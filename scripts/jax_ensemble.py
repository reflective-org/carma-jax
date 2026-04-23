"""JAX ensemble runner (Phase 10.3).

Drives ``make_step_full`` over the 1000-scenario ensemble via
``jax.vmap`` on the batch axis. Each scenario advances the same
180 000-s sulfate simulation (matching the Fortran ensemble's
``nstep × dtime`` setup) and records ``(T_final,
gc_h2so4_final, pc_final)`` for later parity comparison.

This is deliberately scoped to the sulfate-column case that
``make_step_full`` supports today: single-group sulfate, no
transport, no coagulation. That matches the Fortran ensemble
binary (``carma_sulfatetest_ensemble.F90``) exactly — no process
is active on one side and absent on the other.

Output is a compressed NPZ with the same keys as the Fortran
aggregator (``T_final``, ``gc_h2so4_final``, ``pc_final``,
``nstep_ran``, ``status``) so Phase 10.4 can compare them
bit-for-key.
"""

import argparse
import sys
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "tests" / "unit"))

from generate_sulfate_scenarios import load_scenarios
from carma.constants import AVG, BK, WTMOL_H2O
from carma.step_full import make_step_full
from carma.vapor_pressure import vaporp_h2o_murphy2005

# Reuse the env fixture from the step_full tests. _minimal_config
# is rebuilt with Fortran-matching bin grid (see _fortran_config).
from test_step_full import _growth_env, _minimal_config as _test_minimal_config


_GWTMOL_H2SO4 = 98.0

# Fortran sulfatetest bin grid (must match
# carma_sulfatetest_ensemble.F90 for apples-to-apples parity):
_FORTRAN_NBIN = 38
_FORTRAN_RMIN_CM = 2.0e-8       # 20 nm
_FORTRAN_RMRAT = 2.0
_FORTRAN_RHO_SULF = 1.923       # g/cm³


def _fortran_matching_config():
    """Build the CarmaConfig with the same bin grid as the Fortran
    ensemble binary. Uses the same structural layout as the test
    fixture, but with (nbin, rmin, rmrat, rho) taken from the
    Fortran constants."""
    from carma.config import (
        CarmaConfig, ElementConfig, GroupConfig, GasConfig, SoluteConfig,
    )
    import jax.numpy as jnp

    nbin = _FORTRAN_NBIN
    rho = _FORTRAN_RHO_SULF
    rmin = _FORTRAN_RMIN_CM
    rmrat = _FORTRAN_RMRAT
    vmin = (4.0 / 3.0) * np.pi * rmin**3 * rho
    rmass = vmin * rmrat ** np.arange(nbin)
    rmassup = rmass * rmrat**0.5
    r = (3.0 * rmass / (4.0 * np.pi * rho)) ** (1.0 / 3.0)

    group = GroupConfig(
        name="sulfate", ishape=1, ienconc=0,
        is_ice=False, is_cloud=False, is_sulfate=True,
        do_vtran=False, do_drydep=False,
        ifallrtn=1, irhswell=3, rmrat=rmrat, eshape=1.0, rmin=rmin,
        r=jnp.asarray(r), rmass=jnp.asarray(rmass),
        vol=jnp.asarray(rmass / rho),
        dr=jnp.asarray(r * 0.1), dm=jnp.asarray(rmass * 0.1),
        rmassup=jnp.asarray(rmassup),
        rup=jnp.asarray(r * 1.2), rlow=jnp.asarray(r * 0.8),
        rrat=jnp.ones(nbin), rprat=jnp.ones(nbin), arat=jnp.ones(nbin),
    )
    element = ElementConfig(
        name="sulfate_num", rho=jnp.full((nbin,), rho), igroup=0,
        itype=2, icomposition=0, isolute=0, kappa=0.65,
    )
    gas_h2o = GasConfig(name="H2O", wtmol=18.016, ivaprtn=2, icomposition=1,
                         dgc_threshold=0.0, ds_threshold=0.0)
    gas_h2so4 = GasConfig(name="H2SO4", wtmol=_GWTMOL_H2SO4, ivaprtn=4,
                           icomposition=2,
                           dgc_threshold=0.0, ds_threshold=0.0)
    solute = SoluteConfig(name="sulfate", ions=3, wtmol=_GWTMOL_H2SO4, rho=rho)

    return CarmaConfig(
        nbin=nbin, nelem=1, ngroup=1, ngas=2, nsolute=1,
        elements=(element,), groups=(group,),
        gases=(gas_h2o, gas_h2so4), solutes=(solute,),
        coag=None,
        do_coag=False, do_grow=True, do_vtran=False, do_vdiff=False,
        do_thermo=False, do_substep=False, do_explised=False,
        do_incloud=False, do_clearsky=True, do_detrain=False,
        do_pheat=False, do_pheatatm=False, do_cnst_rlh=False,
        itbnd_pc=1, ibbnd_pc=1,
        maxsubsteps=32, minsubsteps=1, maxretries=4, conmax=1e-4,
        cstick=1.0, gsticki=1.0, gstickl=1.0, tstick=1.0, dt_threshold=0.0,
        igash2o=0, igash2so4=1, igasso2=-1,
    )


_minimal_config = _fortran_matching_config


def _init_lognormal_pc(r_bins_cm, mu_nm, sigma_g, total_mass_g_per_cm3=1e-18):
    """Build a lognormal number-density vector per bin, normalised so
    total mass equals ``total_mass_g_per_cm3`` (matches Fortran
    ensemble's ``1e-18 g/g`` tiny seed)."""
    r = np.asarray(r_bins_cm)
    mu_cm = mu_nm * 1e-7
    log_sigma = np.log(sigma_g)
    log_r = np.log(r)
    pc = np.exp(-0.5 * ((log_r - np.log(mu_cm)) / log_sigma) ** 2) \
         / (r * log_sigma * np.sqrt(2 * np.pi))
    rho_sulf = 1.923
    rmass = (4.0 / 3.0) * np.pi * r**3 * rho_sulf
    total = np.sum(pc * rmass)
    if total > 0:
        pc = pc * (total_mass_g_per_cm3 / total)
    return jnp.asarray(pc)


def _env_for_scenario(T_K, p_hPa, rh, h2so4_pptv, mu_nm, sigma_g, cfg):
    """Produce a populated env dict for one scenario, matching the
    make_step_full call signature."""
    base = _growth_env(cfg, T=float(T_K), rh=float(rh),
                        h2so4_ppbv=float(h2so4_pptv) * 1e-3,
                        p_hpa=float(p_hPa))
    r_bins = np.asarray(cfg.groups[0].r)
    pc_sulf = _init_lognormal_pc(r_bins, float(mu_nm), float(sigma_g))
    # base["pc"] is (1, nbin, 1) zero; put our lognormal into element 0.
    pc = jnp.zeros_like(base["pc"])
    pc = pc.at[0, :, 0].set(pc_sulf)
    base["pc"] = pc
    return base


def run_jax_ensemble(scenarios, dtime_s=1800.0, nstep=100):
    """Run every scenario through make_step_full end-to-end.

    Loops in Python for simplicity (not vmap) since each scenario
    runs a sequential time loop; batching in vmap would require
    scan over time within vmap over scenarios. That's a future
    optimisation — the per-call 92 μs warm time × 1000 × 100
    steps ≈ 9 s, fast enough."""
    cfg = _minimal_config()
    print(f"  bin grid: nbin={cfg.nbin}, rmin={float(cfg.groups[0].r[0])*1e7:.1f} nm, "
          f"rmrat={cfg.groups[0].rmrat}, rmax={float(cfg.groups[0].r[-1])*1e4:.2f} μm",
          flush=True)
    step = make_step_full(cfg)

    n = int(scenarios["_n"])
    T_final = np.zeros(n, dtype=np.float64)
    gc_final = np.zeros(n, dtype=np.float64)
    pc_final = np.zeros((n, cfg.nbin), dtype=np.float64)
    nstep_ran = np.full(n, nstep, dtype=np.int64)
    status = ["ok"] * n

    for i in range(n):
        env = _env_for_scenario(
            scenarios["T"][i], scenarios["p"][i], scenarios["rh"][i],
            scenarios["h2so4_pptv"][i], scenarios["aerosol_mu_nm"][i],
            scenarios["aerosol_sigma_g"][i], cfg,
        )
        pc = env["pc"]
        gc = env["gc"]
        t = env["t"]
        env_no_state = {k: v for k, v in env.items()
                        if k not in ("pc", "gc", "t")}

        try:
            for _ in range(nstep):
                pc, gc, t, _ = step(pc=pc, gc=gc, t=t, dtime=dtime_s,
                                    **env_no_state)
        except Exception as exc:            # noqa: BLE001
            status[i] = f"error: {exc.__class__.__name__}: {str(exc)[:200]}"
            continue

        # Convert JAX internal-CGS outputs to Fortran's MMR convention
        # to enable apples-to-apples parity:
        #   gc [g/cm³/z] / rhoa [g/cm³/z]  →  mmr [g/g]
        #   pc [#/cm³/z] × rmass [g] / rhoa → mmr_per_bin [g/g]
        rhoa = float(env["rhoa"][0])          # g/cm³/z
        rmass = np.asarray(cfg.groups[0].rmass)     # g per particle per bin
        T_final[i] = float(t[0])
        gc_final[i] = float(gc[0, 1]) / rhoa                    # g/g
        pc_final[i] = np.asarray(pc[0, :, 0]) * rmass / rhoa   # g/g per bin

        if i % 100 == 0:
            print(f"  scenario {i:4d}/{n}: T={T_final[i]:.1f}K, "
                  f"gc_H2SO4={gc_final[i]:.3e}", flush=True)

    return dict(
        T_final=T_final, gc_h2so4_final=gc_final, pc_final=pc_final,
        nstep_ran=nstep_ran, status=status,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path,
                        default=Path("data/sulfate_scenarios_1000.npz"))
    parser.add_argument("--out", type=Path,
                        default=Path("data/sulfate_jax_outputs.npz"))
    parser.add_argument("--dtime", type=float, default=1800.0)
    parser.add_argument("--nstep", type=int, default=100)
    parser.add_argument("--n", type=int, default=None,
                        help="Override scenario count (for smoke tests)")
    args = parser.parse_args()

    scenarios = load_scenarios(args.scenarios)
    if args.n is not None:
        # Slice to N scenarios for quick runs
        scenarios = {k: (v[:args.n] if hasattr(v, "shape") and v.ndim > 0
                         else v)
                     for k, v in scenarios.items()}
        scenarios["_n"] = args.n

    n = int(scenarios["_n"])
    print(f"Running JAX ensemble over {n} scenarios "
          f"(dtime={args.dtime}s × {args.nstep} steps)...")

    t0 = time.perf_counter()
    results = run_jax_ensemble(scenarios,
                                dtime_s=args.dtime, nstep=args.nstep)
    wall = time.perf_counter() - t0

    n_ok = sum(1 for s in results["status"] if s == "ok")
    print(f"\n  wall time: {wall:.1f} s ({wall/n*1000:.2f} ms / scenario)")
    print(f"  status: {n_ok}/{n} ok, {n - n_ok} errors")

    np.savez_compressed(args.out, **{
        k: v for k, v in results.items() if k != "status"
    })
    # Write status list separately as a plain text sidecar
    (args.out.with_suffix(".status.txt")).write_text(
        "\n".join(f"{i:5d}\t{s}" for i, s in enumerate(results["status"]))
    )
    print(f"  saved to {args.out}")


if __name__ == "__main__":
    main()
