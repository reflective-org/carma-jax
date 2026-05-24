"""Run JAX CARMA over the 100 realistic scenarios.

Same setup as the patched Fortran binary:
  - 60 s outer timestep × 1440 steps (24 hours)
  - H₂SO₄ injected each step as production_rate × dtime
  - Initial seed mass M_total_ug_m3, log-normal (mu_nm, sigma_g)
  - Initial gas H₂SO₄ = 0
  - sulfate-only single-cell

Outputs benchmark_final/outputs/jax_outputs.npz with the same schema
as Fortran (T_final, gc_h2so4_final, pc_final, nstep_ran) plus wall_time_s.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tests" / "unit"))

from jax_ensemble import (_minimal_config, _compute_ppm_coefs,
                           _refresh_env)
from carma.constants import AVG, R_AIR, RPA2CGS
from carma.step_full_faithful import make_step_full_faithful
from carma.prestep import prestep
from carma.precision import DTYPE

M_H2SO4 = 98.078479           # g/mol
_GWTMOL_H2SO4 = 98.078479
_FORTRAN_RHO_SULF = 1.923     # match patched Fortran F90


def init_pc_from_mass(cfg, M_ug_m3, mu_nm, sigma_g, rhoa_g_cm3):
    """Initial log-normal seed (mass-weighted) targeting total mmr M_target.

    Mirrors the Fortran initial-seed block but produces a JAX array.
    Returns pc[nz=1, nbin, nelem=1] in #/cm³ (CARMA internal units).
    """
    r = np.asarray(cfg.groups[0].r)      # cm
    rmass = np.asarray(cfg.groups[0].rmass)   # g/particle
    log_mu_cm = math.log(mu_nm * 1e-7)
    log_sigma = math.log(sigma_g)

    # Mass-weighted log-normal shape
    shape = (np.exp(-0.5 * ((np.log(r) - log_mu_cm) / log_sigma)**2)
              / (r * log_sigma * math.sqrt(2.0 * math.pi))
              * rmass)
    norm = shape.sum()
    if norm <= 0:
        return jnp.zeros((1, cfg.nbin, cfg.nelem), dtype=DTYPE)

    # Total target mmr = M [g/cm³] / rho_air [g/cm³]
    M_target_mmr = (M_ug_m3 * 1e-12) / rhoa_g_cm3
    mmr_per_bin = shape * (M_target_mmr / norm)         # g/g per bin

    # Convert mmr → pc [#/cm³]: pc = mmr × rhoa / rmass
    pc_per_bin = mmr_per_bin * rhoa_g_cm3 / rmass

    pc = jnp.zeros((1, cfg.nbin, cfg.nelem), dtype=DTYPE)
    pc = pc.at[0, :, 0].set(jnp.asarray(pc_per_bin, dtype=DTYPE))
    return pc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenarios", type=Path,
                   default=ROOT / "scenarios" / "realistic_scenarios_100.npz")
    p.add_argument("--out", type=Path,
                   default=ROOT / "outputs" / "jax_outputs.npz")
    p.add_argument("--dtime", type=float, default=60.0,
                   help="outer timestep [s]")
    p.add_argument("--nstep", type=int, default=1440,
                   help="number of outer steps (1440 = 24h at 60s)")
    args = p.parse_args()

    scens = np.load(args.scenarios)
    n = int(scens["_n"])

    cfg = _minimal_config()
    ppm = _compute_ppm_coefs(cfg)
    step = make_step_full_faithful(cfg, ppm_coefs=ppm)
    itype_arr   = jnp.asarray([e.itype for e in cfg.elements])
    ienconc_arr = jnp.asarray([g.ienconc for g in cfg.groups])
    igelem_arr  = jnp.asarray([e.igroup for e in cfg.elements])
    rmass_2d    = jnp.stack(
        [jnp.asarray(g.rmass, dtype=DTYPE) for g in cfg.groups], axis=1,
    )

    print(f"Running JAX realistic ensemble: {n} scenarios, "
          f"{args.nstep} steps × {args.dtime} s ({args.nstep*args.dtime/3600:.0f} h)…",
          flush=True)
    t_start = time.perf_counter()

    nbin = cfg.nbin
    T_final = np.zeros(n)
    gc_final = np.zeros(n)
    pc_final = np.zeros((n, nbin))     # output as mmr (g/g per bin) to match Fortran
    nstep_ran = np.full(n, args.nstep, dtype=np.int64)

    rmass = np.asarray(cfg.groups[0].rmass)

    for i in range(n):
        T_K = float(scens["T"][i])
        p_hPa = float(scens["p"][i])
        rh = float(scens["rh"][i])
        prod_rate = float(scens["h2so4_prod_rate"][i])
        M_ug_m3 = float(scens["M_total_ug_m3"][i])
        mu_nm = float(scens["aerosol_mu_nm"][i])
        sigma_g = float(scens["aerosol_sigma_g"][i])

        p_cgs_val = p_hPa * 100.0 * float(RPA2CGS)
        rhoa_g_cm3 = p_cgs_val / (float(R_AIR) * T_K)

        T = jnp.asarray([T_K], dtype=DTYPE)
        p_cgs = jnp.asarray([p_cgs_val], dtype=DTYPE)

        # H2O from RH × p_vap (Murphy-Koop-like)
        pvapl_Pa = math.exp(54.842763 - 6763.22/T_K
                            - 4.210*math.log(T_K) + 0.000367*T_K)
        h2o_mmr = rh * pvapl_Pa * 18.0 / (29.0 * p_hPa * 100.0)
        # gc in JAX CGS = mmr * rhoa
        gc = jnp.asarray([[h2o_mmr * rhoa_g_cm3, 0.0]], dtype=DTYPE)

        # Initial seed
        pc = init_pc_from_mass(cfg, M_ug_m3, mu_nm, sigma_g, rhoa_g_cm3)
        t = T

        # H₂SO₄ injection per step: gc_h2so4 += prod_rate × dtime × M_H2SO4 / AVG
        # JAX gc is in g/cm³ (= mmr × rhoa). Direct mass injection per cm³:
        dgc_h2so4_per_step = prod_rate * args.dtime * M_H2SO4 / float(AVG)

        for istep in range(args.nstep):
            # Inject H₂SO₄ production
            gc = gc.at[0, 1].set(gc[0, 1] + dgc_h2so4_per_step)

            env_s = _refresh_env(t, p_cgs, gc, cfg, ppm)
            pc, gc, t, pcl, gcl, d_gc, d_t, pconmax = prestep(
                pc, gc, t, pc, gc, t, env_s["zmet"],
                itype_arr, ienconc_arr, igelem_arr, rmass_2d,
                do_substep=True, do_coag=cfg.do_coag,
            )
            pc, gc, t, _diag = step(
                pc=pc, gc=gc, t=t, dtime=float(args.dtime),
                rhoa=env_s["rhoa"], zmet=env_s["zmet"],
                akelvin=env_s["akelvin"], akelvini=env_s["akelvini"],
                gro=env_s["gro"], gro1=env_s["gro1"],
                rup_wet=env_s["rup_wet"],
                rlhe=env_s["rlhe"], rlhm=env_s["rlhm"],
                ckernel=env_s["ckernel"], pconmax=pconmax,
                ds_threshold_arr=env_s["ds_threshold_arr"],
                pcl=pcl, gcl=gcl, told=t, d_gc=d_gc, d_t=d_t,
            )

        # Convert final state to MMR for comparison with Fortran
        T_final[i] = float(t[0])
        gc_final[i] = float(gc[0, 1]) / rhoa_g_cm3   # g/cm³ → g/g
        pc_final[i] = np.asarray(pc[0, :, 0]) * rmass / rhoa_g_cm3   # #/cm³ → mmr g/g

        if (i + 1) % 10 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  [{i+1:3d}/{n}] T={T_K:5.1f}K  "
                  f"prod={prod_rate:.2e}  Ntot={float(jnp.sum(pc[0,:,0])):.2e} #/cm³  "
                  f"{elapsed/(i+1):.1f} s/scen", flush=True)

    elapsed = time.perf_counter() - t_start
    print(f"\nJAX wall time: {elapsed:.0f} s ({elapsed/n*1000:.0f} ms/scenario)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        T_final=T_final, gc_h2so4_final=gc_final, pc_final=pc_final,
        nstep_ran=nstep_ran,
        wall_time_s=np.array([elapsed]),
        n_scenarios=np.array([n]),
    )
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
