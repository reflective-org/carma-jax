"""Visual A/B: PPM vs 1st-order upwind in the diffrax RHS.

Phase 5 replaced the diffrax mass-space advection scheme from upwind
to PPM. This script reruns a controlled condensation scenario through
the diffrax solver with both reconstructions and overlays the final
size distributions, plus the (faithful JAX) reference that already
uses PPM.

The "upwind" run is produced by monkey-patching
``carma_diffrax.ppm_advection.ppm_n_at_boundary`` with a first-order
implementation at runtime, so the rest of the pipeline (env build,
solver, tolerances) is identical between the two runs.

Setup (matches the prior condensation_3way controlled scenarios):
  - Single sulfate group, NBIN=38, rmin=2e-8 cm, rmrat=2
  - Lognormal seed: GMD₀ = 20 nm, GSD = 1.6, N = 1e4 #/cm³
  - p = 90 hPa, T = 250 K, RH = 1.5 % (sulfate stratospheric)
  - H₂SO₄ held fixed (prescribed schedule)
  - dtime = 60 s, nstep = 60  (1 h)
  - No nucleation, no coagulation — pure growth

What to look for:
  - Both PPM and upwind conserve total mass (gas-side reconstructed
    from the bin scheme by construction).
  - PPM should produce a NARROWER, taller peak than upwind, closer
    to the faithful reference (which uses PPM in a different numerical
    framework — semi-implicit Euler + adaptive substeps).

Output:
  plots/diff/ppm/diffrax_ppm_vs_upwind.png
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from carma.constants import AVG, R_AIR, RPA2CGS
from carma.precision import DTYPE
import carma_diffrax.ppm_advection as ppm_mod
from carma_diffrax import diffrax_step
from carma_diffrax.config import DiffraxConfig
from carma_diffrax.rhs import FrozenEnv
from carma_diffrax.state import StateShape


T_K = 250.0
P_HPA = 90.0
RH = 0.015
H2SO4_NUMBER_CM3 = 5.0e8   # constant H2SO4 concentration
DTIME = 60.0
NSTEP = 60                  # 1 h total
GMD_NM = 20.0
GSD = 1.6
N_TOT = 1.0e4


def _setup_inputs():
    """Build the cfg + initial state + env for one outer step."""
    from jax_ensemble import (
        _minimal_config, _compute_ppm_coefs, _refresh_env,
        _init_lognormal_pc,
    )

    cfg = _minimal_config()._replace(do_coag=False, do_grow=True)
    ppm = _compute_ppm_coefs(cfg)
    grp = cfg.groups[0]

    T = jnp.asarray([T_K], dtype=DTYPE)
    p_cgs = jnp.asarray([P_HPA * 100.0 * float(RPA2CGS)], dtype=DTYPE)

    rho_air = float(p_cgs[0]) / (float(R_AIR) * T_K)
    h2so4_mmr = H2SO4_NUMBER_CM3 * (98.078479 / float(AVG)) / rho_air
    h2o_pvap_pa = math.exp(
        54.842763 - 6763.22 / T_K - 4.210 * math.log(T_K) + 0.000367 * T_K
    )
    h2o_mmr = RH * h2o_pvap_pa * 18.016 / (29.0 * P_HPA * 100.0)
    gc = jnp.asarray(
        [[h2o_mmr * rho_air, h2so4_mmr * rho_air]], dtype=DTYPE,
    )

    r = np.asarray(grp.r)
    pc_lognormal = _init_lognormal_pc(
        r, GMD_NM / 2.0, GSD, total_mass_g_per_cm3=N_TOT * 1.0e-22,
    )
    # Normalise to N_TOT particles/cm³ directly.
    pc_total = float(jnp.sum(pc_lognormal))
    pc_lognormal = pc_lognormal * (N_TOT / pc_total)
    pc = jnp.zeros((1, cfg.nbin, cfg.nelem), dtype=DTYPE)
    pc = pc.at[0, :, 0].set(pc_lognormal)

    return cfg, ppm, T, p_cgs, gc, pc, grp, rho_air


def _build_env(T, p_cgs, gc, cfg, ppm, grp):
    from jax_ensemble import _refresh_env
    env_d = _refresh_env(T, p_cgs, gc, cfg, ppm)
    r_wet = (env_d["rup_wet"][0, :, 0] + env_d["rlow_wet"][0, :, 0]) / 2.0
    rmassup_np = np.asarray(grp.rmassup)
    rmass_np = np.asarray(grp.rmass)
    rmasslow_np = np.concatenate(
        [[rmass_np[0] / (grp.rmrat ** 0.5)], rmassup_np[:-1]]
    )
    dm = jnp.asarray(rmassup_np - rmasslow_np, dtype=DTYPE)
    return FrozenEnv(
        akelvin=env_d["akelvin"], akelvini=env_d["akelvini"],
        gro=env_d["gro"], gro1=env_d["gro1"],
        rup_wet=env_d["rup_wet"], r_wet=r_wet,
        rmass=jnp.asarray(rmass_np, dtype=DTYPE)[:, None],
        dm=dm[:, None],
        rmassup=jnp.asarray(rmassup_np, dtype=DTYPE),
        rmrat_val=float(grp.rmrat),
        rhoa=env_d["rhoa"], zmet=env_d["zmet"],
        rlhe=env_d["rlhe"], rlhm=env_d["rlhm"],
        pratt=env_d["pratt"][..., 0], prat=env_d["prat"][..., 0],
        pden1=env_d["pden1"][..., 0], palr=env_d["palr"][..., 0],
    )


def _run(scheme: str):
    """Run the diffrax solver with the requested boundary scheme."""
    cfg, ppm, T, p_cgs, gc, pc, grp, rho_air = _setup_inputs()
    shape = StateShape(nbin=cfg.nbin, nelem=cfg.nelem, ngas=cfg.ngas)
    cfg_d = DiffraxConfig()

    # Hot-swap the PPM function with a 1st-order upwind formula by
    # monkey-patching the symbol both in the module and in the rhs
    # module that imported it. Then clear JAX caches so the previous
    # JIT trace (which baked in the old function) is rebuilt.
    original = ppm_mod.ppm_n_at_boundary
    import carma_diffrax.rhs as rhs_mod
    if scheme == "upwind":
        @jax.jit
        def upwind(pc_elem, dm, pratt, prat, pden1, palr, dmdt):
            del pratt, prat, pden1, palr
            nbin = pc_elem.shape[0]
            zero = jnp.zeros((1,), dtype=DTYPE)
            pc_above = jnp.concatenate([pc_elem[1:], zero])
            dm_above = jnp.concatenate(
                [dm[1:], jnp.ones((1,), dtype=DTYPE)],
            )
            n = jnp.where(
                dmdt > 0, pc_elem / dm, pc_above / dm_above,
            )
            return n.at[nbin - 1].set(DTYPE(0.0))
        ppm_mod.ppm_n_at_boundary = upwind
        rhs_mod.ppm_n_at_boundary = upwind
    else:
        # Make sure we have the PPM version active on this entry.
        ppm_mod.ppm_n_at_boundary = original
        rhs_mod.ppm_n_at_boundary = original

    # Force re-trace: previous step-build cached the old function ref.
    jax.clear_caches()
    # Also rebuild the diffrax_step's _build_jit_step cache (it lives
    # in step.py via @functools.lru_cache on shape+solver_name).
    import carma_diffrax.step as step_mod
    if hasattr(step_mod, "_build_jit_step"):
        # Wipe lru_cache if present
        bjs = step_mod._build_jit_step
        if hasattr(bjs, "cache_clear"):
            bjs.cache_clear()

    try:
        T_s = float(T[0])
        # Hold H2SO4 fixed by resetting each outer step.
        gc_h2so4_target = float(gc[0, 1])
        for _ in range(NSTEP):
            gc = gc.at[0, 1].set(gc_h2so4_target)
            env = _build_env(jnp.atleast_1d(T_s), p_cgs, gc, cfg, ppm, grp)
            pc_1d, gc_1d, T_s, _stats = diffrax_step(
                pc[0], gc[0], T_s, DTIME, env, shape, cfg_d,
            )
            pc = pc_1d[None, :, :]
            gc = gc_1d[None, :]
    finally:
        ppm_mod.ppm_n_at_boundary = original
        import carma_diffrax.rhs as rhs_mod
        rhs_mod.ppm_n_at_boundary = original

    pc_np = np.asarray(pc[0, :, 0])
    return cfg, np.asarray(grp.r), pc_np, rho_air


def main():
    print(f"\n=== PPM vs upwind in diffrax ===")
    print(f"  GMD₀={GMD_NM} nm, GSD={GSD}, N={N_TOT:.0e}, "
          f"[H₂SO₄]={H2SO4_NUMBER_CM3:.0e} #/cm³, "
          f"T={T_K} K, p={P_HPA} hPa, RH={RH:.1%}")
    print(f"  dt={DTIME}s × {NSTEP} = {DTIME*NSTEP/60:.0f} min total")

    t0 = time.perf_counter()
    cfg, r_cm, pc_ppm, rho_air = _run("ppm")
    print(f"  PPM      wall = {time.perf_counter()-t0:.1f}s, "
          f"sum_pc = {pc_ppm.sum():.3e} #/cm³")

    t0 = time.perf_counter()
    _, _, pc_upwind, _ = _run("upwind")
    print(f"  Upwind   wall = {time.perf_counter()-t0:.1f}s, "
          f"sum_pc = {pc_upwind.sum():.3e} #/cm³")

    rmass = np.asarray(cfg.groups[0].rmass)
    mass_ppm = pc_ppm * rmass
    mass_upwind = pc_upwind * rmass

    # Bin-centre dlogD for the dN/dlogD axis
    d_nm = r_cm * 2.0 * 1e7
    log_d = np.log10(d_nm)
    dlogd = np.gradient(log_d)
    dN_dlogD_ppm = pc_ppm / dlogd
    dN_dlogD_upwind = pc_upwind / dlogd

    # Find peak bin index for each
    peak_ppm = int(np.argmax(dN_dlogD_ppm))
    peak_upwind = int(np.argmax(dN_dlogD_upwind))
    print(f"  peak bin: PPM={peak_ppm} (d={d_nm[peak_ppm]:.1f} nm, "
          f"dN/dlogD={dN_dlogD_ppm[peak_ppm]:.3e})")
    print(f"            upwind={peak_upwind} (d={d_nm[peak_upwind]:.1f} nm, "
          f"dN/dlogD={dN_dlogD_upwind[peak_upwind]:.3e})")

    # --- Plots ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(d_nm, dN_dlogD_ppm, "-o", color="#27ae60", ms=4, lw=2,
             label="diffrax + PPM (this PR)")
    ax.plot(d_nm, dN_dlogD_upwind, "-s", color="#c0392b", ms=4, lw=2,
             alpha=0.85, label="diffrax + upwind (baseline)")
    ax.set_xscale("log")
    ax.set_xlabel("diameter [nm]")
    ax.set_ylabel("dN/dlogD [#/cm³]")
    ax.set_title(f"Final number distribution after {DTIME*NSTEP/60:.0f} min "
                  f"of condensation\n(GMD₀={GMD_NM} nm, "
                  f"[H₂SO₄]={H2SO4_NUMBER_CM3:.0e} #/cm³)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(d_nm[0] * 0.9, d_nm[-1] * 1.1)

    ax = axes[1]
    ax.plot(d_nm, mass_ppm, "-o", color="#27ae60", ms=4, lw=2,
             label="PPM")
    ax.plot(d_nm, mass_upwind, "-s", color="#c0392b", ms=4, lw=2,
             alpha=0.85, label="upwind")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("diameter [nm]")
    ax.set_ylabel("mass per bin [g/cm³]")
    ax.set_title("Per-bin mass (log-y)")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    out = REPO / "plots" / "diff" / "ppm" / "diffrax_ppm_vs_upwind.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=110)
    print(f"\nSaved {out}")

    # Conservation check
    tot_n_ppm = float(pc_ppm.sum())
    tot_m_ppm = float(mass_ppm.sum())
    tot_n_up = float(pc_upwind.sum())
    tot_m_up = float(mass_upwind.sum())
    print(f"\n  total number: PPM={tot_n_ppm:.4e}, "
          f"upwind={tot_n_up:.4e}, "
          f"diff = {(tot_n_ppm-tot_n_up)/tot_n_up:.2%}")
    print(f"  total mass:   PPM={tot_m_ppm:.4e}, "
          f"upwind={tot_m_up:.4e}, "
          f"diff = {(tot_m_ppm-tot_m_up)/tot_m_up:.2%}")


if __name__ == "__main__":
    main()
