"""Isolate which dynamic process is causing the bin-shift drift.

For one scenario (scen 3 — high H2SO4, clearest Fortran bimodal):
- Run JAX with both growth+coag (current)
- Run JAX with coag DISABLED (growth only)
- Run JAX with growth DISABLED (coag only on initial pop, no nucleation
  source)
Compare the resulting spectra to see which process is responsible for
JAX's missing grown mode.
"""
import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from jax_ensemble import (
    _fortran_matching_config, _compute_ppm_coefs, _env_for_scenario,
)
from carma.prestep import prestep
from carma.precision import DTYPE
from carma.step_full_faithful import make_step_full_faithful


def run_one(cfg, env, dtime, nstep):
    step = make_step_full_faithful(cfg, ppm_coefs=_compute_ppm_coefs(cfg))
    itype_arr = jnp.asarray([e.itype for e in cfg.elements])
    ienconc_arr = jnp.asarray([g.ienconc for g in cfg.groups])
    igelem_arr = jnp.asarray([e.igroup for e in cfg.elements])
    rmass_2d = jnp.stack(
        [jnp.asarray(g.rmass, dtype=DTYPE) for g in cfg.groups], axis=1)

    pc = env["pc"]; gc = env["gc"]; t = env["t"]
    for _ in range(nstep):
        pc, gc, t, pcl, gcl, d_gc, d_t, pconmax = prestep(
            pc, gc, t, pc, gc, t, env["zmet"],
            itype_arr, ienconc_arr, igelem_arr, rmass_2d,
            do_substep=True, do_coag=cfg.do_coag,
        )
        pc, gc, t, _ = step(
            pc=pc, gc=gc, t=t, dtime=dtime,
            rhoa=env["rhoa"], zmet=env["zmet"],
            akelvin=env["akelvin"], akelvini=env["akelvini"],
            gro=env["gro"], gro1=env["gro1"], rup_wet=env["rup_wet"],
            rlhe=env["rlhe"], rlhm=env["rlhm"],
            ckernel=env["ckernel"], pconmax=pconmax,
            ds_threshold_arr=env["ds_threshold_arr"],
            pcl=pcl, gcl=gcl, told=t, d_gc=d_gc, d_t=d_t,
            dt_threshold=DTYPE(cfg.dt_threshold),
        )
    return pc, gc, t


def main():
    sc = np.load(ROOT / "data/sulfate_scenarios_realistic_1000.npz")
    F = np.load(ROOT / "data/sulfate_fortran_realistic_outputs.npz")
    SCEN = 3
    DTIME, NSTEP = 1800.0, 100
    cfg_full = _fortran_matching_config()
    ppm = _compute_ppm_coefs(cfg_full)
    env = _env_for_scenario(
        sc["T"][SCEN], sc["p"][SCEN], sc["rh"][SCEN], sc["h2so4_pptv"][SCEN],
        sc["aerosol_mu_nm"][SCEN], sc["aerosol_sigma_g"][SCEN], cfg_full, ppm,
    )

    print(f"scen {SCEN}: T={sc['T'][SCEN]:.1f}K, RH={sc['rh'][SCEN]:.2f}, "
          f"H2SO4={sc['h2so4_pptv'][SCEN]:.2f}pptv, mu={sc['aerosol_mu_nm'][SCEN]:.1f}nm, "
          f"sig={sc['aerosol_sigma_g'][SCEN]:.2f}")

    # Initial pc spectrum
    pc_init = np.asarray(env["pc"])[0, :, 0]

    # Variant 1: full (growth + coag + nucleation)
    print("\n[1/3] full (growth + coag + nucleation)...", flush=True)
    pc1, gc1, t1 = run_one(cfg_full, env, DTIME, NSTEP)

    # Variant 2: no coag — growth + nucleation only
    print("[2/3] no coag (growth + nucleation only)...", flush=True)
    cfg_nc = cfg_full._replace(do_coag=False)
    env2 = dict(env)   # copy
    pc2, gc2, t2 = run_one(cfg_nc, env2, DTIME, NSTEP)

    # Variant 3: coag only — disable growth/nucleation by passing
    #   * akelvin / akelvini / gro / gro1 effectively neutralised
    # Easiest: create a config with do_grow=False (won't help since
    # microfast_full doesn't gate on do_grow). Cheaper: zero out gro and
    # rhompe by zeroing gro/gro1 and disabling sulfnuc via
    # do_homogeneous=False. Pass that into make_step_full_faithful.
    print("[3/3] no growth, no nucleation (coag only)...", flush=True)
    step3 = make_step_full_faithful(
        cfg_full, ppm_coefs=_compute_ppm_coefs(cfg_full),
        do_homogeneous=False, do_heterogeneous=False,
    )
    itype_arr = jnp.asarray([e.itype for e in cfg_full.elements])
    ienconc_arr = jnp.asarray([g.ienconc for g in cfg_full.groups])
    igelem_arr = jnp.asarray([e.igroup for e in cfg_full.elements])
    rmass_2d = jnp.stack(
        [jnp.asarray(g.rmass, dtype=DTYPE) for g in cfg_full.groups], axis=1)
    pc = env["pc"]; gc = env["gc"]; t = env["t"]
    env3 = dict(env)
    env3["gro"] = jnp.zeros_like(env["gro"])
    env3["gro1"] = jnp.zeros_like(env["gro1"])
    for _ in range(NSTEP):
        pc, gc, t, pcl, gcl, d_gc, d_t, pconmax = prestep(
            pc, gc, t, pc, gc, t, env3["zmet"],
            itype_arr, ienconc_arr, igelem_arr, rmass_2d,
            do_substep=True, do_coag=cfg_full.do_coag,
        )
        pc, gc, t, _ = step3(
            pc=pc, gc=gc, t=t, dtime=DTIME,
            rhoa=env3["rhoa"], zmet=env3["zmet"],
            akelvin=env3["akelvin"], akelvini=env3["akelvini"],
            gro=env3["gro"], gro1=env3["gro1"], rup_wet=env3["rup_wet"],
            rlhe=env3["rlhe"], rlhm=env3["rlhm"],
            ckernel=env3["ckernel"], pconmax=pconmax,
            ds_threshold_arr=env3["ds_threshold_arr"],
            pcl=pcl, gcl=gcl, told=t, d_gc=d_gc, d_t=d_t,
            dt_threshold=DTYPE(cfg_full.dt_threshold),
        )
    pc3 = pc

    # Extract mmr per bin (g/g)
    rhoa = float(env["rhoa"][0])
    rmass = np.asarray(cfg_full.groups[0].rmass)
    pc1_mmr = np.asarray(pc1[0, :, 0]) * rmass / rhoa
    pc2_mmr = np.asarray(pc2[0, :, 0]) * rmass / rhoa
    pc3_mmr = np.asarray(pc3[0, :, 0]) * rmass / rhoa
    pc_init_mmr = pc_init * rmass / rhoa
    pcF = F["pc_final"][SCEN]

    # Plot
    nbin = cfg_full.nbin
    r_nm = np.asarray(cfg_full.groups[0].r) * 1e7
    import math
    dlogr = math.log10(2.0) / 3.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.loglog(r_nm, pc_init_mmr / dlogr, color="0.5", ls=":", lw=1.5, label="initial")
    ax.loglog(r_nm, pcF        / dlogr, "k-", lw=2, label="Fortran (full, t=50h)")
    ax.loglog(r_nm, pc1_mmr    / dlogr, "r--", lw=2, label="JAX full")
    ax.loglog(r_nm, pc2_mmr    / dlogr, "b-.", lw=2, label="JAX no-coag")
    ax.loglog(r_nm, pc3_mmr    / dlogr, "g:", lw=2, label="JAX coag-only (no growth/nuc)")
    ax.set_xlabel("r (nm)")
    ax.set_ylabel("dM/dlogr (g/g)")
    ax.set_title(f"scen {SCEN} mass: T={sc['T'][SCEN]:.1f}K, "
                 f"H2SO4={sc['h2so4_pptv'][SCEN]:.2f}pptv (50h)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9)

    ax2 = axes[1]
    ax2.loglog(r_nm, pc_init_mmr / rmass / dlogr, color="0.5", ls=":", lw=1.5, label="initial")
    ax2.loglog(r_nm, pcF        / rmass / dlogr, "k-", lw=2, label="Fortran")
    ax2.loglog(r_nm, pc1_mmr    / rmass / dlogr, "r--", lw=2, label="JAX full")
    ax2.loglog(r_nm, pc2_mmr    / rmass / dlogr, "b-.", lw=2, label="JAX no-coag")
    ax2.loglog(r_nm, pc3_mmr    / rmass / dlogr, "g:", lw=2, label="JAX coag-only")
    ax2.set_xlabel("r (nm)")
    ax2.set_ylabel("dN/dlogr ∝ pc/rmass")
    ax2.set_title(f"scen {SCEN} number")
    ax2.grid(True, alpha=0.3, which="both")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    out = ROOT / "data" / f"_isolate_scen{SCEN:03d}.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\nsaved {out}")

    print(f"\nMass conservation check:")
    print(f"  initial Σ mmr   = {pc_init_mmr.sum():.4e}")
    print(f"  Fortran Σ mmr   = {pcF.sum():.4e}")
    print(f"  JAX full        = {pc1_mmr.sum():.4e}")
    print(f"  JAX no-coag     = {pc2_mmr.sum():.4e}")
    print(f"  JAX coag-only   = {pc3_mmr.sum():.4e}")


if __name__ == "__main__":
    main()
