"""Phase 10.4e multi-step mass-conservation diagnostic.

Single-step is perfect (ratio 1.0001). The 400× loss must come
from another step in the loop. This script runs the same
ensemble loop as jax_ensemble.py but logs gas + particle mass
before/after EACH stage:

  1. step_coag (prestep + microslow)
  2. step_full inner: growth (newstate_calc_growth_jit)
  3. step_full inner: sulfate (sulfate_step_one_level)

so we can see exactly which stage drops particle mass without
giving the gas back.
"""

import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "tests" / "unit"))

from generate_sulfate_scenarios import load_scenarios
from jax_ensemble import (_compute_ppm_coefs, _env_for_scenario,
                            _fortran_matching_config)
from carma.precision import DTYPE
from carma.sulfate_step import sulfate_step_one_level
from carma.newstate_calc_jit import newstate_calc_growth_jit
from carma.step import make_step_coag


def _sum_pc_mass(pc, rmass):
    return float((pc[0, :, 0] * rmass).sum())


def diagnose_multistep(scenario_idx=0, nstep=10):
    cfg = _fortran_matching_config()
    ppm = _compute_ppm_coefs(cfg)
    s = load_scenarios(_ROOT / "data" / "sulfate_scenarios_realistic_1000.npz")

    print(f"=== Multi-step diagnostic, scenario {scenario_idx} ===")
    print(f"  T={s['T'][scenario_idx]:.2f}K, p={s['p'][scenario_idx]:.2f}hPa, "
          f"H2SO4={s['h2so4_pptv'][scenario_idx]:.4f}pptv")

    env = _env_for_scenario(
        s["T"][scenario_idx], s["p"][scenario_idx], s["rh"][scenario_idx],
        s["h2so4_pptv"][scenario_idx], s["aerosol_mu_nm"][scenario_idx],
        s["aerosol_sigma_g"][scenario_idx], cfg, ppm,
    )

    rmass = jnp.asarray(cfg.groups[0].rmass)
    rhoa_cgs = float(env["rhoa"][0]) / float(env["zmet"][0])
    dtime = 1800.0

    pc = env["pc"]
    gc = env["gc"]
    t = env["t"]
    ckernel = env["ckernel"]
    pcl, gcl, told = pc, gc, t

    # Diffmass + inuc2bin for sulfate
    rmass_2d = rmass[:, None]
    diffmass = (rmass_2d[:, :, None, None]
                  - rmass_2d[None, None, :, :]).astype(DTYPE)
    nbin = cfg.nbin
    i2_map = jnp.arange(nbin, dtype=jnp.int32) + 1
    i2_map = jnp.where(i2_map < nbin, i2_map,
                        jnp.asarray(-1, dtype=jnp.int32))
    inuc2bin = i2_map.reshape(nbin, 1, 1)

    step_coag = make_step_coag(cfg)

    is_ice_arr = tuple(bool(g.is_ice) for g in cfg.groups)
    # Sulfate grows by H2SO4 (gas index igash2so4=1), not H2O (gas 0).
    igrowgas_arr = tuple(int(cfg.igash2so4) for _ in cfg.elements)
    ienconc_arr = tuple(int(g.ienconc) for g in cfg.groups)
    igroup_arr = tuple(int(e.igroup) for e in cfg.elements)
    gwtmol_arr = tuple(float(g.wtmol) for g in cfg.gases)

    rmass_2d_cfg = rmass[:, None]
    dm_2d_cfg = jnp.asarray(cfg.groups[0].dm)[:, None]

    fmt = "{:14.6e}"
    header = f"{'step':>4s} {'stage':>14s} {'gc_consumed':>14s} {'pc_gained':>14s} {'cum gc':>14s} {'cum pc':>14s} {'ratio':>8s}"
    print()
    print(header)
    gc0 = float(gc[0, 1]) / rhoa_cgs                  # MMR
    pc0 = _sum_pc_mass(pc, rmass) / rhoa_cgs

    print(f"{'0':>4s} {'init':>14s} {'-':>14s} {'-':>14s} "
          f"{fmt.format(0.0):>14s} {fmt.format(0.0):>14s} {'-':>8s}")

    for istep in range(nstep):
        # Stage A: step_coag (prestep + microslow)
        gc_a0 = float(gc[0, 1]) / rhoa_cgs
        pc_a0 = _sum_pc_mass(pc, rmass) / rhoa_cgs
        pc_n, gc_n, t_n, pcl, gcl, _ = step_coag(
            pc, gc, t, pcl, gcl, told, env["zmet"], ckernel, dtime,
        )
        gc_a1 = float(gc_n[0, 1]) / rhoa_cgs
        pc_a1 = _sum_pc_mass(pc_n, rmass) / rhoa_cgs
        gc_diff = gc0 - gc_a1
        pc_diff = pc_a1 - pc0
        ratio = pc_diff / gc_diff if abs(gc_diff) > 1e-30 else float("nan")
        print(f"{istep+1:>4d} {'coag':>14s} {fmt.format(gc_a0 - gc_a1):>14s} "
              f"{fmt.format(pc_a1 - pc_a0):>14s} {fmt.format(gc_diff):>14s} "
              f"{fmt.format(pc_diff):>14s} {ratio:>8.3e}")
        pc, gc, t = pc_n, gc_n, t_n
        told = t

        # Stage B: growth
        gc_b0 = float(gc[0, 1]) / rhoa_cgs
        pc_b0 = _sum_pc_mass(pc, rmass) / rhoa_cgs
        pc_n, gc_n, t_n, _, _ = newstate_calc_growth_jit(
            pc, gc, t, 0, dtime,
            env["rhoa"], env["zmet"], env["rlhe"], env["rlhm"], env["diffus"],
            env["akelvin"], env["akelvini"],
            env["gro"], env["gro1"], env["gro2"],
            env["rup_wet"], rmass_2d_cfg, dm_2d_cfg, env["rlow_wet"],
            env["pratt"], env["prat"], env["pden1"], env["palr"],
            is_ice_arr, igrowgas_arr, ienconc_arr,
            igroup_arr, gwtmol_arr,
            cfg.nbin, cfg.ngroup, cfg.ngas, cfg.nelem,
            ds_threshold_arr=env["ds_threshold_arr"],
        )
        gc_b1 = float(gc_n[0, 1]) / rhoa_cgs
        pc_b1 = _sum_pc_mass(pc_n, rmass) / rhoa_cgs
        gc_diff = gc0 - gc_b1
        pc_diff = pc_b1 - pc0
        ratio = pc_diff / gc_diff if abs(gc_diff) > 1e-30 else float("nan")
        print(f"{'':>4s} {'growth':>14s} {fmt.format(gc_b0 - gc_b1):>14s} "
              f"{fmt.format(pc_b1 - pc_b0):>14s} {fmt.format(gc_diff):>14s} "
              f"{fmt.format(pc_diff):>14s} {ratio:>8.3e}")
        pc, gc, t = pc_n, gc_n, t_n

        # Stage C: sulfate
        gc_c0 = float(gc[0, 1]) / rhoa_cgs
        pc_c0 = _sum_pc_mass(pc, rmass) / rhoa_cgs
        pc_sulf_new, gc_h2so4_new, _ = sulfate_step_one_level(
            pc=pc[0, :, 0], gc_h2so4=gc[0, 1], gc_h2o=gc[0, 0],
            temp=t[0], pvapl=env["pvapl"], zmet=env["zmet"][0], dtime=dtime,
            r_bins=jnp.asarray(cfg.groups[0].r), rmass=rmass,
            rmassup=jnp.asarray(cfg.groups[0].rmassup),
            diffmass=diffmass, inuc2bin=inuc2bin,
            rmrat_val=float(cfg.groups[0].rmrat),
        )
        pc = pc.at[0, :, 0].set(pc_sulf_new)
        gc = gc.at[0, 1].set(gc_h2so4_new)
        gc_c1 = float(gc[0, 1]) / rhoa_cgs
        pc_c1 = _sum_pc_mass(pc, rmass) / rhoa_cgs
        gc_diff = gc0 - gc_c1
        pc_diff = pc_c1 - pc0
        ratio = pc_diff / gc_diff if abs(gc_diff) > 1e-30 else float("nan")
        print(f"{'':>4s} {'sulfate':>14s} {fmt.format(gc_c0 - gc_c1):>14s} "
              f"{fmt.format(pc_c1 - pc_c0):>14s} {fmt.format(gc_diff):>14s} "
              f"{fmt.format(pc_diff):>14s} {ratio:>8.3e}")


if __name__ == "__main__":
    diagnose_multistep(scenario_idx=0, nstep=5)
