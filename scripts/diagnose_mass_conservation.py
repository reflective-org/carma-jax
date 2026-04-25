"""Phase 10.4e single-step mass-conservation diagnostic.

Drives ONE scenario through ONE step of make_step_full and prints
every intermediate quantity that touches the gas-particle mass
exchange. Goal: pinpoint where the ~400× factor between
gas-consumed and particle-mass-gained sneaks in.

Stages traced:
  - initial gc, pc (in MMR)
  - sulfnuc rhompe (#/cm³/z/s) and rnuclg (per seed/s)
  - gasexchange gasprod (g/cm³/z/s)
  - scale factor inside sulfate_step (gate)
  - applied gas delta and particle mass delta
  - mass conservation check
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
from carma.constants import AVG, BK, RGAS, WTMOL_H2O
from carma.gasexchange import gasexchange
from carma.nucleation.sulfnuc import sulfnuc
from carma.precision import DTYPE
from carma.sulfate_utils import wtpct_tabaz


_GWTMOL_H2SO4 = 98.0


def diagnose_one_step(scenario_idx=0):
    cfg = _fortran_matching_config()
    ppm = _compute_ppm_coefs(cfg)
    s = load_scenarios(_ROOT / "data" / "sulfate_scenarios_realistic_1000.npz")

    print(f"=== Scenario {scenario_idx} ===")
    print(f"  T={s['T'][scenario_idx]:.2f}K, p={s['p'][scenario_idx]:.2f}hPa, "
          f"RH={s['rh'][scenario_idx]:.3f}, "
          f"H2SO4={s['h2so4_pptv'][scenario_idx]:.4f}pptv, "
          f"μ={s['aerosol_mu_nm'][scenario_idx]:.2f}nm, "
          f"σg={s['aerosol_sigma_g'][scenario_idx]:.2f}")

    env = _env_for_scenario(
        s["T"][scenario_idx], s["p"][scenario_idx], s["rh"][scenario_idx],
        s["h2so4_pptv"][scenario_idx], s["aerosol_mu_nm"][scenario_idx],
        s["aerosol_sigma_g"][scenario_idx], cfg, ppm,
    )

    pc = env["pc"]            # (1, nbin, nelem)
    gc = env["gc"]            # (1, ngas)
    t = env["t"]              # (1,)
    rhoa = env["rhoa"]
    zmet = env["zmet"]
    pvapl = env["pvapl"]
    rmass = jnp.asarray(cfg.groups[0].rmass)
    rmassup = jnp.asarray(cfg.groups[0].rmassup)
    r_bins = jnp.asarray(cfg.groups[0].r)
    rmrat_val = float(cfg.groups[0].rmrat)
    nbin = cfg.nbin

    rhoa_cgs = float(rhoa[0]) / float(zmet[0])

    print()
    print("=== Initial state (CGS internal units) ===")
    print(f"  gc[H2SO4] = {float(gc[0, 1]):.6e} g/cm³/z")
    print(f"  gc[H2SO4] / rhoa = {float(gc[0, 1]) / rhoa_cgs:.6e} g/g (MMR)")
    print(f"  pc[bin] sum = {float(pc[0, :, 0].sum()):.6e} #/cm³/z")
    pc_mass_initial_cgs = float((pc[0, :, 0] * rmass).sum())
    pc_mass_mmr = pc_mass_initial_cgs / rhoa_cgs
    print(f"  pc · rmass sum = {pc_mass_initial_cgs:.6e} g/cm³/z")
    print(f"  pc · rmass / rhoa = {pc_mass_mmr:.6e} g/g (MMR)")
    print(f"  rhoa = {rhoa_cgs:.6e} g/cm³")

    # --- Run sulfnuc directly ---
    h2o_cgs = float(gc[0, 0]) / float(zmet[0])
    h2so4_cgs = float(gc[0, 1]) / float(zmet[0])
    h2so4_num = h2so4_cgs * float(AVG) / _GWTMOL_H2SO4
    h2o_num = h2o_cgs * float(AVG) / float(WTMOL_H2O)
    rvap = float(RGAS) / float(WTMOL_H2O)
    rh_internal = (h2o_cgs * rvap * float(t[0])) / pvapl
    h2o_mass_wtp = rh_internal * pvapl * float(WTMOL_H2O) / (
        float(BK) * float(t[0]) * float(AVG))
    wtp = float(wtpct_tabaz(t[0], h2o_mass_wtp, pvapl))

    rhompe_1d, rnuclg_1d = sulfnuc(
        temp=t[0], weight_percent=wtp, rh=rh_internal,
        h2so4=h2so4_num, h2so4_cgs=h2so4_cgs,
        h2o=h2o_num, h2o_cgs=h2o_cgs,
        r_bins=r_bins, rmassup=rmassup, rmrat_val=rmrat_val,
        zmet=float(zmet[0]),
    )

    print()
    print("=== sulfnuc output ===")
    print(f"  rhompe sum = {float(rhompe_1d.sum()):.6e} #/cm³/z/s")
    print(f"  rnuclg sum = {float(rnuclg_1d.sum()):.6e}  (per seed per s)")
    print(f"  rhompe · rmass sum = {float((rhompe_1d * rmass).sum()):.6e} g/cm³/z/s")
    print(f"  expected dt·rhompe·rmass at dt=1800 s = "
          f"{float((rhompe_1d * rmass).sum()) * 1800.0:.6e} g/cm³/z")

    # --- Run gasexchange directly ---
    rhompe_2d = rhompe_1d[:, None]
    rnuclg_3d = rnuclg_1d[:, None, None]
    pc_iz = pc[0, :, 0:1]
    rmass_2d = rmass[:, None]
    diffmass = (rmass_2d[:, :, None, None]
                  - rmass_2d[None, None, :, :]).astype(DTYPE)
    i2_map = jnp.arange(nbin, dtype=jnp.int32) + 1
    i2_map = jnp.where(i2_map < nbin, i2_map,
                        jnp.asarray(-1, dtype=jnp.int32))
    inuc2bin = i2_map.reshape(nbin, 1, 1)

    gasprod = gasexchange(
        pc_iz=pc_iz,
        rhompe=rhompe_2d,
        rnuclg=rnuclg_3d,
        growlg=jnp.zeros((nbin, 1), dtype=DTYPE),
        evaplg=jnp.zeros((nbin, 1), dtype=DTYPE),
        rmass=rmass_2d,
        diffmass=diffmass,
        cmf=jnp.zeros((nbin, 1), dtype=DTYPE),
        totevap=jnp.zeros((nbin, 1), dtype=bool),
        inuc2bin=inuc2bin,
        if_nuc=np.asarray([[True]]),
        ienconc=np.asarray([0]),
        igelem=np.asarray([0]),
        inucgas=np.asarray([0]),
        nnuc2elem=np.asarray([1]),
        igrowgas=np.asarray([-1]),
        ngas=1, ngroup=1, nelem=1, nbin=nbin,
    )
    print()
    print("=== gasexchange output ===")
    print(f"  gasprod[0] = {float(gasprod[0]):.6e} g/cm³/z/s")
    print(f"  expected dt·gasprod at dt=1800 s = "
          f"{float(gasprod[0]) * 1800.0:.6e} g/cm³/z (negative = consumed)")

    print()
    print("=== Comparison ===")
    rhompe_mass = float((rhompe_1d * rmass).sum())
    gasprod_mass = -float(gasprod[0])     # positive = gas being consumed
    if abs(rhompe_mass) > 0:
        ratio = gasprod_mass / rhompe_mass
        print(f"  ratio gasprod / (rhompe·rmass) = {ratio:.6e}")
        print(f"    → expect ~1.0 if mass-conserving "
              f"(both should be the gas → particle flux rate)")

    # --- Run sulfate_step_one_level and inspect outputs ---
    from carma.sulfate_step import sulfate_step_one_level
    pc_new, gc_new, diag = sulfate_step_one_level(
        pc=pc[0, :, 0],
        gc_h2so4=gc[0, 1], gc_h2o=gc[0, 0],
        temp=t[0], pvapl=pvapl, zmet=zmet[0],
        dtime=1800.0,
        r_bins=r_bins, rmass=rmass, rmassup=rmassup,
        diffmass=diffmass, inuc2bin=inuc2bin,
        rmrat_val=rmrat_val,
    )

    pc_mass_final_cgs = float((pc_new * rmass).sum())
    gc_delta = float(gc[0, 1]) - float(gc_new)
    pc_delta = pc_mass_final_cgs - pc_mass_initial_cgs
    print()
    print("=== sulfate_step_one_level applied (dt=1800 s) ===")
    print(f"  gc[H2SO4] consumed = {gc_delta:.6e} g/cm³/z")
    print(f"  pc · rmass gained  = {pc_delta:.6e} g/cm³/z")
    print(f"  ratio pc-gain / gc-loss = "
          f"{pc_delta / gc_delta if abs(gc_delta) > 0 else 0:.6e}")
    print(f"    → 1.0 = mass conserved; <<1 = mass disappearing")


if __name__ == "__main__":
    diagnose_one_step(0)
