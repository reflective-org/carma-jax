"""Sulfate nucleation + H2SO4 condensational growth — realistic scenarios.

Tests the full sulfate lifecycle:
  1. Vehkamaki (2002) binary H2SO4-H2O homogeneous nucleation
  2. H2SO4 condensational growth on sulfate particles

Runs TWO realistic scenarios:

  Stratospheric (Junge layer, ~20 km):
    T = 215 K, p = 50 hPa, H2O = 4 ppmv, H2SO4 ~= 1 pptv
    Expected: slow nucleation burst, particles grow to 0.1-0.5 um

  Tropospheric (free troposphere, ~5 km):
    T = 260 K, p = 500 hPa, H2O = 1000 ppmv (RH ~70%), H2SO4 = 10 pptv
    Expected: rapid nucleation events when H2SO4 is supplied, growth to 50nm

References:
- Junge (1961) stratospheric sulfate layer
- Kulmala et al. (2004) new particle formation in troposphere
- Vehkamaki et al. (2002) JGR 107, 4622
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from carma.precision import DTYPE
from carma.constants import *
from carma.bins import setup_bins
from carma.setup_atm import setup_atm
from carma.setup_grow import setup_grow
from carma.setup_gkern import setup_gkern
from carma.growth.growevapl import growevapl
from carma.growth.growp import growp
from carma.growth.evapp import evapp, downgevapply
from carma.solvers.psolve import psolve
from carma.solvers.totalcondensate import totalcondensate
from carma.vapor_pressure import vaporp_h2o_murphy2005, vaporp_h2so4_ayers1980
from carma.supersaturation import supersat
from carma.nucleation.sulfnucrate import binary_nuc_vehk2002, sulfnucrate
from carma.enums import GridType, ElementType
from scripts.run_nuctest_jax import compute_ppm_coefficients


# Molecular weights [g/mol]
WTMOL_H2SO4 = 98.0
WTMOL_H2O_L = 18.016


def setup_sulfate_scenario(scenario_name):
    """Return atmospheric/chemistry config for a named scenario.

    Note: rmin must be above the Kelvin critical radius for growth to
    dominate over evaporation. At T=250K with typical H2SO4 supersaturation,
    this is ~2-5 nm. We use 10 nm as the smallest bin so tests focus on
    grown particles, not new-cluster formation.
    """
    if scenario_name == "stratosphere":
        return dict(
            T=240.0,           # K (within Vehkamaki validity 230-305K)
            p_hpa=50.0,        # hPa
            z_km=20.0,         # km
            h2o_ppmv=5.0,      # ppm by volume (stratospheric value)
            h2so4_pptv=100.0,  # ppt by volume (enhanced for clear growth)
            rmin_cm=5e-6,      # 50 nm smallest bin (above Kelvin barrier)
            rmrat=2.0,         # finer mass ratio
            nbin=32,
            dtime=60.0,        # s
            nstep=200,         # 200 min total (~3.3h)
            title="Stratosphere (Junge layer, ~20 km)",
            h2so4_source_rate=0.0,      # no continuous source
            seed_N_per_cm3=5.0,         # seed particles/cm3 (Junge layer)
            seed_bin_fraction=0.2,      # at bin nbin*0.2 (~0.1um radius)
        )
    elif scenario_name == "troposphere":
        return dict(
            T=270.0,           # K (lower free troposphere)
            p_hpa=700.0,       # hPa
            z_km=3.0,          # km
            h2o_ppmv=3000.0,   # ppm — ~80% RH at 270K
            h2so4_pptv=50.0,   # ppt by volume (polluted boundary layer)
            rmin_cm=5e-7,      # 5 nm smallest bin (closer to Kelvin barrier)
            rmrat=2.0,
            nbin=32,
            dtime=10.0,        # s
            nstep=360,         # 1 hour
            title="Free troposphere (~3 km)",
            h2so4_source_rate=1e7,      # photochemical source [cm-3/s]
            seed_N_per_cm3=500.0,       # pre-existing boundary layer aerosol
            seed_bin_fraction=0.35,     # at bin nbin*0.35 (~35 nm, above Kelvin)
        )
    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")


def run_sulfate_scenario(config, outdir):
    """Run one sulfate scenario and generate diagnostic plots.

    Configuration: 1 group (sulfate aerosol), 1 element (I_INVOLATILE),
    1 gas (H2SO4), 1 secondary gas for H2O.

    For simplicity, we'll run a simplified test:
    - Single group (sulfate) with I_VOLATILE element so it grows by H2SO4
    - Nucleation adds new particles to first few bins
    - Growth moves them to larger sizes
    """
    NBIN = config["nbin"]
    NGROUP = 1
    NELEM = 1
    NGAS = 1  # H2SO4 only for growth
    NZ = 1

    rmin = config["rmin_cm"]
    rmrat = config["rmrat"]
    rho_aer = 1.78  # wet sulfate aerosol particle density [g/cm³]
    rho_sol = 1.38  # H2SO4 solute density

    r1, rmass1, vol1, dr1, dm1, rup1, rlow1, rmassup1 = setup_bins(rmin, rmrat, NBIN, rho_aer)
    print(f"  Bin range: {float(r1[0])*1e7:.3f} to {float(r1[-1])*1e4:.2f} um")

    # --- Element/Group config ---
    igroup_arr = np.array([0], dtype=np.int32)
    itype_arr = np.array([int(ElementType.I_VOLATILE)], dtype=np.int32)
    igrowgas_arr = np.array([0], dtype=np.int32)  # grows by gas 0 (H2SO4)
    ienconc_arr = np.array([0], dtype=np.int32)
    is_ice_arr = np.array([False])
    gwtmol_arr = np.array([WTMOL_H2SO4])

    # --- PPM coefficients ---
    pratt, prat, pden1, palr = compute_ppm_coefficients(
        [np.array(dm1)], [np.array(rmass1)], [np.array(rmassup1)],
        rmrat, NBIN, NGROUP)

    rmass_2d = rmass1[:, None]
    dm_2d = dm1[:, None]
    rup_wet = jnp.broadcast_to(rup1[None, :, None], (NZ, NBIN, NGROUP))
    rlow_wet = jnp.broadcast_to(rlow1[None, :, None], (NZ, NBIN, NGROUP))
    r_wet = jnp.broadcast_to(r1[None, :, None], (NZ, NBIN, NGROUP))
    re = jnp.zeros((NZ, NBIN, NGROUP), dtype=DTYPE)

    # --- Atmosphere ---
    T_init = config["T"]
    p_pa = config["p_hpa"] * 100
    t = jnp.array([DTYPE(T_init)])
    p_cgs = jnp.array([DTYPE(p_pa) * RPA2CGS])
    zc = jnp.array([DTYPE(config["z_km"] * 1e3) * RM2CGS])
    # Rough 100m box
    zl = jnp.array([DTYPE((config["z_km"] * 1e3 - 50.0)) * RM2CGS,
                     DTYPE((config["z_km"] * 1e3 + 50.0)) * RM2CGS])
    rho_air = float(p_pa * 10.0) / (float(R_AIR) * T_init)
    pl = jnp.array([(p_pa + 50 * rho_air * float(GRAV) / 100) * float(RPA2CGS),
                     (p_pa - 50 * rho_air * float(GRAV) / 100) * float(RPA2CGS)])

    rhoa, dz, zmet, zmetl, rmu, thcond, rhoa_wet = setup_atm(
        t, p_cgs, pl, zc, zl, GridType.I_CART)

    print(f"  Atm: T={T_init}K, p={config['p_hpa']} hPa, rhoa={float(rhoa[0]/zmet[0]):.3e} g/cm³")

    # --- Initial conditions ---
    # H2O gas concentration (from ppmv)
    h2o_ppmv = config["h2o_ppmv"]
    # ppmv → mass mixing ratio: mmr = ppmv * (Mw/Mair) * 1e-6
    h2o_mmr = h2o_ppmv * 1e-6 * (WTMOL_H2O_L / WTMOL_AIR)
    h2o_conc = h2o_mmr * float(rhoa[0] / zmet[0])  # g/cm³

    # H2SO4 gas concentration (from pptv)
    h2so4_pptv = config["h2so4_pptv"]
    h2so4_mmr = h2so4_pptv * 1e-12 * (WTMOL_H2SO4 / WTMOL_AIR)
    h2so4_conc = h2so4_mmr * float(rhoa[0] / zmet[0])  # g/cm³
    h2so4_numconc = h2so4_conc / WTMOL_H2SO4 * float(AVG)  # molec/cm³

    print(f"  H2O:   {h2o_ppmv:.1f} ppmv → {h2o_conc:.3e} g/cm³")
    print(f"  H2SO4: {h2so4_pptv:.2f} pptv → {h2so4_numconc:.3e} molec/cm³")

    # Seed particles (above Kelvin barrier)
    pc = jnp.full((NZ, NBIN, NELEM), SMALL_PC, dtype=DTYPE)
    seed_N = config["seed_N_per_cm3"]
    seed_bin = int(NBIN * config["seed_bin_fraction"])
    print(f"  Seed: {seed_N} #/cm³ at bin {seed_bin} (r={float(r1[seed_bin])*1e4:.3f} um)")
    pc = pc.at[0, seed_bin, 0].set(DTYPE(seed_N) * zmet[0])
    gc = jnp.array([[DTYPE(h2so4_conc) * zmet[0]]])  # H2SO4 gas (×zmet for internal units)

    DTIME = config["dtime"]
    NSTEP = config["nstep"]

    # --- Run simulation ---
    print(f"  Running {NSTEP} steps at dt={DTIME}s ({NSTEP*DTIME/60:.0f} min total)...")

    history = {
        "times": [0.0],
        "h2so4_conc": [h2so4_numconc],
        "nuc_rate": [0.0],
        "total_N": [float(pc[0, :, 0].sum() / zmet[0])],
        "total_mass": [float((pc[0, :, 0] * rmass1).sum() / zmet[0])],
        "dist_snapshots": [np.array(pc[0, :, 0] / zmet[0])],
        "snapshot_times": [0.0],
    }

    # Snapshot every N steps
    snap_every = max(1, NSTEP // 5)

    iz = 0
    for istep in range(NSTEP):
        # --- Vapor pressures ---
        pvapl_h2o, pvapi_h2o = vaporp_h2o_murphy2005(t)
        # Ayers H2SO4 vapor pressure (using our H2O result for RH)
        # For this test, assume H2O stays fixed (large reservoir)
        h2o_gc_cgs = jnp.array([DTYPE(h2o_conc)])
        pvapl_h2so4, pvapi_h2so4 = vaporp_h2so4_ayers1980(
            t, h2o_gc_cgs, pvapl_h2o, zmet)

        # H2SO4 supersaturation
        gwtmol_h2so4 = DTYPE(WTMOL_H2SO4)
        ssl_h2so4, ssi_h2so4 = supersat(
            t, gc[:, 0], pvapl_h2so4, pvapi_h2so4, gwtmol_h2so4, zmet)

        # RH for Vehkamaki (fraction 0-1)
        h2o_rvap = RGAS / DTYPE(WTMOL_H2O_L)
        rh = h2o_conc * h2o_rvap * T_init / float(pvapl_h2o[0])
        rh = float(jnp.clip(rh, 0.001, 1.0))

        # --- Nucleation: Vehkamaki ---
        h2so4_current = float(gc[0, 0] / rhoa[0]) * float(rhoa[0] / zmet[0]) / WTMOL_H2SO4 * float(AVG)
        nucrate_cgs, mass_cluster, r_cluster = binary_nuc_vehk2002(
            t[0], DTYPE(rh), DTYPE(h2so4_current), DTYPE(WTMOL_H2SO4))
        nucrate = float(nucrate_cgs)  # #/cm³/s

        # Determine target bin for nucleated cluster
        rmass_np = np.array(rmass1)
        if float(mass_cluster) < float(rmassup1[0]):
            nucbin = 0
        else:
            nucbin = int(np.clip(
                1 + np.floor(np.log(float(mass_cluster) / float(rmassup1[0]))
                             / np.log(rmrat)),
                0, NBIN - 1))

        # --- Setup growth kernels ---
        # Use gas H2SO4 as growth gas
        diffus_arr, rlhe_arr, rlhm_arr = setup_grow(
            t, p_cgs, rhoa, zmet,
            igash2o=-1, igash2so4=0,  # H2SO4 is gas 0
            ngas=NGAS, do_cnst_rlh=False)

        rrat = jnp.ones((NBIN, NGROUP), dtype=DTYPE)
        eshape_arr = jnp.array([1.0])  # sphere
        surfctwa, akelvin, akelvini, gro, gro1, gro2, ft_arr, thcondnc = setup_gkern(
            t, p_cgs, rhoa, zmet, rmu, thcond, diffus_arr, rlhe_arr, rlhm_arr,
            re, r_wet, rlow_wet, rrat, eshape_arr, is_ice_arr,
            gwtmol_arr, igrowgas_arr, 1.0, 1.0, 1.0,
            NBIN, NGROUP, NGAS)

        # Broadcast to (NZ, NGAS)
        pvapl_2d = pvapl_h2so4[:, None]
        pvapi_2d = pvapi_h2so4[:, None]
        supsatl_2d = ssl_h2so4[:, None]
        supsati_2d = ssi_h2so4[:, None]

        pconmax = jnp.zeros((NZ, NGROUP), dtype=DTYPE)
        pconmax = pconmax.at[:, 0].set(jnp.max(pc[:, :, 0], axis=1) / zmet)

        # --- Save condensate, then growevapl ---
        prev_ice, prev_liq = totalcondensate(
            pc, rmass_2d, igroup_arr, is_ice_arr, igrowgas_arr,
            NBIN, NGROUP, NGAS, iz)

        growlg = jnp.zeros((NBIN, NGROUP), dtype=DTYPE)
        evaplg = jnp.zeros((NBIN, NGROUP), dtype=DTYPE)
        growlg, evaplg = growevapl(
            pc, growlg, evaplg,
            supsatl_2d, supsati_2d, pvapl_2d, pvapi_2d,
            akelvin, akelvini, gro, gro1,
            rup_wet, rmass_2d, dm_2d, pconmax,
            pratt, prat, pden1, palr,
            is_ice_arr, igrowgas_arr, ienconc_arr,
            DTIME, iz, NBIN, NGROUP)

        # --- Apply nucleation: add particles to nucbin ---
        # rhompe = homogeneous nucleation production rate [#/cm³/s] × zmet
        rhompe_arr = jnp.zeros((NBIN, NELEM), dtype=DTYPE)
        if nucrate > 0:
            # Scale by source term for troposphere
            rhompe_arr = rhompe_arr.at[nucbin, 0].set(DTYPE(nucrate) * zmet[0])

        # Deplete gas from nucleation (cluster mass × rate)
        gas_loss_nuc = float(nucrate) * float(mass_cluster)  # g/cm³/s

        # --- Inner loop: growp + psolve ---
        growpe_arr = jnp.zeros((NBIN, NELEM), dtype=DTYPE)
        evappe_arr = jnp.zeros((NBIN, NELEM), dtype=DTYPE)
        rnucpe_arr = jnp.zeros((NBIN, NELEM), dtype=DTYPE)
        pc_nucl = jnp.zeros_like(pc)
        rnuclg_zero = jnp.zeros((NBIN, NGROUP, NGROUP), dtype=DTYPE)

        for ibin in range(NBIN):
            growpe_arr = growp(pc, growpe_arr, growlg, pconmax,
                               iz, ibin, 0, 0, 0)
            pc, pc_nucl = psolve(
                pc, pc_nucl, growpe_arr, evappe_arr, rnucpe_arr, rhompe_arr,
                growlg, evaplg, rnuclg_zero, DTIME, iz, ibin, 0, 0, NGROUP)

        # --- Evaporation ---
        evappe_arr = jnp.zeros((NBIN, NELEM), dtype=DTYPE)
        evappe_arr = evapp(
            pc, evappe_arr, evaplg, pconmax, ienconc_arr, itype_arr,
            igroup_arr, iz, NBIN, NGROUP, NELEM)
        pc = downgevapply(pc, evappe_arr,
                          jnp.zeros((NBIN, NELEM), dtype=DTYPE),
                          DTIME, iz, NBIN, NELEM)

        # --- Gas solver ---
        curr_ice, curr_liq = totalcondensate(
            pc, rmass_2d, igroup_arr, is_ice_arr, igrowgas_arr,
            NBIN, NGROUP, NGAS, iz)
        # Gas balance: condensation + nucleation loss
        condensation_loss = (curr_ice[0] - prev_ice[0]) + (curr_liq[0] - prev_liq[0])
        gasprod = -condensation_loss / DTIME - gas_loss_nuc * zmet[0]

        # Optional source (troposphere)
        h2so4_src = config.get("h2so4_source_rate", 0.0)
        # Source is in molec/cm³/s, convert to g/cm³/s
        src_g = h2so4_src * WTMOL_H2SO4 / float(AVG)
        gasprod += src_g * zmet[0]

        gc = gc.at[iz, 0].add(DTIME * gasprod)
        # Don't let gas go negative
        gc = jnp.maximum(gc, DTYPE(0.0))

        # --- Store history ---
        total_N = float(pc[0, :, 0].sum() / zmet[0])
        total_mass = float((pc[0, :, 0] * rmass1).sum() / zmet[0])
        h2so4_conc_now = float(gc[0, 0] / zmet[0])
        h2so4_numconc_now = h2so4_conc_now / WTMOL_H2SO4 * float(AVG)

        history["times"].append(DTIME * (istep + 1))
        history["h2so4_conc"].append(h2so4_numconc_now)
        history["nuc_rate"].append(nucrate)
        history["total_N"].append(total_N)
        history["total_mass"].append(total_mass)

        if (istep + 1) % snap_every == 0 or istep == NSTEP - 1:
            history["dist_snapshots"].append(np.array(pc[0, :, 0] / zmet[0]))
            history["snapshot_times"].append(DTIME * (istep + 1))
            print(f"    t={DTIME*(istep+1)/60:.1f}min: N={total_N:.2e} #/cm³, "
                  f"mass={total_mass:.2e} g/cm³, "
                  f"H2SO4={h2so4_numconc_now:.2e} /cm³, "
                  f"nuc={nucrate:.2e} /cm³/s", flush=True)

    return history, r1, rmass1


def plot_scenario(history, r_cm, rmass, config, outpath):
    """Generate diagnostic plots for a scenario."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    times_min = np.array(history["times"]) / 60.0
    r_um = np.array(r_cm) * 1e4

    # Row 1: Time series
    ax = axes[0, 0]
    ax.semilogy(times_min, history["total_N"], "b-", lw=2)
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Total N [#/cm³]")
    ax.set_title("Particle Number Concentration")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.semilogy(times_min, np.maximum(history["total_mass"], 1e-30), "b-", lw=2)
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Total mass [g/cm³]")
    ax.set_title("Particle Mass Concentration")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.semilogy(times_min, np.maximum(history["h2so4_conc"], 1e-5), "g-", lw=2)
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("H2SO4 [molec/cm³]")
    ax.set_title("H2SO4 Gas Concentration")
    ax.grid(True, alpha=0.3)

    # Row 2: distributions
    ax = axes[1, 0]
    ax.semilogy(times_min[1:], np.maximum(history["nuc_rate"][1:], 1e-10), "r-", lw=2)
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Nuc rate [#/cm³/s]")
    ax.set_title("Vehkamaki Nucleation Rate")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for i, (t_s, dist) in enumerate(zip(history["snapshot_times"], history["dist_snapshots"])):
        ax.semilogy(r_um, np.maximum(dist, 1e-20), "-o", ms=3,
                    label=f"t={t_s/60:.0f}min", alpha=0.7)
    ax.set_xlabel("Radius [um]")
    ax.set_ylabel("dN/dbin [#/cm³]")
    ax.set_title("Size Distribution Evolution")
    ax.legend(fontsize=8)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    # Mass distribution
    for i, (t_s, dist) in enumerate(zip(history["snapshot_times"], history["dist_snapshots"])):
        mass_dist = dist * np.array(rmass)
        ax.semilogy(r_um, np.maximum(mass_dist, 1e-30), "-o", ms=3,
                    label=f"t={t_s/60:.0f}min", alpha=0.7)
    ax.set_xlabel("Radius [um]")
    ax.set_ylabel("dM/dbin [g/cm³]")
    ax.set_title("Mass Distribution Evolution")
    ax.legend(fontsize=8)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Sulfate Nucleation + H2SO4 Condensation — {config['title']}\n"
                 f"T={config['T']}K, p={config['p_hpa']} hPa, "
                 f"H2O={config['h2o_ppmv']} ppmv, H2SO4_init={config['h2so4_pptv']} pptv",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def main():
    outdir = Path("plots/sulfate_test")
    outdir.mkdir(parents=True, exist_ok=True)

    for scenario in ["stratosphere", "troposphere"]:
        print(f"\n{'='*60}")
        print(f"Running {scenario.upper()} scenario")
        print(f"{'='*60}")
        config = setup_sulfate_scenario(scenario)
        history, r_cm, rmass = run_sulfate_scenario(config, outdir)
        plot_path = outdir / f"sulfate_{scenario}.png"
        plot_scenario(history, r_cm, rmass, config, plot_path)
        print(f"  Saved: {plot_path}")

    print(f"\nAll plots in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
