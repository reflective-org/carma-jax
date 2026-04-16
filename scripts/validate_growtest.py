"""Validate JAX growth against Fortran growtest benchmark.

Growtest: single level, ice crystal growth with thermodynamics.
TTL conditions: T=190K, p=9000Pa, H2O=3.5ppm, 0.1 cm^-3 ice in bin 1.
NBIN=24, rmin=1e-4 cm (1um), rmrat=2, dt=100s, 50 steps.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from carma.precision import DTYPE
from carma.constants import (
    BK, PI, RGAS, R_AIR, GRAV, RHO_I, WTMOL_H2O, CP,
    RM2CGS, RPA2CGS, SMALL_PC, FEW_PC,
)
from carma.bins import setup_bins
from carma.setup_atm import setup_atm
from carma.enums import GridType, ElementType, VaporPressureRoutine
from carma.vapor_pressure import vaporp_h2o_murphy2005
from carma.supersaturation import supersat
from carma.setup_grow import setup_grow
from carma.setup_gkern import setup_gkern
from carma.setup_vf import setup_vf
from carma.growth.growevapl import growevapl
from carma.growth.growp import growp
from carma.solvers.psolve import psolve
from carma.solvers.gsolve import gsolve
from carma.solvers.tsolve import tsolve
from carma.solvers.totalcondensate import totalcondensate


def parse_growtest_bench(filepath):
    """Parse carma_growtest.txt benchmark."""
    with open(filepath) as f:
        lines = f.readlines()
    idx = 0
    parts = lines[idx].split()
    ngroup, nelem, nbin, ngas = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
    idx += 1

    radii = np.zeros(nbin)
    rmass_f = np.zeros(nbin)
    for i in range(nbin):
        parts = lines[idx].split()
        radii[i] = float(parts[2]) * 1e-4  # um to cm
        rmass_f[i] = float(parts[3])
        idx += 1

    # Parse timestep blocks
    times = []
    t_changes = []
    rlheats = []
    mmr_data = []
    gas_data = []

    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue

        # Time value
        time_val = float(line)
        times.append(time_val)
        idx += 1

        # t_change, rlheat
        parts = lines[idx].split()
        t_changes.append(float(parts[0]))
        rlheats.append(float(parts[1]))
        idx += 1

        # Per-bin mmr
        mmr = np.zeros(nbin)
        for i in range(nbin):
            parts = lines[idx].split()
            mmr[i] = float(parts[2])
            idx += 1
        mmr_data.append(mmr)

        # Gas state
        parts = lines[idx].split()
        gas_mmr = float(parts[1])
        satliq = float(parts[2])
        satice = float(parts[3])
        gas_data.append((gas_mmr, satliq, satice))
        idx += 1

    return {
        "nbin": nbin, "radii": radii, "rmass": rmass_f,
        "times": np.array(times),
        "t_changes": np.array(t_changes),
        "rlheats": np.array(rlheats),
        "mmr": np.array(mmr_data),
        "gas": gas_data,
    }


def run_jax_growtest():
    """Run JAX growth test matching Fortran growtest setup."""
    NBIN, NELEM, NGROUP, NGAS, NZ = 24, 1, 1, 1, 1
    rmin_cm = 1e-4  # 1 um
    rmrat = 2.0
    rho_particle = float(RHO_I)  # ice density
    dtime = 100.0
    nstep = 50

    # Setup bins
    r, rmass, vol, dr, dm, rup, rlow, rmassup = setup_bins(rmin_cm, rmrat, NBIN, rho_particle)
    rmass_np = np.array(rmass)

    # TTL atmosphere: T=190K, p=9000Pa
    t_init = DTYPE(190.0)
    p_pa = DTYPE(9000.0)
    zc_m = DTYPE(17000.0)
    deltaz_m = DTYPE(100.0)

    # Convert to CGS
    p_cgs = p_pa * RPA2CGS
    zc = jnp.array([zc_m * RM2CGS])
    zl = jnp.array([(zc_m - deltaz_m) * RM2CGS, (zc_m + deltaz_m) * RM2CGS])

    # Compute pressure at edges from hydrostatic
    rho_air_mks = float(p_pa * DTYPE(10.0)) / (float(R_AIR) * float(t_init)) * DTYPE(1e-3) * DTYPE(1e6)
    pl_1 = p_pa - (zc_m - deltaz_m - zc_m) * rho_air_mks * (float(GRAV) / DTYPE(100.0))
    pl_2 = p_pa + (zc_m + deltaz_m - zc_m) * rho_air_mks * (float(GRAV) / DTYPE(100.0))
    # Actually: pl(1) = p - (zl(1)-zc)*rho*(g/100), pl(2) = p - (zl(2)-zc)*rho*(g/100)
    pl_1 = p_pa - (-deltaz_m) * rho_air_mks * (float(GRAV) / DTYPE(100.0))
    pl_2 = p_pa - (deltaz_m) * rho_air_mks * (float(GRAV) / DTYPE(100.0))
    pl = jnp.array([pl_1 * RPA2CGS, pl_2 * RPA2CGS])

    t = jnp.array([t_init])
    p = jnp.array([p_cgs])
    t_orig = float(t_init)

    rhoa, dz, zmet, zmetl, rmu, thcond, rhoa_wet = setup_atm(
        t, p, pl, zc, zl, GridType.I_CART
    )

    # Initial conditions
    # Particles: 0.1 cm^-3 in bin 1 → mmr = N * rmass / rhoa_air
    # In Fortran: mmr(1,1,1) = (0.1 * rmass(1) * 1e3) / rho_air_cgs
    # We work in number concentration directly
    N0 = DTYPE(0.1)  # cm^-3 → but need to account for zmet
    pc = jnp.full((NZ, NBIN, NELEM), SMALL_PC, dtype=DTYPE)
    pc = pc.at[0, 0, 0].set(N0 * zmet[0])  # scale by zmet for internal units

    # Gas: 3.5 ppm H2O → mmr = 3.5e-6 g/g → gc = mmr * rhoa
    mmr_gas_init = DTYPE(3.5e-6)
    gc = jnp.array([[mmr_gas_init * rhoa[0]]])  # (NZ, NGAS) in g/cm^2/z

    # Growth setup
    gwtmol_arr = jnp.array([float(WTMOL_H2O)])
    igrowgas_arr = jnp.array([0])  # element 0 grows by gas 0
    is_ice_arr = jnp.array([True])
    ienconc_arr = jnp.array([0])

    diffus, rlhe, rlhm = setup_grow(
        t, p, rhoa, zmet, igash2o=0, igash2so4=-1, ngas=NGAS, do_cnst_rlh=False,
    )

    # Vapor pressure
    pvapl, pvapi = vaporp_h2o_murphy2005(t)
    pvapl = pvapl[None, :]  # (1,) -> (NZ, NGAS) = (1, 1)
    pvapi = pvapi[None, :]

    # Supersaturation
    supsatl, supsati = supersat(t, gc[:, 0], pvapl[:, 0], pvapi[:, 0], float(WTMOL_H2O), zmet)
    supsatl = supsatl[:, None]  # (NZ, 1) = (1, 1)
    supsati = supsati[:, None]

    # Fall velocity and growth kernels
    r_wet = jnp.broadcast_to(r[None, :, None], (NZ, NBIN, NGROUP))
    rhop_wet = jnp.full((NZ, NBIN, NGROUP), rho_particle, dtype=DTYPE)
    rrat_arr = jnp.ones((NBIN, NGROUP), dtype=DTYPE)
    rprat_arr = jnp.ones((NBIN, NGROUP), dtype=DTYPE)

    vf, re, bpm = setup_vf(None, t, rhoa, zmet, rmu, r_wet, rhop_wet, rrat_arr, rprat_arr)

    # Growth kernel setup
    rlow_wet = jnp.broadcast_to(rlow[None, :, None], (NZ, NBIN, NGROUP))
    eshape_arr = jnp.array([1.0])

    gro, gro1, gro2 = jnp.zeros((NZ, NBIN, NGROUP), dtype=DTYPE), \
                       jnp.zeros((NZ, NBIN, NGROUP), dtype=DTYPE), \
                       jnp.zeros((NZ, NGROUP), dtype=DTYPE)

    # Compute growth kernels using setup_gkern
    from carma.setup_gkern import setup_gkern
    surfctwa, akelvin, akelvini, gro, gro1, gro2, ft_arr, thcondnc = setup_gkern(
        t, p, rhoa, zmet, rmu, thcond, diffus, rlhe, rlhm,
        re, r_wet, rlow_wet, rrat_arr, eshape_arr, is_ice_arr,
        gwtmol_arr, igrowgas_arr, float(1.0), float(1.0), float(1.0),
        NBIN, NGROUP, NGAS,
    )

    # PPM coefficients (precomputed from bin structure)
    # pratt, prat, pden1, palr — these depend on bin mass ratios
    # For uniform rmrat, simplify
    pratt = jnp.zeros((3, NBIN, NGROUP), dtype=DTYPE)
    prat = jnp.zeros((4, NBIN, NGROUP), dtype=DTYPE)
    pden1 = jnp.zeros((NBIN, NGROUP), dtype=DTYPE)
    palr = jnp.zeros((4, NGROUP), dtype=DTYPE)

    # Compute PPM coefficients — EXACT Fortran formulas from carma_mod.F90 lines 532-569
    dm_np = np.array(dm)
    rmass_np2 = np.array(rmass)
    rmassup_np = np.array(rmassup)
    rmrat_val = 2.0

    for ig in range(NGROUP):
        # pratt: gradient coefficients (Fortran domain: i=2..NBIN-1, 1-based = i=1..NBIN-2, 0-based)
        for ibin in range(1, NBIN - 1):
            dm_im1 = dm_np[ibin - 1]
            dm_i = dm_np[ibin]
            dm_ip1 = dm_np[ibin + 1]
            pratt = pratt.at[0, ibin, ig].set(dm_i / (dm_im1 + dm_i + dm_ip1))
            pratt = pratt.at[1, ibin, ig].set((2.0 * dm_im1 + dm_i) / (dm_ip1 + dm_i))
            pratt = pratt.at[2, ibin, ig].set((2.0 * dm_ip1 + dm_i) / (dm_im1 + dm_i))

        # prat, pden1: polynomial coefficients (Fortran domain: i=2..NBIN-2, 1-based = i=1..NBIN-3, 0-based)
        for ibin in range(1, NBIN - 2):
            dm_im1 = dm_np[ibin - 1]
            dm_i = dm_np[ibin]
            dm_ip1 = dm_np[ibin + 1]
            dm_ip2 = dm_np[min(ibin + 2, NBIN - 1)]
            prat = prat.at[0, ibin, ig].set(dm_i / (dm_i + dm_ip1))
            prat = prat.at[1, ibin, ig].set(2.0 * dm_ip1 * dm_i / (dm_i + dm_ip1))
            prat = prat.at[2, ibin, ig].set((dm_im1 + dm_i) / (2.0 * dm_i + dm_ip1))
            prat = prat.at[3, ibin, ig].set((dm_ip2 + dm_ip1) / (2.0 * dm_ip1 + dm_i))
            pden1 = pden1.at[ibin, ig].set(dm_im1 + dm_i + dm_ip1 + dm_ip2)

        # palr: edge coefficients — EXACT from Fortran carma_mod.F90 lines 557-568
        # palr(1) = (rmassup(1) - rmass(1)) / (rmass(2) - rmass(1))  [Fortran 1-based]
        # palr(2) = (rmassup(1)/rmrat - rmass(1)) / (rmass(2) - rmass(1))
        # palr(3) = (rmassup(NBIN-1) - rmass(NBIN-1)) / (rmass(NBIN) - rmass(NBIN-1))
        # palr(4) = (rmassup(NBIN) - rmass(NBIN-1)) / (rmass(NBIN) - rmass(NBIN-1))
        denom_low = rmass_np2[1] - rmass_np2[0]
        denom_high = rmass_np2[NBIN - 1] - rmass_np2[NBIN - 2]
        palr = palr.at[0, ig].set((rmassup_np[0] - rmass_np2[0]) / denom_low)
        palr = palr.at[1, ig].set((rmassup_np[0] / rmrat_val - rmass_np2[0]) / denom_low)
        palr = palr.at[2, ig].set((rmassup_np[NBIN - 2] - rmass_np2[NBIN - 2]) / denom_high)
        palr = palr.at[3, ig].set((rmassup_np[NBIN - 1] - rmass_np2[NBIN - 2]) / denom_high)

    # Time integration
    results = {
        "times": [0.0],
        "t_change": [0.0],
        "rlheat_val": [0.0],
        "mmr_bins": [np.array(pc[0, :, 0]) / float(zmet[0]) * rmass_np / float(rhoa[0])],
        "gas_mmr": [float(gc[0, 0] / rhoa[0])],
    }

    from carma.newstate_calc import newstate_calc_growth

    iz = 0
    rup_wet_arr = jnp.broadcast_to(rup[None, :, None], (NZ, NBIN, NGROUP))
    rlow_wet_arr = jnp.broadcast_to(rlow[None, :, None], (NZ, NBIN, NGROUP))
    rmass_2d = rmass[:, None] * jnp.ones((1, NGROUP), dtype=DTYPE)
    dm_2d = dm[:, None] * jnp.ones((1, NGROUP), dtype=DTYPE)

    for istep in range(nstep):
        pc, gc, t, rlheat_val, nsub = newstate_calc_growth(
            pc, gc, t, iz, dtime,
            rhoa, zmet, rlhe, rlhm, diffus,
            akelvin, akelvini, gro, gro1, gro2,
            rup_wet_arr, rmass_2d, dm_2d, rlow_wet_arr,
            pratt, prat, pden1, palr,
            is_ice_arr, igrowgas_arr, ienconc_arr,
            jnp.array([0]),  # igroup_arr
            gwtmol_arr,
            NBIN, NGROUP, NGAS, NELEM,
            minsubsteps=1, maxsubsteps=128, maxretries=10,
            dt_threshold=0.0,
            ds_threshold_arr=jnp.array([-0.1]),  # sign-change check
            scale_threshold=1.0,
        )

        time = (istep + 1) * dtime
        mmr_bins = np.array(pc[0, :, 0]) / float(zmet[0]) * rmass_np / float(rhoa[0])
        results["times"].append(time)
        results["t_change"].append(float(t[0]) - t_orig)
        results["rlheat_val"].append(rlheat_val)
        results["mmr_bins"].append(mmr_bins)
        results["gas_mmr"].append(float(gc[0, 0] / rhoa[0]))
        if istep < 5 or istep % 10 == 0:
            print(f"  Step {istep+1}/{nstep}: dT={float(t[0])-t_orig:.4e}K, substeps={nsub}", flush=True)

    results["times"] = np.array(results["times"])
    results["t_change"] = np.array(results["t_change"])
    results["rlheat_val"] = np.array(results["rlheat_val"])
    results["mmr_bins"] = np.array(results["mmr_bins"])
    results["gas_mmr"] = np.array(results["gas_mmr"])
    results["radii"] = np.array(r)
    results["rmass"] = rmass_np

    return results


def main():
    bench_path = Path(__file__).parent.parent.parent / "original-carma" / "CARMA" / "tests" / "bench" / "carma_growtest.txt"
    print("Parsing Fortran benchmark...", flush=True)
    fortran = parse_growtest_bench(bench_path)
    print(f"  {fortran['nbin']} bins, {len(fortran['times'])} timesteps", flush=True)

    print("Running JAX growth test...", flush=True)
    jax_data = run_jax_growtest()
    print(f"  {len(jax_data['times'])} timesteps", flush=True)

    # Compare
    nsteps = min(len(fortran["times"]), len(jax_data["times"]))
    outdir = Path(__file__).parent.parent / "plots" / "growtest_validation"
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Temperature change
    ax = axes[0, 0]
    ax.plot(fortran["times"][:nsteps], fortran["t_changes"][:nsteps], "ko-", ms=3, label="Fortran")
    ax.plot(jax_data["times"][:nsteps], jax_data["t_change"][:nsteps], "r^--", ms=3, label="JAX")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("T - T_init [K]")
    ax.set_title("Temperature Change from Latent Heating")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Gas MMR
    ax = axes[0, 1]
    f_gas = [g[0] for g in fortran["gas"][:nsteps]]
    ax.plot(fortran["times"][:nsteps], f_gas, "ko-", ms=3, label="Fortran")
    ax.plot(jax_data["times"][:nsteps], jax_data["gas_mmr"][:nsteps], "r^--", ms=3, label="JAX")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("H2O MMR [g/g]")
    ax.set_title("Water Vapor Mixing Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Size distribution at final step
    ax = axes[1, 0]
    r_um = fortran["radii"] * 1e4
    ax.semilogy(r_um, fortran["mmr"][-1], "ko-", ms=4, label="Fortran (final)")
    ax.semilogy(jax_data["radii"] * 1e4, jax_data["mmr_bins"][-1], "r^--", ms=4, label="JAX (final)")
    ax.semilogy(r_um, fortran["mmr"][0], "k:", alpha=0.5, label="Initial")
    ax.set_xlabel("Radius [um]")
    ax.set_ylabel("MMR [g/g]")
    ax.set_title("Size Distribution")
    ax.legend()
    ax.set_ylim(bottom=1e-20)
    ax.grid(True, alpha=0.3)

    # 4. Total mass conservation
    ax = axes[1, 1]
    f_total = np.array([fortran["mmr"][i].sum() + fortran["gas"][i][0] for i in range(nsteps)])
    j_total = jax_data["mmr_bins"][:nsteps].sum(axis=1) + jax_data["gas_mmr"][:nsteps]
    ax.plot(fortran["times"][:nsteps], (f_total / f_total[0] - 1) * 100, "ko-", ms=3, label="Fortran")
    ax.plot(jax_data["times"][:nsteps], (j_total / j_total[0] - 1) * 100, "r^--", ms=3, label="JAX")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("(Total/Total_init - 1) [%]")
    ax.set_title("Total Mass Conservation (particles + gas)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Growth Test Validation: Fortran vs JAX", fontsize=14)
    fig.tight_layout()
    fig.savefig(outdir / "growtest_comparison.png", dpi=150)
    plt.close(fig)

    # --- Figure 2: Size distribution evolution ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    step_indices = [0, 2, 5, 10, 25, 50]
    step_labels = ["t=0s", "t=200s", "t=500s", "t=1000s", "t=2500s", "t=5000s"]

    for ax, si, label in zip(axes.flat, step_indices, step_labels):
        if si < nsteps:
            f_mmr = fortran["mmr"][si]
            j_mmr = jax_data["mmr_bins"][si]
            ax.semilogy(r_um, f_mmr, "ko-", ms=4, lw=1.5, label="Fortran")
            ax.semilogy(jax_data["radii"] * 1e4, j_mmr, "r^--", ms=4, lw=1.5, label="JAX")
            ax.set_xlabel("Radius [um]")
            ax.set_ylabel("MMR [g/g]")
            ax.set_title(label)
            ax.set_ylim(bottom=1e-20, top=1e-7)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Size Distribution Evolution: Fortran vs JAX", fontsize=14)
    fig.tight_layout()
    fig.savefig(outdir / "growtest_size_evolution.png", dpi=150)
    plt.close(fig)

    # --- Figure 3: Per-bin time evolution for selected bins ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    selected_bins = [0, 3, 6, 9, 12, 15]

    for ax, ib in zip(axes.flat, selected_bins):
        if ib < fortran["nbin"]:
            f_bin = fortran["mmr"][:nsteps, ib]
            j_bin = jax_data["mmr_bins"][:nsteps, ib]
            ax.plot(fortran["times"][:nsteps], f_bin, "k-", lw=2, label="Fortran")
            ax.plot(jax_data["times"][:nsteps], j_bin, "r--", lw=2, label="JAX")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("MMR [g/g]")
            ax.set_title(f"Bin {ib+1} (r={r_um[ib]:.1f} um)")
            ax.set_yscale("symlog", linthresh=1e-15)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Per-Bin MMR Time Evolution", fontsize=14)
    fig.tight_layout()
    fig.savefig(outdir / "growtest_perbin_evolution.png", dpi=150)
    plt.close(fig)

    # --- Figure 4: Supersaturation and ice saturation ratio ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Ice saturation ratio from Fortran (satice from gas output)
    f_satice = [fortran["gas"][i][2] for i in range(nsteps)]
    ax = axes[0]
    ax.plot(fortran["times"][:nsteps], f_satice, "ko-", ms=3, label="Fortran satice")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Ice saturation ratio")
    ax.set_title("Ice Saturation Ratio (Fortran)")
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Peak bin position over time
    ax = axes[1]
    f_peak = [np.argmax(fortran["mmr"][i]) + 1 for i in range(nsteps)]
    j_peak = [np.argmax(jax_data["mmr_bins"][i]) + 1 for i in range(nsteps)]
    ax.plot(fortran["times"][:nsteps], f_peak, "ko-", ms=3, label="Fortran")
    ax.plot(jax_data["times"][:nsteps], j_peak, "r^--", ms=3, label="JAX")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Peak bin index")
    ax.set_title("Peak Bin Position Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Total particle mass (mmr) over time
    ax = axes[2]
    f_pmass = [fortran["mmr"][i].sum() for i in range(nsteps)]
    j_pmass = jax_data["mmr_bins"][:nsteps].sum(axis=1)
    ax.plot(fortran["times"][:nsteps], f_pmass, "ko-", ms=3, label="Fortran")
    ax.plot(jax_data["times"][:nsteps], j_pmass, "r^--", ms=3, label="JAX")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Total particle MMR [g/g]")
    ax.set_title("Total Particle Mass Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Growth Diagnostics", fontsize=14)
    fig.tight_layout()
    fig.savefig(outdir / "growtest_diagnostics.png", dpi=150)
    plt.close(fig)

    # Summary
    print("\n" + "=" * 60)
    print("GROWTEST VALIDATION SUMMARY")
    print("=" * 60)
    print(f"T change (final): Fortran={fortran['t_changes'][-1]:.6e}, JAX={jax_data['t_change'][-1]:.6e}")
    if abs(fortran['t_changes'][-1]) > 1e-10:
        print(f"  Rel err: {abs(jax_data['t_change'][-1] - fortran['t_changes'][-1]) / abs(fortran['t_changes'][-1]):.4e}")
    print(f"Gas MMR (final): Fortran={f_gas[-1]:.6e}, JAX={jax_data['gas_mmr'][-1]:.6e}")
    print(f"Mass conservation: Fortran={(f_total[-1]/f_total[0]-1)*100:.6f}%, JAX={(j_total[-1]/j_total[0]-1)*100:.6f}%")
    print(f"\nPlots saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
