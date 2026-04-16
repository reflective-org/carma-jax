"""1000-scenario growth test: Fortran vs JAX.

Varies T, p, gas_mmr, N0 across scenarios.
24 bins (1-200 um ice crystals), dt=100s, 50 steps.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
import time as timer

from carma.precision import DTYPE
from carma.constants import *
from carma.bins import setup_bins
from carma.setup_atm import setup_atm
from carma.setup_grow import setup_grow
from carma.setup_gkern import setup_gkern
from carma.setup_vf import setup_vf
from carma.vapor_pressure import vaporp_h2o_murphy2005
from carma.supersaturation import supersat
from carma.newstate_calc import newstate_calc_growth
from carma.enums import GridType

SCRIPT_DIR = Path(__file__).parent
FORTRAN_DIR = SCRIPT_DIR / "fortran_runner"
FORTRAN_EXE = FORTRAN_DIR / "test_grow_param"

NBIN, NELEM, NGROUP, NGAS, NZ = 24, 1, 1, 1, 1
RMIN_CM = 1e-4
RMRAT = 2.0
RHO_P = float(RHO_I)
DTIME = 100.0
NSTEP = 50


def parse_grow_output(filepath):
    with open(filepath) as f:
        lines = f.readlines()
    idx = 0
    parts = lines[idx].split()
    nbin = int(parts[0])
    idx += 1
    radii = np.zeros(nbin)
    rmass_f = np.zeros(nbin)
    for i in range(nbin):
        parts = lines[idx].split()
        radii[i] = float(parts[1])
        rmass_f[i] = float(parts[2])
        idx += 1

    times, t_changes, mmr_data, gas_data = [], [], [], []
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        times.append(float(line))
        idx += 1
        parts = lines[idx].split()
        t_changes.append(float(parts[0]))
        idx += 1
        mmr = np.zeros(nbin)
        for i in range(nbin):
            parts = lines[idx].split()
            mmr[i] = float(parts[1])
            idx += 1
        mmr_data.append(mmr)
        parts = lines[idx].split()
        gas_data.append(float(parts[0]))
        idx += 1

    return np.array(times), np.array(t_changes), np.array(mmr_data), np.array(gas_data), radii, rmass_f


def run_fortran(T, p_pa, gas_mmr, N0):
    nml = (f"&grow_params param_T={T:.4f}, param_p={p_pa:.4f}, "
           f"param_gas_mmr={gas_mmr:.6e}, param_N0={N0:.6e}, "
           f"param_dtime={DTIME:.1f}, param_nstep={NSTEP} /\n")
    (FORTRAN_DIR / "grow_params.nml").write_text(nml)
    t0 = timer.time()
    result = subprocess.run([str(FORTRAN_EXE)], cwd=str(FORTRAN_DIR),
                            capture_output=True, text=True, timeout=60)
    t1 = timer.time()
    if result.returncode != 0:
        raise RuntimeError(f"Fortran failed: {result.stderr}")
    times, t_ch, mmr, gas, _, _ = parse_grow_output(FORTRAN_DIR / "carma_growtest_param.txt")
    return t_ch, mmr, gas, t1 - t0


def run_jax(T_val, p_pa_val, gas_mmr_val, N0_val,
            r, rmass, vol, dr, dm, rup, rlow, rmassup,
            pratt, prat, pden1, palr, gwtmol_arr):
    t = jnp.array([DTYPE(T_val)])
    p_cgs = jnp.array([DTYPE(p_pa_val) * RPA2CGS])
    zc = jnp.array([DTYPE(17000.0) * RM2CGS])
    zl = jnp.array([DTYPE(16900.0) * RM2CGS, DTYPE(17100.0) * RM2CGS])

    rho_air_val = float(p_pa_val * 10.0) / (float(R_AIR) * T_val) * 1e3
    pl = jnp.array([DTYPE(p_pa_val + 100.0 * rho_air_val * float(GRAV) / 100.0) * RPA2CGS,
                     DTYPE(p_pa_val - 100.0 * rho_air_val * float(GRAV) / 100.0) * RPA2CGS])

    rhoa, dz, zmet, zmetl, rmu, thcond, rhoa_wet = setup_atm(t, p_cgs, pl, zc, zl, GridType.I_CART)
    diffus, rlhe, rlhm = setup_grow(t, p_cgs, rhoa, zmet, igash2o=0, igash2so4=-1, ngas=1, do_cnst_rlh=False)

    r_wet = jnp.broadcast_to(r[None, :, None], (NZ, NBIN, NGROUP))
    rlow_wet = jnp.broadcast_to(rlow[None, :, None], (NZ, NBIN, NGROUP))
    rup_wet = jnp.broadcast_to(rup[None, :, None], (NZ, NBIN, NGROUP))
    rrat_arr = jnp.ones((NBIN, NGROUP), dtype=DTYPE)
    rhop_wet = jnp.full((NZ, NBIN, NGROUP), RHO_P, dtype=DTYPE)
    rmass_2d = rmass[:, None] * jnp.ones((1, NGROUP), dtype=DTYPE)
    dm_2d = dm[:, None] * jnp.ones((1, NGROUP), dtype=DTYPE)

    vf, re, bpm = setup_vf(None, t, rhoa, zmet, rmu, r_wet, rhop_wet, rrat_arr, rrat_arr)
    _, akelvin, akelvini, gro, gro1, gro2, ft_arr, thcondnc = setup_gkern(
        t, p_cgs, rhoa, zmet, rmu, thcond, diffus, rlhe, rlhm,
        re, r_wet, rlow_wet, rrat_arr, jnp.array([1.0]), jnp.array([True]),
        gwtmol_arr, jnp.array([0]), 1.0, 1.0, 1.0, NBIN, NGROUP, NGAS,
    )

    pc = jnp.full((NZ, NBIN, NELEM), SMALL_PC, dtype=DTYPE)
    pc = pc.at[0, 0, 0].set(DTYPE(N0_val) * zmet[0])
    gc = jnp.array([[DTYPE(gas_mmr_val) * rhoa[0]]])
    t_orig = T_val

    t0 = timer.time()
    for istep in range(NSTEP):
        pc, gc, t, _, _ = newstate_calc_growth(
            pc, gc, t, 0, DTIME,
            rhoa, zmet, rlhe, rlhm, diffus,
            akelvin, akelvini, gro, gro1, gro2,
            rup_wet, rmass_2d, dm_2d, rlow_wet,
            pratt, prat, pden1, palr,
            jnp.array([True]), jnp.array([0]), jnp.array([0]),
            jnp.array([0]), gwtmol_arr,
            NBIN, NGROUP, NGAS, NELEM,
            minsubsteps=1, maxsubsteps=1, maxretries=0,
        )
    t1 = timer.time()

    rmass_np = np.array(rmass)
    mmr_final = np.array(pc[0, :, 0]) / float(zmet[0]) * rmass_np / float(rhoa[0])
    gas_final = float(gc[0, 0] / rhoa[0])
    dt_final = float(t[0]) - t_orig

    return dt_final, mmr_final, gas_final, t1 - t0


def main():
    np.random.seed(42)
    n_scenarios = 100

    r, rmass, vol, dr, dm, rup, rlow, rmassup = setup_bins(RMIN_CM, RMRAT, NBIN, RHO_P)

    # PPM coefficients (exact Fortran formulas)
    dm_np, rmass_np, rmassup_np = np.array(dm), np.array(rmass), np.array(rmassup)
    pratt = jnp.zeros((3, NBIN, NGROUP), dtype=DTYPE)
    prat = jnp.zeros((4, NBIN, NGROUP), dtype=DTYPE)
    pden1 = jnp.zeros((NBIN, NGROUP), dtype=DTYPE)
    palr = jnp.zeros((4, NGROUP), dtype=DTYPE)

    for ibin in range(1, NBIN - 1):
        pratt = pratt.at[0, ibin, 0].set(dm_np[ibin] / (dm_np[ibin-1]+dm_np[ibin]+dm_np[ibin+1]))
        pratt = pratt.at[1, ibin, 0].set((2*dm_np[ibin-1]+dm_np[ibin]) / (dm_np[ibin+1]+dm_np[ibin]))
        pratt = pratt.at[2, ibin, 0].set((2*dm_np[ibin+1]+dm_np[ibin]) / (dm_np[ibin-1]+dm_np[ibin]))
    for ibin in range(1, NBIN - 2):
        dm_ip2 = dm_np[min(ibin+2, NBIN-1)]
        prat = prat.at[0, ibin, 0].set(dm_np[ibin] / (dm_np[ibin]+dm_np[ibin+1]))
        prat = prat.at[1, ibin, 0].set(2*dm_np[ibin+1]*dm_np[ibin] / (dm_np[ibin]+dm_np[ibin+1]))
        prat = prat.at[2, ibin, 0].set((dm_np[ibin-1]+dm_np[ibin]) / (2*dm_np[ibin]+dm_np[ibin+1]))
        prat = prat.at[3, ibin, 0].set((dm_ip2+dm_np[ibin+1]) / (2*dm_np[ibin+1]+dm_np[ibin]))
        pden1 = pden1.at[ibin, 0].set(dm_np[ibin-1]+dm_np[ibin]+dm_np[ibin+1]+dm_ip2)
    denom_low = rmass_np[1] - rmass_np[0]
    denom_high = rmass_np[NBIN-1] - rmass_np[NBIN-2]
    palr = palr.at[0, 0].set((rmassup_np[0] - rmass_np[0]) / denom_low)
    palr = palr.at[1, 0].set((rmassup_np[0] / RMRAT - rmass_np[0]) / denom_low)
    palr = palr.at[2, 0].set((rmassup_np[NBIN-2] - rmass_np[NBIN-2]) / denom_high)
    palr = palr.at[3, 0].set((rmassup_np[NBIN-1] - rmass_np[NBIN-2]) / denom_high)

    gwtmol_arr = jnp.array([float(WTMOL_H2O)])

    # Random scenarios
    T_arr = np.random.uniform(185, 210, n_scenarios)    # TTL range
    p_arr = np.random.uniform(7000, 12000, n_scenarios)  # 70-120 hPa
    gas_arr = np.random.uniform(2e-6, 6e-6, n_scenarios) # 2-6 ppm H2O
    N0_arr = 10.0 ** np.random.uniform(-2, 1, n_scenarios) # 0.01-10 cm^-3

    # Storage
    fortran_total, jax_total = 0.0, 0.0
    rel_err_dT = np.zeros(n_scenarios)
    rel_err_gas = np.zeros(n_scenarios)
    all_f_mmr = np.zeros((n_scenarios, NBIN))
    all_j_mmr = np.zeros((n_scenarios, NBIN))
    all_bin_rel_err = np.full((n_scenarios, NBIN), np.nan)

    print(f"Running {n_scenarios} scenarios (24 bins, dt=100s, 50 steps)...", flush=True)
    t_start = timer.time()

    for i in range(n_scenarios):
        T, p_pa, gas_mmr, N0 = T_arr[i], p_arr[i], gas_arr[i], N0_arr[i]

        f_dT, f_mmr, f_gas, f_elapsed = run_fortran(T, p_pa, gas_mmr, N0)
        fortran_total += f_elapsed

        j_dT, j_mmr, j_gas, j_elapsed = run_jax(
            T, p_pa, gas_mmr, N0,
            r, rmass, vol, dr, dm, rup, rlow, rmassup,
            pratt, prat, pden1, palr, gwtmol_arr,
        )
        jax_total += j_elapsed

        all_f_mmr[i] = f_mmr[-1]
        all_j_mmr[i] = j_mmr

        if abs(f_dT[-1]) > 1e-10:
            rel_err_dT[i] = (j_dT - f_dT[-1]) / f_dT[-1]
        if abs(f_gas[-1]) > 1e-15:
            rel_err_gas[i] = (j_gas - f_gas[-1]) / f_gas[-1]

        for b in range(NBIN):
            if f_mmr[-1][b] > 1e-20:
                all_bin_rel_err[i, b] = (j_mmr[b] - f_mmr[-1][b]) / f_mmr[-1][b]

        if (i + 1) % 100 == 0:
            elapsed = timer.time() - t_start
            print(f"  {i+1}/{n_scenarios} ({elapsed:.0f}s)", flush=True)

    wall = timer.time() - t_start
    ae_dT = np.abs(rel_err_dT)

    print("\n" + "=" * 70)
    print(f"GROWTH TEST: {n_scenarios} SCENARIOS")
    print("=" * 70)
    print(f"\n--- Setup ---")
    print(f"  T range: [{T_arr.min():.0f}, {T_arr.max():.0f}] K")
    print(f"  p range: [{p_arr.min():.0f}, {p_arr.max():.0f}] Pa")
    print(f"  gas_mmr: [{gas_arr.min():.1e}, {gas_arr.max():.1e}]")
    print(f"  N0: [{N0_arr.min():.2e}, {N0_arr.max():.2e}] cm^-3")
    print(f"\n--- Timing ---")
    print(f"  Fortran: {fortran_total:.1f}s ({fortran_total/n_scenarios*1000:.1f} ms/scenario)")
    print(f"  JAX:     {jax_total:.1f}s ({jax_total/n_scenarios*1000:.1f} ms/scenario)")
    print(f"  Speedup: {fortran_total/max(jax_total, 0.001):.1f}x")
    print(f"\n--- Temperature Change Error ---")
    print(f"  Mean:   {ae_dT.mean():.6e}")
    print(f"  Median: {np.median(ae_dT):.6e}")
    print(f"  Max:    {ae_dT.max():.6e}")
    for th in [1e-1, 1e-2, 1e-3, 1e-4]:
        print(f"  % < {th:.0e}: {(ae_dT<th).mean()*100:.1f}%")

    # Plots
    outdir = SCRIPT_DIR.parent / "plots" / "growtest_1000"
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. dT error histogram
    ax = axes[0, 0]
    ax.hist(rel_err_dT * 100, bins=60, color="steelblue", alpha=0.8, edgecolor="white", lw=0.3)
    ax.axvline(np.mean(rel_err_dT) * 100, color="r", ls="--", lw=1.5,
               label=f"mean={np.mean(rel_err_dT)*100:.4f}%")
    ax.set_xlabel("dT relative error [%]")
    ax.set_ylabel("Count")
    ax.set_title("Temperature Change Error")
    ax.legend()
    ax.grid(True, alpha=0.2)

    # 2. CDF
    ax = axes[0, 1]
    sorted_err = np.sort(ae_dT)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
    ax.plot(sorted_err, cdf * 100, "b-", lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("|Relative error|")
    ax.set_ylabel("Cumulative %")
    ax.set_title("CDF of |dT Error|")
    for th in [1e-4, 1e-3, 1e-2]:
        ax.axvline(th, color="gray", ls=":", lw=0.8)
    ax.grid(True, alpha=0.2)

    # 3. Timing
    ax = axes[0, 2]
    ax.bar(["Fortran", "JAX"], [fortran_total, jax_total], color=["#333", "steelblue"])
    ax.set_ylabel("Total time [s]")
    ax.set_title(f"Execution Time ({n_scenarios} x {NSTEP} steps)")
    for j_idx, v in enumerate([fortran_total, jax_total]):
        ax.text(j_idx, v * 1.02, f"{v:.1f}s", ha="center", fontsize=10)
    ax.grid(True, alpha=0.2)

    # 4. Per-bin number error boxplot
    ax = axes[1, 0]
    bin_data, valid_bins = [], []
    for b in range(NBIN):
        col = all_bin_rel_err[:, b]
        valid = ~np.isnan(col)
        if valid.sum() > 5:
            bin_data.append(col[valid] * 100)
            valid_bins.append(b)
    if bin_data:
        bp = ax.boxplot(bin_data, tick_labels=[str(b+1) for b in valid_bins],
                        patch_artist=True, showfliers=True,
                        flierprops=dict(marker=".", ms=1, alpha=0.2),
                        medianprops=dict(color="red", lw=1.5))
        for patch in bp["boxes"]:
            patch.set_facecolor("steelblue")
            patch.set_alpha(0.6)
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("Bin index")
    ax.set_ylabel("MMR relative error [%]")
    ax.set_title("Per-Bin MMR Error (final step)")
    ax.grid(True, alpha=0.2)

    # 5. Per-bin mass error boxplot
    ax = axes[1, 1]
    mass_data, mass_bins = [], []
    for b in range(NBIN):
        f_mass = all_f_mmr[:, b]
        j_mass = all_j_mmr[:, b]
        mask = f_mass > 1e-20
        if mask.sum() > 5:
            re = (j_mass[mask] - f_mass[mask]) / f_mass[mask]
            mass_data.append(re * 100)
            mass_bins.append(b)
    if mass_data:
        bp = ax.boxplot(mass_data, tick_labels=[str(b+1) for b in mass_bins],
                        patch_artist=True, showfliers=True,
                        flierprops=dict(marker=".", ms=1, alpha=0.2),
                        medianprops=dict(color="red", lw=1.5))
        for patch in bp["boxes"]:
            patch.set_facecolor("coral")
            patch.set_alpha(0.6)
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("Bin index")
    ax.set_ylabel("Mass relative error [%]")
    ax.set_title("Per-Bin Mass Error")
    ax.grid(True, alpha=0.2)

    # 6. Error vs T
    ax = axes[1, 2]
    ax.scatter(T_arr, ae_dT, s=5, alpha=0.4, c="steelblue")
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("|dT relative error|")
    ax.set_yscale("log")
    ax.set_title("Error vs Temperature")
    ax.axhline(1e-3, color="r", ls="--", lw=1, label="1e-3")
    ax.legend()
    ax.grid(True, alpha=0.2)

    fig.suptitle(f"Growth Test: {n_scenarios} Scenarios, 24 bins, Fortran vs JAX", fontsize=14)
    fig.tight_layout()
    fig.savefig(outdir / "growtest_1000.png", dpi=150)
    plt.close(fig)

    print(f"\nPlots saved to: {outdir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
