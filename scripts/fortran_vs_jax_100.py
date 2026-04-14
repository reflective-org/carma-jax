"""Run 100 coagulation scenarios in both Fortran CARMA and JAX, compare results.

Varies N0, ck0, and dtime across scenarios. Compares bin-by-bin number
concentrations, tracks timing, and produces diagnostic plots.
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
from carma.constants import RM2CGS, RPA2CGS, SMALL_PC, BK
from carma.bins import setup_bins
from carma.atmosphere_std import get_standard_atmosphere
from carma.setup_atm import setup_atm
from carma.enums import GridType, ElementType
from carma.coagulation.setup_coag import setup_coag
from carma.config import ElementConfig, GroupConfig, CarmaConfig
from carma.microslow import make_microslow

SCRIPT_DIR = Path(__file__).parent
FORTRAN_DIR = SCRIPT_DIR / "fortran_runner"
FORTRAN_EXE = FORTRAN_DIR / "test_coag_param"

NBIN = 20
NELEM = 1
NGROUP = 1
NZ = 1


def parse_output(filepath):
    """Parse carma_coagtest_param.txt output."""
    with open(filepath) as f:
        lines = f.readlines()
    idx = 0
    parts = lines[idx].split()
    nbin = int(parts[0])
    idx += 1

    radii = np.zeros(nbin)
    for i in range(nbin):
        parts = lines[idx].split()
        radii[i] = float(parts[1])
        idx += 1

    times = []
    data = []
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        time_val = float(line)
        times.append(time_val)
        idx += 1
        nd = np.zeros(nbin)
        for i in range(nbin):
            parts = lines[idx].split()
            nd[i] = float(parts[1])
            idx += 1
        data.append(nd)

    return np.array(times), np.array(data), radii


def run_fortran(n0, ck0, dtime, nstep, init_bin):
    """Run one Fortran scenario. Returns (times, nd_array, elapsed_seconds)."""
    nml = (f"&coag_params param_n0={n0:.6e}, param_ck0={ck0:.6e}, "
           f"param_dtime={dtime:.6e}, param_nstep={nstep}, "
           f"param_init_bin={init_bin} /\n")
    nml_path = FORTRAN_DIR / "coag_params.nml"
    out_path = FORTRAN_DIR / "carma_coagtest_param.txt"

    nml_path.write_text(nml)

    t0 = timer.time()
    result = subprocess.run(
        [str(FORTRAN_EXE)],
        cwd=str(FORTRAN_DIR),
        capture_output=True, text=True, timeout=60
    )
    t1 = timer.time()

    if result.returncode != 0:
        raise RuntimeError(f"Fortran failed: {result.stderr}")

    times, nd, radii = parse_output(out_path)
    return times, nd, radii, t1 - t0


def run_jax(n0, ck0, dtime, nstep, init_bin, microslow_fn, zmet, rmass):
    """Run one JAX scenario. Returns (times, nd_array, elapsed_seconds)."""
    ckernel = jnp.full((NZ, NBIN, NBIN, 1, 1), DTYPE(ck0), dtype=DTYPE)

    pc = jnp.full((NZ, NBIN, NELEM), SMALL_PC, dtype=DTYPE)
    pc = pc.at[0, init_bin - 1, 0].set(DTYPE(n0))  # init_bin is 1-based from Fortran

    nd_history = [np.array(pc[0, :, 0])]

    t0 = timer.time()
    for istep in range(nstep):
        pcl = pc
        pconmax = jnp.max(pc[:, :, 0:1] / zmet[:, None, None], axis=1)
        pc = microslow_fn(pc, pcl, ckernel, pconmax, zmet, DTYPE(dtime))
        nd_history.append(np.array(pc[0, :, 0]))
    jax.block_until_ready(pc)
    t1 = timer.time()

    times = np.arange(0, nstep + 1) * dtime
    return times, np.array(nd_history), t1 - t0


def main():
    np.random.seed(42)
    n_scenarios = 1000
    nstep = 20

    # Setup JAX infrastructure once
    rmin_cm, rmrat, rho = 3e-7, 2.0, 2.0
    r, rmass, vol, dr, dm, rup, rlow, rmassup = setup_bins(rmin_cm, rmrat, NBIN, rho)
    rmass_np = np.array(rmass)

    groups = (GroupConfig(
        name='dust', ishape=1, ienconc=0, is_ice=False, is_cloud=False,
        is_sulfate=False, do_vtran=False, do_drydep=False, ifallrtn=1,
        irhswell=0, rmrat=rmrat, eshape=1.0, rmin=rmin_cm,
        r=r, rmass=rmass, vol=vol, dr=dr, dm=dm, rmassup=rmassup, rup=rup, rlow=rlow,
        rrat=jnp.ones(NBIN, dtype=DTYPE), rprat=jnp.ones(NBIN, dtype=DTYPE),
        arat=jnp.ones(NBIN, dtype=DTYPE),
    ),)
    elements = (ElementConfig(
        name='dust', rho=jnp.full(NBIN, rho, dtype=DTYPE), igroup=0,
        itype=int(ElementType.I_INVOLATILE), icomposition=0, isolute=-1, kappa=0.0,
    ),)
    coag = setup_coag(NBIN, 1, 1, groups, elements,
                      np.array([[0]], dtype=np.int32), np.array([[0]], dtype=np.int32))

    microslow_fn = make_microslow(
        nbin=NBIN, nelem=NELEM, ngroup=NGROUP,
        elem_igroup=jnp.array([0]),
        icoag=coag.icoag, volx=coag.volx, icoagelem=coag.icoagelem,
        npairu=coag.npairu, npairl=coag.npairl,
        iup=coag.iup, jup=coag.jup, igup=coag.igup, jgup=coag.jgup,
        ilow=coag.ilow, jlow=coag.jlow, iglow=coag.iglow, jglow=coag.jglow,
        pkernel=coag.pkernel,
        ienconc_arr=jnp.array([0]),
        elem_itypes=jnp.array([int(ElementType.I_INVOLATILE)]),
    )

    # Single-level atmosphere
    deltaz = 100.0 * RM2CGS
    zc = jnp.array([0.5 * deltaz])
    zl = jnp.array([0.0, deltaz])
    p_pa, t = get_standard_atmosphere(zc / RM2CGS)
    pl_pa, _ = get_standard_atmosphere(zl / RM2CGS)
    rhoa, dz, zmet, zmetl, rmu, thcond, rhoa_wet = setup_atm(
        t, p_pa * RPA2CGS, pl_pa * RPA2CGS, zc, zl, GridType.I_CART
    )

    # Warmup JAX JIT
    print("Warming up JAX JIT...", flush=True)
    ck_w = jnp.full((NZ, NBIN, NBIN, 1, 1), 1e-9, dtype=DTYPE)
    pc_w = jnp.full((NZ, NBIN, 1), SMALL_PC, dtype=DTYPE).at[0, 0, 0].set(1e6)
    pcon_w = jnp.max(pc_w[:, :, 0:1] / zmet[:, None, None], axis=1)
    _ = microslow_fn(pc_w, pc_w, ck_w, pcon_w, zmet, DTYPE(600.0))

    # Random scenario parameters
    log_N0 = np.random.uniform(2, 9, n_scenarios)
    N0_arr = 10.0 ** log_N0
    init_bins = np.random.randint(1, 6, n_scenarios)  # 1-based for Fortran, bins 1-5
    # Vary kernel: ck0 = 8*BK*T_ref / (3*eta), T_ref in [150, 400]K
    T_ref = np.random.uniform(150, 400, n_scenarios)
    ck0_arr = 8.0 * 1.38054e-16 * T_ref / (3.0 * 1.85e-4)
    dtime_arr = np.random.uniform(30, 1800, n_scenarios)  # 30s to 30min

    # Results storage
    fortran_times_total = 0.0
    jax_times_total = 0.0
    fortran_times_list = []
    jax_times_list = []

    # Per-scenario error metrics
    rel_err_totalN = np.zeros(n_scenarios)
    rel_err_totalM = np.zeros(n_scenarios)
    max_bin_err = np.zeros(n_scenarios)
    mean_bin_err = np.zeros(n_scenarios)

    print(f"Running {n_scenarios} scenarios (Fortran + JAX, {nstep} steps each)...", flush=True)
    t_total_start = timer.time()

    for i in range(n_scenarios):
        n0 = N0_arr[i]
        ib = init_bins[i]
        ck0 = ck0_arr[i]
        dt = dtime_arr[i]

        # Run Fortran
        f_times, f_nd, f_radii, f_elapsed = run_fortran(n0, ck0, dt, nstep, ib)
        fortran_times_list.append(f_elapsed)
        fortran_times_total += f_elapsed

        # Run JAX
        j_times, j_nd, j_elapsed = run_jax(n0, ck0, dt, nstep, ib, microslow_fn, zmet, rmass_np)
        jax_times_list.append(j_elapsed)
        jax_times_total += j_elapsed

        # Compare final timestep
        f_final = f_nd[-1]
        j_final = j_nd[-1]

        # Total number
        N_f = f_final.sum()
        N_j = j_final.sum()
        if N_f > 1:
            rel_err_totalN[i] = (N_j - N_f) / N_f

        # Total mass
        M_f = (f_final * rmass_np).sum()
        M_j = (j_final * rmass_np).sum()
        if M_f > 1e-50:
            rel_err_totalM[i] = (M_j - M_f) / M_f

        # Per-bin errors (bins > 10 cm^-3)
        mask = f_final > 10.0
        if mask.any():
            bin_errs = np.abs((j_final[mask] - f_final[mask]) / f_final[mask])
            max_bin_err[i] = bin_errs.max()
            mean_bin_err[i] = bin_errs.mean()

        if (i + 1) % 200 == 0:
            elapsed = timer.time() - t_total_start
            print(f"  {i+1}/{n_scenarios} ({elapsed:.1f}s)", flush=True)

    t_total_end = timer.time()

    # Summary
    print("\n" + "=" * 70)
    print(f"FORTRAN vs JAX: {n_scenarios} SCENARIOS COMPARISON")
    print("=" * 70)

    print(f"\n--- Timing ---")
    print(f"  Fortran total:    {fortran_times_total:.2f}s ({fortran_times_total/n_scenarios*1000:.1f} ms/scenario)")
    print(f"  JAX total:        {jax_times_total:.4f}s ({jax_times_total/n_scenarios*1000:.2f} ms/scenario)")
    print(f"  Speedup:          {fortran_times_total/jax_times_total:.1f}x")
    print(f"  Wall time:        {t_total_end - t_total_start:.1f}s")

    print(f"\n--- Total Number Relative Error (JAX vs Fortran) ---")
    ae = np.abs(rel_err_totalN)
    print(f"  Mean |error|:     {ae.mean():.6e}")
    print(f"  Median |error|:   {np.median(ae):.6e}")
    print(f"  Max |error|:      {ae.max():.6e}")
    print(f"  Std:              {rel_err_totalN.std():.6e}")
    for thresh in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        pct = (ae < thresh).mean() * 100
        print(f"  % below {thresh:.0e}:    {pct:.1f}%")

    print(f"\n--- Total Mass Relative Error ---")
    ae_m = np.abs(rel_err_totalM)
    print(f"  Mean |error|:     {ae_m.mean():.6e}")
    print(f"  Max |error|:      {ae_m.max():.6e}")

    print(f"\n--- Per-Bin Error (bins > 10 cm^-3, final step) ---")
    valid = max_bin_err > 0
    if valid.any():
        print(f"  Scenarios with valid bins: {valid.sum()}")
        print(f"  Mean of max |bin err|:    {max_bin_err[valid].mean():.6e}")
        print(f"  Max of max |bin err|:     {max_bin_err[valid].max():.6e}")
        print(f"  Mean of mean |bin err|:   {mean_bin_err[valid].mean():.6e}")

    # --- Plots ---
    outdir = SCRIPT_DIR.parent / "plots" / "fortran_vs_jax_100"
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Total N error histogram
    ax = axes[0, 0]
    ax.hist(rel_err_totalN * 100, bins=50, color="steelblue", alpha=0.8, edgecolor="white", lw=0.3)
    ax.axvline(0, color="k", lw=1)
    mn = np.mean(rel_err_totalN) * 100
    ax.axvline(mn, color="r", ls="--", lw=1.5, label=f"mean={mn:.4f}%")
    ax.set_xlabel("Relative error [%]")
    ax.set_ylabel("Count")
    ax.set_title("Total Number: (JAX-Fortran)/Fortran")
    ax.legend()
    ax.grid(True, alpha=0.2)

    # 2. Total N error CDF
    ax = axes[0, 1]
    sorted_err = np.sort(ae)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
    ax.plot(sorted_err, cdf * 100, "b-", lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("|Relative error|")
    ax.set_ylabel("Cumulative %")
    ax.set_title("CDF of |Total Number Error|")
    for thresh in [1e-6, 1e-5, 1e-4, 1e-3]:
        ax.axvline(thresh, color="gray", ls=":", lw=0.8)
    ax.grid(True, alpha=0.2)

    # 3. Max bin error histogram
    ax = axes[0, 2]
    valid_mbe = max_bin_err[max_bin_err > 0]
    if len(valid_mbe) > 0:
        ax.hist(np.log10(valid_mbe), bins=40, color="coral", alpha=0.8, edgecolor="white", lw=0.3)
        ax.set_xlabel("log10(max |bin error|)")
        ax.set_ylabel("Count")
    ax.set_title("Max Per-Bin Error Distribution")
    ax.grid(True, alpha=0.2)

    # 4. Timing comparison
    ax = axes[1, 0]
    ax.bar(["Fortran", "JAX"], [fortran_times_total, jax_times_total], color=["#333", "steelblue"])
    ax.set_ylabel("Total time [s]")
    ax.set_title(f"Total Execution Time ({n_scenarios} scenarios)")
    for j_idx, v in enumerate([fortran_times_total, jax_times_total]):
        ax.text(j_idx, v + 0.01 * max(fortran_times_total, jax_times_total),
                f"{v:.2f}s", ha="center", fontsize=10)
    ax.grid(True, alpha=0.2)

    # 5. Per-scenario timing
    ax = axes[1, 1]
    ax.scatter(range(n_scenarios), np.array(fortran_times_list) * 1000, s=8, alpha=0.6, label="Fortran", c="#333")
    ax.scatter(range(n_scenarios), np.array(jax_times_list) * 1000, s=8, alpha=0.6, label="JAX", c="steelblue")
    ax.set_xlabel("Scenario index")
    ax.set_ylabel("Time per scenario [ms]")
    ax.set_title("Per-Scenario Execution Time")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.2)

    # 6. Error vs N0
    ax = axes[1, 2]
    ax.scatter(N0_arr, ae, s=10, alpha=0.5, c="steelblue")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N0 [cm$^{-3}$]")
    ax.set_ylabel("|Total N relative error|")
    ax.set_title("Error vs Initial Concentration")
    ax.axhline(1e-6, color="r", ls="--", lw=1, label="1e-6")
    ax.axhline(1e-4, color="orange", ls="--", lw=1, label="1e-4")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    fig.suptitle(f"Fortran vs JAX: {n_scenarios} Random Coagulation Scenarios", fontsize=14)
    fig.tight_layout()
    fig.savefig(outdir / "fortran_vs_jax_comparison.png", dpi=150)
    plt.close(fig)

    print(f"\nPlots saved to: {outdir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
