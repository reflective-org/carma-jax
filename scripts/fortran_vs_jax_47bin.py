"""Fortran vs JAX: 1000 scenarios, 47 bins (0.2nm-8um), dt=60s, 12 hours.

Nucleation-scale bin setup matching CARMA sulfate test conventions.
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
FORTRAN_EXE = SCRIPT_DIR / "fortran_runner" / "test_coag_47bin"
FORTRAN_DIR = SCRIPT_DIR / "fortran_runner"

NBIN = 47
NELEM = 1
NGROUP = 1
NZ = 1
RMIN_CM = 2e-8    # 0.2 nm
RMRAT = 2.0
RHO = 2.0
DT = 60.0         # fixed 60s timestep
NSTEP = 720       # 12 hours = 43200s / 60s


def parse_output(filepath, nbin):
    with open(filepath) as f:
        lines = f.readlines()
    idx = 0
    parts = lines[idx].split()
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
        times.append(float(line))
        idx += 1
        nd = np.zeros(nbin)
        for i in range(nbin):
            parts = lines[idx].split()
            nd[i] = float(parts[1])
            idx += 1
        data.append(nd)
    return np.array(times), np.array(data), radii


def run_fortran(n0, ck0, init_bin):
    nml = (f"&coag_params param_n0={n0:.6e}, param_ck0={ck0:.6e}, "
           f"param_dtime={DT:.6e}, param_nstep={NSTEP}, "
           f"param_init_bin={init_bin} /\n")
    (FORTRAN_DIR / "coag_params.nml").write_text(nml)
    t0 = timer.time()
    result = subprocess.run(
        [str(FORTRAN_EXE)], cwd=str(FORTRAN_DIR),
        capture_output=True, text=True, timeout=120
    )
    t1 = timer.time()
    if result.returncode != 0:
        raise RuntimeError(f"Fortran failed: {result.stderr}")
    times, nd, radii = parse_output(FORTRAN_DIR / "carma_coagtest_param.txt", NBIN)
    return times, nd, radii, t1 - t0


def run_jax(n0, ck0, init_bin, microslow_fn, zmet, rmass_jnp):
    ckernel = jnp.full((NZ, NBIN, NBIN, 1, 1), DTYPE(ck0), dtype=DTYPE)
    pc = jnp.full((NZ, NBIN, NELEM), SMALL_PC, dtype=DTYPE)
    pc = pc.at[0, init_bin - 1, 0].set(DTYPE(n0))

    nd_history = [np.array(pc[0, :, 0])]
    t0 = timer.time()
    for istep in range(NSTEP):
        pcl = pc
        pconmax = jnp.max(pc[:, :, 0:1] / zmet[:, None, None], axis=1)
        pc = microslow_fn(pc, pcl, ckernel, pconmax, zmet, DTYPE(DT))
        nd_history.append(np.array(pc[0, :, 0]))
    jax.block_until_ready(pc)
    t1 = timer.time()
    times = np.arange(0, NSTEP + 1) * DT
    return times, np.array(nd_history), t1 - t0


def main():
    np.random.seed(42)
    n_scenarios = 1000

    # Build JAX infrastructure
    r, rmass, vol, dr, dm, rup, rlow, rmassup = setup_bins(RMIN_CM, RMRAT, NBIN, RHO)
    rmass_np = np.array(rmass)

    groups = (GroupConfig(
        name='dust', ishape=1, ienconc=0, is_ice=False, is_cloud=False,
        is_sulfate=False, do_vtran=False, do_drydep=False, ifallrtn=1,
        irhswell=0, rmrat=RMRAT, eshape=1.0, rmin=RMIN_CM,
        r=r, rmass=rmass, vol=vol, dr=dr, dm=dm, rmassup=rmassup, rup=rup, rlow=rlow,
        rrat=jnp.ones(NBIN, dtype=DTYPE), rprat=jnp.ones(NBIN, dtype=DTYPE),
        arat=jnp.ones(NBIN, dtype=DTYPE),
    ),)
    elements = (ElementConfig(
        name='dust', rho=jnp.full(NBIN, RHO, dtype=DTYPE), igroup=0,
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

    deltaz = 100.0 * RM2CGS
    zc = jnp.array([0.5 * deltaz])
    zl = jnp.array([0.0, deltaz])
    p_pa, t = get_standard_atmosphere(zc / RM2CGS)
    pl_pa, _ = get_standard_atmosphere(zl / RM2CGS)
    rhoa, dz, zmet, zmetl, rmu, thcond, rhoa_wet = setup_atm(
        t, p_pa * RPA2CGS, pl_pa * RPA2CGS, zc, zl, GridType.I_CART
    )

    # Warmup JIT
    print("JIT compiling (47 bins, 720 steps)...", flush=True)
    ck_w = jnp.full((NZ, NBIN, NBIN, 1, 1), 1e-9, dtype=DTYPE)
    pc_w = jnp.full((NZ, NBIN, 1), SMALL_PC, dtype=DTYPE).at[0, 0, 0].set(1e6)
    pcon_w = jnp.max(pc_w[:, :, 0:1] / zmet[:, None, None], axis=1)
    _ = microslow_fn(pc_w, pc_w, ck_w, pcon_w, zmet, DTYPE(DT))
    print("Done.", flush=True)

    # Random parameters
    log_N0 = np.random.uniform(2, 9, n_scenarios)
    N0_arr = 10.0 ** log_N0
    init_bins = np.random.randint(1, 10, n_scenarios)  # bins 1-9 (0.2nm to 1.6nm)
    T_ref = np.random.uniform(150, 400, n_scenarios)
    ck0_arr = 8.0 * 1.38054e-16 * T_ref / (3.0 * 1.85e-4)

    # Storage
    fortran_time_total = 0.0
    jax_time_total = 0.0
    rel_err_totalN = np.zeros(n_scenarios)
    rel_err_totalM = np.zeros(n_scenarios)
    all_fortran_final = np.zeros((n_scenarios, NBIN))
    all_jax_final = np.zeros((n_scenarios, NBIN))
    all_bin_rel_err = np.full((n_scenarios, NBIN), np.nan)

    r_np = np.array(r)

    print(f"Running {n_scenarios} scenarios (47 bins, dt=60s, 12h)...", flush=True)
    t_total = timer.time()

    for i in range(n_scenarios):
        n0 = N0_arr[i]
        ib = init_bins[i]
        ck0 = ck0_arr[i]

        f_times, f_nd, f_radii, f_elapsed = run_fortran(n0, ck0, ib)
        fortran_time_total += f_elapsed

        j_times, j_nd, j_elapsed = run_jax(n0, ck0, ib, microslow_fn, zmet, rmass)
        jax_time_total += j_elapsed

        f_final = f_nd[-1]
        j_final = j_nd[-1]
        all_fortran_final[i] = f_final
        all_jax_final[i] = j_final

        N_f, N_j = f_final.sum(), j_final.sum()
        if N_f > 1:
            rel_err_totalN[i] = (N_j - N_f) / N_f
        M_f = (f_final * rmass_np).sum()
        M_j = (j_final * rmass_np).sum()
        if M_f > 1e-50:
            rel_err_totalM[i] = (M_j - M_f) / M_f

        for b in range(NBIN):
            if f_final[b] > 10.0:
                all_bin_rel_err[i, b] = (j_final[b] - f_final[b]) / f_final[b]

        if (i + 1) % 100 == 0:
            elapsed = timer.time() - t_total
            print(f"  {i+1}/{n_scenarios} ({elapsed:.0f}s)", flush=True)

    wall = timer.time() - t_total

    # Summary
    ae = np.abs(rel_err_totalN)
    print("\n" + "=" * 70)
    print(f"FORTRAN vs JAX: {n_scenarios} SCENARIOS (47 bins, dt=60s, 12h)")
    print("=" * 70)
    print(f"\n--- Setup ---")
    print(f"  Bins: {NBIN} (0.2nm to 8um, rmrat=2)")
    print(f"  Timestep: {DT}s, Steps: {NSTEP}, Total: 12 hours")
    print(f"  N0 range: [{N0_arr.min():.1e}, {N0_arr.max():.1e}] cm^-3")
    print(f"  Init bins: 1-9 (0.2nm to 1.6nm)")
    print(f"\n--- Timing ---")
    print(f"  Fortran: {fortran_time_total:.1f}s ({fortran_time_total/n_scenarios*1000:.1f} ms/scenario)")
    print(f"  JAX:     {jax_time_total:.1f}s ({jax_time_total/n_scenarios*1000:.1f} ms/scenario)")
    print(f"  Speedup: {fortran_time_total/jax_time_total:.1f}x")
    print(f"  Wall:    {wall:.0f}s")
    print(f"\n--- Total Number Error ---")
    print(f"  Mean:   {ae.mean():.6e}")
    print(f"  Median: {np.median(ae):.6e}")
    print(f"  Max:    {ae.max():.6e}")
    for th in [1e-2, 1e-3, 1e-4, 1e-5]:
        print(f"  % < {th:.0e}: {(ae<th).mean()*100:.1f}%")
    print(f"\n--- Mass Error ---")
    print(f"  Mean:   {np.abs(rel_err_totalM).mean():.6e}")
    print(f"  Max:    {np.abs(rel_err_totalM).max():.6e}")

    # --- Plots ---
    outdir = SCRIPT_DIR.parent / "plots" / "fortran_vs_jax_47bin"
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Error histogram
    ax = axes[0, 0]
    ax.hist(rel_err_totalN * 100, bins=60, color="steelblue", alpha=0.8, edgecolor="white", lw=0.3)
    ax.axvline(0, color="k", lw=1)
    ax.axvline(np.mean(rel_err_totalN) * 100, color="r", ls="--", lw=1.5,
               label=f"mean={np.mean(rel_err_totalN)*100:.4f}%")
    ax.set_xlabel("Relative error [%]")
    ax.set_ylabel("Count")
    ax.set_title("Total Number: (JAX-Fortran)/Fortran")
    ax.legend()
    ax.grid(True, alpha=0.2)

    # 2. CDF
    ax = axes[0, 1]
    sorted_err = np.sort(ae)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
    ax.plot(sorted_err, cdf * 100, "b-", lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("|Relative error|")
    ax.set_ylabel("Cumulative %")
    ax.set_title("CDF of |Total Number Error|")
    for th in [1e-6, 1e-5, 1e-4, 1e-3]:
        ax.axvline(th, color="gray", ls=":", lw=0.8)
    ax.grid(True, alpha=0.2)

    # 3. Timing
    ax = axes[0, 2]
    ax.bar(["Fortran", "JAX"], [fortran_time_total, jax_time_total], color=["#333", "steelblue"])
    ax.set_ylabel("Total time [s]")
    ax.set_title(f"Execution Time ({n_scenarios} x 720 steps)")
    for j_idx, v in enumerate([fortran_time_total, jax_time_total]):
        ax.text(j_idx, v * 1.02, f"{v:.1f}s", ha="center", fontsize=10)
    ax.grid(True, alpha=0.2)

    # 4. Per-bin error boxplot
    ax = axes[1, 0]
    bin_data = []
    valid_bins = []
    for b in range(NBIN):
        col = all_bin_rel_err[:, b]
        valid = ~np.isnan(col)
        if valid.sum() > 5:
            bin_data.append(col[valid] * 100)
            valid_bins.append(b)

    if bin_data:
        bp = ax.boxplot(
            bin_data,
            tick_labels=[str(b + 1) for b in valid_bins],
            patch_artist=True, showfliers=True,
            flierprops=dict(marker=".", ms=1, alpha=0.2),
            medianprops=dict(color="red", lw=1.5),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("steelblue")
            patch.set_alpha(0.6)
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("Bin index")
    ax.set_ylabel("Relative error [%]")
    ax.set_title("Per-Bin Error (N > 10 cm$^{-3}$)")
    ax.tick_params(axis="x", labelsize=6)
    ax.grid(True, alpha=0.2)

    # 5. Concentration boxplot
    ax = axes[1, 1]
    positions_f = np.arange(len(valid_bins)) * 2.5
    positions_j = positions_f + 0.8

    conc_f = [all_fortran_final[:, b][all_fortran_final[:, b] > 10] for b in valid_bins]
    conc_j = [all_jax_final[:, b][all_jax_final[:, b] > 10] for b in valid_bins]

    if conc_f:
        bp_f = ax.boxplot(conc_f, positions=positions_f, widths=0.7,
                          patch_artist=True, showfliers=True,
                          flierprops=dict(marker=".", ms=1, alpha=0.2),
                          medianprops=dict(color="black", lw=1.5))
        for p in bp_f["boxes"]:
            p.set_facecolor("#333")
            p.set_alpha(0.5)
        bp_j = ax.boxplot(conc_j, positions=positions_j, widths=0.7,
                          patch_artist=True, showfliers=True,
                          flierprops=dict(marker=".", ms=1, alpha=0.2),
                          medianprops=dict(color="red", lw=1.5))
        for p in bp_j["boxes"]:
            p.set_facecolor("steelblue")
            p.set_alpha(0.5)
        ax.set_yscale("log")
        ax.set_xticks([(positions_f[i] + positions_j[i]) / 2 for i in range(len(valid_bins))])
        ax.set_xticklabels([str(b + 1) for b in valid_bins], fontsize=6)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor="#333", alpha=0.5, label="Fortran"),
                           Patch(facecolor="steelblue", alpha=0.5, label="JAX")],
                  loc="upper right", fontsize=8)
    ax.set_xlabel("Bin index")
    ax.set_ylabel("N [cm$^{-3}$]")
    ax.set_title("Final Concentration by Bin (N > 10)")
    ax.grid(True, alpha=0.2)

    # 6. Example evolution: one scenario
    ax = axes[1, 2]
    # Pick a mid-range scenario
    ex_idx = np.argmin(np.abs(N0_arr - 1e6))
    ex_n0 = N0_arr[ex_idx]
    ex_ib = init_bins[ex_idx]
    ex_ck0 = ck0_arr[ex_idx]
    f_t, f_nd_ex, _, _ = run_fortran(ex_n0, ex_ck0, ex_ib)
    j_t, j_nd_ex, _ = run_jax(ex_n0, ex_ck0, ex_ib, microslow_fn, zmet, rmass)

    # Plot dN/dlogr at t=0, 1h, 6h, 12h
    dlogr = np.log10(r_np[1] / r_np[0])
    for t_hr, ls in [(0, "-"), (1, "--"), (6, "-."), (12, ":")]:
        step = int(t_hr * 3600 / DT)
        if step < len(f_nd_ex):
            ax.plot(r_np * 1e4, f_nd_ex[step] / dlogr, f"k{ls}", lw=1.5, alpha=0.7)
            ax.plot(r_np * 1e4, j_nd_ex[step] / dlogr, f"r{ls}", lw=1.5, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-2)
    ax.set_xlabel("Radius [um]")
    ax.set_ylabel("dN/dlogr [cm$^{-3}$]")
    ax.set_title(f"Example: N0={ex_n0:.0e}, bin {ex_ib}\n(black=Fortran, red=JAX, solid=0h, --=1h, -.=6h, :=12h)")
    ax.grid(True, alpha=0.2)

    fig.suptitle(f"Fortran vs JAX: {n_scenarios} Scenarios, 47 bins (0.2nm-8um), dt=60s, 12h", fontsize=14)
    fig.tight_layout()
    fig.savefig(outdir / "fortran_vs_jax_47bin.png", dpi=150)
    plt.close(fig)

    print(f"\nPlots saved to: {outdir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
