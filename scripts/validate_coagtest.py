"""Validate JAX coagulation against Fortran benchmark.

Parses carma_coagtest.txt, runs matching JAX simulation, generates
comparison plots.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from carma.precision import DTYPE
from carma.constants import PI, RM2CGS, RPA2CGS, SMALL_PC, BK
from carma.bins import setup_bins
from carma.atmosphere_std import get_standard_atmosphere
from carma.setup_atm import setup_atm
from carma.enums import GridType, ElementType
from carma.coagulation.setup_coag import setup_coag
from carma.config import ElementConfig, GroupConfig, CarmaConfig
from carma.setup_vf import setup_vf
from carma.setup_ckern import setup_ckern
from carma.microslow import microslow

# ---------- Parse Fortran benchmark ----------

def parse_coagtest_bench(filepath):
    """Parse carma_coagtest.txt benchmark file."""
    with open(filepath) as f:
        lines = f.readlines()

    idx = 0
    # Line 1: NBIN, NELEM, NGROUP
    parts = lines[idx].split()
    nbin = int(parts[0])
    idx += 1

    # Bin structure: nbin lines of (index, radius_cm, dr_cm)
    radii = np.zeros(nbin)
    dr = np.zeros(nbin)
    for i in range(nbin):
        parts = lines[idx].split()
        radii[i] = float(parts[1])
        dr[i] = float(parts[2])
        idx += 1

    # Time blocks: time value, then nbin lines of (index, number_density)
    times = []
    data = []  # list of (nbin,) arrays

    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        # Time value
        time_val = float(line)
        times.append(time_val)
        idx += 1

        nd = np.zeros(nbin)
        for i in range(nbin):
            parts = lines[idx].split()
            nd[i] = float(parts[1])
            idx += 1
        data.append(nd)

    return {
        "nbin": nbin,
        "radii": radii,  # cm
        "dr": dr,  # cm
        "times": np.array(times),
        "nd": np.array(data),  # (nstep+1, nbin) number density [cm^-3]
    }


# ---------- Run JAX simulation ----------

def run_jax_coagtest():
    """Run the JAX coagulation test matching Fortran setup."""
    NBIN, NELEM, NGROUP, NZ = 20, 1, 1, 80
    rmin_cm = 3e-7  # 3e-7 cm (3 nm) — CARMA uses CGS directly
    rmrat = 2.0
    rho = 2.0
    dtime = 600.0
    nstep = 72
    deltaz_m = 100.0

    # Setup bins
    r, rmass, vol, dr, dm, rup, rlow, rmassup = setup_bins(rmin_cm, rmrat, NBIN, rho)

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

    icoag = np.array([[0]], dtype=np.int32)
    icoagelem = np.array([[0]], dtype=np.int32)
    coag = setup_coag(NBIN, NGROUP, NELEM, groups, elements, icoag, icoagelem)

    config = CarmaConfig(
        nbin=NBIN, nelem=NELEM, ngroup=NGROUP, ngas=0, nsolute=0,
        elements=elements, groups=groups, gases=(), solutes=(),
        coag=coag,
        do_coag=True, do_grow=False, do_vtran=False, do_vdiff=False,
        do_thermo=False, do_substep=False, do_explised=False,
        do_incloud=False, do_clearsky=False, do_detrain=False,
        do_pheat=False, do_pheatatm=False, do_cnst_rlh=False,
        itbnd_pc=1, ibbnd_pc=1,
        maxsubsteps=1, minsubsteps=1, maxretries=5, conmax=0.0,
        cstick=1.0, gsticki=1.0, gstickl=1.0, tstick=1.0, dt_threshold=0.0,
        igash2o=-1, igash2so4=-1, igasso2=-1,
    )

    # Setup atmosphere
    deltaz = deltaz_m * RM2CGS
    zc = jnp.arange(0.5, NZ) * deltaz
    zl = jnp.arange(0.0, NZ + 1) * deltaz
    p_pa, t = get_standard_atmosphere(zc / RM2CGS)
    pl_pa, _ = get_standard_atmosphere(zl / RM2CGS)
    p = p_pa * RPA2CGS
    pl = pl_pa * RPA2CGS
    rhoa, dz, zmet, zmetl, rmu, thcond, rhoa_wet = setup_atm(
        t, p, pl, zc, zl, GridType.I_CART
    )

    # Particle properties
    r_wet = jnp.broadcast_to(r[None, :, None], (NZ, NBIN, NGROUP))
    rhop_wet = jnp.full((NZ, NBIN, NGROUP), rho, dtype=DTYPE)
    rrat_arr = jnp.ones((NBIN, NGROUP), dtype=DTYPE)
    rprat_arr = jnp.ones((NBIN, NGROUP), dtype=DTYPE)
    rmass_arr = rmass[:, None] * jnp.ones((1, NGROUP), dtype=DTYPE)

    vf, re, bpm = setup_vf(
        config, t, rhoa, zmet, rmu, r_wet, rhop_wet, rrat_arr, rprat_arr
    )

    # The Fortran coagtest uses a CONSTANT Brownian kernel (icoagop=I_COAGOP_CONST)
    # ck0 = 8 * BK * 298 / (3 * 1.85e-4)  [from carma_coagtest.F90 line 125]
    ck0 = DTYPE(8.0) * BK * DTYPE(298.0) / (DTYPE(3.0) * DTYPE(1.85e-4))
    ckernel = jnp.full((NZ, NBIN, NBIN, NGROUP, NGROUP), ck0, dtype=DTYPE)

    # Initial conditions: 1e6 cm^-3 in bin 0, level 0
    pc = jnp.full((NZ, NBIN, NELEM), SMALL_PC, dtype=DTYPE)
    pc = pc.at[0, 0, 0].set(DTYPE(1e6))

    # Store time history at level 0
    nd_history = [np.array(pc[0, :, 0])]  # time 0

    # Time integration
    for istep in range(nstep):
        pcl = pc
        pconmax = jnp.max(pc[:, :, 0:1] / zmet[:, None, None], axis=1)
        pc = microslow(config, pc, pcl, ckernel, pconmax, zmet, dtime)
        nd_history.append(np.array(pc[0, :, 0]))

    times = np.arange(0, nstep + 1) * dtime
    nd_jax = np.array(nd_history)  # (nstep+1, NBIN)
    rmass_np = np.array(rmass)
    r_np = np.array(r)
    dr_np = np.array(dr)

    return {
        "times": times,
        "nd": nd_jax,
        "radii": r_np,
        "dr": dr_np,
        "rmass": rmass_np,
    }


# ---------- Plotting ----------

def make_plots(fortran, jax_data, outdir):
    """Generate all comparison plots."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    r_f = fortran["radii"]
    r_j = jax_data["radii"]
    nd_f = fortran["nd"]
    nd_j = jax_data["nd"]
    times_f = fortran["times"]
    times_j = jax_data["times"]
    rmass = jax_data["rmass"]
    dr_f = fortran["dr"]
    dr_j = jax_data["dr"]

    nsteps = min(len(times_f), len(times_j))
    times = times_f[:nsteps]
    nd_f = nd_f[:nsteps]
    nd_j = nd_j[:nsteps]

    # --- Figure 1: Initial and final size distributions (number) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # dN/dlog(r) = N / dlog(r)
    dlogr = np.log10(r_f[1] / r_f[0])

    ax = axes[0]
    ax.plot(r_f * 1e4, nd_f[0] / dlogr, "ko-", ms=5, label="Fortran t=0")
    ax.plot(r_j * 1e4, nd_j[0] / dlogr, "r^--", ms=5, label="JAX t=0")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Radius [um]")
    ax.set_ylabel("dN/dlogr [cm$^{-3}$]")
    ax.set_title("Initial Size Distribution")
    ax.legend()
    ax.set_ylim(bottom=1e-2)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(r_f * 1e4, nd_f[-1] / dlogr, "ko-", ms=5, label=f"Fortran t={times[-1]:.0f}s")
    ax.plot(r_j * 1e4, nd_j[-1] / dlogr, "r^--", ms=5, label=f"JAX t={times[-1]:.0f}s")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Radius [um]")
    ax.set_ylabel("dN/dlogr [cm$^{-3}$]")
    ax.set_title("Final Size Distribution (12 hours)")
    ax.legend()
    ax.set_ylim(bottom=1e-2)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Number Size Distribution: Fortran vs JAX", fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "01_size_distribution_number.png", dpi=150)
    plt.close(fig)

    # --- Figure 2: Mass size distributions ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    mass_f_init = nd_f[0] * rmass
    mass_f_final = nd_f[-1] * rmass
    mass_j_init = nd_j[0] * rmass
    mass_j_final = nd_j[-1] * rmass

    ax = axes[0]
    ax.plot(r_f * 1e4, mass_f_init / dlogr, "ko-", ms=5, label="Fortran t=0")
    ax.plot(r_j * 1e4, mass_j_init / dlogr, "r^--", ms=5, label="JAX t=0")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Radius [um]")
    ax.set_ylabel("dM/dlogr [g/cm$^{3}$]")
    ax.set_title("Initial Mass Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(r_f * 1e4, mass_f_final / dlogr, "ko-", ms=5, label=f"Fortran t={times[-1]:.0f}s")
    ax.plot(r_j * 1e4, mass_j_final / dlogr, "r^--", ms=5, label=f"JAX t={times[-1]:.0f}s")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Radius [um]")
    ax.set_ylabel("dM/dlogr [g/cm$^{3}$]")
    ax.set_title("Final Mass Distribution (12 hours)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Mass Size Distribution: Fortran vs JAX", fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "02_size_distribution_mass.png", dpi=150)
    plt.close(fig)

    # --- Figure 3: Bin-by-bin relative error at selected times ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    time_indices = [1, 12, 36, -1]
    time_labels = ["t=600s (step 1)", "t=7200s (step 12)", "t=21600s (step 36)", f"t={times[-1]:.0f}s (final)"]

    for ax, ti, label in zip(axes.flat, time_indices, time_labels):
        nd_fi = nd_f[ti]
        nd_ji = nd_j[ti]

        # Relative error where Fortran has significant values
        mask = nd_fi > 1e-40
        rel_err = np.full_like(nd_fi, np.nan)
        rel_err[mask] = (nd_ji[mask] - nd_fi[mask]) / nd_fi[mask]

        ax.bar(np.arange(1, len(rel_err) + 1), rel_err * 100, color="steelblue", alpha=0.8)
        ax.set_xlabel("Bin index")
        ax.set_ylabel("Relative error [%]")
        ax.set_title(label)
        ax.axhline(0, color="k", lw=0.5)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Bin-by-Bin Relative Error (JAX - Fortran) / Fortran", fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "03_binwise_relative_error.png", dpi=150)
    plt.close(fig)

    # --- Figure 3b: Concentration + error combined (dual y-axis) ---
    time_indices_3b = [1, 12, 36, -1]
    time_labels_3b = ["t=600s (step 1)", "t=7200s (step 12)", "t=21600s (step 36)", f"t={times[-1]:.0f}s (final)"]
    bins = np.arange(1, fortran["nbin"] + 1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    for ax, ti, label in zip(axes.flat, time_indices_3b, time_labels_3b):
        nd_fi = nd_f[ti]
        nd_ji = nd_j[ti]

        # Relative error
        mask = nd_fi > 1e-40
        rel_err = np.full_like(nd_fi, np.nan)
        rel_err[mask] = (nd_ji[mask] - nd_fi[mask]) / nd_fi[mask]

        # Left axis: number concentration
        color_f = "black"
        color_j = "tab:red"
        ax.semilogy(bins, nd_fi, "o-", color=color_f, ms=5, lw=1.5, label="Fortran N", zorder=3)
        ax.semilogy(bins, nd_ji, "^--", color=color_j, ms=5, lw=1.5, label="JAX N", zorder=3)
        ax.set_xlabel("Bin index")
        ax.set_ylabel("Number concentration [cm$^{-3}$]")
        ax.set_ylim(bottom=max(nd_fi[nd_fi > 0].min() * 0.1, 1e-40) if (nd_fi > 0).any() else 1e-40)
        ax.grid(True, alpha=0.2)

        # Right axis: relative error
        ax2 = ax.twinx()
        color_err = "tab:blue"
        valid = ~np.isnan(rel_err) & (nd_fi > 1e-10)
        ax2.bar(bins[valid], rel_err[valid] * 100, color=color_err, alpha=0.25, width=0.6, label="Rel. error")
        ax2.set_ylabel("Relative error [%]", color=color_err)
        ax2.tick_params(axis="y", labelcolor=color_err)
        ax2.axhline(0, color=color_err, lw=0.5, ls=":")

        ax.set_title(label, fontsize=11)

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

    fig.suptitle("Number Concentration and Relative Error by Bin", fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "03b_concentration_and_error.png", dpi=150)
    plt.close(fig)

    # --- Figure 3c: Per-bin time series with error (selected bins) ---
    selected_bins = [0, 2, 4, 6, 8, 10]  # 0-based
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, ib in zip(axes.flat, selected_bins):
        nd_f_bin = nd_f[:, ib]
        nd_j_bin = nd_j[:, ib]

        mask = nd_f_bin > 1e-40
        rel_err_bin = np.full_like(nd_f_bin, np.nan)
        rel_err_bin[mask] = (nd_j_bin[mask] - nd_f_bin[mask]) / nd_f_bin[mask]

        ax.semilogy(times / 3600, nd_f_bin, "k-", lw=1.5, label="Fortran")
        ax.semilogy(times / 3600, nd_j_bin, "r--", lw=1.5, label="JAX")
        ax.set_xlabel("Time [hours]")
        ax.set_ylabel("N [cm$^{-3}$]")
        ax.set_title(f"Bin {ib+1} (r={r_f[ib]*1e4:.4f} um)", fontsize=10)
        ax.grid(True, alpha=0.2)

        ax2 = ax.twinx()
        valid_t = ~np.isnan(rel_err_bin)
        ax2.plot(times[valid_t] / 3600, rel_err_bin[valid_t] * 100, "b-", alpha=0.5, lw=0.8, label="Rel. error")
        ax2.set_ylabel("Rel. error [%]", color="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:blue")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="best")

    fig.suptitle("Per-Bin Time Series: Concentration and Relative Error", fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "03c_per_bin_timeseries.png", dpi=150)
    plt.close(fig)

    # --- Figure 4: Total number and mass vs time ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    total_n_f = nd_f.sum(axis=1)
    total_n_j = nd_j.sum(axis=1)
    total_m_f = (nd_f * rmass[None, :]).sum(axis=1)
    total_m_j = (nd_j * rmass[None, :]).sum(axis=1)

    ax = axes[0]
    ax.plot(times / 3600, total_n_f, "ko-", ms=3, label="Fortran")
    ax.plot(times / 3600, total_n_j, "r^--", ms=3, label="JAX")
    ax.set_xlabel("Time [hours]")
    ax.set_ylabel("Total number [cm$^{-3}$]")
    ax.set_title("Total Number Concentration vs Time")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(times / 3600, total_m_f, "ko-", ms=3, label="Fortran")
    ax.plot(times / 3600, total_m_j, "r^--", ms=3, label="JAX")
    ax.set_xlabel("Time [hours]")
    ax.set_ylabel("Total mass [g/cm$^{3}$]")
    ax.set_title("Total Mass Concentration vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Integrated Quantities: Fortran vs JAX", fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "04_totals_vs_time.png", dpi=150)
    plt.close(fig)

    # --- Figure 5: Error metrics vs time ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Total number relative error
    ax = axes[0, 0]
    rel_err_n = (total_n_j - total_n_f) / total_n_f
    ax.plot(times / 3600, rel_err_n * 100, "b.-")
    ax.set_xlabel("Time [hours]")
    ax.set_ylabel("Relative error [%]")
    ax.set_title("Total Number: (JAX-Fortran)/Fortran")
    ax.axhline(0, color="k", lw=0.5)
    ax.grid(True, alpha=0.3)

    # Total mass relative error
    ax = axes[0, 1]
    rel_err_m = (total_m_j - total_m_f) / np.maximum(total_m_f, 1e-50)
    ax.plot(times / 3600, rel_err_m * 100, "b.-")
    ax.set_xlabel("Time [hours]")
    ax.set_ylabel("Relative error [%]")
    ax.set_title("Total Mass: (JAX-Fortran)/Fortran")
    ax.axhline(0, color="k", lw=0.5)
    ax.grid(True, alpha=0.3)

    # Max absolute relative error per timestep (over bins with significant concentration)
    ax = axes[1, 0]
    max_rel_err = []
    mean_rel_err = []
    for i in range(nsteps):
        mask = nd_f[i] > 1e-10
        if mask.any():
            re = np.abs((nd_j[i][mask] - nd_f[i][mask]) / nd_f[i][mask])
            max_rel_err.append(re.max())
            mean_rel_err.append(re.mean())
        else:
            max_rel_err.append(0.0)
            mean_rel_err.append(0.0)
    ax.semilogy(times / 3600, max_rel_err, "r.-", label="Max |rel err|")
    ax.semilogy(times / 3600, mean_rel_err, "b.-", label="Mean |rel err|")
    ax.set_xlabel("Time [hours]")
    ax.set_ylabel("|Relative error|")
    ax.set_title("Per-Bin Error Envelope (bins > 1e-10 cm$^{-3}$)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mass conservation check (JAX only — should be perfect)
    ax = axes[1, 1]
    mass_ratio_j = total_m_j / total_m_j[0]
    mass_ratio_f = total_m_f / total_m_f[0]
    ax.plot(times / 3600, (mass_ratio_j - 1) * 1e15, "r.-", label="JAX")
    ax.plot(times / 3600, (mass_ratio_f - 1) * 1e15, "k.-", label="Fortran")
    ax.set_xlabel("Time [hours]")
    ax.set_ylabel("(M/M0 - 1) x 1e15")
    ax.set_title("Mass Conservation (deviation from initial)")
    ax.legend()
    ax.axhline(0, color="k", lw=0.5)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Error Metrics vs Time", fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "05_error_metrics_vs_time.png", dpi=150)
    plt.close(fig)

    # --- Figure 6: Time evolution of size distribution (waterfall) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    step_indices = [0, 6, 12, 24, 36, 48, 60, 72]

    ax = axes[0]
    cmap = plt.cm.viridis
    for idx, si in enumerate(step_indices):
        if si < nsteps:
            color = cmap(idx / len(step_indices))
            ax.plot(r_f * 1e4, nd_f[si] / dlogr, "o-", ms=3, color=color,
                    label=f"t={times[si]/3600:.1f}h")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Radius [um]")
    ax.set_ylabel("dN/dlogr [cm$^{-3}$]")
    ax.set_title("Fortran: Size Distribution Evolution")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=1e-2)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for idx, si in enumerate(step_indices):
        if si < nsteps:
            color = cmap(idx / len(step_indices))
            ax.plot(r_j * 1e4, nd_j[si] / dlogr, "o-", ms=3, color=color,
                    label=f"t={times[si]/3600:.1f}h")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Radius [um]")
    ax.set_ylabel("dN/dlogr [cm$^{-3}$]")
    ax.set_title("JAX: Size Distribution Evolution")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=1e-2)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Size Distribution Time Evolution", fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "06_size_dist_evolution.png", dpi=150)
    plt.close(fig)

    # --- Print summary statistics ---
    print("\n=== VALIDATION SUMMARY ===")
    print(f"Timesteps compared: {nsteps}")
    print(f"Total time: {times[-1]:.0f} s ({times[-1]/3600:.1f} hours)")
    print(f"\nTotal number (final):")
    print(f"  Fortran: {total_n_f[-1]:.6e}")
    print(f"  JAX:     {total_n_j[-1]:.6e}")
    print(f"  Rel err: {rel_err_n[-1]:.4e}")
    print(f"\nTotal mass (final):")
    print(f"  Fortran: {total_m_f[-1]:.6e}")
    print(f"  JAX:     {total_m_j[-1]:.6e}")
    print(f"  Rel err: {rel_err_m[-1]:.4e}")
    print(f"\nMax per-bin relative error (final, bins > 1e-10):")
    print(f"  {max_rel_err[-1]:.4e}")
    print(f"\nMass conservation (JAX): M_final/M_init - 1 = {float(mass_ratio_j[-1]-1):.4e}")
    print(f"Mass conservation (Fortran): M_final/M_init - 1 = {float(mass_ratio_f[-1]-1):.4e}")
    print(f"\nPlots saved to: {outdir.resolve()}")


# ---------- Main ----------

if __name__ == "__main__":
    bench_path = Path(__file__).parent.parent / "tests" / "reference_data" / "carma_coagtest.txt"
    if not bench_path.exists():
        # Try original location
        bench_path = Path(__file__).parent.parent.parent / "original-carma" / "CARMA" / "tests" / "bench" / "carma_coagtest.txt"

    print("Parsing Fortran benchmark...")
    fortran = parse_coagtest_bench(bench_path)
    print(f"  {fortran['nbin']} bins, {len(fortran['times'])} timesteps")

    print("Running JAX coagulation simulation...")
    jax_data = run_jax_coagtest()
    print(f"  {len(jax_data['times'])} timesteps")

    print("Generating plots...")
    outdir = Path(__file__).parent.parent / "plots" / "coagtest_validation"
    make_plots(fortran, jax_data, outdir)
