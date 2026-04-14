"""Statistical validation of JAX coagulation across 1000 scenarios.

Compares JAX total number against analytical Smoluchowski solution for
constant kernel. Checks mass conservation. Uses precomputed setup to
minimize overhead.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time as timer

from carma.precision import DTYPE
from carma.constants import PI, RM2CGS, RPA2CGS, SMALL_PC, BK
from carma.bins import setup_bins
from carma.atmosphere_std import get_standard_atmosphere
from carma.setup_atm import setup_atm
from carma.enums import GridType, ElementType
from carma.coagulation.setup_coag import setup_coag
from carma.config import ElementConfig, GroupConfig, CarmaConfig
from carma.microslow import microslow
from carma.coagulation.coagl import coagl
from carma.coagulation.csolve import csolve
from carma.coagulation.coagp import coagp


def fast_coag_step(config, pc, ckernel, zmet, dtime):
    """Fast single-step coagulation (no overhead from full microslow)."""
    nz, nbin, nelem = pc.shape
    pcl = pc
    pconmax = jnp.max(pc[:, :, 0:1] / zmet[:, None, None], axis=1)
    coaglg = coagl(config, ckernel, pcl, pconmax)
    coagpe = jnp.zeros_like(pc)
    for ielem in range(nelem):
        igroup = config.elements[ielem].igroup
        for ibin in range(nbin):
            coagpe = coagp(config, ckernel, pc, pcl, pconmax, coagpe, ibin, ielem)
            pc = csolve(pc, coagpe, coaglg, zmet, dtime, ibin, ielem, igroup)
    return pc


def main():
    np.random.seed(42)
    n_scenarios = 1000
    nbin = 20
    nz = 1
    nstep = 20

    # Build config ONCE (static setup)
    rmin_cm, rmrat, rho = 3e-7, 2.0, 2.0
    r, rmass, vol, dr, dm, rup, rlow, rmassup = setup_bins(rmin_cm, rmrat, nbin, rho)
    rmass_np = np.array(rmass)

    groups = (GroupConfig(
        name='dust', ishape=1, ienconc=0, is_ice=False, is_cloud=False,
        is_sulfate=False, do_vtran=False, do_drydep=False, ifallrtn=1,
        irhswell=0, rmrat=rmrat, eshape=1.0, rmin=rmin_cm,
        r=r, rmass=rmass, vol=vol, dr=dr, dm=dm, rmassup=rmassup, rup=rup, rlow=rlow,
        rrat=jnp.ones(nbin, dtype=DTYPE), rprat=jnp.ones(nbin, dtype=DTYPE),
        arat=jnp.ones(nbin, dtype=DTYPE),
    ),)
    elements = (ElementConfig(
        name='dust', rho=jnp.full(nbin, rho, dtype=DTYPE), igroup=0,
        itype=int(ElementType.I_INVOLATILE), icomposition=0, isolute=-1, kappa=0.0,
    ),)
    coag = setup_coag(nbin, 1, 1, groups, elements,
                      np.array([[0]], dtype=np.int32), np.array([[0]], dtype=np.int32))
    config = CarmaConfig(
        nbin=nbin, nelem=1, ngroup=1, ngas=0, nsolute=0,
        elements=elements, groups=groups, gases=(), solutes=(), coag=coag,
        do_coag=True, do_grow=False, do_vtran=False, do_vdiff=False,
        do_thermo=False, do_substep=False, do_explised=False,
        do_incloud=False, do_clearsky=False, do_detrain=False,
        do_pheat=False, do_pheatatm=False, do_cnst_rlh=False,
        itbnd_pc=1, ibbnd_pc=1,
        maxsubsteps=1, minsubsteps=1, maxretries=5, conmax=0.0,
        cstick=1.0, gsticki=1.0, gstickl=1.0, tstick=1.0, dt_threshold=0.0,
        igash2o=-1, igash2so4=-1, igasso2=-1,
    )

    # Setup atmosphere ONCE (single level)
    deltaz = 100.0 * RM2CGS
    zc = jnp.array([0.5 * deltaz])
    zl = jnp.array([0.0, deltaz])
    p_pa, t = get_standard_atmosphere(zc / RM2CGS)
    pl_pa, _ = get_standard_atmosphere(zl / RM2CGS)
    p, pl = p_pa * RPA2CGS, pl_pa * RPA2CGS
    rhoa, dz, zmet, zmetl, rmu, thcond, rhoa_wet = setup_atm(t, p, pl, zc, zl, GridType.I_CART)

    # Random parameters
    log_N0 = np.random.uniform(2, 8, n_scenarios)
    N0_arr = 10.0 ** log_N0
    init_bins = np.random.randint(0, 5, n_scenarios)
    log_ck0 = np.random.uniform(-11, -8, n_scenarios)
    ck0_arr = 10.0 ** log_ck0
    dtime_arr = np.random.uniform(60, 1200, n_scenarios)

    # Results storage
    rel_err_totalN = np.zeros(n_scenarios)
    mass_conserv = np.zeros(n_scenarios)
    N0_used = np.zeros(n_scenarios)
    tau_arr = np.zeros(n_scenarios)  # dimensionless coag parameter

    # Per-bin error storage (for bins > threshold)
    all_bin_errors_gt10 = []
    all_bin_errors_gt100 = []
    all_bin_errors_gt1000 = []

    # Warmup: run one scenario to trigger any lazy init
    print("Warming up...", flush=True)
    ck_warmup = jnp.full((nz, nbin, nbin, 1, 1), 1e-9, dtype=DTYPE)
    pc_warmup = jnp.full((nz, nbin, 1), SMALL_PC, dtype=DTYPE).at[0, 0, 0].set(1e6)
    _ = fast_coag_step(config, pc_warmup, ck_warmup, zmet, 600.0)
    print("Warmup done.", flush=True)

    print(f"Running {n_scenarios} scenarios...", flush=True)
    t_start = timer.time()

    for i in range(n_scenarios):
        N0 = N0_arr[i]
        ib = init_bins[i]
        ck0 = ck0_arr[i]
        dt = dtime_arr[i]
        t_final = nstep * dt

        # Constant kernel
        ckernel = jnp.full((nz, nbin, nbin, 1, 1), ck0, dtype=DTYPE)

        # Initial conditions
        pc = jnp.full((nz, nbin, 1), SMALL_PC, dtype=DTYPE)
        pc = pc.at[0, ib, 0].set(DTYPE(N0))

        mass_init = float((pc[0, :, 0] * rmass).sum())

        # Time integration
        for istep in range(nstep):
            pc = fast_coag_step(config, pc, ckernel, zmet, dt)

        nd_final = np.array(pc[0, :, 0])
        N_jax = nd_final.sum()
        mass_final = float((nd_final * rmass_np).sum())

        # Analytical Smoluchowski: N(t) = N0 / (1 + N0*K*t/2)
        N_anal = N0 / (1.0 + N0 * ck0 * t_final / 2.0)
        tau = N0 * ck0 * t_final / 2.0

        N0_used[i] = N0
        tau_arr[i] = tau
        rel_err_totalN[i] = (N_jax - N_anal) / N_anal if N_anal > 1 else 0.0
        mass_conserv[i] = abs(mass_final / mass_init - 1.0) if mass_init > 0 else 0.0

        # Per-bin: collect errors relative to a reference
        # Use analytical total as sanity, but bin-level we check against itself
        # (no bin-level analytical solution exists for discrete bins)
        # Instead: track which bins have >threshold and store the concentration
        for threshold, err_list in [
            (10, all_bin_errors_gt10),
            (100, all_bin_errors_gt100),
            (1000, all_bin_errors_gt1000),
        ]:
            mask = nd_final > threshold
            if mask.any():
                # Relative deviation from Smoluchowski for total is our proxy
                # per-bin we can only check mass conservation quality
                pass

        if (i + 1) % 200 == 0:
            elapsed = timer.time() - t_start
            rate = (i + 1) / elapsed
            eta = (n_scenarios - i - 1) / rate
            print(f"  {i+1}/{n_scenarios} ({elapsed:.0f}s, ~{eta:.0f}s left)", flush=True)

    elapsed = timer.time() - t_start
    print(f"Done in {elapsed:.1f}s ({n_scenarios/elapsed:.1f} scenarios/s)", flush=True)

    # --- Summary ---
    # Filter: only scenarios where analytical N > 10 (meaningful comparison)
    N_anal_all = N0_arr / (1.0 + N0_arr * ck0_arr * nstep * dtime_arr / 2.0)
    significant = N_anal_all > 10
    errs_sig = rel_err_totalN[significant]

    print("\n" + "=" * 70)
    print("STATISTICAL VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Scenarios: {n_scenarios} ({significant.sum()} with N_final > 10)")
    print(f"N0 range: [{N0_arr.min():.1e}, {N0_arr.max():.1e}] cm^-3")
    print(f"K range:  [{ck0_arr.min():.1e}, {ck0_arr.max():.1e}] cm^3/s")
    print(f"dt range: [{dtime_arr.min():.0f}, {dtime_arr.max():.0f}] s")
    print(f"tau range:[{tau_arr.min():.2e}, {tau_arr.max():.2e}]")

    print(f"\n--- Total Number vs Smoluchowski (N_final > 10 cm^-3) ---")
    print(f"  Samples:        {len(errs_sig)}")
    print(f"  Mean |error|:   {np.abs(errs_sig).mean():.6e}")
    print(f"  Median |error|: {np.median(np.abs(errs_sig)):.6e}")
    print(f"  Max |error|:    {np.abs(errs_sig).max():.6e}")
    print(f"  Std:            {errs_sig.std():.6e}")
    for thresh in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        pct = (np.abs(errs_sig) < thresh).mean() * 100
        print(f"  % below {thresh:.0e}:  {pct:.1f}%")

    print(f"\n--- Mass Conservation ---")
    print(f"  Mean |M/M0-1|:  {mass_conserv.mean():.4e}")
    print(f"  Max |M/M0-1|:   {mass_conserv.max():.4e}")
    print(f"  % < 1e-14:      {(mass_conserv < 1e-14).mean()*100:.1f}%")

    # --- Plots ---
    outdir = Path(__file__).parent.parent / "plots" / "statistical_validation"
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Histogram of total N relative error
    ax = axes[0, 0]
    ax.hist(errs_sig * 100, bins=80, color="steelblue", alpha=0.8, edgecolor="white", lw=0.3)
    ax.axvline(0, color="k", lw=1)
    mn = np.mean(errs_sig) * 100
    ax.axvline(mn, color="r", lw=1.5, ls="--", label=f"mean={mn:.4f}%")
    ax.set_xlabel("Relative error [%]")
    ax.set_ylabel("Count")
    ax.set_title(f"Total Number Error vs Smoluchowski (n={len(errs_sig)})")
    ax.legend()
    ax.grid(True, alpha=0.2)

    # 2. |Error| vs tau (dimensionless coagulation parameter)
    ax = axes[0, 1]
    ax.scatter(tau_arr[significant], np.abs(errs_sig), s=5, alpha=0.4, c="steelblue")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\tau = N_0 K t_{final} / 2$")
    ax.set_ylabel("|Relative error|")
    ax.set_title("Error vs Coagulation Extent")
    ax.axhline(1e-6, color="r", ls="--", lw=1.5, label="1e-6 target")
    ax.axhline(1e-4, color="orange", ls="--", lw=1, label="1e-4")
    ax.axhline(1e-2, color="green", ls="--", lw=1, label="1e-2")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # 3. Mass conservation
    ax = axes[1, 0]
    mc_log = np.log10(np.maximum(mass_conserv, 1e-18))
    ax.hist(mc_log, bins=50, color="forestgreen", alpha=0.8, edgecolor="white", lw=0.3)
    ax.axvline(-14, color="r", ls="--", lw=1.5, label="1e-14 (machine eps)")
    ax.set_xlabel("log10(|M_final/M_init - 1|)")
    ax.set_ylabel("Count")
    ax.set_title(f"Mass Conservation (n={n_scenarios})")
    ax.legend()
    ax.grid(True, alpha=0.2)

    # 4. |Error| vs N0
    ax = axes[1, 1]
    ax.scatter(N0_arr[significant], np.abs(errs_sig), s=5, alpha=0.4, c="steelblue")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N0 [cm$^{-3}$]")
    ax.set_ylabel("|Relative error|")
    ax.set_title("Error vs Initial Concentration")
    ax.axhline(1e-6, color="r", ls="--", lw=1.5, label="1e-6")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    fig.suptitle(f"Statistical Validation: {n_scenarios} Random Scenarios", fontsize=14)
    fig.tight_layout()
    fig.savefig(outdir / "statistical_validation.png", dpi=150)
    plt.close(fig)

    # 5. CDF of absolute error
    fig, ax = plt.subplots(figsize=(8, 5))
    sorted_err = np.sort(np.abs(errs_sig))
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
    ax.plot(sorted_err, cdf * 100, "b-", lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("|Relative error|")
    ax.set_ylabel("Cumulative %")
    ax.set_title(f"CDF of |Relative Error| (n={len(errs_sig)} scenarios)")
    for thresh in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        pct = (sorted_err < thresh).mean() * 100
        ax.axvline(thresh, color="gray", ls=":", lw=0.8)
        ax.text(thresh * 1.2, max(10, pct - 5), f"{pct:.0f}%", fontsize=8, color="gray")
    ax.axhline(99, color="r", ls="--", lw=0.8, alpha=0.5)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(outdir / "error_cdf.png", dpi=150)
    plt.close(fig)

    print(f"\nPlots saved to: {outdir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
