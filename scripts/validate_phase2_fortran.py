"""Phase 2 validation: 1000 scenarios, computed kernel, Fortran vs JAX.

47 bins (0.2nm-8um), dt=60s, 12 hours. Uses COMPUTED Brownian+gravitational
kernel (not constant ck0) — tests the full Phase 2 chain:
atmosphere → fall velocity → coagulation kernel → coagulation.
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
from carma.constants import RM2CGS, RPA2CGS, SMALL_PC, BK, PI
from carma.bins import setup_bins
from carma.atmosphere_std import get_standard_atmosphere
from carma.setup_atm import setup_atm
from carma.enums import GridType, ElementType
from carma.coagulation.setup_coag import setup_coag
from carma.config import ElementConfig, GroupConfig, CarmaConfig
from carma.setup_vf import setup_vf
from carma.setup_ckern import setup_ckern
from carma.microslow import make_microslow

SCRIPT_DIR = Path(__file__).parent
FORTRAN_DIR = SCRIPT_DIR / "fortran_runner"
FORTRAN_EXE = FORTRAN_DIR / "test_phase2"

NBIN = 47
NZ = 1
RMIN_CM = 2e-8
RMRAT = 2.0
RHO = 2.0
DT = 60.0
NSTEP = 720  # 12 hours


def parse_phase2_output(filepath):
    with open(filepath) as f:
        lines = f.readlines()
    idx = 0
    parts = lines[idx].split()
    nbin = int(parts[0])
    nstep = int(parts[1])
    t_val = float(parts[2])
    p_val = float(parts[3])
    idx += 1

    radii = np.zeros(nbin)
    for i in range(nbin):
        parts = lines[idx].split()
        radii[i] = float(parts[1])
        idx += 1

    nd_final = np.zeros(nbin)
    for i in range(nbin):
        parts = lines[idx].split()
        nd_final[i] = float(parts[1])
        idx += 1

    return radii, nd_final, t_val, p_val


def run_fortran(init_nd, nstep):
    nml = f"&phase2_params param_dtime={DT:.1f}, param_nstep={nstep} /\n"
    (FORTRAN_DIR / "phase2_params.nml").write_text(nml)
    nd_lines = "\n".join(f"{v:.6e}" for v in init_nd) + "\n"
    (FORTRAN_DIR / "init_nd.txt").write_text(nd_lines)

    t0 = timer.time()
    result = subprocess.run(
        [str(FORTRAN_EXE)], cwd=str(FORTRAN_DIR),
        capture_output=True, text=True, timeout=120
    )
    t1 = timer.time()
    if result.returncode != 0:
        raise RuntimeError(f"Fortran failed: {result.stderr}")
    radii, nd_final, t_val, p_val = parse_phase2_output(
        FORTRAN_DIR / "carma_phase2_out.txt"
    )
    return nd_final, t1 - t0


def main():
    np.random.seed(42)
    n_scenarios = 1000

    # JAX setup
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

    config = CarmaConfig(
        nbin=NBIN, nelem=1, ngroup=1, ngas=0, nsolute=0,
        elements=elements, groups=groups, gases=(), solutes=(), coag=coag,
        do_coag=True, do_grow=False, do_vtran=False, do_vdiff=False,
        do_thermo=False, do_substep=False, do_explised=False,
        do_incloud=False, do_clearsky=False, do_detrain=False,
        do_pheat=False, do_pheatatm=False, do_cnst_rlh=False,
        itbnd_pc=1, ibbnd_pc=1, maxsubsteps=1, minsubsteps=1, maxretries=5, conmax=0.0,
        cstick=1.0, gsticki=1.0, gstickl=1.0, tstick=1.0, dt_threshold=0.0,
        igash2o=-1, igash2so4=-1, igasso2=-1,
    )

    # Atmosphere (same as Fortran: z=50m, standard atmosphere)
    deltaz = 100.0 * RM2CGS
    zc = jnp.array([0.5 * deltaz])
    zl = jnp.array([0.0, deltaz])
    p_pa, t = get_standard_atmosphere(zc / RM2CGS)
    pl_pa, _ = get_standard_atmosphere(zl / RM2CGS)
    p_cgs, pl_cgs = p_pa * RPA2CGS, pl_pa * RPA2CGS
    rhoa, dz, zmet, zmetl, rmu, thcond, rhoa_wet = setup_atm(
        t, p_cgs, pl_cgs, zc, zl, GridType.I_CART
    )

    # Compute fall velocity and kernel
    r_wet = jnp.broadcast_to(r[None, :, None], (NZ, NBIN, 1))
    rhop_wet = jnp.full((NZ, NBIN, 1), RHO, dtype=DTYPE)
    rrat_arr = jnp.ones((NBIN, 1), dtype=DTYPE)
    rprat_arr = jnp.ones((NBIN, 1), dtype=DTYPE)
    rmass_arr = rmass[:, None] * jnp.ones((1, 1), dtype=DTYPE)

    vf, re, bpm = setup_vf(config, t, rhoa, zmet, rmu, r_wet, rhop_wet, rrat_arr, rprat_arr)
    ckernel = setup_ckern(config, t, rhoa, zmet, rmu, r_wet, rrat_arr, rprat_arr,
                          bpm, rmass_arr, re, vf)

    print(f"JAX computed kernel range: {float(ckernel.min()):.4e} to {float(ckernel.max()):.4e}", flush=True)

    # Build JIT time stepper
    microslow_fn = make_microslow(
        nbin=NBIN, nelem=1, ngroup=1,
        elem_igroup=jnp.array([0]),
        icoag=coag.icoag, volx=coag.volx, icoagelem=coag.icoagelem,
        npairu=coag.npairu, npairl=coag.npairl,
        iup=coag.iup, jup=coag.jup, igup=coag.igup, jgup=coag.jgup,
        ilow=coag.ilow, jlow=coag.jlow, iglow=coag.iglow, jglow=coag.jglow,
        pkernel=coag.pkernel,
        ienconc_arr=jnp.array([0]),
        elem_itypes=jnp.array([int(ElementType.I_INVOLATILE)]),
    )

    @jax.jit
    def run_jax_sim(pc, ckernel, zmet):
        def step(pc, _):
            pcl = pc
            pconmax = jnp.max(pc[:, :, 0:1] / zmet[:, None, None], axis=1)
            return microslow_fn(pc, pcl, ckernel, pconmax, zmet, DTYPE(DT)), None
        pc_final, _ = jax.lax.scan(step, pc, None, length=NSTEP)
        return pc_final

    # Warmup
    print("JIT compiling...", flush=True)
    pc_w = jnp.full((NZ, NBIN, 1), SMALL_PC, dtype=DTYPE).at[0, 0, 0].set(1e6)
    _ = run_jax_sim(pc_w, ckernel, zmet)
    print("Done.", flush=True)

    # Random scenarios: bins 1-35 populated
    all_init_nd = np.zeros((n_scenarios, NBIN))
    for i in range(n_scenarios):
        for b in range(35):
            all_init_nd[i, b] = 10.0 ** np.random.uniform(3, 9)

    # Storage
    fortran_total = 0.0
    jax_total = 0.0
    rel_err_totalN = np.zeros(n_scenarios)
    rel_err_totalM = np.zeros(n_scenarios)
    all_f_final = np.zeros((n_scenarios, NBIN))
    all_j_final = np.zeros((n_scenarios, NBIN))
    all_bin_rel_err = np.full((n_scenarios, NBIN), np.nan)

    print(f"Running {n_scenarios} scenarios (computed kernel, 47 bins, dt=60s, 12h)...", flush=True)
    t_start = timer.time()

    for i in range(n_scenarios):
        init_nd = all_init_nd[i]

        # Fortran
        f_nd, f_elapsed = run_fortran(init_nd, NSTEP)
        fortran_total += f_elapsed
        all_f_final[i] = f_nd

        # JAX
        t0 = timer.time()
        pc = jnp.full((NZ, NBIN, 1), SMALL_PC, dtype=DTYPE)
        pc = pc.at[0, :, 0].set(jnp.array(init_nd, dtype=DTYPE))
        pc_final = run_jax_sim(pc, ckernel, zmet)
        jax.block_until_ready(pc_final)
        t1 = timer.time()
        jax_total += t1 - t0

        j_nd = np.array(pc_final[0, :, 0])
        all_j_final[i] = j_nd

        # Compare
        N_f, N_j = f_nd.sum(), j_nd.sum()
        if N_f > 1:
            rel_err_totalN[i] = (N_j - N_f) / N_f
        M_f = (f_nd * rmass_np).sum()
        M_j = (j_nd * rmass_np).sum()
        if M_f > 1e-50:
            rel_err_totalM[i] = (M_j - M_f) / M_f

        for b in range(NBIN):
            if f_nd[b] > 10.0:
                all_bin_rel_err[i, b] = (j_nd[b] - f_nd[b]) / f_nd[b]

        if (i + 1) % 100 == 0:
            elapsed = timer.time() - t_start
            print(f"  {i+1}/{n_scenarios} ({elapsed:.0f}s)", flush=True)

    wall = timer.time() - t_start
    ae = np.abs(rel_err_totalN)

    print("\n" + "=" * 70)
    print(f"PHASE 2 VALIDATION: COMPUTED KERNEL ({n_scenarios} scenarios)")
    print("=" * 70)
    print(f"\n--- Setup ---")
    print(f"  Bins: {NBIN} (0.2nm-8um), dt={DT}s, {NSTEP} steps (12h)")
    print(f"  Kernel: COMPUTED (Brownian + gravitational, Fuchs collection)")
    print(f"  Init: bins 1-35 each 1e3-1e9 cm^-3 randomly")
    print(f"\n--- Timing ---")
    print(f"  Fortran: {fortran_total:.1f}s ({fortran_total/n_scenarios*1000:.1f} ms/scenario)")
    print(f"  JAX:     {jax_total:.1f}s ({jax_total/n_scenarios*1000:.1f} ms/scenario)")
    print(f"  Speedup: {fortran_total/max(jax_total, 0.001):.1f}x")
    print(f"  Wall:    {wall:.0f}s")
    print(f"\n--- Total Number Error ---")
    print(f"  Mean:   {ae.mean():.6e}")
    print(f"  Median: {np.median(ae):.6e}")
    print(f"  Max:    {ae.max():.6e}")
    for th in [1e-1, 1e-2, 1e-3, 1e-4]:
        print(f"  % < {th:.0e}: {(ae<th).mean()*100:.1f}%")
    print(f"\n--- Mass Error ---")
    print(f"  Mean:   {np.abs(rel_err_totalM).mean():.6e}")
    print(f"  Max:    {np.abs(rel_err_totalM).max():.6e}")

    # Plots
    outdir = SCRIPT_DIR.parent / "plots" / "phase2_computed_kernel"
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Error histogram
    ax = axes[0, 0]
    ax.hist(rel_err_totalN * 100, bins=60, color="steelblue", alpha=0.8, edgecolor="white", lw=0.3)
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
    for th in [1e-4, 1e-3, 1e-2, 1e-1]:
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
    bin_data = []
    valid_bins = []
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
    ax.set_ylabel("Number error [%]")
    ax.set_title("Per-Bin Number Error (N > 10)")
    ax.tick_params(axis="x", labelsize=5)
    ax.grid(True, alpha=0.2)

    # 5. Per-bin mass error boxplot
    ax = axes[1, 1]
    mass_bin_data = []
    mass_valid_bins = []
    for b in range(NBIN):
        f_mass = all_f_final[:, b] * rmass_np[b]
        j_mass = all_j_final[:, b] * rmass_np[b]
        mask = f_mass > 1e-50
        if mask.sum() > 5:
            re = (j_mass[mask] - f_mass[mask]) / f_mass[mask]
            mass_bin_data.append(re * 100)
            mass_valid_bins.append(b)
    if mass_bin_data:
        bp = ax.boxplot(mass_bin_data, tick_labels=[str(b+1) for b in mass_valid_bins],
                        patch_artist=True, showfliers=True,
                        flierprops=dict(marker=".", ms=1, alpha=0.2),
                        medianprops=dict(color="red", lw=1.5))
        for patch in bp["boxes"]:
            patch.set_facecolor("coral")
            patch.set_alpha(0.6)
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("Bin index")
    ax.set_ylabel("Mass error [%]")
    ax.set_title("Per-Bin Mass Error")
    ax.tick_params(axis="x", labelsize=5)
    ax.grid(True, alpha=0.2)

    # 6. Example scenario
    ax = axes[1, 2]
    r_np = np.array(r)
    dlogr = np.log10(r_np[1] / r_np[0])
    ex = n_scenarios // 2
    ax.semilogy(r_np * 1e4, all_f_final[ex] / dlogr, "ko-", ms=3, label="Fortran")
    ax.semilogy(r_np * 1e4, all_j_final[ex] / dlogr, "r^--", ms=3, label="JAX")
    ax.set_xscale("log")
    ax.set_xlabel("Radius [um]")
    ax.set_ylabel("dN/dlogr [cm$^{-3}$]")
    ax.set_title(f"Example scenario #{ex} (final)")
    ax.legend()
    ax.set_ylim(bottom=1e-2)
    ax.grid(True, alpha=0.2)

    fig.suptitle(f"Phase 2: Computed Kernel, {n_scenarios} Scenarios, 47 bins, 12h", fontsize=14)
    fig.tight_layout()
    fig.savefig(outdir / "phase2_computed_kernel.png", dpi=150)
    plt.close(fig)

    print(f"\nPlots saved to: {outdir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
