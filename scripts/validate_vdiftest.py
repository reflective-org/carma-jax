"""Validate JAX Brownian diffusion against Fortran carma_vdiftest.

Fortran vdiftest: tiny dust particles diffusing in the mesosphere
- NZ=240, dz=100m, 80-104 km
- 1 bin (rmin=2e-8 cm, 0.2 nm), 1 element, 1 group, dust rho=2.65 g/cm^3
- vf_const = 1e-10 cm/s (effectively off)
- do_vdiff = True
- Initial: narrow Gaussian at 90 km, width 3 km
- dt=1000s, 600 steps (~7 days)
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
from carma.atmosphere_std import get_standard_atmosphere
from carma.setup_atm import setup_atm
from carma.setup_vf import setup_vf_jit
from carma.setup_bdif import setup_bdif
from carma.transport.vertical import vertical
from carma.enums import GridType, BoundaryCondition


def parse_vdiftest_bench(filepath):
    """Parse carma_vdiftest.txt (same format as falltest)."""
    with open(filepath) as f:
        lines = f.readlines()
    nz = int(lines[0].strip())
    idx = 1
    zc = np.zeros(nz)
    dz = np.zeros(nz)
    for i in range(nz):
        parts = lines[idx].split()
        zc[i] = float(parts[1])
        dz[i] = float(parts[2])
        idx += 1

    times = []
    mmr_bins = []
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        try:
            t_val = float(line)
        except ValueError:
            idx += 1
            continue
        times.append(t_val)
        idx += 1
        mmr = np.zeros(nz)
        for i in range(nz):
            parts = lines[idx].split()
            mmr[i] = float(parts[1])
            idx += 1
        mmr_bins.append(mmr)

    return {"nz": nz, "zc": zc, "dz": dz,
            "times": np.array(times), "mmr": np.array(mmr_bins)}


def main():
    bench_path = (Path(__file__).parent.parent.parent
                  / "original-carma" / "CARMA" / "tests" / "bench" / "carma_vdiftest.txt")
    fortran = parse_vdiftest_bench(bench_path)
    print(f"Fortran benchmark: NZ={fortran['nz']}, timesteps={len(fortran['times'])}")

    # --- Setup matching Fortran ---
    NZ = 240
    NBIN = 1
    NELEM = 1
    NGROUP = 1
    dtime = 1000.0
    nstep = 600
    deltaz_m = 100.0
    zmin_m = 80000.0
    vf_const = 1e-10  # cm/s (~off, only diffusion matters)
    rmin_cm = 2e-8
    rmrat = 2.0
    rho_aer = 2.65

    r, rmass, vol, dr, dm, rup, rlow, rmassup = setup_bins(rmin_cm, rmrat, NBIN, rho_aer)

    zc_m = np.arange(NZ) * deltaz_m + zmin_m + deltaz_m / 2
    zl_m = np.arange(NZ + 1) * deltaz_m + zmin_m

    p_pa, t_k = get_standard_atmosphere(jnp.array(zc_m, dtype=DTYPE))
    pl_pa, _ = get_standard_atmosphere(jnp.array(zl_m, dtype=DTYPE))

    p_cgs = p_pa * RPA2CGS
    pl_cgs = pl_pa * RPA2CGS
    zc = jnp.array(zc_m) * RM2CGS
    zl = jnp.array(zl_m) * RM2CGS

    rhoa, dz, zmet, zmetl, rmu, thcond, rhoa_wet = setup_atm(
        t_k, p_cgs, pl_cgs, zc, zl, GridType.I_CART)

    rhoa_cgs = rhoa / zmet
    print(f"Atm mesosphere: p={float(p_cgs[0]):.2e} - {float(p_cgs[-1]):.2e} dyne/cm²")
    print(f"Atm mesosphere: T={float(t_k[0]):.1f} - {float(t_k[-1]):.1f} K")
    # Compare rhoa (ideal gas) vs rhoa_wet (hydrostatic) — should be ~equal
    rho_ratio = rhoa_wet / rhoa_cgs
    print(f"rhoa_wet/rhoa ratio: min={float(rho_ratio.min()):.4f} "
          f"max={float(rho_ratio.max()):.4f} mean={float(rho_ratio.mean()):.4f}")

    # --- Particle properties for setup_vf / setup_bdif ---
    # Spherical dust: rrat = rprat = 1 (no shape correction)
    r_wet = jnp.broadcast_to(jnp.array(r, dtype=DTYPE)[None, :, None],
                              (NZ, NBIN, NGROUP))
    rhop_wet = jnp.full((NZ, NBIN, NGROUP), DTYPE(rho_aer), dtype=DTYPE)
    rrat = jnp.ones((NBIN, NGROUP), dtype=DTYPE)
    rprat = jnp.ones((NBIN, NGROUP), dtype=DTYPE)

    # Fall velocity (bpm is what we need from this; vf overridden by vf_const)
    vf_computed, re, bpm = setup_vf_jit(
        t_k, rhoa, zmet, rmu, r_wet, rhop_wet, rrat, rprat)

    # Override vf with the constant from Fortran test
    vf = jnp.full((NZ + 1, NBIN, NGROUP), DTYPE(vf_const), dtype=DTYPE)

    # Brownian diffusion coefficient
    dkz = setup_bdif(t_k, rmu, r_wet, bpm, rprat, zmetl, GridType.I_CART)
    print(f"dkz range: {float(dkz.min()):.2e} - {float(dkz.max()):.2e} cm²/s")
    print(f"  dkz samples [cm²/s]:")
    for iz in [0, 50, 100, 150, 200, 239]:
        print(f"    z={zc_m[iz]/1000:6.1f}km T={float(t_k[iz]):6.1f} "
              f"rhoa={float(rhoa_cgs[iz]):.3e} rmu={float(rmu[iz]):.3e} "
              f"bpm={float(bpm[iz,0,0]):.2e} dkz_ctr={float(dkz[iz,0,0]):.3e}")

    # --- Initial condition: narrow Gaussian at 90 km, bin 0 ---
    rho_dry_si = np.array(p_pa) / (287.0 * np.array(t_k))  # kg/m^3
    mmr_init = 1e-10 * np.exp(-5.0 * ((zc_m - 90000.0) / 3000.0) ** 2) / rho_dry_si

    pc_init_bin0 = mmr_init * np.array(rhoa_wet)
    pc = jnp.zeros((NZ, NBIN, NELEM), dtype=DTYPE)
    pc = pc.at[:, 0, 0].set(jnp.array(pc_init_bin0, dtype=DTYPE))

    print(f"Initial total: {float(jnp.sum(pc[:, 0, 0] * dz)):.4e} g/cm²")
    print(f"Fortran init max MMR: {fortran['mmr'][0].max():.3e} at bin {np.argmax(fortran['mmr'][0])}")
    print(f"JAX init max MMR: {float(mmr_init.max()):.3e} at bin {int(np.argmax(mmr_init))}")

    # Boundary conditions: FIXED_CONC (CARMA default)
    vd = jnp.zeros((NBIN, NGROUP), dtype=DTYPE)
    pc_topbnd = jnp.zeros((NBIN, NELEM), dtype=DTYPE)
    pc_botbnd = jnp.zeros((NBIN, NELEM), dtype=DTYPE)
    ftoppart = jnp.zeros((NBIN, NELEM), dtype=DTYPE)
    fbotpart = jnp.zeros((NBIN, NELEM), dtype=DTYPE)

    igroup_arr = np.array([0], dtype=np.int32)
    grp_do_vtran = np.array([True])
    grp_do_drydep = np.array([False])

    # --- Time loop ---
    print(f"\nRunning {nstep} steps at dt={dtime}s...")
    snapshot_steps = [0, 50, 100, 200, 400, 600]
    snapshots = {0: np.array(pc[:, 0, 0])}

    for istep in range(nstep):
        pc, sedflux = vertical(
            pc, vf, dkz, vd, dz, zc, zl, rhoa, zmet, DTYPE(dtime),
            itbnd_pc=int(BoundaryCondition.I_FIXED_CONC),
            ibbnd_pc=int(BoundaryCondition.I_FIXED_CONC),
            pc_topbnd=pc_topbnd, pc_botbnd=pc_botbnd,
            ftoppart=ftoppart, fbotpart=fbotpart,
            igroup_arr=igroup_arr,
            grp_do_vtran=grp_do_vtran, grp_do_drydep=grp_do_drydep,
            igridv=GridType.I_CART, nbin=NBIN, nelem=NELEM, ngroup=NGROUP)
        if (istep + 1) in snapshot_steps:
            snapshots[istep + 1] = np.array(pc[:, 0, 0])
        if (istep + 1) % 100 == 0:
            tot = float(jnp.sum(pc[:, 0, 0] * dz))
            print(f"  Step {istep+1:3d}: total={tot:.4e} g/cm²")

    # --- Compare with Fortran ---
    outdir = Path("plots/vdiftest_validation")
    outdir.mkdir(parents=True, exist_ok=True)

    jax_final = np.array(pc[:, 0, 0])
    fort_final = fortran["mmr"][-1]
    jax_mmr_final = jax_final / np.array(rhoa_wet)

    print(f"\nError at snapshot times:")
    for step in [50, 100, 200, 400, 600]:
        jax_s = snapshots.get(step)
        if jax_s is None or step >= len(fortran["mmr"]):
            continue
        jax_mmr_s = jax_s / np.array(rhoa_wet)
        fort_s = fortran["mmr"][step]
        m = fort_s > 1e-30
        if m.any():
            re = np.abs(jax_mmr_s[m] - fort_s[m]) / fort_s[m]
            print(f"  t={step*dtime/3600:5.1f}h: median={np.median(re)*100:6.3f}%, "
                  f"max={re.max()*100:6.2f}%, peak JAX={int(np.argmax(jax_mmr_s))} "
                  f"Fort={int(np.argmax(fort_s))}")

    print(f"\nFinal state:")
    print(f"  Max MMR: JAX={jax_mmr_final.max():.3e}, Fort={fort_final.max():.3e}")
    print(f"  At peak: JAX={np.argmax(jax_mmr_final)}, Fort={np.argmax(fort_final)}")

    mask = fort_final > 1e-30
    if mask.any():
        rel_err = np.abs(jax_mmr_final[mask] - fort_final[mask]) / fort_final[mask]
        print(f"  Rel error (active bins): max={rel_err.max():.2e}, "
              f"median={np.median(rel_err):.2e}")

    # Plot: profiles + error
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    for step_idx, step in enumerate([0, 100, 200, 400, 600]):
        if step < len(fortran["mmr"]):
            jax_mmr = snapshots.get(step, None)
            if jax_mmr is not None:
                jax_mmr = jax_mmr / np.array(rhoa_wet)
                fort_mmr = fortran["mmr"][step]
                color = f"C{step_idx}"
                hrs = step * dtime / 3600
                ax.plot(np.maximum(jax_mmr, 1e-50), zc_m / 1000, "-", color=color,
                        lw=2, label=f"JAX t={hrs:.0f}h")
                ax.plot(np.maximum(fort_mmr, 1e-50), zc_m / 1000, "--", color=color,
                        lw=1.5, alpha=0.8, label=f"Fort t={hrs:.0f}h")
    ax.set_xscale("log")
    ax.set_xlabel("MMR [g/g]")
    ax.set_ylabel("Altitude [km]")
    ax.set_title("Vertical Profile: JAX vs Fortran")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1e-25, 1e-9)
    ax.set_ylim(80, 104)

    ax = axes[1]
    mask = fort_final > 1e-30
    rel_err = np.abs(jax_mmr_final[mask] - fort_final[mask]) / fort_final[mask]
    z_active = (zc_m[mask]) / 1000
    ax.plot(rel_err * 100, z_active, "ro-", ms=3, lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("|JAX - Fortran| / Fortran  [%]")
    ax.set_ylabel("Altitude [km]")
    ax.set_title(f"Final Error at t={nstep*dtime/3600:.0f}h\n"
                 f"median={np.median(rel_err)*100:.3f}%, max={rel_err.max()*100:.2f}%")
    ax.grid(True, alpha=0.3)
    ax.axvline(1.0, color="gray", ls="--", alpha=0.5, label="1% target")
    ax.axvline(0.1, color="green", ls="--", alpha=0.5, label="0.1%")
    ax.legend(fontsize=9)
    ax.set_ylim(80, 104)

    fig.suptitle("Brownian Diffusion Validation (rmin=2e-8 cm, mesosphere)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "vdiftest_validation.png", dpi=150)
    plt.close(fig)
    print(f"\nPlot: {outdir / 'vdiftest_validation.png'}")


if __name__ == "__main__":
    main()
