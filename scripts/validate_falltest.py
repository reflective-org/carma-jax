"""Validate JAX vertical transport against Fortran carma_falltest.

Fortran falltest: dust sedimentation with constant vf=2 cm/s
- NZ=110, dz=100m, 0-11 km
- 8 bins, 1 element, 1 group, dust rho=2.65 g/cm^3
- Initial: Gaussian at 8 km, only bin 1 has particles
- dt=1000s, 600 steps (100 hours), implicit solver
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
from carma.transport.vertical import vertical
from carma.enums import GridType, BoundaryCondition


def parse_falltest_bench(filepath):
    """Parse carma_falltest.txt.

    Format:
        line 1: NZ
        lines 2..NZ+1: i, zc(m), dz(m)
        then time blocks:
            time value
            NZ lines: i, mmr(g/g), num_density(?)
    """
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
    # --- Parse Fortran benchmark ---
    bench_path = (Path(__file__).parent.parent.parent
                  / "original-carma" / "CARMA" / "tests" / "bench" / "carma_falltest.txt")
    fortran = parse_falltest_bench(bench_path)
    print(f"Fortran benchmark: NZ={fortran['nz']}, timesteps={len(fortran['times'])}")

    # --- JAX setup matching Fortran ---
    NZ = 110
    NBIN = 8
    NELEM = 1
    NGROUP = 1
    dtime = 1000.0
    nstep = 600
    deltaz_m = 100.0
    zmin_m = 0.0
    vf_const = 2.0  # cm/s
    rmin_cm = 7.5e-4
    rmrat = 2.0
    rho_aer = 2.65

    # Setup bins (matching Fortran)
    r, rmass, vol, dr, dm, rup, rlow, rmassup = setup_bins(rmin_cm, rmrat, NBIN, rho_aer)

    # Setup atmosphere (US Standard Atmosphere at layer centers)
    zc_m = np.arange(NZ) * deltaz_m + zmin_m + deltaz_m / 2  # centers
    zl_m = np.arange(NZ + 1) * deltaz_m + zmin_m  # edges

    # Get T, p from standard atmosphere
    p_pa, t_k = get_standard_atmosphere(jnp.array(zc_m, dtype=DTYPE))
    pl_pa, _ = get_standard_atmosphere(jnp.array(zl_m, dtype=DTYPE))

    # Convert to CGS
    p_cgs = p_pa * RPA2CGS  # dyne/cm^2
    pl_cgs = pl_pa * RPA2CGS
    zc = jnp.array(zc_m) * RM2CGS  # cm
    zl = jnp.array(zl_m) * RM2CGS

    rhoa, dz, zmet, zmetl, rmu, thcond, rhoa_wet = setup_atm(
        t_k, p_cgs, pl_cgs, zc, zl, GridType.I_CART)
    print(f"Atm: p range = {float(p_cgs[0]):.3e} - {float(p_cgs[-1]):.3e} dyne/cm^2")
    print(f"Atm: T range = {float(t_k[0]):.1f} - {float(t_k[-1]):.1f} K")

    # rhoa is rho_cgs * zmet; for Cartesian zmet=1, so rhoa = rho_cgs
    rhoa_cgs = rhoa / zmet  # g/cm^3

    # --- Initial condition: Gaussian in bin 0 only ---
    # Fortran: mmr(i,1,1) = 1e-10 * exp(-((zc - 8000)/3000)^2) / (p/287/T)
    # rho_dry_SI = p/(287 T) in SI (kg/m^3)
    # We need to convert mmr to pc (number density × zmet)
    # mmr is kg/kg. To get number concentration: mmr * rho_air / mass_per_particle
    # But Fortran stores pc = mmr * rhoa_wet internally (see CARMASTATE_SetBin)

    # For bin 1 only, the initial Gaussian in kg/kg:
    rho_dry_si = np.array(p_pa) / (287.0 * np.array(t_k))  # kg/m^3
    mmr_init = 1e-10 * np.exp(-((zc_m - 8000.0) / 3000.0) ** 2) / rho_dry_si  # kg/kg

    # Convert to pc = mmr * rhoa_wet [g/cm^3] (that's what Fortran stores)
    # rhoa_wet is in g/cm^3 for cartesian
    pc_init_bin0 = mmr_init * np.array(rhoa_wet)  # g/cm^3 (stored units)

    # pc shape: (NZ, NBIN, NELEM). Only bin 0 has particles.
    pc = jnp.zeros((NZ, NBIN, NELEM), dtype=DTYPE)
    pc = pc.at[:, 0, 0].set(jnp.array(pc_init_bin0, dtype=DTYPE))

    print(f"\nInitial bin 1 total: {float(jnp.sum(pc[:, 0, 0] * dz)):.4e}")
    print(f"Fortran initial bin 1: {fortran['mmr'][0].max():.4e}")

    # --- Fall velocity: constant 2 cm/s downward (at all layer edges) ---
    vf = jnp.full((NZ + 1, NBIN, NGROUP), DTYPE(vf_const), dtype=DTYPE)
    # No diffusion for falltest (set to zero)
    dkz = jnp.zeros((NZ + 1, NBIN, NGROUP), dtype=DTYPE)
    # No dry dep
    vd = jnp.zeros((NBIN, NGROUP), dtype=DTYPE)
    # Boundary fluxes zero
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
    outdir = Path("plots/falltest_validation")
    outdir.mkdir(parents=True, exist_ok=True)

    # Final state comparison
    jax_final = np.array(pc[:, 0, 0])
    fort_final = fortran["mmr"][-1]  # this is mmr (kg/kg)
    # Convert JAX pc (g/cm³) to mmr (g/g): pc / rhoa_cgs
    jax_mmr_final = jax_final / np.array(rhoa_cgs)

    print(f"\nFinal state:")
    print(f"  JAX total:    {float(jnp.sum(pc[:, 0, 0] * dz)):.4e} g/cm²")
    print(f"  Max MMR: JAX={jax_mmr_final.max():.3e}, Fort={fort_final.max():.3e}")
    print(f"  At peak: JAX={np.argmax(jax_mmr_final)}, Fort={np.argmax(fort_final)}")

    # Rel error (where Fortran > 1e-30)
    mask = fort_final > 1e-30
    if mask.any():
        rel_err = np.abs(jax_mmr_final[mask] - fort_final[mask]) / fort_final[mask]
        print(f"  Rel error (active bins): max={rel_err.max():.2e}, "
              f"median={np.median(rel_err):.2e}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    for step_idx, step in enumerate([0, 100, 200, 400, 600]):
        if step < len(fortran["mmr"]):
            jax_mmr = snapshots.get(step, None)
            if jax_mmr is not None:
                jax_mmr = jax_mmr / np.array(rhoa_cgs)
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
    ax.set_xlim(1e-20, 1e-9)
    ax.set_ylim(0, 11)

    ax = axes[1]
    # Relative error at each step
    mask = fort_final > 1e-30
    rel_err = np.abs(jax_mmr_final[mask] - fort_final[mask]) / fort_final[mask]
    z_active = (zc_m[mask]) / 1000
    ax.plot(rel_err * 100, z_active, "ro-", ms=4, lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("|JAX - Fortran| / Fortran  [%]")
    ax.set_ylabel("Altitude [km]")
    ax.set_title(f"Final Error at t={nstep*dtime/3600:.0f}h\n"
                 f"median={np.median(rel_err)*100:.3f}%, max={rel_err.max()*100:.2f}%")
    ax.grid(True, alpha=0.3)
    ax.axvline(1.0, color="gray", ls="--", alpha=0.5, label="1% target")
    ax.axvline(0.1, color="green", ls="--", alpha=0.5, label="0.1%")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 11)

    fig.suptitle("Vertical Transport Validation: Constant vf=2 cm/s Sedimentation", fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "falltest_validation.png", dpi=150)
    plt.close(fig)
    print(f"\nPlot: {outdir / 'falltest_validation.png'}")


if __name__ == "__main__":
    main()
