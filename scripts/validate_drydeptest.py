"""Validate JAX dry deposition against Fortran carma_drydeptest.

Fortran drydeptest:
- NZ=150, dz=100m, 0-15 km
- 2 groups: group 1 (drydep), group 2 (no drydep); 16 bins, rmin=1e-6, rmrat=4.32
- Ocean only: ocnfv=2.0 cm/s, ocnram=40.0 s/cm, ocnfrac=1.0
- Gaussian MMR at 8 km in every bin
- Output: bin 14 only (r ~ 5.7e-4 cm = 5.7 μm)
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
from carma.setup_vdry import setup_vdry
from carma.transport.vertical import vertical
from carma.enums import GridType, BoundaryCondition


def parse_drydeptest_bench(filepath):
    """Parse carma_drydeptest.txt.

    Format:
        line 1: NZ NELEM
        lines 2..NZ+1: i zc dz
        time blocks:
            time value
            NELEM*NZ lines: ielem i mmr num_density
    """
    with open(filepath) as f:
        lines = f.readlines()
    hdr = lines[0].split()
    nz, nelem = int(hdr[0]), int(hdr[1])
    idx = 1
    zc = np.zeros(nz)
    dz = np.zeros(nz)
    for i in range(nz):
        parts = lines[idx].split()
        zc[i] = float(parts[1])
        dz[i] = float(parts[2])
        idx += 1

    times = []
    mmr_blocks = []  # list of arrays (NELEM, NZ)
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        tokens = line.split()
        if len(tokens) == 1:
            try:
                t_val = float(tokens[0])
            except ValueError:
                idx += 1
                continue
            times.append(t_val)
            idx += 1
            mmr = np.zeros((nelem, nz))
            for _ in range(nelem * nz):
                parts = lines[idx].split()
                ie = int(parts[0]) - 1
                iz = int(parts[1]) - 1
                mmr[ie, iz] = float(parts[2])
                idx += 1
            mmr_blocks.append(mmr)
        else:
            idx += 1

    return {"nz": nz, "nelem": nelem, "zc": zc, "dz": dz,
            "times": np.array(times), "mmr": np.array(mmr_blocks)}


def main():
    bench_path = (Path(__file__).parent.parent.parent
                  / "original-carma" / "CARMA" / "tests" / "bench" / "carma_drydeptest.txt")
    fortran = parse_drydeptest_bench(bench_path)
    print(f"Fortran benchmark: NZ={fortran['nz']}, NELEM={fortran['nelem']}, "
          f"timesteps={len(fortran['times'])}")

    # --- Fortran test params ---
    NZ = 150
    NBIN = 16
    NGROUP = 2
    NELEM = 2
    OUTBIN = 13  # 0-based (Fortran OUTBIN=14)
    dtime = 1000.0
    nstep = 600
    deltaz_m = 100.0
    zmin_m = 0.0
    rmin_cm = 1e-6
    rmrat = 4.32
    rho_aer = 2.65

    # Surface parameters (ocean only)
    lndfv, lndram, lndfrac = 1.5, 60.0, 0.0
    ocnfv, ocnram, ocnfrac = 2.0, 40.0, 1.0
    icefv, iceram, icefrac = 2.5, 20.0, 0.0

    r, rmass, vol, dr, dm, rup, rlow, rmassup = setup_bins(rmin_cm, rmrat, NBIN, rho_aer)
    print(f"Bin {OUTBIN+1} radius: {float(r[OUTBIN]):.3e} cm")

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

    # Particle properties: spherical
    r_wet = jnp.broadcast_to(jnp.array(r, dtype=DTYPE)[None, :, None],
                              (NZ, NBIN, NGROUP))
    rhop_wet = jnp.full((NZ, NBIN, NGROUP), DTYPE(rho_aer), dtype=DTYPE)
    rrat = jnp.ones((NBIN, NGROUP), dtype=DTYPE)
    rprat = jnp.ones((NBIN, NGROUP), dtype=DTYPE)

    vf, re, bpm = setup_vf_jit(
        t_k, rhoa, zmet, rmu, r_wet, rhop_wet, rrat, rprat)

    dkz = jnp.zeros((NZ + 1, NBIN, NGROUP), dtype=DTYPE)  # no Brownian diffusion

    # Group 1 does drydep, group 2 doesn't
    grp_do_drydep = np.array([True, False])
    grp_do_vtran = np.array([True, True])

    vd = setup_vdry(
        vf, r_wet, bpm, t_k, rmu, rhoa, zmet, zmetl,
        lndfv, ocnfv, icefv, lndram, ocnram, iceram,
        lndfrac, ocnfrac, icefrac,
        igroup_arr=None, grp_do_drydep=grp_do_drydep, igridv=GridType.I_CART)
    print(f"vd range: group 1 (drydep) {float(vd[:,0].min()):.3e} - {float(vd[:,0].max()):.3e} cm/s")
    print(f"vd range: group 2 (no drydep, = vfall) {float(vd[:,1].min()):.3e} - {float(vd[:,1].max()):.3e} cm/s")
    print(f"vd[OUTBIN=14] drydep={float(vd[OUTBIN,0]):.4e} no_drydep={float(vd[OUTBIN,1]):.4e}")

    # --- Initial condition: Gaussian at 8 km, all bins and elements ---
    rho_dry_si = np.array(p_pa) / (287.0 * np.array(t_k))
    mmr_init_profile = 1e-10 * np.exp(-((zc_m - 8000.0) / 3000.0) ** 2) / rho_dry_si
    pc_init_profile = mmr_init_profile * np.array(rhoa_wet)  # g/cm^3

    pc = jnp.zeros((NZ, NBIN, NELEM), dtype=DTYPE)
    for ibin in range(NBIN):
        for ielem in range(NELEM):
            pc = pc.at[:, ibin, ielem].set(jnp.array(pc_init_profile, dtype=DTYPE))

    # Element → group mapping: element i → group i
    igroup_arr = np.array([0, 1], dtype=np.int32)

    pc_topbnd = jnp.zeros((NBIN, NELEM), dtype=DTYPE)
    pc_botbnd = jnp.zeros((NBIN, NELEM), dtype=DTYPE)
    ftoppart = jnp.zeros((NBIN, NELEM), dtype=DTYPE)
    fbotpart = jnp.zeros((NBIN, NELEM), dtype=DTYPE)

    print(f"\nInitial bin 14 max MMR [element 0, 1]: "
          f"{float((pc[:, OUTBIN, 0] / rhoa_wet).max()):.4e}, "
          f"{float((pc[:, OUTBIN, 1] / rhoa_wet).max()):.4e}")
    print(f"Fortran init bin 14 max MMR [elem 0, 1]: "
          f"{fortran['mmr'][0, 0, :].max():.4e}, {fortran['mmr'][0, 1, :].max():.4e}")
    print(f"Fortran at step 20 max MMR [elem 0, 1]: "
          f"{fortran['mmr'][20, 0, :].max():.4e}, {fortran['mmr'][20, 1, :].max():.4e}")

    print(f"\nRunning {nstep} steps at dt={dtime}s...")
    snapshot_steps = [0, 50, 100, 200, 400, 600]
    snapshots = {0: {0: np.array(pc[:, OUTBIN, 0]), 1: np.array(pc[:, OUTBIN, 1])}}

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
            snapshots[istep + 1] = {
                0: np.array(pc[:, OUTBIN, 0]),
                1: np.array(pc[:, OUTBIN, 1]),
            }
        if (istep + 1) % 100 == 0:
            tot_dd = float(jnp.sum(pc[:, OUTBIN, 0] * dz))
            tot_nd = float(jnp.sum(pc[:, OUTBIN, 1] * dz))
            print(f"  Step {istep+1:3d}: drydep total={tot_dd:.3e}, no_drydep total={tot_nd:.3e}")

    outdir = Path("plots/drydeptest_validation")
    outdir.mkdir(parents=True, exist_ok=True)

    rhoa_wet_np = np.array(rhoa_wet)

    # Final-state comparison per element (mmr bin 14)
    print(f"\nFinal state (bin {OUTBIN+1}):")
    for ie in range(NELEM):
        name = "drydep" if ie == 0 else "no_drydep"
        jax_pc = np.array(pc[:, OUTBIN, ie])
        jax_mmr = jax_pc / rhoa_wet_np
        fort_mmr = fortran["mmr"][-1, ie, :]
        mask = fort_mmr > 1e-30
        if mask.any():
            rel_err = np.abs(jax_mmr[mask] - fort_mmr[mask]) / fort_mmr[mask]
            print(f"  {name:10s}: max MMR JAX={jax_mmr.max():.3e} Fort={fort_mmr.max():.3e} "
                  f"median_err={np.median(rel_err)*100:.3f}% max_err={rel_err.max()*100:.2f}%")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ie, (ax, name) in enumerate(zip(axes[:2], ["drydep", "no_drydep"])):
        for step_idx, step in enumerate([0, 100, 200, 400, 600]):
            if step >= len(fortran["mmr"]):
                continue
            snap = snapshots.get(step)
            if snap is None:
                continue
            jax_mmr = snap[ie] / rhoa_wet_np
            fort_mmr = fortran["mmr"][step, ie, :]
            color = f"C{step_idx}"
            hrs = step * dtime / 3600
            ax.plot(np.maximum(jax_mmr, 1e-50), zc_m / 1000, "-", color=color,
                    lw=2, label=f"JAX t={hrs:.0f}h")
            ax.plot(np.maximum(fort_mmr, 1e-50), zc_m / 1000, "--", color=color,
                    lw=1.5, alpha=0.8, label=f"Fort t={hrs:.0f}h")
        ax.set_xscale("log")
        ax.set_xlabel("MMR [g/g]")
        ax.set_ylabel("Altitude [km]")
        ax.set_title(f"{name.upper()} (group {ie+1}, bin {OUTBIN+1})")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1e-20, 1e-9)
        ax.set_ylim(0, 15)

    # Final error panel
    ax = axes[2]
    for ie, name in enumerate(["drydep", "no_drydep"]):
        jax_mmr = np.array(pc[:, OUTBIN, ie]) / rhoa_wet_np
        fort_mmr = fortran["mmr"][-1, ie, :]
        mask = fort_mmr > 1e-30
        if mask.any():
            rel_err = np.abs(jax_mmr[mask] - fort_mmr[mask]) / fort_mmr[mask]
            z_active = zc_m[mask] / 1000
            ax.plot(rel_err * 100, z_active, "o-", ms=3, lw=1,
                    label=f"{name} (med={np.median(rel_err)*100:.2f}%)")
    ax.set_xscale("log")
    ax.set_xlabel("|JAX - Fortran| / Fortran  [%]")
    ax.set_ylabel("Altitude [km]")
    ax.set_title(f"Final Error at t={nstep*dtime/3600:.0f}h")
    ax.grid(True, alpha=0.3)
    ax.axvline(1.0, color="gray", ls="--", alpha=0.5, label="1%")
    ax.axvline(0.1, color="green", ls="--", alpha=0.5, label="0.1%")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 15)

    fig.suptitle("Dry Deposition Validation (Ocean, Zhang 2001)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "drydeptest_validation.png", dpi=150)
    plt.close(fig)
    print(f"\nPlot: {outdir / 'drydeptest_validation.png'}")


if __name__ == "__main__":
    main()
