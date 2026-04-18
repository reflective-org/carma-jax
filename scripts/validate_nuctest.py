"""Validate JAX nucleation against Fortran nuctest benchmark.

Nuctest: sulfate aerosol freezing into ice crystals at TTL conditions.
2 groups, 3 elements, 16 bins, T=205K, p=90hPa, dt=1s, 100 steps.
Koop 2000 + Murray 2010 freezing + ice growth.
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
from carma.setup_atm import setup_atm
from carma.setup_grow import setup_grow
from carma.setup_gkern import setup_gkern
from carma.setup_vf import setup_vf
from carma.vapor_pressure import vaporp_h2o_murphy2005
from carma.supersaturation import supersat
from carma.growth.growevapl import growevapl
from carma.nucleation.freezaerl_koop2000 import freezaerl_koop2000
from carma.nucleation.freezglaerl_murray2010 import freezglaerl_murray2010
from carma.microfast import microfast_step
from carma.enums import GridType, ElementType


def parse_nuctest_bench(filepath):
    """Parse carma_nuctest.txt benchmark."""
    with open(filepath) as f:
        lines = f.readlines()
    idx = 0
    parts = lines[idx].split()
    ngroup, nelem, nbin, ngas = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
    idx += 1

    # Bin structure per group
    radii = {}
    rmass_f = {}
    for ig in range(ngroup):
        radii[ig] = np.zeros(nbin)
        rmass_f[ig] = np.zeros(nbin)
        for i in range(nbin):
            parts = lines[idx].split()
            radii[ig][i] = float(parts[2]) * 1e-4  # um to cm
            rmass_f[ig][i] = float(parts[3])
            idx += 1

    # Parse timestep blocks
    times = []
    mmr_data = []  # list of (nelem, nbin) arrays
    gas_data = []

    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue

        # Time value
        try:
            time_val = float(line)
        except ValueError:
            idx += 1
            continue
        times.append(time_val)
        idx += 1

        # Per-element, per-bin mmr
        mmr = np.zeros((nelem, nbin))
        for ie in range(nelem):
            for ib in range(nbin):
                parts = lines[idx].split()
                mmr[ie, ib] = float(parts[2])
                idx += 1
        mmr_data.append(mmr)

        # Gas state: "igas mmr_gas satliq satice"
        parts = lines[idx].split()
        gas_data.append(float(parts[1]))
        idx += 1

    return {
        "ngroup": ngroup, "nelem": nelem, "nbin": nbin, "ngas": ngas,
        "radii": radii, "rmass": rmass_f,
        "times": np.array(times),
        "mmr": mmr_data,  # list of (nelem, nbin) arrays
        "gas": np.array(gas_data),
    }


def main():
    # Parse Fortran benchmark
    bench_path = Path(__file__).parent.parent.parent / "original-carma" / "CARMA" / "build" / "carma_nuctest.txt"
    if not bench_path.exists():
        bench_path = Path(__file__).parent.parent.parent / "original-carma" / "CARMA" / "tests" / "bench" / "carma_nuctest.txt"

    print("Parsing Fortran benchmark...", flush=True)
    fortran = parse_nuctest_bench(bench_path)
    print(f"  {fortran['ngroup']} groups, {fortran['nelem']} elements, "
          f"{fortran['nbin']} bins, {len(fortran['times'])} timesteps", flush=True)

    # Print Fortran initial and final state
    print(f"\nFortran initial (t=0):")
    for ie in range(fortran['nelem']):
        nonzero = np.sum(fortran['mmr'][0][ie] > 0)
        total = np.sum(fortran['mmr'][0][ie])
        print(f"  Element {ie+1}: {nonzero} non-zero bins, total mmr = {total:.4e}")
    print(f"  Gas mmr = {fortran['gas'][0]:.4e}")

    print(f"\nFortran final (t={fortran['times'][-1]:.0f}s):")
    for ie in range(fortran['nelem']):
        nonzero = np.sum(fortran['mmr'][-1][ie] > 0)
        total = np.sum(fortran['mmr'][-1][ie])
        peak_bin = np.argmax(fortran['mmr'][-1][ie])
        print(f"  Element {ie+1}: {nonzero} non-zero bins, total mmr = {total:.4e}, peak bin = {peak_bin+1}")
    print(f"  Gas mmr = {fortran['gas'][-1]:.4e}")

    # Plot Fortran results
    outdir = Path(__file__).parent.parent / "plots" / "nuctest_validation"
    outdir.mkdir(parents=True, exist_ok=True)

    nsteps = len(fortran['times'])
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Element 1 (sulfate) evolution
    ax = axes[0, 0]
    r_um = fortran['radii'][0] * 1e4
    for si, label in [(0, 't=0'), (10, 't=10s'), (50, 't=50s'), (-1, f't={fortran["times"][-1]:.0f}s')]:
        if si < nsteps:
            ax.semilogy(r_um, fortran['mmr'][si][0], 'o-', ms=3, label=label)
    ax.set_xlabel("Radius [um]")
    ax.set_ylabel("MMR [g/g]")
    ax.set_title("Element 1: Sulfate Aerosol")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=1e-20)
    ax.grid(True, alpha=0.3)

    # 2. Element 2 (ice volatile) evolution
    ax = axes[0, 1]
    r_um2 = fortran['radii'][1] * 1e4
    for si, label in [(0, 't=0'), (10, 't=10s'), (50, 't=50s'), (-1, f't={fortran["times"][-1]:.0f}s')]:
        if si < nsteps:
            mmr_vals = fortran['mmr'][si][1]
            mmr_vals = np.where(mmr_vals > 0, mmr_vals, 1e-30)
            ax.semilogy(r_um2, mmr_vals, 'o-', ms=3, label=label)
    ax.set_xlabel("Radius [um]")
    ax.set_ylabel("MMR [g/g]")
    ax.set_title("Element 2: Ice Crystal (volatile)")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=1e-20)
    ax.grid(True, alpha=0.3)

    # 3. Element 3 (ice core) evolution
    ax = axes[1, 0]
    for si, label in [(0, 't=0'), (10, 't=10s'), (50, 't=50s'), (-1, f't={fortran["times"][-1]:.0f}s')]:
        if si < nsteps:
            mmr_vals = fortran['mmr'][si][2]
            mmr_vals = np.where(mmr_vals > 0, mmr_vals, 1e-30)
            ax.semilogy(r_um2, mmr_vals, 'o-', ms=3, label=label)
    ax.set_xlabel("Radius [um]")
    ax.set_ylabel("MMR [g/g]")
    ax.set_title("Element 3: Ice Core (sulfate mass)")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=1e-20)
    ax.grid(True, alpha=0.3)

    # 4. Gas + total particle mass
    ax = axes[1, 1]
    ax.plot(fortran['times'], fortran['gas'] * 1e6, 'b-', lw=2, label="H2O gas [ppm]")
    total_particle = [sum(fortran['mmr'][i].sum() for _ in [0]) for i in range(nsteps)]
    # Actually sum all element mmr
    total_particle = [fortran['mmr'][i].sum() for i in range(nsteps)]
    ax.plot(fortran['times'], np.array(total_particle) * 1e6, 'r-', lw=2, label="Total particle [ppm]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Mixing ratio [ppm]")
    ax.set_title("Gas and Particle Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Fortran Nuctest: Sulfate Freezing → Ice Growth", fontsize=14)
    fig.tight_layout()
    fig.savefig(outdir / "nuctest_fortran.png", dpi=150)
    plt.close(fig)

    print(f"\nFortran plots saved to: {outdir.resolve()}")
    print("\n(JAX comparison will be added once the full microfast driver is validated)")


if __name__ == "__main__":
    main()
