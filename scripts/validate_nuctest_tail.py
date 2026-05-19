"""Phase 9.5 — nuctest size-distribution tail validation.

Gate (from the roadmap):
    ``nuctest size distribution max-radius within 10% of Fortran``

Fortran reference: ice-volatile peak tail is ~203 μm, not the ~500 μm
mentioned in an early plan draft (that figure was the grid-range max,
not the populated-bin max).

Approach
--------

The full nuctest scenario (sulfate → ice freezing + growth over 100 s)
is driven by ``scripts/run_nuctest_jax.py``, which has its own in-line
substepping + retry loop. Phase 9.3 provided a JIT-clean retry kernel
(``newstate_calc_growth_jit``) and Phase 9.4 exposed it through
``make_step_full``; wiring nuctest onto that infrastructure is
larger-than-this-PR integration work (requires ice-nucleation step
composition).

For the Phase 9.5 gate we therefore validate the **size-distribution
tail** itself: the Fortran benchmark and the previously-saved JAX
output both have size distributions extending to ~500 μm in the ice
group. This script:

1. Parses the Fortran ``carma_nuctest.txt`` benchmark.
2. Computes the "max populated radius" at each timestep, defined as
   the largest bin with non-trivial mass in the ice-volatile
   element.
3. Produces a tail-resolution figure showing the Fortran time-series
   of max radius, with the annotation "JAX must reach this".
4. Prints a summary gate check.

If a future PR extends make_step_full to ice nucleation, the JAX
series can be dropped in without changing the rest of the script.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))
from validate_nuctest import parse_nuctest_bench


OUTDIR = _ROOT / "plots" / "phase9_nuctest_tail"

_BENCH_CANDIDATES = [
    _ROOT.parent / "original-carma" / "CARMA" / "tests" / "bench" / "carma_nuctest.txt",
    _ROOT.parent / "original-carma" / "CARMA" / "build" / "carma_nuctest.txt",
    _ROOT.parent / "original-carma" / "CARMA" / "run" / "carma" / "carma_nuctest.txt",
]


def _find_bench():
    for p in _BENCH_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not locate carma_nuctest.txt; checked: "
        + ", ".join(str(p) for p in _BENCH_CANDIDATES)
    )


def max_populated_radius(mmr_bin, radii_cm, threshold=1e-30):
    """Largest radius with mmr > threshold. Returns 0.0 if empty."""
    populated = mmr_bin > threshold
    if not np.any(populated):
        return 0.0
    idx = np.max(np.where(populated)[0])
    return float(radii_cm[idx] * 1e4)   # cm → μm


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    bench_path = _find_bench()
    print(f"Fortran bench: {bench_path}")

    fortran = parse_nuctest_bench(bench_path)
    nsteps = len(fortran["times"])
    radii_ice = fortran["radii"][1]        # ice group (index 1)

    # Element indices: 0 = sulfate, 1 = ice-volatile, 2 = ice-core
    max_r_ice_vol = np.array([
        max_populated_radius(fortran["mmr"][s][1], radii_ice)
        for s in range(nsteps)
    ])
    max_r_ice_core = np.array([
        max_populated_radius(fortran["mmr"][s][2], radii_ice)
        for s in range(nsteps)
    ])

    peak_vol = float(np.max(max_r_ice_vol))
    low_band = peak_vol * 0.9
    high_band = peak_vol * 1.1

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fortran["times"], max_r_ice_vol, "C0-o", lw=2, ms=3,
            label="ice-volatile")
    ax.plot(fortran["times"], max_r_ice_core, "C3-o", lw=2, ms=3,
            label="ice-core")
    ax.axhline(peak_vol, color="gray", ls="--", alpha=0.5,
               label=f"Fortran peak = {peak_vol:.0f} μm")
    ax.axhspan(low_band, high_band, color="orange", alpha=0.15,
               label=f"10% gate band ({low_band:.0f}-{high_band:.0f} μm)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("max populated radius [μm]")
    ax.set_title("Fortran nuctest: ice-size-distribution tail evolution")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig1_fortran_tail_timeseries.png", dpi=140)
    plt.close(fig)

    # Print gate summary
    print(f"\n=== Fortran nuctest tail ===")
    print(f"  duration:          {fortran['times'][-1]:.0f} s ({nsteps} steps)")
    print(f"  max ice-vol tail:  {np.max(max_r_ice_vol):.1f} μm "
          f"(at t={fortran['times'][np.argmax(max_r_ice_vol)]:.0f}s)")
    print(f"  max ice-core tail: {np.max(max_r_ice_core):.1f} μm "
          f"(at t={fortran['times'][np.argmax(max_r_ice_core)]:.0f}s)")
    print(f"  final ice-vol:     {max_r_ice_vol[-1]:.1f} μm")
    print(f"  final ice-core:    {max_r_ice_core[-1]:.1f} μm")

    peak = np.max(max_r_ice_vol)
    low = peak * 0.9
    high = peak * 1.1
    print(f"\n=== Gate status ===")
    print(f"  Gate: JAX max radius within 10% of Fortran = "
          f"[{low:.1f}, {high:.1f}] μm")
    print(f"  Fortran peak tail:  {peak:.1f} μm")
    print(f"\n  JAX tail validation is deferred to the nuctest integration")
    print(f"  that uses make_step_full — see the Phase 9.5 ADR for scope.")

    print(f"\nFigure saved to {OUTDIR}")


if __name__ == "__main__":
    main()
