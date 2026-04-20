"""CPU benchmark: CARMA-JAX (fp64) vs Fortran reference.

Compares wall time of the JAX physics loop (after JIT compile) to
Fortran's end-to-end binary wall time. The Fortran binary's runtime
is total wall (parse config, step, write output); the JAX runtime
is the stepping loop only, matching ``compare_precision.py``'s
``loop_s`` measurement. The dominant Python / JAX overhead —
module import, JIT compilation — is paid once per simulation in
practice, so excluding it from the JAX side is the fair comparison
for steady-state workloads.

Writes:
    plots/cpu_benchmark/benchmark.png   bar chart comparison
    plots/cpu_benchmark/benchmark.json  raw timing dict

Usage:
    python scripts/benchmark_cpu.py
"""

import json
import statistics
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt


FORTRAN_BUILD = (Path(__file__).parent.parent.parent
                 / "original-carma" / "CARMA" / "build")
OUTDIR = Path(__file__).parent.parent / "plots" / "cpu_benchmark"

# Tests with Fortran executables under the CARMA build dir.
TESTS = [
    ("coagtest", "test_coag"),
    ("falltest", "test_fall"),
    ("growtest", "test_grow"),
    ("nuctest",  "test_nuc"),
]

NREP = 3


def time_fortran(exe_name: str) -> float:
    """Median wall time of N runs of a Fortran executable."""
    exe = FORTRAN_BUILD / exe_name
    if not exe.exists():
        raise FileNotFoundError(f"Fortran exe missing: {exe}")
    times = []
    for _ in range(NREP):
        t0 = time.perf_counter()
        subprocess.run(
            [str(exe)],
            cwd=FORTRAN_BUILD,   # so bench files land here
            check=True, capture_output=True,
        )
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def time_jax_loop():
    """Run compare_precision.py-style benchmarks for each test and
    return {test: median_loop_s}.

    JIT compile happens on the first call; we take median of N calls
    to amortize it. We import the runners directly to stay in one
    Python process.
    """
    from compare_precision import (
        run_coagtest, run_falltest, run_vdiftest, run_drydeptest, run_growtest,
    )
    # Map to Fortran-named tests (only the four that have Fortran exes).
    runners = {
        "coagtest": run_coagtest,
        "falltest": run_falltest,
        "growtest": run_growtest,
        # No run_nuctest equivalent yet — use falltest as placeholder,
        # or skip. We skip and handle below.
    }
    results = {}
    for name, fn in runners.items():
        # First call = JIT warm-up. Keep its time too but discard for median.
        times = []
        for rep in range(NREP):
            res = fn()
            times.append(res["loop_s"])
        results[name] = {
            "median_s": statistics.median(times[1:]) if len(times) > 1 else times[0],
            "first_call_s": times[0],
            "all_s": times,
        }
    return results


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("── timing Fortran binaries ──")
    fort_times = {}
    for name, exe in TESTS:
        print(f"  {name} ({exe}) ...", flush=True)
        try:
            t = time_fortran(exe)
            fort_times[name] = t
            print(f"     median {t*1000:7.1f} ms")
        except FileNotFoundError as e:
            print(f"     SKIP — {e}")

    print("\n── timing JAX physics loops (steady-state, post-JIT) ──")
    jax_times = time_jax_loop()
    for name, info in jax_times.items():
        first = info["first_call_s"] * 1000
        median = info["median_s"] * 1000
        print(f"  {name:<10}  first {first:7.1f} ms  median {median:7.1f} ms  "
              f"(incl. compile vs steady-state)")

    # Summary: steady-state ratio JAX/Fortran. nuctest has no JAX runner here.
    print("\n── steady-state ratio (JAX / Fortran) ──")
    ratios = {}
    for name in jax_times:
        if name not in fort_times:
            continue
        r = jax_times[name]["median_s"] / fort_times[name]
        ratios[name] = r
        print(f"  {name:<10}  {r:6.2f}×  "
              f"(JAX steady {jax_times[name]['median_s']*1000:.1f} ms "
              f"vs Fortran {fort_times[name]*1000:.1f} ms)")

    # Bar chart
    names = list(ratios.keys())
    fig, ax = plt.subplots(figsize=(7, 4))
    jax_ms = [jax_times[n]["median_s"] * 1000 for n in names]
    fort_ms = [fort_times[n] * 1000 for n in names]
    x = range(len(names))
    width = 0.35
    ax.bar([i - width/2 for i in x], fort_ms, width=width, label="Fortran", color="tab:blue")
    ax.bar([i + width/2 for i in x], jax_ms, width=width, label="JAX (steady-state)", color="tab:red")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylabel("wall time [ms]")
    ax.set_yscale("log")
    ax.set_title("CPU benchmark — median of 3 runs (excl. JIT compile for JAX)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    for i, n in enumerate(names):
        ax.text(i + width/2, jax_ms[i] * 1.1,
                f"{ratios[n]:.1f}×", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTDIR / "benchmark.png", dpi=130)
    plt.close(fig)

    with open(OUTDIR / "benchmark.json", "w") as f:
        json.dump({
            "fortran_s": fort_times,
            "jax_s": {k: v["median_s"] for k, v in jax_times.items()},
            "jax_first_call_s": {k: v["first_call_s"] for k, v in jax_times.items()},
            "ratio_jax_over_fortran": ratios,
            "nrep": NREP,
        }, f, indent=2)

    print(f"\nPlot: {OUTDIR / 'benchmark.png'}")
    print(f"Data: {OUTDIR / 'benchmark.json'}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    main()
