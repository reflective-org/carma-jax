"""Microbenchmark for make_step_full (Phase 9.4).

Measures:

- **Trace + compile time** (cold first call).
- **Per-step wall time** after warm-up.
- **vmap batch scaling** over 10 / 100 / 1000 scenarios — this is
  the mode Phase 10's ensemble uses.

Output goes to ``plots/phase9_step_full/bench_step_full.png`` plus a
summary table on stdout. The numbers are CPU-only — run this on the
machine that will host the ensemble to get representative timings.
"""

import sys
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# Make the tests/ directory importable so we can reuse the fixture.
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
from tests.unit.test_step_full import _growth_env, _minimal_config
from carma.step_full import make_step_full


OUTDIR = Path(__file__).parent.parent / "plots" / "phase9_step_full"


def _time_call(fn, *args, n_warmup=3, n_iter=30, **kwargs):
    """Warm up then measure per-call wall time."""
    for _ in range(n_warmup):
        out = fn(*args, **kwargs)
        jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        out = fn(*args, **kwargs)
        jax.block_until_ready(out)
    dt = (time.perf_counter() - t0) / n_iter
    return dt


def bench_single_column():
    cfg = _minimal_config()
    step = make_step_full(cfg)
    env = _growth_env(cfg)

    # Cold call
    t0 = time.perf_counter()
    out = step(dtime=60.0, **env)
    jax.block_until_ready(out)
    trace_compile_s = time.perf_counter() - t0
    print(f"  cold (trace + compile): {trace_compile_s*1000:7.2f} ms")

    dt_warm = _time_call(step, dtime=60.0, **env, n_iter=100)
    print(f"  warm step:              {dt_warm*1000:7.3f} ms  ({1/dt_warm:.1f} step/s)")
    return trace_compile_s, dt_warm


def bench_vmap_batch(batch_sizes=(1, 10, 100, 1000)):
    """vmap-batch scaling on CPU. Replicates the pc/gc/t state
    across a batch axis while keeping env tables broadcast (same
    env for all scenarios — representative of the Phase 10
    ensemble where env is derived from common bin tables)."""
    cfg = _minimal_config()
    step = make_step_full(cfg)
    env = _growth_env(cfg)

    # Only pc, gc, t are varied per-scenario in a real ensemble.
    # All env arrays are broadcast (in_axes=None); pvapl is a scalar
    # and also broadcast.
    def call(pc, gc, t_arr):
        return step(
            pc=pc, gc=gc, t=t_arr, dtime=60.0,
            rhoa=env["rhoa"], zmet=env["zmet"],
            rlhe=env["rlhe"], rlhm=env["rlhm"], diffus=env["diffus"],
            akelvin=env["akelvin"], akelvini=env["akelvini"],
            gro=env["gro"], gro1=env["gro1"], gro2=env["gro2"],
            rup_wet=env["rup_wet"], rlow_wet=env["rlow_wet"],
            pratt=env["pratt"], prat=env["prat"],
            pden1=env["pden1"], palr=env["palr"],
            pvapl=env["pvapl"],
            ds_threshold_arr=env["ds_threshold_arr"],
        )

    vcall = jax.jit(jax.vmap(call, in_axes=(0, 0, 0)))

    results = {}
    for n in batch_sizes:
        pc_b = jnp.broadcast_to(env["pc"][None], (n,) + env["pc"].shape)
        gc_b = jnp.broadcast_to(env["gc"][None], (n,) + env["gc"].shape)
        t_b = jnp.broadcast_to(env["t"][None], (n,) + env["t"].shape)

        # Warm up
        for _ in range(3):
            out = vcall(pc_b, gc_b, t_b)
            jax.block_until_ready(out)

        t0 = time.perf_counter()
        n_iter = 10
        for _ in range(n_iter):
            out = vcall(pc_b, gc_b, t_b)
            jax.block_until_ready(out)
        dt = (time.perf_counter() - t0) / n_iter

        per_col = dt / n
        print(
            f"  vmap batch={n:4d}: {dt*1000:8.2f} ms total"
            f" / {per_col*1000:8.4f} ms per column")
        results[n] = (dt, per_col)
    return results


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Single-column step_full ===")
    trace_s, warm_s = bench_single_column()

    print("\n=== vmap batch scaling ===")
    batch = bench_vmap_batch()

    # Plot batch efficiency
    sizes = sorted(batch.keys())
    per_col = [batch[n][1] * 1e3 for n in sizes]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sizes, per_col, "C0-o", lw=2, ms=6)
    ax.axhline(warm_s * 1000, color="gray", ls="--",
               label=f"single-column (no vmap) = {warm_s*1000:.2f} ms")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("vmap batch size")
    ax.set_ylabel("per-column wall time [ms]")
    ax.set_title("make_step_full: vmap-batch scaling on CPU")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "bench_step_full.png", dpi=140)
    plt.close(fig)
    print(f"\nFigure saved to {OUTDIR / 'bench_step_full.png'}")


if __name__ == "__main__":
    main()
