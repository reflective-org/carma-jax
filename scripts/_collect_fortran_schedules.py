"""Run the patched Fortran ensemble binary across all scenarios and
aggregate the per-step substep schedule (Phase 10.6 prerequisite).

For each scenario the patched binary writes
    output.json                  — same JSON the orchestrator already parses
    output.json.schedule.bin     — 2 × nstep int64: (nsubsteps, nretries)

We collect both into a single NPZ with shape (n_scenarios, nstep).
"""
import argparse
import concurrent.futures as cf
import json
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from fortran_orchestrator import iter_scenarios


def _run_one(args):
    binary, scen, work_dir, timeout_s, nstep = args
    work_dir.mkdir(parents=True, exist_ok=True)
    scen_path = work_dir / "scenario.txt"
    out_path = work_dir / "output.json"
    sched_path = out_path.with_suffix(".json.schedule.bin")
    scen_path.write_text(scen.to_text_line() + "\n")

    cmd = shlex.split(binary) + [str(scen_path), str(out_path)]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout_s,
    )
    if result.returncode != 0 or not out_path.exists():
        return {"status": f"err: rc={result.returncode}",
                "stderr": result.stderr[:300]}

    out = json.loads(out_path.read_text())
    if not sched_path.exists():
        return {"status": "err: no schedule.bin"}

    sched = np.fromfile(sched_path, dtype=np.int64)
    if sched.size != 2 * nstep:
        return {"status": f"err: schedule size {sched.size} != {2*nstep}"}
    out["nsubsteps_per_step"] = sched[:nstep].tolist()
    out["nretries_per_step"]  = sched[nstep:].tolist()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", required=True)
    p.add_argument("--scenarios", type=Path,
                   default=REPO / "data/sulfate_scenarios_realistic_1000.npz")
    p.add_argument("--out-outputs", type=Path,
                   default=REPO / "data/sulfate_fortran_realistic_outputs.npz")
    p.add_argument("--out-schedules", type=Path,
                   default=REPO / "data/sulfate_fortran_substep_schedules.npz")
    p.add_argument("--nstep", type=int, default=100)
    p.add_argument("--n", type=int, default=None,
                   help="optional cap for smoke test")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--timeout", type=float, default=600.0)
    args = p.parse_args()

    scens = dict(np.load(args.scenarios))
    if args.n is not None:
        for k in list(scens):
            v = scens[k]
            if hasattr(v, "shape") and v.ndim > 0:
                scens[k] = v[:args.n]
        scens["_n"] = args.n

    n = int(scens["_n"])
    print(f"Running Fortran ensemble + schedule capture on {n} scenarios "
          f"(nstep={args.nstep}, workers={args.workers})...", flush=True)

    with tempfile.TemporaryDirectory(prefix="phase106_") as work_root:
        work_root = Path(work_root)
        scen_list = list(iter_scenarios(scens))
        futs = {}
        with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, scen in enumerate(scen_list):
                futs[ex.submit(_run_one,
                               (args.binary, scen,
                                work_root / f"s{i:05d}", args.timeout,
                                args.nstep))] = i
            outputs = [None] * n
            t0 = time.perf_counter()
            done = 0
            for fut in cf.as_completed(futs):
                i = futs[fut]
                outputs[i] = fut.result()
                done += 1
                if done % 50 == 0 or done == n:
                    print(f"  {done}/{n}  ({(time.perf_counter()-t0):.1f}s)",
                          flush=True)

    n_ok = sum(1 for o in outputs if "status" not in o or o.get("status") == "ok")
    print(f"  ok: {n_ok}/{n}")

    # Aggregate parity-style outputs (mirrors fortran_orchestrator.aggregate_outputs)
    T_final = np.array([o["T_final"] for o in outputs], dtype=np.float64)
    gc = np.array([o["gc_h2so4_final"] for o in outputs], dtype=np.float64)
    pc = np.array([o["pc_final"] for o in outputs], dtype=np.float64)
    nstep_ran = np.array([o["nstep_ran"] for o in outputs], dtype=np.int64)
    np.savez_compressed(args.out_outputs,
        T_final=T_final, gc_h2so4_final=gc, pc_final=pc, nstep_ran=nstep_ran)
    print(f"  saved outputs npz to {args.out_outputs}")

    # Aggregate substep schedules. Fortran's cstate%f_nsubstep is
    # accumulated across ALL Step calls for the cstate's lifetime
    # (carmastate_mod.F90:540-543 only resets it on FIRST Create), so
    # the values we got are cumulative. Per-outer-step ntsubsteps is
    # the diff with 0 prepended.
    sched_cum = np.array([o["nsubsteps_per_step"] for o in outputs], dtype=np.int64)
    rets_cum  = np.array([o["nretries_per_step"]  for o in outputs], dtype=np.int64)
    nsubsteps = np.diff(sched_cum, axis=1, prepend=0)
    nretries  = np.diff(rets_cum,  axis=1, prepend=0)
    np.savez_compressed(args.out_schedules,
        nsubsteps=nsubsteps, nretries=nretries,
        nsubsteps_cumulative=sched_cum, nretries_cumulative=rets_cum)
    print(f"  saved schedules npz to {args.out_schedules}  shape={nsubsteps.shape}")
    print(f"  per-step nsubsteps: p50={np.percentile(nsubsteps, 50):.0f}  "
          f"p95={np.percentile(nsubsteps, 95):.0f}  max={nsubsteps.max()}")


if __name__ == "__main__":
    main()
