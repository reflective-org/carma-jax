"""Run the Fortran diagnostic binary on all 1000 realistic scenarios.

For each scenario, runs ``test_sulfate_diagnostic`` for ``nstep_max``
timesteps (default 1) and writes the per-substep array dumps to
``data/diff/scen_<NNN>/``. Parallel via ProcessPoolExecutor.

Usage:
    python scripts/run_diagnostic_ensemble.py
    python scripts/run_diagnostic_ensemble.py --nstep-max 5
    python scripts/run_diagnostic_ensemble.py --workers 4 --n 100
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[1]
_BINARY = (_REPO_ROOT / ".." / "original-carma" / "CARMA" / "build"
           / "test_sulfate_diagnostic").resolve()
_SCENARIOS = _REPO_ROOT / "data" / "sulfate_scenarios_realistic_1000.npz"
_OUT_BASE = _REPO_ROOT / "data" / "diff"


def _run_one(args):
    idx, scenario_text, out_dir, nstep_max, timeout = args
    out_dir.mkdir(parents=True, exist_ok=True)
    # Wipe any stale files from a previous run for this scenario
    for f in out_dir.glob("substep_*.bin"):
        f.unlink()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(scenario_text)
        scen_path = f.name
    try:
        r = subprocess.run(
            [str(_BINARY), scen_path, str(out_dir), str(nstep_max)],
            capture_output=True, text=True, timeout=timeout,
        )
        if r.returncode != 0:
            return idx, False, r.stderr[:300]
        # Sanity: at least one of the per-substep files exists
        if not (out_dir / f"substep_{1:04d}_pc.bin").exists():
            return idx, False, "missing substep_0001_pc.bin"
        return idx, True, ""
    finally:
        os.unlink(scen_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=Path, default=_SCENARIOS,
                        help=f"NPZ with scenarios (default {_SCENARIOS})")
    parser.add_argument("--out-base", type=Path, default=_OUT_BASE,
                        help=f"Output base dir (default {_OUT_BASE})")
    parser.add_argument("--nstep-max", type=int, default=1,
                        help="Stop after this many substeps (default 1)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel worker processes")
    parser.add_argument("--n", type=int, default=None,
                        help="Run only first N scenarios (default: all)")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Per-scenario timeout in seconds")
    args = parser.parse_args()

    if not _BINARY.exists():
        print(f"ERROR: diagnostic binary missing at {_BINARY}", file=sys.stderr)
        print("Run scripts/fortran_patch/apply_diagnostic_patch.sh first.",
              file=sys.stderr)
        sys.exit(1)

    sc = np.load(args.scenarios)
    n_total = int(sc["T"].shape[0]) if args.n is None else min(args.n, sc["T"].shape[0])

    print(f"Running diagnostic dumps for {n_total} scenarios "
          f"(nstep_max={args.nstep_max}, workers={args.workers})...")

    tasks = []
    for i in range(n_total):
        text = (
            f"{float(sc['T'][i]):.6f}  {float(sc['p'][i]):.6f}  "
            f"{float(sc['rh'][i]):.6f}  {float(sc['h2so4_pptv'][i]):.6f}  "
            f"{float(sc['aerosol_mu_nm'][i]):.6f}  {float(sc['aerosol_sigma_g'][i]):.6f}\n"
        )
        out_dir = args.out_base / f"scen_{i:03d}"
        tasks.append((i, text, out_dir, args.nstep_max, args.timeout))

    t0 = time.perf_counter()
    failed = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_run_one, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            idx, ok, msg = fut.result()
            done += 1
            if not ok:
                failed.append((idx, msg))
            if done % 50 == 0 or done == n_total:
                wall = time.perf_counter() - t0
                rate = done / wall
                eta = (n_total - done) / rate if rate > 0 else 0
                print(f"  {done}/{n_total} ok={done - len(failed)} "
                      f"failed={len(failed)} ({wall:.1f}s, {rate:.1f}/s, "
                      f"ETA {eta:.0f}s)", flush=True)

    wall = time.perf_counter() - t0
    print(f"\nDone in {wall:.1f}s. {n_total - len(failed)}/{n_total} succeeded.")
    if failed:
        print(f"FAILURES (first 10):")
        for idx, msg in failed[:10]:
            print(f"  scen {idx}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
