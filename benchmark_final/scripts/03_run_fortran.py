"""Run the patched Fortran realistic-ensemble binary over all scenarios.

Drives ../original-carma/CARMA/build/test_sulfate_realistic, in parallel.
Writes aggregated outputs (T_final, gc_h2so4_final, pc_final, nstep_ran)
into benchmark_final/outputs/fortran_outputs.npz and prints total wall time.
"""
import argparse
import concurrent.futures as cf
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIN = (ROOT.parent.parent / "original-carma" / "CARMA" / "build"
                / "test_sulfate_realistic")
SCENARIO_PATH = ROOT / "scenarios" / "realistic_scenarios_100.npz"
DEFAULT_OUT = ROOT / "outputs" / "fortran_outputs.npz"


def scenario_line(scen, i):
    return (f"{float(scen['T'][i])!r} {float(scen['p'][i])!r} "
            f"{float(scen['rh'][i])!r} "
            f"{float(scen['h2so4_prod_rate'][i])!r} "
            f"{float(scen['M_total_ug_m3'][i])!r} "
            f"{float(scen['aerosol_mu_nm'][i])!r} "
            f"{float(scen['aerosol_sigma_g'][i])!r}\n")


def run_one(binary, scenario_line_text, work_dir, idx, timeout_s,
            do_coag=True, do_grow=True):
    scen_path = work_dir / f"scen_{idx:04d}.txt"
    out_path = work_dir / f"out_{idx:04d}.json"
    sched_path = Path(str(out_path) + ".schedule.bin")
    scen_path.write_text(scenario_line_text)
    cmd = [str(binary), str(scen_path), str(out_path),
           "1" if do_coag else "0",
           "1" if do_grow else "0"]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return idx, None, None, f"timeout (>{timeout_s}s)"
    if result.returncode != 0:
        return idx, None, None, result.stderr[:300]
    try:
        out_data = json.loads(out_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return idx, None, None, f"read: {e}"
    # Read substep schedule (int64[nstep] nsubsteps + int64[nstep] nretries)
    schedule = None
    if sched_path.exists():
        raw = np.fromfile(sched_path, dtype=np.int64)
        n_half = raw.size // 2
        schedule = (raw[:n_half], raw[n_half:])
    return idx, out_data, schedule, None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, default=DEFAULT_BIN)
    p.add_argument("--scenarios", type=Path, default=SCENARIO_PATH)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--timeout", type=int, default=1800,
                   help="per-scenario timeout in seconds (default 1800 = 30 min)")
    p.add_argument("--no-coag", action="store_true",
                   help="disable coagulation in Fortran")
    p.add_argument("--no-grow", action="store_true",
                   help="disable growth/condensation/nucleation in Fortran")
    args = p.parse_args()
    do_coag = not args.no_coag
    do_grow = not args.no_grow

    if not args.binary.exists():
        print(f"ERROR: binary not found at {args.binary}\n"
              f"Run benchmark_final/fortran/build_realistic.sh first.")
        return

    scens = np.load(args.scenarios)
    n = int(scens["_n"])
    print(f"Running Fortran realistic ensemble over {n} scenarios "
          f"({args.workers} workers)…")

    t_start = time.perf_counter()
    results = [None] * n
    errors = []
    with tempfile.TemporaryDirectory(prefix="bench_realistic_") as work_dir:
        work_dir = Path(work_dir)
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [
                ex.submit(run_one, args.binary, scenario_line(scens, i),
                          work_dir, i, args.timeout, do_coag, do_grow)
                for i in range(n)
            ]
            done = 0
            schedules = [None] * n
            for fut in cf.as_completed(futs):
                idx, out, schedule, err = fut.result()
                done += 1
                if out is None:
                    errors.append((idx, err))
                    print(f"  [{done:3d}/{n}] scen {idx} FAILED: {err}", flush=True)
                else:
                    results[idx] = out
                    schedules[idx] = schedule
                    if done % 10 == 0:
                        print(f"  [{done:3d}/{n}] done", flush=True)
    elapsed = time.perf_counter() - t_start

    print(f"\nFortran wall time: {elapsed:.1f} s  "
          f"({elapsed/n*1000:.0f} ms/scenario)")
    if errors:
        print(f"Errors in {len(errors)} scenarios:")
        for idx, err in errors[:5]:
            print(f"  scen {idx}: {err[:200]}")

    nbin = 38
    T_final = np.zeros(n)
    gc_final = np.zeros(n)
    pc_final = np.zeros((n, nbin))
    nstep_ran = np.zeros(n, dtype=np.int64)
    status = []
    for i, r in enumerate(results):
        if r is None:
            status.append("error")
            continue
        T_final[i] = r["T_final"]
        gc_final[i] = r["gc_h2so4_final"]
        pc_final[i] = r["pc_final"]
        nstep_ran[i] = r["nstep_ran"]
        status.append(r.get("status", "ok"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        T_final=T_final, gc_h2so4_final=gc_final, pc_final=pc_final,
        nstep_ran=nstep_ran,
        wall_time_s=np.array([elapsed]),
        n_scenarios=np.array([n]),
    )
    print(f"\nSaved: {args.out}")

    # Aggregate per-scenario substep schedules into one (N, nstep) array.
    valid_schedules = [s for s in schedules if s is not None]
    if valid_schedules:
        first_nstep = valid_schedules[0][0].shape[0]
        nsub_arr = np.zeros((n, first_nstep), dtype=np.int64)
        nret_arr = np.zeros((n, first_nstep), dtype=np.int64)
        for i, s in enumerate(schedules):
            if s is not None:
                nsub_arr[i], nret_arr[i] = s
        sched_out = args.out.with_name(args.out.stem + "_schedules.npz")
        np.savez_compressed(sched_out, nsubsteps_history=nsub_arr,
                             nretries_history=nret_arr)
        print(f"Saved substep schedules: {sched_out}")


if __name__ == "__main__":
    main()
