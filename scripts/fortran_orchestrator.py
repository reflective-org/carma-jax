"""Fortran scenario-driven orchestrator (Phase 10.2).

Drives a scenario-enabled Fortran sulfatetest binary once per scenario
in the ensemble NPZ, parses its output, and returns a dict of arrays.

The orchestrator is **ready to drive** a Fortran binary that reads
a scenario file — but we deliberately don't commit the Fortran
modifications to ``../original-carma/CARMA/tests/carma_sulfatetest.F90``
from this PR. See ``docs/decisions/0025-fortran-orchestrator-design.md``
for the one-time Fortran-side changes needed.

Interface contract with the Fortran binary (spelled out in the ADR):

1. Reads scenario from ``SCENARIO_PATH`` (env var or first argv);
   file format is one-line whitespace-separated:
       ``T p rh h2so4_pptv aerosol_mu_nm aerosol_sigma_g``
2. Writes scenario output to ``OUTPUT_PATH`` (env var or second argv)
   as JSON with keys:
       ``T_final, gc_h2so4_final, pc_final (list), nstep_ran, status``

Until the Fortran side is updated, the orchestrator runs against a
Python-implemented *mock* binary shipped alongside (see
``scripts/_mock_fortran_sulfate.py``), which is sufficient for
unit-testing the orchestrator itself.
"""

import concurrent.futures as cf
import json
import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class ScenarioInput:
    """Flat record describing one scenario."""
    T: float
    p: float
    rh: float
    h2so4_pptv: float
    aerosol_mu_nm: float
    aerosol_sigma_g: float

    def to_text_line(self) -> str:
        """One-line whitespace-separated format the Fortran reads."""
        return (
            f"{self.T:.6e} {self.p:.6e} {self.rh:.6e} "
            f"{self.h2so4_pptv:.6e} {self.aerosol_mu_nm:.6e} "
            f"{self.aerosol_sigma_g:.6e}"
        )


def iter_scenarios(scenarios: dict) -> Iterable[ScenarioInput]:
    n = int(scenarios["_n"])
    for i in range(n):
        yield ScenarioInput(
            T=float(scenarios["T"][i]),
            p=float(scenarios["p"][i]),
            rh=float(scenarios["rh"][i]),
            h2so4_pptv=float(scenarios["h2so4_pptv"][i]),
            aerosol_mu_nm=float(scenarios["aerosol_mu_nm"][i]),
            aerosol_sigma_g=float(scenarios["aerosol_sigma_g"][i]),
        )


def run_one(binary: str, scen: ScenarioInput,
            work_dir: Path = None, timeout_s: float = 60.0) -> dict:
    """Invoke ``binary`` on one scenario, return parsed JSON output.

    The binary (real Fortran or mock) is called with two positional
    args: ``scenario_file`` and ``output_file``. Output is parsed as
    JSON. Raises on non-zero exit or malformed JSON.
    """
    work_dir = work_dir or Path(tempfile.mkdtemp(prefix="carma_scen_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    scen_path = work_dir / "scenario.txt"
    out_path = work_dir / "output.json"
    scen_path.write_text(scen.to_text_line() + "\n")

    cmd = shlex.split(binary) + [str(scen_path), str(out_path)]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout_s,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Fortran orchestrator failed (exit {result.returncode}):\n"
            f"  scenario: {scen}\n"
            f"  stderr: {result.stderr[:500]}"
        )
    if not out_path.exists():
        raise RuntimeError(
            f"Fortran orchestrator produced no output file at {out_path}")
    return json.loads(out_path.read_text())


def run_ensemble(binary: str, scenarios: dict,
                 work_root: Path = None,
                 max_workers: int = None,
                 timeout_s: float = 60.0) -> list:
    """Run all scenarios in parallel. Returns a list of output dicts
    (one per scenario, in order).

    Args:
        binary: shell-quoted path to the Fortran binary (or mock).
        scenarios: dict from generate_sulfate_scenarios.
        work_root: working directory base; created if missing.
        max_workers: process-pool size (defaults to os.cpu_count).
        timeout_s: per-scenario wall-time cap.
    """
    work_root = work_root or Path(tempfile.mkdtemp(prefix="carma_ens_"))
    work_root.mkdir(parents=True, exist_ok=True)

    n = int(scenarios["_n"])
    scen_list = list(iter_scenarios(scenarios))

    results = [None] * n
    with cf.ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(run_one, binary, scen,
                      work_dir=work_root / f"scen_{i:05d}",
                      timeout_s=timeout_s): i
            for i, scen in enumerate(scen_list)
        }
        for fut in cf.as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()

    return results


def aggregate_outputs(outputs: list) -> dict:
    """Stack per-scenario JSON outputs into numpy arrays.

    Each output JSON has keys ``T_final``, ``gc_h2so4_final``,
    ``pc_final`` (list), ``nstep_ran``, ``status``. Returns a dict
    of ``(n_scenarios, ...)`` arrays + status list.
    """
    n = len(outputs)
    T_final = np.array([o["T_final"] for o in outputs], dtype=np.float64)
    gc_final = np.array([o["gc_h2so4_final"] for o in outputs], dtype=np.float64)
    pc_list = [o["pc_final"] for o in outputs]
    # Assume all outputs have same nbin
    nbin = len(pc_list[0]) if pc_list else 0
    pc_final = np.zeros((n, nbin), dtype=np.float64)
    for i, pc in enumerate(pc_list):
        pc_final[i] = pc
    nstep = np.array([o["nstep_ran"] for o in outputs], dtype=np.int64)
    status = [o.get("status", "ok") for o in outputs]
    return dict(
        T_final=T_final,
        gc_h2so4_final=gc_final,
        pc_final=pc_final,
        nstep_ran=nstep,
        status=status,
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True,
                        help="Path or shell-quoted command to Fortran/mock binary")
    parser.add_argument("--scenarios", type=Path,
                        default=Path("data/sulfate_scenarios_1000.npz"))
    parser.add_argument("--out", type=Path,
                        default=Path("data/sulfate_fortran_outputs.npz"))
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--n", type=int, default=None,
                        help="Override scenario count (for smoke tests)")
    args = parser.parse_args()

    from generate_sulfate_scenarios import load_scenarios
    scenarios = load_scenarios(args.scenarios)
    if args.n is not None:
        scenarios = {k: (v[:args.n] if hasattr(v, "shape") and v.ndim > 0
                         else v)
                     for k, v in scenarios.items()}
        scenarios["_n"] = args.n
    n = int(scenarios["_n"])
    print(f"Running {n} scenarios through {args.binary}...")

    outputs = run_ensemble(
        args.binary, scenarios,
        max_workers=args.max_workers, timeout_s=args.timeout,
    )
    agg = aggregate_outputs(outputs)
    np.savez_compressed(args.out, **{
        k: v for k, v in agg.items() if k != "status"
    })
    print(f"Saved aggregated outputs to {args.out}")

    n_ok = sum(1 for s in agg["status"] if s == "ok")
    print(f"  status: {n_ok}/{n} ok, {n - n_ok} failed")


if __name__ == "__main__":
    main()
