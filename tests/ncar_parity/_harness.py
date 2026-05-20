"""Shared harness for NCAR-test parity comparisons.

The Fortran tests at ``../original-carma/CARMA/tests/`` write a fixed
plain-text format to ``tests/bench/<testname>.txt``. Layout:

  line 1: NGROUP NELEM NBIN NGAS                              (4 ints)
  next NGROUP·NBIN lines: igroup ibin r[µm] rmass[g]          (bin geom)
  one line containing "0" (a scalar, often zero)
  one line "0.0  0.0" (a pair, often time/relhum init)
  next NELEM·NBIN lines: ielem ibin mmr_init
  next NGAS lines: igas mmr_gas relhum_l relhum_i             (gas init)

  Then per step (nstep times):
      line: "    <time>"
      line: "    <T_or_other> <something>"      (test-specific)
      NELEM·NBIN lines: ielem ibin mmr
      NGAS  lines: igas mmr_gas relhum_l relhum_i

This module provides `parse_bench(path)` returning a `BenchData`
namedtuple with the parsed arrays plus a `compare(...)` helper for
the per-bin relative-error gates the parity tests apply.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class BenchData:
    """Parsed contents of a CARMA test bench file."""
    NGROUP: int
    NELEM: int
    NBIN: int
    NGAS: int
    r_um: np.ndarray            # (NGROUP, NBIN) - bin-center radius [µm]
    rmass_g: np.ndarray         # (NGROUP, NBIN) - bin-center mass [g]
    mmr_init: np.ndarray        # (NELEM, NBIN) - initial particle mmr
    mmr_gas_init: np.ndarray    # (NGAS,) - initial gas mmr
    times: np.ndarray           # (nstep,) - time stamps from each step block
    mmr_history: np.ndarray     # (nstep, NELEM, NBIN)
    mmr_gas_history: np.ndarray # (nstep, NGAS)

    @property
    def mmr_final(self) -> np.ndarray:
        return self.mmr_history[-1]

    @property
    def mmr_gas_final(self) -> np.ndarray:
        return self.mmr_gas_history[-1]


def _tokens(path: Path) -> Iterator[str]:
    """Yield whitespace-separated tokens from the file. The CARMA tests
    write Fortran free-format; lines are not significant for the data
    (only for line breaks)."""
    with open(path) as f:
        for line in f:
            for tok in line.split():
                yield tok


def parse_bench(path: str | Path,
                 pre_mmr_skip: int = 3,
                 per_step_skip: int = 3) -> BenchData:
    """Parse a CARMA test bench file.

    The exact byte layout differs from test to test in how many scalars
    the F90 source writes between sections. Override these per-test:

      ``pre_mmr_skip``  — tokens between bin geometry and initial mmr
                          (growtest=3, sulfatetest=4, nuctest=1).
      ``per_step_skip`` — tokens between the time and the mmr block
                          in each per-step record (growtest=3,
                          sulfatetest=3, nuctest=1).
    """
    path = Path(path)
    it = _tokens(path)

    NGROUP = int(next(it)); NELEM = int(next(it))
    NBIN   = int(next(it)); NGAS  = int(next(it))

    # Bin geometry
    r_um = np.zeros((NGROUP, NBIN))
    rmass_g = np.zeros((NGROUP, NBIN))
    for ig in range(NGROUP):
        for ib in range(NBIN):
            ig_read = int(next(it)); ib_read = int(next(it))
            assert ig_read == ig + 1 and ib_read == ib + 1, \
                f"bin geom mismatch at ({ig_read}, {ib_read})"
            r_um[ig, ib] = float(next(it))
            rmass_g[ig, ib] = float(next(it))

    # Test-specific scalars between bin geometry and initial mmr.
    for _ in range(pre_mmr_skip):
        next(it)

    # Initial particle mmr
    mmr_init = np.zeros((NELEM, NBIN))
    for ie in range(NELEM):
        for ib in range(NBIN):
            ie_read = int(next(it)); ib_read = int(next(it))
            mmr_init[ie, ib] = float(next(it))

    # Initial gas mmr (3 trailing numbers ignored — relhum / ice relhum)
    mmr_gas_init = np.zeros(NGAS)
    for ig in range(NGAS):
        ig_read = int(next(it))
        mmr_gas_init[ig] = float(next(it))
        _rh_l = float(next(it)); _rh_i = float(next(it))

    # Time-stepped block — repeat until EOF
    times_list: list[float] = []
    mmr_list: list[np.ndarray] = []
    gas_list: list[np.ndarray] = []
    while True:
        try:
            time_tok = next(it)
        except StopIteration:
            break
        times_list.append(float(time_tok))
        # Test-specific scalars before the mmr block.
        for _ in range(per_step_skip):
            next(it)
        # NELEM × NBIN mmr lines
        mmr_step = np.zeros((NELEM, NBIN))
        for ie in range(NELEM):
            for ib in range(NBIN):
                ie_read = int(next(it)); ib_read = int(next(it))
                mmr_step[ie, ib] = float(next(it))
        mmr_list.append(mmr_step)
        # NGAS gas lines (3 trailing numbers each, ignored)
        gas_step = np.zeros(NGAS)
        for ig in range(NGAS):
            ig_read = int(next(it))
            gas_step[ig] = float(next(it))
            _rh_l = float(next(it)); _rh_i = float(next(it))
        gas_list.append(gas_step)

    return BenchData(
        NGROUP=NGROUP, NELEM=NELEM, NBIN=NBIN, NGAS=NGAS,
        r_um=r_um, rmass_g=rmass_g,
        mmr_init=mmr_init, mmr_gas_init=mmr_gas_init,
        times=np.asarray(times_list),
        mmr_history=np.stack(mmr_list) if mmr_list else np.zeros((0, NELEM, NBIN)),
        mmr_gas_history=np.stack(gas_list) if gas_list else np.zeros((0, NGAS)),
    )


def compare(predicted: np.ndarray, reference: np.ndarray,
            rtol: float, atol: float, label: str = "") -> dict:
    """Per-bin relative-error gate. Returns a summary dict.

    Bins where `reference` is below `atol` are excluded from rel-err
    statistics (they're effectively zero and rel-err would blow up).
    """
    predicted = np.asarray(predicted); reference = np.asarray(reference)
    assert predicted.shape == reference.shape, \
        f"{label}: shape mismatch {predicted.shape} vs {reference.shape}"
    active = np.abs(reference) > atol
    if not active.any():
        return dict(label=label, n_active=0, max_rel=0.0, p95_rel=0.0,
                     median_rel=0.0, passed=True)
    denom = np.maximum(np.abs(reference), np.abs(predicted))
    rel_err = np.abs(predicted - reference) / np.where(denom > 0, denom, 1.0)
    rel_active = rel_err[active]
    return dict(
        label=label,
        n_active=int(active.sum()),
        n_total=int(active.size),
        max_rel=float(rel_active.max()),
        p95_rel=float(np.percentile(rel_active, 95)),
        median_rel=float(np.median(rel_active)),
        passed=bool(rel_active.max() <= rtol),
        rtol=rtol, atol=atol,
    )


# Convenience paths used by the per-test files.
NCAR_TESTS = Path(__file__).resolve().parents[2] / ".." / "original-carma" / "CARMA" / "tests"
NCAR_BENCH = NCAR_TESTS / "bench"


def bench_path(testname: str) -> Path:
    p = (NCAR_BENCH / f"{testname}.txt").resolve()
    if not p.exists():
        raise FileNotFoundError(
            f"NCAR bench file not found at {p}. "
            "Did you clone original-carma alongside the repo?"
        )
    return p
