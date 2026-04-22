"""Unit tests for the Fortran scenario orchestrator (Phase 10.2).

Uses a Python mock binary (``scripts/_mock_fortran_sulfate.py``) in
place of the real Fortran binary. This keeps the tests fast and
decoupled from the external Fortran build; once the real binary is
available, the same tests can run against it by changing the
``_BINARY`` env override.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))
from fortran_orchestrator import (
    ScenarioInput, aggregate_outputs, iter_scenarios, run_ensemble, run_one,
)
from generate_sulfate_scenarios import generate_scenarios


_MOCK_BINARY = f"{sys.executable} {_ROOT / 'scripts' / '_mock_fortran_sulfate.py'}"


def _scen():
    return ScenarioInput(
        T=220.0, p=50.0, rh=0.5,
        h2so4_pptv=1.0, aerosol_mu_nm=50.0, aerosol_sigma_g=1.8,
    )


# --- single-scenario plumbing ---

def test_scenario_input_text_line_roundtrip():
    s = _scen()
    line = s.to_text_line()
    parts = [float(x) for x in line.split()]
    assert len(parts) == 6
    assert pytest.approx(parts[0]) == 220.0
    assert pytest.approx(parts[3]) == 1.0


def test_run_one_with_mock_binary(tmp_path):
    out = run_one(_MOCK_BINARY, _scen(), work_dir=tmp_path)
    assert out["status"] == "ok"
    assert out["nstep_ran"] == 100
    assert isinstance(out["pc_final"], list)
    assert len(out["pc_final"]) == 38


def test_run_one_fails_on_bad_binary(tmp_path):
    with pytest.raises((RuntimeError, FileNotFoundError)):
        run_one("/nonexistent_binary_xyz", _scen(), work_dir=tmp_path)


# --- iter_scenarios ---

def test_iter_scenarios_count():
    sc = generate_scenarios(5, seed=1)
    seen = list(iter_scenarios(sc))
    assert len(seen) == 5
    assert all(isinstance(s, ScenarioInput) for s in seen)


# --- end-to-end ensemble on mock ---

def test_run_ensemble_small(tmp_path):
    sc = generate_scenarios(4, seed=1)
    outputs = run_ensemble(
        _MOCK_BINARY, sc,
        work_root=tmp_path, max_workers=2,
    )
    assert len(outputs) == 4
    assert all(o["status"] == "ok" for o in outputs)
    # Different scenarios → different gc_h2so4_final
    gcs = {o["gc_h2so4_final"] for o in outputs}
    assert len(gcs) > 1


def test_aggregate_outputs_shapes():
    sc = generate_scenarios(4, seed=2)
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(_ROOT)
        outputs = run_ensemble(_MOCK_BINARY, sc, max_workers=2)
    agg = aggregate_outputs(outputs)
    assert agg["T_final"].shape == (4,)
    assert agg["gc_h2so4_final"].shape == (4,)
    assert agg["pc_final"].shape == (4, 38)
    assert agg["nstep_ran"].shape == (4,)
    assert len(agg["status"]) == 4


def test_ensemble_preserves_scenario_order():
    """Output index i must correspond to scenario i (not first-completed
    order from the process pool)."""
    sc = generate_scenarios(6, seed=3)
    outputs = run_ensemble(_MOCK_BINARY, sc, max_workers=3)

    # Re-run one scenario directly and compare to its index in the ensemble
    scen_list = list(iter_scenarios(sc))
    direct = run_one(_MOCK_BINARY, scen_list[2])
    assert pytest.approx(outputs[2]["gc_h2so4_final"]) == direct["gc_h2so4_final"]
    assert pytest.approx(outputs[2]["T_final"]) == direct["T_final"]
