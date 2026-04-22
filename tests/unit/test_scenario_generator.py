"""Unit tests for the Phase 10.1 sulfate-scenario generator."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Make scripts/ importable
_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))
from generate_sulfate_scenarios import (
    _BOUNDS, _LOG_AXES, generate_scenarios, load_scenarios, save_scenarios,
)


def test_count_matches_request():
    s = generate_scenarios(50, seed=1)
    for k in _BOUNDS:
        assert s[k].shape == (50,)


def test_all_values_within_bounds():
    s = generate_scenarios(200, seed=2)
    for k, (lo, hi) in _BOUNDS.items():
        arr = s[k]
        assert (arr >= lo).all() and (arr <= hi).all(), (
            f"{k}: out of bounds — min={arr.min()}, max={arr.max()}, "
            f"expected [{lo}, {hi}]")


def test_log_axes_use_log_uniform():
    """For log-uniform axes, the log-spaced histogram is roughly flat."""
    s = generate_scenarios(2000, seed=3)
    for k in _LOG_AXES:
        log_arr = np.log10(s[k])
        # bin into 5 buckets; expect roughly 400 per bucket within ±25%
        hist, _ = np.histogram(log_arr, bins=5)
        target = len(s[k]) / 5
        assert (hist > 0.6 * target).all() and (hist < 1.4 * target).all(), (
            f"{k}: not log-uniform; histogram = {hist}")


def test_linear_axes_use_uniform():
    s = generate_scenarios(2000, seed=4)
    for k in _BOUNDS:
        if k in _LOG_AXES:
            continue
        hist, _ = np.histogram(s[k], bins=5)
        target = len(s[k]) / 5
        assert (hist > 0.6 * target).all() and (hist < 1.4 * target).all(), (
            f"{k}: not uniform; histogram = {hist}")


def test_seed_is_deterministic():
    s1 = generate_scenarios(100, seed=42)
    s2 = generate_scenarios(100, seed=42)
    for k in _BOUNDS:
        assert np.array_equal(s1[k], s2[k])


def test_different_seeds_give_different_samples():
    s1 = generate_scenarios(100, seed=1)
    s2 = generate_scenarios(100, seed=2)
    # At least one parameter must differ
    diffs = [not np.array_equal(s1[k], s2[k]) for k in _BOUNDS]
    assert any(diffs)


def test_save_load_roundtrip(tmp_path):
    s = generate_scenarios(75, seed=9)
    out = tmp_path / "scen.npz"
    save_scenarios(s, out)
    s_loaded = load_scenarios(out)
    for k in _BOUNDS:
        assert np.array_equal(s[k], s_loaded[k])
    assert int(s_loaded["_n"]) == 75
    assert int(s_loaded["_seed"]) == 9


def test_metadata_round_trip():
    s = generate_scenarios(50, seed=7)
    assert s["_seed"] == 7
    assert s["_n"] == 50


def test_latin_hypercube_no_duplicate_rows():
    """Sanity: LHS gives unique scenarios."""
    s = generate_scenarios(500, seed=10)
    arr = np.stack([s[k] for k in _BOUNDS], axis=1)   # (500, 6)
    unique = np.unique(arr, axis=0)
    assert unique.shape[0] == arr.shape[0]
