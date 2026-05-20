"""Smoke tests for the NCAR-parity harness.

Verifies that `parse_bench` can read several bench files in the NCAR
test suite and that the parsed shapes / step counts match what the
F90 source declares.
"""
import pytest

from tests.ncar_parity._harness import parse_bench, bench_path, compare


# Tests whose bench file follows the "standard" 4-int header layout
# (NGROUP NELEM NBIN NGAS). Other tests (coagtest, swelltest, ...) use
# custom layouts; they're covered case-by-case in dedicated test files.
#
# pre_mmr_skip / per_step_skip vary per test because the F90 source
# writes different scalars between sections (free vs formatted output).
@pytest.mark.parametrize(
    "testname,expected_ngroup,expected_nelem,expected_nbin,expected_ngas,"
    "pre_mmr,per_step",
    [
        # growtest: pre=write(0)+write(0,0)=3 tokens; per_step=time then (t-t0, rlh)=2 tokens
        ("carma_growtest",     1, 1, 24, 1, 3, 2),
        # sulfatetest: pre=write(0)+write(0,0,0.0)=4 tokens; per_step=(nsub_diff, nret_diff, dt)=3
        ("carma_sulfatetest",  1, 1, 38, 2, 4, 3),
        # nuctest: pre=write(0)=1; per_step=just time, then directly mmr
        ("carma_nuctest",      2, 3, 16, 1, 1, 0),
    ],
)
def test_parse_bench_basic_shapes(testname, expected_ngroup, expected_nelem,
                                    expected_nbin, expected_ngas,
                                    pre_mmr, per_step):
    """The parser yields the dimensions declared by the F90 source."""
    b = parse_bench(bench_path(testname),
                     pre_mmr_skip=pre_mmr, per_step_skip=per_step)
    assert b.NGROUP == expected_ngroup, (
        f"{testname}: expected NGROUP={expected_ngroup}, got {b.NGROUP}"
    )
    assert b.NELEM == expected_nelem, (
        f"{testname}: expected NELEM={expected_nelem}, got {b.NELEM}"
    )
    assert b.NBIN == expected_nbin, (
        f"{testname}: expected NBIN={expected_nbin}, got {b.NBIN}"
    )
    assert b.NGAS == expected_ngas, (
        f"{testname}: expected NGAS={expected_ngas}, got {b.NGAS}"
    )


def test_parse_growtest_internals():
    """Specific shape and value checks on the growtest bench."""
    b = parse_bench(bench_path("carma_growtest"),
                     pre_mmr_skip=3, per_step_skip=2)
    assert b.r_um.shape == (1, 24)
    assert b.rmass_g.shape == (1, 24)
    assert b.mmr_init.shape == (1, 24)
    assert b.mmr_gas_init.shape == (1,)

    # rmrat=2 in mass → radius ratio is 2^(1/3) ≈ 1.26
    ratio = b.r_um[0, 1] / b.r_um[0, 0]
    assert abs(ratio - 2 ** (1 / 3)) / ratio < 0.02

    # Mass-bin doubling. Bench has only 3-digit exponential precision so
    # use a loose 1% tolerance.
    assert abs(b.rmass_g[0, 1] / b.rmass_g[0, 0] - 2.0) < 1e-2

    # Initial water vapor at 3.5e-6 g/g (per F90 source)
    assert abs(b.mmr_gas_init[0] - 3.5e-6) / 3.5e-6 < 0.01

    # Only bin 0 has initial particles
    assert b.mmr_init[0, 0] > 0
    assert (b.mmr_init[0, 1:] == 0).all()

    # 50 steps of dtime=100 s → final time = 5000 s
    assert len(b.times) == 50
    assert abs(b.times[-1] - 5000.0) < 1e-3


def test_compare_pass_and_fail():
    """The relative-error gate behaves sensibly."""
    import numpy as np
    ref = np.array([1.0, 2.0, 3.0, 0.0])

    # Exact match — passes any rtol
    r = compare(ref.copy(), ref, rtol=1e-12, atol=1e-30)
    assert r["passed"], r
    assert r["max_rel"] == 0.0

    # 1% perturbation on each non-zero bin
    pred = ref * 1.01
    r = compare(pred, ref, rtol=2e-2, atol=1e-30)
    assert r["passed"], r
    assert abs(r["max_rel"] - 0.01 / 1.01) < 1e-6  # max(|a-b|)/max(|a|,|b|)

    # Fails when over tolerance
    r = compare(pred, ref, rtol=1e-3, atol=1e-30)
    assert not r["passed"], r

    # Bins below atol contribute nothing
    pred_with_zero = np.array([1.0, 2.0, 3.0, 1e-50])
    r = compare(pred_with_zero, ref, rtol=1e-12, atol=1e-30)
    assert r["n_active"] == 3   # zero-ref bin excluded
