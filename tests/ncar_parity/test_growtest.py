"""Parity scaffolding for the NCAR ``carma_growtest``.

The NCAR test (Bardeen) at
``../original-carma/CARMA/tests/carma_growtest.F90`` runs:

  - 1 ice-crystal group, 24 bins (rmin=1e-4 cm, rmrat=2, sphere)
  - 1 gas: H2O (Murphy 2005 vapour pressure)
  - Processes: growth (deposition onto ice) + thermo; NO nucleation,
    NO coagulation, NO sulfate
  - TTL-like setup: p=90 hPa, T=190 K, z=17 km
  - Initial: 0.1 #/cm³ in bin 0 (lognormal seed at the smallest bin),
             mmr_gas(H2O) = 3.5e-6 g/g (near ice saturation).
  - dtime = 100 s, nstep = 50 (5000 s total).

This is the cleanest Tier-1 test to add second because it shares zero
physics with the sulfate test path: pure deposition / evaporation
onto ice in mass space, no nucleation or coagulation. It is the right
case to validate the PPM mass-advection scheme (Phase 5) against.

Current scaffolding status
--------------------------
- ``test_growtest_bench_parse`` — runs *now*. It validates that the
  bench file parses with the correct token-skip counts and that the
  Fortran totals are internally consistent (water-vapour decrease ≈
  ice-mass increase). This is the parser regression gate.

- ``test_growtest_faithful_totals`` — placeholder, marked ``xfail``
  with a structural reason: ``src/carma/microfast_full.py`` is
  hardcoded for sulfate (line 133: ``is_ice_arr = (False,)``; line
  134: ``igrowgas_arr = (igas_h2so4,)``). Running ice + H2O-growth
  through the same step requires either a multi-species refactor of
  microfast_full (planned for Phase 5+) or a dedicated ice-only step
  driver. Until then the assertion would be impossible to construct.

The growtest bench file is checked in via the parent ``original-carma``
repo, so this test runs as long as the sibling repo is present.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from tests.ncar_parity._harness import bench_path, compare, parse_bench


# Exact token skips for growtest's bench format. ``per_step_skip = 2``
# (two reals between time and mmr block: ``t - t_orig`` and ``rlheat``)
# vs sulfatetest's 3. See parser docstring in ``_harness.py``.
GROWTEST_PARSE_KWARGS = dict(pre_mmr_skip=3, per_step_skip=2)


def _parse_growtest():
    return parse_bench(bench_path("carma_growtest"), **GROWTEST_PARSE_KWARGS)


def test_growtest_bench_parse():
    """Parser regression gate: confirm we read the bench correctly.

    Three independent checks:
      (a) Header dimensions match the F90 source constants.
      (b) Initial state has the expected seed in bin 0 and 3.5e-6 H2O.
      (c) Mass change between init and final is consistent within ~1 %:
          ``Δ(ice mass) ≈ -Δ(gas mmr)`` (H2O is the only gas).
    """
    bench = _parse_growtest()

    assert (bench.NGROUP, bench.NELEM, bench.NBIN, bench.NGAS) == (1, 1, 24, 1)
    assert bench.r_um[0, 0] == pytest.approx(1.0, rel=1e-3)
    assert bench.r_um[0, -1] == pytest.approx(203.0, rel=1e-2)
    assert bench.mmr_init[0, 0] == pytest.approx(2.36e-9, rel=1e-2)
    assert (bench.mmr_init[0, 1:] == 0.0).all()
    assert bench.mmr_gas_init[0] == pytest.approx(3.5e-6, rel=1e-3)
    assert len(bench.times) == 50
    assert bench.times[0] == pytest.approx(100.0)
    assert bench.times[-1] == pytest.approx(5000.0)

    # Mass conservation in the bench itself: ice mass gained ≈ gas mass lost.
    ice_init = bench.mmr_init[0].sum()
    ice_final = bench.mmr_final[0].sum()
    gas_init = bench.mmr_gas_init[0]
    gas_final = bench.mmr_gas_final[0]
    delta_ice = ice_final - ice_init
    delta_gas = gas_final - gas_init
    print(f"\n  Δ ice = {delta_ice:+.3e}  Δ gas = {delta_gas:+.3e}  "
          f"(should sum to ~0)")
    # 2 % tolerance — ice and gas are different storages, modest
    # numerical drift across 50 steps is allowed.
    assert abs(delta_ice + delta_gas) / max(abs(delta_ice), 1e-30) < 0.02


@pytest.mark.slow
@pytest.mark.xfail(
    reason="Running carma_growtest through the faithful JAX step requires "
            "lifting src/carma/microfast_full.py's hardcoded sulfate "
            "assumptions (is_ice_arr=(False,), igrowgas_arr=(igas_h2so4,), "
            "wtpct_tabaz etc.). Blocked on Phase 5 / multi-species refactor. "
            "Scaffolding kept so the moment that lands, flipping this test "
            "to a real parity gate is trivial.",
    strict=False,
)
def test_growtest_faithful_totals():
    """Placeholder for the real parity test (Phase 5+).

    When unblocked, this should:
      1. Build an ice-only ``CarmaConfig`` mirroring carma_growtest.F90
         (NBIN=24, rmin=1e-4 cm, rmrat=2, sphere, is_ice=True,
         RHO_I=0.93, 1 gas = H2O, ivaprtn=2).
      2. Run 50 × 100 s through the faithful step.
      3. Compare total ice mmr and final gas mmr to bench within ~5 %.
      4. Compare per-bin distribution under a looser ~30 % tolerance.
    """
    bench = _parse_growtest()
    # The test below cannot execute today; mark via raise so the
    # xfail is recorded as expected-fail rather than passing silently.
    raise NotImplementedError(
        "Awaiting multi-species refactor of microfast_full — see xfail reason."
    )
