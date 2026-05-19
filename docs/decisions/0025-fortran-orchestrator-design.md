# ADR 0025: Fortran scenario orchestrator — JSON-over-files, deferred Fortran patch

## Status

Accepted — Phase 10.2.

## Context

The plan for Phase 10.2 reads:

> Modify ``carma_sulfatetest.F90`` to read (T, p, RH, H₂SO₄, ...)
> from a text file; rebuild once; Python orchestrator runs the
> resulting binary 1000 times, collects outputs.

Two parts: the Fortran modification, and the Python orchestrator
that drives it.

Modifying the Fortran source is a real intervention in a shared
repo (``../original-carma``). The orchestrator plumbing, testing,
and parsing logic can be built and tested against a Python mock
binary without touching the Fortran.

## Decision

Split the work:

1. **Orchestrator (this PR).** ``scripts/fortran_orchestrator.py``
   drives any binary that honors the Phase 10.2 interface
   contract — read one scenario from ``argv[1]``, write one JSON
   output to ``argv[2]``. Parallelised via
   ``concurrent.futures.ProcessPoolExecutor``; preserves scenario
   index order in the output list.

2. **Mock binary (this PR).** ``scripts/_mock_fortran_sulfate.py``
   implements the interface with trivial physics (exponential
   H₂SO₄ decay). Lets every orchestrator test run in <1 s without
   a Fortran build.

3. **Fortran patch (full, ready to apply).**
   ``scripts/fortran_patch/carma_sulfatetest_ensemble.F90`` is a
   complete working Fortran source mirroring
   ``carma_sulfatetest.F90`` with the hard-coded constants
   replaced by scenario-file reads. ``apply_patch.sh`` copies it
   into ``../original-carma/CARMA/tests/``, appends one
   ``create_standard_test`` entry to the CMakeLists, and rebuilds
   to produce ``SULFATE_ENSEMBLE.exe``.

   The patch is kept in our repo (not committed to
   ``../original-carma``) so upstream pulls don't clobber it;
   re-running ``apply_patch.sh`` is idempotent.

## Interface contract

Invocation:

```
$ binary <scenario_file> <output_file>
```

Scenario file (read by Fortran):

```
T_K  p_hPa  rh_fraction  h2so4_pptv  aerosol_mu_nm  aerosol_sigma_g
```

Output file (written by Fortran) — JSON:

```json
{
  "T_final": 220.0,
  "gc_h2so4_final": 1.2345e-13,
  "pc_final": [..., 38 values ...],
  "nstep_ran": 100,
  "status": "ok"
}
```

## Alternatives considered

- **Stdin/stdout pipes.** Rejected — harder to debug, and parallel
  Fortran processes sharing stdin is fragile.
- **Binary format (unformatted Fortran record).** Rejected — JSON
  is human-readable, easy to grep, trivially parseable by Python.
- **Commit Fortran patch directly.** Rejected — the user flagged
  shared-repo edits as something to coordinate on. The template
  captures the exact diff needed and lets the user apply it when
  convenient without blocking the orchestrator PR.

## Consequences

- 7 unit tests pass via the Python mock (no Fortran required).
- Once the Fortran patch lands, swap ``_MOCK_BINARY`` for the real
  binary path and re-run the tests for end-to-end Fortran parity.
- The orchestrator supports arbitrary ``max_workers`` — a 1000-
  scenario ensemble will take ~scenario_time × 1000 /
  max_workers seconds wall-clock. On 8 cores and 1-s Fortran calls
  that's ~2 minutes; full 30-s Fortran runs ~1 hour.
- Crash-handling: per-scenario ``subprocess.run`` is isolated;
  one failed scenario doesn't abort the ensemble. Per-scenario
  status tracked in the output JSON.

## Validation

- 7 unit tests covering scenario serialisation, single-scenario
  run, bad-binary handling, count from iter_scenarios, end-to-end
  ensemble on mock, aggregate-shape sanity, scenario-order
  preservation.
- Full suite: 206/206 unit tests pass.
