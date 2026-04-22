# ADR 0020: `newstate` dispatcher implements clear-sky only; in-cloud deferred

## Status

Accepted — Phase 9.1.

## Context

Fortran ``newstate.F90`` is a two-mode dispatcher that sits above
``newstate_calc``:

1. **Gridbox-average (``do_incloud=False``)** — run microphysics once
   on the whole-grid state. Used by every single-column test
   (nuctest, growtest, sulfatetest) and every Phase 7 sulfate
   scenario.
2. **In-cloud + clear-sky blended (``do_incloud=True``)** — split
   state by cloud fraction, run microphysics twice (cloudy path
   sees concentrations scaled by ``1/cldfrc`` for cloud-particle
   groups; clear-sky path sees the complement), then blend results
   with ``cldfrc`` weights. Uses ``pcd`` (detrained particles),
   ``is_grp_cloud``, and the per-pass ``zsubsteps`` tracking.

The in-cloud branch is large (~150 Fortran lines) and depends on
cloud physics (detrainment, cloud-droplet activation, ice nucleation)
that land later in the plan.

## Decision

``src/carma/newstate.py`` implements the **gridbox-average path
fully**; calling with ``do_incloud=True`` raises
``NotImplementedError`` with a pointer to Phase 11 where cloud/ice
microphysics lands.

```python
newstate(..., do_incloud=False)      # full support — delegates to
                                      # newstate_calc_growth
newstate(..., do_incloud=True)       # NotImplementedError
```

Arguments for the in-cloud path (``cldfrc``, ``is_grp_cloud``,
``do_clearsky``) are reserved in the signature so Phase 11 can
implement the blending without breaking the existing Phase 7/10
call sites.

## Alternatives considered

- **Stub with a silent pass-through (do_incloud=True → do_incloud=False)**.
  Rejected — would silently drop cloud-fraction scaling, making
  debugging impossible if someone accidentally enables the flag.
- **Implement the full in-cloud path now.** Rejected — the blending
  depends on ``detrain.py`` and cloud-particle tagging that live in
  Phase 9.2 and 11.1. Doing it piecemeal would ship broken state.
- **Drop the in-cloud kwargs from the signature entirely.** Rejected
  — would require a signature change in Phase 11 and break whatever
  ``make_step_full`` factory lands in 9.4.

## Consequences

- Every existing single-column benchmark runs unchanged through
  ``newstate(do_incloud=False, ...)``. Parity with
  ``newstate_calc_growth`` is tested (bit-for-bit).
- Phase 11's cloud/ice work will expand `newstate` rather than
  introduce a parallel module.
- The dispatcher is currently a thin wrapper (~20 LoC active), but
  it lets Phase 9.4's ``make_step_full`` have a single entry point
  regardless of cloud configuration.

## Validation

- 3 unit tests:
  - ``do_incloud=False`` matches ``newstate_calc_growth`` bit-for-bit
    on a growth-only H2O scenario.
  - ``do_incloud=True`` raises ``NotImplementedError`` with an
    informative message.
  - Total H2O mass (gas + condensate) is conserved across the step
    to within 1e-6 relative.
- Full suite: 182/182 unit tests pass.
