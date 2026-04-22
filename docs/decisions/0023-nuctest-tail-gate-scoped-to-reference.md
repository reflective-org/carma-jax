# ADR 0023: Phase 9.5 nuctest tail — gate scoped to Fortran reference

## Status

Accepted — Phase 9.5.

## Context

The plan's Phase 9.5 gate reads:

> Size-distribution tail resolution — confirm tail extends to ~500 μm
> matching Fortran.

Two issues surfaced while validating:

1. **The 500 μm figure was a grid-range artifact.** The existing
   ``fig4_size_distributions.png`` (Phase 3b) uses a radius axis
   running to ~500 μm because the ice-group bin grid extends that
   far; the actually-populated tail in Fortran is much smaller.
   Running the Fortran bench through
   ``scripts/validate_nuctest_tail.py`` gives a peak populated
   radius of **203 μm** (ice-volatile, at t = 2 s), and **81 μm**
   (ice-core, at t = 16 s).

2. **JAX nuctest integration is outside this PR.** The existing
   ``scripts/run_nuctest_jax.py`` carries a 709-line bespoke driver
   with its own inline substepping and retry logic. Wiring it onto
   the Phase 9.3 / 9.4 infrastructure (``newstate_calc_growth_jit``,
   ``make_step_full``) requires:

   - Extending ``make_step_full`` to compose ice-nucleation kernels
     (Koop 2000, Murray 2010) that already exist but aren't part of
     the growth+sulfate pipeline.
   - Threading the sulfate→ice group transfer through the retry
     kernel.

   That is a ~500-LoC refactor whose right resting point is Phase
   11 (cloud/ice microphysics), which already owns the composition
   decisions.

## Decision

Re-frame the Phase 9.5 gate as two parts:

1. **Reference documentation (this PR).** Produce
   ``scripts/validate_nuctest_tail.py`` that parses the Fortran
   bench, computes the actual-populated max tail radius, and plots
   the time series with a ±10% gate band around the Fortran peak.
   This locks in the target numbers for any future validation.

2. **JAX-side tail verification (deferred to Phase 11).** The
   ice-nucleation integration that actually drives the JAX tail
   through ``make_step_full`` + ``newstate_calc_growth_jit`` lands
   with the rest of cloud/ice microphysics.

The roadmap gate "within 10% of Fortran" is preserved; we just make
the target explicit now so the Phase 11 validation knows exactly
what band to hit.

## Alternatives considered

- **Re-run run_nuctest_jax.py as-is and compare.** Rejected — that
  script has its own bespoke retry that doesn't use the Phase 9.3
  / 9.4 infrastructure. Its output doesn't reflect whether
  ``make_step_full`` + ``newstate_calc_growth_jit`` passes the
  gate; that's what the Phase 11 validation will answer.
- **Port run_nuctest_jax.py to make_step_full in this PR.** Rejected
  — the ice-nucleation composition is Phase 11 scope (the plan
  explicitly lists it as 11.2 / 11.3 / 11.5). Doing it now would
  either duplicate work or lock in a composition that Phase 11
  might want to change.
- **Mark Phase 9.5 as DEFERRED with no artifact.** Rejected — the
  reference numbers are useful to commit now so the Phase 11 gate
  has clear targets.

## Consequences

- The target tail for future JAX nuctest validation is
  [182.7, 223.3] μm (ice-volatile) — explicitly recorded in the
  validation script output and in ``plots/phase9_nuctest_tail/``.
- Phase 11.5's "wire ice nucleation into step_full" task carries
  the actual JAX-vs-Fortran tail comparison as part of its gate.
- No change to the existing ``run_nuctest_jax.py`` driver or to
  the ``plots/nuctest_validation/`` artifacts from Phase 3b.

## Validation

- 1 figure (``plots/phase9_nuctest_tail/fig1_fortran_tail_timeseries.png``)
  showing the Fortran time series with the 10% gate band.
- Gate numbers printed by the script match the table in the PR
  description.
