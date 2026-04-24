# ADR 0026: Phase 10 parity reveals a physics-configuration gap

## Status

Accepted as finding — Phase 10.4.

## Context

Phase 10.4 ran JAX (`make_step_full`) and Fortran
(`carma_sulfatetest_ensemble`) over the same 1000-scenario
ensemble (Phase 10.1 NPZ). Both finished 1000/1000 ok. The
parity comparison (`scripts/compare_ensembles.py`) gave:

| output | JAX–Fortran median rel err | ≤1% pass rate |
|---|---|---|
| `T_final` | 1.04e-7 | **100 %** |
| `gc_h2so4_final` (mmr) | 1.00 | 0 % |
| `pc_final` (mmr per bin, median over bins) | 1.00 | 2.4 % |

Gate: ≥95% ≤1%; actual: 2.4%. **Fails.**

## Diagnosis

Per-scenario inspection (scenario 0: T=272 K, 0.02 pptv H₂SO₄)
shows:

- Fortran `pc_final[37]` (top bin) = **1.12e-2 g/g** — all the
  particle mass has grown into the top bin over 50 h of simulated
  time.
- JAX `pc_final[37]` = **3.9e-11 g/g** — 9 orders of magnitude
  smaller. Particles are still essentially at their initial
  lognormal-seeded sizes.

Root cause: the JAX ensemble uses the growth environment from the
step_full **test fixture** (`_growth_env`), which sets
`gro = 1e-30` (nearly zero by design — the unit tests verify
adaptive-retry logic without driving real growth). The Fortran
binary uses `setup_gkern`'s computed growth kernel with the
scenario's T, p, RH.

So the two runs are solving different equations: Fortran is doing
full condensational growth + CARMA's coagulation + thermo
coupling; JAX is running an essentially-static initial distribution
through `newstate_calc_growth_jit` with negligible growth rates
(plus sulfate homogeneous nucleation + gas exchange from
`sulfate_step_one_level`).

## Decision

Commit the ensemble outputs + parity script as-is. Phase 10.4's
deliverable is the comparison infrastructure and the honest
finding; passing the gate is a follow-on task blocked on two real
pieces of work:

1. **Wire `setup_gkern` into `jax_ensemble.py`** so the growth
   kernel reflects each scenario's (T, p, RH) instead of the test
   fixture's dead values. This needs the scenario → env mapping
   that the test fixture currently stubs.
2. **Add coagulation to `make_step_full`** (or explicitly disable
   `do_coag=.true.` in the Fortran driver for parity). Fortran's
   default has coag enabled; JAX's `make_step_full` doesn't include
   it.

Either direction is legitimate:

- **Option A (expand JAX).** Add growth kernel wiring + coagulation
  composition into `make_step_full`. Closes the physics gap but is
  significant new work. The Phase 9.4 ADR (0022) already flagged
  the transport+coag composition as deferred; this would unblock it.
- **Option B (restrict Fortran).** Modify
  `carma_sulfatetest_ensemble.F90` to disable `do_grow` and
  `do_coag` (only leave nucleation + gas exchange) so the two
  sides solve the same reduced-physics problem. Cheapest; matches
  what `make_step_full` actually covers today.

Recommended next step is Option B for this PR's gate (so the
10.4 comparison actually validates what we've built), then
Option A in a separate phase (to validate the full Phase 9.4 /
Phase 11 composition).

## Update after attempting Option B

Tried flipping ``do_grow=.false., do_coag=.false.`` in
``carma_sulfatetest_ensemble.F90``. Fortran segfaults. Reading
``microfast.F90``:

```fortran
  if (do_grow) then          ! line 57
    ...
    call sulfnuc(carma, cstate, iz, rc)   ! line 79 — NUCLEATION is here
    ...
    call growevapl(...)
    ...
  endif
```

**Fortran CARMA couples sulfate nucleation to the growth branch** —
``sulfnuc`` only runs if ``do_grow=.true.``. Option B as
conceived ("Fortran does nucleation + gas exchange only") is not
achievable by flipping flags. Would require surgery in
``microfast.F90`` to expose a ``do_nucleate_only`` path.

That makes Option B no cheaper than Option A. The revised
recommendation is:

- **Option A-lite (proposed next)**: wire ``setup_gkern`` into the
  JAX ensemble so growth kernels reflect each scenario's T/p/RH.
  Leaves coagulation composed-out of JAX; Fortran still has coag
  enabled. Quantify the coag-only residual separately.
- **Option A-full (later)**: also add ``make_step_coag`` into
  ``make_step_full`` for bit-parity on coag. Larger phase-11-ish
  scope.

The current state of this PR — both ensembles ran, parity
comparison infrastructure works, gap clearly documented — is
already a useful deliverable even before we close the gate.

## Update — Option A-lite landed

Wired ``setup_atm + setup_grow + setup_gkern`` into
``scripts/jax_ensemble.py`` (per-scenario growth kernels using
the real Fortran-equivalent code path). Re-ran the JAX ensemble
(92 s, 1000/1000 ok) and the parity comparison.

| metric | before A-lite | after A-lite |
|---|---|---|
| `pc_final` median-bin ≤1% pass | 2.4 % | **65.8 %** |
| `gc_h2so4_final` ≤1% pass | 0 % | 0 % |

The `pc_final` improvement is dramatic. Median error
``p50 = 6e-25`` — half the ensemble agrees with Fortran to
near-machine precision now. The distribution is bimodal:
~67% pass with ~zero error, ~33% fail at ~100% error.

Diagnostic on the 326 failing scenarios:

| param | passing median | failing median |
|---|---|---|
| T [K] | 228 | 270 |
| RH | 0.57 | 0.46 |
| H₂SO₄ [pptv] | 1.37 | 0.57 |

Failing scenarios are **warm, dry, low-H₂SO₄** — exactly the
regime where nucleation is slow and coagulation drives most of
the size-distribution evolution. Confirms the residual gap is
coagulation: Fortran has it (`do_coag=.true., I_COLLEC_FUCHS`),
JAX's `make_step_full` does not.

`gc_h2so4` parity didn't improve because gas depletion is
dominated by Fortran's coag-fueled accumulation into the top
bin (the bin pulls down gas as a sink). Adding coag to JAX
should close this.

## Next step — Option A-full

Add `make_step_coag` composition into `make_step_full`. The
infrastructure is all in place (Phase 5b shipped
`make_step_coag`); just need to thread it through and verify
parity. Estimated 1–2 days of work.

The Phase 10.4 gate (≥95% ≤1%) will likely pass once coag is
added, given that two-thirds of scenarios already pass with
near-machine-precision agreement.

## Consequences

- Phase 10.4 gate technically fails on the current `make_step_full`
  because the Fortran reference exercises physics JAX hasn't
  composed yet. This is a composition gap, not an accuracy bug —
  the individual components have all been validated in isolation
  (Phases 1–8).
- The comparison script, NPZs, and figures are useful regardless:
  they document the gap precisely and make fixing it a
  verification exercise.
- Fortran-side data (`data/sulfate_fortran_outputs.npz`) is
  committed so that whichever direction we take, we don't need
  to re-run the ~3-minute Fortran ensemble.

## Validation

- Fortran ensemble: 1000/1000 ok, 3.2 min wall time, 245 KB NPZ.
- JAX ensemble: 1000/1000 ok, 17 s wall time, 311 KB NPZ.
- Parity script runs end-to-end and emits CDFs + worst-case table.
- Gate fails as expected given the physics gap; ADR captures the
  cause.
