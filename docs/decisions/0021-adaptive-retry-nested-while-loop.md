# ADR 0021: Adaptive retry uses nested `lax.while_loop` for JIT cleanness

## Status

Accepted — Phase 9.3.

## Context

`newstate_calc.newstate_calc_growth` (Phase 5e) implements
adaptive substepping with a Python ``while True`` retry loop wrapping
a Python ``for isubstep in range(ntsubsteps)``. Both loops use
traced state (pc, gc, t, rc) for the control flow, which means:

- The outer ``while`` can't be compiled — it inspects ``rc`` at the
  Python level.
- The inner ``for`` is a fixed Python range, but the retry doubles
  ``ntsubsteps``, so the graph grows on every retry attempt.

This prevents `make_step_full` (Phase 9.4) from being a single JIT'd
closure. The goal of 9.3 is to fix that.

## Decision

Port the retry logic into nested `jax.lax.while_loop`:

- **Outer loop** (retry): state `(pc, gc, t, rlh_total, nts, rc,
  nretries)`. Body runs one inner loop with `nts` substeps from the
  saved state; if it returns `rc == RC_WARNING_RETRY`, doubles
  `nts` (clipped to `maxsubsteps`) and increments `nretries`. Exits
  when `rc != RC_WARNING_RETRY` or `nretries >= maxretries + 1`.
- **Inner loop** (substep): state `(pc, gc, t, rlh, i, rc)`. Body
  calls `microfast_growth` and increments `i`. Exits on
  `i == nts` or `rc == RC_WARNING_RETRY` (early break).

The sentinel initial `rc = RC_WARNING_RETRY` forces the outer loop
to run at least once. Successful first attempt → inner returns
`rc = RC_OK` → outer cond fails → exit with `nretries = 0`.

Return value is `(pc, gc, t, rlh_total, nts_used)` matching the
Python-while variant.

## Alternatives considered

- **Mask-based fori_loop up to maxsubsteps.** Rejected — wastes
  compute when `ntsubsteps < maxsubsteps`; also doesn't handle the
  outer retry.
- **Single `lax.while_loop` combining both levels.** Rejected —
  would need a complex flat state and more conditionals per
  iteration; the nested form mirrors Fortran's structure and is
  easier to review.
- **Keep the Python-while version.** Rejected — `make_step_full`
  needs end-to-end JIT; can't have a Python loop in the middle.

## Implementation notes

- `microfast_growth` returns `rc` as `int64` (from `jnp.where` of
  Python ints). The initial `rc` in both loops must be `int64` to
  keep `while_loop` carry types consistent. Tested — int32 init
  produces `TypeError: carry input and output have different dtypes`.
- `nts_used` in the returned tuple is the count that actually ran
  successfully (or the last attempted count if retries exhausted).
- The Python-while variant (`newstate_calc_growth` in
  `newstate_calc.py`) is **kept**, not replaced — it remains useful
  for debugging and for callers that want per-attempt logging. The
  JIT variant lives in a new module (`newstate_calc_jit.py`) so the
  two implementations are side-by-side.

## Consequences

- `make_step_full` (Phase 9.4) can now be a single `jax.jit`'d
  closure end-to-end.
- Parity test in `tests/unit/test_newstate_calc_jit.py` verifies
  bit-for-bit equivalence with the Python-while version on the
  no-retry-needed scenario.
- JIT compilation succeeds (`test_jit_compiles_under_jax_jit`).
- Column mass conservation within 1e-6 (growth scenario).

## Validation

- 3 unit tests in `tests/unit/test_newstate_calc_jit.py`:
  - Parity with `newstate_calc_growth` (same output when no retry
    is triggered).
  - `jax.jit` compiles the wrapper and produces consistent output.
  - H2O mass (gas + condensate) conserved within 1e-6 relative.
- Full suite: 185/185 unit tests pass.
