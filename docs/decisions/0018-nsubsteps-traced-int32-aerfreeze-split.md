# ADR 0018: `nsubsteps` returns traced int32; aerfreeze force is a companion helper

## Status

Accepted — Phase 8.4.

## Context

Fortran's `nsubsteps.F90` returns a Python-level integer that drives
a static substep loop. Porting directly to JAX has two wrinkles:

1. **Static vs traced loop bound.** `jax.lax.fori_loop(0, n, body, init)`
   supports a traced `n`, but most existing CARMA-JAX microstep loops
   use a static `ntsubsteps` fixed at factory time (`make_step_microfast`)
   to enable aggressive specialisation. Adaptive substepping means
   `n` varies per call, so the step loop needs to be re-specialised or
   accept a traced bound.

2. **Aerfreeze temperature threshold.** The Fortran aerfreeze branch
   needs both `supsati` and `T`. Threading `T` through the main
   `nsubsteps` signature forces every caller (even those who don't
   configure aerfreeze) to pass it.

## Decision

- `nsubsteps(...)` returns a **traced `jnp.int32` scalar**. Callers
  can cast to Python int (forcing a JIT specialisation on that value,
  acceptable when the counts are quantised to a small set like
  `{1, 2, 4, 8, 16, 32}`) or keep it traced and use `lax.fori_loop`
  with a dynamic bound (Phase 9).

- The aerfreeze short-circuit is split out as a separate
  `aerfreeze_force(supsati, temp, iz, ...)` helper returning a
  traced bool. The `nsubsteps` main kernel does **not** carry `T`;
  callers that configure aerfreeze OR-in the aerfreeze flag on top
  of the returned count themselves (pseudocode):

  ```python
  n = nsubsteps(...)
  if aerfreeze_force(...):
      n = maxsubsteps
  ```

## Alternatives considered

- **Return a Python int via a non-JIT kernel.** Rejected — the
  growth-rate branch reads `gro`, `gro1`, `supsatl`, `pvapl`, which
  are traced in the normal step path. Forcing materialisation at the
  Python boundary would break the step-level JIT.
- **Fold `aerfreeze_force` into `nsubsteps`.** Rejected — it's the
  only consumer of `T` in the kernel and is often disabled; keeping
  it separate is cleaner and documents the dependency.
- **Use `lax.cond` for the activation / aerfreeze branches.** Not
  needed — both reduce to `jnp.where(force_max, maxsubsteps,
  n_growth)` which is cheaper.

## Consequences

- `make_step_microfast` and friends stay static-ntsubsteps for now.
  Phase 9's `step_full` will wire adaptive substepping via
  `lax.fori_loop(0, n_traced, ...)` once the adaptive-retry path is
  in place.
- Callers that want aerfreeze must compose the two calls explicitly;
  documented in the module docstring.
- The growth-rate limit uses `jnp.take` (via fancy indexing) to read
  `gro`/`gro1` at the traced `ibin_small` — that produces a gather
  op at graph build time, small overhead.

## Validation

- 10 unit tests:
  - `do_substep=False` → 1
  - Trivial-growth (small `gro`, large `dm`) → min
  - Fast-growth → scales up then caps at max
  - Activation with DROPACT forces max
  - Low `pconmax` skips growth path (→ min)
  - `aerfreeze_force` fires only when cold AND ss > 0.4; false otherwise
  - JIT equivalence
- Full suite: 171/171 unit tests pass.
