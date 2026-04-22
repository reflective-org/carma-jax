# ADR 0015: `fixcorecol` is a per-(group, bin) column-vectorised fixer

## Status

Accepted — Phase 8.1.

## Context

Fortran's `fixcorecol.F90` loops `(igroup, ibin)` and runs a per-column
two-pass scan: one pass collects positive vs negative ``concgas_md``
contributions over all vertical levels; a second pass applies the
mass-conserving rebalance. The routine is small but the algorithm is
genuinely two-pass — a single ``lax.scan`` over ``iz`` cannot compute
``total_mass`` / ``missing_mass`` and simultaneously rescale.

## Decision

The JAX port mirrors the Fortran structure:

- Python loops over the two small axes: ``igroup`` (typically 1–3)
  and ``ibin`` (20–50, static).
- Vectorised reductions over ``iz`` using `jnp.sum(jnp.where(...))`,
  then a broadcast rescale.
- Groups without core elements (``ncore == 0``) are skipped at the
  Python level — no traced conditional.

Because ``total_mass`` and ``missing_mass`` are both scalars per
bin, we use `jnp.where(enough, pc_new_enough, pc_new_short)` to
choose between the two branches without a `lax.cond`. Both branches
are always evaluated; the cost is small because the work is ``O(nz)``
per bin.

## Alternatives considered

- **Single `lax.scan` over iz.** Impossible — the scan would need two
  passes, doubling code and complicating the rebalance formula.
- **`vmap` over bins.** Rejected — would force the bin index through
  the `.at[...].set(...)` updates, producing scatter-update graphs
  that are harder to read than the Python loop. Bin count is static
  and small.
- **Drop the Fortran short-mass clamp.** Kept it — when total column
  mass is insufficient to cover deficits, Fortran zeroes `pc[iepart]`
  to `total_core / rmass`. Same here. This is not mass-conserving but
  matches Fortran for reproducibility.

## Consequences

- `fixcorecol` is JIT-clean and composable under `jax.jit`.
- Column-total mass is conserved to machine precision (1e-16 in the
  synthetic validation figure; 1e-12 in the test gate).
- The companion `coremasscheck` is split into three static modes
  (`"roundoff"`, `"always"`, `"never"`) so callers choose diagnostic
  vs repair behavior without a traced string.

## Validation

- 13 unit tests:
  - `fixcorecol`: healthy-no-op, mass conservation under deficit,
    insufficient-mass clamp, `ncore=0` bypass, JIT equivalence.
  - `coremasscheck`: flag-exceeded diagnostic, roundoff-scale auto-fix,
    large-error no-fix in roundoff mode, always-mode forced fix,
    never-mode no modification, zero-num bypass, JIT, unknown-behavior
    validation.
- 1 validation figure (`fig1_fixcorecol_profile.png`) showing the
  before/after ``pc[number]`` profile, the free-mass-per-level trace,
  and the column-total rel err (1.4e-16).
