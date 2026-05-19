# ADR 0009: `rhopart` returns repaired `pc` alongside `rhop`

## Status

Accepted — Phase 7.3.

## Context

Fortran's `rhopart.F90` does two things in a single subroutine:

1. Computes `rhop[iz, ibin, igroup]` — dry particle density per bin.
2. Calls `getwetr` three times per bin (for `r`, `rlow`, `rup`) to populate the wet-radius arrays.
3. In-place: when numerical diffusion from advection produces `m_core > m_total`, it truncates `pc[iz, ibin, iepart] := m_core / rmass` and returns the updated density.

We have to port (1) and (3) as pure functions; (2) is orchestration that belongs in the `step_full` factory alongside `hygroscopicity` and the other wet-state updates, so it's deferred to Phase 9.

## Decision

`rhopart(pc, ...)` returns a tuple `(rhop, pc_repaired)`:

- `rhop`: `(NZ, NBIN, NGROUP)` bulk density.
- `pc_repaired`: `(NZ, NBIN, NELEM)` with the number-element clamped in bins where `m_core > m_total`.

Callers that don't care about the safety repair can just discard `pc_repaired`; callers that match Fortran behaviour use it.

## Alternatives considered

- **Return only `rhop`, do the repair in a separate `coremasscheck` step** — would require the caller to re-derive `m_core` and `m_total`, duplicating work. Rejected.
- **Raise on `m_core > m_total`** — Fortran silently repairs because this happens routinely under advection. Raising would break the first integration test that exercises transport + coag together. Rejected.
- **Mutate `pc` in place via `state._replace(pc=...)`** — that's the caller's responsibility, not `rhopart`'s. Rejected.
- **Split into `dry_density` + `repair_coremass`** — adds an API without simplifying; Fortran keeps them together for a reason. Rejected.

## Consequences

- `rhopart` is now the canonical "bulk density + core-mass safety" function. Phase 8 `fixcorecol` / `coremasscheck` handle column-level conservation on top of the bin-level repair that `rhopart` already does.
- The wet-radius piece of Fortran `rhopart` (the three `getwetr` calls) is NOT ported here. It will be orchestrated in Phase 9's `step_full` factory, which has the full state in scope.
- `rhop` is always in `[min(rhoelem), max(rhoelem)]` except under the core-truncation branch, where it equals the pure-core density. This is asserted by `test_rhopart_result_in_valid_range`.

## Validation

- 9 unit tests, all pass.
- Key assertion: for shell + one core, `rhop` matches the analytic volume-mixing formula `1 / ((1 - f_core) / ρ_shell + f_core / ρ_core)` to 1e-10 relative.
- 3 validation figures in `plots/phase7_rhopart/` show JAX-vs-analytic overlays (to plotting precision), three-element heatmap, and the safety clamp activation at `m_core / m_total = 1`.
