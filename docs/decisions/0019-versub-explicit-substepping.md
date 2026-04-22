# ADR 0019: `versub` as an alternative explicit-substepping sedimentation solver

## Status

Accepted — Phase 8.5.

## Context

Fortran CARMA ships two vertical sedimentation solvers:

- `versol.F90` — implicit (Thomas-algorithm) tridiagonal solver.
- `versub.F90` — explicit first-order upwind with internal CFL-based
  substepping.

The `GroupConfig.ifallrtn` flag selects between them per group. Phase
3 ported `versol`; Phase 8.5 adds `versub` for use cases where the
substepped explicit scheme is preferable (large CFL, irregular grids,
faster per call).

## Decision

`src/carma/transport/versub.py` implements the explicit substepping
solver as a pure function matching the Fortran signature. Key JAX
choices:

- **Traced substep count**. `nstep_sed = floor(1 + cfl_max)`, doubled
  when any edge has opposing up/down velocities. The count is a
  traced `jnp.int32`; the substep loop is `jax.lax.fori_loop(0,
  nstep_sed, body, init)` which accepts a traced upper bound.
- **Vectorised flux expression**. The per-level flux
  `-cv[i]·dn[i] + cv[i-1]·up[i] + cv[i+1]·dn[i+1] - cv[i]·up[i+1]`
  is built via array shifts, with boundary corrections applied to the
  first/last element using `.at[].add()`.
- **Boundary conditions**. Fortran branches on `igridv ∈ {SIG,
  HYBRID}` vs Cartesian to swap top/bottom roles; we resolve this at
  trace time via a Python `if`.

## Alternatives considered

- **Static max-substeps with masking.** Rejected — would run a fixed
  32 iterations every call; `fori_loop` with a traced bound is
  cheaper on CPU and matches Fortran exactly.
- **Reuse `versol`'s CFL branching (uc=0/1).** Rejected — `versol` is
  a single-step solver (one implicit solve); mixing substepping into
  it would conflate two solvers and make the dispatch harder to read.

## Unexpected observation: diffusion ordering

Fortran's docstring claims versub is "more diffusive than versol".
Under our JAX ports (both first-order upwind), **versub is actually
less diffusive than versol at high CFL** because:

- `versub` substeps to keep each micro-step within CFL; each micro-step
  is an explicit first-order upwind sweep, which at CFL≤1 is close to
  the exact advection.
- `versol` switches to implicit (uc=1) at CFL>1 — implicit first-order
  upwind is more diffusive than CFL-bounded explicit steps.

The Fortran comparison presumably refers to a PPM configuration of
`versol`; our `versol.py` uses implicit Thomas throughout, so the
ordering inverts. The validation figure demonstrates this: at CFL≈2,
versub keeps a sharp pulse while versol broadens. At CFL<1 both
agree to plotting precision.

This is a physics-correctness note, not a bug. If Phase 10's
sulfate ensemble identifies scenarios where the diffusion matters,
we can revisit.

## Consequences

- `versub` is JIT-clean, composable under `vmap` over bin/group axes.
- `vertical.py` dispatch (Phase 9) will select `versub` vs `versol`
  via `GroupConfig.ifallrtn`.
- 8 unit tests verify pulse propagation, zero-velocity identity,
  non-negativity, CFL-triggered substepping, opposing-velocity double
  step, flux boundary, JIT equivalence.
- 2 figures in `plots/phase8_versub/`: pulse propagation over 12 h,
  versub vs versol at low vs high CFL.

## Validation

- 8 unit tests pass.
- Full suite: 179/179 unit tests pass.
