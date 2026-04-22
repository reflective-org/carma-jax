# ADR 0011: `sulfhetnucrate` takes `r_preexist` as a direct scalar

## Status

Accepted — Phase 7.4b.

## Context

Fortran's `sulfhetnucrate.F90` reads the pre-existing-particle radius from the module-level array `r(nucbin, igroup)` indexed by the target bin and group. Our JAX port prefers pure functions whose inputs match their physical content rather than carrying state indices.

## Decision

`sulfhetnucrate(..., r_preexist, ...)` takes the seed radius as a scalar (or broadcastable array) directly. Indexing into the bin/group tables is the caller's responsibility — the caller already knows which bin holds the pre-existing seed and can pass `r[nucbin, igroup]`. This mirrors the pattern we already use in Phase 7.1–7.3 (`wetr.get_wetr`, `rhopart.rhopart`, `sulfate_utils.*`).

The Fortran orchestration that loops over `nucbin` (in `sulfnuc.F90`, ported in Phase 7.5) will vectorise or vmap this function over the bin axis. Since `sulfhetnucrate` is pure and depends only on scalar `r_preexist`, composition under `vmap(..., in_axes=(..., 0, ...))` is trivial.

## Alternatives considered

- **Take `nucbin` + `igroup` + full `r` array.** Rejected — couples the function to CARMA state shape and adds an indexing step that hides what physics actually depends on.
- **Take `r_preexist` as a batched array.** The function already supports this via broadcasting — no special handling needed beyond the scalar path.

## Consequences

- Phase 7.5 `sulfnuc.py` will `vmap` `sulfhetnucrate` over the bin axis to compute rates for every candidate seed bin in one call.
- Unit tests compose the function with a single scalar seed radius, exactly as the Fortran does per-bin.
- No performance cost — `jax.vmap` on a 20-bin axis is free on CPU/GPU for this computation.

## Validation

- 8 unit tests pass. Reference stratospheric point (T=220 K, RH=50%, H2SO4=10⁸, 50-nm seed) produces 2.9e-2 embryos/s per seed — physically sensible for polar stratospheric sulfate background.
- 4 figures show the Fletcher factor piecewise continuity at xm=1, r² seed-size scaling, heterogeneous-vs-homogeneous comparison, and a (T, seed-size) heatmap.
