# ADR 0012: `sulfnuc` is a per-group, per-level pure function

## Status

Accepted — Phase 7.5.

## Context

Fortran's `sulfnuc.F90` is called from `microfast.F90` inside nested
`(iz, igroup)` loops. Each invocation reads global `gc`, `wtpct`,
`supsatl`, and bin tables and mutates module-level `rhompe` /
`rnuclg` arrays. The two physical branches (homogeneous and
heterogeneous) share the Zhao-Turco saddle search, and the
heterogeneous branch is gated on the homogeneous result (`rstar > 0`).

The JAX port must preserve that gating, remain JIT-clean, and compose
cleanly under `vmap` over vertical levels and groups.

## Decision

`sulfnuc(...)` is a pure function for a **single group, single level**
that:

1. Calls `homogeneous_nucleation(...)` to obtain `(nucrate, nucbin,
   rstar, ftry)`. Homogeneous method is selected by a Python-level
   `method` string (`"ZhaoTurco"` or `"Vehkamaki"`); the string is
   resolved at trace time, not threaded as a traced integer.
2. Calls `heterogeneous_nucleation(...)` which `vmap`s
   `sulfhetnucrate` over the bin axis (matching ADR 0011).
3. Returns bin-indexed `rhompe` (only `nucbin` is non-zero) and
   `rnuclg` (shape `(nbin,)`). Caller scatters into the outer
   `(nbin, nelem)` or `(nbin, igroup, ienucto)` array.

The signature is:

```python
def sulfnuc(
    temp, weight_percent, rh,
    h2so4, h2so4_cgs, h2o, h2o_cgs,
    r_bins, rmassup, rmrat_val, zmet,
    method="ZhaoTurco",
    do_homogeneous=True,
    do_heterogeneous=True,
    ...
) -> (rhompe, rnuclg)
```

Gating rules:

- When `method="Vehkamaki"`, heterogeneous nucleation returns zero
  (Vehkamaki does not produce `rstar`/`ftry`).
- When `do_heterogeneous=False`, heterogeneous nucleation is skipped.
- When the homogeneous Zhao saddle search fails, both arrays are zero.

## Alternatives considered

- **Thread a Python-integer `method_id`**. Rejected — `method_id`
  would be traced and require `lax.switch` or `lax.cond` for dispatch.
  Static string argument keeps both paths JIT-compileable as separate
  closures.
- **One function over `(iz, igroup)`**. Rejected — ties the API to the
  outer shape and forces reshaping when orchestration evolves.
  Per-group vmap is cheap and keeps the inner function small.
- **Return the full `(nbin, nelem)` tensor**. Rejected — caller owns
  which element slot each group's number-density element occupies;
  keeping the driver bin-shape-only matches the "pure physics, no
  state bookkeeping" pattern of Phase 7.1–7.4.
- **Merge homogeneous and heterogeneous into one kernel that runs
  both passes in one call to the Zhao path**. Fortran already
  duplicates the Zhao call (sulfnucrate + sulfhetnucrate both call it
  internally). We match the Fortran pattern for reviewability;
  optimising the duplicate call can happen in a later pass if the
  ensemble benchmark in Phase 10 identifies it as a hot spot.

## Consequences

- The outer Phase 7.6 `make_step_sulfate` factory binds `method`
  and process flags at compile time and `vmap`s over `iz` and
  `igroup`.
- Unit tests exercise both branches in isolation
  (`homogeneous_nucleation`, `heterogeneous_nucleation`) and the
  combined driver. Parity with `sulfhetnucrate` direct call verified.
- JIT is tested with `method` as a `static_argnames` entry.
- `β1` and `β2` are computed inside the driver rather than exposed;
  callers never need to pass collision coefficients.

## Validation

- 17 unit tests pass. Coverage: β-coefficient sanity, homogeneous
  rate positivity at the stratospheric reference point, no-saddle
  gate, Vehkamaki fallback, unknown-method error, zmet scaling,
  heterogeneous vs direct `sulfhetnucrate` per-bin parity,
  homogeneous/heterogeneous disable flags, JIT equivalence.
- 4 figures show rhompe vs T, bin placement of the critical cluster,
  per-bin rnuclg at three T, and a (T, H2SO4) heatmap of the ratio
  of heterogeneous to homogeneous summed rates.
