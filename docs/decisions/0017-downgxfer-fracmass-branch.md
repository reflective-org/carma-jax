# ADR 0017: `downgxfer` implements the Fortran `fracmass` branch

## Status

Accepted — Phase 8.3.

## Context

`upgxfer` (Phase 3 port) uses the simplified ``elemass = rmass[ifrom,
igfrom]`` even when the source is a number element of a multi-core
group. Fortran's `downgxfer.F90` has a more detailed branch: when the
source group has cores AND the source element type is a pure number
(``itype ≤ I_VOLATILE``), the effective element mass is the volatile
fraction:

```
fracmass = 1 - Σ core_mass / (pc[num] · rmass)
elemass  = fracmass · rmass
```

This matters when the source bin's internal mixture (core vs.
volatile) determines how much mass transfers per nucleation event —
common in ice → liquid dissolution scenarios where the ice bin
carries a core but the production rate should depend on the volatile
water fraction only.

## Decision

`downgxfer` implements the `fracmass` branch faithfully. Activation
is a Python-level decision (resolved at trace time):

```python
need_fracmass = (ncore[igfrom] > 0
                  and itype_from <= int(ElementType.I_VOLATILE))
```

When true, the routine reads every core element index from
``icorelem[0:ncore, igfrom]``, sums their `pc` contributions at the
source bin, and builds `fracmass` with a safe denominator guard.

`upgxfer` is **not** updated in this phase. The Phase-3 simplified
version is still correct for every current upgxfer call site (growth
into pure number elements, where ncore=0 on the source). When
Phase 9 `step_full` wires heterogeneous ice nucleation into the
upgrade direction, we'll revisit upgxfer.

## Alternatives considered

- **Always compute `fracmass`.** Rejected — introduces a useless
  division per bin when `ncore = 0`, wasting graph ops and risking
  roundoff-induced 0/0 traps.
- **Merge `upgxfer` and `downgxfer` into one function.** Rejected —
  the direction gate (`ielem < iefrom` vs `ielem > iefrom`) and the
  `rnucpe` reset semantics differ. A unified API would require extra
  flags and obscure what each call does.

## Consequences

- `downgxfer` is JIT-clean; 8 unit tests verify direction gating,
  `reset` default / override, pconmax / rnuclg gates, fracmass branch
  activation, and JIT equivalence.
- The `reset=True` default mirrors Fortran's `rnucpe(:,:) = 0` at
  the top of `downgxfer.F90`. Callers that want to accumulate
  upgxfer + downgxfer into the same array pass `reset=False` on the
  second call.
- The `mass_factor` expression is a Python-level switch on small
  integer `ipow` values (-2, -1, 0, 1, 2). This compiles to a
  direct arithmetic op with no `jnp.power` overhead for the common
  cases, and falls through to `elemass ** ipow` for general powers.

## Validation

- 8 unit tests: up-direction fires, source-≤-target skip, reset
  default vs override, `pconmax < FEW_PC` gate, `rnuclg = 0` gate,
  fracmass branch with synthetic cores, JIT equivalence.
- Full suite: 161/161 unit tests pass.
