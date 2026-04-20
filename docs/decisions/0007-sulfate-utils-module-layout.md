# ADR 0007: `sulfate_utils` as top-level pure-function module

## Status

Accepted — Phase 7.1.

## Context

`sulfate_utils.F90` exposes three pure functions — `wtpct_tabaz`, `sulfate_density`, `sulfate_surf_tens` — that sulfate nucleation, hygroscopicity, and the H2SO4 condensational-growth path all need. The Fortran module scopes them with shared module-level data arrays (`dnwtp`, `dnc0`, `dnc1`). In JAX we have to pick:

1. Port each function into the module that first needs it (e.g. wtpct into `vapor_pressure.py`, density into `setup_grow.py`).
2. Keep them together in a dedicated `sulfate_utils.py` module.

Other Phase 7 pieces (`sulfnuc`, `hygroscopicity`, `wetr`, `rhopart`, `gasexchange`) also call these utilities.

## Decision

Port `sulfate_utils.F90` to a single `src/carma/sulfate_utils.py` that mirrors the Fortran module 1-to-1. Keep it at top level of `src/carma/` (not under `nucleation/` or `growth/`) because the functions are physics primitives used by several downstream modules.

Tables (`_DNWTP`, `_DNC0`, `_DNC1`, `_STWTP`, `_STC0`, `_STC1`) live as module-level `jnp.asarray(..., dtype=DTYPE)` constants — hashable enough to close over inside `jax.jit`, and clearly map to the Fortran `data` statements.

All three functions are pure (no state argument), vectorised (pass scalars or arrays), and JIT/vmap safe.

## Alternatives considered

- **Inline the tables at each call site.** Rejected — the 46-entry density table would be duplicated in every consumer module, risking transcription errors.
- **Split into `sulfate/` subpackage.** Rejected — Phase 7 only has six flat modules; a subpackage is overkill until cloud/ice doubles the count (Phase 11).
- **Return an `rc` failure code like Fortran.** Rejected — JAX idiom is to clamp silently (wt% in [1, 100], T in [180, 380] K) and trust static input validation. A downstream `assert` in `create_config` catches gross input errors.

## Consequences

- Future `sulfnuc` / `hygroscopicity` / `wetr` / `rhopart` / `gasexchange` modules import from `carma.sulfate_utils` directly.
- The piecewise branch of `wtpct_tabaz` is implemented with `jnp.where` on three masks (activity < 0.05, 0.05–0.85, > 0.85). Each branch's coefficients are selected in parallel; Fortran's four coefficient pairs per branch become 12 `jnp.where` lines. Slightly verbose but fully JIT-clean.
- The `do while (wtp > wtp_tab(i))` loop in Fortran's table lookup becomes a single `jnp.searchsorted(side="left")` — faster under JIT and avoids Python-level control flow on traced values.
- Fortran's "if i == 1 or wtp == dnwtp(i): return den2" shortcut is emulated by clamping `i_upper` into `[1, n-1]` and clipping `frac` into `[0, 1]`. The extrapolation behaviour matches the Fortran `temp_loc = clamp(temp, 180, 380)` rule in both temperature and wt%.

## Validation

`scripts/validate_sulfate_utils.py` generates five figures (branch continuity, density/surface tension curves, heatmap over stratosphere-to-troposphere envelope, canonical reference points). Canonical sanity check:

| Scenario | T [K] | RH | wt% | ρ [g/cm³] | σ [erg/cm²] |
|----------|-------|-----|-----|------------|--------------|
| Stratosphere | 220 | 10% | 59.1 | 1.547 | 78.8 |
| PSC | 195 | 50% | 37.6 | 1.353 | 84.8 |
| UT/LS | 230 | 20% | 53.3 | 1.476 | 79.8 |
| Troposphere | 275 | 60% | 36.7 | 1.287 | 77.7 |

Values consistent with Carslaw et al. (1995) composition tables to within the Tabazadeh 1997 fit accuracy.
