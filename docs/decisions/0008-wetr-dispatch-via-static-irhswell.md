# ADR 0008: `get_wetr` dispatches via static `irhswell`, with Fitzgerald/Gerber deferred

## Status

Accepted — Phase 7.2.

## Context

`wetr.F90` exposes five swelling paths behind the `irhswell` switch: `I_NO_SWELLING`, `I_FITZGERALD`, `I_GERBER`, `I_WTPCT_H2SO4`, `I_PETTERS`. Fortran uses `if (irhswell == ...) then` blocks that only exercise one branch per group per call. Each branch has its own composition sub-switch: Fitzgerald has 9 (NH42SO4, NACL, etc.), Gerber has 4.

Two questions:
1. How should the JAX port dispatch between the five branches?
2. Which branches do we port in Phase 7?

## Decision

**Dispatch**: `get_wetr` takes `irhswell` as a Python int; a Python-level `if` chooses the branch at trace time. Branches are independent functions (`_wetr_petters`, `_wetr_wtpct`). This keeps each branch's JIT graph small — no branch not selected is traced.

**Scope**: Phase 7.2 ports only the three branches the sulfate workflow needs:

- `I_NO_SWELLING` — trivial.
- `I_PETTERS` — κ-Köhler (Petters & Kreidenweis 2007) with Yu (2015) low-T rescale below 190 K.
- `I_WTPCT_H2SO4` — binary H2SO4/H2O with Tabazadeh wt%, `sulfate_density`, and a one-pass Kelvin iteration.

`I_FITZGERALD` and `I_GERBER` raise `NotImplementedError` with a pointer to the later phase that will port them (sea-salt work).

## Alternatives considered

- **Port all five now.** Rejected — Fitzgerald's 9-composition enum and Gerber's 4-composition enum add ~150 LoC of lookup code that has no sulfate consumer. Would bloat the PR without testable downstream benefit.
- **JIT-traced branch dispatch via `jax.lax.switch`.** Rejected — `irhswell` is a static config flag (lives on `GroupConfig`), not a traced value. Python-level branching is cleaner; no need to pay for `lax.switch`'s always-evaluate-all-branches cost.
- **Single JIT'd function that takes all optional args.** Rejected — would force every call site to supply kwargs it doesn't use (e.g. `kappa` for `I_WTPCT_H2SO4`). Cleaner to fail fast with `ValueError` when a required arg is missing.

## Consequences

- Calling a branch without its required kwargs raises `ValueError`; calling `I_FITZGERALD` or `I_GERBER` raises `NotImplementedError`. Both surface at trace time, not compile time.
- Each branch JIT-compiles independently when called. A simulation that only uses `I_WTPCT_H2SO4` never pays the `I_PETTERS` compile cost.
- Adding Fitzgerald/Gerber later is a pure extension — no call-site changes needed, no enum renaming.
- The `SwellMethod` enum in `carma.enums` already exposes all five values; only three are honoured today.

## Validation

- 15 unit tests (`tests/unit/test_wetr_hygroscopicity.py`), all passing.
- Key assertion: `I_PETTERS` output matches the closed-form PK07 Eq. 6 (`rwet = rdry · (1 + RH·κ/(1-RH))^(1/3)`) to 1e-10 relative.
- `I_WTPCT_H2SO4` output at stratospheric conditions (T=220 K, RH=10%) gives growth factor ≈ 1.14 and wet density ≈ 1.56 g/cm³ — both in the range reported by Carslaw 1995.
- 4 validation figures in `plots/phase7_wetr_hygroscopicity/`: κ-Köhler sweep matches analytic reference to plotting precision; low-T transition at 190 K is visually smooth; H2SO4 GF vs RH curves behave as expected.
