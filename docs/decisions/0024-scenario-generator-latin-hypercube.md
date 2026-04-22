# ADR 0024: Phase 10 scenario generator — Latin hypercube with log-uniform axes

## Status

Accepted — Phase 10.1.

## Context

Phase 10's headline deliverable is a 1000-scenario H₂SO₄ ensemble
covering stratosphere-to-troposphere conditions:

| param | range | units |
|---|---|---|
| T | [180, 310] | K |
| p | [1, 1013] | hPa |
| RH | [0.01, 1.05] | (fraction) |
| H₂SO₄ | [0.01, 100] | pptv |
| aerosol μ | [5, 500] | nm |
| aerosol σ_g | [1.2, 2.5] | (geom. SD) |

Random Monte Carlo on 1000 samples leaves visible gaps in the
parameter hypercube — birthday-paradox clumping is severe at
N=1000 over 6 dimensions. Latin hypercube and scrambled Halton
both stratify; Halton has stratification artefacts in higher
dimensions, LHS does not.

Two of the axes (H₂SO₄ and aerosol μ) span 3-4 orders of magnitude.
Linear sampling on those axes wastes >90% of scenarios at the high
end of the range; log-uniform sampling gives equal coverage of each
order.

## Decision

Use ``scipy.stats.qmc.LatinHypercube`` with:

- **Linear sampling** on T, p, RH, σ_g — these have ≤2× dynamic
  range.
- **Log-uniform sampling** on H₂SO₄ and aerosol μ — orders-of-
  magnitude dynamic range.

Generator is deterministic by seed (default 42); the same seed
always produces the same scenario list. The 1000-scenario NPZ is
committed to ``data/sulfate_scenarios_1000.npz`` (47 KB).

## Alternatives considered

- **Pure Monte Carlo.** Rejected — visible clumping, would need
  ~10× more scenarios for similar coverage.
- **Scrambled Halton.** Acceptable but the Phase 10 plan explicitly
  mentions LHS as the first option; LHS is also the well-known
  default in atmospheric ensemble work.
- **CMIP-profile-based sampling** (load real T/p profiles from a
  reanalysis dataset). Rejected for now — would require a new
  data dependency. Useful as a follow-on if the LHS ensemble misses
  important real-world combinations.
- **Linear sampling everywhere.** Rejected — wastes >90% of the
  H₂SO₄ axis on the upper-tropospheric end.

## Consequences

- 1000 scenarios fit in 47 KB compressed NPZ → committable.
- Reproducibility: seed=42 always gives the same scenarios.
- Phase 10.2 (Fortran orchestrator) and Phase 10.3 (JAX vmap
  ensemble) both load the same NPZ — same input to both
  implementations is a hard requirement for parity comparison.

## Validation

- 9 unit tests:
  - count, in-bounds, log-uniform vs linear-uniform histograms,
    seed reproducibility, save/load round-trip, no-duplicate-rows.
- 2 figures:
  - 1-D histograms confirming uniform coverage.
  - Key 2-D projections (T×p, T×RH, H₂SO₄×μ) confirming no
    clumping or empty regions.
- Full suite: 199/199 unit tests pass.
