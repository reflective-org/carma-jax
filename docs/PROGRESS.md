# CARMA-JAX Progress Tracker

## Current Phase: 1 (Foundation + Coagulation) — COMPLETE

### Phase 1 Milestones

- [x] Project scaffolding (pyproject.toml, directories)
- [x] CLAUDE.md with architecture rules
- [x] Documentation files (ROADMAP, PROGRESS, ASSUMPTIONS, REFERENCES)
- [x] `precision.py` — dtype, thresholds
- [x] `constants.py` — physical constants in CGS
- [x] `enums.py` — IntEnum classes
- [x] `config.py` — CarmaConfig NamedTuple
- [x] `state.py` — CarmaState NamedTuple
- [x] `bins.py` — bin structure computation
- [x] `atmosphere_std.py` — US Standard Atmosphere 1976
- [x] `setup_atm.py` — atmospheric properties
- [x] `setup_vf.py` — fall velocity (Stokes/transitional/high-Re)
- [x] `coagulation/setup_coag.py` — mapping tables
- [x] `setup_ckern.py` — coagulation kernels (Brownian + gravitational)
- [x] `coagulation/coagl.py` — loss rates (vectorized with einsum)
- [x] `coagulation/coagp.py` — production terms (vectorized gather+reduce)
- [x] `coagulation/csolve.py` — explicit Euler solver
- [x] `microslow.py` — JIT-compiled coagulation driver (make_microslow factory)
- [x] Unit tests passing (26 tests)
- [x] Fortran benchmark validation (single scenario, 0.012% error)
- [x] 1000-scenario Fortran comparison (20-bin, all below 0.016%)
- [x] 1000-scenario Fortran comparison (47-bin, 0.2nm–8um, all below 0.017%)
- [x] Multi-bin initial conditions (all 35 bins populated, same accuracy)
- [x] Mass-per-bin error analysis (flat ~0.02% across all bins)
- [x] JIT time loop via jax.lax.scan (4.7x faster than Fortran on CPU)

### Phase 1 Validation Summary

| Test | Scenarios | Max Error | Mass Conservation |
|------|-----------|-----------|-------------------|
| Fortran coagtest benchmark | 1 | 0.012% total N | Machine precision |
| 20-bin random (dt=60-1800s) | 1000 | 0.016% total N | Machine precision |
| 47-bin random (dt=60s, 12h) | 1000 | 0.017% total N | Machine precision |
| 47-bin multi-bin init | 1000 | 0.017% total N | Machine precision |

### Performance

| Metric | Fortran | JAX (CPU) | Speedup |
|--------|---------|-----------|---------|
| 47-bin, 720 steps | 42 ms | 8.9 ms | **4.7x** |
| Per coag step (JIT'd) | — | 0.05 ms | — |

### Phase 2-6: Not Started
