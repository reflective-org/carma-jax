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

## Phase 2: Thermodynamics & Growth Setup — COMPLETE

### Modules Implemented
- [x] `vapor_pressure.py` — Buck 1981, Murphy 2005, Goff 1946 (H2O), Ayers 1980 (H2SO4)
- [x] `supersaturation.py` — standard and in-cloud variants
- [x] `setup_grow.py` — gas diffusivity, temperature-dependent latent heats
- [x] `setup_gkern.py` — surface tensions, Kelvin factors, ventilation, growth kernels
- [x] `setup_vf.py` — vectorized and JIT-compiled (0.006ms per call)
- [x] `setup_ckern.py` — vectorized and JIT-compiled (0.040ms per call)
- [x] 40 unit tests passing (14 new for Phase 2)
- [x] 1000-scenario Fortran comparison with computed kernel

### Phase 2 Validation (computed kernel, 1000 scenarios)

| Metric | Value |
|--------|-------|
| Mean total N error vs Fortran | 5.8e-5 (0.006%) |
| Median total N error | 4.8e-7 |
| Max total N error | 2.7e-4 (0.027%) |
| Mass error max | 4.2e-7 |
| JAX vs Fortran (CPU) | **8.7x faster** |

### Known Caveat

The coagulation kernel is currently computed **once** and reused for all timesteps. This is valid only when environmental variables (T, p, particle radii) are constant. In the full model (Phase 3+), the kernel must be recomputed when these change. The JIT-compiled `setup_ckern` at 0.04ms/call makes per-step recomputation feasible. See `docs/ASSUMPTIONS.md` for strategy options.

## Phase 3a: Growth + Evaporation — COMPLETE

### Modules Implemented
- [x] `solvers/psolve.py` — implicit Euler particle solver
- [x] `solvers/gsolve.py` — forward Euler gas solver (mass conservation)
- [x] `solvers/tsolve.py` — forward Euler temperature solver (latent heat)
- [x] `solvers/totalcondensate.py` — ice/liquid mass accumulator
- [x] `growth/pheat.py` — mass growth rate dm/dt with Kelvin curvature
- [x] `growth/growp.py` — bin-to-bin growth production
- [x] `growth/growevapl.py` — PPM (Colella-Woodward 1984) growth/evaporation rates
- [x] `growth/evapp.py` — evaporation production (within-group, no cores)
- [x] `microfast_growth.py` — JIT-compiled full growth step
- [x] `setup_gkern_jit.py` — JIT-compiled growth kernel setup
- [x] `newstate_calc.py` — adaptive substepping framework
- [x] 40 unit tests passing
- [x] Single scenario validation: 0.023% error vs Fortran
- [x] 1000-scenario validation (supersaturated conditions)

### Phase 3a Validation (1000 scenarios, 24 bins, 50 steps)

| Metric | Value |
|--------|-------|
| Median dT error | **5.3e-5** |
| 84% below 1% | |
| 94% below 10% | |
| Fortran vs JAX speed | **0.8-5.5x** (depends on substepping) |

### Physics correctness
- Full kernel recomputation every step (vapor pressure, diffusivity, latent heats, Knudsen, ventilation, gro/gro1)
- Adaptive substepping via `jax.lax.while_loop` (doubles substeps on supersaturation sign change)
- PPM coefficients from exact Fortran formulas (carma_mod.F90 lines 532-569)
- See ADR 0003 for compromise documentation

### Known Issues
- 15% of supersaturated scenarios have >0.1% error (mostly in high bins 20-24)
- Likely from subtle PPM differences or evaporation handling at bin boundaries
- Fortran growtest also runs without substepping — errors are not from missing substeps
- Absolute errors remain small (<1 mK for all scenarios)

### Phase 3b-6: Not Started
