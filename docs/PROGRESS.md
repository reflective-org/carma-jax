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

## Phase 3b: Nucleation — COMPLETE

### Modules Implemented
- [x] `nucleation/sulfnucrate.py` — Vehkamaki 2002 binary H2SO4-H2O (verified vs paper Fig 8)
- [x] `nucleation/freezaerl_koop2000.py` — Koop 2000 homogeneous aerosol freezing
- [x] `nucleation/freezglaerl_murray2010.py` — Murray 2010 glassy aerosol heterogeneous freezing
- [x] `growth/upgxfer.py` — nucleation production transfer between groups
- [x] `setup_nuc.py` — bin/element nucleation mapping tables
- [x] `microfast.py` — multi-group orchestrator (nucleation + growth + solvers)
- [x] 2-group nuctest runner (sulfate → ice, 16 bins, 3 elements)

### Critical Bug Fixes During Phase 3b

1. **`setup_bins`**: Use exact Fortran formulas. Was using geometric mean for
   `rmassup`; Fortran uses `rmassup = 2*rmrat/(rmrat+1) * rmass`. Also fixed
   `dr` formula to match Fortran's `vrfact * (rmass/rho)^(1/3)`.

2. **`growp`**: Use GROUP-level `igrowgas` (from number concentration element),
   not per-element. Core mass elements must participate in growth — when ice
   crystals grow from bin i-1 to bin i, ALL elements (volatile + core) move
   together. Matches Fortran `growp.F90` which checks `igrowgas(iepart)`.

3. **`evapp`**: Only process elements belonging to the current group. Was
   applying ice evaporation rate to sulfate elements. Matches Fortran
   `evap_ingrp.F90` which loops `isub = 1..nelemg(ig)`.

4. **Initial conditions**: Match Fortran's approximate rhoa in nuctest
   (uses `100 mbar / R_AIR / 200K` instead of actual conditions). Without
   this, all bins had systematic 14-23% offset.

5. **Nucleation mapping**: Use TWO pairs (element 0→1 for ice number,
   element 0→2 for ice core mass). Previously only had 0→2, so ice
   number was never produced from sulfate freezing.

6. **`growevapl` per-bin threshold**: The PPM formula divides `dmdt` by `pc`.
   When a bin has `pc ≈ SMALL_PC = 1e-50`, this produces astronomical
   growth rates (e.g., 2.5e35/s), which corrupt the core mass element.
   Added per-bin relative threshold: `pc > max(FEW_PC, pc_group_max * 1e-20)`.

7. **Koop `rhosol`**: Use H2SO4 SOLUTE density (1.38 g/cm³) in `volrat`
   calculation, NOT sulfate particle density (1.78). The Fortran uses
   `rhosol(isol)` which is the solute density. Fixed ~20% rate error in
   bins 9-16.

### Phase 3b Validation (Fortran nuctest, dt=1s, 100 steps)

| Quantity | JAX | Fortran | Rel Error |
|----------|-----|---------|-----------|
| Sulfate MMR total | 3.40e-11 | 3.34e-11 | 1.94% |
| Ice volatile MMR total | 1.606e-05 | 1.622e-05 | 0.97% |
| Ice core MMR total | 9.73e-11 | 9.79e-11 | 0.66% |
| Gas MMR (H2O) | 2.394e-05 | 2.380e-05 | 0.60% |
| Volatile conservation | 1.5e-15 | — | Machine precision |

### Bin-by-bin validation at t=1s (all bins)

- Sulfate: max 0.31% error (bin 6), most < 0.2%
- Ice volatile: < 0.1% error in all non-zero bins
- Ice core: < 0.1% error in all non-zero bins
- Gas: 0.08% error

### Known Issues

- **Size distribution tail**: At t=100s, Fortran ice distribution extends
  to ~500 μm, but JAX cuts off at ~200 μm. Totals match within 1%, but
  the shape differs — the Fortran uses internal adaptive substepping
  (newstate_calc.F90) which resolves the growth/advection better than
  our dt=1 single-step approach. This is a refinement to fix in Phase 5
  (orchestration with substepping).
- **H2SO4 condensational growth on sulfate** not yet validated in an
  end-to-end test. Infrastructure exists (vapor pressure, diffusivity,
  growth kernels work for any gas), but needs a dedicated stratospheric
  and tropospheric scenario test. Planned before Phase 4.

### Phase 4-6: Not Started
