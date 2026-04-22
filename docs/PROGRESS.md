# CARMA-JAX Progress Tracker

## Current Phase: 8.3 (downgxfer) — IN PROGRESS

## Phase Summary

| Phase | Status | Gate |
|-------|--------|------|
| 1: Foundation + coagulation | COMPLETE | coagtest within 0.02% |
| 2: Thermodynamics + growth setup | COMPLETE | falltest ready, vapor pressure validated |
| 3a: Growth (one-group) | COMPLETE | growtest T rel err 1.15e-3 |
| 3b: Nucleation + microfast multi-group | COMPLETE | nuctest, sulfatetest |
| 4: Vertical transport | COMPLETE | falltest, vdiftest, drydeptest all <0.1% median |
| 5a: prestep + utilities + step_transport | COMPLETE | PR #7 merged |
| 5b: make_step_coag factory | COMPLETE | PR #8 merged |
| 5c: make_step_transport + public API | COMPLETE | PR #9 merged |
| 5d: JIT-readiness cleanups in microfast | COMPLETE | PR #10 merged |
| 5e: microfast_growth JITs + make_step_microfast | COMPLETE | PR #11 |
| growtest cliff fix | COMPLETE | PR #14 |
| 6a: fp64 precision-comparison harness | COMPLETE | PR #13 |
| 6b: fp32 overlay — float32 evaluated, parked | COMPLETE | PR #15 |
| 6c: CPU benchmark JAX vs Fortran | COMPLETE | PR #16 — JAX 1.2× / 0.5× / 2.1× of Fortran |
| 7.1: sulfate_utils (wtpct, density, surf_tens) | COMPLETE | 14 unit tests, 5 figures |
| 7.2: wetr (κ-Köhler + WTPCT) + hygroscopicity | COMPLETE | 15 unit tests, 4 figures |
| 7.3: rhopart (multi-element bin density) | COMPLETE | 9 unit tests, 3 figures |
| 7.4a: binary_nuc_zhao1995 (classical H2SO4/H2O) | COMPLETE | 8 unit tests, 4 figures |
| 7.4b: sulfhetnucrate (heterogeneous nucleation) | COMPLETE | 8 unit tests, 4 figures |
| 7.5: sulfnuc (hom + het driver) | COMPLETE | 17 unit tests, 4 figures |
| 7.6a: gasexchange (nuc + growth/evap gas flux) | COMPLETE | 10 unit tests, 3 figures |
| 7.6b: sulfate_step + make_step_sulfate factory | COMPLETE | 8 unit tests, 4 figures, mass cons <1e-4 @ 6h |
| 8.1: fixcorecol + coremasscheck | COMPLETE | 13 unit tests, 1 figure, col mass cons 1e-16 |
| 8.2: evap_mono + evap_poly | COMPLETE | 11 unit tests, 1 figure |
| 8.3: downgxfer (evap-direction nucleation xfer) | COMPLETE | 8 unit tests |

## Phase 1 (Foundation + Coagulation) — COMPLETE

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

## Phase 4: Vertical transport — COMPLETE

### Implemented
- `transport/vertadv.py` — PPM advection rates (Colella-Woodward 1984) with monotonicity limiter
- `transport/vertdif.py` — Brownian diffusion rates at layer edges
- `transport/versol.py` — Thomas tridiagonal solver via `jax.lax.scan`
- `transport/vertical.py` — JIT'd driver, `vmap` over (NBIN, NELEM)
- `setup_bdif.py` — Einstein-Stokes diffusivity + layer-edge interpolation
- `setup_vdry.py` — Zhang 2001 dry deposition velocity

### Validation

| Test | Process | Median err | Peak MMR: JAX vs Fortran |
|------|---------|-----------|--------------------------|
| falltest | PPM sedimentation | 0.053% | 3.159e-18 vs 3.160e-18 |
| vdiftest | Brownian diffusion | 0.053% | 8.290e-06 vs 8.290e-06 |
| drydeptest | Zhang 2001 drydep | 0.063% | 1.081e-10 vs 1.080e-10 |

### Key fixes found during validation
- `vertdif.py` top-boundary index was Fortran-style (`nz-1`) giving `log(1)/0` at the top edge. Fixed to `max(0, nz-2)` for 0-based indexing.
- Validation scripts now convert `pc ↔ mmr` via `rhoa_wet` (hydrostatic, CARMA internal) instead of `rhoa` (ideal gas) — these differ by ~3% in the mesosphere.
- Falltest uses `I_FIXED_CONC` (CARMA default), not `I_FLUX_SPEC`.

## Phase 5 (Orchestration) — COMPLETE

### Implemented
- `utils/smallconc.py` — `smallconc()` (per-element floor) + `maxconc()` (per-(level, group) max)
- `prestep.py` — d_gc/d_t substep increments, rewind gc/t, apply smallconc, compute pconmax
- `step.py` — three step factories: `make_step_transport`, `make_step_coag`, `make_step_microfast`
- `__init__.py` — public API surface (`carma.make_step_*`, `carma.CarmaConfig`, etc.)
- JIT-readiness cleanups in `newstate_calc.microfast_growth` and `growth/growevapl` so the microfast chain compiles end-to-end

### Speedups from JIT-ing the factories
| Script | Before | After | Speedup |
|--------|--------|-------|---------|
| validate_coagtest.py | ~12s physics loop | ~0.1s | ~100x |
| validate_growtest.py | 9.4s | 3.5s | ~2.7x |
| validate_falltest.py | ~60s | 2.3s | ~25x |
| validate_vdiftest.py | ~180s | 2.5s | ~70x |
| validate_drydeptest.py | ~25 min | 2.7s | ~550x |

### Deferred to Phase 6 / later

These are tracked here explicitly so we don't lose them:

1. **Adaptive retry inside JIT** — `newstate_calc_growth` currently uses a Python-level `while True` + `for isubstep in range(ntsubsteps)` retry loop. A JIT-native implementation would use `jax.lax.while_loop` for the retry doubling and `jax.lax.fori_loop(0, max_substeps, body_with_mask)` for the substep sweep. Callers that need retry (only triggered when supersaturation sign-flips with large magnitude) still go through the Python wrapper.

2. **`step_full(config)` composition** — a single driver that chains `prestep → vertical → microslow → microfast` conditionally on the `do_*` flags. Blocked on the adaptive retry work because multi-level microfast with substepping needs the retry story settled. The three factories in isolation already cover every benchmark we have.

3. **Multi-level microfast orchestration** — `make_step_microfast` currently operates at one `iz`. Fortran's `newstate_calc.F90` iterates top-down (or bottom-up for sigma grids) and threads sedimentation-into-layer accumulations (`dpc_sed`) between levels. Single-column tests don't exercise this; stratospheric column tests eventually will.

4. **In-cloud / clear-sky split** — scale particle concentrations by `cldfrc`, run the microphysics twice, recombine. Not exercised by any current benchmark (all tests are cloud-free or set cldfrc=1). Can wait until we run a CAM-coupled scenario.

5. **Utility modules still missing** — `fixcorecol`, `coremasscheck`, `zeromicro` (the last is implicit in JAX because fresh arrays are allocated every call, but the first two matter for core-mass conservation in coupled nucleation+coagulation runs).

6. **Size distribution shape at large radii** — the nuctest at t=100s shows Fortran's ice distribution extending to ~500 μm vs JAX's ~200 μm. Totals match within 1% and mass conservation is machine-precision, but the tail shape is under-resolved. Fortran's adaptive substepping catches it — our fixed-ntsubsteps drivers don't yet. Expected to resolve once the adaptive retry in item 1 is JIT'd.

7. **Float32 path** — evaluated in Phase 6b (PR #15) and *parked*. `precision.py` already parameterises the dtype via `CARMA_DTYPE=fp32`, and `scripts/compare_precision.py` + `scripts/plot_precision_overlay.py` measure the cost. Findings: coagtest and vdiftest run cleanly in fp32 (precision cost < 0.1%); falltest / drydeptest / growtest blow up because `SMALL_PC = 1e-50` underflows in fp32 (min subnormal ~1.4e-45), turning the `jnp.maximum(pc, SMALL_PC)` denominator-floor into a true zero. Fixable with dtype-aware constants + a few `.astype(float64)` islands (vapor pressure is the strongest candidate), but not worth the maintenance cost while Fortran-parity precision is the reference and CPU throughput isn't a bottleneck. Revisit only if (a) CPU fp64 becomes a throughput blocker, or (b) we target GPU batched runs where the 2× memory / ~2–3× throughput benefit justifies the mixed-precision plumbing. The harness stays in-tree as measurement infrastructure for that future decision.

8. **CPU throughput benchmark vs Fortran** — ran Phase 6c (`scripts/benchmark_cpu.py`, `plots/cpu_benchmark/benchmark.png`). JAX fp64 vs Fortran wall time, median of 3 steady-state runs after a warm-up call that drains the JIT cache into the hot path. `.block_until_ready()` on both sides so we're timing actual compute, not async dispatch.

    | test | Fortran | JAX (steady) | ratio |
    |------|---------|--------------|-------|
    | coagtest | 10 ms | 12 ms | 1.2× |
    | falltest | 61 ms | 31 ms | **0.5×** (JAX faster) |
    | growtest | 5 ms | 10 ms | 2.1× |

    Implication: **JAX fp64 is competitive with Fortran on CPU.** No case for pursuing fp32 on performance grounds. Item 7 (fp32) stays parked.

    Earlier draft numbers in PR #16 initially reported a 319× gap on growtest. That was a measurement bug — growtest's `loop_s` was timing the entire `run_jax_growtest()` function (atmosphere setup, growth-kernel setup, config build, JIT compile, *and* the stepping loop), while the other runners measured only the stepping loop. Fixed by adding explicit warm-up + `.block_until_ready()` and reporting `step_loop_s` separately.

9. **GPU / vmap benchmarks** — every JIT'd factory should compose with `jax.vmap` over a column batch axis already, but we haven't run the numbers.

9. **GPU / vmap benchmarks** — every JIT'd factory should compose with `jax.vmap` over a column batch axis already, but we haven't run the numbers.
