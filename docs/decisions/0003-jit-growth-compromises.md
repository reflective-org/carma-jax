# ADR 0003: JIT Growth Chain Compromises

## Status

Active — MUST be resolved before production use. Compromises 1 and 2 affect physics correctness and are not acceptable for scientific work. The current JIT version is for validation speed only.

## Context

The JIT-compiled `microfast_growth.py` achieves 4,200x speedup over the Python-loop version by eliminating all Python loops. However, it makes several simplifications relative to the full Fortran CARMA implementation. These are documented here for Phase 5 resolution.

## Compromises

### 1. Growth kernel frozen per timestep

**What Fortran does:** Recomputes `gro`, `gro1`, `akelvin` every timestep inside `CARMASTATE_Step` because T changes from latent heating, which changes vapor pressure, diffusivity, and thermal conductivity.

**What JIT does:** Uses `gro`, `gro1`, `akelvin` computed once from the initial T, reused for all 50 steps.

**Impact:** Small for the growtest (T changes by 3.5 mK). Could matter for stronger growth or longer simulations. The outlier scenarios (17/1000 with >10% error) may be partly caused by this.

**Fix:** Move `setup_gkern` inside the JIT time loop (ADR 0002 carry-state design). Cost: ~0.04ms/step additional.

### 2. No adaptive substepping

**What Fortran does:** `newstate_calc.F90` detects when supersaturation changes sign or exceeds thresholds, then doubles substeps and retries from saved state.

**What JIT does:** Single step per call, no convergence check, no retry.

**Impact:** Causes instability for extreme growth scenarios (low T, high supersaturation). 17/1000 scenarios show >10% error.

**Fix:** Implement substepping as `jax.lax.while_loop` inside the JIT function (ADR 0002). The fixed-max-iteration with masking pattern works here.

### 3. Single group, single element, single gas hardcoded

**What Fortran does:** Loops over NGROUP, NELEM, NGAS with full element-type dispatch (I_INVOLATILE, I_VOLATILE, I_COREMASS, etc.).

**What JIT does:** Assumes NGROUP=1, NELEM=1, NGAS=1 with ice (I_VOLATILE) element type.

**Impact:** Cannot handle multi-group simulations (e.g., sulfate + ice), mixed-composition particles, or multiple condensing gases.

**Fix:** Generalize with Python loops over groups at trace time (they unroll during JIT compilation since NGROUP is static). Same pattern as coagulation loss computation.

### 4. Single vertical level (NZ=1)

**What Fortran does:** Loops over NZ levels, each with independent microphysics.

**What JIT does:** Hardcodes iz=0.

**Impact:** Cannot do multi-level simulations with vertical transport.

**Fix:** `jax.vmap` over levels, or include the level loop inside the JIT scan. Straightforward since levels are independent for microphysics.

### 5. Vapor pressure inlined

**What Fortran does:** Calls `vaporp()` dispatcher which selects from 4 routines based on `ivaprtn` config.

**What JIT does:** Murphy 2005 formulas written directly in the JIT function.

**Impact:** Cannot switch vapor pressure parameterization without editing the JIT function.

**Fix:** Use `jax.lax.switch` for runtime dispatch, or generate separate JIT functions per vapor pressure routine (trace-time dispatch since `ivaprtn` is static config).

### 6. Simplified total condensate

**What Fortran does:** `totalcondensate.F90` loops over elements, bins, groups, separates ice from liquid, handles core mass subtraction.

**What JIT does:** `jnp.sum(pc * rmass)` — assumes all particle mass is volatile condensate.

**Impact:** Wrong for multi-element particles with non-volatile cores.

**Fix:** Generalize the condensate sum with element-type masking.

## Scientific integrity warning

**Compromises 1 and 2 are NOT acceptable for scientific simulations.** They change the physics:

- **Frozen kernel (#1)**: The growth rate depends on temperature through vapor pressure (exponential), diffusivity (T^1.94), and thermal conductivity (linear). Freezing these while T changes is physically wrong. Even small T changes compound over many steps.

- **No substepping (#2)**: Without convergence checking, the solver can overshoot equilibrium. This produces unphysical oscillations and wrong final states for fast-growing scenarios.

The correct approach: **always recompute all physics every step, then optimize the computation speed.** Never skip physics to gain speed. Our JIT-compiled `setup_vf_jit` (0.006ms) and `setup_ckern_jit` (0.04ms) are already fast enough to call every step. The growth kernel setup (`setup_gkern`) needs the same JIT treatment.

Compromises 3-6 are structural limitations (dimensions, generality) that don't affect the physics for the cases they support.

## What is NOT compromised

- **PPM algorithm**: identical to Fortran (same coefficients, same monotonicity, same Courant handling)
- **psolve formula**: identical implicit Euler
- **Gas solver**: identical forward Euler from condensate change
- **Temperature solver**: identical latent heat feedback
- **Kelvin curvature**: identical exponential formula
- **Growth rate (dmdt)**: identical formula from pheat.F90

## Resolution Plan

Phase 5 will implement the full carry-state `jax.lax.scan` architecture (ADR 0002), which naturally resolves compromises 1, 2, and 4. Compromises 3, 5, and 6 are resolved by generalizing the factory pattern with trace-time config dispatch.
