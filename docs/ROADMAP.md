# CARMA-JAX Porting Roadmap

## Strategy

Port CARMA process-by-process from Fortran 90 to JAX. Validate each process against Fortran benchmark outputs before moving to the next. Six phases, each gated by integration tests.

## Phase 1: Foundation + Coagulation

**Goal**: Project scaffolding, core data structures, first complete physics process.
**Gate**: `pytest tests/integration/test_coagtest.py` passes with rtol < 1e-10.

| Module | Fortran Source | Status |
|--------|---------------|--------|
| `precision.py` | `carma_precision_mod.F90` | Pending |
| `constants.py` | `carma_constants_mod.F90` | Pending |
| `enums.py` | `carma_enums_mod.F90` | Pending |
| `config.py` | `carma_types_mod.F90` (carma_type) | Pending |
| `state.py` | `carma_types_mod.F90` (carmastate_type) | Pending |
| `bins.py` | setupbins logic | Pending |
| `atmosphere_std.py` | `atmosphere_mod.F90` (tests) | Pending |
| `setup_atm.py` | `setupatm.F90` | Pending |
| `coagulation/setup_coag.py` | `setupcoag.F90` | Pending |
| `setup_ckern.py` | `setupckern.F90` | Pending |
| `coagulation/coagl.py` | `coagl.F90` | Pending |
| `coagulation/coagp.py` | `coagp.F90` | Pending |
| `coagulation/csolve.py` | `csolve.F90` | Pending |
| `microslow.py` | `microslow.F90` | Pending |

## Phase 2: Thermodynamics & Growth Setup

**Goal**: Vapor pressure, supersaturation, fall velocity, growth kernels.
**Gate**: `pytest tests/integration/test_falltest.py` passes.

| Module | Fortran Source | Status |
|--------|---------------|--------|
| `vapor_pressure.py` | `vaporp_h2o_*.F90`, `vaporp_h2so4_*.F90` | Pending |
| `supersaturation.py` | `supersat.F90` | Pending |
| `setup_vf.py` | `setupvf.F90`, `setupvf_std.F90`, `setupvf_std_shape.F90`, `setupvf_heymsfield2010.F90` | Pending |
| `setup_grow.py` | `setupgrow.F90` | Pending |
| `setup_gkern.py` | `setupgkern.F90` | Pending |
| `swelling.py` | wet radius routines | Pending |

## Phase 3: Fast Microphysics (Growth + Nucleation)

**Goal**: Coupled growth/evaporation/nucleation system.
**Gate**: `pytest tests/integration/test_growtest.py test_nuctest.py test_sulfatetest.py` pass.

| Module | Fortran Source | Status |
|--------|---------------|--------|
| `growth/growevapl.py` | `growevapl.F90` | Pending |
| `growth/growp.py` | `growp.F90` | Pending |
| `growth/evapp.py` | `evapp.F90` | Pending |
| `growth/evap_ingrp.py` | `evap_ingrp.F90` | Pending |
| `growth/evap_mono.py` | `evap_mono.F90` | Pending |
| `growth/evap_poly.py` | `evap_poly.F90` | Pending |
| `growth/upgxfer.py` | `upgxfer.F90` | Pending |
| `growth/downgxfer.py` | `downgxfer.F90` | Pending |
| `growth/downgevapply.py` | `downgevapply.F90` | Pending |
| `nucleation/sulfnuc.py` | `sulfnuc.F90` | Pending |
| `nucleation/sulfnucrate.py` | `sulfnucrate.F90` | Pending |
| `nucleation/sulfhetnucrate.py` | `sulfhetnucrate.F90` | Pending |
| `nucleation/hetnucl.py` | `hetnucl.F90` | Pending |
| `nucleation/actdropl.py` | `actdropl.F90` | Pending |
| `nucleation/freezaerl.py` | `freezaerl_*.F90`, `freezglaerl_*.F90` | Pending |
| `nucleation/freezdropl.py` | `freezdropl.F90` | Pending |
| `nucleation/melticel.py` | `melticel.F90` | Pending |
| `solvers/psolve.py` | `psolve.F90` | Pending |
| `solvers/gsolve.py` | `gsolve.F90` | Pending |
| `solvers/tsolve.py` | `tsolve.F90` | Pending |
| `solvers/totalcondensate.py` | `totalcondensate.F90` | Pending |
| `microfast.py` | `microfast.F90` | Pending |

## Phase 4: Vertical Transport

**Goal**: PPM advection, Brownian diffusion, tridiagonal solver, dry deposition.
**Gate**: `pytest tests/integration/test_falltest.py test_vdiftest.py test_drydeptest.py` pass.

| Module | Fortran Source | Status |
|--------|---------------|--------|
| `transport/vertadv.py` | `vertadv.F90` | Pending |
| `transport/vertdif.py` | `vertdif.F90` | Pending |
| `transport/versol.py` | `versol.F90` | Pending |
| `transport/versub.py` | `versub.F90` | Pending |
| `transport/vertical.py` | `vertical.F90` | Pending |
| `setup_bdif.py` | `setupbdif.F90` | Pending |
| `setup_vdry.py` | `setupvdry.F90` | Pending |

## Phase 5: Orchestration & Substepping

**Goal**: Complete timestep driver with adaptive substepping.
**Gate**: ALL Fortran benchmark tests pass.

| Module | Fortran Source | Status |
|--------|---------------|--------|
| `utils/smallconc.py` | `smallconc.F90` | Pending |
| `utils/maxconc.py` | `maxconc.F90` | Pending |
| `utils/fixcorecol.py` | `fixcorecol.F90` | Pending |
| `utils/coremasscheck.py` | `coremasscheck.F90` | Pending |
| `utils/zeromicro.py` | `zeromicro.F90` | Pending |
| `prestep.py` | `prestep.F90` | Pending |
| `newstate_calc.py` | `newstate_calc.F90` | Pending |
| `newstate.py` | `newstate.F90` | Pending |
| `step.py` | `step.F90` | Pending |

## Phase 6: Performance & Production

**Goal**: GPU/TPU readiness, float32 mode, benchmarks.
**Gate**: GPU benchmarks show meaningful speedup. Float32 tests pass.

- vmap end-to-end (1, 16, 256, 1024 columns)
- Float32 mode with relaxed tolerances
- GPU/TPU benchmarks
- Profiling and optimization
- Documentation and examples
