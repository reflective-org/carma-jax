# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**carma-jax** is a JAX port of CARMA (Community Aerosol and Radiation Model for Atmospheres), a bin microphysical model for simulating aerosol and cloud particle processes in the atmosphere. The original Fortran 90 implementation lives at `../original-carma/CARMA/`.

CARMA models: homogeneous/heterogeneous nucleation, condensational growth, evaporation, binary coagulation, cloud droplet activation, ice crystal freezing/melting, sedimentation, Brownian diffusion, and dry deposition. It supports sulfate aerosols, water droplets, ice crystals (various shapes), fractal aggregates, and mixed/coated particles.

## Build and Test Commands

```bash
pip install -e ".[dev]"          # Install in development mode
pytest tests/ -v                 # Run all tests
pytest tests/unit/ -v            # Unit tests only
pytest tests/integration/ -v     # Fortran benchmark comparison tests
pytest tests/jax/ -v             # JIT/vmap compilation tests
pytest tests/unit/test_foo.py -v # Single test file
pytest -k "test_name" -v         # Single test by name
```

## Architecture Rules

- **Pure functional**: No mutation. All functions take state in, return new state. Use `NamedTuple._replace()` for state updates, `jnp.ndarray.at[].set()` for array updates.
- **JIT-first**: Every public function must compile under `jax.jit`. No Python-level loops over dynamic (traced) values.
- **Static config, dynamic state**: `CarmaConfig` (model dimensions, process flags, precomputed tables) is static to JIT — passed via `functools.partial` or `static_argnums`. `CarmaState` (atmospheric arrays) is fully traced.
- **CGS internally**: Match Fortran's internal unit system (cm, g, s, erg) for numerical validation. MKS conversion only at public API boundary (`create_state` input, output extraction).
- **Float64 default**: Use `jax.config.update("jax_enable_x64", True)`. Parameterize dtype via `precision.py` for optional float32.
- **vmap for columns**: Single-column physics, parallelized via `jax.vmap` over batch dimension.
- **Process isolation**: Each physics process in its own subpackage (`coagulation/`, `growth/`, `nucleation/`, `transport/`, `solvers/`), independently testable.

## JAX Translation Patterns

- Fortran `do while` with dynamic iterations → `jax.lax.while_loop` with bounded max iterations
- Fortran `select case` on config values → Python `if/elif` (resolves at trace time since config is static)
- Fortran tridiagonal solver (Thomas algorithm) → `jax.lax.scan` for forward/backward passes
- Fortran nested bin/element loops → `jnp.einsum` or vectorized array ops where possible; Python loops over small static dimensions (NGROUP=2-10)
- Conditional physics (`if pconmax > FEW_PC`) → `jnp.where` computing both branches
- Adaptive substepping retry → `jax.lax.while_loop` + `jax.lax.fori_loop` with fixed max and masking

## Code Layout

```
src/carma/
  precision.py, constants.py, enums.py     # Foundation (Tier 0)
  config.py, state.py, bins.py             # Data structures
  atmosphere_std.py, setup_atm.py          # Atmosphere
  vapor_pressure.py, supersaturation.py    # Thermodynamics
  setup_vf.py, setup_grow.py, setup_gkern.py, setup_ckern.py  # Setup routines
  coagulation/                             # Coagulation process
  growth/                                  # Growth/evaporation process
  nucleation/                              # Nucleation processes
  solvers/                                 # psolve, gsolve, tsolve
  transport/                               # Vertical transport (PPM, tridiagonal)
  utils/                                   # smallconc, maxconc, fixcorecol
  microslow.py, microfast.py               # Process drivers
  prestep.py, newstate.py, newstate_calc.py, step.py  # Orchestration
```

## Validation Rules

- Every ported module gets unit tests (analytical cases, edge cases, JIT compilation check)
- Every completed process gets integration tests comparing against Fortran bench files in `tests/reference_data/`
- Target: relative error < 1e-10 for simple processes (coagulation, fall), < 1e-6 for coupled processes (growth+nucleation)
- Conservation laws (mass, number) checked in every integration test

## Documentation Rules

- Key decisions recorded in `docs/decisions/NNNN-title.md` (ADR format: context, decision, consequences)
- Progress tracked in `docs/PROGRESS.md` — updated after each phase completion
- Assumptions and hypotheses logged in `docs/ASSUMPTIONS.md` with status (confirmed/rejected/open)
- References and sources (papers, Fortran files) tracked in `docs/REFERENCES.md`
- Full porting roadmap in `docs/ROADMAP.md`

## Reference Fortran Implementation

The original source is in `../original-carma/CARMA/source/base/` (~90 F90 files). Key files:

- **carma_types_mod.F90** — Core data structures: `carma_type`, `carmastate_type`, `carmagroup_type`, `carmaelement_type`, `carmagas_type`, `carmasolute_type`
- **step.F90** → **newstate.F90** → **newstate_calc.F90** → **microfast.F90** / **microslow.F90** — Timestep call chain
- **carma_constants_mod.F90** — Physical constants in CGS
- **carma_enums_mod.F90** — Enumeration flags for all physics modes

### Fortran Model Workflow

```
Initialization: CARMA_Create → CARMAGROUP_Create → CARMAELEMENT_Create → CARMA_Initialize
Per-timestep:   CARMASTATE_Create → SetBin/SetGas → Step → GetBin/Gas/State → Destroy
```

### Unit System

CGS internally (cm, g, s, erg). MKS input interface. Conversions: `RM2CGS=100`, `RPA2CGS=10`, `RMB2CGS=1000`.

### Key Solver Characteristics (confirmed from code analysis)

- **psolve, csolve, gsolve, tsolve**: All explicit Euler — NOT iterative. `pc_new = (pc_old + dt*source) / (1 + dt*loss)`
- **versol**: Only truly implicit solver — tridiagonal Thomas algorithm for PPM vertical advection
- **Adaptive substepping**: `newstate_calc.F90` retry logic doubles substeps on convergence failure, max `maxretries` attempts

### Fortran Test Suite

32+ test cases in `../original-carma/CARMA/tests/` with benchmark outputs in `tests/bench/`. Tests are single-column, use US Standard Atmosphere 1976, and output text files parseable for validation.
