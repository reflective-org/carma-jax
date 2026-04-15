# CARMA-JAX

JAX port of [CARMA](https://github.com/ESCOMP/CARMA) (Community Aerosol and Radiation Model for Atmospheres) — a bin microphysical model for simulating aerosol and cloud particle processes.

Designed for GPU, CPU, and TPU acceleration.

## Status

**Phase 1 complete: Coagulation**

Validated against Fortran CARMA across 4000+ random scenarios:
- 47 bins (0.2 nm to 8 um), 12-hour simulations at dt=60s
- All scenarios below 0.017% total number error vs Fortran
- Mass conservation at machine precision (1e-15)
- 4.7x faster than Fortran on single-core CPU
- Fully JIT-compiled (time loop, bin loop, pair summation)

Remaining phases (growth, nucleation, transport, orchestration) are planned — see [docs/ROADMAP.md](docs/ROADMAP.md).

## Features

- **Pure JAX** — no dependencies beyond JAX/jaxlib
- **JIT-compiled end-to-end** — `jax.lax.scan` over timesteps and bins
- **GPU/TPU ready** — `jax.vmap` over atmospheric columns
- **Float64 precision** — matches Fortran to machine epsilon for mass conservation
- **Process isolation** — each physics process independently testable
- **Bin-resolved microphysics** — geometric mass bins from sub-nm to tens of um

## Installation

```bash
pip install -e ".[dev]"
```

For GPU:
```bash
pip install -e ".[gpu]"
```

## Quick Start

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from carma.precision import DTYPE
from carma.bins import setup_bins
from carma.coagulation.setup_coag import setup_coag
from carma.microslow import make_microslow
from carma.config import ElementConfig, GroupConfig, CarmaConfig
from carma.enums import ElementType
import numpy as np

# 47 bins from 0.2nm to 8um
NBIN = 47
r, rmass, vol, dr, dm, rup, rlow, rmassup = setup_bins(
    rmin=2e-8, rmrat=2.0, nbin=NBIN, rho=2.0
)

# Configure particle group and element
groups = (GroupConfig(
    name='aerosol', ishape=1, ienconc=0, is_ice=False, is_cloud=False,
    is_sulfate=False, do_vtran=False, do_drydep=False, ifallrtn=1,
    irhswell=0, rmrat=2.0, eshape=1.0, rmin=2e-8,
    r=r, rmass=rmass, vol=vol, dr=dr, dm=dm, rmassup=rmassup,
    rup=rup, rlow=rlow,
    rrat=jnp.ones(NBIN, dtype=DTYPE),
    rprat=jnp.ones(NBIN, dtype=DTYPE),
    arat=jnp.ones(NBIN, dtype=DTYPE),
),)
elements = (ElementConfig(
    name='dust', rho=jnp.full(NBIN, 2.0, dtype=DTYPE), igroup=0,
    itype=int(ElementType.I_INVOLATILE), icomposition=0,
    isolute=-1, kappa=0.0,
),)

# Setup coagulation tables
coag = setup_coag(NBIN, 1, 1, groups, elements,
                  np.array([[0]]), np.array([[0]]))

# Build JIT-compiled coagulation stepper
microslow_fn = make_microslow(
    nbin=NBIN, nelem=1, ngroup=1,
    elem_igroup=jnp.array([0]),
    icoag=coag.icoag, volx=coag.volx, icoagelem=coag.icoagelem,
    npairu=coag.npairu, npairl=coag.npairl,
    iup=coag.iup, jup=coag.jup, igup=coag.igup, jgup=coag.jgup,
    ilow=coag.ilow, jlow=coag.jlow, iglow=coag.iglow, jglow=coag.jglow,
    pkernel=coag.pkernel,
    ienconc_arr=jnp.array([0]),
    elem_itypes=jnp.array([int(ElementType.I_INVOLATILE)]),
)

# Initial condition: 1e6 cm^-3 in bin 1
NZ = 1
pc = jnp.full((NZ, NBIN, 1), 1e-50, dtype=DTYPE)
pc = pc.at[0, 0, 0].set(1e6)

# Constant Brownian kernel
ck0 = 8 * 1.38054e-16 * 298 / (3 * 1.85e-4)
ckernel = jnp.full((NZ, NBIN, NBIN, 1, 1), ck0, dtype=DTYPE)
zmet = jnp.ones(NZ, dtype=DTYPE)

# Run 720 steps (12 hours at dt=60s) inside JIT
@jax.jit
def run_simulation(pc, ckernel, zmet):
    def step(pc, _):
        pcl = pc
        pconmax = jnp.max(pc[:, :, 0:1] / zmet[:, None, None], axis=1)
        return microslow_fn(pc, pcl, ckernel, pconmax, zmet, DTYPE(60.0)), None
    pc_final, _ = jax.lax.scan(step, pc, None, length=720)
    return pc_final

pc_final = run_simulation(pc, ckernel, zmet)
print(f"Total N: {float(pc_final[0, :, 0].sum()):.2e} cm^-3")
```

## Testing

```bash
pytest tests/unit/ -v           # 26 unit tests
python scripts/validate_coagtest.py       # Fortran benchmark comparison
python scripts/fortran_vs_jax_47bin.py    # 1000-scenario validation
```

## Project Structure

```
src/carma/
  precision.py, constants.py, enums.py   — Foundation
  config.py, state.py, bins.py           — Data structures
  atmosphere_std.py, setup_atm.py        — Atmosphere
  setup_vf.py, setup_ckern.py            — Setup routines
  coagulation/                           — Coagulation physics
  microslow.py                           — Coagulation driver (JIT)
  growth/, nucleation/, solvers/         — Future phases
  transport/, utils/                     — Future phases
```

## Documentation

- [ROADMAP.md](docs/ROADMAP.md) — 6-phase porting plan
- [PROGRESS.md](docs/PROGRESS.md) — Phase completion tracker
- [ASSUMPTIONS.md](docs/ASSUMPTIONS.md) — Design assumptions with status
- [REFERENCES.md](docs/REFERENCES.md) — Papers and Fortran sources
- [decisions/0001-use-namedtuples.md](docs/decisions/0001-use-namedtuples.md) — Architecture decision records

## Validation Plots

Plots from 1000-scenario Fortran vs JAX comparison are in `plots/`:
- `fortran_vs_jax_47bin/` — Error distributions, per-bin analysis, timing
- `coagtest_validation/` — Single benchmark comparison with time evolution
