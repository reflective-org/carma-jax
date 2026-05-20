# 0001 — NCAR-test parity track

## Context

`../original-carma/CARMA/tests/` contains ~32 dedicated Fortran test
cases written by the original CARMA team at NCAR (Chuck Bardeen et al.).
Each isolates one physics mode (growth, coagulation, nucleation,
sedimentation, dry-dep, sulfate variants, optics, etc.) and writes a
reference text output to `tests/bench/<testname>.txt`.

Up to now our validation was *bespoke* — we hand-built sulfate
ensembles. This decision establishes the NCAR test suite as our primary
**standing parity gate** for any change in `src/carma/` or
`src/carma_diffrax/`.

## Status (tier-status)

The 32 NCAR tests are triaged by the JAX-port capability needed:

### Tier 1 — currently runnable
Physics is fully ported and we have a step-function path.

| NCAR test | JAX path | Status |
|---|---|---|
| `carma_sulfatetest`         | faithful JAX (`step_full_faithful`) | **GREEN totals (5%), per-bin xfail (~67%)** |
| `carma_sulfate_vehkamaki_test` | faithful JAX, `method="Vehkamaki"` | **GREEN totals (5%), per-bin xfail (~24%)** |
| `carma_nuctest`             | needs NGROUP=2 NELEM=3 config | TBD |
| `carma_growtest`            | needs ice (RHO_I) group | TBD |
| `carma_growintest`          | needs ice + diagnostic output | TBD |
| `carma_growclrtest`         | needs ice + clear-sky | TBD |
| `carma_growsubtest`         | needs ice + substep stress test | TBD |
| `carma_swelltest`           | hygroscopic swelling only, no time stepping | TBD — custom bench format |
| `carma_kappawetrtest`       | kappa-Köhler wet radius | TBD — custom bench format |

### Tier 2 — needs operator-split coag in diffrax
Faithful JAX has coag (already used in Tier-1 sulfatetest above).
Diffrax needs the `coag_step.py` wrapper from the existing plan.

| NCAR test | Status |
|---|---|
| `carma_coagtest`         | needs coag-only setup; custom 3-int header |
| `carma_coagonly_ensemble` | ensemble version |

### Tier 3 — needs multi-level (NZ>1) or vertical transport
We have `versol`/`versub`/`vertdif` in `src/carma/` but none of the
diffrax or faithful paths exercise NZ>1 columns yet.

| NCAR test |
|---|
| `carma_falltest` (sedimentation) |
| `carma_drydeptest` (dry deposition) |
| `carma_pheattest` (particle radiative heating) |
| `carma_vdiftest` (vertical diffusion) |
| `carma_scfalltest`, `carma_sigmafalltest`, `carma_sigmadrydeptest` |

### Tier 4 — needs new species / new physics
Some kernels are partially ported (`hetnucl`, `freezaerl_koop2000/mohler2010/tabazadeh2000`,
fractal optics) but not wired into a step path.

| NCAR test |
|---|
| `carma_bcoctest`, `carma_bc2gtest` (BC/OC carbon model) |
| `carma_aluminum_2nc_test` (aluminum two-nucleate) |
| `carma_fractalmicrotest`, `carma_fractalopticstest` |
| `carma_mietest`, `carma_fractaloptics_2nc_test` |
| `carma_nuc2test` (two-mode nucleation) |
| `carma_sulfhettest`, `carma_sulfhet_vehkamaki_test` (heterogeneous sulfate) |
| `carma_sulfate_2nc_test`, `carma_sulfate_ccdon_test` (two-nucleate sulfate) |
| `carma_cetest`, `carma_test`, `carma_inittest`, `carma_history` |
| `carma_kappawetrtest`, `carma_swelltest` (swelling-only, no step path) |

## Decisions

1. **Tolerances by path**:
   - Faithful JAX vs Fortran on **totals**: 10% rel err. Same algorithm,
     same physics — gap should be small but discretization can drift.
   - Faithful JAX vs Fortran on **per-bin**: 5% rel err is the *aspiration*.
     Currently per-bin disagreement on sulfatetest is ~67% (ZhaoTurco) /
     24% (Vehkamaki). Tests are `xfail` until we close the gap.
   - Diffrax vs Fortran: 5% rel err on totals (different discretization;
     Kvaerno5 + upwind vs PPM is expected to differ on shape).

2. **Bench parsing**: each NCAR test writes a slightly different layout
   in the "between-section scalars" zone. Per-test `pre_mmr_skip` and
   `per_step_skip` kwargs on `parse_bench` declare how many tokens to
   consume between (a) bin geometry and initial mmr, (b) time and the
   mmr block. See `tests/ncar_parity/_harness.py`.

3. **Test organisation**: one pytest file per NCAR test. Each builds
   the matching JAX config, runs the relevant JAX path, calls the
   harness `compare()`, and either passes or is marked `xfail` with
   a reason capturing the known disagreement.

## Known gaps (with diagnoses)

### Sulfatetest per-bin shape gap

Faithful JAX vs `carma_sulfatetest` (ZhaoTurco) totals agree to ~5 %
but per-bin shape disagrees by ~67 % median. Centroid analysis
(`tests/ncar_parity/diagnose_sulfatetest_gap.py`) shows:

| step | F centroid | J centroid | Δ |
|---|---|---|---|
| 5  | bin 19 | bin 22 | +3 |
| 10 | bin 22 | bin 24 | +2 |
| 25 | bin 25 | bin 26 | +1 |
| 99 | bin 26 | bin 27 | +0.8 |

The faithful JAX particle population grows into 1–3 bins HIGHER than
Fortran during the rapid-nucleation phase. With `rmrat=2` in mass,
+3 bins means 8× larger mean particle mass at peak. Same physics
(growth + ZhaoTurco + coag), different discretization in either the
substep dynamics or the bin-crossing handling.

Vehkamaki variant tests at 24% per-bin (better than ZhaoTurco's 67%)
and 1% on gas (vs 46%), suggesting most of the gap is ZhaoTurco-specific
rate-integration drift, not a general port bug.

## How to extend

Per-test stub:
```python
# tests/ncar_parity/test_<name>.py
from tests.ncar_parity._harness import parse_bench, bench_path, compare

@pytest.mark.slow
def test_<name>_totals():
    bench = parse_bench(bench_path("<name>"),
                         pre_mmr_skip=…, per_step_skip=…)
    # Build matching JAX config & initial state from F90 source as spec.
    # Run JAX for the F90's nstep × dtime.
    # Compare totals: total mass + final gas.
```

The F90 source at `original-carma/CARMA/tests/<testname>.F90` is the
authoritative spec for each test's configuration and initial conditions.
