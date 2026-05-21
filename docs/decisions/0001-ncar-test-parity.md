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

### ZhaoTurco rate point-sample diagnosis

`tests/ncar_parity/diagnose_zhaoturco_rate.py` evaluates our
`binary_nuc_zhao1995` at the exact `carma_sulfatetest` initial
conditions (T=250 K, [H₂SO₄]=2.6e8 cm⁻³, RH=1.5%):

|   | JAX | Bench-implied | ratio |
|---|---|---|---|
| Rate [#/cm³/s] | 4.5e-4 | ~7.0e-4 | **0.64** |

Vehkamäki on the same point agrees with Fortran within ~5%, so the
bug lives inside `binary_nuc_zhao1995` itself (not in the dispatcher
or nucbin placement).

The rate-dominating term is `exhom = exp(-gstar/kT) ≈ 7.18e-18`. A 36%
rate ratio corresponds to only a 0.8% difference in `gstar`.

### ZhaoTurco term-by-term diff (Phase 6.6c)

Patched `sulfnucrate.F90` with a one-shot `write()` of every saddle
intermediate at the first call to `binary_nuc_zhao1995` (saved fixture:
`tests/ncar_parity/fixtures/zhaoturco_dump_fortran.txt`). Ran
`tests/ncar_parity/diff_zhaoturco_intermediates.py` to recompute the
JAX intermediates at the **exact same inputs** (read from the Fortran
dump rather than reconstructed from MMR).

Headline: **the rate function itself agrees to ~9 %, not 36 %.**

| term            | F90              | JAX              | rel diff |
|---|---|---|---|
| saddle_i        | 25               | 25               | 0 %      |
| pa_i            | 5.41886e+00      | 5.41966e+00      | +0.01 %  |
| pb_i            | 1.45588e+00      | 1.45677e+00      | +0.06 %  |
| c1_i            | -2.50692e+00     | -2.51166e+00     | -0.19 %  |
| fct_i           |  1.65823e+00     |  1.65478e+00     | -0.21 %  |
| fct_ip1         | -8.94077e-02     | -9.30028e-02     | **-4.02 %** |
| xfrac           |  5.11592e-02     |  5.32119e-02     | +4.01 %  |
| wstar           | 78.9488          | 78.9468          | -0.00 %  |
| sigma           | 72.8404          | 72.8411          |  +0.00 % |
| gstar           | 1.34814e-12      | 1.34513e-12      | -0.22 %  |
| ftry            | -39.0614         | -38.9741         | +0.22 %  |
| exhom           | 1.08610e-17      | 1.18509e-17      | **+9.11 %** |
| **nucrate_cgs** | **7.0326e-04**   | **7.6709e-04**   | **+9.08 %** |

So `binary_nuc_zhao1995` over-predicts by ~9 % at identical inputs,
not under-predicts by 36 %. The original 36 % gap is dominated by an
**input mismatch** ~5 % in `h2so4_cgs` (Python's diagnostic used a
crude `P/(R_AIR·T)` for `rho_air` while Fortran's `CARMASTATE_Create`
flow produces a slightly different effective density), amplified
through the exponential ftry term.

The residual 9 % at identical inputs comes from cumulative drift:
`pa/pb` table values differ by 0.05 %, propagating into `c1/c2`
(0.2 %) and `fct` (0.2 %); because `fct_hi` is close to zero at the
saddle, `xfrac = fct_hi/(fct_hi - fct_lo)` amplifies the 0.2 %
`fct_hi` drift to ~4 %, then `gstar/ftry` and `exp(ftry)` amplify
further. RGAS, BK, AVG, M_air all match between F90 and JAX exactly,
so the source is in the Lin–Tabazadeh or Ayers–Kulmala
parameterisations or in `_DNWTP/_DNC0/_DNC1/_DNPOT` rounding.

**Practical takeaway:** the ZhaoTurco rate routine is not buggy
enough to chase further — fixing it would close ~9 % of the test
gap, not 36 %. The bulk of the sulfatetest disagreement (40 % gas,
67 % per-bin) lives in the **substepping or bin-crossing** layer,
not in the per-call nucleation rate. That's the right place to look
next (e.g. comparison of nucrate trajectories across an entire
1800 s step, not just first call).

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
