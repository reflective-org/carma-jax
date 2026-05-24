# Fortran patch — realistic-ensemble benchmark

`carma_sulfatetest_realistic.F90` is a patched version of the standard
`carma_sulfatetest_ensemble.F90` (committed under
`scripts/fortran_patch/`). It exists so we can run Fortran CARMA at the
**60 s timestep** matched to JAX and with the **H₂SO₄ continuous
production** forcing.

## What the patch changes

| Aspect | Standard ensemble (`scripts/fortran_patch/`) | Realistic ensemble (here) |
|---|---|---|
| Outer timestep `dtime` | 1800 s (30 min) | **60 s** |
| Number of outer steps | 100 (50 h sim) | **1440** (24 h sim) |
| Scenario file format | 6 fields: `T p rh h2so4_pptv mu_nm sigma_g` | **7 fields:** `T p rh h2so4_prod_rate M_total_ug_m3 mu_nm sigma_g` |
| H₂SO₄ initial concentration | `mmr(:,2) = h2so4_pptv × 1e-12 × 98/29` | **0** (no initial H₂SO₄) |
| H₂SO₄ per-step forcing | none | **`+= prod_rate × dt × M_H2SO4 / N_A / ρ_air`** every step |
| Seed normalisation | total mmr = 1e-18 g/g (arbitrary) | **target mmr = M_total_ug_m3 × 1e-12 / ρ_air** (real µg/m³) |

The H₂SO₄ injection happens **before** the microphysics call each
outer step, so the freshly produced H₂SO₄ is available for nucleation /
growth within that step (instead of accumulating then being processed
on the next step). At 60 s timestep this is the realistic limit.

## Build

```bash
./benchmark_final/fortran/build_realistic.sh
```

Drops the F90 file into `../original-carma/CARMA/tests/`, registers it
with CMake if not already, and builds the binary at
`../original-carma/CARMA/build/test_sulfate_realistic`.

## Run one scenario manually (for sanity-check)

```
$ echo "275.0 850.0 0.4 1e7 5.0 50.0 1.8" > /tmp/scen.txt
$ ../original-carma/CARMA/build/test_sulfate_realistic /tmp/scen.txt /tmp/out.json
$ cat /tmp/out.json
```

Fields in the scenario line (whitespace-separated):

1. `T_K` — temperature [K]
2. `p_hPa` — pressure [hPa]
3. `rh_fraction` — relative humidity over liquid [0–1]
4. `h2so4_prod_rate` — H₂SO₄ production rate [molecules/cm³/s]
5. `M_total_ug_m3` — initial seed mass concentration [µg/m³]
6. `mu_nm` — log-normal seed geometric mean diameter [nm]
7. `sigma_g` — log-normal seed geometric standard deviation

## Output

Same JSON schema as the standard ensemble: `T_final`, `gc_h2so4_final`,
`pc_final` (NBIN-element array), `nstep_ran`, `status` — plus the
`<output_path>.schedule.bin` sidecar with the substep history.
