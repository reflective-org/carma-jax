# Final CARMA-JAX vs Fortran benchmark — realistic atmospheric ensemble

This directory contains the **final benchmark** comparing JAX and Fortran
CARMA on a realistic atmospheric ensemble. Unlike the earlier 1000-scenario
sulfate benchmark (which used independent uniform sampling and treated
H₂SO₄ as a one-shot initial condition), this benchmark uses:

- **Physically consistent (T, p) pairs** sampled from the US Standard
  Atmosphere over 0–30 km altitude
- **Continuous H₂SO₄ production rate** (1e1 – 1e9 molecules/cm³/s) instead
  of an initial-only concentration — represents SO₂ → H₂SO₄ oxidation by
  OH chemistry happening upstream of CARMA
- **Mass-based initial seed** (0.1 – 50 µg/m³) — total particle number is
  derived from the seed log-normal parameters and the total mass
- **Realistic RH** (≤ 60%) — avoids saturation/water-uptake artefacts
- **60 s outer timestep × 24 h total** = 1440 steps (matches what real
  GCMs use for fast chemistry)
- **Initial composition** is pure sulfuric acid

Both Fortran (patched binary, also at 60 s) and JAX (same setup) run the
same 100 scenarios and outputs are compared bin-by-bin and on integrated
moments.

## Directory layout

```
benchmark_final/
├── README.md                                       # this file
├── scenarios/
│   └── realistic_scenarios_100.npz                 # the 100 scenarios
├── fortran/
│   ├── carma_sulfatetest_realistic.F90             # patched Fortran source
│   ├── build_realistic.sh                          # build script
│   └── README.md                                   # what the patch does
├── outputs/
│   ├── fortran_outputs.npz                         # Fortran ensemble results
│   └── jax_outputs.npz                             # JAX ensemble results
├── plots/
│   ├── 01_initial_conditions.png                   # input variable distributions
│   ├── 02_total_number_comparison.png              # total N(JAX) vs N(Fortran)
│   ├── 03_total_mass_comparison.png                # total M(JAX) vs M(Fortran)
│   ├── 04_binbybin_number.png                      # per-bin number rel-err
│   └── 05_binbybin_mass.png                        # per-bin mass rel-err
└── scripts/
    ├── 01_generate_scenarios.py                    # samples 100 realistic scenarios
    ├── 02_plot_initial_conditions.py               # makes 01_initial_conditions.png
    ├── 03_run_jax_realistic.py                     # runs JAX over the 100 scenarios
    └── 04_plot_comparison.py                       # makes comparison plots
```

## Scenario parameter ranges

| Parameter | Range | Distribution | How |
|---|---|---|---|
| Altitude | 0 – 30 km | Uniform | drives T,p via US Std Atm |
| Temperature `T` | derived from altitude | + N(0, 5 K) perturbation | physically consistent with `p` |
| Pressure `p` | derived from altitude | — | physically consistent with `T` |
| Rel. humidity `RH` (over liquid) | 0 – 60% | Uniform | |
| H₂SO₄ **production rate** | 1e1 – 1e9 molecules/cm³/s | Log-uniform | continuous source |
| Initial mass conc. `M₀` | 0.1 – 50 µg/m³ | Log-uniform | total seed mass |
| Seed mode diameter `D_g` | 10 – 800 nm | Log-uniform | geometric mean |
| Seed sigma `σ_g` | 1.05 – 2.5 | Uniform | geometric std |
| Initial composition | pure H₂SO₄ | — | sulfate-only |

## Initial number from mass + size distribution

For a log-normal distribution with geometric mean diameter `D_g` and
geometric standard deviation `σ_g`, the mean particle mass is

```
m̄ = (π/6) · ρ_sulf · D_g³ · exp(4.5 · ln²σ_g)
```

so the total number concentration is

```
N₀ = M₀ / m̄
```

This gives atmospherically realistic N (~10 – 50,000 #/cm³ depending on
the (M₀, D_g, σ_g) combination).
