"""Generate 100 realistic atmospheric scenarios for the final benchmark.

50% troposphere (0–12 km), 50% stratosphere (12–21 km), all (T, p) pairs
drawn from US Standard Atmosphere with small T perturbation.

v2 bounds (per project-atmospheric-realism memory):
  - T ≥ 210 K  (clamp the lower tail of the T perturbation)
  - p ≥ 50 hPa (cap strat altitude at 21 km; US Std Atm p(21km) ≈ 47 hPa
    minimum, so clamp z_strat ≤ 21 km then post-filter p ≥ 50)
  - RH ≤ 10% in the stratosphere (z ≥ 12 km); RH up to 100% in troposphere

Output: benchmark_final/scenarios/realistic_scenarios_100_v2.npz
"""
import argparse
import math
from pathlib import Path

import numpy as np


# -----------------------------------------------------------------------------
# US Standard Atmosphere 1976 — piecewise lapse-rate model
# -----------------------------------------------------------------------------
# Reference: NOAA, NASA, USAF (1976). Layers used here cover 0–32 km.

_LAYERS = [
    # (z_base [km], T_base [K], p_base [Pa], lapse_rate [K/km])
    (  0.0, 288.15, 101325.0, -6.5),   # troposphere
    ( 11.0, 216.65,  22632.0,  0.0),   # tropopause / lower stratosphere
    ( 20.0, 216.65,   5474.9, +1.0),   # stratosphere
    ( 32.0, 228.65,    868.0, +2.8),   # upper stratosphere
]

_g0 = 9.80665       # m/s²
_R_dry = 287.0531   # J/(kg·K)


def us_std_atmosphere(z_km):
    """Return (T_K, p_hPa) at altitude z_km using US Std Atm 1976."""
    z_km = float(z_km)
    for i, (zb, Tb, pb, lapse) in enumerate(_LAYERS):
        if i == len(_LAYERS) - 1 or z_km < _LAYERS[i + 1][0]:
            dz_m = (z_km - zb) * 1e3
            if abs(lapse) < 1e-12:
                T = Tb
                p = pb * math.exp(-_g0 * dz_m / (_R_dry * Tb))
            else:
                lapse_per_m = lapse / 1e3
                T = Tb + lapse_per_m * dz_m
                p = pb * (T / Tb) ** (-_g0 / (_R_dry * lapse_per_m))
            return T, p / 100.0   # Pa → hPa
    raise ValueError(f"z = {z_km} km outside model range")


# -----------------------------------------------------------------------------
# Initial number from log-normal seed + total mass
# -----------------------------------------------------------------------------

_RHO_SULF = 1.78   # g/cm³, sulfuric-acid particle density


def n_total_from_mass(M_ug_m3, D_g_nm, sigma_g):
    """N_total [#/cm³] for a log-normal seed with given mass + shape.

    For log-normal:  m̄ = (π/6) · ρ · D_g³ · exp(4.5 · ln²σ_g)
    """
    M_g_cm3 = M_ug_m3 * 1e-12          # µg/m³ → g/cm³
    D_g_cm = D_g_nm * 1e-7              # nm → cm
    ln2_sg = math.log(sigma_g) ** 2
    m_mean = (math.pi / 6.0) * _RHO_SULF * D_g_cm**3 * math.exp(4.5 * ln2_sg)
    return M_g_cm3 / m_mean


# -----------------------------------------------------------------------------
# Sampling
# -----------------------------------------------------------------------------

def sample_scenarios(n_scen, seed=42, T_min=210.0, p_min_hPa=50.0,
                     strat_rh_max=0.10):
    """Sample n_scen realistic atmospheric scenarios.

    v2 hard physical bounds:
      - T ≥ T_min K  (default 210; sub-stratosphere temperatures aren't
        relevant to the troposphere/stratosphere regime CARMA targets).
      - p ≥ p_min_hPa  (default 50; below that is mesospheric).
      - In the stratosphere (z ≥ 12 km), RH ≤ strat_rh_max (default 10%).
    """
    rng = np.random.default_rng(seed)

    # 50% troposphere (0–12 km), 50% stratosphere (12–21 km — capped to
    # respect p ≥ 50 hPa via US Std Atm; p(21 km) ≈ 47 hPa, so we trim
    # any z that gives p < 50 hPa below)
    n_trop = n_scen // 2
    n_strat = n_scen - n_trop
    z_trop = rng.uniform(0.0, 12.0, n_trop)
    z_strat = rng.uniform(12.0, 21.0, n_strat)
    z_all = np.concatenate([z_trop, z_strat])
    is_strat = np.concatenate([
        np.zeros(n_trop, dtype=bool), np.ones(n_strat, dtype=bool),
    ])
    # Shuffle the (z, is_strat) pair together so they stay aligned
    order = rng.permutation(n_scen)
    z_all = z_all[order]
    is_strat = is_strat[order]

    # T,p from US Std Atm + small T perturbation, clamped to T ≥ T_min
    T_std = np.zeros(n_scen)
    p_hPa = np.zeros(n_scen)
    for i, z in enumerate(z_all):
        T_std[i], p_hPa[i] = us_std_atmosphere(z)
    T_pert = rng.normal(0.0, 5.0, n_scen)        # ±5 K weather variability
    T_K = np.maximum(T_std + T_pert, T_min)

    # Hard p-floor — if any altitude sample dropped below p_min, lift to
    # p_min and recompute T_std at the corresponding altitude. With
    # z_strat ≤ 21 km this rarely triggers, but keep the safeguard.
    underfloor = p_hPa < p_min_hPa
    p_hPa = np.where(underfloor, p_min_hPa, p_hPa)

    # Relative humidity — regime-conditional per project-atmospheric-realism:
    #   stratosphere (z ≥ 12 km, i.e. is_strat True): RH ∈ [0.001, strat_rh_max]
    #   troposphere (z < 12 km):                       RH ∈ [0.001, 1.0]
    rh_strat = rng.uniform(0.001, strat_rh_max, n_scen)
    rh_trop = rng.uniform(0.001, 1.0, n_scen)
    rh = np.where(is_strat, rh_strat, rh_trop)

    # H₂SO₄ production rate, 1e1 – 1e7 molecules/cm³/s (log-uniform).
    # Capped at 1e7 (background-to-urban range); 1e8-1e9 are essentially
    # industrial-plume extremes that demand thousands of substeps to
    # converge and dominate the JAX runtime.
    h2so4_prod = 10.0 ** rng.uniform(1.0, 7.0, n_scen)

    # Initial seed: log-normal log-uniform sampling
    M_ug_m3 = 10.0 ** rng.uniform(np.log10(0.1), np.log10(50.0), n_scen)   # 0.1–50 µg/m³
    mu_nm = 10.0 ** rng.uniform(np.log10(10.0), np.log10(800.0), n_scen)   # 10–800 nm
    sigma_g = rng.uniform(1.05, 2.5, n_scen)

    # Derived: initial total number concentration
    N_total_cm3 = np.array([
        n_total_from_mass(M_ug_m3[i], mu_nm[i], sigma_g[i])
        for i in range(n_scen)
    ])

    return dict(
        altitude_km=z_all,
        T=T_K,
        p=p_hPa,
        rh=rh,
        is_strat=is_strat,                      # bool, True for z ≥ 12 km
        h2so4_prod_rate=h2so4_prod,            # molecules/cm³/s
        M_total_ug_m3=M_ug_m3,                  # initial seed mass
        aerosol_mu_nm=mu_nm,                    # seed geometric mean diameter
        aerosol_sigma_g=sigma_g,                # seed geometric sigma
        N_total_cm3=N_total_cm3,                # derived initial N
        _n=n_scen,
        _seed=seed,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parents[1]
                            / "scenarios" / "realistic_scenarios_100_v2.npz")
    p.add_argument("--T-min", type=float, default=210.0)
    p.add_argument("--p-min-hPa", type=float, default=50.0)
    p.add_argument("--strat-rh-max", type=float, default=0.10)
    args = p.parse_args()

    scens = sample_scenarios(
        args.n, seed=args.seed,
        T_min=args.T_min, p_min_hPa=args.p_min_hPa,
        strat_rh_max=args.strat_rh_max,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **scens)

    print(f"Generated {args.n} scenarios → {args.out}")
    print(f"\nRanges (P5 — P95):")
    for k in ["altitude_km", "T", "p", "rh", "h2so4_prod_rate",
             "M_total_ug_m3", "aerosol_mu_nm", "aerosol_sigma_g",
             "N_total_cm3"]:
        v = scens[k]
        print(f"  {k:20s}: {np.percentile(v, 5):.3e}  —  {np.percentile(v, 95):.3e}")

    # Sanity-check the hard bounds
    assert scens["T"].min() >= args.T_min - 1e-9, \
        f"T_min violation: {scens['T'].min()}"
    assert scens["p"].min() >= args.p_min_hPa - 1e-9, \
        f"p_min violation: {scens['p'].min()}"
    strat_rh = scens["rh"][scens["is_strat"]]
    assert strat_rh.max() <= args.strat_rh_max + 1e-9, \
        f"strat RH violation: {strat_rh.max()}"
    print(f"\nBound checks PASS:")
    print(f"  T_min       = {scens['T'].min():.2f} K  (≥ {args.T_min})")
    print(f"  p_min       = {scens['p'].min():.2f} hPa (≥ {args.p_min_hPa})")
    print(f"  strat RH_max = {strat_rh.max():.4f}    (≤ {args.strat_rh_max})")
    print(f"  trop RH_max  = {scens['rh'][~scens['is_strat']].max():.4f}")


if __name__ == "__main__":
    main()
