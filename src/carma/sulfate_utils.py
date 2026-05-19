"""Sulfate physical-chemistry utilities for CARMA-JAX.

Pure-function JAX ports of three Fortran routines in
``sulfate_utils.F90``:

- ``wtpct_tabaz(temp, h2o_mass, h2o_vp)``
  H2SO4 weight-percent composition of a binary H2SO4/H2O particle,
  Tabazadeh et al. (GRL 1997). Piecewise activity fits valid for
  T = 185-260 K, activity = 0.01-1.0.

- ``sulfate_density(wtp, temp)``
  Binary H2SO4/H2O solution density [g/cm^3], linear-in-T fits to
  Washburn (NRC 1928) tables. Linear extrapolation to 180-380 K
  validated by Beyer, Ravishankara & Lovejoy (JGR 1996).

- ``sulfate_surf_tens(wtp, temp)``
  Surface tension [erg/cm^2 = dyn/cm], linear-in-T fits to
  Sabinina & Terpugow (1935) measurements, Mills (1996 thesis).

All three are pure functions: they take arrays in, return arrays
out, and compose under ``jax.jit`` / ``jax.vmap``. No Python
branching on traced values.

Ported from: sulfate_utils.F90 (CARMA base).
"""

import jax
import jax.numpy as jnp

from carma.constants import AVG, BK, WTMOL_H2O
from carma.precision import DTYPE


# ---------------------------------------------------------------------------
# Tables (from sulfate_utils.F90)
# ---------------------------------------------------------------------------

# wt% grid (46 pts) used by sulfate_density
_DNWTP = jnp.asarray([
    0.0, 1.0, 5.0, 10.0, 20.0, 25.0, 30.0, 35.0, 40.0,
    41.0, 45.0, 50.0, 53.0, 55.0, 56.0, 60.0, 65.0, 66.0, 70.0,
    72.0, 73.0, 74.0, 75.0, 76.0, 78.0, 79.0, 80.0, 81.0, 82.0,
    83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0,
    93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 100.0,
], dtype=DTYPE)

# Density coefficients: density = DNC0 + DNC1 * T  (at each wt% above)
_DNC0 = jnp.asarray([
    1.0, 1.13185, 1.17171, 1.22164, 1.3219, 1.37209,
    1.42185, 1.4705, 1.51767, 1.52731, 1.56584, 1.61834, 1.65191,
    1.6752, 1.68708, 1.7356, 1.7997, 1.81271, 1.86696, 1.89491,
    1.9092, 1.92395, 1.93904, 1.95438, 1.98574, 2.00151, 2.01703,
    2.03234, 2.04716, 2.06082, 2.07363, 2.08461, 2.09386, 2.10143,
    2.10764, 2.11283, 2.11671, 2.11938, 2.12125, 2.1219, 2.12723,
    2.12654, 2.12621, 2.12561, 2.12494, 2.12093,
], dtype=DTYPE)

_DNC1 = jnp.asarray([
    0.0, -0.000435022, -0.000479481, -0.000531558, -0.000622448,
    -0.000660866, -0.000693492, -0.000718251, -0.000732869, -0.000735755,
    -0.000744294, -0.000761493, -0.000774238, -0.00078392, -0.000788939,
    -0.00080946, -0.000839848, -0.000845825, -0.000874337, -0.000890074,
    -0.00089873, -0.000908778, -0.000920012, -0.000932184, -0.000959514,
    -0.000974043, -0.000988264, -0.00100258, -0.00101634, -0.00102762,
    -0.00103757, -0.00104337, -0.00104563, -0.00104458, -0.00104144,
    -0.00103719, -0.00103089, -0.00102262, -0.00101355, -0.00100249,
    -0.00100934, -0.000998299, -0.000990961, -0.000985845, -0.000984529,
    -0.000989315,
], dtype=DTYPE)

# Surface-tension tables (15 pts): σ = STC0 + STC1 * T
_STWTP = jnp.asarray([
    0.0, 23.8141, 38.0279, 40.6856, 45.335, 52.9305, 56.2735,
    59.8557, 66.2364, 73.103, 79.432, 85.9195, 91.7444, 97.6687, 100.0,
], dtype=DTYPE)

_STC0 = jnp.asarray([
    117.564, 103.303, 101.796, 100.42, 98.4993, 91.8866,
    88.3033, 86.5546, 84.471, 81.2939, 79.3556, 75.608, 70.0777,
    63.7412, 61.4591,
], dtype=DTYPE)

_STC1 = jnp.asarray([
    -0.153641, -0.0982007, -0.0872379, -0.0818509,
    -0.0746702, -0.0522399, -0.0407773, -0.0357946, -0.0317062,
    -0.025825, -0.0267212, -0.0269204, -0.0276187, -0.0302094,
    -0.0303081,
], dtype=DTYPE)


# ---------------------------------------------------------------------------
# 1. Tabazadeh 1997 weight percent
# ---------------------------------------------------------------------------

def wtpct_tabaz(temp, h2o_mass, h2o_vp):
    """H2SO4 weight-percent composition of a binary H2SO4/H2O particle.

    Implements Tabazadeh et al. (GRL 1997) piecewise fits. Valid for
    T = 185-260 K, water activity = 0.01-1.0. At activities outside
    that range, the formula extrapolates and the result is clamped
    to [1, 100]%.

    Args:
        temp: Temperature [K]. Scalar or array.
        h2o_mass: Water vapour mass concentration [g/cm^3].
            (h2o_mass * AVG / WTMOL_H2O gives number density.)
        h2o_vp: Water equilibrium vapour pressure over liquid water
            [dyn/cm^2 = erg/cm^3].

    Returns:
        Weight percent H2SO4 in [1, 100].

    Matches Fortran's ``wtpct_tabaz`` in ``sulfate_utils.F90``.
    """
    temp = jnp.asarray(temp, dtype=DTYPE)
    h2o_mass = jnp.asarray(h2o_mass, dtype=DTYPE)
    h2o_vp = jnp.asarray(h2o_vp, dtype=DTYPE)

    # Number density of water [1/cm^3] from mass concentration
    h2o_num = h2o_mass * AVG / WTMOL_H2O
    # Partial pressure in dyn/cm^2, then convert to mb (÷ 1000)
    p_h2o = h2o_num * BK * temp / DTYPE(1000.0)
    vp_h2o = h2o_vp / DTYPE(1000.0)

    # Water activity; activity > 1 is clamped in the high branch,
    # activity < 0.05 uses the low branch with a tiny-activity floor.
    activ_raw = p_h2o / vp_h2o

    # Three coefficient sets (a, b, c, d) for the two temperature curves
    # (contl at 190 K, conth at 260 K) across three activity regions.
    # Low branch: activ < 0.05  → use tiny-activity floor on activ
    a1_lo, b1_lo, c1_lo, d1_lo = (
        DTYPE(12.37208932), DTYPE(-0.16125516114),
        DTYPE(-30.490657554), DTYPE(-2.1133114241),
    )
    a2_lo, b2_lo, c2_lo, d2_lo = (
        DTYPE(13.455394705), DTYPE(-0.1921312255),
        DTYPE(-34.285174607), DTYPE(-1.7620073078),
    )
    # Mid branch: 0.05 ≤ activ ≤ 0.85
    a1_m, b1_m, c1_m, d1_m = (
        DTYPE(11.820654354), DTYPE(-0.20786404244),
        DTYPE(-4.807306373), DTYPE(-5.1727540348),
    )
    a2_m, b2_m, c2_m, d2_m = (
        DTYPE(12.891938068), DTYPE(-0.23233847708),
        DTYPE(-6.4261237757), DTYPE(-4.9005471319),
    )
    # High branch: activ > 0.85 → clamp activity to 1
    a1_h, b1_h, c1_h, d1_h = (
        DTYPE(-180.06541028), DTYPE(-0.38601102592),
        DTYPE(-93.317846778), DTYPE(273.88132245),
    )
    a2_h, b2_h, c2_h, d2_h = (
        DTYPE(-176.95814097), DTYPE(-0.36257048154),
        DTYPE(-90.469744201), DTYPE(267.45509988),
    )

    # Activity choice per branch (floor / clip applied where Fortran did)
    activ_low = jnp.maximum(activ_raw, DTYPE(1e-32))
    activ_mid = activ_raw
    activ_high = jnp.minimum(activ_raw, DTYPE(1.0))

    # Branch masks (bool arrays, broadcast over inputs)
    m_low = activ_raw < DTYPE(0.05)
    m_high = activ_raw > DTYPE(0.85)
    m_mid = (~m_low) & (~m_high)

    activ = jnp.where(m_low, activ_low,
                     jnp.where(m_high, activ_high, activ_mid))

    a1 = jnp.where(m_low, a1_lo, jnp.where(m_high, a1_h, a1_m))
    b1 = jnp.where(m_low, b1_lo, jnp.where(m_high, b1_h, b1_m))
    c1 = jnp.where(m_low, c1_lo, jnp.where(m_high, c1_h, c1_m))
    d1 = jnp.where(m_low, d1_lo, jnp.where(m_high, d1_h, d1_m))
    a2 = jnp.where(m_low, a2_lo, jnp.where(m_high, a2_h, a2_m))
    b2 = jnp.where(m_low, b2_lo, jnp.where(m_high, b2_h, b2_m))
    c2 = jnp.where(m_low, c2_lo, jnp.where(m_high, c2_h, c2_m))
    d2 = jnp.where(m_low, d2_lo, jnp.where(m_high, d2_h, d2_m))

    contl = a1 * (activ ** b1) + c1 * activ + d1
    conth = a2 * (activ ** b2) + c2 * activ + d2

    # Linear blend from 190 K (contl) to 260 K (conth)
    contt = contl + (conth - contl) * ((temp - DTYPE(190.0)) / DTYPE(70.0))
    conwtp = contt * DTYPE(98.0) + DTYPE(1000.0)
    wtp = (DTYPE(100.0) * contt * DTYPE(98.0)) / conwtp

    return jnp.clip(wtp, DTYPE(1.0), DTYPE(100.0))


# ---------------------------------------------------------------------------
# 2. Binary H2SO4/H2O solution density
# ---------------------------------------------------------------------------

def sulfate_density(wtp, temp):
    """Binary H2SO4/H2O solution density [g/cm^3].

    Linear-in-T fits at 46 weight-percent points; linearly interpolated
    between bracketing points. Temperature clamped to [180, 380] K.

    Args:
        wtp: Weight percent H2SO4 [0, 100]. Scalar or array.
        temp: Temperature [K]. Scalar or array.

    Returns:
        Density [g/cm^3].

    Matches Fortran's ``sulfate_density`` in ``sulfate_utils.F90``.
    """
    wtp = jnp.asarray(wtp, dtype=DTYPE)
    temp = jnp.asarray(temp, dtype=DTYPE)
    temp_loc = jnp.clip(temp, DTYPE(180.0), DTYPE(380.0))

    return _interp_lookup(wtp, temp_loc, _DNWTP, _DNC0, _DNC1)


# ---------------------------------------------------------------------------
# 3. Surface tension
# ---------------------------------------------------------------------------

def sulfate_surf_tens(wtp, temp):
    """Surface tension of binary H2SO4/H2O solution [erg/cm^2 = dyn/cm].

    Linear-in-T fits at 15 weight-percent points; linearly interpolated
    between bracketing points. Temperature clamped to [180, 380] K.

    Args:
        wtp: Weight percent H2SO4 [0, 100]. Scalar or array.
        temp: Temperature [K]. Scalar or array.

    Returns:
        Surface tension [erg/cm^2 = dyn/cm].

    Matches Fortran's ``sulfate_surf_tens`` in ``sulfate_utils.F90``.
    """
    wtp = jnp.asarray(wtp, dtype=DTYPE)
    temp = jnp.asarray(temp, dtype=DTYPE)
    temp_loc = jnp.clip(temp, DTYPE(180.0), DTYPE(380.0))

    return _interp_lookup(wtp, temp_loc, _STWTP, _STC0, _STC1)


# ---------------------------------------------------------------------------
# Shared interpolation helper (Fortran's i-1/i linear blend)
# ---------------------------------------------------------------------------

def _interp_lookup(wtp, temp_loc, wtp_tab, c0_tab, c1_tab):
    """Linear interpolation over a (wtp, temp) table matching Fortran's
    ``do while (wtp > wtp_tab[i]): i+=1`` + ``(den1*frac + den2*(1-frac))``.

    Given bracket indices ``i`` (upper) and ``i-1`` (lower) such that
    ``wtp_tab[i-1] < wtp <= wtp_tab[i]``:
        frac = (wtp_tab[i] - wtp) / (wtp_tab[i] - wtp_tab[i-1])
        result = c0[i-1]*frac + c1[i-1]*frac*T + c0[i]*(1-frac) + c1[i]*(1-frac)*T

    Clamping rules (from Fortran):
      - wtp ≤ wtp_tab[0] → use i=0 only
      - wtp ≥ wtp_tab[-1] → use i=last only
    """
    n = wtp_tab.shape[0]
    # searchsorted with side='left' gives i such that wtp_tab[i-1] < wtp ≤ wtp_tab[i]
    # (matches Fortran's `do while (wtp > wtp_tab(i))` then use i).
    i_upper = jnp.searchsorted(wtp_tab, wtp, side="left")
    # Clamp to [1, n-1] so i-1 stays in-range. When i_upper = 0, the
    # `i.eq.1 .or. wtp.eq.dnwtp(i)` branch triggers and we just return
    # den2; emulated here by setting frac = 0 so den2 dominates.
    i_upper = jnp.clip(i_upper, 1, n - 1)
    i_lower = i_upper - 1

    wtp_u = wtp_tab[i_upper]
    wtp_l = wtp_tab[i_lower]
    den2 = c0_tab[i_upper] + c1_tab[i_upper] * temp_loc
    den1 = c0_tab[i_lower] + c1_tab[i_lower] * temp_loc

    # Fortran explicitly short-circuits when i==1 or wtp==wtp_tab[i];
    # we mirror it with frac_denom > 0 guard to avoid 0/0 at the
    # tabulated endpoints (these only matter for wtp = wtp_tab[0]).
    denom = wtp_u - wtp_l
    safe_denom = jnp.where(denom > DTYPE(0.0), denom, DTYPE(1.0))
    frac = (wtp_u - wtp) / safe_denom
    # When i_upper clamped up to 1 because wtp < wtp_tab[0], treat as
    # pure-lower (frac=1, return den1). When wtp > wtp_tab[-1], frac<0
    # extrapolates linearly along the last segment — Fortran would
    # stop at i=last and use den2; match that by clamping frac ≥ 0.
    frac = jnp.clip(frac, DTYPE(0.0), DTYPE(1.0))

    return den1 * frac + den2 * (DTYPE(1.0) - frac)
