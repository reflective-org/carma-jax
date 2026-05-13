"""Unit conversion utilities for the CARMA-JAX public API.

CARMA operates internally in CGS (cm, g, s, erg). All conversions
are collected here so the boundary between CGS and SI is explicit
and testable.

CGS ↔ SI reference:
  Length:   1 m  = 100 cm          (RM2CGS = 100)
  Pressure: 1 Pa = 10 dyne/cm²     (RPA2CGS = 10)
  Velocity: 1 m/s = 100 cm/s
  Density:  1 kg/m³ = 1e-3 g/cm³
  Area:     1 m² = 1e4 cm²

Internal concentration convention:
  gc [g/cm³/z] = q [kg/kg] × ρ_air [g/cm³]   (zmet = 1 in Cartesian)
  pc [#/cm³/z] = n [#/kg]  × ρ_air [g/cm³] × 1e-3   (1 g = 1e-3 kg)
"""
import jax.numpy as jnp

from carma.constants import R_AIR, RM2CGS, RPA2CGS
from carma.precision import DTYPE


# ---------------------------------------------------------------------------
# Scalar converters (operate on scalars or arrays element-wise)
# ---------------------------------------------------------------------------

def pa_to_cgs(p_pa):
    """Pressure Pa → dyne/cm²."""
    return DTYPE(RPA2CGS) * jnp.asarray(p_pa, dtype=DTYPE)


def cgs_to_pa(p_cgs):
    """Pressure dyne/cm² → Pa."""
    return jnp.asarray(p_cgs, dtype=DTYPE) / DTYPE(RPA2CGS)


def m_to_cm(x_m):
    """Length m → cm."""
    return DTYPE(RM2CGS) * jnp.asarray(x_m, dtype=DTYPE)


def cm_to_m(x_cm):
    """Length cm → m."""
    return jnp.asarray(x_cm, dtype=DTYPE) / DTYPE(RM2CGS)


def ms_to_cms(v_ms):
    """Velocity m/s → cm/s."""
    return DTYPE(RM2CGS) * jnp.asarray(v_ms, dtype=DTYPE)


def cms_to_ms(v_cms):
    """Velocity cm/s → m/s."""
    return jnp.asarray(v_cms, dtype=DTYPE) / DTYPE(RM2CGS)


def m2s_to_cm2s(d_m2s):
    """Diffusivity m²/s → cm²/s."""
    return DTYPE(RM2CGS ** 2) * jnp.asarray(d_m2s, dtype=DTYPE)


def m2_to_cm2_flux(f_per_m2s):
    """Flux per m²/s → per cm²/s (divide by 1e4)."""
    return jnp.asarray(f_per_m2s, dtype=DTYPE) * DTYPE(1e-4)


def cm2_to_m2_flux(f_per_cm2s):
    """Flux per cm²/s → per m²/s (multiply by 1e4)."""
    return jnp.asarray(f_per_cm2s, dtype=DTYPE) * DTYPE(1e4)


# ---------------------------------------------------------------------------
# Air density from ideal-gas law
# ---------------------------------------------------------------------------

def air_density_cgs(T_k, p_cgs):
    """Dry-air number density → mass density [g/cm³].

    Uses ideal-gas law: ρ = p / (R_air · T) with R_air in CGS.

    Args:
        T_k:    Temperature [K], any shape.
        p_cgs:  Pressure [dyne/cm²], same shape.

    Returns:
        ρ_air [g/cm³].
    """
    T = jnp.asarray(T_k, dtype=DTYPE)
    p = jnp.asarray(p_cgs, dtype=DTYPE)
    return p / (DTYPE(R_AIR) * T)


# ---------------------------------------------------------------------------
# Gas concentrations: mixing ratio ↔ CGS
# ---------------------------------------------------------------------------

def mmr_to_gc(q_kg_per_kg, rho_air_cgs, zmet):
    """Mass mixing ratio [kg/kg] → internal gas conc [g/cm³/z].

    Args:
        q_kg_per_kg:   Mass mixing ratio [kg/kg], shape (NZ,) or (NZ, NGAS).
        rho_air_cgs:   Air density [g/cm³], shape (NZ,).
        zmet:          Vertical metric (dimensionless), shape (NZ,).

    Returns:
        gc [g/cm³/z], same shape as q_kg_per_kg.
    """
    q = jnp.asarray(q_kg_per_kg, dtype=DTYPE)
    rho = jnp.asarray(rho_air_cgs, dtype=DTYPE)
    zm = jnp.asarray(zmet, dtype=DTYPE)
    if q.ndim == 2:
        # (NZ, NGAS): broadcast rho/zmet along gas axis
        return q * rho[:, None] * zm[:, None]
    return q * rho * zm


def gc_to_mmr(gc_cgs, rho_air_cgs, zmet):
    """Internal gas conc [g/cm³/z] → mass mixing ratio [kg/kg].

    Args:
        gc_cgs:        Gas concentration [g/cm³/z].
        rho_air_cgs:   Air density [g/cm³], shape (NZ,).
        zmet:          Vertical metric, shape (NZ,).

    Returns:
        q [kg/kg], same shape as gc_cgs.
    """
    gc = jnp.asarray(gc_cgs, dtype=DTYPE)
    rho = jnp.asarray(rho_air_cgs, dtype=DTYPE)
    zm = jnp.asarray(zmet, dtype=DTYPE)
    if gc.ndim == 2:
        return gc / (rho[:, None] * zm[:, None])
    return gc / (rho * zm)


# ---------------------------------------------------------------------------
# Particle concentrations: number mixing ratio ↔ CGS
# ---------------------------------------------------------------------------

def nmr_to_pc(n_per_kg, rho_air_cgs, zmet):
    """Number mixing ratio [#/kg_air] → internal particle conc [#/cm³/z].

    Args:
        n_per_kg:      Number mixing ratio [#/kg_air], shape (NZ, NBIN).
        rho_air_cgs:   Air density [g/cm³], shape (NZ,).
        zmet:          Vertical metric, shape (NZ,).

    Returns:
        pc [#/cm³/z], shape (NZ, NBIN).
    """
    n = jnp.asarray(n_per_kg, dtype=DTYPE)
    rho = jnp.asarray(rho_air_cgs, dtype=DTYPE)
    zm = jnp.asarray(zmet, dtype=DTYPE)
    # n [#/kg] × ρ [g/cm³] × 1e-3 [kg/g] = #/cm³; × zmet = #/cm³/z
    return n * rho[:, None] * zm[:, None] * DTYPE(1e-3)


def pc_to_nmr(pc_cgs, rho_air_cgs, zmet):
    """Internal particle conc [#/cm³/z] → number mixing ratio [#/kg_air].

    Args:
        pc_cgs:        Particle concentration [#/cm³/z], shape (NZ, NBIN).
        rho_air_cgs:   Air density [g/cm³], shape (NZ,).
        zmet:          Vertical metric, shape (NZ,).

    Returns:
        n [#/kg_air], shape (NZ, NBIN).
    """
    pc = jnp.asarray(pc_cgs, dtype=DTYPE)
    rho = jnp.asarray(rho_air_cgs, dtype=DTYPE)
    zm = jnp.asarray(zmet, dtype=DTYPE)
    return pc / (rho[:, None] * zm[:, None] * DTYPE(1e-3))
