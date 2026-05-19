"""Heymsfield & Westbrook (2010) fall velocity for ice particles.

Ported from `setupvf_heymsfield2010.F90`. Computes per-(level, bin)
fall velocity vf [cm/s], Reynolds number re, and slip-correction
factor bpm. Used as an alternative to the standard CARMA setupvf for
ice-only groups — the Heymsfield-Westbrook formulation requires
particle area ratio (`arat`) and mass (`rmass`), and is single-regime
(no Stokes / transitional / high-Re branching).

Formula (per altitude k, bin i, group j):
    rhoa_cgs = rhoa / zmet
    vg       = sqrt(8/π · R_AIR · T)
    rmfp     = 2·μ / (ρ_a · vg)
    rkn      = rmfp / (r_wet · rrat)
    bpm      = 1 + 1.246·rkn + 0.42·rkn·exp(-0.87/rkn)
    dmax     = 2·r_wet·rrat
    x        = (ρ_a / μ²) · 8·m·g / (π · √arat)            (Best/Davies number)
    x       *= bpm                                           (slip correction)
    re       = (δ₀² / 4) · (√(1 + 4·√x / (δ₀²·√c₀)) − 1)²
    vf       = μ · re / (ρ_a · dmax)

with c₀ = 0.35, δ₀ = 8.0 (Heymsfield & Westbrook 2010 fit constants).

This routine does *not* perform the vf boundary-interpolation step
that `setupvf_std` does; the Fortran subroutine writes vf only at
cell centers (`vf(k,i,j)`). The caller can interpolate to layer
edges (geometric mean of adjacent centers) externally if needed —
that pattern is used elsewhere in CARMA but is not part of this
kernel.
"""
import jax
import jax.numpy as jnp

from carma.constants import GRAV, PI, R_AIR
from carma.precision import DTYPE, POWMAX


_C0 = DTYPE(0.35)
_DELTA0 = DTYPE(8.0)
_DELTA0_SQ = _DELTA0 ** 2


@jax.jit
def setup_vf_heymsfield2010_jit(t, rhoa, zmet, rmu, r_wet, rrat, rmass, arat):
    """Heymsfield & Westbrook 2010 fall velocity for ice particles.

    Args:
        t:      Temperature, shape ``(NZ,)`` [K].
        rhoa:   Air column density (ρ·zmet), shape ``(NZ,)`` [g/cm²].
        zmet:   Vertical metric, shape ``(NZ,)``.
        rmu:    Dynamic viscosity, shape ``(NZ,)`` [g/cm/s].
        r_wet:  Wet particle radius, shape ``(NZ, NBIN, NGROUP)`` [cm].
        rrat:   Radius ratio (shape factor), shape ``(NBIN, NGROUP)``.
        rmass:  Particle mass, shape ``(NBIN, NGROUP)`` [g].
        arat:   Projected area ratio, shape ``(NBIN, NGROUP)``.

    Returns:
        Tuple ``(vf, re, bpm)``:
            vf:  Fall velocity at cell centers, shape ``(NZ, NBIN, NGROUP)`` [cm/s].
            re:  Reynolds number, shape ``(NZ, NBIN, NGROUP)``.
            bpm: Cunningham slip correction, shape ``(NZ, NBIN, NGROUP)``.
    """
    rhoa_cgs = rhoa / zmet                                     # (NZ,)
    vg = jnp.sqrt(DTYPE(8.0) / PI * R_AIR * t)                 # (NZ,)
    rmfp = DTYPE(2.0) * rmu / (rhoa_cgs * vg)                  # (NZ,)

    r_eff = r_wet * rrat[None, :, :]                           # (NZ, NBIN, NGROUP)
    rkn = rmfp[:, None, None] / r_eff                          # (NZ, NBIN, NGROUP)

    expon = jnp.maximum(-POWMAX, -DTYPE(0.87) / rkn)
    bpm = (DTYPE(1.0) + DTYPE(1.246) * rkn
           + DTYPE(0.42) * rkn * jnp.exp(expon))

    dmax = DTYPE(2.0) * r_eff                                  # (NZ, NBIN, NGROUP)

    # Best/Davies number, then apply slip correction (Seinfeld & Pandis 8.46).
    x = (rhoa_cgs[:, None, None] / (rmu[:, None, None] ** 2)
         * (DTYPE(8.0) * rmass[None, :, :] * GRAV
            / (PI * jnp.sqrt(arat[None, :, :]))))
    x = x * bpm

    re = (_DELTA0_SQ / DTYPE(4.0)
          * (jnp.sqrt(DTYPE(1.0)
                      + DTYPE(4.0) * jnp.sqrt(x)
                        / (_DELTA0_SQ * jnp.sqrt(_C0)))
             - DTYPE(1.0)) ** 2)

    vf = rmu[:, None, None] * re / (rhoa_cgs[:, None, None] * dmax)
    return vf, re, bpm


def setup_vf_heymsfield2010(t, rhoa, zmet, rmu, r_wet, rrat, rmass, arat):
    """Convenience wrapper matching the original Fortran signature."""
    return setup_vf_heymsfield2010_jit(
        t, rhoa, zmet, rmu, r_wet, rrat, rmass, arat)
