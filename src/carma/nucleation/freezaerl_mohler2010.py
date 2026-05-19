"""Aerosol → ice nucleation via Möhler et al. (2010).

Computes per-bin ice-nucleation loss rate for sulfate aerosol particles
freezing depositionally / immersionally onto ice. Used by the upper-
troposphere / cirrus regime in CARMA when a group has nucleation
process ``I_AERFREEZE | I_AF_MOHLER_2010`` configured.

Ported from: ``freezaerl_mohler2010.F90``.

Reference: Möhler, O. et al., presented at the AMS Cloud Physics
Workshop (2010). The published parameterisation is:

    log10(j_nuc) = 97.973292 - 154.67476·(s_i+1) - 0.84952712·T
                 + 1.0049467·(s_i+1)·T

with j_nuc in [cm⁻³ s⁻¹], where ``s_i`` is ice supersaturation
(Kelvin-corrected) and ``T`` is in [K]. The per-bin loss rate is then

    rnuclg[bin] = min(1e20, j_nuc · volrat[bin] · vol_dry[bin])

where ``volrat`` is the wet/dry volume ratio of the freezing sulfate
solution droplet, computed from a wt%-vs-water-activity polynomial
(Carslaw 1995 fit, three regimes).

Gates:

- ``T ≤ 240 K`` (parameterisation domain)
- ``pconmax > FEW_PC`` (skip if essentially zero particles)
- ``s_i (Kelvin-corrected) > 0.3`` (Jensen-Toon 1994 critical
  supersaturation for sulfate homogeneous freezing)
"""
import jax.numpy as jnp

from carma.constants import RHO_W, FEW_PC
from carma.precision import DTYPE


_SIFREEZE = DTYPE(0.3)


def freezaerl_mohler2010(
    t_val,
    supsati_val, supsatl_val,
    akelvin_val, akelvini_val,
    r_bins, vol_bins,
    rhosol_val, pconmax_val,
):
    """Compute Möhler 2010 aerosol freezing nucleation rates.

    Args:
        t_val:         Temperature [K], scalar.
        supsati_val:   Ice supersaturation (over flat surface), scalar.
        supsatl_val:   Liquid supersaturation, scalar.
        akelvin_val:   Kelvin factor for liquid water [cm], scalar.
        akelvini_val:  Kelvin factor for ice [cm], scalar.
        r_bins:        Bin centre radii [cm], shape ``(NBIN,)``.
        vol_bins:      Bin dry volumes [cm³], shape ``(NBIN,)``.
        rhosol_val:    Solute density [g/cm³] (e.g. 1.78 for sulfate).
        pconmax_val:   Max particle concentration for this group, scalar.

    Returns:
        rnuclg: per-bin nucleation loss rate [s⁻¹], shape ``(NBIN,)``.
        Caller adds this to the global ``rnuclg[:, igroup, ignucto]``
        accumulator.
    """
    # Per-bin Kelvin-corrected ssi.
    fkelvi = jnp.exp(akelvini_val / r_bins)
    ssi = supsati_val / fkelvi

    # Möhler 2010 nucleation rate [cm⁻³ s⁻¹].
    rlogj = (DTYPE(97.973292)
             - DTYPE(154.67476) * (ssi + DTYPE(1.0))
             - DTYPE(0.84952712) * t_val
             + DTYPE(1.0049467) * (ssi + DTYPE(1.0)) * t_val)
    rjj = DTYPE(10.0) ** rlogj

    # Clip ssl to (-1, 0]: parameterisation breaks down at supersaturated
    # liquid where the wt% polynomial returns negative values.
    ssl_clip = jnp.maximum(DTYPE(-1.0), jnp.minimum(DTYPE(0.0), supsatl_val))

    # Kelvin-corrected water activity (per bin).
    fkelv = jnp.exp(akelvin_val / r_bins)
    aw = (DTYPE(1.0) + ssl_clip) / fkelv

    # H2SO4 wt% vs aw via Carslaw fit, three regimes.
    aw_safe = jnp.maximum(aw, DTYPE(1e-30))    # guard the negative powers
    contl_lo = (DTYPE(12.37208932)  * aw_safe ** DTYPE(-0.16125516114)
                - DTYPE(30.490657554) * aw - DTYPE(2.1133114241))
    conth_lo = (DTYPE(13.455394705) * aw_safe ** DTYPE(-0.1921312255)
                - DTYPE(34.285174604) * aw - DTYPE(1.7620073078))
    contl_md = (DTYPE(11.820654354) * aw_safe ** DTYPE(-0.20786404244)
                - DTYPE(4.807306373) * aw - DTYPE(5.1727540348))
    conth_md = (DTYPE(12.891938068) * aw_safe ** DTYPE(-0.23233847708)
                - DTYPE(6.4261237757) * aw - DTYPE(4.9005471319))
    contl_hi = (DTYPE(-180.06541028) * aw_safe ** DTYPE(-0.38601102592)
                - DTYPE(93.317846778) * aw + DTYPE(273.88132245))
    conth_hi = (DTYPE(-176.95814097) * aw_safe ** DTYPE(-0.36257048154)
                - DTYPE(90.469744201) * aw + DTYPE(267.45509988))

    contl = jnp.where(aw < DTYPE(0.05), contl_lo,
                       jnp.where(aw <= DTYPE(0.85), contl_md, contl_hi))
    conth = jnp.where(aw < DTYPE(0.05), conth_lo,
                       jnp.where(aw <= DTYPE(0.85), conth_md, conth_hi))

    H2SO4m = contl + (conth - contl) * (t_val - DTYPE(190.0)) / DTYPE(70.0)
    WT_raw = (DTYPE(98.0) * H2SO4m) / (DTYPE(1000.0) + DTYPE(98.0) * H2SO4m)
    WT = DTYPE(100.0) * jnp.maximum(DTYPE(0.0), jnp.minimum(DTYPE(1.0), WT_raw))

    # Wet/dry volume ratio. Guard the WT==0 case (sentinel 1e10).
    WT_safe = jnp.maximum(WT, DTYPE(1e-30))
    volrat_pos = (rhosol_val / RHO_W *
                  ((DTYPE(100.0) - WT_safe) / WT_safe) + DTYPE(1.0))
    volrat = jnp.where(WT > DTYPE(0.0), volrat_pos, DTYPE(1e10))

    # Per-bin loss rate, capped at 1e20 for numerical stability (Fortran
    # imposes the same cap).
    nuclg = jnp.minimum(DTYPE(1e20), rjj * volrat * vol_bins)

    # Combined gate: T ≤ 240 K, pconmax > FEW_PC, ssi (Kelvin-corrected)
    # > 0.3. The first two are scalar, the third is per-bin.
    cold_enough = t_val <= DTYPE(240.0)
    has_particles = pconmax_val > FEW_PC
    super_enough = ssi > _SIFREEZE
    gate = cold_enough & has_particles & super_enough

    return jnp.where(gate, nuclg, DTYPE(0.0))
