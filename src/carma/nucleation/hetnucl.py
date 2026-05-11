"""Heterogeneous deposition ice nucleation (Keesee 1989 / Rapp-Thomas 2006).

Ported from `hetnucl.F90`. Computes per-bin loss rate for water vapor
freezing directly onto aerosol cores in the polar mesospheric cloud
(PMC) regime. Classical nucleation theory with a contact-angle
geometric factor (Pruppacher & Klett eq. 9-22).

Pipeline:

    rlogs = log(supsati + 1)
    ag    = 2·g·σ_ia / (R·T·ρ_i·rlogs)        # Rapp-Thomas eq. 2 critical germ
    fh    = ½·(1 + ((1-m·x)/φ)³ + fv3 + fv4)   # P&K 9-22 geometric factor
    ΔF    = 4π·ag²·σ_ia − (4π·ρ_i·ag³·k·T·rlogs) / (3·m_w)
    expon = (2·gdes − gsd − fh·ΔF) / (k·T)
    rate  = min(1e10, zeld·k·T·diflen·ag·sin(θ)·4π·r²·n_h2o² /
                       (fh·m_w·vibfreq) · exp(expon))

Gates:
    p < 1000 dyne/cm² (= 1 hPa, mesospheric only — Fortran note: "only
        trying to model PMC particles, so turn off where the CAM
        microphysics takes over").
    supsati > 0
    pconmax > FEW_PC

The Fortran source pins `rmiv = 0.95` hardcoded; we match. Future
Trainer et al. 2008 temperature-dependent form is commented out
upstream.
"""
import jax.numpy as jnp

from carma.constants import AVG, BK, RGAS, RHO_I, PI, FEW_PC
from carma.precision import DTYPE


# Heterogeneous nucleation factors (hetnucl.F90:63-67)
_GDES    = DTYPE(2.9e-13)    # desorption energy [erg]
_GSD     = DTYPE(2.9e-14)    # surface diffusion energy [erg]
_ZELD    = DTYPE(0.1)        # Zeldovich factor
_VIBFREQ = DTYPE(1.0e13)     # vibration frequency [1/s]
_DIFLEN  = DTYPE(0.1e-7)     # diffusion length [cm]
_RMIV    = DTYPE(0.95)       # cos(contact angle); upstream hardcodes 0.95

# Mesospheric pressure gate
_P_GATE  = DTYPE(1.0e3)      # dyne/cm² = 1 hPa


def hetnucl(t_val, p_val, supsati_val, gc_h2o, gwtmol_h2o,
            surfctia_val, r_bins, pconmax_val):
    """Compute per-bin heterogeneous deposition ice nucleation rate.

    Args:
        t_val:        Temperature [K], scalar.
        p_val:        Pressure [dyne/cm²], scalar.
        supsati_val:  Ice supersaturation, scalar.
        gc_h2o:       H₂O gas mass density [g/cm³], scalar.
        gwtmol_h2o:   Water molecular weight [g/mol], scalar (typ 18.016).
        surfctia_val: Ice-air surface tension [dyne/cm], scalar.
        r_bins:       Bin centre radii [cm], shape ``(NBIN,)``.
        pconmax_val:  Max particle concentration for the group, scalar.

    Returns:
        rnuclg: per-bin nucleation loss rate [s⁻¹], shape ``(NBIN,)``.
    """
    rmw = gwtmol_h2o / AVG
    r_h2o = RGAS / gwtmol_h2o
    rnh2o = gc_h2o * r_h2o / BK

    # log(S+1), with guard for supsati ≤ 0 (will be gated to zero anyway)
    rlogs = jnp.log(jnp.maximum(supsati_val + DTYPE(1.0), DTYPE(1e-30)))

    # Critical germ radius (Rapp-Thomas 2006 eq. 2). Guard the
    # supsati→0 limit where rlogs → 0; gate masks the result.
    ag = (DTYPE(2.0) * gwtmol_h2o * surfctia_val
          / (RGAS * t_val * RHO_I * jnp.maximum(rlogs, DTYPE(1e-300))))

    # P&K 9-22 heterogeneous nucleation geometric factor.
    contang = jnp.arccos(_RMIV)
    x = r_bins / ag           # per-bin
    phi_arg = jnp.maximum(
        DTYPE(1.0) - DTYPE(2.0) * _RMIV * x + x ** 2,
        DTYPE(0.0),
    )
    phih = jnp.sqrt(phi_arg)
    rath = (x - _RMIV) / jnp.maximum(phih, DTYPE(1e-300))

    fv3h_raw = x ** 3 * (DTYPE(2.0) - DTYPE(3.0) * rath + rath ** 3)
    fv4h_raw = DTYPE(3.0) * _RMIV * x ** 2 * (rath - DTYPE(1.0))
    # Fortran zeros these at the boundary where rath ≈ ±1 to avoid
    # cancellation noise (hetnucl.F90:136-137).
    fv3h = jnp.where(jnp.abs(rath) > DTYPE(1.0) - DTYPE(1e-8),
                     DTYPE(0.0), fv3h_raw)
    fv4h = jnp.where(jnp.abs(rath) > DTYPE(1.0) - DTYPE(1e-10),
                     DTYPE(0.0), fv4h_raw)

    fh = DTYPE(0.5) * (
        DTYPE(1.0)
        + ((DTYPE(1.0) - _RMIV * x) / jnp.maximum(phih, DTYPE(1e-300))) ** 3
        + fv3h + fv4h
    )

    # Gibbs free energy of ice germ formation (Rapp-Thomas 2006 eq. 3).
    delfg = (DTYPE(4.0) * PI * ag ** 2 * surfctia_val
             - DTYPE(4.0) * PI * RHO_I * ag ** 3 * BK * t_val * rlogs
               / (DTYPE(3.0) * rmw))

    # Activation exponent
    expon = ((DTYPE(2.0) * _GDES - _GSD - fh * delfg) / (BK * t_val))

    # Per-bin nucleation rate (capped at 1e10 for stability)
    rate_raw = (_ZELD * BK * t_val * _DIFLEN * ag * jnp.sin(contang)
                * DTYPE(4.0) * PI * r_bins ** 2 * rnh2o ** 2
                / (jnp.maximum(fh, DTYPE(1e-300)) * rmw * _VIBFREQ)
                * jnp.exp(expon))
    rate = jnp.minimum(DTYPE(1.0e10), rate_raw)

    # Combined gates: p < 1000 dyne/cm², supsati > 0, pconmax > FEW_PC.
    p_ok = p_val < _P_GATE
    super_ok = supsati_val > DTYPE(0.0)
    has_particles = pconmax_val > FEW_PC
    gate = p_ok & super_ok & has_particles

    return jnp.where(gate, rate, DTYPE(0.0))
