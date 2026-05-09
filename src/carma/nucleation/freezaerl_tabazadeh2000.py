"""Aerosol → ice nucleation via Tabazadeh et al. (2000).

Computes per-bin ice-nucleation loss rate for sulfate aerosol particles
freezing depositionally / immersionally onto ice. Used by the
upper-troposphere / cirrus regime in CARMA when a group has nucleation
process ``I_AERFREEZE | I_AF_TABAZADEH_2000`` configured.

Ported from: ``freezaerl_tabazadeh2000.F90``.

Reference: Tabazadeh, A. et al., 2000, GRL 27, 1111. Classical
nucleation theory with empirical fits for activation energy
(Koop lab data), sulfate solution density (Myhre 1998), and
surface tension at the H₂SO₄/H₂O–ice interface.

Pipeline (per bin):

    1. Kelvin-corrected water activity a_w (from ssl).
    2. Carslaw 1995 H₂SO₄ wt% polynomial (3 a_w regimes).
    3. Sulfate solution density via Myhre 1998 (10-term wt% polynomial).
    4. Activation energy via Koop lab data (split T < 220 K vs ≥).
    5. Surface tension at sulfate-solution / ice interface
       (interpolated between S260, S220, S180).
    6. Critical ice germ radius (classical nucleation theory).
    7. Gibbs free energy of germ formation.
    8. Nucleation rate ∝ √(σT) · v_rat · v_dry · exp((-Ea − ΔF)/kT).

Gates:

- ``pconmax > FEW_PC`` (skip if essentially zero particles)
- ``supsati > 0.3`` (Jensen-Toon 1994 critical supersaturation)
- Negative rates clamped to zero (parameterisation can return negative
  values at high temperatures, mirrored from Fortran).
"""
import jax.numpy as jnp

from carma.constants import RHO_W, RHO_I, PI, BK, RGAS, T0, FEW_PC
from carma.precision import DTYPE, ONE


_SIFREEZE = DTYPE(0.3)
_PRENUC = DTYPE(2.075e33) * RHO_W / RHO_I
_SIGICEA = DTYPE(105.0)


def _wtpct_from_activity(act):
    """Carslaw 1995 H₂SO₄ wt% from water activity (3 regimes).

    Same polynomial as Möhler 2010 — kept inline here rather than
    factored out so each kernel can be read against its own Fortran
    source independently.
    """
    act_safe = jnp.maximum(act, DTYPE(1e-30))   # guard negative powers
    contl_lo = (DTYPE(12.37208932)  * act_safe ** DTYPE(-0.16125516114)
                - DTYPE(30.490657554) * act - DTYPE(2.1133114241))
    conth_lo = (DTYPE(13.455394705) * act_safe ** DTYPE(-0.1921312255)
                - DTYPE(34.285174604) * act - DTYPE(1.7620073078))
    contl_md = (DTYPE(11.820654354) * act_safe ** DTYPE(-0.20786404244)
                - DTYPE(4.807306373) * act - DTYPE(5.1727540348))
    conth_md = (DTYPE(12.891938068) * act_safe ** DTYPE(-0.23233847708)
                - DTYPE(6.4261237757) * act - DTYPE(4.9005471319))
    contl_hi = (DTYPE(-180.06541028) * act_safe ** DTYPE(-0.38601102592)
                - DTYPE(93.317846778) * act + DTYPE(273.88132245))
    conth_hi = (DTYPE(-176.95814097) * act_safe ** DTYPE(-0.36257048154)
                - DTYPE(90.469744201) * act + DTYPE(267.45509988))
    contl = jnp.where(act < DTYPE(0.05), contl_lo,
                       jnp.where(act <= DTYPE(0.85), contl_md, contl_hi))
    conth = jnp.where(act < DTYPE(0.05), conth_lo,
                       jnp.where(act <= DTYPE(0.85), conth_md, conth_hi))
    return contl, conth


def freezaerl_tabazadeh2000(
    t_val, supsati_val, supsatl_val,
    akelvin_val,
    r_bins, vol_bins,
    rhosol_val, gwtmol_val,
    pconmax_val,
):
    """Compute Tabazadeh 2000 aerosol freezing nucleation rates.

    Args:
        t_val:         Temperature [K], scalar.
        supsati_val:   Ice supersaturation, scalar.
        supsatl_val:   Liquid supersaturation, scalar.
        akelvin_val:   Kelvin factor for liquid water [cm], scalar.
        r_bins:        Bin centre radii [cm], shape ``(NBIN,)``.
        vol_bins:      Bin dry volumes [cm³], shape ``(NBIN,)``.
        rhosol_val:    Solute density [g/cm³] (e.g. 1.78 for sulfate).
        gwtmol_val:    Water molecular weight [g/mol] (typically 18.0).
        pconmax_val:   Max particle concentration for this group, scalar.

    Returns:
        rnuclg: per-bin nucleation loss rate [s⁻¹], shape ``(NBIN,)``.
    """
    # Mean ice density and latent-heat-of-fusion integrals over [T0, T].
    # Singular at t_val = T0; mask later via the ssi/pconmax gate.
    dT = t_val - T0
    dT_safe = jnp.where(jnp.abs(dT) > DTYPE(1e-30), dT, DTYPE(1e-30))
    rhoibar = ((DTYPE(0.916) * dT
                - DTYPE(1.75e-4) / DTYPE(2.0) * dT ** 2
                - DTYPE(5.0e-7) * dT ** 3 / DTYPE(3.0))
               / dT_safe)
    rlhbar = ((DTYPE(79.7) * dT
               + DTYPE(0.485) / DTYPE(2.0) * dT ** 2
               - DTYPE(2.5e-3) / DTYPE(3.0) * dT ** 3)
              / dT_safe * DTYPE(4.186e7) * DTYPE(18.0))

    # Activity from clamped ssl, Kelvin-corrected per bin.
    ssl_clip = jnp.maximum(DTYPE(-1.0), jnp.minimum(DTYPE(0.0), supsatl_val))
    fkelv = jnp.exp(akelvin_val / r_bins)
    act_clip = jnp.minimum(ONE, ssl_clip + DTYPE(1.0))   # min(1, ssl+1)
    act = act_clip / fkelv

    # H₂SO₄ wt% via Carslaw polynomial.
    contl, conth = _wtpct_from_activity(act)
    H2SO4m = contl + (conth - contl) * (t_val - DTYPE(190.0)) / DTYPE(70.0)
    WT_raw = (DTYPE(98.0) * H2SO4m) / (DTYPE(1000.0) + DTYPE(98.0) * H2SO4m)
    WT = DTYPE(100.0) * WT_raw                              # 0..100 %
    WT_safe = jnp.maximum(WT, DTYPE(1e-30))

    # Wet/dry volume ratio.
    vrat = rhosol_val / RHO_W * ((DTYPE(100.0) - WT_safe) / WT_safe) + DTYPE(1.0)

    # Sulfate solution density (Myhre 1998), 10-term wt% polynomial.
    wtfrac = WT / DTYPE(100.0)
    C1 = t_val - DTYPE(273.15)
    C2 = C1 ** 2
    C3 = C1 ** 3
    C4 = C1 ** 4
    A0  = DTYPE(999.8426)    + DTYPE(334.5402e-4) * C1 - DTYPE(569.1304e-5) * C2
    A1  = DTYPE(547.2659)    - DTYPE(530.0445e-2) * C1 + DTYPE(118.7671e-4) * C2 + DTYPE(599.0008e-6) * C3
    A2  = DTYPE(526.295e+1)  + DTYPE(372.0445e-1) * C1 + DTYPE(120.1909e-3) * C2 - DTYPE(414.8594e-5) * C3 + DTYPE(119.7973e-7) * C4
    A3  = DTYPE(-621.3958e+2) - DTYPE(287.7670)    * C1 - DTYPE(406.4638e-3) * C2 + DTYPE(111.9488e-4) * C3 + DTYPE(360.7768e-7) * C4
    A4  = DTYPE(409.0293e+3) + DTYPE(127.0854e+1) * C1 + DTYPE(326.9710e-3) * C2 - DTYPE(137.7435e-4) * C3 - DTYPE(263.3585e-7) * C4
    A5  = DTYPE(-159.6989e+4) - DTYPE(306.2836e+1) * C1 + DTYPE(136.6499e-3) * C2 + DTYPE(637.3031e-5) * C3
    A6  = DTYPE(385.7411e+4) + DTYPE(408.3717e+1) * C1 - DTYPE(192.7785e-3) * C2
    A7  = DTYPE(-580.8064e+4) - DTYPE(284.4401e+1) * C1
    A8  = DTYPE(530.1976e+4) + DTYPE(809.1053)    * C1
    A9  = DTYPE(-268.2616e+4)
    A10 = DTYPE(576.4288e+3)
    den = (A0
           + wtfrac    * A1
           + wtfrac**2 * A2
           + wtfrac**3 * A3
           + wtfrac**4 * A4
           + wtfrac**5 * A5
           + wtfrac**6 * A6
           + wtfrac**7 * A7
           + wtfrac**8 * A8
           + wtfrac**9 * A9
           + wtfrac**10 * A10)

    # Activation energy (Koop lab data); two T regimes.
    A0_hi = DTYPE(104525.93058)
    A1_hi = DTYPE(-1103.7644651)
    A2_hi = DTYPE(1.070332702)
    A3_hi = DTYPE(0.017386254322)
    A4_hi = DTYPE(-1.5506854268e-06)
    A5_hi = DTYPE(-3.2661912497e-07)
    A6_hi = DTYPE(6.467954459e-10)
    A0_lo = DTYPE(-17459.516183)
    A1_lo = DTYPE(458.45827551)
    A2_lo = DTYPE(-4.8492831317)
    A3_lo = DTYPE(0.026003658878)
    A4_lo = DTYPE(-7.1991577798e-05)
    A5_lo = DTYPE(8.9049094618e-08)
    A6_lo = DTYPE(-2.4932257419e-11)
    is_hi = t_val > DTYPE(220.0)
    Aa = jnp.where(is_hi, A0_hi, A0_lo)
    Ab = jnp.where(is_hi, A1_hi, A1_lo)
    Ac = jnp.where(is_hi, A2_hi, A2_lo)
    Ad = jnp.where(is_hi, A3_hi, A3_lo)
    Ae = jnp.where(is_hi, A4_hi, A4_lo)
    Af = jnp.where(is_hi, A5_hi, A5_lo)
    Ag = jnp.where(is_hi, A6_hi, A6_lo)
    diffact = ((Aa
                + Ab * t_val
                + Ac * t_val ** 2
                + Ad * t_val ** 3
                + Ae * t_val ** 4
                + Af * t_val ** 5
                + Ag * t_val ** 6) * DTYPE(1.0e-13))

    # Surface tension at sulfate-solution side: interpolate S260, S220, S180.
    c0, c1_, c2_, c3_, c4_, c5_ = (DTYPE(77.40682664), DTYPE(-0.006963123274),
                                     DTYPE(-0.009682499074), DTYPE(0.00088797988),
                                     DTYPE(-2.384669516e-05), DTYPE(2.095358048e-07))
    S260 = c0 + c1_*WT + c2_*WT**2 + c3_*WT**3 + c4_*WT**4 + c5_*WT**5
    d0, d1_, d2_, d3_, d4_, d5_ = (DTYPE(82.01197792),  DTYPE(0.5312072092),
                                     DTYPE(-0.1050692123), DTYPE(0.005415260617),
                                     DTYPE(-0.0001145573827), DTYPE(8.969257061e-07))
    S220 = d0 + d1_*WT + d2_*WT**2 + d3_*WT**3 + d4_*WT**4 + d5_*WT**5
    e0, e1_, e2_, e3_, e4_, e5_ = (DTYPE(85.75507114),  DTYPE(0.09541966318),
                                     DTYPE(-0.1103647657), DTYPE(0.007485866933),
                                     DTYPE(-0.0001912224154), DTYPE(1.736789787e-06))
    S180 = e0 + e1_*WT + e2_*WT**2 + e3_*WT**3 + e4_*WT**4 + e5_*WT**5
    sigma_hi = S260 + (DTYPE(260.0) - t_val) * (S220 - S260) / DTYPE(40.0)
    sigma_lo = S220 + (DTYPE(220.0) - t_val) * (S180 - S220) / DTYPE(40.0)
    sigsula = jnp.where(t_val >= DTYPE(220.0), sigma_hi, sigma_lo)
    sigsulice = jnp.abs(sigsula - _SIGICEA)

    # Critical ice germ radius (classical nucleation theory).
    log_T0_over_T = jnp.log(T0 / t_val)
    log_ssl_p1 = jnp.log(ssl_clip + DTYPE(1.0) + DTYPE(1e-30))   # ssl_clip ≥ -1
    ag_denom = (rlhbar * rhoibar * log_T0_over_T
                + rhoibar * RGAS * DTYPE(0.5) * (T0 + t_val) * log_ssl_p1)
    ag_denom_safe = jnp.where(jnp.abs(ag_denom) > DTYPE(1e-300),
                                ag_denom, DTYPE(1e-300))
    ag_raw = DTYPE(2.0) * gwtmol_val * sigsulice / ag_denom_safe
    ag = jnp.where(ag_raw < DTYPE(0.0), DTYPE(1e10), ag_raw)

    # Gibbs free energy of germ formation.
    delfg = DTYPE(4.0) / DTYPE(3.0) * PI * sigsulice * ag ** 2

    # Final nucleation rate.
    expon = (-diffact - delfg) / BK / t_val
    expon = jnp.maximum(DTYPE(-100.0) * ONE, expon)
    rate_pos = (_PRENUC
                * jnp.sqrt(sigsulice * t_val)
                * vrat * vol_bins
                * jnp.exp(expon))
    rate_pos = jnp.maximum(DTYPE(0.0), rate_pos)            # clamp negative

    # Gates: pconmax > FEW_PC, supsati > 0.3 (per-scenario), WT > 0.
    has_particles = pconmax_val > FEW_PC
    super_enough = supsati_val > _SIFREEZE
    wt_ok = WT > DTYPE(0.0)
    gate = has_particles & super_enough & wt_ok

    return jnp.where(gate, rate_pos, DTYPE(0.0))
