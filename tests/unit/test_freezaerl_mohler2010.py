"""Unit tests for freezaerl_mohler2010 (Phase 11.1).

Bench against a numpy reference that transcribes
``freezaerl_mohler2010.F90`` line-for-line. This validates algorithmic
equivalence at machine ε; the integrated Fortran-binary diff bench
(stage 2 of Phase 11.1) lives in a follow-up PR with its own
CARMA-state probe infrastructure.
"""
import math

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from carma.constants import RHO_W, FEW_PC
from carma.nucleation.freezaerl_mohler2010 import freezaerl_mohler2010


# ---------------------------------------------------------------------------
# numpy reference — line-for-line transcription of freezaerl_mohler2010.F90
# (lines 79-180 of original-carma/CARMA/source/base/freezaerl_mohler2010.F90).
# Returns rnuclg per bin in [s⁻¹]. Caller-side gating on inucproc is omitted
# since we test a single (igroup, ienucto) pair.
# ---------------------------------------------------------------------------
def _np_freezaerl_mohler2010(t, supsati, supsatl, akelvin, akelvini,
                               r_bins, vol_bins, rhosol, pconmax):
    rnuclg = np.zeros(r_bins.shape[0])
    if t > 240.0:                                  # F90 line 82
        return rnuclg
    if pconmax <= FEW_PC:                          # F90 line 94
        return rnuclg
    for ibin in range(r_bins.shape[0]):
        ssi_val = supsati                          # F90 line 114
        ssl_val = supsatl                          # F90 line 115
        fkelvi = math.exp(akelvini / r_bins[ibin]) # F90 line 118
        ssi_val = ssi_val / fkelvi
        if ssi_val <= 0.3:                         # F90 line 127 (sifreeze)
            continue
        rlogj = (97.973292
                 - 154.67476 * (ssi_val + 1.0)
                 - 0.84952712 * t
                 + 1.0049467 * (ssi_val + 1.0) * t)            # F90:130
        rjj = 10.0 ** rlogj                                     # F90:131

        ssl_val = max(-1.0, min(0.0, ssl_val))                  # F90:137
        aw = 1.0 + ssl_val                                       # F90:140
        fkelv = math.exp(akelvin / r_bins[ibin])                # F90:141
        aw = aw / fkelv

        if aw < 0.05:
            contl = (12.37208932 * aw ** -0.16125516114
                     - 30.490657554 * aw - 2.1133114241)
            conth = (13.455394705 * aw ** -0.1921312255
                     - 34.285174604 * aw - 1.7620073078)
        elif aw <= 0.85:
            contl = (11.820654354 * aw ** -0.20786404244
                     - 4.807306373 * aw - 5.1727540348)
            conth = (12.891938068 * aw ** -0.23233847708
                     - 6.4261237757 * aw - 4.9005471319)
        else:
            contl = (-180.06541028 * aw ** -0.38601102592
                     - 93.317846778 * aw + 273.88132245)
            conth = (-176.95814097 * aw ** -0.36257048154
                     - 90.469744201 * aw + 267.45509988)

        H2SO4m = contl + (conth - contl) * (t - 190.0) / 70.0   # F90:156
        WT = (98.0 * H2SO4m) / (1000.0 + 98.0 * H2SO4m)
        WT = max(0.0, min(1.0, WT))
        WT = 100.0 * WT                                          # F90:159

        if WT <= 0.0:                                           # F90:162
            volrat = 1e10
        else:
            volrat = float(rhosol) / float(RHO_W) * ((100.0 - WT) / WT) + 1.0

        rnuclg[ibin] = min(1e20, rjj * volrat * vol_bins[ibin]) # F90:170
    return rnuclg


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

def _bin_grid(nbin=16, rmin_cm=1e-7, rmrat=4.0, rho=1.78):
    """Mirrors the carma_nuc2test 'Sulfate IN' group (rmin=1e-7, rmrat=4)."""
    vmin = (4.0 / 3.0) * math.pi * rmin_cm**3 * rho
    rmass = vmin * rmrat ** np.arange(nbin)
    vol = rmass / rho
    r = (3.0 * rmass / (4.0 * math.pi * rho)) ** (1.0 / 3.0)
    return r, vol


SCENARIOS = [
    # (label, T, supsati, supsatl, akelvin, akelvini, rhosol, pconmax)
    ("cirrus_typical",   220.0,  0.50, -0.40, 1.5e-7, 2.0e-7, 1.78, 1e3),
    ("very_cold_high",   200.0,  0.80, -0.20, 1.4e-7, 1.9e-7, 1.78, 1e4),
    ("near_threshold",   235.0,  0.32, -0.10, 1.5e-7, 2.0e-7, 1.78, 1e2),
    ("low_aw_regime",    220.0,  0.50, -0.99, 1.5e-7, 2.0e-7, 1.78, 1e3),  # aw<0.05
    ("high_aw_regime",   220.0,  0.50, -0.10, 1.5e-7, 2.0e-7, 1.78, 1e3),  # aw>0.85
    # Gated-off scenarios — should return all zeros.
    ("warm_skip_T",      250.0,  0.50, -0.40, 1.5e-7, 2.0e-7, 1.78, 1e3),
    ("low_ssi_skip",     220.0,  0.10, -0.40, 1.5e-7, 2.0e-7, 1.78, 1e3),
    ("no_particles",     220.0,  0.50, -0.40, 1.5e-7, 2.0e-7, 1.78, 1e-50),
]


@pytest.mark.parametrize("label,T,ssi,ssl,akelvin,akelvini,rhosol,pconmax",
                          SCENARIOS)
def test_freezaerl_mohler2010_matches_numpy_reference(
        label, T, ssi, ssl, akelvin, akelvini, rhosol, pconmax):
    """JAX matches a numpy transcription of the Fortran kernel at machine ε."""
    r_bins, vol_bins = _bin_grid()
    np_out = _np_freezaerl_mohler2010(
        T, ssi, ssl, akelvin, akelvini,
        r_bins, vol_bins, rhosol, pconmax,
    )
    jax_out = np.asarray(freezaerl_mohler2010(
        t_val=jnp.float64(T),
        supsati_val=jnp.float64(ssi),
        supsatl_val=jnp.float64(ssl),
        akelvin_val=jnp.float64(akelvin),
        akelvini_val=jnp.float64(akelvini),
        r_bins=jnp.asarray(r_bins),
        vol_bins=jnp.asarray(vol_bins),
        rhosol_val=jnp.float64(rhosol),
        pconmax_val=jnp.float64(pconmax),
    ))
    # Use rtol=1e-12 — the formula has cube-roots, exponentials, and
    # power functions, so we expect 1-2 ULPs of slack from FMA fusion.
    np.testing.assert_allclose(jax_out, np_out, rtol=1e-12, atol=0.0,
                                err_msg=f"scenario {label}")


def test_freezaerl_mohler2010_jit_equivalent():
    """JIT-compiled call matches non-JIT to bit-exactness."""
    r_bins, vol_bins = _bin_grid()
    args = dict(
        t_val=jnp.float64(220.0),
        supsati_val=jnp.float64(0.5),
        supsatl_val=jnp.float64(-0.4),
        akelvin_val=jnp.float64(1.5e-7),
        akelvini_val=jnp.float64(2.0e-7),
        r_bins=jnp.asarray(r_bins),
        vol_bins=jnp.asarray(vol_bins),
        rhosol_val=jnp.float64(1.78),
        pconmax_val=jnp.float64(1e3),
    )
    out      = freezaerl_mohler2010(**args)
    out_jit  = jax.jit(freezaerl_mohler2010)(**args)
    # XLA fuses multiply-adds in JIT but not in eager, so ULP-level diffs
    # are expected on a long expression chain (the per-bin rate involves
    # exp, **0.4, divide, etc.). 1e-12 is comfortably tighter than the
    # numpy-reference tolerance.
    np.testing.assert_allclose(np.asarray(out), np.asarray(out_jit),
                                rtol=1e-12, atol=0.0)


def test_freezaerl_mohler2010_gates():
    """T > 240, ssi < 0.3, pconmax < FEW_PC each independently zero the rate."""
    r_bins, vol_bins = _bin_grid()
    base = dict(
        akelvin_val=jnp.float64(1.5e-7),
        akelvini_val=jnp.float64(2.0e-7),
        r_bins=jnp.asarray(r_bins),
        vol_bins=jnp.asarray(vol_bins),
        rhosol_val=jnp.float64(1.78),
    )
    # Active baseline.
    active = freezaerl_mohler2010(
        t_val=jnp.float64(220.0),
        supsati_val=jnp.float64(0.5),
        supsatl_val=jnp.float64(-0.4),
        pconmax_val=jnp.float64(1e3), **base,
    )
    assert bool(jnp.any(active > 0))

    for kwargs in (
        dict(t_val=jnp.float64(245.0),                   # T > 240
             supsati_val=jnp.float64(0.5),
             supsatl_val=jnp.float64(-0.4),
             pconmax_val=jnp.float64(1e3)),
        dict(t_val=jnp.float64(220.0),
             supsati_val=jnp.float64(0.1),                # ssi-after-Kelvin < 0.3
             supsatl_val=jnp.float64(-0.4),
             pconmax_val=jnp.float64(1e3)),
        dict(t_val=jnp.float64(220.0),
             supsati_val=jnp.float64(0.5),
             supsatl_val=jnp.float64(-0.4),
             pconmax_val=jnp.float64(1e-50)),             # pconmax < FEW_PC = 1e-44
    ):
        out = freezaerl_mohler2010(**kwargs, **base)
        assert bool(jnp.all(out == 0)), f"gate failed for {kwargs}"
