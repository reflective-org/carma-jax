"""Polydisperse total evaporation into a CN (nucleation-target) group.

Ported from ``evap_poly.F90``. When the evaporating source-group cores
have non-trivial size spread (``coresig > sig_mono``), the CN cores are
distributed across **all** target bins using a log-normal core-mass PDF
skewed by ``m⁻³ᐟ²`` (see Turco, NASA Technical Paper 1362, for the
derivation; the -3/2 skew guarantees the pdf mean matches ``coreavg``).

The distribution is normalised piecewise — bins below ``iavg`` and
bins at/above ``iavg`` get renormalised separately so that both the
total core number and total core mass are conserved exactly. The
Fortran code uses empty-side weights ``weightl, weights`` to handle
the case where the PDF tail escapes past the grid.

Returns an ``evappe`` delta that the caller adds to its accumulator.
"""

import jax.numpy as jnp

from carma.precision import DTYPE


_POWMAX = DTYPE(700.0)           # max exponent before exp underflow/overflow
_ALMOST_ONE = DTYPE(1.0 - 1e-12)
_ALMOST_ZERO = DTYPE(1e-12)


def evap_poly(
    evdrop,
    evcore,
    coreavg,
    coresig,
    iavg,
    ieto,
    igto,
    ncore_ig,
    icorelem,
    ievp2elem,
    ig,
    rmass,
    dm,
    nbin,
    nelem,
):
    """Return an ``evappe`` delta for one polydisperse total-evap event.

    Args:
        evdrop: Evaporated particle-number rate [#/z/s].
        evcore: Per-core evap rates, shape ``(ncore_max,)``.
        coreavg: Average core mass per source particle [g].
        coresig: ``log(smf / cmf²)`` — log of the core-mass variance.
            Assumed ``> 0`` by the dispatcher that calls this.
        iavg: Target bin that straddles the core-mass mean (0-based).
        ieto: Target number-element index.
        igto: Target group index.
        ncore_ig: ``ncore[ig]`` (Python int).
        icorelem: ``(ncore_max, ngroup)`` int (static).
        ievp2elem: ``(nelem,)`` — evap-core destination element.
        ig: Source group index.
        rmass: ``(nbin, ngroup)`` bin mass.
        dm: ``(nbin, ngroup)`` bin-width in mass.
        nbin, nelem: Static dimensions.

    Returns:
        ``evappe_delta`` of shape ``(nbin, nelem)``.
    """
    rmass_to = rmass[:, igto]                              # (nbin,)
    dm_to = dm[:, igto]                                    # (nbin,)

    coreavg = jnp.asarray(coreavg, dtype=DTYPE)
    coresig = jnp.asarray(coresig, dtype=DTYPE)

    valid = (coreavg > DTYPE(0.0)) & (coresig > DTYPE(0.0))
    safe_coreavg = jnp.where(valid, coreavg, DTYPE(1.0))
    safe_coresig = jnp.where(valid, coresig, DTYPE(1.0))

    # prob ∝ rmass⁻³ᐟ² · exp(-ln²(rmass/coreavg) / (2 coresig))
    expon = -jnp.log(rmass_to / safe_coreavg) ** 2 / (
        DTYPE(2.0) * safe_coresig
    )
    expon = jnp.where(valid, expon, DTYPE(0.0))
    expon = jnp.maximum(expon, -_POWMAX)
    prob = rmass_to ** DTYPE(-1.5) * jnp.exp(expon)

    # Split bins around iavg (Fortran uses ito < iavg)
    ito_idx = jnp.arange(nbin)
    is_small = ito_idx < iavg
    is_large = ~is_small

    pdm = prob * dm_to
    rn_norms = jnp.sum(jnp.where(is_small, pdm, DTYPE(0.0)))
    rn_norml = jnp.sum(jnp.where(is_large, pdm, DTYPE(0.0)))
    rm_norms_raw = jnp.sum(jnp.where(is_small, pdm * rmass_to, DTYPE(0.0)))
    rm_norml_raw = jnp.sum(jnp.where(is_large, pdm * rmass_to, DTYPE(0.0)))

    kount_s = jnp.sum(is_small.astype(jnp.int32))
    kount_l = jnp.sum(is_large.astype(jnp.int32))

    # Weight computation — Fortran distinguishes three empty / non-empty
    # cases. Replicated here with jnp.where.
    has_small = kount_s > 0
    has_large = kount_l > 0

    safe_rns = jnp.where(rn_norms > DTYPE(0.0), rn_norms, DTYPE(1.0))
    safe_rnl = jnp.where(rn_norml > DTYPE(0.0), rn_norml, DTYPE(1.0))
    rm_norms_mean = rm_norms_raw / safe_rns
    rm_norml_mean = rm_norml_raw / safe_rnl

    weightl_both = (coreavg - rm_norms_mean) / (
        jnp.where(
            jnp.abs(rm_norml_mean - rm_norms_mean) > DTYPE(1e-300),
            rm_norml_mean - rm_norms_mean,
            DTYPE(1.0),
        )
    )
    weightl_both = jnp.clip(weightl_both, DTYPE(0.0), _ALMOST_ONE)
    weightl_both = jnp.where(weightl_both > _ALMOST_ONE, DTYPE(1.0),
                              weightl_both)
    weightl_both = jnp.where(weightl_both < _ALMOST_ZERO, DTYPE(0.0),
                              weightl_both)

    weightl = jnp.where(
        ~has_small, DTYPE(1.0),
        jnp.where(~has_large, DTYPE(0.0), weightl_both),
    )
    weights = DTYPE(1.0) - weightl

    # Renormalise prob and build number-production term
    prob_small = prob * weights / jnp.where(rn_norms > DTYPE(0.0),
                                             rn_norms, DTYPE(1.0))
    prob_large = prob * weightl / jnp.where(rn_norml > DTYPE(0.0),
                                             rn_norml, DTYPE(1.0))
    prob_renorm = jnp.where(is_small, prob_small, prob_large)

    num_term = evdrop * prob_renorm * dm_to                  # (nbin,)

    evappe_delta = jnp.zeros((nbin, nelem), dtype=DTYPE)
    evappe_delta = evappe_delta.at[:, ieto].add(num_term)

    # Core elements
    for ic in range(1, ncore_ig):
        ie2cn = int(ievp2elem[int(icorelem[ic, ig])])
        if ie2cn < 0:
            continue
        core_term = rmass_to * evcore[ic] * prob_renorm * dm_to  # (nbin,)
        evappe_delta = evappe_delta.at[:, ie2cn].add(core_term)

    # If the dispatcher's validity conditions are violated (coreavg<=0
    # or coresig<=0) we should produce no contribution.
    evappe_delta = jnp.where(valid, evappe_delta, DTYPE(0.0))
    return evappe_delta
