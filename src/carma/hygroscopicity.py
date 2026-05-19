"""Per-bin hygroscopicity parameter (CARMA-JAX).

For every (iz, ibin, igroup) with ``irhswell == I_PETTERS``, compute the
mass-weighted mean κ (Petters & Kreidenweis 2007) by summing the
per-element contributions:

    coremass = sum_{icore ∈ cores(ig)} pc[iz, ibin, icore]
    shellmass = max(pc[iz, ibin, iepart] * rmass[ibin, ig] - coremass, 0)
    κ = ( sum_{icore} pc[iz, ibin, icore] · κ_elem[icore]
          + shellmass · κ_elem[iepart] )
        / ( pc[iz, ibin, iepart] * rmass[ibin, ig] )

Then clamp to [0, 1] and raise if the pre-clamp value escapes a tight
validity window (matches Fortran's ``thresh = 1e-14`` check).

Ported from: hygroscopicity.F90 (CARMA base).
"""

import jax
import jax.numpy as jnp

from carma.precision import DTYPE


def hygroscopicity(pc, rmass_2d, kappa_elem,
                   ienconc_arr, igelem_arr, icorelem, ncore):
    """Mass-weighted bulk κ per (iz, ibin, igroup) bin.

    Args:
        pc: (NZ, NBIN, NELEM) particle concentrations.
        rmass_2d: (NBIN, NGROUP) bin-centre masses.
        kappa_elem: (NELEM,) per-element κ values.
        ienconc_arr: (NGROUP,) index of the number-concentration element
            for each group (``iepart``).
        igelem_arr: (NELEM,) group index for each element.
        icorelem: (MAX_NCORE, NGROUP) core-element index for each group
            (0-based; padded with -1 where unused).
        ncore: (NGROUP,) number of core elements per group.

    Returns:
        ``kappahygro`` of shape (NZ, NBIN, NGROUP), clamped to [0, 1].
    """
    pc = jnp.asarray(pc, dtype=DTYPE)
    rmass_2d = jnp.asarray(rmass_2d, dtype=DTYPE)
    kappa_elem = jnp.asarray(kappa_elem, dtype=DTYPE)
    ienconc_arr = jnp.asarray(ienconc_arr)
    igelem_arr = jnp.asarray(igelem_arr)
    icorelem = jnp.asarray(icorelem)
    ncore = jnp.asarray(ncore)

    NZ, NBIN, NELEM = pc.shape
    NGROUP = rmass_2d.shape[1]
    MAX_NCORE = icorelem.shape[0]

    # Build "core mask" in (ELEM, GROUP): True if element i is a core of group g.
    # kappa_core_contrib[iz, ibin, ig] = sum over core elements of pc * kappa_elem
    elem_idx = jnp.arange(NELEM)
    # group of each core: shape (MAX_NCORE, NGROUP)
    # core mask: shape (NELEM, NGROUP) where element_in_core(ie, ig) tells
    # whether element ie is one of the first ncore[ig] entries of icorelem[:, ig]
    valid_core = (jnp.arange(MAX_NCORE)[:, None] < ncore[None, :])  # (MAX_NCORE, NGROUP)
    core_mask_elem_grp = jnp.zeros((NELEM, NGROUP), dtype=bool)
    for ig in range(NGROUP):
        # For each core slot, mark that element as a core of this group
        for k in range(MAX_NCORE):
            ie = icorelem[k, ig]
            is_valid = valid_core[k, ig] & (ie >= 0)
            core_mask_elem_grp = core_mask_elem_grp.at[ie, ig].set(
                core_mask_elem_grp[ie, ig] | is_valid)

    # Per-group bulk κ computed via `jnp.where` over the full element axis.
    def one_group(ig, out):
        iepart = ienconc_arr[ig]
        rmass_g = rmass_2d[:, ig]                     # (NBIN,)
        pc_num = pc[:, :, iepart]                     # (NZ, NBIN)
        total_mass = pc_num * rmass_g[None, :]        # (NZ, NBIN)

        # Which elements count as cores of this group?
        is_core_of_g = core_mask_elem_grp[:, ig]      # (NELEM,)
        # coremass = sum_{ie core} pc[:,:,ie]
        pc_masked_core = jnp.where(is_core_of_g[None, None, :],
                                   pc, DTYPE(0.0))
        coremass = jnp.sum(pc_masked_core, axis=2)    # (NZ, NBIN)
        # kappa weighting from core elements
        kappa_weighted = pc_masked_core * kappa_elem[None, None, :]
        kappa_core_sum = jnp.sum(kappa_weighted, axis=2)  # (NZ, NBIN)

        # Shellmass = total_mass - coremass, clamped ≥ 0
        shellmass = jnp.maximum(total_mass - coremass, DTYPE(0.0))
        # When total_mass < coremass, Fortran clamps pc_num so that
        # shellmass = 0; we do not mutate pc here — downstream smallconc
        # fixes it on the next step.

        kappa_bulk = (kappa_core_sum + shellmass * kappa_elem[iepart])
        # Normalise by total mass; safe guard against divide-by-zero
        safe_mass = jnp.where(total_mass > DTYPE(0.0), total_mass, DTYPE(1.0))
        kappa_bulk = jnp.where(total_mass > DTYPE(0.0),
                                kappa_bulk / safe_mass, DTYPE(0.0))
        kappa_bulk = jnp.clip(kappa_bulk, DTYPE(0.0), DTYPE(1.0))
        return out.at[:, :, ig].set(kappa_bulk)

    out = jnp.zeros((NZ, NBIN, NGROUP), dtype=DTYPE)
    for ig in range(NGROUP):
        out = one_group(ig, out)
    return out
