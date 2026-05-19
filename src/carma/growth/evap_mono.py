"""Monodisperse total evaporation into a CN (nucleation-target) group.

Ported from ``evap_mono.F90``. Given a total evaporation event from
one source bin ``(ig, ibin)``, this routine scatters the evaporated
particle number ``evdrop`` and secondary core masses ``evcore[ic]``
into the target CN group ``igto`` + number element ``ieto``.

Three layouts, depending on where the average core mass ``coreavg``
lies relative to the target bin grid:

- ``too_small`` or ``nuc_small``: all mass goes into bin 0 (Fortran
  ``jbin=1``).
- ``too_big``: all mass goes into the top bin (Fortran ``jbin=NBIN``).
- Otherwise: split between ``iavg-1`` and ``iavg`` with weight
  ``fracmass = (rmass[iavg] - coreavg) / diffmass[iavg, iavg-1]``
  so total core mass and total number are both conserved.

This function is a pure single-event update. It returns the
incremental contribution to ``evappe``; callers add it to their
accumulator.
"""

import jax.numpy as jnp

from carma.precision import DTYPE


def evap_mono(
    evdrop,
    evcore,
    coreavg,
    iavg,
    ieto,
    igto,
    too_big,
    too_small,
    nuc_small,
    ncore_ig,
    icorelem,
    ievp2elem,
    ig,
    rmass,
    diffmass,
    nbin,
    nelem,
    conserve_mass=True,
):
    """Return an ``evappe`` delta for one monodisperse total-evap event.

    Args:
        evdrop: Evaporated particle-number rate [#/z/s], scalar.
        evcore: Per-core evap rates, shape ``(ncore_max,)``; index
            ``ic >= 1`` holds ``evdrop · pc[iecore] / coretot``.
        coreavg: Average core mass per source particle [g].
        iavg: Target bin (0-based). Valid only when neither
            ``too_big`` nor ``too_small`` nor ``nuc_small``.
        ieto: Target group's number-concentration element index.
        igto: Target group index.
        too_big, too_small, nuc_small: Flags from the dispatcher
            (Python bools).
        ncore_ig: ``ncore[ig]`` — number of core elements in source
            group. Python int (static).
        icorelem: ``(ncore_max, ngroup)`` int (static).
        ievp2elem: ``(nelem,)`` — element that receives evap-core mass
            when core index ic evaporates. ``-1`` (Fortran 0) = skip.
        ig: Source group index (int).
        rmass: ``(nbin, ngroup)`` bin mass.
        diffmass: ``(nbin, ngroup, nbin, ngroup)`` — ``rmass[i2, g2] -
            rmass[i, g]``.
        nbin, nelem: Static dimensions (Python ints).
        conserve_mass: If True (default), scale the number added by
            ``coreavg / rmass[jbin, igto]`` in the boundary cases so
            total core mass is preserved. Matches Fortran default.

    Returns:
        ``evappe_delta`` of shape ``(nbin, nelem)``. Add to caller's
        accumulator.
    """
    evappe_delta = jnp.zeros((nbin, nelem), dtype=DTYPE)

    boundary = bool(too_big) or bool(too_small) or bool(nuc_small)

    if boundary:
        jbin = nbin - 1 if too_big else 0
        if conserve_mass:
            factor = coreavg / rmass[jbin, igto]
        else:
            factor = DTYPE(1.0)

        evappe_delta = evappe_delta.at[jbin, ieto].add(factor * evdrop)

        for ic in range(1, ncore_ig):
            ie2cn = int(ievp2elem[int(icorelem[ic, ig])])
            if ie2cn < 0:
                continue
            evappe_delta = evappe_delta.at[jbin, ie2cn].add(
                factor * evcore[ic] * rmass[jbin, igto]
            )
        return evappe_delta

    # Interior case: split between iavg-1 and iavg.
    # fracmass = (rmass[iavg] - coreavg) / (rmass[iavg] - rmass[iavg-1])
    denom = diffmass[iavg, igto, iavg - 1, igto]
    fracmass = (rmass[iavg, igto] - coreavg) / denom

    evappe_delta = evappe_delta.at[iavg - 1, ieto].add(evdrop * fracmass)
    evappe_delta = evappe_delta.at[iavg, ieto].add(
        evdrop * (DTYPE(1.0) - fracmass)
    )

    for ic in range(1, ncore_ig):
        ie2cn = int(ievp2elem[int(icorelem[ic, ig])])
        if ie2cn < 0:
            continue
        evappe_delta = evappe_delta.at[iavg - 1, ie2cn].add(
            rmass[iavg - 1, igto] * evcore[ic] * fracmass
        )
        evappe_delta = evappe_delta.at[iavg, ie2cn].add(
            rmass[iavg, igto] * evcore[ic] * (DTYPE(1.0) - fracmass)
        )

    return evappe_delta
