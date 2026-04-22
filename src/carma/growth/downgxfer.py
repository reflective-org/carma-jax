"""Down-direction nucleation production transfer (``downgxfer``).

Ports ``downgxfer.F90`` — the evaporation-direction complement to
``upgxfer``. For each ``(ibin, ielem)`` it accumulates the nucleation
production rate ``rnucpe[ibin, ielem]`` from every source element
``iefrom`` with ``iefrom > ielem`` (i.e. the source element has a
larger index than the target — the "downgrade" direction typical of
ice → liquid or more-complex-to-simpler transitions).

The mass-power ``ipow = ipow_to - ipow_from`` is derived from element
types the same way ``upgxfer`` does it:

- ``I_INVOLATILE`` / ``I_VOLATILE`` → 0 (number)
- ``I_COREMASS`` / ``I_VOLCORE`` → 1 (mass)
- ``I_CORE2MOM`` → 2 (second moment)

The source-element mass used in the rate depends on whether the source
group has cores AND the source element is a pure number element
(``itype ≤ I_VOLATILE``):

- No cores, or source is already a core-type: ``elemass = rmass``.
- Multicomponent group + volatile source element: subtract the total
  core mass to isolate the volatile fraction:
      ``elemass = (1 - Σ core / (pc · rmass)) · rmass``.

  This is the "fracmass" branch from Fortran. ``upgxfer`` skips it in
  the current port (see that module's note); ``downgxfer`` implements
  it faithfully because Phase 9's ``step_full`` needs correct
  bin-to-bin mass flux for heterogeneous ice nucleation.
"""

import jax.numpy as jnp

from carma.constants import FEW_PC
from carma.enums import ElementType
from carma.precision import DTYPE


def _ipow_of(itype_val):
    """Fortran ``ipow`` for a given element type."""
    if itype_val in (ElementType.I_INVOLATILE, ElementType.I_VOLATILE):
        return 0
    if itype_val in (ElementType.I_COREMASS, ElementType.I_VOLCORE):
        return 1
    return 2


def downgxfer(
    rnucpe,
    rnuclg,
    pc,
    rmass,
    pconmax,
    ielem,
    ibin,
    iz,
    nnucelem,
    inucelem,
    nnucbin,
    inucbin,
    igelem_arr,
    itype_arr,
    ienconc_arr,
    icorelem,
    ncore,
    nbin,
    ngroup,
    reset=True,
):
    """Accumulate down-direction nucleation production into ``rnucpe``.

    Args:
        rnucpe: Nucleation production array, shape ``(nbin, nelem)``.
            If ``reset=True``, the full array is zeroed first
            (matching the Fortran ``rnucpe(:,:) = 0`` at the top).
        rnuclg: Nucleation loss rates, shape
            ``(nbin, ngroup, ngroup)`` — ``[src_bin, src_grp, tgt_grp]``.
        pc: Particle concentrations, shape ``(nz, nbin, nelem)``.
        rmass: Bin mass, shape ``(nbin, ngroup)``.
        pconmax: Max concentration per group, shape ``(nz, ngroup)``.
        ielem: Target element index.
        ibin: Target bin index.
        iz: Vertical level index.
        nnucelem: Number of source elements per target element,
            shape ``(nelem,)`` int (static).
        inucelem: Source element indices per target,
            shape ``(max_nuc_elem, nelem)`` int (static).
        nnucbin: Number of source bins per target bin,
            shape ``(ngroup, nbin, ngroup)`` int (static).
        inucbin: Source bin indices per target bin,
            shape ``(max_nuc_bin, ngroup, nbin, ngroup)`` int (static).
        igelem_arr: Group per element, shape ``(nelem,)``.
        itype_arr: Element type per element, shape ``(nelem,)``.
        ienconc_arr: Number-conc element per group, shape ``(ngroup,)``.
        icorelem: Core element indices per group,
            shape ``(ncore_max, ngroup)``. ``-1`` = unused slot.
        ncore: Number of core elements per group, shape ``(ngroup,)``.
        nbin, ngroup: Static dimensions.
        reset: Zero the whole ``rnucpe`` before accumulating
            (default True; matches Fortran).

    Returns:
        Updated ``rnucpe`` of shape ``(nbin, nelem)``.

    Notes:
        All index arrays (``nnucelem``, ``inucelem``, ``nnucbin``,
        ``inucbin``, ``igelem_arr``, ``itype_arr``, ``ienconc_arr``,
        ``icorelem``, ``ncore``) must be Python-level arrays (numpy
        or native). They drive Python-level loops resolved at trace
        time, as in upgxfer / gasexchange.
    """
    if reset:
        rnucpe = jnp.zeros_like(rnucpe)

    igroup = int(igelem_arr[ielem])           # target group
    itype_to = int(itype_arr[ielem])
    ipow_to = _ipow_of(itype_to)

    n_nuc = int(nnucelem[ielem])
    for jefrom in range(n_nuc):
        iefrom = int(inucelem[jefrom, ielem])

        # Down-direction gate: only when source element index > target
        if iefrom <= ielem:
            continue

        igfrom = int(igelem_arr[iefrom])
        itype_from = int(itype_arr[iefrom])
        ipow_from = _ipow_of(itype_from)
        ipow = ipow_to - ipow_from

        n_nucbin = int(nnucbin[igfrom, ibin, igroup])
        nc_from = int(ncore[igfrom])
        # Python-level branch: only resolve fracmass if the group has
        # cores AND the source is a number/volatile element.
        need_fracmass = (nc_from > 0
                          and itype_from <= int(ElementType.I_VOLATILE))

        for jfrom in range(n_nucbin):
            ifrom = int(inucbin[jfrom, igfrom, ibin, igroup])

            has_particles = pconmax[iz, igfrom] > FEW_PC
            has_nucleation = rnuclg[ifrom, igfrom, igroup] > DTYPE(0.0)
            active = has_particles & has_nucleation

            rmass_src = rmass[ifrom, igfrom]

            if need_fracmass:
                ip = int(ienconc_arr[igfrom])
                totmass = pc[iz, ifrom, ip] * rmass_src
                rmasscore = jnp.zeros((), dtype=DTYPE)
                for ic in range(nc_from):
                    icore = int(icorelem[ic, igfrom])
                    rmasscore = rmasscore + pc[iz, ifrom, icore]
                safe_totmass = jnp.where(
                    totmass > DTYPE(0.0), totmass, DTYPE(1.0),
                )
                fracmass = DTYPE(1.0) - rmasscore / safe_totmass
                elemass = fracmass * rmass_src
            else:
                elemass = rmass_src

            # elemass raised to ipow (integer; small)
            if ipow == 0:
                mass_factor = DTYPE(1.0)
            elif ipow == 1:
                mass_factor = elemass
            elif ipow == -1:
                safe = jnp.where(elemass > DTYPE(0.0), elemass, DTYPE(1.0))
                mass_factor = DTYPE(1.0) / safe
            elif ipow == 2:
                mass_factor = elemass ** 2
            elif ipow == -2:
                safe = jnp.where(elemass > DTYPE(0.0), elemass, DTYPE(1.0))
                mass_factor = DTYPE(1.0) / (safe ** 2)
            else:
                mass_factor = elemass ** ipow

            rnucprod = (
                rnuclg[ifrom, igfrom, igroup]
                * pc[iz, ifrom, iefrom]
                * mass_factor
            )
            rnucprod = jnp.where(active, rnucprod, DTYPE(0.0))

            rnucpe = rnucpe.at[ibin, ielem].add(rnucprod)

    return rnucpe
