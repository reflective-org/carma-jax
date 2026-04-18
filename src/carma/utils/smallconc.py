"""Floor enforcement for particle concentrations (CARMA-JAX).

Ensures pc >= SMALL_PC for number-density elements. For core-mass and
second-moment elements, uses an FIX_COREF-scaled floor that depends on
bin mass. When the number-density in a bin is below SMALL_PC (or the
core-mass element is below its floor), the element is clamped to the
FIX_COREF-scaled value.

Ported from: smallconc.F90
"""

import jax
import jax.numpy as jnp

from carma.constants import FIX_COREF, SMALL_PC
from carma.enums import ElementType
from carma.precision import DTYPE


def smallconc(pc, itype_arr, ienconc_arr, igelem_arr, rmass_2d):
    """Apply per-element floors to all particle concentrations.

    Args:
        pc: (NZ, NBIN, NELEM) particle concentrations.
        itype_arr: (NELEM,) element type (int, matches ElementType enum).
        ienconc_arr: (NGROUP,) number-density element index for each group.
        igelem_arr: (NELEM,) group index for each element.
        rmass_2d: (NBIN, NGROUP) bin-center mass.

    Returns:
        pc with all elements floored.
    """
    itype_arr = jnp.asarray(itype_arr)
    ienconc_arr = jnp.asarray(ienconc_arr)
    igelem_arr = jnp.asarray(igelem_arr)

    # Element-dependent group and number element (NELEM,)
    ig_of_e = igelem_arr                   # (NELEM,)
    ip_of_e = ienconc_arr[ig_of_e]         # (NELEM,) — number-density elem per element

    is_number = (jnp.arange(pc.shape[2]) == ip_of_e)
    is_core = (itype_arr == int(ElementType.I_COREMASS)) | \
              (itype_arr == int(ElementType.I_VOLCORE))
    is_mom2 = (itype_arr == int(ElementType.I_CORE2MOM))

    # rmass per (NBIN, NELEM), using group-of-element mapping
    rmass_by_elem = rmass_2d[:, ig_of_e]  # (NBIN, NELEM)

    small_core = SMALL_PC * rmass_by_elem * FIX_COREF
    small_mom2 = SMALL_PC * (rmass_by_elem * FIX_COREF) ** 2
    # small_val per (NBIN, NELEM), undefined for number elements (masked later)
    small_val = jnp.where(
        is_core[None, :],
        small_core,
        jnp.where(is_mom2[None, :], small_mom2, DTYPE(0.0)),
    )  # (NBIN, NELEM)

    # Number-density pc at the corresponding number element per element
    pc_num = pc[:, :, ip_of_e]  # (NZ, NBIN, NELEM) — same shape as pc

    # Non-number element: clamp to small_val if number is too small OR element is too small.
    needs_clamp = (pc_num <= SMALL_PC) | (pc < small_val[None, :, :])
    pc_nonnum = jnp.where(needs_clamp, small_val[None, :, :], pc)

    # Number element: pc = max(pc, SMALL_PC)
    pc_num_clamp = jnp.maximum(pc, SMALL_PC)

    return jnp.where(is_number[None, None, :], pc_num_clamp, pc_nonnum)


def maxconc(pc, ienconc_arr, zmet):
    """Compute max particle concentration per group per level.

    pconmax[iz, igrp] = max over bins of pc[iz, :, ienconc_arr[igrp]] / zmet[iz]

    Args:
        pc: (NZ, NBIN, NELEM).
        ienconc_arr: (NGROUP,).
        zmet: (NZ,).

    Returns:
        pconmax: (NZ, NGROUP).
    """
    # Gather number-density element per group: (NZ, NBIN, NGROUP)
    ienconc_j = jnp.asarray(ienconc_arr)
    pc_num = pc[:, :, ienconc_j]  # (NZ, NBIN, NGROUP)
    pconmax = jnp.max(pc_num, axis=1) / zmet[:, None]  # (NZ, NGROUP)
    return pconmax
