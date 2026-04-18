"""Nucleation mapping table setup for CARMA-JAX.

Computes bin-to-bin and element-to-element nucleation mapping tables.
These are static (computed once at initialization).
Ported from: carma_mod.F90 (setupnuc section, lines 577-724)
"""

import numpy as np
import jax.numpy as jnp


def setup_nuc(nbin, ngroup, nelem, groups_rmass,
              nuc_from_elem, nuc_to_elem, nuc_proc):
    """Build nucleation mapping tables.

    Args:
        nbin: Number of bins.
        ngroup: Number of groups.
        nelem: Number of elements.
        groups: Tuple of GroupConfig (need rmass per group).
        elements: Tuple of ElementConfig (need igroup, itype).
        groups_rmass: List of rmass arrays per group, each shape (NBIN,).
        nuc_from_elem: List of source element indices for each nucleation pair.
        nuc_to_elem: List of target element indices for each nucleation pair.
        nuc_proc: List of nucleation process flags for each pair.

    Returns:
        Dict with nucleation tables.
    """
    # --- Bin mapping: inuc2bin ---
    inuc2bin = np.zeros((nbin, ngroup, ngroup), dtype=np.int32)

    for igfrom in range(ngroup):
        for igto in range(ngroup):
            rmass_from = np.array(groups_rmass[igfrom])
            rmass_to = np.array(groups_rmass[igto])

            for ifrom in range(nbin):
                inuc2bin[ifrom, igfrom, igto] = -1  # no mapping
                for ibto in range(nbin - 1, -1, -1):
                    if rmass_to[ibto] >= rmass_from[ifrom]:
                        inuc2bin[ifrom, igfrom, igto] = ibto

    # --- Build nnucbin and inucbin from inuc2bin ---
    max_nuc_per_bin = nbin * ngroup
    nnucbin = np.zeros((ngroup, nbin, ngroup), dtype=np.int32)
    inucbin = np.zeros((max_nuc_per_bin, ngroup, nbin, ngroup), dtype=np.int32)

    for igroup in range(ngroup):  # target group
        for ibin in range(nbin):  # target bin
            for igfrom in range(ngroup):  # source group
                count = 0
                for ifrom in range(nbin):  # source bin
                    if inuc2bin[ifrom, igfrom, igroup] == ibin:
                        inucbin[count, igfrom, ibin, igroup] = ifrom
                        count += 1
                nnucbin[igfrom, ibin, igroup] = count

    # --- Element mapping ---
    # if_nuc[iefrom, ieto] = True if nucleation from iefrom to ieto
    if_nuc = np.zeros((nelem, nelem), dtype=bool)
    inucproc_arr = np.zeros((nelem, nelem), dtype=np.int32)

    for i, (src, tgt, proc) in enumerate(zip(nuc_from_elem, nuc_to_elem, nuc_proc)):
        if src >= 0 and tgt >= 0:
            if_nuc[src, tgt] = True
            inucproc_arr[src, tgt] = proc

    # nnucelem[ielem] = count of source elements that nucleate TO ielem
    # inucelem[j, ielem] = source element index
    nnucelem = np.zeros(nelem, dtype=np.int32)
    max_nuc_elem = nelem
    inucelem = np.full((max_nuc_elem, nelem), -1, dtype=np.int32)

    for ieto in range(nelem):
        count = 0
        for iefrom in range(nelem):
            if if_nuc[iefrom, ieto]:
                inucelem[count, ieto] = iefrom
                count += 1
        nnucelem[ieto] = count

    return {
        "inuc2bin": jnp.array(inuc2bin),
        "nnucbin": jnp.array(nnucbin),
        "inucbin": jnp.array(inucbin),
        "nnucelem": jnp.array(nnucelem),
        "inucelem": jnp.array(inucelem),
        "inucproc": jnp.array(inucproc_arr),
        "if_nuc": jnp.array(if_nuc),
    }
