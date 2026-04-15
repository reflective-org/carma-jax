"""Coagulation mapping table setup for CARMA-JAX.

Builds all static coagulation arrays: destination bins, volume fractions,
production kernels, and bin pair lists. These are computed once at
configuration time and do not depend on atmospheric state.
Ported from: setupcoag.F90
"""

import numpy as np

from carma.config import CoagConfig


def setup_coag(nbin, ngroup, nelem, groups, elements, icoag_input, icoagelem_input):
    """Build coagulation mapping tables.

    Args:
        nbin: Number of size bins.
        ngroup: Number of particle groups.
        nelem: Number of elements.
        groups: Tuple of GroupConfig.
        elements: Tuple of ElementConfig.
        icoag_input: (NGROUP, NGROUP) coagulation target group, 0=no coag.
            Uses 0-based group indices, -1 for no coagulation.
        icoagelem_input: (NELEM, NGROUP) source element for production.
            -1 for no contribution.

    Returns:
        CoagConfig with all precomputed tables.
    """
    # Work in numpy for setup (not traced by JIT)
    icoag = np.array(icoag_input, dtype=np.int32)
    icoagelem = np.array(icoagelem_input, dtype=np.int32)

    # Symmetrize icoag
    for ig in range(ngroup):
        for jg in range(ig):
            if icoag[ig, jg] < 0 and icoag[jg, ig] >= 0:
                icoag[ig, jg] = icoag[jg, ig]
            elif icoag[jg, ig] < 0 and icoag[ig, jg] >= 0:
                icoag[jg, ig] = icoag[ig, jg]

    # Determine coagulation operation type (default: calculated)
    icoagop = np.full((ngroup, ngroup), 2, dtype=np.int32)  # I_COAGOP_CALC

    # Calculate kbin: destination bin for collision products
    kbin = np.zeros((ngroup, ngroup, ngroup, nbin, nbin), dtype=np.int32)
    for ig in range(ngroup):
        for jg in range(ngroup):
            igrp = icoag[ig, jg]
            if igrp < 0:
                continue
            rmass_igrp = np.array(groups[igrp].rmass)
            for i in range(nbin):
                for j in range(nbin):
                    rmsum = float(groups[ig].rmass[i]) + float(groups[jg].rmass[j])
                    # Find bin where rmsum falls
                    kb = nbin - 1  # Default: last bin
                    for ibin in range(nbin - 1):
                        if rmsum >= rmass_igrp[ibin] and rmsum < rmass_igrp[ibin + 1]:
                            kb = ibin
                            break
                    kbin[igrp, ig, jg, i, j] = kb

    # Calculate volx: partial volume loss fraction
    volx = np.zeros((ngroup, ngroup, ngroup, nbin, nbin), dtype=np.float64)
    for ig in range(ngroup):
        for jg in range(ngroup):
            igrp = icoag[ig, jg]
            if igrp < 0:
                continue
            if igrp != ig:
                # Complete loss to different group: volx not used for loss
                continue
            rmrat_igrp = float(groups[igrp].rmrat)
            for i in range(nbin):
                for j in range(nbin):
                    rmsum = float(groups[ig].rmass[i]) + float(groups[jg].rmass[j])
                    kb = kbin[igrp, ig, jg, i, j]
                    rmkbin = float(groups[igrp].rmass[kb])

                    if kb == i:
                        # Product falls in same bin as particle i: partial loss
                        denom = rmrat_igrp * rmkbin - rmkbin
                        if abs(denom) > 1e-30:
                            frac = (rmrat_igrp * rmkbin - rmsum) / denom
                            mi = float(groups[ig].rmass[i])
                            volx[igrp, ig, jg, i, j] = 1.0 - frac * mi / rmsum
                        else:
                            volx[igrp, ig, jg, i, j] = 1.0
                    else:
                        volx[igrp, ig, jg, i, j] = 1.0

    # Calculate pkernel: production kernel factors (6 types)
    pkernel = np.zeros((nbin, nbin, ngroup, ngroup, ngroup, 6), dtype=np.float64)
    for ig in range(ngroup):
        for jg in range(ngroup):
            igrp = icoag[ig, jg]
            if igrp < 0:
                continue
            rmrat_igrp = float(groups[igrp].rmrat)
            for i in range(nbin):
                for j in range(nbin):
                    rmsum = float(groups[ig].rmass[i]) + float(groups[jg].rmass[j])
                    kb = kbin[igrp, ig, jg, i, j]
                    rmk = float(groups[igrp].rmass[kb])

                    if kb < nbin - 1:
                        denom = rmrat_igrp * rmk - rmk
                        if abs(denom) > 1e-30:
                            pkernl = (rmrat_igrp * rmk - rmsum) / denom
                            pkernu = (rmsum - rmk) / denom
                        else:
                            pkernl = 1.0
                            pkernu = 0.0
                    else:
                        # Last bin: all production goes to this bin
                        pkernl = rmsum / rmk if abs(rmk) > 1e-30 else 1.0
                        pkernu = 0.0

                    rmi = float(groups[ig].rmass[i])
                    rmk_up = rmk * rmrat_igrp

                    pkernel[i, j, ig, jg, igrp, 0] = pkernu * rmi / rmsum  # upper, number
                    pkernel[i, j, ig, jg, igrp, 1] = pkernl * rmi / rmsum  # lower, number
                    pkernel[i, j, ig, jg, igrp, 2] = pkernu * rmk_up / rmsum  # upper, core
                    pkernel[i, j, ig, jg, igrp, 3] = pkernl * rmk / rmsum  # lower, core
                    pkernel[i, j, ig, jg, igrp, 4] = pkernu * (rmk_up / rmsum) ** 2  # upper, 2nd mom
                    pkernel[i, j, ig, jg, igrp, 5] = pkernl * (rmk / rmsum) ** 2  # lower, 2nd mom

    # Build bin pair lists (upper and lower contributors)
    max_pairs = nbin * nbin * ngroup * ngroup  # upper bound
    npairu = np.zeros((ngroup, nbin), dtype=np.int32)
    npairl = np.zeros((ngroup, nbin), dtype=np.int32)
    iup = np.zeros((ngroup, nbin, max_pairs), dtype=np.int32)
    jup = np.zeros((ngroup, nbin, max_pairs), dtype=np.int32)
    igup = np.zeros((ngroup, nbin, max_pairs), dtype=np.int32)
    jgup = np.zeros((ngroup, nbin, max_pairs), dtype=np.int32)
    ilow = np.zeros((ngroup, nbin, max_pairs), dtype=np.int32)
    jlow = np.zeros((ngroup, nbin, max_pairs), dtype=np.int32)
    iglow = np.zeros((ngroup, nbin, max_pairs), dtype=np.int32)
    jglow = np.zeros((ngroup, nbin, max_pairs), dtype=np.int32)

    for ig in range(ngroup):
        for jg in range(ngroup):
            igrp = icoag[ig, jg]
            if igrp < 0:
                continue
            for i in range(nbin):
                for j in range(nbin):
                    kb = kbin[igrp, ig, jg, i, j]

                    # Upper contribution: kb+1 is the target bin
                    if kb + 1 < nbin:
                        target = kb + 1
                        n = npairu[igrp, target]
                        iup[igrp, target, n] = i
                        jup[igrp, target, n] = j
                        igup[igrp, target, n] = ig
                        jgup[igrp, target, n] = jg
                        npairu[igrp, target] = n + 1

                    # Lower contribution: kb is the target bin
                    # Exclude self-bin case (i==target_bin AND ig==igrp)
                    if not (i == kb and ig == igrp):
                        target = kb
                        n = npairl[igrp, target]
                        ilow[igrp, target, n] = i
                        jlow[igrp, target, n] = j
                        iglow[igrp, target, n] = ig
                        jglow[igrp, target, n] = jg
                        npairl[igrp, target] = n + 1

    # Trim pair arrays to actual max
    actual_max = max(int(npairu.max()), int(npairl.max()), 1)

    import jax.numpy as jnp

    return CoagConfig(
        icoag=jnp.array(icoag),
        icoagelem=jnp.array(icoagelem),
        icoagop=jnp.array(icoagop),
        volx=jnp.array(volx),
        npairl=jnp.array(npairl),
        npairu=jnp.array(npairu),
        ilow=jnp.array(ilow[:, :, :actual_max]),
        jlow=jnp.array(jlow[:, :, :actual_max]),
        iglow=jnp.array(iglow[:, :, :actual_max]),
        jglow=jnp.array(jglow[:, :, :actual_max]),
        iup=jnp.array(iup[:, :, :actual_max]),
        jup=jnp.array(jup[:, :, :actual_max]),
        igup=jnp.array(igup[:, :, :actual_max]),
        jgup=jnp.array(jgup[:, :, :actual_max]),
        pkernel=jnp.array(pkernel),
    )
