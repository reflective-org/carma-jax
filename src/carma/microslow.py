"""Slow microphysics driver (coagulation) for CARMA-JAX.

Fully vectorized and JIT-compilable. Pair summation uses gather+reduce,
bin loop uses jax.lax.fori_loop for sequential dependency.
Ported from: microslow.F90
"""

import jax
import jax.numpy as jnp
from functools import partial

from carma.constants import FEW_PC
from carma.coagulation.coagl import coagl


def _compute_production_for_bin(ibin, pc, pcl, ckernel, pconmax,
                                 pair_i, pair_j, pair_ig, pair_jg,
                                 pair_iefrom, pair_je, pair_pk,
                                 pair_count, nbin, ngroup):
    """Vectorized production for one bin from all contributing pairs.

    Args:
        ibin: Target bin index.
        pc, pcl: (NZ, NBIN, NELEM) concentrations.
        ckernel: (NZ, NBIN, NBIN, NGROUP, NGROUP).
        pconmax: (NZ, NGROUP).
        pair_i, pair_j, pair_ig, pair_jg: (MAX_PAIRS,) source indices.
        pair_iefrom, pair_je: (MAX_PAIRS,) element indices.
        pair_pk: (MAX_PAIRS,) pkernel weights.
        pair_count: Number of valid pairs.
        nbin, ngroup: Dimensions.

    Returns:
        Production rate for this bin, shape (NZ,).
    """
    max_pairs = pair_i.shape[0]
    nz = pc.shape[0]
    nelem = pc.shape[2]

    # Validity mask
    valid = jnp.arange(max_pairs) < pair_count  # (MAX_PAIRS,)

    # Gather pc[:, i, iefrom] for each pair
    flat_pc = pc.reshape(nz, -1)  # (NZ, NBIN*NELEM)
    pc_idx = pair_i * nelem + pair_iefrom  # (MAX_PAIRS,)
    pc_from = flat_pc[:, pc_idx]  # (NZ, MAX_PAIRS)

    # Gather pcl[:, j, je] for each pair
    flat_pcl = pcl.reshape(nz, -1)
    pcl_idx = pair_j * nelem + pair_je
    pcl_partner = flat_pcl[:, pcl_idx]  # (NZ, MAX_PAIRS)

    # Gather ckernel[:, i, j, ig, jg]
    flat_ck = ckernel.reshape(nz, -1)  # (NZ, NBIN*NBIN*NGROUP*NGROUP)
    ck_idx = (pair_i * nbin * ngroup * ngroup +
              pair_j * ngroup * ngroup +
              pair_ig * ngroup +
              pair_jg)
    ck_vals = flat_ck[:, ck_idx]  # (NZ, MAX_PAIRS)

    # Active mask
    active_ig = pconmax[:, pair_ig] > FEW_PC  # (NZ, MAX_PAIRS)
    active_jg = pconmax[:, pair_jg] > FEW_PC
    active = (active_ig & active_jg).astype(pc.dtype)

    # Production per pair, masked and summed
    prod = pc_from * pcl_partner * ck_vals * pair_pk[None, :] * valid[None, :].astype(pc.dtype) * active
    return prod.sum(axis=1)  # (NZ,)


def make_microslow(nbin, nelem, ngroup, elem_igroup,
                   icoag, volx, icoagelem,
                   npairu, npairl,
                   iup, jup, igup, jgup,
                   ilow, jlow, iglow, jglow,
                   pkernel, ienconc_arr, elem_itypes):
    """Create a JIT-compiled microslow function with baked-in config.

    This factory function captures all static configuration in a closure,
    returning a pure function of dynamic arrays that can be JIT-compiled.

    Args:
        nbin, nelem, ngroup: Dimensions.
        elem_igroup: (NELEM,) group index per element.
        icoag: (NGROUP, NGROUP) coagulation target group.
        volx: (NGROUP, NGROUP, NGROUP, NBIN, NBIN) volume fractions.
        icoagelem: (NELEM, NGROUP) source element mapping.
        npairu, npairl: (NGROUP, NBIN) pair counts.
        iup/jup/igup/jgup: (NGROUP, NBIN, MAX_PAIRS) upper pair indices.
        ilow/jlow/iglow/jglow: (NGROUP, NBIN, MAX_PAIRS) lower pair indices.
        pkernel: (NBIN, NBIN, NGROUP, NGROUP, NGROUP, 6) production kernels.
        ienconc_arr: (NGROUP,) number concentration element per group.
        elem_itypes: (NELEM,) element types.

    Returns:
        JIT-compiled function: microslow(pc, pcl, ckernel, pconmax, zmet, dtime) -> pc
    """
    from carma.enums import ElementType

    # Determine pkernel index for each element
    i_pkern_upper = []
    i_pkern_lower = []
    for ie in range(nelem):
        itype = int(elem_itypes[ie])
        if itype == ElementType.I_COREMASS or itype == ElementType.I_VOLCORE:
            i_pkern_upper.append(2)
            i_pkern_lower.append(3)
        elif itype == ElementType.I_CORE2MOM:
            i_pkern_upper.append(4)
            i_pkern_lower.append(5)
        else:
            i_pkern_upper.append(0)
            i_pkern_lower.append(1)

    # Precompute combined pair arrays for each (ielem, ibin):
    # Merge upper + lower pairs into one array with precomputed pk values
    max_pairs_u = int(npairu.max()) if npairu.size > 0 else 0
    max_pairs_l = int(npairl.max()) if npairl.size > 0 else 0
    max_pairs_total = max_pairs_u + max_pairs_l

    # For each element and bin, precompute the merged pair arrays
    # Shape: (NELEM, NBIN, MAX_TOTAL_PAIRS)
    import numpy as np

    all_pair_i = np.zeros((nelem, nbin, max_pairs_total), dtype=np.int32)
    all_pair_j = np.zeros((nelem, nbin, max_pairs_total), dtype=np.int32)
    all_pair_ig = np.zeros((nelem, nbin, max_pairs_total), dtype=np.int32)
    all_pair_jg = np.zeros((nelem, nbin, max_pairs_total), dtype=np.int32)
    all_pair_iefrom = np.zeros((nelem, nbin, max_pairs_total), dtype=np.int32)
    all_pair_je = np.zeros((nelem, nbin, max_pairs_total), dtype=np.int32)
    all_pair_pk = np.zeros((nelem, nbin, max_pairs_total), dtype=np.float64)
    all_pair_count = np.zeros((nelem, nbin), dtype=np.int32)

    icoagelem_np = np.array(icoagelem)
    ienconc_np = np.array(ienconc_arr)
    pkernel_np = np.array(pkernel)

    for ie in range(nelem):
        pkern_u = i_pkern_upper[ie]
        pkern_l = i_pkern_lower[ie]

        for ib in range(nbin):
            idx = 0
            # Upper pairs
            for igrp in range(ngroup):
                nu = int(npairu[igrp, ib])
                for q in range(nu):
                    i_src = int(iup[igrp, ib, q])
                    j_src = int(jup[igrp, ib, q])
                    ig_src = int(igup[igrp, ib, q])
                    jg_src = int(jgup[igrp, ib, q])
                    iefrom = int(icoagelem_np[ie, ig_src])
                    if iefrom < 0:
                        continue
                    je = int(ienconc_np[jg_src])
                    pk = float(pkernel_np[i_src, j_src, ig_src, jg_src, igrp, pkern_u])

                    all_pair_i[ie, ib, idx] = i_src
                    all_pair_j[ie, ib, idx] = j_src
                    all_pair_ig[ie, ib, idx] = ig_src
                    all_pair_jg[ie, ib, idx] = jg_src
                    all_pair_iefrom[ie, ib, idx] = iefrom
                    all_pair_je[ie, ib, idx] = je
                    all_pair_pk[ie, ib, idx] = pk
                    idx += 1

            # Lower pairs
            for igrp in range(ngroup):
                nl = int(npairl[igrp, ib])
                for q in range(nl):
                    i_src = int(ilow[igrp, ib, q])
                    j_src = int(jlow[igrp, ib, q])
                    ig_src = int(iglow[igrp, ib, q])
                    jg_src = int(jglow[igrp, ib, q])
                    iefrom = int(icoagelem_np[ie, ig_src])
                    if iefrom < 0:
                        continue
                    je = int(ienconc_np[jg_src])
                    pk = float(pkernel_np[i_src, j_src, ig_src, jg_src, igrp, pkern_l])

                    all_pair_i[ie, ib, idx] = i_src
                    all_pair_j[ie, ib, idx] = j_src
                    all_pair_ig[ie, ib, idx] = ig_src
                    all_pair_jg[ie, ib, idx] = jg_src
                    all_pair_iefrom[ie, ib, idx] = iefrom
                    all_pair_je[ie, ib, idx] = je
                    all_pair_pk[ie, ib, idx] = pk
                    idx += 1

            all_pair_count[ie, ib] = idx

    # Convert to JAX arrays (these are captured in the closure)
    _pair_i = jnp.array(all_pair_i)
    _pair_j = jnp.array(all_pair_j)
    _pair_ig = jnp.array(all_pair_ig)
    _pair_jg = jnp.array(all_pair_jg)
    _pair_iefrom = jnp.array(all_pair_iefrom)
    _pair_je = jnp.array(all_pair_je)
    _pair_pk = jnp.array(all_pair_pk)
    _pair_count = jnp.array(all_pair_count)
    _elem_igroup = jnp.array(elem_igroup)
    _icoag = jnp.array(icoag)
    _volx = jnp.array(volx)
    _ienconc = jnp.array(ienconc_arr)

    @jax.jit
    def microslow_jit(pc, pcl, ckernel, pconmax, zmet, dtime):
        """JIT-compiled coagulation step.

        Args:
            pc: (NZ, NBIN, NELEM) particle concentrations.
            pcl: (NZ, NBIN, NELEM) saved concentrations.
            ckernel: (NZ, NBIN, NBIN, NGROUP, NGROUP) coagulation kernel.
            pconmax: (NZ, NGROUP) max concentrations.
            zmet: (NZ,) vertical metric.
            dtime: Timestep [s].

        Returns:
            Updated pc (NZ, NBIN, NELEM).
        """
        # Step 1: Compute loss rates (already vectorized)
        coaglg = _compute_loss(ckernel, pcl, pconmax)

        # Step 2: Sequential bin loop — production and solve
        def step_elem_bin(pc, indices):
            ie, ib = indices
            ig = _elem_igroup[ie]

            # Gather precomputed pair arrays for this (elem, bin)
            p_i = _pair_i[ie, ib]
            p_j = _pair_j[ie, ib]
            p_ig = _pair_ig[ie, ib]
            p_jg = _pair_jg[ie, ib]
            p_iefrom = _pair_iefrom[ie, ib]
            p_je = _pair_je[ie, ib]
            p_pk = _pair_pk[ie, ib]
            p_count = _pair_count[ie, ib]

            # Vectorized production
            prod = _compute_production_for_bin(
                pc, pcl, ckernel, pconmax,
                p_i, p_j, p_ig, p_jg, p_iefrom, p_je, p_pk,
                p_count, nbin, ngroup
            )

            # Solve: implicit Euler
            ppd = prod / zmet
            pls = coaglg[:, ib, ig] / zmet
            pc_old = pc[:, ib, ie]
            pc_new = (pc_old + dtime * ppd) / (1.0 + pls * dtime)
            pc = pc.at[:, ib, ie].set(pc_new)

            return pc, None

        # Build (ielem, ibin) index pairs — element outer, bin inner
        elem_indices = jnp.repeat(jnp.arange(nelem), nbin)
        bin_indices = jnp.tile(jnp.arange(nbin), nelem)
        scan_indices = jnp.stack([elem_indices, bin_indices], axis=1)

        pc, _ = jax.lax.scan(step_elem_bin, pc, scan_indices)
        return pc

    # Extract icoag and ienconc as Python values at factory time (not traced)
    import numpy as _np
    _icoag_py = _np.array(icoag, dtype=int)
    _ienconc_py = [int(ienconc_arr[g]) for g in range(ngroup)]
    _volx_np = _np.array(volx)

    # Pre-extract which (ig, jg) pairs are active and their properties
    _loss_pairs = []
    for ig in range(ngroup):
        for jg in range(ngroup):
            igrp = int(_icoag_py[ig, jg])
            if igrp < 0:
                continue
            _loss_pairs.append((ig, jg, igrp, _ienconc_py[jg]))

    def _compute_loss(ckernel, pcl, pconmax):
        """Vectorized coagulation loss rate computation."""
        nz = pcl.shape[0]
        coaglg = jnp.zeros((nz, nbin, ngroup), dtype=pcl.dtype)

        for ig, jg, igrp, je in _loss_pairs:
            active = ((pconmax[:, jg] > FEW_PC) &
                      (pconmax[:, ig] > FEW_PC)).astype(pcl.dtype)
            pcl_je = pcl[:, :, je]

            if igrp == ig:
                volx_slice = _volx[igrp, ig, jg, :nbin - 1, :]
                kern_slice = ckernel[:, :nbin - 1, :, ig, jg]
                loss = jnp.einsum("zij,zj,ij->zi", kern_slice, pcl_je, volx_slice)
                loss = loss * active[:, None]
                coaglg = coaglg.at[:, :nbin - 1, ig].add(loss)
            else:
                kern_slice = ckernel[:, :, :, ig, jg]
                loss = jnp.einsum("zij,zj->zi", kern_slice, pcl_je)
                loss = loss * active[:, None]
                coaglg = coaglg.at[:, :, ig].add(loss)

        return coaglg

    def _compute_production_for_bin(pc, pcl, ckernel, pconmax,
                                     p_i, p_j, p_ig, p_jg, p_iefrom, p_je, p_pk,
                                     p_count, nbin_, ngroup_):
        """Vectorized production for one bin."""
        nz = pc.shape[0]
        nelem_ = pc.shape[2]
        max_p = p_i.shape[0]

        valid = (jnp.arange(max_p) < p_count).astype(pc.dtype)

        # Gather pc[:, i, iefrom]
        flat_pc = pc.reshape(nz, -1)
        pc_idx = p_i * nelem_ + p_iefrom
        pc_from = flat_pc[:, pc_idx]

        # Gather pcl[:, j, je]
        flat_pcl = pcl.reshape(nz, -1)
        pcl_idx = p_j * nelem_ + p_je
        pcl_partner = flat_pcl[:, pcl_idx]

        # Gather ckernel[:, i, j, ig, jg]
        flat_ck = ckernel.reshape(nz, -1)
        ck_idx = (p_i * nbin_ * ngroup_ * ngroup_ +
                  p_j * ngroup_ * ngroup_ +
                  p_ig * ngroup_ +
                  p_jg)
        ck_vals = flat_ck[:, ck_idx]

        # Active mask
        active = ((pconmax[:, p_ig] > FEW_PC) &
                  (pconmax[:, p_jg] > FEW_PC)).astype(pc.dtype)

        prod = pc_from * pcl_partner * ck_vals * p_pk[None, :] * valid[None, :] * active
        return prod.sum(axis=1)

    return microslow_jit


def microslow(config, pc, pcl, ckernel, pconmax, zmet, dtime):
    """Convenience wrapper that builds and calls the JIT-compiled microslow.

    For repeated calls with the same config, use make_microslow() once
    and call the returned function directly to avoid rebuilding.
    """
    _fn = make_microslow(
        nbin=config.nbin, nelem=config.nelem, ngroup=config.ngroup,
        elem_igroup=jnp.array([e.igroup for e in config.elements]),
        icoag=config.coag.icoag,
        volx=config.coag.volx,
        icoagelem=config.coag.icoagelem,
        npairu=config.coag.npairu, npairl=config.coag.npairl,
        iup=config.coag.iup, jup=config.coag.jup,
        igup=config.coag.igup, jgup=config.coag.jgup,
        ilow=config.coag.ilow, jlow=config.coag.jlow,
        iglow=config.coag.iglow, jglow=config.coag.jglow,
        pkernel=config.coag.pkernel,
        ienconc_arr=jnp.array([g.ienconc for g in config.groups]),
        elem_itypes=jnp.array([e.itype for e in config.elements]),
    )
    return _fn(pc, pcl, ckernel, pconmax, zmet, dtime)
