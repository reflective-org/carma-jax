"""Per-bin dry particle density for multi-element bins (CARMA-JAX).

Computes ``rhop[iz, ibin, igroup]`` as a mass-weighted mean over the
core elements and the shell (number-concentration) element.

Ported from: rhopart.F90 (CARMA base) — only the dry-density block.
The wet-radius calls to `getwetr` that Fortran's `rhopart` also drives
live in `wetr.py`; orchestrating them alongside `rhopart` and
`hygroscopicity` happens in Phase 9 (`step_full`).

Formula (cores present):

    m_core  = sum_{c ∈ cores(ig)} pc[iz, ibin, c]
    v_core  = sum_{c ∈ cores(ig)} pc[iz, ibin, c] / rhoelem[ibin, c]
    m_total = pc[iz, ibin, iepart] * rmass[ibin, ig]
    v_shell = (m_total - m_core) / rhoelem[ibin, iepart]
    rhop    = m_total / (v_shell + v_core)

Safety (numerical diffusion in advection can push m_core > m_total):

    if m_core > m_total:
        rhop = m_core / v_core                            # pure-core density
        pc[iz, ibin, iepart] = m_core / rmass[ibin, ig]   # repair number

If the group has no cores, ``rhop = rhoelem[ibin, iepart]``.

Returns both the density array and a corrected ``pc`` (so callers
can optionally adopt the core-mass repair — the Fortran does it
in-place).
"""

import jax
import jax.numpy as jnp

from carma.precision import DTYPE


def rhopart(pc, rmass_2d, rhoelem,
            ienconc_arr, igelem_arr, icorelem, ncore):
    """Mass-weighted dry density per (iz, ibin, igroup) bin.

    Args:
        pc: (NZ, NBIN, NELEM) particle concentrations.
        rmass_2d: (NBIN, NGROUP) bin-centre masses.
        rhoelem: (NBIN, NELEM) per-element density.
        ienconc_arr: (NGROUP,) number-concentration element index per group.
        igelem_arr: (NELEM,) group index per element (unused here but kept
            for call-site symmetry with `hygroscopicity`).
        icorelem: (MAX_NCORE, NGROUP) core element indices, padded with -1
            where unused.
        ncore: (NGROUP,) number of core elements per group.

    Returns:
        ``(rhop, pc_repaired)`` where:
            rhop:  (NZ, NBIN, NGROUP) dry-particle mass density [g/cm³]
            pc_repaired: (NZ, NBIN, NELEM) with the number-element clamped
                in bins where core mass exceeds total mass.
    """
    pc = jnp.asarray(pc, dtype=DTYPE)
    rmass_2d = jnp.asarray(rmass_2d, dtype=DTYPE)
    rhoelem = jnp.asarray(rhoelem, dtype=DTYPE)
    ienconc_arr = jnp.asarray(ienconc_arr)
    icorelem = jnp.asarray(icorelem)
    ncore = jnp.asarray(ncore)

    NZ, NBIN, NELEM = pc.shape
    NGROUP = rmass_2d.shape[1]
    MAX_NCORE = icorelem.shape[0]

    # Build (NELEM, NGROUP) mask: element e is a core of group g.
    # Static in NELEM × NGROUP so Python loops unroll at trace time.
    core_mask = jnp.zeros((NELEM, NGROUP), dtype=bool)
    valid = jnp.arange(MAX_NCORE)[:, None] < ncore[None, :]   # (MAX_NCORE, NGROUP)
    for ig in range(NGROUP):
        for k in range(MAX_NCORE):
            ie = icorelem[k, ig]
            is_valid = valid[k, ig] & (ie >= 0)
            core_mask = core_mask.at[ie, ig].set(
                core_mask[ie, ig] | is_valid)

    rhop = jnp.zeros((NZ, NBIN, NGROUP), dtype=DTYPE)
    pc_out = pc

    for ig in range(NGROUP):
        iepart = ienconc_arr[ig]
        rmass_g = rmass_2d[:, ig]                            # (NBIN,)
        rho_shell = rhoelem[:, iepart]                        # (NBIN,)

        pc_num = pc_out[:, :, iepart]                         # (NZ, NBIN)
        m_total = pc_num * rmass_g[None, :]                   # (NZ, NBIN)

        # Per-bin core mass + volume
        is_core_of_g = core_mask[:, ig]                       # (NELEM,)
        pc_core = jnp.where(is_core_of_g[None, None, :], pc_out, DTYPE(0.0))
        m_core = jnp.sum(pc_core, axis=2)                     # (NZ, NBIN)
        # v_core = sum pc_core / rhoelem[ibin, ie]; reshape rhoelem to (1, NBIN, NELEM)
        rho_elem_b = jnp.where(is_core_of_g[None, None, :],
                               rhoelem[None, :, :], DTYPE(1.0))
        v_core = jnp.sum(
            jnp.where(is_core_of_g[None, None, :],
                      pc_core / rho_elem_b, DTYPE(0.0)),
            axis=2,
        )

        # Three regimes:
        #   (A) group has no cores at all   → rhop = rho_shell
        #   (B) m_core == 0                 → rhop = rho_shell
        #   (C) 0 < m_core ≤ m_total        → full mixed-density formula
        #   (D) m_core > m_total            → rhop = m_core/v_core, pc_num := m_core/rmass
        has_core_group = ncore[ig] > 0

        # Safe denominators (avoid 0/0 where branches won't be selected)
        safe_v_core = jnp.where(v_core > DTYPE(0.0), v_core, DTYPE(1.0))
        v_shell = (m_total - m_core) / jnp.where(
            rho_shell[None, :] > DTYPE(0.0), rho_shell[None, :], DTYPE(1.0))
        v_shell = jnp.maximum(v_shell, DTYPE(0.0))
        safe_v_total = jnp.where(
            (v_shell + v_core) > DTYPE(0.0), v_shell + v_core, DTYPE(1.0))

        rho_mixed = m_total / safe_v_total
        rho_truncated = m_core / safe_v_core

        # (C) vs (D)
        core_exceeds = m_core > m_total
        rho_cored = jnp.where(core_exceeds, rho_truncated, rho_mixed)

        # (B): no core mass in this bin
        has_core_bin = m_core > DTYPE(0.0)
        rho_with_cores = jnp.where(has_core_bin, rho_cored, rho_shell[None, :])

        # (A): group has no cores at all
        rho_group = jnp.where(has_core_group, rho_with_cores, rho_shell[None, :])

        rhop = rhop.at[:, :, ig].set(rho_group)

        # Repair pc_num where core mass exceeded total mass
        if True:   # (Python-level, always included)
            pc_num_repaired = jnp.where(
                core_exceeds & has_core_group,
                m_core / rmass_g[None, :],
                pc_num,
            )
            pc_out = pc_out.at[:, :, iepart].set(pc_num_repaired)

    return rhop, pc_out
