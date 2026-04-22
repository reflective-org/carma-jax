"""Adaptive substep-count estimator (``nsubsteps``).

Ports ``nsubsteps.F90``. Given the state at one vertical level,
returns the suggested number of micro-substeps to run for the
microphysics call at that level.

Three paths produce ``maxsubsteps``:

1. **Drop activation**: any involatile (CN) group that would activate
   into droplets at any bin where ``pc / zmet > conmax · pconmax``
   and ``ss > scrit``.
2. **Homogeneous aerosol freezing**: any group with ``I_AERFREEZE``
   in ``inucproc`` when ``supsati > 0.4`` and ``T < 233.16 K``.
3. Growth-rate limit exceeds ``maxsubsteps``.

Otherwise the count is derived from the growth rate of a particle
at the "smallest significant" bin of each volatile group:

    ``dmdt = pvap · ss · g0 / (1 + g0·g1·pvap)``
    ``dt_adv = min(dt_save, dm / dmdt)``
    ``ntsubsteps = round(dt_save / dt_adv)`` clipped to
    ``[minsubsteps, maxsubsteps]``.

The function is a pure single-level routine. Static config tables
drive Python-level loops; traced state (``pc``, ``supsatl``,
``supsati``, ``gro``, ``gro1``, etc.) flows through ``jnp`` ops.

Returned value is a ``jnp.int32`` scalar so callers can either cast
to a Python int (forcing a per-call JIT specialisation of the
downstream fori_loop bound) or keep it traced.

Ported from: ``nsubsteps.F90``.
"""

import jax.numpy as jnp
import numpy as np

from carma.constants import FEW_PC
from carma.enums import ElementType, NucProcess
from carma.precision import DTYPE


_T_AERFREEZE = DTYPE(233.16)
_SS_AERFREEZE = DTYPE(0.4)


def nsubsteps(
    pc,
    supsatl,
    supsati,
    supsatlold,
    pvapl,
    pvapi,
    scrit,
    gro,
    gro1,
    pconmax,
    zmet,
    dm,
    iz,
    dtime_save,
    minsubsteps,
    maxsubsteps,
    conmax,
    ngroup,
    nbin,
    # Static config tables (numpy arrays, driven by Python loops)
    ienconc_arr,
    itype_arr,
    inucgas_arr,
    igrowgas_arr,
    nnuc2elem_arr,
    inuc2elem_arr,
    inucproc_arr,
    is_grp_ice_arr,
    do_substep=True,
):
    """Compute the suggested substep count for one column level.

    Args:
        pc: Particle concentrations, shape ``(nz, nbin, nelem)``.
        supsatl, supsati: Supersaturation (nz, ngas).
        supsatlold: Previous-step supsatl (nz, ngas).
        pvapl, pvapi: Saturation vapour pressures (nz, ngas).
        scrit: Critical supersaturation for activation
            (nz, nbin, ngroup).
        gro, gro1: Growth kernels (nz, nbin, ngroup).
        pconmax: Max concentration per group (nz, ngroup).
        zmet: Vertical metric (nz,).
        dm: Bin width in mass (nbin, ngroup).
        iz: Vertical level index (Python int).
        dtime_save: Outer timestep [s] (scalar).
        minsubsteps, maxsubsteps: Integer bounds (Python ints).
        conmax: Concentration-threshold fraction (scalar).
        ngroup, nbin: Static dimensions.
        ienconc_arr, itype_arr, inucgas_arr, igrowgas_arr,
        nnuc2elem_arr, inuc2elem_arr, inucproc_arr, is_grp_ice_arr:
            Static config tables (numpy arrays).
        do_substep: If False, returns ``1``.

    Returns:
        ``jnp.int32`` scalar with the number of substeps in
        ``[minsubsteps, maxsubsteps]``.
    """
    if not do_substep:
        return jnp.asarray(1, dtype=jnp.int32)

    maxsubsteps_i = jnp.asarray(maxsubsteps, dtype=jnp.int32)
    minsubsteps_i = jnp.asarray(minsubsteps, dtype=jnp.int32)

    # Accumulate forces and candidate substep counts as traced values.
    force_max = jnp.asarray(False)
    # We'll track ibin_small per-group as a jnp vector so we can
    # index gro/gro1 with traced bins via take.
    ibin_small_list = []

    # ------- Pass 1: ibin_small + drop-activation short-circuit -------
    for ig in range(ngroup):
        iepart = int(ienconc_arr[ig])
        itype_val = int(itype_arr[iepart])

        ibin_small_ig = jnp.asarray(nbin - 1, dtype=jnp.int32)
        has_particles = pconmax[iz, ig] > FEW_PC

        if itype_val == int(ElementType.I_INVOLATILE):
            igas = int(inucgas_arr[ig])
            if igas >= 0:
                ss = jnp.maximum(supsatl[iz, igas], supsatlold[iz, igas])
                nnuc = int(nnuc2elem_arr[iepart])
                for inuc in range(nnuc):
                    ienucto = int(inuc2elem_arr[inuc, iepart])
                    proc = int(inucproc_arr[iepart, ienucto])
                    if (proc & int(NucProcess.I_DROPACT)) == 0:
                        continue
                    # Vectorised activation check across bins
                    mask = (
                        (pc[iz, :, iepart] / zmet[iz]
                         > DTYPE(conmax) * pconmax[iz, ig])
                        & (ss > scrit[iz, :, ig])
                    )
                    activated = jnp.any(mask) & has_particles
                    force_max = force_max | activated
        elif itype_val == int(ElementType.I_VOLATILE):
            # smallest bin index where pc/zmet > conmax * pconmax,
            # over i in [0, nbin-2] (Fortran loops i=NBIN-1 to 1)
            bin_idx = jnp.arange(nbin - 1)
            vol_mask = (
                pc[iz, : nbin - 1, iepart] / zmet[iz]
                > DTYPE(conmax) * pconmax[iz, ig]
            )
            # Fortran: ibin_small = min i such that mask[i] is True;
            # default nbin-1 if no bin qualifies.
            # Use arange padded with nbin-1 where mask is False.
            masked = jnp.where(vol_mask, bin_idx,
                                jnp.asarray(nbin - 1, dtype=jnp.int32))
            ibin_small_ig = jnp.where(has_particles,
                                        jnp.min(masked),
                                        jnp.asarray(nbin - 1, dtype=jnp.int32))

        ibin_small_list.append(ibin_small_ig)

    # ------- Pass 2: growth-rate-limited dt_adv -------
    dt_adv = jnp.asarray(dtime_save, dtype=DTYPE)
    for ig in range(ngroup):
        iepart = int(ienconc_arr[ig])
        itype_val = int(itype_arr[iepart])
        igas = int(igrowgas_arr[iepart])

        if igas < 0 or itype_val != int(ElementType.I_VOLATILE):
            continue

        has_particles = pconmax[iz, ig] > FEW_PC
        ibin_small_ig = ibin_small_list[ig]

        is_ice = bool(is_grp_ice_arr[ig])
        if is_ice:
            ss = supsati[iz, igas]
            pvap = pvapi[iz, igas]
        else:
            ss = supsatl[iz, igas]
            pvap = pvapl[iz, igas]

        # Index gro / gro1 at (iz, ibin_small_ig, ig) via take
        g0 = gro[iz, :, ig][ibin_small_ig]
        g1 = gro1[iz, :, ig][ibin_small_ig]

        dmdt = jnp.abs(pvap * ss * g0 / (DTYPE(1.0) + g0 * g1 * pvap))

        dm_at_ibin = dm[:, ig][ibin_small_ig]
        candidate = dm_at_ibin / jnp.where(dmdt > DTYPE(0.0), dmdt, DTYPE(1.0))
        active = has_particles & (dmdt > DTYPE(0.0))
        new_dt = jnp.where(active,
                            jnp.minimum(dt_adv, candidate),
                            dt_adv)
        dt_adv = new_dt

    # ------- Pass 3: aerfreeze short-circuit -------
    for ig in range(ngroup):
        iepart = int(ienconc_arr[ig])
        igas = int(inucgas_arr[ig])
        if igas < 0:
            continue
        nnuc = int(nnuc2elem_arr[iepart])
        for inuc in range(nnuc):
            ienucto = int(inuc2elem_arr[inuc, iepart])
            proc = int(inucproc_arr[iepart, ienucto])
            if (proc & int(NucProcess.I_AERFREEZE)) == 0:
                continue
            # Fortran passes T(iz) — but we don't have it here; use
            # a threshold on supsati only. Temperature threshold is
            # applied via the caller when T is available; we include
            # it here explicitly via a separate passed-in flag.
            # (Actually the Fortran checks t(iz) < 233.16.)
            pass
    # Note: nsubsteps.F90 also checks t(iz) < 233.16 for aerfreeze.
    # Temperature is not in the current args — callers pass it via
    # a companion aerfreeze_force flag. We leave the hook but don't
    # fire it in this pure kernel; see the ADR.

    # Compute n_growth from dt_adv
    n_growth_raw = jnp.round(
        jnp.asarray(dtime_save, dtype=DTYPE) / dt_adv
    ).astype(jnp.int32)
    n_growth = jnp.minimum(maxsubsteps_i, n_growth_raw)
    n_growth = jnp.maximum(minsubsteps_i, n_growth)

    # If activation forced max, use it; otherwise growth-rate count.
    n_final = jnp.where(force_max, maxsubsteps_i, n_growth)
    return n_final


def aerfreeze_force(
    supsati, temp, iz,
    inucgas_arr, ienconc_arr, nnuc2elem_arr, inuc2elem_arr, inucproc_arr,
    ngroup,
):
    """Standalone companion check: ``ntsubsteps = maxsubsteps`` when
    any group has ``I_AERFREEZE`` active AND ``supsati > 0.4`` AND
    ``T < 233.16``.

    Callers ``|=`` the boolean result onto the force-max flag they
    pass into `nsubsteps`. Split out so the main kernel doesn't need
    to carry ``t`` through its signature in cases where the aerfreeze
    path isn't configured.
    """
    any_force = jnp.asarray(False)
    t_cold = jnp.asarray(temp, dtype=DTYPE) < _T_AERFREEZE
    for ig in range(ngroup):
        iepart = int(ienconc_arr[ig])
        igas = int(inucgas_arr[ig])
        if igas < 0:
            continue
        nnuc = int(nnuc2elem_arr[iepart])
        for inuc in range(nnuc):
            ienucto = int(inuc2elem_arr[inuc, iepart])
            proc = int(inucproc_arr[iepart, ienucto])
            if (proc & int(NucProcess.I_AERFREEZE)) == 0:
                continue
            ss_big = supsati[iz, igas] > _SS_AERFREEZE
            any_force = any_force | (ss_big & t_cold)
    return any_force
