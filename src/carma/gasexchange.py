"""Gas-phase production rates from nucleation + growth/evaporation.

Ported from ``gasexchange.F90``. Returns the gas-production vector
``gasprod[igas] = Σ_group (gprod_nuc + gprod_grow)`` at a single
vertical level.

**Nucleation path** (for each group that has a condensing gas and a
non-empty nucleation-target element list):

- Homogeneous: every bin loses gas equal to ``rhompe[i] · rmass[i]``
  (the mass flux into the freshly-nucleated cluster in bin ``i``).
- Heterogeneous: seed bin ``i`` loses gas equal to
  ``pc[i] · rnuclg[i] · diffmass[i2, ig2, i, igroup]`` where
  ``i2 = inuc2bin[i, igroup, ig2]`` is the target bin; the gate
  ``i2 == -1`` (Fortran's ``0``) disables that transfer.

**Growth/evaporation path** (for each element with a growth gas):

- For ``i in [0, nbin-2]``:
  - Evap contribution: ``evaplg[i+1] · pc[i+1] · gasgain``
    where ``gasgain = (1 - cmf[i+1]) · rmass[i+1]`` if ``totevap[i+1]``
    else ``diffmass[i+1, i]``.
  - Growth contribution: ``-growlg[i] · pc[i] · diffmass[i+1, i]``.
- Plus the total-evaporation term out of bin 0:
  ``evaplg[0] · pc[0] · (1 - cmf[0]) · rmass[0]``.

The function is a pure single-level call; caller vmaps over ``iz``
or the factory binds ``iz`` through a lax.fori_loop.

Fortran-to-JAX conventions:

- All indices are 0-based (Fortran ``0`` sentinel → ``-1``).
- Static dimensions (``ngroup``, ``nelem``, ``ngas``) resolve as
  Python ints at trace time; we therefore use Python loops over
  those axes and vectorised array ops over ``nbin``.

This port covers the code path that's live today. The commented-out
latent-heat branch in gasexchange.F90 is not ported; ``rlprod`` from
latent heating is handled downstream by ``gsolve``.
"""

import jax.numpy as jnp

from carma.precision import DTYPE


def gasexchange(
    pc_iz,
    rhompe,
    rnuclg,
    growlg,
    evaplg,
    rmass,
    diffmass,
    cmf,
    totevap,
    inuc2bin,
    if_nuc,
    ienconc,
    igelem,
    inucgas,
    nnuc2elem,
    igrowgas,
    ngas,
    ngroup,
    nelem,
    nbin,
):
    """Gas production vector at one vertical level.

    Args:
        pc_iz: Particle concentrations at this level, shape
            ``(nbin, nelem)``.
        rhompe: Homogeneous production rate, shape ``(nbin, nelem)``.
        rnuclg: Heterogeneous nucleation rate per seed,
            shape ``(nbin, ngroup, ngroup)``. Last axis is the target
            group index (``ig2``).
        growlg: Growth loss rate, shape ``(nbin, ngroup)``.
        evaplg: Evaporation loss rate, shape ``(nbin, ngroup)``.
        rmass: Bin mass, shape ``(nbin, ngroup)``.
        diffmass: Mass difference between target ``(i2, ig2)`` and
            source ``(i, igroup)``, shape ``(nbin, ngroup, nbin, ngroup)``.
        cmf: Core mass fraction, shape ``(nbin, ngroup)``.
        totevap: Total-evap flag, shape ``(nbin, ngroup)``, bool.
        inuc2bin: Target bin for heterogeneous nucleation,
            shape ``(nbin, ngroup, ngroup)``. ``-1`` means no target.
        if_nuc: Per-element nucleation map, shape ``(nelem, nelem)``,
            bool. ``if_nuc[ielem, ienuc2]`` → does ``ielem`` nucleate
            particles into ``ienuc2``.
        ienconc: Number-concentration element per group,
            shape ``(ngroup,)`` int.
        igelem: Group of each element, shape ``(nelem,)`` int.
        inucgas: Condensing gas of each group, shape ``(ngroup,)`` int;
            ``-1`` means no condensing gas.
        nnuc2elem: Number of nucleation-target elements per element,
            shape ``(nelem,)`` int.
        igrowgas: Growth gas of each element, shape ``(nelem,)`` int;
            ``-1`` means no growth gas.
        ngas, ngroup, nelem, nbin: Static dimensions (Python ints).

    Returns:
        ``gasprod[igas]`` of shape ``(ngas,)``, in units of
        ``g / cm³ / z / s`` (sign convention: positive means gas
        is being produced, negative means consumed).
    """
    pc_iz = jnp.asarray(pc_iz, dtype=DTYPE)
    gasprod = jnp.zeros(ngas, dtype=DTYPE)

    for igroup in range(ngroup):
        igas_nuc = int(inucgas[igroup])
        ielem_num = int(ienconc[igroup])
        nnuc_count = int(nnuc2elem[ielem_num])

        # --- Nucleation branch ---
        if igas_nuc >= 0 and nnuc_count > 0:
            for ienuc2 in range(nelem):
                if not bool(if_nuc[ielem_num, ienuc2]):
                    continue
                ig2 = int(igelem[ienuc2])

                # Homogeneous: sum over bins of rhompe · rmass
                hom_term = jnp.sum(
                    rhompe[:, ielem_num] * rmass[:, igroup]
                )

                # Heterogeneous: sum over source bins of
                #   pc · rnuclg · diffmass[target_bin, ig2, src_bin, igroup]
                # where target_bin = inuc2bin[src_bin, igroup, ig2];
                # -1 targets contribute 0.
                i2_vec = inuc2bin[:, igroup, ig2]                    # (nbin,)
                has_target = i2_vec >= 0
                # Gather diffmass at (target_bin, ig2) for every src i:
                #   diffmass[i2_vec[i], ig2, i, igroup]
                # Use jnp.take along the first axis, then index the
                # remaining axes per-i with advanced indexing.
                i2_safe = jnp.where(has_target, i2_vec, 0)
                # diffmass[i2_safe, ig2, arange(nbin), igroup]
                bin_arange = jnp.arange(nbin)
                dm_gather = diffmass[i2_safe, ig2, bin_arange, igroup]  # (nbin,)
                dm_gather = jnp.where(has_target, dm_gather, DTYPE(0.0))

                het_term = jnp.sum(
                    pc_iz[:, ielem_num]
                    * rnuclg[:, igroup, ig2]
                    * dm_gather
                )

                gasprod = gasprod.at[igas_nuc].add(-(hom_term + het_term))

        # --- Growth/evap branch ---
        igas_grow = int(igrowgas[ielem_num])
        if igas_grow >= 0:
            # Bins 0..nbin-2 (Fortran 1..nbin-1)
            # diffmass[i+1, igroup, i, igroup]
            dm_up = diffmass[1:, igroup, :-1, igroup].diagonal()      # (nbin-1,)

            totevap_up = totevap[1:, igroup]                           # (nbin-1,)
            cmf_up = cmf[1:, igroup]
            rmass_up = rmass[1:, igroup]
            pc_up = pc_iz[1:, ielem_num]
            evaplg_up = evaplg[1:, igroup]

            gasgain = jnp.where(
                totevap_up,
                (DTYPE(1.0) - cmf_up) * rmass_up,
                dm_up,
            )
            evap_term = jnp.sum(evaplg_up * pc_up * gasgain)

            # Growth from bin i into bin i+1
            growth_term = jnp.sum(
                growlg[:-1, igroup]
                * pc_iz[:-1, ielem_num]
                * dm_up
            )

            # Total-evap out of bin 0
            bin0_evap = (
                evaplg[0, igroup] * pc_iz[0, ielem_num]
                * (DTYPE(1.0) - cmf[0, igroup]) * rmass[0, igroup]
            )

            gasprod = gasprod.at[igas_grow].add(
                evap_term - growth_term + bin0_evap
            )

    return gasprod
