"""Single-column sulfate microphysics step.

Composes Phase 7.1-7.6a primitives into one advance of a column:

    wtpct → sulfnuc → gasexchange → explicit Euler apply → (pc, gc)

The step is a pure function; the ``make_step_sulfate`` factory binds
a ``CarmaConfig`` + bin tables to it and returns a JIT-compiled
substepping closure.

Scope
-----

This is the minimum-viable sulfate kernel for a single group with a
single condensing gas (H2SO4):

- Homogeneous nucleation → particles produced in ``nucbin``.
- Heterogeneous nucleation → number transferred from seed bin ``i``
  into ``inuc2bin[i]``; both arrays are taken from the caller-supplied
  setup (typically ``i → i+1`` for monotone sulfate growth onto self).
- H2SO4 gas balance via ``gasexchange``.
- H2O gas, temperature, and latent-heat feedback are NOT updated —
  the driver treats them as prescribed, matching ``carma_sulfatetest``
  which runs at fixed T, p, and RH. The temperature and gas arrays
  appear in the signature so callers can swap in a full energy /
  H2O-balance update later without changing the kernel interface.

Mass conservation
-----------------

Total sulfate mass (gas ``h2so4_cgs`` + particle mass
``Σ pc[i] · rmass[i]``) is conserved to floating-point precision:
``gasprod`` from ``gasexchange`` is exactly the negative of the
particle-side mass gain ``rhompe · rmass + Σ pc · rnuclg · Δm`` by
construction.

Unit conventions
----------------

- Internal state is in CGS. ``gc`` is ``g / cm³ / z`` (the Fortran
  convention; divide by ``zmet`` to get ``g / cm³``).
- ``pc`` is ``# / cm³ / z``.
- ``rmass`` is ``g`` per particle per bin.

Ported as the integration of: ``sulfnuc.F90`` (driver),
``gasexchange.F90`` (gas flux), and the explicit-Euler particle
update that Fortran's ``psolve`` performs for the nucleation branch.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from carma.constants import AVG, BK, RGAS, WTMOL_H2O
from carma.gasexchange import gasexchange
from carma.nucleation.sulfnuc import sulfnuc
from carma.precision import DTYPE
from carma.sulfate_utils import wtpct_tabaz


# Static (Python-level) config arrays for the minimal single-group
# sulfate problem. Kept outside jit'd code so gasexchange sees
# Python ints when it reads ``if_nuc[0, 0]`` etc.
_IF_NUC_SULF = np.asarray([[True]])
_IENCONC_SULF = np.asarray([0])
_IGELEM_SULF = np.asarray([0])
_INUCGAS_SULF = np.asarray([0])
_NNUC2ELEM_SULF = np.asarray([1])
_IGROWGAS_DISABLED = np.asarray([-1])


_GWTMOL_H2SO4 = DTYPE(98.0)
_GWTMOL_H2O = DTYPE(18.0)


def _compute_state(gc_h2so4, gc_h2o, temp, pvapl, zmet):
    """Convert column gas state into the molar quantities sulfnuc needs.

    Returns ``(h2so4_num, h2so4_cgs, h2o_num, h2o_cgs, rh, wtp)``.
    """
    h2so4_cgs = gc_h2so4 / zmet
    h2o_cgs = gc_h2o / zmet

    h2so4_num = h2so4_cgs * AVG / _GWTMOL_H2SO4
    h2o_num = h2o_cgs * AVG / _GWTMOL_H2O

    # Relative humidity = (gc_h2o_cgs · R_vap · T) / pvapl
    rvap = RGAS / _GWTMOL_H2O
    rh = (h2o_cgs * rvap * temp) / pvapl

    # Activity for wtpct_tabaz = gc_h2o / (gc_h2o + h2o at saturation)
    h2o_mass_wtp = rh * pvapl * _GWTMOL_H2O / (BK * temp * AVG)
    wtp = wtpct_tabaz(temp, h2o_mass_wtp, pvapl)

    return h2so4_num, h2so4_cgs, h2o_num, h2o_cgs, rh, wtp


def sulfate_step_one_level(
    pc, gc_h2so4, gc_h2o, temp, pvapl, zmet, dtime,
    r_bins, rmass, rmassup, diffmass, inuc2bin,
    rmrat_val,
    method="ZhaoTurco",
    do_homogeneous=True,
    do_heterogeneous=True,
):
    """Advance a single-column sulfate state by ``dtime``.

    Args:
        pc: Particle number concentration, shape ``(nbin,)``.
        gc_h2so4: H2SO4 gas concentration [g/cm³/z] (scalar).
        gc_h2o: H2O gas concentration [g/cm³/z] (scalar).
        temp: Temperature [K] (scalar).
        pvapl: Saturation vapour pressure over liquid water [dyn/cm²].
        zmet: Vertical metric [cm/z-unit] (scalar).
        dtime: Timestep [s] (scalar).
        r_bins: Wet radius per bin [cm], shape ``(nbin,)``.
        rmass: Particle mass per bin [g], shape ``(nbin,)``.
        rmassup: Upper bin boundary mass [g], shape ``(nbin,)``.
        diffmass: Target-minus-source mass,
            shape ``(nbin, 1, nbin, 1)`` for single-group sulfate.
        inuc2bin: Heterogeneous target bin per source bin,
            shape ``(nbin, 1, 1)`` int (``-1`` for no target).
        rmrat_val: Bin mass ratio (scalar).
        method: ``"ZhaoTurco"`` (default) or ``"Vehkamaki"``.
        do_homogeneous, do_heterogeneous: disable either branch.

    Returns:
        ``(pc_new, gc_h2so4_new, diag)`` where ``diag`` is a dict with
        ``rhompe``, ``rnuclg``, ``gasprod``.
    """
    nbin = pc.shape[0]

    h2so4_num, h2so4_cgs, h2o_num, h2o_cgs, rh, wtp = _compute_state(
        gc_h2so4, gc_h2o, temp, pvapl, zmet,
    )

    # --- Rates ---
    rhompe_1d, rnuclg_1d = sulfnuc(
        temp=temp, weight_percent=wtp, rh=rh,
        h2so4=h2so4_num, h2so4_cgs=h2so4_cgs,
        h2o=h2o_num, h2o_cgs=h2o_cgs,
        r_bins=r_bins, rmassup=rmassup, rmrat_val=rmrat_val, zmet=zmet,
        method=method,
        do_homogeneous=do_homogeneous,
        do_heterogeneous=do_heterogeneous,
    )

    # Reshape into gasexchange's expected layout (nbin, nelem=1),
    # (nbin, ngroup=1, ngroup=1), etc.
    rhompe = rhompe_1d[:, None]
    rnuclg = rnuclg_1d[:, None, None]
    pc_iz = pc[:, None]
    rmass_2d = rmass[:, None]

    # Gas exchange — single group, single gas, grow branch off (no growth
    # in the minimal sulfate step; it's handled by the growth kernel in
    # the full model but omitted here to keep the Phase-7 kernel scoped
    # to the pure nucleation flux).
    gasprod = gasexchange(
        pc_iz=pc_iz, rhompe=rhompe, rnuclg=rnuclg,
        growlg=jnp.zeros((nbin, 1), dtype=DTYPE),
        evaplg=jnp.zeros((nbin, 1), dtype=DTYPE),
        rmass=rmass_2d, diffmass=diffmass,
        cmf=jnp.zeros((nbin, 1), dtype=DTYPE),
        totevap=jnp.zeros((nbin, 1), dtype=bool),
        inuc2bin=inuc2bin,
        if_nuc=_IF_NUC_SULF,
        ienconc=_IENCONC_SULF,
        igelem=_IGELEM_SULF,
        inucgas=_INUCGAS_SULF,
        nnuc2elem=_NNUC2ELEM_SULF,
        igrowgas=_IGROWGAS_DISABLED,
        ngas=1, ngroup=1, nelem=1, nbin=nbin,
    )
    gasprod_h2so4 = gasprod[0]

    # --- Apply rates (semi-implicit + mass-conserving gate) ---
    #
    # Nucleation rates can be fast compared to the 60-s outer timestep
    # (a saturated stratospheric column can deplete H2SO4 in under a
    # second). The Fortran ``psolve`` uses the implicit loss form
    # ``pc_new = pc / (1 + dt · loss_rate)`` — we mirror that for the
    # heterogeneous transfer and gate the whole mass flux on the
    # available gas to preserve ``Δ(gc) = -Δ(particle mass)``
    # exactly.

    # Heterogeneous: semi-implicit transfer. Fraction of each bin that
    # transfers out over [0, dt] is ``dt·λ / (1 + dt·λ)``.
    lam = rnuclg_1d                                   # per-seed rate [1/s]
    het_frac = dtime * lam / (DTYPE(1.0) + dtime * lam)
    het_num_transferred = pc * het_frac               # (nbin,)

    src_tgt = inuc2bin[:, 0, 0]                       # (nbin,) int
    has_target = src_tgt >= 0
    src_tgt_safe = jnp.where(has_target, src_tgt, 0)
    het_contrib = jnp.where(has_target, het_num_transferred, DTYPE(0.0))

    diffmass_het = jnp.diagonal(
        jnp.diagonal(diffmass, axis1=1, axis2=3), axis1=0, axis2=1
    )  # (nbin,) — mass gained per seed
    diffmass_het = jnp.where(has_target, diffmass_het, DTYPE(0.0))

    # Homogeneous mass demand: rhompe[nucbin] particles of rmass[nucbin]
    hom_mass_demand = jnp.sum(dtime * rhompe_1d * rmass)
    het_mass_demand = jnp.sum(het_contrib * diffmass_het)
    total_mass_demand = hom_mass_demand + het_mass_demand

    # Scale so we never consume more gas than exists.
    scale = jnp.where(
        total_mass_demand > DTYPE(1e-300),
        jnp.minimum(DTYPE(1.0), gc_h2so4 / total_mass_demand),
        DTYPE(1.0),
    )

    # Apply scaled rates.
    pc_new = pc + scale * dtime * rhompe_1d

    het_gain = jnp.zeros(nbin, dtype=DTYPE).at[src_tgt_safe].add(het_contrib)
    pc_new = pc_new + scale * (het_gain - het_contrib)
    pc_new = jnp.maximum(pc_new, DTYPE(0.0))

    gc_h2so4_new = gc_h2so4 - scale * total_mass_demand
    gc_h2so4_new = jnp.maximum(gc_h2so4_new, DTYPE(0.0))

    diag = {
        "rhompe": rhompe_1d,
        "rnuclg": rnuclg_1d,
        "gasprod_h2so4": gasprod_h2so4,
    }
    return pc_new, gc_h2so4_new, diag


def make_step_sulfate(config, igroup=0, igas_h2so4=1, igas_h2o=0,
                      method="ZhaoTurco", ntsubsteps=1):
    """Build a JIT'd single-column sulfate stepper from a ``CarmaConfig``.

    Args:
        config: CarmaConfig (provides bin tables and group geometry).
        igroup: Group index hosting the sulfate particles (default 0).
        igas_h2so4: Gas-index of H2SO4 in ``gc``. Defaults to 1
            (matches carma_sulfatetest, which has ``igash2so4=2`` in
            Fortran's 1-based indexing).
        igas_h2o: Gas-index of H2O in ``gc``. Defaults to 0.
        method: ``"ZhaoTurco"`` (default) or ``"Vehkamaki"``.
        ntsubsteps: Static micro-substeps per call.

    Usage:
        step_sulfate = make_step_sulfate(config)
        for istep in range(nstep):
            pc, gc, diag = step_sulfate(
                pc, gc, temp, pvapl, zmet, dtime,
            )

    Returns a closure accepting ``(pc, gc, temp, pvapl, zmet, dtime)``
    that returns ``(pc_new, gc_new, diag)``. ``pc`` is ``(nbin,)`` for
    the nucleating group; ``gc`` is ``(ngas,)``.
    """
    group = config.groups[igroup]
    nbin = int(config.nbin)

    r_bins = jnp.asarray(group.r, dtype=DTYPE)
    rmass = jnp.asarray(group.rmass, dtype=DTYPE)
    rmassup = jnp.asarray(group.rmassup, dtype=DTYPE)
    rmrat_val = float(group.rmrat)

    # Build diffmass(tgt_bin, 1, src_bin, 1) from rmass.
    rmass_col = rmass[:, None]
    diffmass = (rmass_col[:, :, None, None]
                - rmass_col[None, None, :, :])
    diffmass = diffmass.astype(DTYPE)

    # Heterogeneous target mapping: sulfate-onto-self grows bin i → i+1,
    # top bin has no target.
    i2_map = jnp.arange(nbin, dtype=jnp.int32) + 1
    i2_map = jnp.where(i2_map < nbin, i2_map,
                        jnp.asarray(-1, dtype=jnp.int32))
    inuc2bin = i2_map.reshape(nbin, 1, 1)

    _nts = int(ntsubsteps)

    @jax.jit
    def step_fn(pc, gc, temp, pvapl, zmet, dtime):
        dt_sub = DTYPE(dtime) / DTYPE(_nts)

        def body(_, carry):
            pc_c, gc_c = carry
            gc_h2so4 = gc_c[igas_h2so4]
            gc_h2o = gc_c[igas_h2o]
            pc_c, gc_h2so4_new, _ = sulfate_step_one_level(
                pc_c, gc_h2so4, gc_h2o, temp, pvapl, zmet, dt_sub,
                r_bins, rmass, rmassup, diffmass, inuc2bin,
                rmrat_val, method=method,
            )
            gc_c = gc_c.at[igas_h2so4].set(gc_h2so4_new)
            return (pc_c, gc_c)

        pc_f, gc_f = jax.lax.fori_loop(0, _nts, body, (pc, gc))

        # One extra call to return the diag from the final substep
        gc_h2so4 = gc_f[igas_h2so4]
        gc_h2o = gc_f[igas_h2o]
        _, _, diag = sulfate_step_one_level(
            pc_f, gc_h2so4, gc_h2o, temp, pvapl, zmet, DTYPE(0.0),
            r_bins, rmass, rmassup, diffmass, inuc2bin,
            rmrat_val, method=method,
        )
        return pc_f, gc_f, diag

    return step_fn
