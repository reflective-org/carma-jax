"""Sulfate nucleation driver (homogeneous + heterogeneous).

Ports the per-group, per-level logic of ``sulfnuc.F90``:

1. **Homogeneous path**. Calls the selected rate (Zhao-Turco or
   Vehkamaki) once to obtain (nucrate, nucbin, rstar). Accumulates
   ``rhompe[nucbin] += nucrate``.

2. **Heterogeneous path**. Runs only if the homogeneous Zhao-Turco
   call produced a positive critical radius ``rstar``. For every
   candidate seed bin, ``sulfhetnucrate`` is vmapped over the bin
   axis to produce ``rnuclg[bin]``.

The driver is a pure function for a single column/level and a single
aerosol group. Caller composes over (iz, igroup) via ``jax.vmap`` or
Python loops in the Phase-7.6 orchestrator.

Unit bookkeeping is explicit:

- ``nucrate_cgs`` from the rate primitives is ``# cm⁻³ s⁻¹``
  (homogeneous volumetric rate) or ``# s⁻¹ per seed``
  (heterogeneous, bin-local). The homogeneous rate is multiplied by
  ``zmet`` to match the Fortran internal ``# z⁻¹ s⁻¹`` convention
  before being slotted into the bin-indexed ``rhompe`` array.

Ported from: sulfnuc.F90, sulfnucrate.F90 (dispatcher branch).
"""

import jax
import jax.numpy as jnp

from carma.constants import PI, RGAS
from carma.nucleation.sulfhetnucrate import sulfhetnucrate
from carma.nucleation.sulfnucrate import (
    binary_nuc_vehk2002,
    binary_nuc_zhao1995,
)
from carma.precision import DTYPE


_GWTMOL_H2SO4 = DTYPE(98.0)
_GWTMOL_H2O = DTYPE(18.0)


def _collision_betas(temp, gwtmol_h2so4=_GWTMOL_H2SO4, gwtmol_h2o=_GWTMOL_H2O):
    """Gas-kinetic collision rate coefficients β1 (H2SO4) and β2 (H2O).

    From sulfnucrate.F90: ``β = √(R T / (2π · M))``.
    """
    rb = RGAS * temp / DTYPE(2.0) / PI
    beta1 = jnp.sqrt(rb / gwtmol_h2so4)
    beta2 = jnp.sqrt(rb / gwtmol_h2o)
    return beta1, beta2


def _nucbin_from_mass(mass_cluster_dry, rmassup, rmrat_val):
    """Locate the target bin for a critical cluster of given dry mass.

    Fortran convention (1-based): bin = 1 if mass < rmassup[0];
    otherwise bin = 2 + floor(log(mass / rmassup[0]) / log(rmrat)).
    Converted to 0-based. May return ``nbin`` (one past the end) to
    signal "larger than every bin" — caller must gate.
    """
    nbin = rmassup.shape[0]
    raw = DTYPE(1.0) + jnp.floor(
        jnp.log(mass_cluster_dry / rmassup[0]) / jnp.log(DTYPE(rmrat_val))
    )
    idx = jnp.where(
        mass_cluster_dry < rmassup[0],
        jnp.asarray(0, dtype=jnp.int32),
        raw.astype(jnp.int32),
    )
    # Fortran gates nucrate on nucbin <= NBIN, so idx == nbin is "no bin".
    return jnp.clip(idx, 0, nbin)


def homogeneous_nucleation(
    temp, weight_percent, rh,
    h2so4, h2so4_cgs, h2o, h2o_cgs,
    rmassup, rmrat_val, zmet,
    method="ZhaoTurco",
    gwtmol_h2so4=_GWTMOL_H2SO4,
    gwtmol_h2o=_GWTMOL_H2O,
):
    """Homogeneous H2SO4/H2O nucleation rate + target bin.

    Args:
        temp, weight_percent, rh: atmospheric state.
        h2so4, h2so4_cgs: H2SO4 gas [molec/cm³] and [g/cm³].
        h2o, h2o_cgs: H2O gas [molec/cm³] and [g/cm³].
        rmassup: upper-bin-boundary masses [g], shape ``(nbin,)``.
        rmrat_val: bin mass ratio (scalar).
        zmet: vertical metric converting cm⁻³ → z⁻¹.
        method: ``"ZhaoTurco"`` or ``"Vehkamaki"``.

    Returns:
        nucrate_z: homogeneous rate in internal units [# z⁻¹ s⁻¹].
        nucbin: 0-based target bin index (or ``nbin`` if cluster
            exceeds every bin — rate is zeroed in that case).
        rstar: critical-cluster radius [cm] (0 when no saddle /
            Vehkamaki path with sub-threshold H2SO4).
        ftry: Zhao-Turco critical-Gibbs exponent (0 for Vehkamaki).
    """
    beta1, _ = _collision_betas(temp, gwtmol_h2so4, gwtmol_h2o)

    if method == "ZhaoTurco":
        nucrate_cgs, mass_cluster_dry, rstar, ftry = binary_nuc_zhao1995(
            temp, weight_percent, rh, h2so4, h2so4_cgs, h2o, h2o_cgs,
            beta1, gwtmol_h2so4, gwtmol_h2o,
        )
    elif method == "Vehkamaki":
        nucrate_cgs, mass_cluster_dry, rstar = binary_nuc_vehk2002(
            temp, rh, h2so4, gwtmol_h2so4,
        )
        # Vehkamaki doesn't expose ftry; heterogeneous path not supported
        # via this branch.
        ftry = jnp.zeros_like(nucrate_cgs)
        # Vehkamaki skipped below 1e4 cm⁻³ (Fortran skips the call itself).
        nucrate_cgs = jnp.where(
            jnp.asarray(h2so4, dtype=DTYPE) >= DTYPE(1e4),
            nucrate_cgs, DTYPE(0.0),
        )
    else:
        raise ValueError(
            f"sulfnuc: unknown homogeneous method {method!r}; "
            "expected 'ZhaoTurco' or 'Vehkamaki'."
        )

    nbin = rmassup.shape[0]
    nucbin = _nucbin_from_mass(mass_cluster_dry, rmassup, rmrat_val)
    nucrate_z = nucrate_cgs * DTYPE(zmet)
    # Zero if critical mass exceeds every bin.
    nucrate_z = jnp.where(nucbin < nbin, nucrate_z, DTYPE(0.0))
    return nucrate_z, nucbin, rstar, ftry


def heterogeneous_nucleation(
    temp, weight_percent, rh,
    h2so4, h2so4_cgs, h2o, h2o_cgs,
    r_bins,
    gwtmol_h2so4=_GWTMOL_H2SO4,
    gwtmol_h2o=_GWTMOL_H2O,
):
    """Heterogeneous nucleation rate per bin, for a single group.

    Only produces non-zero rates when the Zhao-Turco path finds a
    saddle (rstar > 0); ``sulfhetnucrate`` handles the gate internally.

    Args:
        r_bins: wet radii of this group's bins [cm], shape ``(nbin,)``.

    Returns:
        rnuclg: heterogeneous nucleation rate for each seed bin
            [embryos / seed / s], shape ``(nbin,)``. Multiply by
            particle number per bin to get bin-wise embryo-production.
    """
    beta1, beta2 = _collision_betas(temp, gwtmol_h2so4, gwtmol_h2o)

    def _rate_for_bin(r_pre):
        return sulfhetnucrate(
            temp, weight_percent, rh,
            h2so4, h2so4_cgs, h2o, h2o_cgs,
            beta1, beta2, r_pre,
            gwtmol_h2so4=gwtmol_h2so4,
            gwtmol_h2o=gwtmol_h2o,
        )

    return jax.vmap(_rate_for_bin)(jnp.asarray(r_bins, dtype=DTYPE))


def sulfnuc(
    temp, weight_percent, rh,
    h2so4, h2so4_cgs, h2o, h2o_cgs,
    r_bins, rmassup, rmrat_val, zmet,
    method="ZhaoTurco",
    do_homogeneous=True,
    do_heterogeneous=True,
    gwtmol_h2so4=_GWTMOL_H2SO4,
    gwtmol_h2o=_GWTMOL_H2O,
):
    """Sulfate nucleation driver for a single group, single level.

    Mirrors the Fortran ``sulfnuc`` loop body (one iteration) — runs
    both homogeneous and heterogeneous nucleation, returning the
    bin-indexed production rates that ``newstate_calc`` needs.

    The heterogeneous path is automatically gated on the homogeneous
    Zhao-Turco ``rstar > 0``: when the saddle search fails, all
    rnuclg entries are zero. When ``do_heterogeneous`` is False, the
    gate is overridden with zeros regardless of rstar.

    When ``method='Vehkamaki'``, heterogeneous nucleation cannot run
    (no ftry/rstar in the Vehkamaki parameterisation); rnuclg is zero.
    Caller must select ZhaoTurco to get heterogeneous rates.

    Args:
        temp: Temperature [K].
        weight_percent: H2SO4 weight percent (0-100).
        rh: Relative humidity over liquid water (0-1).
        h2so4, h2so4_cgs: H2SO4 gas [molec/cm³] and [g/cm³].
        h2o, h2o_cgs: H2O gas [molec/cm³] and [g/cm³].
        r_bins: wet-radius per bin [cm], shape ``(nbin,)``.
        rmassup: upper-bin-boundary masses [g], shape ``(nbin,)``.
        rmrat_val: mass ratio between adjacent bins (scalar).
        zmet: vertical metric [cm / z-unit].
        method: ``'ZhaoTurco'`` (default) or ``'Vehkamaki'``.
        do_homogeneous, do_heterogeneous: disable either branch.

    Returns:
        rhompe: homogeneous bin-wise production [# z⁻¹ s⁻¹], shape
            ``(nbin,)``. All zero except at the target nucbin.
        rnuclg: heterogeneous per-seed production [embryos / seed / s],
            shape ``(nbin,)``. Caller multiplies by ``pc[bin]`` to get
            a total rate.
    """
    nbin = rmassup.shape[0]

    # --- Homogeneous ---
    nucrate_z, nucbin, rstar, _ = homogeneous_nucleation(
        temp, weight_percent, rh, h2so4, h2so4_cgs, h2o, h2o_cgs,
        rmassup, rmrat_val, zmet, method=method,
        gwtmol_h2so4=gwtmol_h2so4, gwtmol_h2o=gwtmol_h2o,
    )
    if not do_homogeneous:
        nucrate_z = jnp.zeros_like(nucrate_z)

    bin_idx = jnp.arange(nbin, dtype=jnp.int32)
    rhompe = jnp.where(bin_idx == nucbin, nucrate_z, DTYPE(0.0))

    # --- Heterogeneous ---
    if do_heterogeneous and method == "ZhaoTurco":
        rnuclg = heterogeneous_nucleation(
            temp, weight_percent, rh, h2so4, h2so4_cgs, h2o, h2o_cgs,
            r_bins,
            gwtmol_h2so4=gwtmol_h2so4, gwtmol_h2o=gwtmol_h2o,
        )
        # Redundant with sulfhetnucrate's internal gate on rstar>0,
        # but makes the do_het=False override explicit:
        rnuclg = jnp.where(rstar > DTYPE(0.0), rnuclg, DTYPE(0.0))
    else:
        rnuclg = jnp.zeros(nbin, dtype=DTYPE)

    return rhompe, rnuclg
