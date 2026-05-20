"""3-group CARMA config: sulfate + cloud_water + ice_crystal.

Mirrors the structure of `scripts/jax_ensemble.py:_fortran_matching_config`
but adds two extra groups. All three groups share the same bin grid
(rmin = 2e-8 cm, rmrat = 2, nbin = 38) — adequate for a first cut, even
though cloud water and ice are usually carried on a coarser, larger-radius
grid in production. Group-specific density and is_ice/is_cloud flags
differentiate the physics.
"""
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from carma.bins import setup_bins
from carma.coagulation.setup_coag import setup_coag
from carma.config import (
    CarmaConfig, ElementConfig, GasConfig, GroupConfig, SoluteConfig,
)
from carma.precision import DTYPE


# Same bin grid as the Fortran sulfate ensemble — keeps the existing
# realistic_scenarios_100 inputs valid for the sulfate sub-group of the
# multispecies config.
_NBIN = 38
_RMIN_CM = 2.0e-8
_RMRAT = 2.0

# Group densities [g/cm^3]
_RHO_SULFATE = 1.923    # H2SO4-H2O at strat conditions (Fortran default)
_RHO_WATER = 1.000      # liquid water
_RHO_ICE = 0.917        # ice at 273 K

# Molecular weights [g/mol]
_GWTMOL_H2O = 18.01528
_GWTMOL_H2SO4 = 98.078479

# Gas indices in the multispecies layout
_IGAS_H2O = 0
_IGAS_H2SO4 = 1

# Group indices
_IGROUP_SULFATE = 0
_IGROUP_CLOUD_WATER = 1
_IGROUP_ICE = 2

# Element indices (one number-element per group)
_IELEM_SULFATE = 0
_IELEM_CLOUD_WATER = 1
_IELEM_ICE = 2


class MultiSpeciesConfig(NamedTuple):
    """Static, JIT-safe bundle of CARMA config + extra index arrays.

    The `cfg` field is a stock `CarmaConfig`. The remaining fields are
    derived once at config build time and used inside the RHS to dispatch
    per-group physics without re-deriving them on every call.
    """
    cfg: CarmaConfig
    # element-indexed arrays (length nelem):
    igrowgas: np.ndarray            # (nelem,) — growth gas index per element
    igelem: np.ndarray              # (nelem,) — group index per element
    # group-indexed arrays (length ngroup):
    is_ice: np.ndarray              # (ngroup,) — bool
    is_cloud: np.ndarray            # (ngroup,) — bool
    is_sulfate: np.ndarray          # (ngroup,) — bool
    ienconc: np.ndarray             # (ngroup,) — number-element index
    # per-group bin geometry stacked (ngroup, nbin):
    rmass_2d: jnp.ndarray           # bin-center mass [g]
    dm_2d: jnp.ndarray              # bin true width [g] = rmassup - rmasslow
    rmassup_2d: jnp.ndarray         # upper bin boundary mass [g]
    r_2d: jnp.ndarray               # bin-center radius [cm]


def _build_group(name, ienconc, is_ice, is_cloud, is_sulfate, rho):
    """One GroupConfig with the standard bin grid."""
    r, rmass, vol, dr, dm, rup, rlow, rmassup = setup_bins(
        _RMIN_CM, _RMRAT, _NBIN, rho,
    )
    return GroupConfig(
        name=name, ishape=1, ienconc=ienconc,
        is_ice=is_ice, is_cloud=is_cloud, is_sulfate=is_sulfate,
        do_vtran=False, do_drydep=False,
        ifallrtn=1, irhswell=(3 if is_sulfate else 0),
        rmrat=_RMRAT, eshape=1.0, rmin=_RMIN_CM,
        r=r, rmass=rmass, vol=vol, dr=dr, dm=dm,
        rmassup=rmassup, rup=rup, rlow=rlow,
        rrat=jnp.ones(_NBIN), rprat=jnp.ones(_NBIN), arat=jnp.ones(_NBIN),
    )


def make_multispecies_config() -> MultiSpeciesConfig:
    """Build a 3-group sulfate+water+ice CARMA config plus dispatch arrays.

    Returns:
        MultiSpeciesConfig containing the CarmaConfig and the per-element /
        per-group dispatch arrays needed by the multi-species RHS.
    """
    # --- Groups ---
    g_sulf = _build_group(
        "sulfate", ienconc=_IELEM_SULFATE,
        is_ice=False, is_cloud=False, is_sulfate=True,
        rho=_RHO_SULFATE,
    )
    g_water = _build_group(
        "cloud_water", ienconc=_IELEM_CLOUD_WATER,
        is_ice=False, is_cloud=True, is_sulfate=False,
        rho=_RHO_WATER,
    )
    g_ice = _build_group(
        "ice_crystal", ienconc=_IELEM_ICE,
        is_ice=True, is_cloud=True, is_sulfate=False,
        rho=_RHO_ICE,
    )
    groups = (g_sulf, g_water, g_ice)

    # --- Elements (one number element per group, no coated particles) ---
    e_sulf = ElementConfig(
        name="sulfate_num", rho=jnp.full((_NBIN,), _RHO_SULFATE),
        igroup=_IGROUP_SULFATE,
        itype=2, icomposition=0, isolute=0, kappa=0.65,
    )
    e_water = ElementConfig(
        name="cloud_water_num", rho=jnp.full((_NBIN,), _RHO_WATER),
        igroup=_IGROUP_CLOUD_WATER,
        itype=2, icomposition=1, isolute=-1, kappa=0.0,
    )
    e_ice = ElementConfig(
        name="ice_num", rho=jnp.full((_NBIN,), _RHO_ICE),
        igroup=_IGROUP_ICE,
        itype=2, icomposition=1, isolute=-1, kappa=0.0,
    )
    elements = (e_sulf, e_water, e_ice)

    # --- Gases ---
    gas_h2o = GasConfig(
        name="H2O", wtmol=_GWTMOL_H2O, ivaprtn=2, icomposition=1,
        dgc_threshold=0.1, ds_threshold=0.1,
    )
    gas_h2so4 = GasConfig(
        name="H2SO4", wtmol=_GWTMOL_H2SO4, ivaprtn=4, icomposition=2,
        dgc_threshold=0.1, ds_threshold=0.1,
    )
    gases = (gas_h2o, gas_h2so4)

    # --- Solute (sulfate only) ---
    solute = SoluteConfig(
        name="sulfate", ions=3, wtmol=_GWTMOL_H2SO4, rho=_RHO_SULFATE,
    )
    solutes = (solute,)

    # --- Coagulation tables: same-group only (no cross-group coag in cut 2) ---
    # icoag[ig, jg] gives target group for coagulation of (ig, jg). Same-group
    # collisions return to that group; cross-group are deferred.
    icoag = np.array([
        [_IGROUP_SULFATE,     -1,                  -1],
        [-1,                  _IGROUP_CLOUD_WATER, -1],
        [-1,                  -1,                  _IGROUP_ICE],
    ], dtype=np.int32)
    # icoagelem[ielem, jg] — target element when this element coagulates with
    # any particle from group jg. -1 means no contribution.
    icoagelem = np.array([
        [_IELEM_SULFATE,     -1,                  -1],
        [-1,                 _IELEM_CLOUD_WATER, -1],
        [-1,                 -1,                  _IELEM_ICE],
    ], dtype=np.int32)
    coag_cfg = setup_coag(
        nbin=_NBIN, ngroup=3, nelem=3,
        groups=groups, elements=elements,
        icoag_input=icoag, icoagelem_input=icoagelem,
    )

    cfg = CarmaConfig(
        nbin=_NBIN, nelem=3, ngroup=3, ngas=2, nsolute=1,
        elements=elements, groups=groups, gases=gases, solutes=solutes,
        coag=coag_cfg,
        do_coag=True, do_grow=True, do_vtran=False, do_vdiff=False,
        do_thermo=True, do_substep=True, do_explised=False,
        do_incloud=False, do_clearsky=True, do_detrain=False,
        do_pheat=False, do_pheatatm=False, do_cnst_rlh=False,
        itbnd_pc=1, ibbnd_pc=1,
        maxsubsteps=32, minsubsteps=1, maxretries=16, conmax=1e-4,
        cstick=1.0, gsticki=1.0, gstickl=1.0, tstick=1.0, dt_threshold=1.0,
        igash2o=_IGAS_H2O, igash2so4=_IGAS_H2SO4, igasso2=-1,
    )

    # Build dispatch arrays.
    is_ice = np.array([bool(g.is_ice) for g in groups], dtype=bool)
    is_cloud = np.array([bool(g.is_cloud) for g in groups], dtype=bool)
    is_sulfate = np.array([bool(g.is_sulfate) for g in groups], dtype=bool)
    ienconc = np.array([int(g.ienconc) for g in groups], dtype=np.int32)
    igelem = np.array([int(e.igroup) for e in elements], dtype=np.int32)

    # element -> growth-gas index. Sulfate grows from H2SO4; water+ice from H2O.
    def _growth_gas(elem):
        grp = groups[elem.igroup]
        if grp.is_sulfate:
            return _IGAS_H2SO4
        if grp.is_ice or grp.is_cloud:
            return _IGAS_H2O
        return -1
    igrowgas = np.array([_growth_gas(e) for e in elements], dtype=np.int32)

    # Stacked bin geometry: shape (ngroup, nbin)
    rmass_2d = jnp.stack(
        [jnp.asarray(g.rmass, dtype=DTYPE) for g in groups], axis=0,
    )
    rmassup_2d = jnp.stack(
        [jnp.asarray(g.rmassup, dtype=DTYPE) for g in groups], axis=0,
    )
    # True bin width: rmassup - rmasslow (rmasslow[0] = rmass[0]/sqrt(rmrat),
    # rmasslow[i] = rmassup[i-1]).
    dm_list = []
    for g in groups:
        rmass_np = np.asarray(g.rmass)
        rmassup_np = np.asarray(g.rmassup)
        rmasslow_np = np.concatenate(
            [[rmass_np[0] / (g.rmrat ** 0.5)], rmassup_np[:-1]]
        )
        dm_list.append(jnp.asarray(rmassup_np - rmasslow_np, dtype=DTYPE))
    dm_2d = jnp.stack(dm_list, axis=0)
    r_2d = jnp.stack(
        [jnp.asarray(g.r, dtype=DTYPE) for g in groups], axis=0,
    )

    return MultiSpeciesConfig(
        cfg=cfg,
        igrowgas=igrowgas,
        igelem=igelem,
        is_ice=is_ice,
        is_cloud=is_cloud,
        is_sulfate=is_sulfate,
        ienconc=ienconc,
        rmass_2d=rmass_2d,
        dm_2d=dm_2d,
        rmassup_2d=rmassup_2d,
        r_2d=r_2d,
    )
