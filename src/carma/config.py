"""Configuration data structures for CARMA-JAX.

All configuration types are NamedTuples, static to JIT.
Ported from: carma_types_mod.F90 (carma_type and sub-types).
"""

from typing import NamedTuple

import jax.numpy as jnp


class ElementConfig(NamedTuple):
    """Per-element configuration. Maps to carmaelement_type."""
    name: str
    rho: jnp.ndarray  # (NBIN,) mass density [g/cm^3]
    igroup: int  # Group index this element belongs to
    itype: int  # ElementType enum value
    icomposition: int  # Compound specification
    isolute: int  # Solute index (-1 = none)
    kappa: float  # Hygroscopicity parameter


class GroupConfig(NamedTuple):
    """Per-group configuration. Maps to carmagroup_type."""
    name: str
    ishape: int  # Shape enum value
    ienconc: int  # Element index for number concentration
    is_ice: bool
    is_cloud: bool
    is_sulfate: bool
    do_vtran: bool
    do_drydep: bool
    ifallrtn: int  # FallRoutine enum value
    irhswell: int  # SwellMethod enum value
    rmrat: float  # Mass ratio between bins
    eshape: float  # Length/diameter ratio (aspect ratio)
    rmin: float  # Radius of first bin [cm]
    # Precomputed bin arrays (NBIN,):
    r: jnp.ndarray  # Bin center radii [cm]
    rmass: jnp.ndarray  # Bin center masses [g]
    vol: jnp.ndarray  # Bin volumes [cm^3]
    dr: jnp.ndarray  # Bin widths in radius [cm]
    dm: jnp.ndarray  # Bin widths in mass [g]
    rmassup: jnp.ndarray  # Upper bin boundary mass [g]
    rup: jnp.ndarray  # Upper bin boundary radius [cm]
    rlow: jnp.ndarray  # Lower bin boundary radius [cm]
    # Shape correction factors (NBIN,):
    rrat: jnp.ndarray  # Radius ratio for shape
    rprat: jnp.ndarray  # Drag ratio for shape
    arat: jnp.ndarray  # Area ratio for shape


class GasConfig(NamedTuple):
    """Per-gas configuration. Maps to carmagas_type."""
    name: str
    wtmol: float  # Molecular weight [g/mol]
    ivaprtn: int  # VaporPressureRoutine enum value
    icomposition: int  # GasComposition enum value
    dgc_threshold: float  # Convergence threshold for gas concentration
    ds_threshold: float  # Convergence threshold for supersaturation


class SoluteConfig(NamedTuple):
    """Per-solute configuration. Maps to carmasolute_type."""
    name: str
    ions: int  # Number of ions in solution
    wtmol: float  # Molecular weight [g/mol]
    rho: float  # Density [g/cm^3]


class CoagConfig(NamedTuple):
    """Precomputed coagulation mapping tables. From setupcoag."""
    icoag: jnp.ndarray  # (NGROUP, NGROUP) target group index
    icoagelem: jnp.ndarray  # (NELEM, NGROUP) target element index
    icoagop: jnp.ndarray  # (NGROUP, NGROUP) coagulation operation type
    volx: jnp.ndarray  # Volume fraction array
    # Bin pair arrays for production (padded to MAX_PAIRS):
    npairl: jnp.ndarray  # (NGROUP, NBIN) number of loss pairs
    npairu: jnp.ndarray  # (NGROUP, NBIN) number of production pairs
    ilow: jnp.ndarray  # (NGROUP, NBIN, MAX_PAIRS) source bin i (loss)
    jlow: jnp.ndarray  # (NGROUP, NBIN, MAX_PAIRS) source bin j (loss)
    iglow: jnp.ndarray  # (NGROUP, NBIN, MAX_PAIRS) source group i (loss)
    jglow: jnp.ndarray  # (NGROUP, NBIN, MAX_PAIRS) source group j (loss)
    iup: jnp.ndarray  # (NGROUP, NBIN, MAX_PAIRS) source bin i (production)
    jup: jnp.ndarray  # (NGROUP, NBIN, MAX_PAIRS) source bin j (production)
    igup: jnp.ndarray  # (NGROUP, NBIN, MAX_PAIRS) source group i (production)
    jgup: jnp.ndarray  # (NGROUP, NBIN, MAX_PAIRS) source group j (production)


class CarmaConfig(NamedTuple):
    """Complete static model configuration. Maps to carma_type.

    Passed to JIT-compiled functions as a static argument (via
    functools.partial or static_argnums).
    """
    # Dimensions
    nbin: int
    nelem: int
    ngroup: int
    ngas: int
    nsolute: int

    # Sub-configurations
    elements: tuple  # tuple[ElementConfig, ...]
    groups: tuple  # tuple[GroupConfig, ...]
    gases: tuple  # tuple[GasConfig, ...]
    solutes: tuple  # tuple[SoluteConfig, ...]

    # Process tables
    coag: CoagConfig

    # Process flags
    do_coag: bool
    do_grow: bool
    do_vtran: bool
    do_vdiff: bool
    do_thermo: bool
    do_substep: bool
    do_explised: bool
    do_incloud: bool
    do_clearsky: bool
    do_detrain: bool
    do_pheat: bool
    do_pheatatm: bool
    do_cnst_rlh: bool

    # Boundary conditions
    itbnd_pc: int  # Top boundary type (BoundaryCondition)
    ibbnd_pc: int  # Bottom boundary type (BoundaryCondition)

    # Substepping parameters
    maxsubsteps: int
    minsubsteps: int
    maxretries: int
    conmax: float

    # Accommodation coefficients
    cstick: float  # Coagulation sticking probability
    gsticki: float  # Growth sticking probability (ice)
    gstickl: float  # Growth sticking probability (liquid)
    tstick: float  # Thermal accommodation coefficient
    dt_threshold: float  # Temperature convergence threshold

    # Gas indices (-1 = not present)
    igash2o: int
    igash2so4: int
    igasso2: int
