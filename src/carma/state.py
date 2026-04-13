"""Per-column atmospheric state for CARMA-JAX.

CarmaState is a NamedTuple with all JAX arrays, fully traced by JIT.
Ported from: carma_types_mod.F90 (carmastate_type).
"""

from typing import NamedTuple

import jax.numpy as jnp


class CarmaState(NamedTuple):
    """Per-column atmospheric state. All fields are JAX arrays.

    This is the mutable state that evolves during a timestep.
    Functional updates via state._replace(field=new_value).

    Array dimensions:
        NZ: number of vertical levels
        NBIN: number of size bins
        NELEM: number of particle elements
        NGROUP: number of particle groups
        NGAS: number of gas species
    """
    # --- Grid ---
    zc: jnp.ndarray  # (NZ,) altitude at layer centers [cm]
    zl: jnp.ndarray  # (NZ+1,) altitude at layer edges [cm]
    dz: jnp.ndarray  # (NZ,) layer thickness [cm]
    zmet: jnp.ndarray  # (NZ,) vertical metric at centers
    zmetl: jnp.ndarray  # (NZ+1,) vertical metric at edges

    # --- Atmospheric state ---
    t: jnp.ndarray  # (NZ,) temperature [K]
    p: jnp.ndarray  # (NZ,) pressure [dyne/cm^2]
    pl: jnp.ndarray  # (NZ+1,) pressure at edges [dyne/cm^2]
    rhoa: jnp.ndarray  # (NZ,) air density * zmet [g/cm^2/z]
    rhoa_wet: jnp.ndarray  # (NZ,) wet air density [g/cm^3]
    rmu: jnp.ndarray  # (NZ,) dynamic viscosity [g/cm/s]
    thcond: jnp.ndarray  # (NZ,) thermal conductivity [erg/cm/s/K]

    # --- Primary state ---
    pc: jnp.ndarray  # (NZ, NBIN, NELEM) particle concentrations [#/cm^3/z]
    gc: jnp.ndarray  # (NZ, NGAS) gas concentrations [g/cm^3/z]

    # --- Saved state for substepping ---
    pcl: jnp.ndarray  # (NZ, NBIN, NELEM) pc at start of step
    gcl: jnp.ndarray  # (NZ, NGAS) gc at start of step
    told: jnp.ndarray  # (NZ,) temperature at start of step
    d_gc: jnp.ndarray  # (NZ, NGAS) gas increment from transport
    d_t: jnp.ndarray  # (NZ,) temperature increment from transport

    # --- Cloud state ---
    cldfrc: jnp.ndarray  # (NZ,) cloud fraction [0-1]

    # --- Particle properties (recomputed per step) ---
    rhop: jnp.ndarray  # (NZ, NBIN, NGROUP) particle density [g/cm^3]
    r_wet: jnp.ndarray  # (NZ, NBIN, NGROUP) wet radius [cm]
    rhop_wet: jnp.ndarray  # (NZ, NBIN, NGROUP) wet density [g/cm^3]
    vf: jnp.ndarray  # (NZ+1, NBIN, NGROUP) fall velocity [cm/s]
    re: jnp.ndarray  # (NZ, NBIN, NGROUP) Reynolds number
    bpm: jnp.ndarray  # (NZ, NBIN, NGROUP) Cunningham slip correction
    dkz: jnp.ndarray  # (NZ+1, NBIN, NGROUP) diffusion coefficient
    vd: jnp.ndarray  # (NBIN, NGROUP) dry deposition velocity [cm/s]

    # --- Growth properties ---
    diffus: jnp.ndarray  # (NZ, NGAS) gas diffusivity [cm^2/s]
    rlhe: jnp.ndarray  # (NZ, NGAS) latent heat of evaporation [cm^2/s^2]
    rlhm: jnp.ndarray  # (NZ, NGAS) latent heat of melting [cm^2/s^2]
    pvapl: jnp.ndarray  # (NZ, NGAS) sat vapor pressure liquid [dyne/cm^2]
    pvapi: jnp.ndarray  # (NZ, NGAS) sat vapor pressure ice [dyne/cm^2]
    supsatl: jnp.ndarray  # (NZ, NGAS) supersaturation over liquid
    supsati: jnp.ndarray  # (NZ, NGAS) supersaturation over ice
    akelvin: jnp.ndarray  # (NZ, NGAS) Kelvin curvature factor [cm]
    gro: jnp.ndarray  # (NZ, NBIN, NGROUP) growth kernel [g cm^3/erg/s]
    gro1: jnp.ndarray  # (NZ, NBIN, NGROUP) growth conduction [s/g]
    gro2: jnp.ndarray  # (NZ, NGROUP) growth radiation [g/erg]
    thcondnc: jnp.ndarray  # (NZ, NBIN, NGROUP) corrected thermal conductivity
    scrit: jnp.ndarray  # (NZ, NBIN, NGROUP) critical supersaturation
    ft: jnp.ndarray  # (NZ, NBIN, NGROUP) thermal ventilation factor

    # --- Coagulation kernel ---
    ckernel: jnp.ndarray  # (NZ, NBIN, NBIN, NGROUP, NGROUP) coag kernel

    # --- Diagnostics ---
    rlheat: jnp.ndarray  # (NZ,) latent heating rate [K/s]
    partheat: jnp.ndarray  # (NZ,) particle heating rate [K/s]
    pc_nucl: jnp.ndarray  # (NZ, NBIN, NELEM) nucleation production
    sedimentationflux: jnp.ndarray  # (NBIN, NELEM) sedimentation flux
    pconmax: jnp.ndarray  # (NZ, NGROUP) max concentration per group
    zsubsteps: jnp.ndarray  # (NZ,) substeps used per level

    # --- Boundary conditions ---
    ftoppart: jnp.ndarray  # (NBIN, NELEM) top flux
    fbotpart: jnp.ndarray  # (NBIN, NELEM) bottom flux
    pc_topbnd: jnp.ndarray  # (NBIN, NELEM) top boundary concentration
    pc_botbnd: jnp.ndarray  # (NBIN, NELEM) bottom boundary concentration

    # --- Timestep ---
    dtime: float  # Current substep timestep [s]
    dtime_orig: float  # Original timestep [s]
    time: float  # Simulation time [s]
