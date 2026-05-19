"""CARMA-JAX public coupling API.

``carma.coupling`` is the stable SI-units interface intended for GCM
integration. It wraps the internal CGS machinery so callers never need
to know about units, grid conventions, or CGS idioms.

Quickstart::

    from carma.coupling import create_sulfate_config, carma_column_step_mks
    from carma.column_step import make_column_step_full

    config  = create_sulfate_config(nbin=38)       # once
    step_fn = make_column_step_full(config)         # once

    result  = carma_column_step_mks(               # per-timestep
        config, step_fn,
        T_k=T, p_pa=p, q_h2o=q_h2o, q_h2so4=q_h2so4,
        n_aerosol=n_aer, dtime_s=1800.0,
        z_m=zc, dz_m=dz,
    )
"""
from carma.coupling.driver import (
    carma_column_step_mks,
    compute_ppm_coefs,
    create_sulfate_config,
)
from carma.coupling.units import (
    air_density_cgs,
    cgs_to_pa,
    cm2_to_m2_flux,
    cms_to_ms,
    gc_to_mmr,
    m2_to_cm2_flux,
    m_to_cm,
    mmr_to_gc,
    ms_to_cms,
    nmr_to_pc,
    pa_to_cgs,
    pc_to_nmr,
)

__all__ = [
    "carma_column_step_mks",
    "compute_ppm_coefs",
    "create_sulfate_config",
    "air_density_cgs",
    "cgs_to_pa",
    "cm2_to_m2_flux",
    "cms_to_ms",
    "gc_to_mmr",
    "m2_to_cm2_flux",
    "m_to_cm",
    "mmr_to_gc",
    "ms_to_cms",
    "nmr_to_pc",
    "pa_to_cgs",
    "pc_to_nmr",
]
