"""carma_diffrax.multispecies — multi-group diffrax solver.

Extends the sulfate-only `carma_diffrax` to handle three particle groups
(sulfate, cloud_water, ice_crystal) with phase transitions between them
(CCN activation, droplet freezing, ice melting).

Reuses everything possible from the sulfate cut: pheat for per-bin growth
rates, sulfnuc for H2SO4 homogeneous nucleation, vapor pressure modules,
the upwind number-density flux scheme, and the bin-mass-balance gas update.
"""

from carma_diffrax.multispecies.config import (
    make_multispecies_config,
    MultiSpeciesConfig,
)
from carma_diffrax.multispecies.state import (
    MultiSpeciesShape, pack as pack_ms, unpack as unpack_ms,
)
from carma_diffrax.multispecies.rhs import make_rhs_ms, FrozenEnvMS
from carma_diffrax.multispecies.step import diffrax_step_ms
from carma_diffrax.multispecies.env_builder import refresh_env_ms

__all__ = [
    "make_multispecies_config", "MultiSpeciesConfig",
    "MultiSpeciesShape", "pack_ms", "unpack_ms",
    "make_rhs_ms", "FrozenEnvMS", "diffrax_step_ms", "refresh_env_ms",
]
