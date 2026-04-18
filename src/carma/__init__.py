"""CARMA-JAX — JAX port of the CARMA aerosol microphysics model.

Public API surface for building and running single-column or batched
simulations. For full config/state NamedTuple definitions see
``carma.config`` and ``carma.state``.

The ``make_step_*`` factories pre-bind a ``CarmaConfig`` into a
JIT-compiled closure that takes only the per-step state arrays. For
simulations that don't need to rebuild the kernel between steps this
is the cheapest path.

Currently implemented drivers:
    make_step_transport(config) — prestep + vertical (sedimentation,
        Brownian diffusion, dry deposition).
    make_step_coag(config) — prestep + microslow (coagulation).

The microfast (growth + nucleation) driver is still called directly
via the Phase 3 scripts; integration into the unified step() awaits
a JIT refactor of microfast_growth.
"""

from carma.config import (
    CarmaConfig,
    CoagConfig,
    ElementConfig,
    GasConfig,
    GroupConfig,
    SoluteConfig,
)
from carma.state import CarmaState
from carma.prestep import prestep
from carma.step import make_step_coag, make_step_transport, step_transport
from carma.utils.smallconc import maxconc, smallconc

__all__ = [
    "CarmaConfig",
    "CarmaState",
    "CoagConfig",
    "ElementConfig",
    "GasConfig",
    "GroupConfig",
    "SoluteConfig",
    "make_step_coag",
    "make_step_transport",
    "maxconc",
    "prestep",
    "smallconc",
    "step_transport",
]
