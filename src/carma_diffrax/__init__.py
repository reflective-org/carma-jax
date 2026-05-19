"""carma_diffrax — diffrax-based parallel implementation of CARMA chemistry solvers.

This package provides an alternative to the hand-rolled semi-implicit Euler
chemistry solvers in `carma.solvers` (psolve, gsolve, tsolve). It treats the
coupled particle / gas / temperature state as a single stiff ODE and integrates
it with `diffrax.Kvaerno5` and `diffrax.PIDController`.

Scope (cut 1): sulfate only (H2SO4 + H2O), single column. Coagulation and
vertical transport are not handled here — use the faithful `carma` package.

The faithful Fortran port at `carma` is the bit-exact reference and is left
untouched. This package is a parallel implementation aimed at pushing past
the explicit-Euler stability ceiling that limits 1800 s outer timesteps.
"""

from carma_diffrax.config import DiffraxConfig
from carma_diffrax.state import pack, unpack
from carma_diffrax.step import diffrax_step

__all__ = ["DiffraxConfig", "diffrax_step", "pack", "unpack"]
