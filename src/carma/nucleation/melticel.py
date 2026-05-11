"""Ice crystal → droplet melting (placeholder kludge).

Ported from `melticel.F90`. Mirror of `freezdropl`; the F90 source
also self-identifies as "temporary simple kludge":

    rnuclg[ibin] = 100.0  [s⁻¹]   if T > T0  AND  pconmax > FEW_PC
                 = 0                otherwise

T0 = 273.16 K, so the melting gate trips above the triple point.
Note the gate uses `pconmax` (the per-group max concentration) here,
not the per-bin `pc` — F90:59. We mirror that.
"""
import jax.numpy as jnp

from carma.constants import T0, FEW_PC
from carma.precision import DTYPE


_RNUCLG_CONST = DTYPE(100.0)   # F90:63 — constant rate when gate fires


def melticel(t_val, pconmax_val, nbin):
    """Compute per-bin ice-melting nucleation rate.

    Args:
        t_val:        Temperature [K], scalar.
        pconmax_val:  Max particle concentration for this group, scalar.
        nbin:         Number of bins.

    Returns:
        rnuclg: per-bin loss rate [s⁻¹], shape ``(NBIN,)``.
    """
    warm_enough = t_val > T0
    has_particles = pconmax_val > FEW_PC
    gate = warm_enough & has_particles
    return jnp.where(gate, jnp.full((nbin,), _RNUCLG_CONST), jnp.zeros((nbin,), dtype=DTYPE))
