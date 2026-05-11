"""Droplet → ice freezing (placeholder kludge).

Ported from `freezdropl.F90`. CARMA's current droplet-freezing
parameterisation is a placeholder constant-rate kludge (the F90 source
self-identifies as "temporary simple kludge"):

    rnuclg[ibin] = 100.0  [s⁻¹]   if T < T0 - 40 K  AND  pc[ibin] > FEW_PC
                 = 0                otherwise

T0 = 273.16 K, so the freezing gate trips at T < 233.16 K. Trivial
arithmetic; we still wire it up cleanly so it composes into the
broader Phase 11 microphysics pipeline.
"""
import jax.numpy as jnp

from carma.constants import T0, FEW_PC
from carma.precision import DTYPE


_RNUCLG_CONST = DTYPE(100.0)   # F90:62 — constant rate when gate fires
_T_FREEZE = T0 - DTYPE(40.0)   # F90:61 — freezing onset


def freezdropl(t_val, pc_bins):
    """Compute per-bin droplet-freezing nucleation rate.

    Args:
        t_val:   Temperature [K], scalar.
        pc_bins: Particle concentration [#/cm³/z] per bin, shape
                 ``(NBIN,)`` — gates the kernel on each bin individually
                 (F90 uses ``pc(iz, ibin, iepart) > FEW_PC`` per-bin,
                 not the per-group ``pconmax`` like Möhler).

    Returns:
        rnuclg: per-bin loss rate [s⁻¹], shape ``(NBIN,)``.
    """
    cold_enough = t_val < _T_FREEZE
    has_particles = pc_bins > FEW_PC
    gate = cold_enough & has_particles
    return jnp.where(gate, _RNUCLG_CONST, DTYPE(0.0))
