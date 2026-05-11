"""Cloud droplet activation (placeholder kludge).

Ported from `actdropl.F90`. CARMA's current in-source droplet
activation parameterisation is a constant-rate kludge — when the
gates fire, the rate is exactly 1000 s⁻¹:

    rnuclg[ibin] = 1000.0  [s⁻¹]   if  T ≥ T₀ − 40 K
                                    AND pconmax > FEW_PC
                                    AND supsatl > scrit[ibin]
                                    AND pc[ibin] > SMALL_PC
                                    AND target droplet bin not evaporating
                 = 0                otherwise

The full Köhler activation integral (Twomey / Abdul-Razzak / etc.)
lives in `adgaquad_mod` and is used by the CAM coupling layer rather
than by this in-source subroutine. When CARMA is replaced upstream
with a proper parameterisation, transcribe the new formula here.

`scrit` is the bin-specific critical Köhler supersaturation, computed
during CARMA setup from the Kelvin term + solute term for each bin.
We accept it as an input rather than re-deriving it — it's part of
the kernel's external dependency surface.
"""
import jax.numpy as jnp

from carma.constants import T0, FEW_PC, SMALL_PC
from carma.precision import DTYPE


_RNUCLG_CONST = DTYPE(1000.0)         # actdropl.F90:94
_T_ACTIVATE = T0 - DTYPE(40.0)         # actdropl.F90:46 — drops can activate above T0-40


def actdropl(t_val, supsatl_val, pconmax_val,
             scrit_bins, pc_bins, evappe_target_bins):
    """Compute per-bin cloud-droplet activation rate.

    Args:
        t_val:              Temperature [K], scalar.
        supsatl_val:        Liquid supersaturation, scalar.
        pconmax_val:        Max particle concentration for this group, scalar.
        scrit_bins:         Per-bin critical Köhler supersaturation,
                            shape ``(NBIN,)``. Activation requires
                            ``supsatl > scrit[bin]``.
        pc_bins:            Per-bin source-element concentration
                            [#/cm³/z], shape ``(NBIN,)`` — gate on
                            ``pc[bin] > SMALL_PC``.
        evappe_target_bins: Per-bin evaporation flag for the target bin
                            (1.0 = evaporating, 0.0 = not), shape
                            ``(NBIN,)`` — gate off where target droplet
                            is evaporating. Pass zeros if not modelling
                            target-bin evaporation here.

    Returns:
        rnuclg: per-bin loss rate [s⁻¹], shape ``(NBIN,)``.
    """
    warm_enough = t_val >= _T_ACTIVATE
    has_particles = pconmax_val > FEW_PC
    super_enough = supsatl_val > scrit_bins
    pc_above_floor = pc_bins > SMALL_PC
    not_evap = evappe_target_bins <= DTYPE(0.0)

    gate = warm_enough & has_particles & super_enough & pc_above_floor & not_evap
    return jnp.where(gate, _RNUCLG_CONST, DTYPE(0.0))
