"""diffeqsolve wrapper for the multispecies RHS."""
import diffrax
import jax

from carma_diffrax.config import DiffraxConfig
from carma_diffrax.multispecies.rhs import FrozenEnvMS, make_rhs_ms
from carma_diffrax.multispecies.config import MultiSpeciesConfig
from carma_diffrax.multispecies.state import MultiSpeciesShape, pack, unpack


_SOLVER_REGISTRY = {
    "Kvaerno3": diffrax.Kvaerno3,
    "Kvaerno4": diffrax.Kvaerno4,
    "Kvaerno5": diffrax.Kvaerno5,
    "KenCarp4": diffrax.KenCarp4,
}


def _resolve_solver(name: str):
    if name not in _SOLVER_REGISTRY:
        raise ValueError(
            f"Unknown solver {name!r}. Available: {sorted(_SOLVER_REGISTRY)}."
        )
    return _SOLVER_REGISTRY[name]()


def diffrax_step_ms(pc0, gc0, T0, dtime, env: FrozenEnvMS,
                     ms: MultiSpeciesConfig,
                     shape: MultiSpeciesShape,
                     cfg_d: DiffraxConfig,
                     **rhs_toggles):
    """One outer-step multispecies diffrax integration.

    Args mirror the sulfate `diffrax_step` plus the `ms` config bundle.
    `rhs_toggles` are forwarded to `make_rhs_ms` (do_homogeneous_nuc,
    do_ccn_activation, etc.).
    """
    rhs = make_rhs_ms(env, ms, shape, **rhs_toggles)
    term = diffrax.ODETerm(rhs)
    solver = _resolve_solver(cfg_d.solver_name)
    controller = diffrax.PIDController(rtol=cfg_d.rtol, atol=cfg_d.atol)

    y0 = pack(pc0, gc0, T0)
    sol = diffrax.diffeqsolve(
        term, solver,
        t0=0.0, t1=float(dtime), dt0=None, y0=y0,
        stepsize_controller=controller,
        max_steps=cfg_d.max_steps,
        saveat=diffrax.SaveAt(t1=True),
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
    )
    y_end = sol.ys[-1]
    pc, gc, T = unpack(y_end, shape)
    stats = {
        "num_accepted_steps": int(sol.stats["num_accepted_steps"]),
        "num_rejected_steps": int(sol.stats["num_rejected_steps"]),
        "num_steps": int(sol.stats["num_steps"]),
        "result": sol.result,
        "successful": bool(sol.result == diffrax.RESULTS.successful),
    }
    return pc, gc, T, stats
