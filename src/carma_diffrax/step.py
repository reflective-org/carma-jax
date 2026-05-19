"""step.py — one outer-step integration via diffrax.

Replaces the entire `make_substep_loop_jit` + adaptive-retry mechanism from
the faithful port with a single `diffrax.diffeqsolve(Kvaerno5, PIDController)`
call. The PID controller adapts the internal step size to whatever the
chemistry requires; the user only sees the outer-step result.
"""
import diffrax
import jax

from carma_diffrax.config import DiffraxConfig
from carma_diffrax.rhs import FrozenEnv, make_rhs
from carma_diffrax.state import StateShape, pack, unpack


_SOLVER_REGISTRY = {
    "Kvaerno3": diffrax.Kvaerno3,
    "Kvaerno4": diffrax.Kvaerno4,
    "Kvaerno5": diffrax.Kvaerno5,
    "KenCarp4": diffrax.KenCarp4,
}


def _resolve_solver(name: str):
    if name not in _SOLVER_REGISTRY:
        raise ValueError(
            f"Unknown solver {name!r}. "
            f"Available: {sorted(_SOLVER_REGISTRY)}."
        )
    return _SOLVER_REGISTRY[name]()


def diffrax_step(pc0, gc0, T0, dtime, env: FrozenEnv,
                 shape: StateShape, cfg: DiffraxConfig):
    """Integrate (pc, gc, T) over one outer step `dtime` with diffrax.

    Args:
        pc0: (nbin, nelem) particle conc at outer-step start, [#/cm^3]
        gc0: (ngas,) gas mass conc, [g/cm^3 × zmet]
        T0:  () scalar temperature [K]
        dtime: outer-step duration [s] (e.g. 60.0 or 1800.0)
        env:   FrozenEnv built once at outer-step boundary
        shape: StateShape(nbin, nelem, ngas)
        cfg:   DiffraxConfig — solver + tolerances

    Returns:
        (pc, gc, T, stats) where stats is a dict with
        ``num_accepted_steps``, ``num_rejected_steps``, ``num_steps``.
    """
    rhs = make_rhs(env, shape)
    term = diffrax.ODETerm(rhs)
    solver = _resolve_solver(cfg.solver_name)
    controller = diffrax.PIDController(rtol=cfg.rtol, atol=cfg.atol)

    y0 = pack(pc0, gc0, T0)
    sol = diffrax.diffeqsolve(
        term, solver,
        t0=0.0, t1=float(dtime), dt0=None, y0=y0,
        stepsize_controller=controller,
        max_steps=cfg.max_steps,
        saveat=diffrax.SaveAt(t1=True),
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
    )
    y_end = sol.ys[-1]
    pc, gc, T = unpack(y_end, shape)
    stats = {
        "num_accepted_steps": int(sol.stats["num_accepted_steps"]),
        "num_rejected_steps": int(sol.stats["num_rejected_steps"]),
        "num_steps": int(sol.stats["num_steps"]),
        "result": str(sol.result),
    }
    return pc, gc, T, stats
