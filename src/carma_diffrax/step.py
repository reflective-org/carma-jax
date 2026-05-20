"""step.py — one outer-step integration via diffrax.

Replaces the entire `make_substep_loop_jit` + adaptive-retry mechanism from
the faithful port with a single `diffrax.diffeqsolve(Kvaerno5, PIDController)`
call. The PID controller adapts the internal step size to whatever the
chemistry requires; the user only sees the outer-step result.

JIT-cache amortization (Phase 2D-pre): env is threaded through as a
diffeqsolve ``args`` parameter instead of being captured in the RHS
closure. That way the same compiled function services every outer step
of every scenario, regardless of which env is passed in.
"""
from functools import lru_cache

import diffrax
import jax
import jax.numpy as jnp

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
            f"Unknown solver {name!r}. Available: {sorted(_SOLVER_REGISTRY)}."
        )
    return _SOLVER_REGISTRY[name]()


@lru_cache(maxsize=64)
def _build_jit_step(shape: StateShape, solver_name: str,
                     rtol: float, atol: float, max_steps: int):
    """Build a JIT-compiled outer-step function bound to (shape, solver, tol).

    The returned `step(pc0, gc0, T0, dtime, env)` is `@jax.jit`-wrapped and
    can be called repeatedly with different (pc0, gc0, T0, dtime, env)
    without retracing — the env pytree shape is the only thing that matters
    for cache hits.
    """
    rhs = make_rhs(shape)
    term = diffrax.ODETerm(rhs)
    solver = _resolve_solver(solver_name)
    controller = diffrax.PIDController(rtol=rtol, atol=atol)

    @jax.jit
    def step(pc0, gc0, T0, dtime, env):
        y0 = pack(pc0, gc0, T0)
        sol = diffrax.diffeqsolve(
            term, solver,
            t0=0.0, t1=dtime, dt0=None, y0=y0,
            args=env,
            stepsize_controller=controller,
            max_steps=max_steps,
            saveat=diffrax.SaveAt(t1=True),
            adjoint=diffrax.RecursiveCheckpointAdjoint(),
        )
        y_end = sol.ys[-1]
        pc, gc, T = unpack(y_end, shape)
        return pc, gc, T, sol.stats, sol.result

    return step


def diffrax_step(pc0, gc0, T0, dtime, env: FrozenEnv,
                 shape: StateShape, cfg: DiffraxConfig):
    """Integrate (pc, gc, T) over one outer step `dtime` with diffrax.

    Args:
        pc0: (nbin, nelem) — particle conc at outer-step start [#/cm^3]
        gc0: (ngas,) — gas mass conc [g/cm^3 × zmet]
        T0:  () — scalar temperature [K]
        dtime: outer-step duration [s]
        env: FrozenEnv built at outer-step boundary
        shape: StateShape — static; resolved at JIT trace time
        cfg: DiffraxConfig — solver name + tolerances + max_steps

    Returns:
        (pc, gc, T, stats) where stats dict has num_accepted_steps,
        num_rejected_steps, num_steps, result, successful.
    """
    step_jit = _build_jit_step(
        shape, cfg.solver_name, cfg.rtol, cfg.atol, cfg.max_steps,
    )
    pc, gc, T, stats_raw, result = step_jit(pc0, gc0, T0, dtime, env)
    stats = {
        "num_accepted_steps": int(stats_raw["num_accepted_steps"]),
        "num_rejected_steps": int(stats_raw["num_rejected_steps"]),
        "num_steps": int(stats_raw["num_steps"]),
        "result": result,
        "successful": bool(result == diffrax.RESULTS.successful),
    }
    return pc, gc, T, stats
