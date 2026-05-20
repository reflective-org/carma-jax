"""diffeqsolve wrapper for the multispecies RHS."""
from functools import lru_cache

import diffrax
import jax

from carma_diffrax.config import DiffraxConfig
from carma_diffrax.multispecies.rhs import FrozenEnvMS, make_rhs_ms
from carma_diffrax.multispecies.config import MultiSpeciesConfig
from carma_diffrax.multispecies.state import MultiSpeciesShape, pack, unpack
from carma_diffrax.step import _build_root_finder


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


def _build_jit_step_ms(ms: MultiSpeciesConfig, shape: MultiSpeciesShape,
                        solver_name: str, rtol: float, atol: float,
                        max_steps: int,
                        pcoeff: float, icoeff: float, dcoeff: float,
                        factormin: float, factormax: float, safety: float,
                        root_finder_name: str, linear_solver_name: str,
                        do_homogeneous_nuc: bool,
                        do_ccn_activation: bool,
                        do_droplet_freezing: bool,
                        do_ice_melting: bool):
    """Build the JIT-compiled outer-step function for the multispecies path."""
    rhs = make_rhs_ms(
        ms, shape,
        do_homogeneous_nuc=do_homogeneous_nuc,
        do_ccn_activation=do_ccn_activation,
        do_droplet_freezing=do_droplet_freezing,
        do_ice_melting=do_ice_melting,
    )
    term = diffrax.ODETerm(rhs)
    root_finder = _build_root_finder(
        root_finder_name, rtol, atol, linear_solver_name,
    )
    solver_cls = _SOLVER_REGISTRY[solver_name]
    solver = solver_cls(root_finder=root_finder)
    controller = diffrax.PIDController(
        rtol=rtol, atol=atol,
        pcoeff=pcoeff, icoeff=icoeff, dcoeff=dcoeff,
        factormin=factormin, factormax=factormax, safety=safety,
    )

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


# Module-level LRU cache to amortize the closure-build cost. Keys are
# hashable, so we use (id(ms), shape, solver_name, rtol, atol, ...).
@lru_cache(maxsize=64)
def _cached_jit_step(ms_id, shape, solver_name, rtol, atol, max_steps,
                      pcoeff, icoeff, dcoeff, factormin, factormax, safety,
                      root_finder_name, linear_solver_name,
                      do_hom_nuc, do_ccn, do_freeze, do_melt):
    ms = _MS_REGISTRY[ms_id]
    return _build_jit_step_ms(
        ms, shape, solver_name, rtol, atol, max_steps,
        pcoeff, icoeff, dcoeff, factormin, factormax, safety,
        root_finder_name, linear_solver_name,
        do_hom_nuc, do_ccn, do_freeze, do_melt,
    )


# Maps id(ms) → ms instance, so cache keys stay hashable.
_MS_REGISTRY: dict = {}


def diffrax_step_ms(pc0, gc0, T0, dtime, env: FrozenEnvMS,
                     ms: MultiSpeciesConfig,
                     shape: MultiSpeciesShape,
                     cfg_d: DiffraxConfig,
                     **rhs_toggles):
    """One outer-step multispecies diffrax integration."""
    do_hom = rhs_toggles.get("do_homogeneous_nuc", True)
    do_ccn = rhs_toggles.get("do_ccn_activation", True)
    do_frz = rhs_toggles.get("do_droplet_freezing", True)
    do_mlt = rhs_toggles.get("do_ice_melting", True)
    ms_id = id(ms)
    _MS_REGISTRY[ms_id] = ms
    step_jit = _cached_jit_step(
        ms_id, shape, cfg_d.solver_name, cfg_d.rtol, cfg_d.atol,
        cfg_d.max_steps,
        cfg_d.pcoeff, cfg_d.icoeff, cfg_d.dcoeff,
        cfg_d.factormin, cfg_d.factormax, cfg_d.safety,
        cfg_d.root_finder_name, cfg_d.linear_solver_name,
        do_hom, do_ccn, do_frz, do_mlt,
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
