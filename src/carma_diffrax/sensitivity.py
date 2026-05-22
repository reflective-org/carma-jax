"""Convenience wrappers for end-to-end ``jax.grad`` over carma_diffrax.

Phase 3.1 (single-scenario, single-input gradient PoC).

The forward pipeline `scenario → final aerosol mass` is naturally a
pure JAX function once we route a single scalar input (here:
``h2so4_prod_rate``) through the same machinery the ensemble runner
uses. This module wraps that pipeline so callers can do::

    f = make_final_mass_wrt_prod_rate(scenario, nstep=...)
    grad = jax.grad(f)(prod_rate)

without worrying about env construction, packing, etc.

Limits of this first PoC
------------------------
- Single scenario, single scalar input (prod_rate).
- Differentiable mass output (``Σ pc · rmass`` at end of run).
- Diffrax's ``RecursiveCheckpointAdjoint`` (the default) handles the
  per-outer-step adjoint, so memory usage is O(log N) in the number
  of inner solver steps.

Later phases:
  - 3.2: multi-input ``(prod_rate, T0, mu_nm, sigma_g, M_ug_m3)``.
  - 3.3: vmap over the 100-scenario ensemble (needs Phase 2D-true).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx

from carma.constants import AVG, R_AIR, RPA2CGS
from carma.precision import DTYPE
from carma_diffrax.config import DiffraxConfig
from carma_diffrax.rhs import FrozenEnv, make_rhs
from carma_diffrax.state import StateShape, pack, unpack

# scripts/ is on sys.path in test/benchmark callers; ensure it's
# importable here too so this module is self-contained.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))


def _build_initial_pc(grp, mu_nm, sigma_g, M_ug_m3, rhoa_g_cm3):
    """Lognormal-mass seed normalised to ``M_ug_m3``.

    Mirrors ``benchmark_final/scripts/15_run_diffrax_1800s.py:74-87``
    so the gradient PoC starts from the same initial state the
    production runner uses.
    """
    r = np.asarray(grp.r)
    rmass_np = np.asarray(grp.rmass)
    log_mu_cm = math.log(mu_nm * 1e-7)
    log_sigma = math.log(sigma_g)
    shape_pdf = (
        np.exp(-0.5 * ((np.log(r) - log_mu_cm) / log_sigma) ** 2)
        / (r * log_sigma * math.sqrt(2.0 * math.pi))
        * rmass_np
    )
    norm = shape_pdf.sum()
    if norm > 0:
        M_target_mmr = (M_ug_m3 * 1e-12) / rhoa_g_cm3
        mmr_per_bin = shape_pdf * (M_target_mmr / norm)
        pc_per_bin = mmr_per_bin * rhoa_g_cm3 / rmass_np
    else:
        pc_per_bin = np.zeros_like(r)
    return jnp.asarray(pc_per_bin, dtype=DTYPE)[:, None]


def _build_env(T, p_cgs, gc, cfg, ppm, grp, dm):
    """Construct a ``FrozenEnv`` for the current ``(T, p_cgs, gc)``.

    Wraps ``scripts/jax_ensemble.py:_refresh_env`` and slices the
    PPM coefs to the single sulfate group. The output FrozenEnv is
    consumed by ``diffrax_step``.
    """
    from jax_ensemble import _refresh_env
    env_d = _refresh_env(T, p_cgs, gc, cfg, ppm)
    r_wet = (env_d["rup_wet"][0, :, 0] + env_d["rlow_wet"][0, :, 0]) / 2.0
    return FrozenEnv(
        akelvin=env_d["akelvin"], akelvini=env_d["akelvini"],
        gro=env_d["gro"], gro1=env_d["gro1"],
        rup_wet=env_d["rup_wet"], r_wet=r_wet,
        rmass=jnp.asarray(grp.rmass, dtype=DTYPE)[:, None],
        dm=dm[:, None],
        rmassup=jnp.asarray(grp.rmassup, dtype=DTYPE),
        rmrat_val=float(grp.rmrat),
        rhoa=env_d["rhoa"], zmet=env_d["zmet"],
        rlhe=env_d["rlhe"], rlhm=env_d["rlhm"],
        pratt=env_d["pratt"][..., 0], prat=env_d["prat"][..., 0],
        pden1=env_d["pden1"][..., 0], palr=env_d["palr"][..., 0],
    )


def _build_diff_step(shape: StateShape, rtol: float, atol: float,
                       max_steps: int = 20_000):
    """Build a diffrax step that supports both forward and reverse autodiff.

    Diverges from production ``diffrax_step`` in one way only:
    ``adjoint=DirectAdjoint`` instead of ``RecursiveCheckpointAdjoint``.
    DirectAdjoint pushes both forward and reverse-mode derivatives
    through the solver via plain autodiff (no ``custom_vjp``), which
    is what we need for ``jax.jvp``. Memory is O(N) in solver steps
    rather than O(log N), so this isn't suitable for production but
    fine for a single-outer-step PoC.
    """
    # Nucleation off for the PoC — sulfnuc has double-where guards
    # whose cotangents are NaN. Issue tracked for future work.
    term = diffrax.ODETerm(make_rhs(shape, do_homogeneous=False))
    solver = diffrax.Kvaerno5(
        root_finder=optx.Chord(rtol=rtol, atol=atol),
    )
    controller = diffrax.PIDController(
        rtol=rtol, atol=atol,
        pcoeff=0.3, icoeff=0.3, dcoeff=0.0,
        factormin=0.5, factormax=5.0, safety=0.8,
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
            adjoint=diffrax.DirectAdjoint(),
        )
        y_end = sol.ys[-1]
        pc, gc, T = unpack(y_end, shape)
        return pc, gc, T

    return step


def make_final_mass_wrt_prod_rate(scenario: dict, nstep: int = 1,
                                    dtime: float = 1800.0,
                                    rtol: float = 1e-5, atol: float = 1e-5
                                    ) -> Callable[[float], jnp.ndarray]:
    """Build ``f(prod_rate) -> final_total_aerosol_mass``.

    Everything else in the scenario (T0, p, RH, GMD, GSD, M) is held
    fixed; only the H2SO4 production rate is the differentiable input.

    The function returns a JAX scalar (``DTYPE``) so it can be passed
    directly to ``jax.grad``, ``jax.vmap``, or ``jax.jit``.

    Returns:
        A callable taking ``prod_rate`` [molec/cm³/s] and returning
        the final total particle mass [g/cm³] after ``nstep`` outer
        steps of duration ``dtime``.
    """
    from jax_ensemble import _minimal_config, _compute_ppm_coefs

    cfg = _minimal_config()._replace(do_coag=True, do_grow=True)
    ppm = _compute_ppm_coefs(cfg)
    grp = cfg.groups[0]
    shape = StateShape(nbin=cfg.nbin, nelem=1, ngas=2)

    rmass_np = np.asarray(grp.rmass)
    rmassup_np = np.asarray(grp.rmassup)
    rmasslow_np = np.concatenate(
        [[rmass_np[0] / (grp.rmrat ** 0.5)], rmassup_np[:-1]]
    )
    dm = jnp.asarray(rmassup_np - rmasslow_np, dtype=DTYPE)
    rmass_jnp = jnp.asarray(rmass_np, dtype=DTYPE)

    # Fixed parts of the scenario.
    T_K = float(scenario["T"])
    p_hPa = float(scenario["p"])
    rh = float(scenario["rh"])
    mu_nm = float(scenario["aerosol_mu_nm"])
    sigma_g = float(scenario["aerosol_sigma_g"])
    M_ug_m3 = float(scenario["M_total_ug_m3"])

    p_cgs_val = p_hPa * 100.0 * float(RPA2CGS)
    rhoa_g_cm3 = p_cgs_val / (float(R_AIR) * T_K)
    p_cgs = jnp.asarray([p_cgs_val], dtype=DTYPE)

    pvapl_Pa = math.exp(
        54.842763 - 6763.22 / T_K - 4.210 * math.log(T_K) + 0.000367 * T_K
    )
    h2o_mmr = rh * pvapl_Pa * 18.0 / (29.0 * p_hPa * 100.0)
    gc0_h2o = h2o_mmr * rhoa_g_cm3
    gc0_h2so4 = 0.0
    pc0 = _build_initial_pc(grp, mu_nm, sigma_g, M_ug_m3, rhoa_g_cm3)

    diff_step = _build_diff_step(shape, rtol, atol)
    H2SO4_PER_MOLEC_G = 98.078479 / float(AVG)   # g per molecule

    def f(prod_rate):
        """``prod_rate`` [molec/cm³/s] → final particle mass [g/cm³]."""
        prod_rate = jnp.asarray(prod_rate, dtype=DTYPE)
        dgc_per_step = prod_rate * DTYPE(dtime) * DTYPE(H2SO4_PER_MOLEC_G)

        pc = pc0
        gc = jnp.asarray([[gc0_h2o, gc0_h2so4]], dtype=DTYPE)
        T_scalar = jnp.asarray(T_K, dtype=DTYPE)

        for _ in range(nstep):
            gc = gc.at[0, 1].set(gc[0, 1] + dgc_per_step)
            env = _build_env(
                jnp.atleast_1d(T_scalar), p_cgs, gc, cfg, ppm, grp, dm,
            )
            pc, gc_1d, T_scalar = diff_step(
                pc, gc[0], T_scalar, DTYPE(dtime), env,
            )
            gc = gc_1d[None, :]

        # Total particle mass = Σ_b pc[b] · rmass[b]
        return jnp.sum(pc[:, 0] * rmass_jnp)

    return f
