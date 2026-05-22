"""Operator-split coagulation wrapper for the diffrax sulfate path.

The sulfate `carma_diffrax/rhs.py` RHS does not include coagulation —
the ODE only evolves condensation, evaporation, nucleation and the
coupled gas/temperature state. To stay faithful to Fortran's
`newstate_calc.F90` flow, coag is run as a **pre-diffeqsolve** step
on each outer step (operator splitting).

This mirrors `src/carma/step_full_faithful.py:124-176`, where
`microslow_jit(pc, ckernel, pconmax, zmet, dtime)` is called once at
the start of each outer step. We do the same here.

Why operator splitting (not coag-inside-the-ODE)?

1. Fortran does it this way already (CARMASTATE_Step → newstate →
   newstate_calc applies microslow before the substep retry).
2. The continuous-time form of binary coagulation
   ``dn_i/dt = ½ Σ_jk K_jk n_j n_k δ(m_i = m_j+m_k)
              - n_i Σ_j K_ij n_j``
   requires an extra reduction inside the RHS at every diffeqsolve
   call; with rmrat=2 binning and 38 bins this isn't expensive, but
   it conflates two physical timescales (coag is slow, condensation
   is fast) and would force the implicit solver to track both.
3. Operator splitting is the standard CARMA pattern and keeps the
   diffrax RHS clean.

Limitations of operator splitting:
- One coag-update per `dtime`, so the coag effective timestep is
  `dtime`. For very long `dtime` and high pconmax, this can be
  inaccurate. The faithful path inherits the same limit.
"""
from __future__ import annotations

from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from carma.constants import FEW_PC
from carma.precision import DTYPE


class CoagBundle(NamedTuple):
    """Static + dynamic inputs for one operator-split coag call.

    ``microslow_jit`` is the JIT closure returned by
    :func:`carma.microslow.make_microslow`. ``ckernel`` is the
    coag-kernel array (NZ, NBIN, NBIN, NGROUP, NGROUP) built by
    :func:`carma.setup_ckern.setup_ckern_jit` — it depends on the
    current ``(T, rhoa, ...)`` so it's part of the dynamic env, not
    a static closure.

    Construct one of these from your existing `env_dict` produced by
    ``scripts/jax_ensemble.py:_refresh_env`` plus a one-time call to
    :func:`build_microslow_for_sulfate`.
    """
    microslow_jit: Callable
    ckernel: jnp.ndarray         # (1, nbin, nbin, ngroup, ngroup)
    zmet: jnp.ndarray            # (1,) — vertical metric


def build_microslow_for_sulfate(cfg):
    """One-time setup: build the JIT'd microslow closure for the sulfate
    config returned by ``_minimal_config()``.

    Mirrors ``step_full_faithful.py:124-139``. Returns a JIT closure
    of signature ``(pc, pcl, ckernel, pconmax, zmet, dtime) -> pc``.
    """
    from carma.microslow import make_microslow

    nbin = int(cfg.nbin)
    nelem = int(cfg.nelem)
    ngroup = int(cfg.ngroup)

    return make_microslow(
        nbin=nbin, nelem=nelem, ngroup=ngroup,
        elem_igroup=jnp.array([e.igroup for e in cfg.elements]),
        icoag=cfg.coag.icoag, volx=cfg.coag.volx,
        icoagelem=cfg.coag.icoagelem,
        npairu=cfg.coag.npairu, npairl=cfg.coag.npairl,
        iup=cfg.coag.iup, jup=cfg.coag.jup,
        igup=cfg.coag.igup, jgup=cfg.coag.jgup,
        ilow=cfg.coag.ilow, jlow=cfg.coag.jlow,
        iglow=cfg.coag.iglow, jglow=cfg.coag.jglow,
        pkernel=cfg.coag.pkernel,
        ienconc_arr=jnp.array([g.ienconc for g in cfg.groups]),
        elem_itypes=jnp.array([e.itype for e in cfg.elements]),
    )


def apply_coag(pc_2d, coag: Optional[CoagBundle], dtime: float):
    """Apply one operator-split coag step to ``pc``.

    Args:
        pc_2d: (nbin, nelem) — particle conc at outer-step start.
            Will be lifted to (1, nbin, nelem) for the microslow call.
        coag: :class:`CoagBundle` or ``None``. If ``None``, the call is a
            no-op and ``pc_2d`` is returned unchanged — this lets callers
            switch coag on/off with a single bundle argument.
        dtime: outer-step duration [s].

    Returns:
        (nbin, nelem) pc after one coag update.
    """
    if coag is None:
        return pc_2d

    # Lift to (NZ=1, nbin, nelem); mirror step_full_faithful.py:174-176.
    pc_3d = pc_2d[None, :, :]
    pcl_3d = pc_3d            # Fortran's prestep snapshot equivalent

    # pconmax = max over bins (per group). For sulfate single-group
    # single-element this is just pc_3d.max() along the bin axis.
    pconmax = jnp.max(pc_3d, axis=1, keepdims=False)   # (1, nelem) → reinterpret as (1, ngroup) for single-group
    # `make_microslow` expects pconmax of shape (NZ, NGROUP). For single-
    # group sulfate, nelem == ngroup == 1, so the shape matches directly.

    pc_after = coag.microslow_jit(
        pc_3d, pcl_3d, coag.ckernel, pconmax, coag.zmet, DTYPE(dtime),
    )
    return pc_after[0]   # back to (nbin, nelem)
