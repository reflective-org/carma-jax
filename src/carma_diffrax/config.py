"""Configuration for the diffrax-based solver.

Tolerances and solver choice are decoupled from the faithful port's
`CarmaConfig` so they can be tuned without touching the reference implementation.
"""
from typing import NamedTuple


class DiffraxConfig(NamedTuple):
    """Static configuration for diffrax_step.

    Tolerances (rtol, atol) and solver_name are the obvious knobs.

    PID gains tuned via Phase 2B sweep (21_sweep_pid_gains.py on scenario 21):
    the defaults below are from the ``PI_balanced_safer`` config, which cut
    step-rejection rate from 56% (pure I controller) to 34% — about 45%
    fewer total Newton iterations.

    rtol / atol  : passed to ``diffrax.PIDController(rtol, atol)``.
    max_steps    : safety cap on internal diffrax steps per diffeqsolve.
    solver_name  : DIRK solver name; resolved in step.py.
    pcoeff       : PID proportional gain.
    icoeff       : PID integral gain.
    dcoeff       : PID derivative gain.
    factormin    : min step shrinkage ratio (default 0.2 → 0.5 = less aggressive).
    factormax    : max step growth ratio (default 10 → 5 = smoother).
    safety       : multiplicative margin on step size (default 0.9 → 0.7 = more conservative).
    """
    rtol: float = 1.0e-5
    atol: float = 1.0e-5
    max_steps: int = 10_000
    solver_name: str = "Kvaerno5"
    pcoeff: float = 0.3
    icoeff: float = 0.3
    dcoeff: float = 0.0
    factormin: float = 0.5
    factormax: float = 5.0
    safety: float = 0.7
    # Newton root-finder choice for the implicit DIRK stage.
    # Chord (optimistix) gave a 1.6× speedup on scenario 21 vs VeryChord
    # by accepting bigger diffrax outer steps (rejection 41% → 11%) with
    # the same final accuracy. Phase 2C sweep.
    root_finder_name: str = "Chord"
    # Linear solver used inside the Newton step. Dense LU is optimal for
    # the 41-dim (sulfate) / 117-dim (multispecies) state — iterative
    # solvers add overhead without exploitable sparsity at this size.
    linear_solver_name: str = "LU"
