"""Configuration for the diffrax-based solver.

Tolerances and solver choice are decoupled from the faithful port's
`CarmaConfig` so they can be tuned without touching the reference implementation.
"""
from typing import NamedTuple


class DiffraxConfig(NamedTuple):
    """Static configuration for diffrax_step.

    rtol / atol : passed to `diffrax.PIDController(rtol, atol)`.
    max_steps   : safety cap on the number of internal diffrax steps per
                  diffeqsolve call. The PIDController controls actual count.
    solver_name : "Kvaerno5" (default) or "KenCarp4" — both DIRK methods
                  suitable for stiff chemistry. Resolved to a diffrax
                  solver instance inside step.py.
    """
    rtol: float = 1.0e-5
    atol: float = 1.0e-5
    max_steps: int = 10_000
    solver_name: str = "Kvaerno5"
