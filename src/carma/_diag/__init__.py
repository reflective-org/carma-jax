"""Differential validation utilities (Phase 0).

Reads per-substep dumps produced by the Fortran diagnostic binary
``test_sulfate_diagnostic`` and exposes them as numpy arrays so the
JAX kernels can be diff'd against Fortran's actual state.

The dumps are produced by ``scripts/fortran_patch/carma_sulfatetest_diagnostic.F90``.
See ``docs/port/tracker.md`` for the validation workflow.
"""

from carma._diag.fortran_reader import (
    read_substep,
    iter_substeps,
    SubstepDump,
)

__all__ = ["read_substep", "iter_substeps", "SubstepDump"]
