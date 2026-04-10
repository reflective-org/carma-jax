# CARMA-JAX Assumptions and Hypotheses

## Confirmed

- **CGS unit system matches Fortran for numerical validation** — Fortran uses CGS internally with MKS interface. Keeping CGS internally avoids conversion-induced numerical drift during validation. (Confirmed: reading `carma_constants_mod.F90` and `CARMASTATE_Create`)

- **All main solvers are explicit Euler, NOT iterative** — `psolve`, `csolve`, `gsolve`, `tsolve` all use `pc_new = (pc_old + dt*source) / (1 + dt*loss)` or simple forward Euler. No Newton-Raphson iteration. (Confirmed: reading all solver source files)

- **Only implicit solver is versol (tridiagonal Thomas algorithm)** — Used for PPM vertical advection. Forward/backward sweep maps to `jax.lax.scan`. (Confirmed: reading `versol.F90`)

- **Coagulation is independent of growth/nucleation** — `microslow()` (coagulation) runs before `microfast()` (growth/nucleation) with no feedback path. Can be ported and validated independently. (Confirmed: reading `newstate_calc.F90` call sequence)

## Open

- **NamedTuples sufficient for state management** — Hypothesis: we won't need equinox.Module or custom pytree registration. NamedTuple._replace() and jnp.at[].set() are sufficient for all state updates. Status: Will evaluate during Phase 1 implementation.

- **Float64 required for Fortran-matching precision** — Hypothesis: float32 will not match Fortran to 1e-10. Need float64 for validation. Float32 added later as optional mode with relaxed tolerances.

- **Coagulation kernel memory is manageable** — `ckernel(NZ,NBIN,NBIN,NGROUP,NGROUP)` at NZ=80, NBIN=20, NGROUP=1 is ~2.4 MB. At NZ=200, NBIN=40, NGROUP=5 would be 128 MB. May need on-the-fly computation for large problems.

- **Python loops over NGROUP (2-10) are acceptable** — Hypothesis: the overhead of Python loops over small static dimensions is negligible compared to vectorized inner operations. Alternative: full vectorization with padding.

## Rejected

(None yet)
