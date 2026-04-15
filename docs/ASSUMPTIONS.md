# CARMA-JAX Assumptions and Hypotheses

## Confirmed

- **CGS unit system matches Fortran for numerical validation** — Fortran uses CGS internally with MKS interface. Keeping CGS internally avoids conversion-induced numerical drift during validation. (Confirmed: reading `carma_constants_mod.F90` and `CARMASTATE_Create`)

- **All main solvers are explicit Euler, NOT iterative** — `psolve`, `csolve`, `gsolve`, `tsolve` all use `pc_new = (pc_old + dt*source) / (1 + dt*loss)` or simple forward Euler. No Newton-Raphson iteration. (Confirmed: reading all solver source files)

- **Only implicit solver is versol (tridiagonal Thomas algorithm)** — Used for PPM vertical advection. Forward/backward sweep maps to `jax.lax.scan`. (Confirmed: reading `versol.F90`)

- **Coagulation is independent of growth/nucleation** — `microslow()` (coagulation) runs before `microfast()` (growth/nucleation) with no feedback path. Can be ported and validated independently. (Confirmed: reading `newstate_calc.F90` call sequence, validated across 1000 scenarios)

- **NamedTuples sufficient for state management** — `NamedTuple._replace()` and `jnp.at[].set()` work well for all state updates in coagulation. No need for equinox. (Confirmed: Phase 1 implementation)

- **Float64 required for Fortran-matching precision** — Float64 achieves machine-precision mass conservation and <0.02% agreement with Fortran. Float32 would lose several digits. (Confirmed: Phase 1 validation)

- **Python loops over NGROUP are acceptable at trace time** — For NGROUP=1, group loops resolve at trace time inside JIT. For multi-group, these are small static loops that unroll during compilation. (Confirmed: coagulation loss computation with einsum)

## Open

- **Coagulation kernel memory is manageable** — `ckernel(NZ,NBIN,NBIN,NGROUP,NGROUP)` at NZ=1, NBIN=47, NGROUP=1 is ~17 KB. At NZ=200, NBIN=47, NGROUP=5 would be ~44 MB. May need on-the-fly computation for large multi-group problems.

- **Remaining ~0.02% error vs Fortran is from mmr conversion** — Hypothesis: the systematic error is from Fortran converting through mass mixing ratios each timestep (N→mmr→N roundtrip introduces floating-point drift). JAX operates directly in number concentration. Not a code bug. Status: consistent across all 4000+ scenarios tested, but not definitively proven.

## Rejected

(None yet)
