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

- **Coagulation kernel recomputation strategy** — CRITICAL for Phase 3+. Currently the kernel is computed once and reused for all timesteps. This is only valid when T, p, particle radii, and air density don't change. In the full model (growth, nucleation, thermodynamics, transport), these all change every timestep, requiring kernel recomputation. Fortran CARMA recomputes every step via `setupckern` inside `CARMASTATE_Step`. Our JIT-compiled `setup_ckern_jit` is fast (0.04ms for 47 bins) so per-step recomputation is feasible, but we need a smart strategy:
  - **Option A**: Recompute every step (simple, matches Fortran). Cost: 0.04ms/step × 720 steps = 29ms — acceptable.
  - **Option B**: Recompute only when state changes significantly (e.g., T changes > threshold). Saves computation but adds branching complexity.
  - **Option C**: Compute kernel on-the-fly inside the coagulation loop (fuse setup_ckern into microslow). Avoids materializing the full (NZ,NBIN,NBIN,NG,NG) array in memory.
  - Decision deferred to Phase 5 orchestration. The JIT-compiled setup functions are fast enough that Option A is likely sufficient.

- **Size distribution tail with dt=1s** — In the nuctest at t=100s, Fortran ice distribution extends to ~500 μm, but JAX cuts off at ~200 μm. Totals match within 1%, but the tail shape differs. Hypothesis: the Fortran uses internal adaptive substepping (`newstate_calc.F90` doubles substeps when supersaturation sign changes) which resolves the rapid mass-space advection during the initial growth burst (t<10s). Without substepping at dt=1s, our PPM implementation loses some tail shape. Matching totals suggests the conservation laws are correct, but the advection is slightly under-resolved. Will be fixed in Phase 5 when the full `newstate_calc` orchestration with adaptive substepping is ported. Note: dt=0.01s gives the same totals with slightly better distribution, confirming this is a dt-resolution issue, not a physics error.

- **H2SO4 condensational growth** — Infrastructure exists (Ayers 1980 vapor pressure in `vapor_pressure.py`, H2SO4 diffusivity in `setup_grow.py`, growth kernels in `setup_gkern.py`), but end-to-end validation with sulfate nucleation + condensation has not been run. Planned as stratospheric and tropospheric background scenarios before Phase 4.

## Rejected

(None yet)
