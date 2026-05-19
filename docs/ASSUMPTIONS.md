# CARMA-JAX Assumptions and Hypotheses

## Confirmed

- **CGS unit system matches Fortran for numerical validation** — Fortran uses CGS internally with MKS interface. Keeping CGS internally avoids conversion-induced numerical drift during validation. (Confirmed: reading `carma_constants_mod.F90` and `CARMASTATE_Create`)

- **All main solvers are explicit Euler, NOT iterative** — `psolve`, `csolve`, `gsolve`, `tsolve` all use `pc_new = (pc_old + dt*source) / (1 + dt*loss)` or simple forward Euler. No Newton-Raphson iteration. (Confirmed: reading all solver source files)

- **Only implicit solver is versol (tridiagonal Thomas algorithm)** — Used for PPM vertical advection. Forward/backward sweep maps to `jax.lax.scan`. (Confirmed: reading `versol.F90`)

- **Coagulation is independent of growth/nucleation** — `microslow()` (coagulation) runs before `microfast()` (growth/nucleation) with no feedback path. Can be ported and validated independently. (Confirmed: reading `newstate_calc.F90` call sequence, validated across 1000 scenarios)

- **NamedTuples sufficient for state management** — `NamedTuple._replace()` and `jnp.at[].set()` work well for all state updates in coagulation. No need for equinox. (Confirmed: Phase 1 implementation)

- **Float64 required for Fortran-matching precision** — Float64 achieves machine-precision mass conservation and <0.02% agreement with Fortran. Float32 would lose several digits. (Confirmed: Phase 1 validation; Phase 6b overlay quantified this — coagtest/vdiftest stay within 0.1% in fp32 but falltest/drydeptest/growtest blow up because SMALL_PC=1e-50 underflows below fp32's subnormal range.)

- **Python loops over NGROUP are acceptable at trace time** — For NGROUP=1, group loops resolve at trace time inside JIT. For multi-group, these are small static loops that unroll during compilation. (Confirmed: coagulation loss computation with einsum)

- **Coagulation kernel recomputation strategy = Option A (per-step)** — The JIT-compiled setup routines are fast enough that recomputing every step (matching Fortran) is the right default. `make_step_coag` passes `ckernel` in as a traced arg; callers that vary atmospheric state between steps just rebuild it. (Resolved in Phase 5b.)

- **`rhoa_wet` vs `rhoa_dry` for pc ↔ mmr conversion** — CARMA internally stores `pc = mmr * rhoa_wet`, where `rhoa_wet` is the hydrostatic layer-mean density (`|dp|/g/dz`), *not* the ideal-gas center density (`p/R_air/T`). These differ by ~3% in the mesosphere. All validation scripts now convert via `rhoa_wet`. (Resolved during vdiftest validation, Phase 4.)

- **CARMA default particle BC is `I_FIXED_CONC`** — not `I_FLUX_SPEC`. This makes the top and bottom boundaries absorbing, so mass that advects past the surface leaves the column. (Resolved during falltest validation, Phase 4.)

## Open

- **Coagulation kernel memory is manageable** — `ckernel(NZ,NBIN,NBIN,NGROUP,NGROUP)` at NZ=1, NBIN=47, NGROUP=1 is ~17 KB. At NZ=200, NBIN=47, NGROUP=5 would be ~44 MB. May need on-the-fly computation for large multi-group problems.

- **Remaining ~0.02% error vs Fortran is from mmr conversion** — Hypothesis: the systematic error is from Fortran converting through mass mixing ratios each timestep (N→mmr→N roundtrip introduces floating-point drift). JAX operates directly in number concentration. Not a code bug. Status: consistent across all 4000+ scenarios tested, but not definitively proven.

- **Coagulation kernel recomputation strategy** — CRITICAL for Phase 3+. Currently the kernel is computed once and reused for all timesteps. This is only valid when T, p, particle radii, and air density don't change. In the full model (growth, nucleation, thermodynamics, transport), these all change every timestep, requiring kernel recomputation. Fortran CARMA recomputes every step via `setupckern` inside `CARMASTATE_Step`. Our JIT-compiled `setup_ckern_jit` is fast (0.04ms for 47 bins) so per-step recomputation is feasible, but we need a smart strategy:
  - **Option A**: Recompute every step (simple, matches Fortran). Cost: 0.04ms/step × 720 steps = 29ms — acceptable.
  - **Option B**: Recompute only when state changes significantly (e.g., T changes > threshold). Saves computation but adds branching complexity.
  - **Option C**: Compute kernel on-the-fly inside the coagulation loop (fuse setup_ckern into microslow). Avoids materializing the full (NZ,NBIN,NBIN,NG,NG) array in memory.
  - Decision deferred to Phase 5 orchestration. The JIT-compiled setup functions are fast enough that Option A is likely sufficient.

- **Size distribution tail with fixed ntsubsteps** — In the nuctest at t=100s, Fortran ice distribution extends to ~500 μm, but JAX cuts off at ~200 μm. Totals match within 1%, mass conservation is machine-precision. Fortran uses adaptive substepping (doubles when supersaturation sign-flips); our `make_step_microfast` runs fixed `ntsubsteps` and does not retry. Expected to resolve once the JIT-native adaptive retry lands (deferred — see PROGRESS.md Phase 5 deferred list, item 1). Note: smaller fixed `dtime` converges toward the Fortran tail, confirming this is a dt-resolution artefact, not a physics bug.

- **H2SO4 condensational growth** — Infrastructure exists (Ayers 1980 vapor pressure, H2SO4 diffusivity, growth kernels) and is exercised by `validate_sulfate_realistic.py`. Not yet driven through the unified `make_step_microfast` factory (which currently assumes the growtest element/gas layout); the realistic sulfate scripts still call the Phase 3 driver directly.

## Parked

- **fp32 production mode** — technically feasible (make `SMALL_PC`, `FEW_PC`, `POWMAX`, `ALMOST_ZERO` dtype-aware + add `.astype(float64)` islands around saturation vapor pressure). Not pursued because the reference Fortran is fp64 and the project's story is "matches Fortran to machine precision." Revisit only if CPU fp64 becomes a throughput bottleneck or we target GPU batched ensembles where 2× memory / 2–3× throughput justifies the mixed-precision plumbing. (Phase 6b — PR #15. The harness stays in-tree as measurement infrastructure.)

## Rejected

(None yet)
