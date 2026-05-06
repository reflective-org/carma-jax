# Deferred work

Items that are explicitly out of scope for the current phase but tracked
for future work. Each entry has a one‑line rationale and a pointer to
where the deferral was decided.

## Performance

### Time benchmarking of the faithful chain
**What:** Wall‑time microbench of `step_full_faithful` end‑to‑end (per
outer‑step and per inner substep), broken down by `microslow` /
`microfast_full` / env‑builder calls. Both single‑column and `vmap`'d
batch dispatch.

**Why deferred:** Phase 9 prioritised correctness over speed; the inner
microfast is JIT'd via `lax.scan` but the outer adaptive‑retry loop is
Python and the env builders run every outer step (Phase 10.5). Numbers
for the *current* implementation aren't yet representative of the final
shape.

**Pointer:** `scripts/benchmark_step_full.py` (legacy) →
needs `benchmark_step_full_faithful.py` once Phase 10.5 lands.

### `vmap` / `lax.while_loop` of the outer step
**What:** The outer step's adaptive‑retry loop is a Python `while` on a
traced `rc`. To `vmap` over scenarios or `jit` the whole outer step we
need `lax.while_loop` with a worst‑case substep ceiling.

**Why deferred:** Documented in `step_full_faithful.py` module docstring;
the Python loop is fine for differential bench / development but blocks
the 1000‑scenario ensemble from running as a single `jax.jit` call.
Currently the ensemble loops scenarios in Python (~325 ms/scenario).

**Pointer:** `src/carma/step_full_faithful.py:30‑38`,
`src/carma/newstate_calc_full.py:18‑26`.

## Numerics

### Adaptive substepping for coagulation
**What:** `microslow` (coag) currently runs once per outer step at the
full `dtime = 1800 s`. Both Fortran and JAX do this. For scenarios with
very dense aerosol (large pc × ckernel × dt product) implicit‑Euler
coag may benefit from internal substepping the same way microfast does
for nucleation/growth.

**Why deferred:** Out of scope for parity (Fortran has the same
limitation); not currently a parity gap source. Becomes relevant when
porting the model to denser regimes (boundary layer, condensational
cloud) where coag rates are large.

**Pointer:** `src/carma/microslow.py`, `src/carma/step_full_faithful.py:160‑163`.

### FP‑boundary retry‑decision drift
**What:** JAX uses fewer adaptive substeps than Fortran on the same
scenario (often 4–16× fewer at step 1 — see Phase 10.4 parity table).
The mechanism is identical (doubling on `gc < 0` / `|Δt| > 1 K` /
`|Δgc/gc| > 0.1`) but the boundary check is FP‑sensitive: JAX's FMA
fusion produces slightly different `gc` values than Fortran's
non‑fused path, so the boundary trips at different substep counts.

**Effect on parity:** Total mass remains bit‑perfect (mass conservation
is symbolic, not boundary‑sensitive). Peak bin position matches in
99% of scenarios. The remaining 33% gap on the strict per‑bin gate
(per‑scenario median ≤ 1 %, max ≤ 5 %) is essentially this effect —
**Phase 10.5 (per‑step env refresh) and Phase 10.6 (prescribed‑substep
diagnostic) together confirmed it**:

- Phase 10.5: rebuilding wet radii / Kelvin / kernels every Step (as
  Fortran does) shifts the JAX result by < 10⁻¹³ relative — the env
  drift is not the cause.
- Phase 10.6: feeding JAX Fortran's exact ntsubsteps for every outer
  step (bypassing JAX's adaptive retry) drops the strict‑gate pass
  rate from 66.8 % to **95.2 %**, total mass P95 from 2.4×10⁻³ to
  4.6×10⁻⁶, and per‑bin median P95 from 2.0×10⁻² to 2.9×10⁻⁵
  (3 orders of magnitude across the board). 284 scenarios moved from
  fail → pass; 0 moved the other way.

So the gap is essentially all retry‑boundary drift, with a small
residual (~5% scenarios) where 65 536 substeps × 100 outer steps =
6.5 M FP ops still accumulate beyond machine ε for the highest‑H₂SO₄
scenarios.

**Why deferred:** The natural fixes (disabling FMA fusion via
`XLA_FLAGS=--xla_cpu_enable_fast_math=false`, manually re‑ordering the
gsolve `gc < 0` test) all sacrifice JIT performance for marginal
parity gain. Phase 9.2 already validated that *given Fortran's
ntsubsteps* JAX matches at machine ε per substep — so the math is
right; only the adaptive‑retry boundary differs. Phase 10.6 quantifies
this: with prescribed schedule, JAX is faithful to Fortran across the
full ensemble; the adaptive variant differs in tail‑bin amplitudes
because retries trigger at different points. Acceptable as a documented
FP property of the port.

**Pointer:** `src/carma/solvers/gsolve.py:66`,
`docs/decisions/0021-adaptive-retry-nested-while-loop.md`,
Phase 10.4 commit `085e697`, Phase 10.5 commit `992d18e`,
Phase 10.6 commit (TBD).

## Scope

### Multi‑group / cloud / ice
The faithful chain (`step_full_faithful`, `microfast_full`) is scoped to
the sulfate single‑group test. Multi‑element groups, cloud droplet
activation, ice freezing/melting, and heterogeneous nucleation onto
sulfate cores are explicit Phase 11 work.

**Pointer:** `docs/ROADMAP.md` Phase 11, `microfast_full.py:34‑42`.

### Vertical transport
Sedimentation, vertical advection (PPM), vertical diffusion, dry
deposition all have Phase 4 unit tests but are not wired into
`step_full_faithful` (single‑cell test). Phase 12+ work.

**Pointer:** `docs/ROADMAP.md` Phase 4 (modules complete) + Phase 12
(integration deferred).
