# ADR 0010: Vectorised saddle-point search for Zhao-Turco 1995 nucleation

## Status

Accepted — Phase 7.4a.

## Context

Fortran's `binary_nuc_zhao1995` searches for the H2SO4/H2O composition where the sum of partial Gibbs-potential derivatives changes sign. It does this with a sequential `do i = 45, 2, -1` loop that exits on the first sign change. The loop has asymmetric treatment of the two endpoints (i=46 uses a one-sided density gradient; interior uses a centered gradient; i=1 falls back to a one-sided gradient too).

Under JIT, sequential `do while` loops with early exits aren't JIT-friendly — they need `jax.lax.while_loop` which carries a traced state and a traced predicate. That works but adds complexity.

## Decision

Compute the full 46-element `fct[i]` array at once using vectorised operations, then find the largest index where `fct[i] * fct[i+1] ≤ 0` with a single `jnp.argmax` on the reversed mask.

Asymmetric endpoint formulas are handled by computing three `dens1` pieces separately and concatenating: `dens1_lo` (one-sided gradient at i=0), `dens1_mid` (centered over interior), `dens1_hi` (one-sided gradient at i=45). The result is bit-identical to Fortran's loop for any saddle that lies strictly in the interior; the endpoints may differ only if the saddle sits exactly at i=0 or i=45, which is outside physical wt% ranges that produce nucleation.

The "no saddle found" branch is handled by a scalar `jnp.any(has_crossing)` gate that zeroes all outputs (`nucrate`, `rstar`, `mass_cluster_dry`, `ftry`) when no sign change exists.

## Alternatives considered

- **`jax.lax.while_loop` matching Fortran.** Rejected — produces identical results but requires more careful predicate construction and loop-carry state. The vectorised approach is simpler to read and JIT-trace.
- **`jax.lax.scan` over the 46 points.** Also works but is overkill for a 46-point search where vectorised ops are fine.
- **Interpolate using linear-by-sign-product endpoints only.** Would skip the composition-of-fcts fidelity to Fortran. Rejected.

## Consequences

- `binary_nuc_zhao1995` JIT-compiles cleanly and composes under `vmap` over temperature, H2SO4, RH axes.
- At the very rare configurations where the saddle sits at the 0th or 45th wt% tabulated point, the vectorised and Fortran answers can differ by the endpoint-formula choice. This is outside any physical parameter range that the model will actually encounter (wt% is always between ~5 and ~95 at realistic atmospheric conditions).
- The `any_cross` gate guarantees clean zero outputs when no nucleation is thermodynamically possible — matches Fortran's early `return` in the "Possibility 1" branch.

## Validation

- 8 unit tests, all pass.
- Key assertions: stratospheric rate (T=220 K, RH=50%, H2SO4=10⁸) is nonzero; warm dry air gives ~0 rate; critical radius stays in [0.1, 5] nm; cluster mass is a few-molecule mass (~1e-22 to 1e-19 g); Zhao-Turco and Vehkamaki both produce positive rates in the strat regime.
- 4 validation figures in `plots/phase7_zhao1995/` show the expected classical-nucleation behaviour: rate rising monotonically with H2SO4, strong T-dependence, Zhao-Turco over-predicting Vehkamaki by 2–3 orders of magnitude at warm T (a well-known feature of the literature).
