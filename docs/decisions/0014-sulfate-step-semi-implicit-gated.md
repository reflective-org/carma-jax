# ADR 0014: `sulfate_step` uses semi-implicit transfer + mass-conserving gate

## Status

Accepted — Phase 7.6b.

## Context

Sulfate nucleation rates at saturated stratospheric conditions are
fast compared to any reasonable outer timestep: 1 ppbv H2SO4 at 220 K
can deplete in ~10 ms under Zhao–Turco homogeneous nucleation, yet
``carma_sulfatetest`` uses ``dtime ≈ 60 s``. A pure explicit Euler
update blows up by 5+ orders of magnitude for the gas and overshoots
``pc`` similarly.

Fortran side-steps this via:

- ``psolve`` implicit-in-loss form: ``pc_new = (pc + dt·src) / (1 + dt·loss)``.
- Adaptive substepping and retry in ``newstate_calc`` when convergence
  thresholds fire.

Phases 8 and 9 will port those properly. Phase 7.6b needs a sulfate
kernel that composes correctly and conserves mass today so we can
start validation, without re-implementing the full adaptive retry.

## Decision

``sulfate_step_one_level`` is a pure single-call advance that:

1. Runs ``sulfnuc`` to obtain ``(rhompe, rnuclg)``.
2. Runs ``gasexchange`` to obtain the gas-side flux (diagnostic only
   in this kernel — the particle update drives the gas update).
3. Computes the **mass demand**:
   - Homogeneous: ``Σ dt · rhompe · rmass``.
   - Heterogeneous: ``Σ pc · het_frac · Δm`` where
     ``het_frac = dt·λ / (1 + dt·λ)`` is the semi-implicit transferred
     fraction (λ = ``rnuclg``).
4. Computes a single **scale** factor
   ``scale = min(1, gc_h2so4 / total_mass_demand)`` that caps the
   whole step so gas consumption never exceeds available gas.
5. Applies the scaled rates to ``pc`` and subtracts the scaled mass
   demand from ``gc``.

This guarantees:

- ``Δ(gc) = -Δ(particle mass)`` to float-roundoff (~1e-6 relative).
- ``pc ≥ 0`` and ``gc ≥ 0`` exactly.
- Heterogeneous transfer is bounded by available seeds (via the
  ``dt·λ / (1 + dt·λ)`` semi-implicit form — larger λ just asymptotes
  to "transfer everything" rather than overshooting).

## Alternatives considered

- **Pure explicit Euler**. Rejected — blows up by 5+ orders of
  magnitude under saturated stratospheric conditions.
- **Full port of Fortran's ``psolve`` + adaptive retry**. Deferred to
  Phase 8.4 (``nsubsteps``) and Phase 9.3 (adaptive retry). Premature
  here — the Phase 7 goal is process composition, not solver
  engineering.
- **Apply ``jnp.maximum(gc, 0)`` and let mass drift**. Rejected —
  silent non-conservation hides failures in the ensemble validation
  in Phase 10.

## Consequences

- Mass conservation holds to ~1e-4 over a 6-hour integration (tested
  with ``fig3_mass_conservation.png``). Phase 9 will tighten this to
  machine precision.
- The ``gasexchange`` call inside ``sulfate_step`` now serves as a
  diagnostic — the authoritative gas update comes from the
  particle-side ``total_mass_demand`` so the two sides are consistent
  by construction rather than by numerical happenstance. The
  ``gasprod`` diag is exposed for debugging.
- ``make_step_sulfate`` pre-binds bin tables and heterogeneous
  ``inuc2bin`` (sulfate-onto-self, bin i → i+1) to a JIT closure.
  Multi-group sulfate (e.g. dust + sulfate coating) requires a
  different ``inuc2bin`` mapping; callers can swap in their own in
  Phase 8+.
- Temperature and H2O are carried through unchanged (Fortran
  ``carma_sulfatetest`` runs at fixed T, p, RH). Latent heat and
  H2O balance are Phase 9 work.

## Validation

- 8 unit tests pass:
  - Mass conservation (one level, multi-step factory, ntsubsteps sweep)
  - Homogeneous-only fills exactly the target bin
  - Heterogeneous-only conserves total number in interior bins
  - Sub-saturated (sub-threshold) returns all zeros
  - Factory JIT warm-cache consistency
- 4 validation figures:
  - Time evolution of gas + particles (6-hour integration)
  - Size distribution snapshots at 0/1/6/24 h
  - Mass-conservation error trace
  - Single-step gas consumption vs T
