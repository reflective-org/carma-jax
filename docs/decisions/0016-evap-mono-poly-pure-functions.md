# ADR 0016: `evap_mono` and `evap_poly` are pure per-event scatter functions

## Status

Accepted — Phase 8.2.

## Context

Fortran ``evap_mono.F90`` and ``evap_poly.F90`` are called from the
dispatcher ``evapp.F90`` for each ``(ig, ibin)`` that undergoes total
evaporation. Each call mutates the module-level ``evappe(ibin, ielem)``
array, scattering evaporated number and core mass into the target CN
group. The dispatcher chooses between the two based on runtime flags
(``if_sec_mom``, ``coresig > sig_mono``, ``nuc_small``).

Porting this without restructuring would couple the JAX versions to
the dispatcher and force either shared mutable state or awkward
argument threading.

## Decision

``evap_mono`` and ``evap_poly`` are **pure functions** that each return
a ``(nbin, nelem)`` delta — the evappe contribution for a single
evaporation event. Callers add the delta to their accumulator.

Signatures take every piece of state the routine reads: ``evdrop``,
``evcore``, ``coreavg``, ``coresig`` (poly only), ``iavg``, ``ieto``,
``igto``, boundary flags (mono only), the ``rmass`` / ``dm`` /
``diffmass`` tables, and the static ``icorelem`` / ``ievp2elem`` /
``ncore`` index tables.

The full Fortran dispatcher (``evapp.F90``) is **not** ported in this
phase. It will be composed in Phase 9 when ``step_full`` needs to
dispatch on ``itype_arr`` and ``if_sec_mom``. The pure functions are
sufficient for unit-testing the scatter physics today.

## Alternatives considered

- **Port the full dispatcher in this PR**. Rejected — the dispatcher
  involves static branches on element-type and ``if_sec_mom`` flags
  that change ``evappe`` dimensions (number of target bins touched);
  the reviewer cost is high without a clear benefit until Phase 9
  needs it.
- **Take a reduced-shape ``evappe`` output (e.g., only the two bins
  ``evap_mono`` touches)**. Rejected — makes the callers responsible
  for knowing which bins each event writes. A dense ``(nbin, nelem)``
  delta is simpler and the extra zeros cost ~100 words of memory.

## Consequences

- Both functions are JIT-clean; tested under ``jax.jit``.
- ``ievp2elem = -1`` (Fortran's ``0``) is handled with a Python-level
  ``continue`` — so unmapped cores simply don't contribute. The check
  is resolved at trace time, not in the graph.
- ``evap_poly`` uses ``jnp.where`` to handle Fortran's three empty /
  non-empty kount branches in a single expression — no ``lax.cond``
  needed because the branches share a common graph structure.
- When Phase 9 wires up the dispatcher, it will call
  ``evap_mono``/``evap_poly`` inside a static ``for ibin`` loop and
  reduce via plain ``+=``. Compiled cost is the same as a single
  fused call.

## Validation

- 11 unit tests pass:
  - ``evap_mono``: number conservation in interior split; core-mass
    conservation; ``too_small`` and ``too_big`` branches place mass in
    the right boundary bin; ``conserve_mass=False`` → unit factor;
    unmapped core (``ievp2elem=-1``) is dropped; JIT equivalence.
  - ``evap_poly``: number conservation across the full CN grid; guard
    against ``coreavg=0``; verifies spread on both sides of ``iavg``;
    JIT equivalence.
- 1 validation figure showing the three ``evap_mono`` regimes and
  ``evap_poly`` scatter for three ``coresig`` values side by side.
