# ADR 0013: `gasexchange` uses Python loops over static dims, array ops over bins

## Status

Accepted — Phase 7.6a.

## Context

The Fortran ``gasexchange`` loops over ``igroup``, ``ienuc2``, and
``ibin`` and scatters into ``gprod_nuc[igroup, igas]`` and
``gprod_grow[igroup, igas]``. The final ``gasprod[igas]`` is a reduction
over group. Three index axes are small and static to a configured
model (``ngroup``, ``nelem``, ``ngas`` typically 1–3); one is medium
(``nbin`` ≈ 20–50).

A direct vectorisation over all four axes would build a 4-D
``diffmass``-gather tensor and a dense ``if_nuc`` selector, which costs
graph size and obscures physics. Static loops resolve at trace time
and keep each inner ``jnp`` statement on a ``(nbin,)`` vector.

## Decision

``gasexchange`` is a pure function. Its signature exposes:

- **Dynamic** arrays (traced): ``pc_iz``, ``rhompe``, ``rnuclg``,
  ``growlg``, ``evaplg``, ``diffmass``, ``cmf``, ``totevap``,
  ``inuc2bin``, ``rmass``.
- **Static / config** arrays (Python-level ints or numpy arrays,
  dereferenced via ``int(arr[i])``): ``if_nuc``, ``ienconc``,
  ``igelem``, ``inucgas``, ``nnuc2elem``, ``igrowgas``. Plus dimensions
  ``ngas``, ``ngroup``, ``nelem``, ``nbin``.

Implementation:

- Python ``for`` loops over ``igroup`` and ``ienuc2``.
- Vectorised ``jnp`` ops (diagonals, broadcasted multiplies,
  reductions) along the ``nbin`` axis.
- No ``lax.fori_loop`` — graph would not benefit, and the static
  unroll is strictly smaller for these sizes.

JIT is achieved by capturing the static config arrays in a closure
(tested in ``test_gasexchange_jits``) rather than via
``static_argnames`` (numpy arrays are not hashable for JIT's static
argument mechanism). In the eventual Phase 7.6b factory, these arrays
live on ``CarmaConfig`` and are bound via ``functools.partial``.

## Alternatives considered

- **Full 4-D vectorisation with masks.** Rejected — would require
  ``(ngroup, nelem, nbin, nbin)`` intermediate, larger than the
  physics, and would waste work on zero-flux channels.
- **``lax.fori_loop`` / ``lax.scan`` over groups.** Rejected —
  ``inucgas[igroup]`` would be traced, so Fortran's dispatch on
  "does this group have a condensing gas?" cannot early-exit without
  ``lax.cond``. Python loops handle the dispatch at trace time.
- **``static_argnames`` for every integer table.** Rejected — arrays
  of ints are not hashable. Closure capture (tested) is cleaner.

## Consequences

- Function signature lists config arrays alongside physics arrays,
  which is ugly but explicit. The Phase 7.6b factory will close over
  the config arrays and expose only ``(pc, rhompe, rnuclg, ...)`` as
  the JIT-compiled callable.
- Any caller that wants JIT must capture static config via closure
  or ``functools.partial``. Documented in the module docstring.
- Growth / evap path is included in the port even though a pure
  mass-conservation formulation (the existing ``gsolve``) can derive
  it from condensate change. The explicit-rate formulation matches
  Fortran step-by-step and lets us keep gas-exchange separable from
  the solver.

## Validation

- 10 unit tests pass:
  - shape and zero-input sanity
  - homogeneous-only with single-bin injection matches
    ``-rhompe · rmass`` exactly
  - heterogeneous-only with single-bin injection matches
    ``-pc · rnuclg · Δm`` exactly
  - top-bin heterogeneous with ``inuc2bin=-1`` zeros correctly
  - growth, partial-evap, total-evap out of bin 0
  - ``totevap`` flag routes gasgain through ``(1-cmf)·rmass`` not
    ``diffmass``
  - Full combined path matches the sum of individual-path outputs
  - JIT via closure capture produces the same answer as eager
- 3 validation figures show H2SO4 loss vs T, vs H2SO4, and as a
  (T, H2SO4) heatmap with a homogeneous = heterogeneous dominance
  line.
