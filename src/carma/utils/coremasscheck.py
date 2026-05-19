"""Per-level core-mass sanity check and roundoff fixer.

Ports ``coremasscheck.F90``. At one vertical level, for every bin of
every group that has core masses, verify
``Σ core_mass ≤ pc[iepart] · rmass`` and optionally repair
violations.

Unlike ``fixcorecol``, this routine is **not mass-conserving** when it
repairs — it raises ``pc[iepart]`` to absorb the excess core mass.
That matches Fortran's behaviour: the routine is used both as a
diagnostic (log + abort on large errors) and as an unconditional
roundoff cleanup. The JAX port drops the logging path and always
applies the fix.

Three modes, selected by ``behavior`` (static to JIT):

- ``"roundoff"`` (default): only repair when the relative error is
  ≤ 1e-14 (Fortran's implicit behavior). Larger errors are left in
  place — callers using this mode should check the returned
  ``any_exceeded`` flag.
- ``"always"``: repair every violation, regardless of magnitude.
  Matches Fortran's ``fixcoremass=.true.``.
- ``"never"``: never repair. Useful as a pure diagnostic (returns the
  input unchanged; ``any_exceeded`` still flags the violation).

All three modes return the post-fix ``pc`` and a scalar boolean
``any_exceeded`` indicating whether any bin exceeded total mass.
"""

import jax.numpy as jnp

from carma.precision import DTYPE


_ROUNDOFF = DTYPE(1e-14)


def coremasscheck(pc, iz, rmass, icorelem, ienconc, ncore,
                  behavior="roundoff"):
    """Diagnose (and optionally fix) core-mass > total-mass at one level.

    Args:
        pc: Particle concentrations, shape ``(nz, nbin, nelem)``.
        iz: Vertical level index (Python int).
        rmass: Bin mass, shape ``(nbin, ngroup)``.
        icorelem: Core-element indices, shape ``(ncore_max, ngroup)``.
        ienconc: Number-concentration element per group.
        ncore: Number of core elements per group.
        behavior: ``"roundoff"``, ``"always"``, or ``"never"``.

    Returns:
        ``(pc_new, any_exceeded)`` where:
          - ``pc_new`` has shape ``(nz, nbin, nelem)`` and differs
            from ``pc`` only at ``[iz, ibin, iepart]`` for violating
            bins when the ``behavior`` mode allows the fix.
          - ``any_exceeded`` is a scalar boolean tracing whether any
            (igroup, ibin) at ``iz`` had coremass > total.
    """
    if behavior not in ("roundoff", "always", "never"):
        raise ValueError(
            f"coremasscheck: unknown behavior {behavior!r}. "
            "Expected 'roundoff', 'always', or 'never'.")

    pc = jnp.asarray(pc, dtype=DTYPE)
    rmass = jnp.asarray(rmass, dtype=DTYPE)

    ngroup = len(ncore)
    nbin = rmass.shape[0]
    any_exceeded = jnp.asarray(False)

    for igroup in range(ngroup):
        nc = int(ncore[igroup])
        if nc == 0:
            continue

        iepart = int(ienconc[igroup])
        core_idxs = [int(icorelem[i, igroup]) for i in range(nc)]

        for ibin in range(nbin):
            num = pc[iz, ibin, iepart]
            only_if_positive = num > DTYPE(0.0)

            coremass = jnp.zeros((), dtype=DTYPE)
            for ic in core_idxs:
                coremass = coremass + pc[iz, ibin, ic]

            total = num * rmass[ibin, igroup]
            exceeded = (coremass > total) & only_if_positive
            any_exceeded = any_exceeded | exceeded

            # Relative error vs coremass (Fortran's definition)
            safe_core = jnp.where(coremass > DTYPE(0.0), coremass, DTYPE(1.0))
            rel_err = (coremass - total) / safe_core
            is_roundoff = rel_err <= _ROUNDOFF

            if behavior == "never":
                should_fix = jnp.asarray(False)
            elif behavior == "always":
                should_fix = exceeded
            else:  # roundoff
                should_fix = exceeded & is_roundoff

            fixed = coremass / rmass[ibin, igroup]
            new_val = jnp.where(should_fix, fixed, num)
            pc = pc.at[iz, ibin, iepart].set(new_val)

    return pc, any_exceeded
