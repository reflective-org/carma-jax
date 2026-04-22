"""Column-scale mass-conserving fixer for negative concentration-element
mass introduced by advection.

Ports ``fixcorecol.F90``. PPM-style advection can produce tracer/tracer
imbalances where the sum of core masses exceeds the concentration-element
mass at a single level. This routine redistributes "free" concentration
mass (``pc[iepart] · rmass - total_core``) from levels where it is
positive into levels where it went negative, preserving the total column
mass of the concentration element.

Algorithm, per ``(igroup, ibin)``:

1. Compute ``total_core[iz] = Σ_i pc[iz, ibin, icorelem[i, igroup]]``.
2. Compute ``concgas_md[iz] = pc[iz, ibin, iepart] · rmass - total_core``.
3. Integrate over the column:
   - ``total_mass = Σ_{iz : concgas_md > 0} concgas_md · dz``
   - ``missing_mass = Σ_{iz : concgas_md < 0} -concgas_md · dz``
4. If ``total_mass >= missing_mass``:
      scale positive levels by
      ``factor = (total_mass - missing_mass) / total_mass``,
      zero out negative levels.
   Otherwise: there isn't enough mass to cover the holes — clamp the
   entire column to ``total_core / rmass`` (this case reports as an
   error in Fortran but our JAX kernel keeps going).

The function is a pure ``(nz, nbin, nelem)`` → ``(nz, nbin, nelem)``
array update. Groups without core masses (``ncore == 0``) are skipped.
"""

import jax
import jax.numpy as jnp
import numpy as np

from carma.precision import DTYPE


def fixcorecol(pc, rmass, dz, icorelem, ienconc, ncore):
    """Mass-conserving redistribution of concentration-element mass.

    Args:
        pc: Particle concentrations, shape ``(nz, nbin, nelem)``.
        rmass: Bin mass, shape ``(nbin, ngroup)``.
        dz: Layer thickness, shape ``(nz,)``.
        icorelem: Core-element indices, shape ``(ncore_max, ngroup)``;
            unused slots hold ``-1``. Numpy array (Python-static).
        ienconc: Number-concentration element per group,
            shape ``(ngroup,)``. Numpy array.
        ncore: Number of core elements per group, shape ``(ngroup,)``.
            Numpy array (Python-static — drives the core loop).

    Returns:
        Updated ``pc`` (same shape).
    """
    pc = jnp.asarray(pc, dtype=DTYPE)
    rmass = jnp.asarray(rmass, dtype=DTYPE)
    dz = jnp.asarray(dz, dtype=DTYPE)

    ngroup = len(ncore)
    nbin = rmass.shape[0]

    for igroup in range(ngroup):
        nc = int(ncore[igroup])
        if nc == 0:
            continue

        iepart = int(ienconc[igroup])

        # Static Python loop over bins (small, no benefit from vectorising)
        for ibin in range(nbin):
            # total_core[iz] = Σ_i pc[iz, ibin, icorelem[i, igroup]]
            core_idxs = [int(icorelem[i, igroup]) for i in range(nc)]
            total_core = jnp.zeros_like(pc[:, ibin, iepart])
            for ic in core_idxs:
                total_core = total_core + pc[:, ibin, ic]

            concgas_md = pc[:, ibin, iepart] * rmass[ibin, igroup] - total_core

            pos = concgas_md > DTYPE(0.0)
            neg = concgas_md < DTYPE(0.0)

            total_mass = jnp.sum(jnp.where(pos, concgas_md * dz, DTYPE(0.0)))
            missing_mass = jnp.sum(
                jnp.where(neg, -concgas_md * dz, DTYPE(0.0))
            )

            enough = total_mass >= missing_mass
            # factor only valid when total_mass > 0; guard against divide-by-0
            safe_total = jnp.where(total_mass > DTYPE(0.0),
                                    total_mass, DTYPE(1.0))
            factor = (total_mass - missing_mass) / safe_total

            pc_scaled = (total_core + factor * concgas_md) / rmass[ibin, igroup]
            pc_zeroed = total_core / rmass[ibin, igroup]

            # If enough mass: scale positives, zero negatives.
            # Otherwise: zero everything out to total_core / rmass.
            pc_new_enough = jnp.where(pos, pc_scaled,
                                       jnp.where(neg, pc_zeroed,
                                                 pc[:, ibin, iepart]))
            pc_new_short = pc_zeroed
            pc_new = jnp.where(enough, pc_new_enough, pc_new_short)

            # Only modify if any level had total_core > pc[iepart]*rmass
            needs_fix = jnp.any(total_core > pc[:, ibin, iepart]
                                  * rmass[ibin, igroup])
            pc_final = jnp.where(needs_fix, pc_new, pc[:, ibin, iepart])

            pc = pc.at[:, ibin, iepart].set(pc_final)

    return pc
