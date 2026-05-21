"""Piecewise-Parabolic-Method (PPM) reconstruction in mass space.

Lifts Steps 1-5 of the Colella-Woodward (1984) PPM scheme out of
``src/carma/growth/growevapl.py`` so the diffrax RHS can use them.
The faithful CARMA path uses ``growevapl`` which does the full PPM
flux calculation **with** CFL-aware time averaging (Step 6) — that
produces a per-substep growth/evap **loss rate**, not an instantaneous
boundary value, and is incompatible with a continuous-time RHS.

What we keep here:
  - Step 1: gradient estimation + monotonicity limiter
  - Step 2: bin-boundary value reconstruction
  - Step 3: per-bin parabola endpoints ``(al, ar)`` with edge handling
  - Step 4: overshoot prevention (Colella-Woodward 1.10)

What we drop:
  - Step 5/6: ``dela``/``a6``/x_grow CFL averaging. Instead we take
    the upwind parabola endpoint at each boundary:

      ``n_at_boundary[b] = ar[b]    if dmdt[b] > 0   (flow b → b+1)``
      ``                = al[b+1]   if dmdt[b] < 0   (flow b+1 → b)``

  This is the standard third-order spatial reconstruction used in
  continuous-time semi-discrete schemes; the temporal integrator
  (diffrax Kvaerno5) handles the time discretisation.

Mass-space-only: works on cell averages ``dpc = pc / dm``. No
position/radius assumptions beyond bin mass widths.

Shape conventions for a single-group sulfate diffrax path
---------------------------------------------------------
All arrays are 1-D over the bin axis (length ``nbin``):

  ``pc_elem``  (nbin,)   — particle number-density per bin [#/cm³]
  ``dm``       (nbin,)   — bin mass width [g]
  ``pratt``    (3, nbin) — PPM gradient coefficients (Step 1)
  ``prat``     (4, nbin) — PPM boundary coefficients (Step 2)
  ``pden1``    (nbin,)   — PPM denominator (Step 2)
  ``palr``     (4,)      — edge slope factors (Step 3 boundaries)
  ``dmdt``     (nbin,)   — boundary mass-growth rate; ``dmdt[b]`` is
                            the velocity at boundary between bin b
                            and b+1. ``dmdt[nbin-1]`` is unused
                            (no upper boundary), pass 0.

Output:
  ``n_at_boundary`` (nbin,) — upwind n at boundary b; the trailing
  entry (b = nbin-1) is forced to 0 since there is no bin above.

Reusing the same ``pratt/prat/pden1/palr`` arrays the faithful path
already builds via ``scripts/jax_ensemble.py:_compute_ppm_coefs`` —
this module is purely a different way of *consuming* them.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from carma.precision import DTYPE


@jax.jit
def ppm_n_at_boundary(pc_elem, dm, pratt, prat, pden1, palr, dmdt):
    """PPM-reconstructed upwind ``n`` at each bin boundary.

    See module docstring for shapes. Returns ``(nbin,)`` with the
    last entry forced to 0 (no boundary above bin nbin-1).
    """
    nbin = pc_elem.shape[0]

    # Cell-average n = pc / dm
    dpc = pc_elem / dm

    # ---- Step 1: gradient estimation + monotonicity limiter ----
    # Per Colella-Woodward eqns 1.7-1.8, the limited gradient at i is:
    #
    #   dela[i] = ratt1 · (ratt2 · (d_{i+1}-d_i) + ratt3 · (d_i-d_{i-1}))
    #
    #   delma[i] = sgn(dela) · min(|dela|, 2|d-d_{i+1}|, 2|d-d_{i-1}|)
    #              if (d_{i+1}-d_i)(d_i-d_{i-1}) > 0 else 0
    #
    # We compute the formula everywhere, then zero out the edge entries.
    d_lo = jnp.roll(dpc, 1)          # d_{i-1}, garbage at i=0
    d_hi = jnp.roll(dpc, -1)         # d_{i+1}, garbage at i=nbin-1

    dela_raw = pratt[0] * (pratt[1] * (d_hi - dpc) + pratt[2] * (dpc - d_lo))
    cond_mono = (d_hi - dpc) * (dpc - d_lo) > DTYPE(0.0)
    sgn = jnp.sign(dela_raw)
    abs_lim = jnp.minimum(
        jnp.abs(dela_raw),
        jnp.minimum(
            DTYPE(2.0) * jnp.abs(dpc - d_hi),
            DTYPE(2.0) * jnp.abs(dpc - d_lo),
        ),
    )
    delma = jnp.where(cond_mono, sgn * abs_lim, DTYPE(0.0))
    # Zero the edge entries (0 and nbin-1) where the formula referenced
    # garbage from roll wrap-around.
    edge_mask = jnp.ones(nbin, dtype=DTYPE).at[0].set(DTYPE(0.0))\
                                            .at[nbin - 1].set(DTYPE(0.0))
    delma = delma * edge_mask

    # ---- Step 2: boundary value reconstruction (aju) ----
    # Colella-Woodward eqn 1.6: the value of d at the boundary between
    # bin i and i+1, valid for i in [1, nbin-3]. Edge bins use Step 3
    # formulas instead.
    delma_hi = jnp.roll(delma, -1)
    dm_hi = jnp.roll(dm, -1)
    aju = dpc + prat[0] * (d_hi - dpc) + (
        prat[1] * (prat[2] - prat[3]) * (d_hi - dpc)
        - dm * prat[2] * delma_hi
        + dm_hi * prat[3] * delma
    ) / pden1

    # ---- Step 3: per-bin parabola endpoints (al, ar) ----
    # Interior: al[i] = aju[i-1], ar[i] = aju[i].
    # Edge bins (0, 1, nbin-2, nbin-1) use linear extrapolation from
    # palr.
    al = jnp.roll(aju, 1)    # al[i] = aju[i-1] for i >= 1
    ar = aju                  # ar[i] = aju[i]

    # Edge formulas (depend only on dpc/palr, computable upfront):
    edge_diff_low = dpc[1] - dpc[0]
    edge_diff_high = dpc[nbin - 1] - dpc[nbin - 2]
    al_at_1 = dpc[0] + palr[0] * edge_diff_low
    al_at_0 = dpc[0] + palr[1] * edge_diff_low
    ar_at_nbin_minus_2 = dpc[nbin - 2] + palr[2] * edge_diff_high
    ar_at_nbin_minus_1 = dpc[nbin - 2] + palr[3] * edge_diff_high

    # Apply in dependency order: ar[0] uses the new al[1]; al[nbin-1]
    # uses the new ar[nbin-2].
    al = al.at[1].set(al_at_1)
    al = al.at[0].set(al_at_0)
    ar = ar.at[0].set(al_at_1)
    ar = ar.at[nbin - 2].set(ar_at_nbin_minus_2)
    al = al.at[nbin - 1].set(ar_at_nbin_minus_2)
    ar = ar.at[nbin - 1].set(ar_at_nbin_minus_1)

    # ---- Step 4: overshoot prevention (Colella-Woodward 1.10) ----
    # If the cell average is outside [al, ar], the parabola can
    # overshoot; clamp to a constant (al = ar = d). Then check the
    # one-sided overshoot test and flip the offending endpoint.
    outside = (ar - dpc) * (dpc - al) <= DTYPE(0.0)
    al = jnp.where(outside, dpc, al)
    ar = jnp.where(outside, dpc, ar)
    diff = ar - al
    test1 = diff * (dpc - DTYPE(0.5) * (al + ar))
    test2 = diff ** 2 / DTYPE(6.0)
    al = jnp.where(
        test1 > test2,
        DTYPE(3.0) * dpc - DTYPE(2.0) * ar,
        al,
    )
    ar = jnp.where(
        test1 < -test2,
        DTYPE(3.0) * dpc - DTYPE(2.0) * al,
        ar,
    )

    # ---- Upwind selection at each boundary ----
    # ``ar[b]`` is the parabola value at the right edge of bin b
    # (which is the left edge of bin b+1). ``al[b+1]`` is the same
    # geometric location but viewed from bin b+1's parabola — they
    # are *not* equal in general because PPM is discontinuous at
    # boundaries (that's exactly why we need an upwind rule).
    al_above = jnp.roll(al, -1)
    n_at_boundary = jnp.where(dmdt > DTYPE(0.0), ar, al_above)

    # No boundary above bin nbin-1.
    n_at_boundary = n_at_boundary.at[nbin - 1].set(DTYPE(0.0))
    return n_at_boundary
