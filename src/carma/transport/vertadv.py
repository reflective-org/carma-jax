"""PPM vertical advection rates for CARMA-JAX.

Computes upward and downward advective transport rates at layer interfaces
using the Piecewise Parabolic Method (Colella & Woodard 1984).
Ported from: vertadv.F90
"""

import jax.numpy as jnp

from carma.enums import BoundaryCondition
from carma.precision import DTYPE


def vertadv(vtrans, cvert, dz, zc, zl, dtime, itbnd, ibbnd,
            cvert_tbnd, cvert_bbnd, do_explised=False):
    """Compute PPM vertical advection transport rates.

    Args:
        vtrans: Vertical velocity at layer edges (NZ+1,) [cm/s].
                Positive = upward. May be modified at boundaries.
        cvert: Quantity being transported, at layer centers (NZ,).
        dz: Layer thickness (NZ,) [cm].
        zc: Layer center altitude (NZ,) [cm].
        zl: Layer edge altitude (NZ+1,) [cm].
        dtime: Timestep [s].
        itbnd: Top boundary condition (BoundaryCondition enum).
        ibbnd: Bottom boundary condition.
        cvert_tbnd: Quantity at top boundary (for I_FIXED_CONC).
        cvert_bbnd: Quantity at bottom boundary.
        do_explised: If True, use simple upwind sorting (explicit sedimentation).

    Returns:
        (vertadvu, vertadvd) both shape (NZ+1,):
        Upward and downward transport rates [cm/s] at layer edges.
    """
    nz = cvert.shape[0]

    if do_explised:
        # Simple upwind: split positive/negative into up/down
        vertadvd = jnp.where(vtrans < DTYPE(0.0), -vtrans, DTYPE(0.0))
        vertadvu = jnp.where(vtrans >= DTYPE(0.0), vtrans, DTYPE(0.0))
        return vertadvu, vertadvd

    # Enforce flux BC at physical boundaries
    vtrans = vtrans.at[0].set(jnp.where(ibbnd == int(BoundaryCondition.I_FLUX_SPEC),
                                        DTYPE(0.0), vtrans[0]))
    vtrans = vtrans.at[nz].set(jnp.where(itbnd == int(BoundaryCondition.I_FLUX_SPEC),
                                        DTYPE(0.0), vtrans[nz]))

    # Fortran uses 1-based: nzm1=max(1,NZ-1), nzm2=max(1,NZ-2), itwo=min(2,NZ).
    # Python 0-based equivalents:
    nzm1 = max(0, nz - 2)  # Fortran NZ-1 (1-based) = Python nz-2 (0-based)
    nzm2 = max(0, nz - 3)  # Fortran NZ-2 = Python nz-3
    itwo = min(1, nz - 1)  # Fortran 2 = Python 1 (0-based)

    # --- Step 1: gradient + monotonicity (k=1..nz-2 in 0-based) ---
    dela = jnp.zeros(nz, dtype=DTYPE)
    delma = jnp.zeros(nz, dtype=DTYPE)

    # Use vectorized interior
    dpc = cvert / dz  # all
    # interior indices: 1 to nz-2 (0-based) correspond to k=2..NZ-1 (1-based)
    k_int = jnp.arange(1, nz - 1)
    dpc_k = dpc[1:-1]
    dpc_kp = dpc[2:]
    dpc_km = dpc[:-2]
    dz_k = dz[1:-1]
    dz_kp = dz[2:]
    dz_km = dz[:-2]

    ratt1 = dz_k / (dz_km + dz_k + dz_kp)
    ratt2 = (DTYPE(2.0) * dz_km + dz_k) / (dz_kp + dz_k)
    ratt3 = (DTYPE(2.0) * dz_kp + dz_k) / (dz_km + dz_k)
    dela_int = ratt1 * (ratt2 * (dpc_kp - dpc_k) + ratt3 * (dpc_k - dpc_km))

    cond = (dpc_kp - dpc_k) * (dpc_k - dpc_km) > DTYPE(0.0)
    limited = jnp.minimum(
        jnp.abs(dela_int),
        jnp.minimum(
            DTYPE(2.0) * jnp.abs(dpc_k - dpc_kp),
            DTYPE(2.0) * jnp.abs(dpc_k - dpc_km),
        ),
    ) * jnp.sign(dela_int)
    delma_int = jnp.where(cond, limited, DTYPE(0.0))

    dela = dela.at[1:-1].set(dela_int)
    delma = delma.at[1:-1].set(delma_int)

    # --- Step 2: boundary estimates aju[k] for k=1..nz-3 (0-based) ---
    aju = jnp.zeros(nz, dtype=DTYPE)
    if nz >= 4:
        k_aj = jnp.arange(1, nz - 2)
        dpc_k_a = dpc[1:-2]
        dpc_kp_a = dpc[2:-1]
        dpc_km_a = dpc[:-3]
        dz_k_a = dz[1:-2]
        dz_kp_a = dz[2:-1]
        dz_km_a = dz[:-3]
        dz_kpp_a = dz[3:]
        rat1 = dz_k_a / (dz_k_a + dz_kp_a)
        rat2 = DTYPE(2.0) * dz_kp_a * dz_k_a / (dz_k_a + dz_kp_a)
        rat3 = (dz_km_a + dz_k_a) / (DTYPE(2.0) * dz_k_a + dz_kp_a)
        rat4 = (dz_kpp_a + dz_kp_a) / (DTYPE(2.0) * dz_kp_a + dz_k_a)
        den1 = dz_km_a + dz_k_a + dz_kp_a + dz_kpp_a
        aju_int = (dpc_k_a
                   + rat1 * (dpc_kp_a - dpc_k_a)
                   + DTYPE(1.0) / den1 * (
                       rat2 * (rat3 - rat4) * (dpc_kp_a - dpc_k_a)
                       - dz_k_a * rat3 * delma[2:-1]
                       + dz_kp_a * rat4 * delma[1:-2]))
        aju = aju.at[1:-2].set(aju_int)

    # --- Step 3: al[k], ar[k] from aju + boundary linear extrapolation ---
    al = jnp.zeros(nz, dtype=DTYPE)
    ar = jnp.zeros(nz, dtype=DTYPE)

    # Interior k = 2..nz-3 (0-based, Fortran k=3..NZ-2)
    if nz >= 5:
        al = al.at[2:-2].set(aju[1:-3])
        ar = ar.at[2:-2].set(aju[2:-2])

    # Bottom boundary extrapolation (Fortran bins 1 & 2 = Python bins 0 & itwo)
    if nz >= 2:
        ar = ar.at[itwo].set(aju[itwo])
        # slope_bot: Fortran uses (cvert(itwo)/dz(itwo) - cvert(1)/dz(1)) / (zc(itwo) - zc(1))
        slope_bot = (cvert[itwo] / dz[itwo] - cvert[0] / dz[0]) / (zc[itwo] - zc[0])
        # Fortran: al(itwo) = cvert(1)/dz(1) + (zl(itwo) - zc(1)) * slope_bot
        al_itwo = cvert[0] / dz[0] + (zl[itwo] - zc[0]) * slope_bot
        al = al.at[itwo].set(al_itwo)
        # Fortran: ar(1) = al(itwo); al(1) = cvert(1)/dz(1) - (zc(1)-zl(1))*slope_bot
        ar = ar.at[0].set(al_itwo)
        al_0 = cvert[0] / dz[0] - (zc[0] - zl[0]) * slope_bot
        al = al.at[0].set(al_0)

        # Top boundary extrapolation (Fortran bins NZ-1 & NZ = Python bins nzm1 & nz-1)
        # Fortran: al(nzm1) = aju(nzm2)
        al = al.at[nzm1].set(aju[nzm2])
        # slope_top: Fortran uses (cvert(NZ)/dz(NZ) - cvert(nzm1)/dz(nzm1)) / (zc(NZ) - zc(nzm1))
        slope_top = (cvert[nz - 1] / dz[nz - 1] - cvert[nzm1] / dz[nzm1]) / (zc[nz - 1] - zc[nzm1])
        # Fortran: ar(nzm1) = cvert(nzm1)/dz(nzm1) + (zl(NZ)-zc(nzm1))*slope_top
        ar_nzm1 = cvert[nzm1] / dz[nzm1] + (zl[nz - 1] - zc[nzm1]) * slope_top
        ar = ar.at[nzm1].set(ar_nzm1)

        # Fortran: al(NZ) = ar(nzm1); ar(NZ) = cvert(nzm1)/dz(nzm1) + (zl(NZ+1)-zc(nzm1))*slope_top
        al = al.at[nz - 1].set(ar_nzm1)
        ar_last = cvert[nzm1] / dz[nzm1] + (zl[nz] - zc[nzm1]) * slope_top
        ar = ar.at[nz - 1].set(ar_last)

    # Floor boundaries to non-negative
    al = al.at[0].set(jnp.maximum(al[0], DTYPE(0.0)))
    ar = ar.at[nz - 1].set(jnp.maximum(ar[nz - 1], DTYPE(0.0)))

    # --- Step 4: monotonicity enforcement (all k=0..nz-1) ---
    outside = (ar - dpc) * (dpc - al) <= DTYPE(0.0)
    al = jnp.where(outside, dpc, al)
    ar = jnp.where(outside, dpc, ar)

    diff = ar - al
    test_val = diff * (dpc - DTYPE(0.5) * (al + ar))
    test_bound = diff * diff / DTYPE(6.0)
    al = jnp.where(test_val > test_bound, DTYPE(3.0) * dpc - DTYPE(2.0) * ar, al)
    ar = jnp.where(test_val < -test_bound, DTYPE(3.0) * dpc - DTYPE(2.0) * al, ar)

    # --- Step 5: recompute dela, a6 for flux formula ---
    dela = ar - al
    a6 = DTYPE(6.0) * (dpc - DTYPE(0.5) * (ar + al))

    # --- Step 6: flux computation at interior layer edges k=1..nz-1 (0-based) ---
    # vertadvu[k+1], vertadvd[k+1] written for k=0..nz-2 (Fortran k=1..NZ-1)
    vertadvu = jnp.zeros(nz + 1, dtype=DTYPE)
    vertadvd = jnp.zeros(nz + 1, dtype=DTYPE)

    # k from 0 to nz-2 (0-based) → edge k+1 ranges 1..nz-1 (0-based)
    k_arr = jnp.arange(nz - 1)
    com2 = (dz[:-1] + dz[1:]) / DTYPE(2.0)  # (nz-1,)
    vtrans_e = vtrans[1:nz]  # edges 1..nz-1 (0-based): interior edges
    x = vtrans_e * dtime / dz[:-1]  # dz[k]
    xpos = jnp.abs(x)

    # Upward: vtrans > 0
    pc_k = cvert[:-1]  # cvert[k] for k=0..nz-2
    pc_kp = cvert[1:]   # cvert[k+1]

    up_mask = vtrans_e > DTYPE(0.0)
    x_lt_1 = x < DTYPE(1.0)
    up_ppm = jnp.where(
        (pc_k != DTYPE(0.0)) & x_lt_1,
        vtrans_e * com2 * (
            (ar[:-1] - DTYPE(0.5) * dela[:-1] * x
             + (x / DTYPE(2.0) - (x * x) / DTYPE(3.0)) * a6[:-1])
            / jnp.where(pc_k != DTYPE(0.0), pc_k, DTYPE(1.0))
        ),
        vtrans_e,  # upwind fallback
    )
    up_vals = jnp.where(up_mask, up_ppm, DTYPE(0.0))
    vertadvu = vertadvu.at[1:nz].set(up_vals)

    # Downward: vtrans < 0
    dn_mask = vtrans_e < DTYPE(0.0)
    x_gt_neg1 = x > DTYPE(-1.0)
    dn_ppm = jnp.where(
        (pc_kp != DTYPE(0.0)) & x_gt_neg1,
        (-vtrans_e) * com2 * (
            (al[1:] + DTYPE(0.5) * dela[1:] * xpos
             + (xpos / DTYPE(2.0) - (xpos * xpos) / DTYPE(3.0)) * a6[1:])
            / jnp.where(pc_kp != DTYPE(0.0), pc_kp, DTYPE(1.0))
        ),
        -vtrans_e,  # upwind fallback
    )
    dn_vals = jnp.where(dn_mask, dn_ppm, DTYPE(0.0))
    vertadvd = vertadvd.at[1:nz].set(dn_vals)

    # --- Step 7: Lower boundary if I_FIXED_CONC ---
    # k = 0 (Fortran 1), edge 0 (Fortran 1)
    is_fixed_b = (ibbnd == int(BoundaryCondition.I_FIXED_CONC))
    com2_0 = (dz[0] + dz[itwo]) / DTYPE(2.0)
    x0 = vtrans[0] * dtime / dz[0]
    xpos0 = jnp.abs(x0)
    vt0 = vtrans[0]

    # vt0 > 0: use cvert_bbnd
    up0 = jnp.where(
        (cvert_bbnd != DTYPE(0.0)) & (x0 < DTYPE(1.0)),
        vt0 / jnp.where(cvert_bbnd != DTYPE(0.0), cvert_bbnd, DTYPE(1.0)) * com2_0
        * (ar[0] - DTYPE(0.5) * dela[0] * x0
           + (x0 / DTYPE(2.0) - (x0 * x0) / DTYPE(3.0)) * a6[0]),
        vt0,
    )
    # vt0 < 0: use cvert[0]
    dn0 = jnp.where(
        (cvert[0] != DTYPE(0.0)) & (x0 > DTYPE(-1.0)),
        (-vt0) / jnp.where(cvert[0] != DTYPE(0.0), cvert[0], DTYPE(1.0)) * com2_0
        * (al[0] + DTYPE(0.5) * dela[0] * xpos0
           + (xpos0 / DTYPE(2.0) - (xpos0 * xpos0) / DTYPE(3.0)) * a6[0]),
        -vt0,
    )
    vertadvu = vertadvu.at[0].set(
        jnp.where(is_fixed_b & (vt0 > DTYPE(0.0)), up0, vertadvu[0]))
    vertadvd = vertadvd.at[0].set(
        jnp.where(is_fixed_b & (vt0 < DTYPE(0.0)), dn0, vertadvd[0]))

    # --- Step 8: Upper boundary if I_FIXED_CONC ---
    is_fixed_t = (itbnd == int(BoundaryCondition.I_FIXED_CONC))
    com2_t = (dz[nz - 1] + dz[nzm1]) / DTYPE(2.0)
    xt = vtrans[nz] * dtime / dz[nz - 1]
    xpost = jnp.abs(xt)
    vtt = vtrans[nz]

    upt = jnp.where(
        (cvert[nz - 1] != DTYPE(0.0)) & (xt < DTYPE(1.0)),
        vtt / jnp.where(cvert[nz - 1] != DTYPE(0.0), cvert[nz - 1], DTYPE(1.0)) * com2_t
        * (ar[nz - 1] - DTYPE(0.5) * dela[nz - 1] * xt
           + (xt / DTYPE(2.0) - (xt * xt) / DTYPE(3.0)) * a6[nz - 1]),
        vtt,
    )
    dnt = jnp.where(
        (cvert_tbnd != DTYPE(0.0)) & (xt > DTYPE(-1.0)),
        (-vtt) / jnp.where(cvert_tbnd != DTYPE(0.0), cvert_tbnd, DTYPE(1.0)) * com2_t
        * (al[nz - 1] + DTYPE(0.5) * dela[nz - 1] * xpost
           + (xpost / DTYPE(2.0) - (xpost * xpost) / DTYPE(3.0)) * a6[nz - 1]),
        -vtt,
    )
    vertadvu = vertadvu.at[nz].set(
        jnp.where(is_fixed_t & (vtt > DTYPE(0.0)), upt, vertadvu[nz]))
    vertadvd = vertadvd.at[nz].set(
        jnp.where(is_fixed_t & (vtt < DTYPE(0.0)), dnt, vertadvd[nz]))

    return vertadvu, vertadvd
