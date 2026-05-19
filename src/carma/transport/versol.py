"""Tridiagonal Thomas solver for vertical transport (CARMA-JAX).

Solves the implicit vertical transport equation
    al(k)*c(k+1) + bl(k)*c(k) + ul(k)*c(k-1) = dl(k)

via Thomas algorithm (forward sweep + backward substitution), mapped
onto `jax.lax.scan` for sequential recurrences.

Ported from: versol.F90 — the only truly implicit solver in CARMA.
"""

import jax
import jax.numpy as jnp

from carma.enums import BoundaryCondition, GridType
from carma.precision import DTYPE


def versol(cvert, dz, dtime,
           itbnd, ibbnd, ftop, fbot, cvert_tbnd, cvert_bbnd,
           vertadvu, vertadvd, vertdifu, vertdifd, igridv):
    """Solve vertical transport equation implicitly (or explicitly) per bin.

    Args:
        cvert: Quantity at layer centers (NZ,).
        dz: Layer thickness (NZ,) [cm].
        dtime: Timestep [s].
        itbnd, ibbnd: Boundary condition flags.
        ftop: Downward flux across upper boundary.
        fbot: Upward flux across lower boundary.
        cvert_tbnd, cvert_bbnd: Quantities at boundaries.
        vertadvu, vertadvd: Advection rates at layer edges (NZ+1,).
        vertdifu, vertdifd: Diffusion rates at layer edges (NZ+1,).
        igridv: Grid type.

    Returns:
        cvert: Updated concentrations (NZ,).
    """
    nz = cvert.shape[0]

    # --- Determine explicit vs implicit (uc=0 or uc=1) ---
    # CFL check: if dz/dt - total_fluxes < 0 anywhere, use implicit
    cour = dz / dtime - (
        vertdifu[1:] + vertdifd[:-1] + vertadvu[1:] + vertadvd[:-1]
    )  # (NZ,)
    uc = jnp.where(jnp.any(cour < DTYPE(0.0)), DTYPE(1.0), DTYPE(0.0))

    # --- Shifted neighbors with boundary values ---
    # ctempl[k] = cvert[k-1], ctempu[k] = cvert[k+1]
    ctempl = jnp.concatenate([jnp.zeros(1, dtype=DTYPE), cvert[:-1]])
    ctempu = jnp.concatenate([cvert[1:], jnp.zeros(1, dtype=DTYPE)])

    # Override boundary values if fixed conc
    is_fixed_b = (ibbnd == int(BoundaryCondition.I_FIXED_CONC))
    is_fixed_t = (itbnd == int(BoundaryCondition.I_FIXED_CONC))
    ctempl = ctempl.at[0].set(jnp.where(is_fixed_b, cvert_bbnd, DTYPE(0.0)))
    ctempu = ctempu.at[nz - 1].set(jnp.where(is_fixed_t, cvert_tbnd, DTYPE(0.0)))

    # --- Tridiagonal coefficients ---
    # al(k)*c(k+1) + bl(k)*c(k) + ul(k)*c(k-1) = dl(k)
    al = uc * (vertdifd[1:] + vertadvd[1:])  # (NZ,) -- c(k+1) coefficient
    ul = uc * (vertdifu[:-1] + vertadvu[:-1])  # c(k-1) coefficient

    flux_total = (vertdifd[:-1] + vertdifu[1:]
                  + vertadvd[:-1] + vertadvu[1:])  # all at level k
    bl = -(uc * flux_total + dz / dtime)

    dl = cvert * ((DTYPE(1.0) - uc) * flux_total - dz / dtime) \
        - (DTYPE(1.0) - uc) * (
            (vertdifu[:-1] + vertadvu[:-1]) * ctempl
            + (vertdifd[1:] + vertadvd[1:]) * ctempu
        )
    # divcor*dz term is zero (not used in CARMA)

    # --- Apply flux boundary conditions ---
    # ftop = downward flux at top, fbot = upward flux at bottom
    is_flux_t = (itbnd == int(BoundaryCondition.I_FLUX_SPEC))
    is_flux_b = (ibbnd == int(BoundaryCondition.I_FLUX_SPEC))

    if igridv in (GridType.I_SIG, GridType.I_HYBRID):
        # Sigma/hybrid: top at index 0, bottom at index NZ-1
        dl = dl.at[0].set(jnp.where(is_flux_t, dl[0] - ftop, dl[0]))
        dl = dl.at[nz - 1].set(jnp.where(is_flux_b, dl[nz - 1] - fbot, dl[nz - 1]))
    else:  # Cartesian
        dl = dl.at[nz - 1].set(jnp.where(is_flux_t, dl[nz - 1] - ftop, dl[nz - 1]))
        dl = dl.at[0].set(jnp.where(is_flux_b, dl[0] - fbot, dl[0]))

    # --- Forward sweep (Thomas) ---
    # el[0] = dl[0]/bl[0]
    # fl[0] = al[0]/bl[0]
    # el[k] = (dl[k] - ul[k]*el[k-1]) / (bl[k] - ul[k]*fl[k-1])
    # fl[k] = al[k] / (bl[k] - ul[k]*fl[k-1])
    el0 = dl[0] / bl[0]
    fl0 = al[0] / bl[0]

    def forward_step(carry, inputs):
        el_prev, fl_prev = carry
        al_k, bl_k, ul_k, dl_k = inputs
        denom = bl_k - ul_k * fl_prev
        el_k = (dl_k - ul_k * el_prev) / denom
        fl_k = al_k / denom
        return (el_k, fl_k), (el_k, fl_k)

    # Scan over k=1..nz-1
    if nz > 1:
        init = (el0, fl0)
        inputs = (al[1:], bl[1:], ul[1:], dl[1:])
        (_, _), (el_rest, fl_rest) = jax.lax.scan(forward_step, init, inputs)
        el = jnp.concatenate([jnp.array([el0]), el_rest])
        fl = jnp.concatenate([jnp.array([fl0]), fl_rest])
    else:
        el = jnp.array([el0])
        fl = jnp.array([fl0])

    # --- Backward substitution ---
    # cvert[nz-1] = el[nz-1]
    # cvert[k] = el[k] - fl[k]*cvert[k+1]
    c_last = el[nz - 1]

    def backward_step(carry, inputs):
        c_next = carry
        el_k, fl_k = inputs
        c_k = el_k - fl_k * c_next
        return c_k, c_k

    if nz > 1:
        # Scan from k=nz-2 down to 0 (reverse)
        el_rev = el[:-1][::-1]
        fl_rev = fl[:-1][::-1]
        _, cvert_rev = jax.lax.scan(backward_step, c_last, (el_rev, fl_rev))
        cvert_new = jnp.concatenate([cvert_rev[::-1], jnp.array([c_last])])
    else:
        cvert_new = jnp.array([c_last])

    return cvert_new
