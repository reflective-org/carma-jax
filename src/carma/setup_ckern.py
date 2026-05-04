"""Coagulation kernel computation for CARMA-JAX.

Fully vectorized over (NZ, NBIN, NBIN, NGROUP, NGROUP) — no Python loops
in the kernel computation. JIT-compilable.
Ported from: setupckern.F90
"""

import jax
import jax.numpy as jnp

from carma.constants import BK, GRAV, PI
from carma.precision import DTYPE


# Van der Waals enhancement constants (Chan & Mozurkewich, JAS 2001).
# Mirrors setupckern.F90:127-134. Applies between sulfate-sulfate group pairs.
_VW_A1 = DTYPE(0.0757)
_VW_A3 = DTYPE(0.0015)
_VW_B0 = DTYPE(0.0151)
_VW_B1 = DTYPE(-0.186)
_VW_B3 = DTYPE(-0.0163)
_VW_HAM = DTYPE(6.4e-13)        # erg, Hamaker constant


@jax.jit
def setup_ckern_jit(t, rhoa, zmet, rmu, r_wet, rrat, rprat, bpm, rmass, re, vf,
                    cstick, use_vw=None):
    """Compute coagulation kernels — fully vectorized, JIT-compiled.

    Computes Brownian + gravitational collection kernel for all
    (i1, i2, j1, j2) pairs simultaneously via broadcasting.

    Args:
        t: Temperature (NZ,) [K].
        rhoa: Air density * zmet (NZ,) [g/cm^2/z].
        zmet: Vertical metric (NZ,).
        rmu: Dynamic viscosity (NZ,) [g/cm/s].
        r_wet: Wet particle radius (NZ, NBIN, NGROUP) [cm].
        rrat: Radius ratio for shape (NBIN, NGROUP).
        rprat: Drag ratio for shape (NBIN, NGROUP).
        bpm: Cunningham slip correction (NZ, NBIN, NGROUP).
        rmass: Particle mass (NBIN, NGROUP) [g].
        re: Reynolds number (NZ, NBIN, NGROUP).
        vf: Fall velocity (NZ+1, NBIN, NGROUP) [cm/s].
        cstick: Sticking coefficient (scalar).
        use_vw: Optional (NGROUP, NGROUP) bool array. Where True, the
            Van der Waals enhancement (Chan & Mozurkewich 2001) replaces
            ``cstick`` with ``Einf/Enot`` for that group pair (mirrors
            setupckern.F90:271-279). Default None → no VdW (raw cstick
            everywhere). Fortran sets it to ``is_sulfate[j1] AND
            is_sulfate[j2]``.

    Returns:
        ckernel: (NZ, NBIN, NBIN, NGROUP, NGROUP) [cm^3/s].
    """
    nz = t.shape[0]
    nbin = r_wet.shape[1]
    ngroup = r_wet.shape[2]

    rhoa_cgs = rhoa / zmet  # (NZ,)

    # Common thermal quantities
    temp1 = BK * t        # (NZ,)
    temp2 = DTYPE(6.0) * PI * rmu  # (NZ,)

    # === Particle properties for ALL bins/groups at once ===
    # Shape: (NZ, NBIN, NGROUP)
    r_eff = r_wet * rrat[None, :, :]

    # Brownian diffusivity: D = kT * bpm / (6*pi*mu*r)
    diff = temp1[:, None, None] * bpm / (temp2[:, None, None] * r_eff)  # (NZ, NBIN, NGROUP)

    # Mean thermal velocity: g = sqrt(8*kT / (pi*m))
    g_vel = jnp.sqrt(
        DTYPE(8.0) * temp1[:, None, None] / (PI * rmass[None, :, :])
    )  # (NZ, NBIN, NGROUP)

    # Mean free path for Brownian: l = 8*D / (pi*g)
    rlb = DTYPE(8.0) * diff / (PI * g_vel)  # (NZ, NBIN, NGROUP)

    # Fuchs transition correction: delta
    dt1 = (DTYPE(2.0) * r_eff + rlb) ** 3
    dt2 = (DTYPE(4.0) * r_eff**2 + rlb**2) ** DTYPE(1.5)
    delt_p = (
        jnp.sqrt(DTYPE(2.0)) / (DTYPE(6.0) * r_eff * rlb) * (dt1 - dt2)
        - DTYPE(2.0) * r_eff
    )  # (NZ, NBIN, NGROUP)

    # === Broadcasting to (NZ, NBIN_i1, NBIN_i2, NGROUP_j1, NGROUP_j2) ===
    # Particle 1: axes (NZ, i1, 1, j1, 1)
    # Particle 2: axes (NZ, 1, i2, 1, j2)

    r1 = r_eff[:, :, None, :, None]           # (NZ, NB, 1, NG, 1)
    r2 = r_eff[:, None, :, None, :]           # (NZ, 1, NB, 1, NG)
    d1 = diff[:, :, None, :, None]
    d2 = diff[:, None, :, None, :]
    g1 = g_vel[:, :, None, :, None]
    g2 = g_vel[:, None, :, None, :]
    dt1_b = delt_p[:, :, None, :, None]
    dt2_b = delt_p[:, None, :, None, :]

    # === Brownian kernel ===
    rp = r1 + r2                              # (NZ, NB, NB, NG, NG)
    dp = d1 + d2

    # Van der Waals enhancement (Chan & Mozurkewich 2001) — replaces
    # ``cstick`` with ``Einf/Enot`` for sulfate-sulfate pairs.
    # Mirrors setupckern.F90:271-279.
    if use_vw is not None:
        # hp = ham/(kT) * 4·r1·r2 / (r1+r2)^2
        temp1_b = temp1[:, None, None, None, None]      # (NZ,1,1,1,1)
        hp = _VW_HAM / temp1_b * (
            DTYPE(4.0) * r1 * r2 / jnp.maximum(rp ** 2, DTYPE(1e-300))
        )
        hpln = jnp.log(DTYPE(1.0) + hp)
        Enot = DTYPE(1.0) + _VW_A1 * hpln + _VW_A3 * hpln ** 3
        sqrt_hp = jnp.sqrt(jnp.maximum(hp, DTYPE(0.0)))
        Einf = (
            DTYPE(1.0)
            + jnp.sqrt(jnp.maximum(hp / DTYPE(3.0), DTYPE(0.0)))
            / (DTYPE(1.0) + _VW_B0 * sqrt_hp)
            + _VW_B1 * hpln + _VW_B3 * hpln ** 3
        )
        cstick_vw = Einf / jnp.maximum(Enot, DTYPE(1e-300))

        # use_vw is (NG, NG); broadcast to (1, 1, 1, NG, NG)
        use_vw_b = jnp.asarray(use_vw, dtype=bool)[None, None, None, :, :]
        cstick_eff = jnp.where(use_vw_b, cstick_vw, DTYPE(cstick))
    else:
        cstick_eff = DTYPE(cstick)

    gg = jnp.sqrt(g1**2 + g2**2) * cstick_eff
    delt = jnp.sqrt(dt1_b**2 + dt2_b**2)
    term1 = rp / (rp + delt)
    term2 = DTYPE(4.0) * dp / (gg * rp)
    cbr = DTYPE(4.0) * PI * rp * dp / (term1 + term2)

    # === Gravitational collection kernel ===
    # Fall velocities at cell centers: (NZ, NBIN, NGROUP)
    vf_center = vf[:nz, :, :]
    vf1 = vf_center[:, :, None, :, None] * zmet[:, None, None, None, None]
    vf2 = vf_center[:, None, :, None, :] * zmet[:, None, None, None, None]
    dv = jnp.abs(vf1 - vf2)

    # Determine larger/smaller particle
    is_2_larger = r2 >= r1
    r_larg = jnp.maximum(r1, r2)
    r_smal = jnp.minimum(r1, r2)

    re1 = re[:, :, None, :, None]
    re2 = re[:, None, :, None, :]
    re_larg = jnp.where(is_2_larger, re2, re1)

    vf_smal = jnp.where(is_2_larger, vf1, vf2)
    vf_larg = jnp.where(is_2_larger, vf2, vf1)

    # Stokes number
    sk = jnp.where(
        r_larg > 0,
        jnp.abs(vf_smal) * jnp.abs(vf_larg - vf_smal)
        / jnp.maximum(r_larg * GRAV, DTYPE(1e-30)),
        DTYPE(0.0),
    )

    # Langmuir efficiency e1 (high Re limit)
    e1 = jnp.where(
        sk >= DTYPE(0.08333334),
        (sk / (sk + DTYPE(0.25))) ** 2,
        DTYPE(0.0),
    )

    # Fuchs efficiency e3 (low Re limit)
    e3 = jnp.where(
        sk >= DTYPE(1.214),
        DTYPE(1.0) / (
            DTYPE(1.0)
            + DTYPE(0.75) * jnp.log(jnp.maximum(DTYPE(2.0) * sk, DTYPE(1e-30)))
            / jnp.maximum(sk - DTYPE(1.214), DTYPE(1e-30))
        ) ** 2,
        DTYPE(0.0),
    )

    # Interpolate between regimes
    re60 = re_larg / DTYPE(60.0)
    e_langmuir = jnp.where(
        re_larg < DTYPE(1.0), e3,
        jnp.where(
            re_larg > DTYPE(1000.0), e1,
            (e3 + re60 * e1) / (DTYPE(1.0) + re60),
        ),
    )

    # Fuchs efficiency from radius ratio
    pr = r_smal / jnp.maximum(r_larg, DTYPE(1e-30))
    e_fuchs = (pr / (DTYPE(1.414) * (DTYPE(1.0) + pr))) ** 2

    e_coll = jnp.maximum(e_fuchs, e_langmuir)

    # Coalescence efficiency (Beard & Ochs, 1984)
    beta = (
        jnp.log(jnp.maximum(r_smal * DTYPE(1e4), DTYPE(1e-30)))
        + DTYPE(0.44) * jnp.log(jnp.maximum(r_larg * DTYPE(50.0), DTYPE(1e-30)))
    )
    b_coal = DTYPE(0.0946) * beta - DTYPE(0.319)
    a_coal = jnp.sqrt(b_coal**2 + DTYPE(0.00441))
    x_coal = (
        jnp.sign(a_coal - b_coal)
        * jnp.abs(a_coal - b_coal) ** (DTYPE(1.0) / DTYPE(3.0))
        - jnp.sign(a_coal + b_coal)
        * jnp.abs(a_coal + b_coal) ** (DTYPE(1.0) / DTYPE(3.0))
        + DTYPE(0.459)
    )
    e_coal = jnp.clip(x_coal, DTYPE(0.5), DTYPE(1.0))

    # Gravitational kernel
    cgr = e_coal * e_coll * PI * rp**2 * dv

    # Combined kernel
    ckernel = cbr + cgr

    return ckernel


def setup_ckern(config, t, rhoa, zmet, rmu, r_wet, rrat, rprat, bpm, rmass, re, vf):
    """Convenience wrapper matching the original signature.

    Builds the ``use_vw`` mask from the per-group ``is_sulfate`` flags
    (mirrors setupckern.F90:234) and forwards to ``setup_ckern_jit``.
    """
    is_sulfate = jnp.asarray(
        [bool(g.is_sulfate) for g in config.groups], dtype=bool,
    )
    use_vw = is_sulfate[:, None] & is_sulfate[None, :]
    return setup_ckern_jit(
        t, rhoa, zmet, rmu, r_wet, rrat, rprat, bpm, rmass, re, vf,
        cstick=DTYPE(config.cstick),
        use_vw=use_vw,
    )
