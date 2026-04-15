"""Coagulation kernel computation for CARMA-JAX.

Computes the state-dependent coagulation kernel from atmospheric properties
and particle characteristics. Includes Brownian, convective enhancement,
and gravitational collection terms.
Ported from: setupckern.F90
"""

import jax.numpy as jnp

from carma.constants import BK, GRAV, PI
from carma.precision import DTYPE


def setup_ckern(config, t, rhoa, zmet, rmu, r_wet, rrat, rprat, bpm, rmass, re, vf):
    """Compute coagulation kernels for all bin/group pairs.

    Args:
        config: CarmaConfig.
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

    Returns:
        ckernel: Coagulation kernel (NZ, NBIN, NBIN, NGROUP, NGROUP) [cm^3/s].
    """
    nz = t.shape[0]
    nbin = config.nbin
    ngroup = config.ngroup

    ckernel = jnp.zeros((nz, nbin, nbin, ngroup, ngroup), dtype=DTYPE)

    # CGS air density
    rhoa_cgs = rhoa / zmet  # (NZ,)

    for j1 in range(ngroup):
        for j2 in range(ngroup):
            if config.coag.icoag[j1, j2] < 0:
                continue

            for i1 in range(nbin):
                # Particle 1 properties
                r1 = r_wet[:, i1, j1] * rrat[i1, j1]  # (NZ,)
                m1 = rmass[i1, j1]
                bpm1 = bpm[:, i1, j1]

                # Brownian diffusivity
                temp1 = BK * t  # (NZ,)
                temp2 = DTYPE(6.0) * PI * rmu  # (NZ,)
                di = temp1 * bpm1 / (temp2 * r1)  # (NZ,)

                # Mean thermal velocity
                gi = jnp.sqrt(DTYPE(8.0) * temp1 / (PI * m1))  # (NZ,)

                # Mean free path for Brownian motion
                rlbi = DTYPE(8.0) * di / (PI * gi)  # (NZ,)

                # Transition regime correction (Fuchs)
                dti1 = (DTYPE(2.0) * r1 + rlbi) ** 3
                dti2 = (DTYPE(4.0) * r1**2 + rlbi**2) ** DTYPE(1.5)
                dti = (
                    jnp.sqrt(DTYPE(2.0))
                    / (DTYPE(6.0) * r1 * rlbi)
                    * (dti1 - dti2)
                    - DTYPE(2.0) * r1
                )

                for i2 in range(nbin):
                    # Particle 2 properties
                    r2 = r_wet[:, i2, j2] * rrat[i2, j2]  # (NZ,)
                    m2 = rmass[i2, j2]
                    bpm2 = bpm[:, i2, j2]

                    dj = temp1 * bpm2 / (temp2 * r2)
                    gj = jnp.sqrt(DTYPE(8.0) * temp1 / (PI * m2))
                    rlbj = DTYPE(8.0) * dj / (PI * gj)
                    dtj1 = (DTYPE(2.0) * r2 + rlbj) ** 3
                    dtj2 = (DTYPE(4.0) * r2**2 + rlbj**2) ** DTYPE(1.5)
                    dtj = (
                        jnp.sqrt(DTYPE(2.0))
                        / (DTYPE(6.0) * r2 * rlbj)
                        * (dtj1 - dtj2)
                        - DTYPE(2.0) * r2
                    )

                    # Sticking coefficient (default=1 for non-sulfate)
                    cstick_val = DTYPE(config.cstick)

                    # Brownian coagulation kernel
                    rp = r1 + r2
                    dp = di + dj
                    gg = jnp.sqrt(gi**2 + gj**2) * cstick_val
                    delt = jnp.sqrt(dti**2 + dtj**2)
                    term1 = rp / (rp + delt)
                    term2 = DTYPE(4.0) * dp / (gg * rp)
                    cbr = DTYPE(4.0) * PI * rp * dp / (term1 + term2)  # (NZ,)

                    # Gravitational collection kernel
                    # Determine larger particle
                    r_larg = jnp.maximum(r1, r2)
                    is_2_larger = r2 >= r1

                    re_1 = re[:, i1, j1]
                    re_2 = re[:, i2, j2]
                    re_larg = jnp.where(is_2_larger, re_2, re_1)

                    vf1 = vf[:nz, i1, j1] * zmet
                    vf2 = vf[:nz, i2, j2] * zmet
                    dv = jnp.abs(vf1 - vf2)

                    # Collection efficiency (simplified: constant for now)
                    # Fuchs/Langmuir composite
                    r_smal = jnp.minimum(r1, r2)
                    vf_smal = jnp.where(is_2_larger, vf1, vf2)
                    vf_larg = jnp.where(is_2_larger, vf2, vf1)

                    # Stokes number
                    sk = jnp.where(
                        r_larg > 0,
                        jnp.abs(vf_smal) * jnp.abs(vf_larg - vf_smal) / (r_larg * GRAV),
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
                        DTYPE(1.0)
                        / (
                            DTYPE(1.0)
                            + DTYPE(0.75)
                            * jnp.log(DTYPE(2.0) * sk)
                            / jnp.maximum(sk - DTYPE(1.214), DTYPE(1e-30))
                        )
                        ** 2,
                        DTYPE(0.0),
                    )

                    # Interpolate between low and high Re
                    re60 = re_larg / DTYPE(60.0)
                    e_langmuir = jnp.where(
                        re_larg < DTYPE(1.0),
                        e3,
                        jnp.where(
                            re_larg > DTYPE(1000.0),
                            e1,
                            (e3 + re60 * e1) / (DTYPE(1.0) + re60),
                        ),
                    )

                    # Fuchs efficiency from radius ratio
                    pr = r_smal / jnp.maximum(r_larg, DTYPE(1e-30))
                    e_fuchs = (pr / (DTYPE(1.414) * (DTYPE(1.0) + pr))) ** 2

                    e_coll = jnp.maximum(e_fuchs, e_langmuir)

                    # Coalescence efficiency (Beard & Ochs, 1984)
                    # For small particles, set to 1.0
                    beta = jnp.log(jnp.maximum(r_smal * DTYPE(1e4), DTYPE(1e-30))) + DTYPE(
                        0.44
                    ) * jnp.log(jnp.maximum(r_larg * DTYPE(50.0), DTYPE(1e-30)))
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

                    # Gravitational collection kernel
                    cgr = e_coal * e_coll * PI * rp**2 * dv

                    # Combined kernel
                    ckernel = ckernel.at[:, i1, i2, j1, j2].set(cbr + cgr)

    return ckernel
