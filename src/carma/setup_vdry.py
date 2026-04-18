"""Dry deposition velocity for CARMA-JAX.

Computes vd(NBIN, NGROUP) [cm/s] = vf_surface + 1/(ra + rs)
weighted over land/ocean/ice fractions. rs uses Zhang (2001)
resistance model with Brownian + impaction + interception.

Ported from: setupvdry.F90, calcrs.F90
"""

import jax.numpy as jnp

from carma.constants import BK, GRAV, PI
from carma.enums import GridType
from carma.precision import DTYPE


_XKAR = DTYPE(0.4)     # Von Karman
_EPS0 = DTYPE(3.0)     # Zhang 2001 empirical constant


def _calc_rs(ustar, tmp, radi, cc, vfall, rmu_surf, rhoa_surf, landidx):
    """Surface resistance rs [s/cm] — Zhang 2001.

    Args:
        ustar: Friction velocity [cm/s] (scalar).
        tmp: Surface temperature [K] (scalar).
        radi: Wet radius at surface [cm] (NBIN, NGROUP).
        cc: Slip correction bpm at surface (NBIN, NGROUP).
        vfall: Fall velocity at surface [cm/s] (NBIN, NGROUP).
        rmu_surf: Viscosity at surface [g/cm/s] (scalar).
        rhoa_surf: Dry air density at surface [g/cm^3] (scalar).
        landidx: 1=land, 2=ocean, 3=sea ice.
    """
    eta = rmu_surf / rhoa_surf  # kinematic viscosity [cm^2/s]

    # Schmidt exponent: 2/3 land, 1/2 ocean/ice
    lam = DTYPE(2.0 / 3.0) if landidx == 1 else DTYPE(0.5)

    db = (BK * tmp * cc) / (DTYPE(6.0) * PI * rmu_surf * radi)  # Brownian diffusivity
    sc = eta / db
    ebrn = sc ** (-lam)

    st = vfall * ustar ** 2 / (GRAV * eta)
    eimp = (st / (DTYPE(0.8) + st)) ** 2

    if landidx == 1:
        eint = DTYPE(0.3) * (
            DTYPE(0.01) * radi / (radi + DTYPE(1e-3))
            + DTYPE(0.99) * radi / (radi + DTYPE(8e-2))
        )
    else:
        eint = jnp.zeros_like(radi)

    rs = jnp.where(
        ustar > DTYPE(0.0),
        DTYPE(1.0) / (_EPS0 * ustar * (ebrn + eimp + eint)),
        DTYPE(0.0),
    )
    return rs


def setup_vdry(vf, r_wet, bpm, t, rmu, rhoa, zmet, zmetl,
               lndfv, ocnfv, icefv, lndram, ocnram, iceram,
               lndfrac, ocnfrac, icefrac,
               igroup_arr, grp_do_drydep, igridv):
    """Compute dry deposition velocities.

    vd(ibin, igroup) = lndfrac*vd_lnd + ocnfrac*vd_ocn + icefrac*vd_ice
    per-surface vd = vfall + 1/(ra + rs)

    Args:
        vf: Fall velocity (NZ+1, NBIN, NGROUP) [cm/s].
        r_wet: Wet radius (NZ, NBIN, NGROUP) [cm].
        bpm: Slip correction (NZ, NBIN, NGROUP).
        t: Temperature (NZ,) [K].
        rmu: Viscosity (NZ,) [g/cm/s].
        rhoa: Air density * zmet (NZ,) [g/cm^2/z].
        zmet, zmetl: Vertical metrics.
        lndfv, ocnfv, icefv: Friction velocities [cm/s] (scalars).
        lndram, ocnram, iceram: Aerodynamic resistances [s/cm] (scalars).
        lndfrac, ocnfrac, icefrac: Surface fractions (scalars).
        igroup_arr: (NELEM,) group index per element — unused here but kept
            for API symmetry with vertical driver.
        grp_do_drydep: (NGROUP,) bool — whether drydep is active per group.
        igridv: Grid type.

    Returns:
        vd: Dry deposition velocity (NBIN, NGROUP) [cm/s].
    """
    nbin = r_wet.shape[1]
    ngroup = r_wet.shape[2]

    if igridv == GridType.I_CART:
        ibot = 0
        ibotp1 = 0  # bottom-edge index for vf = index 0
        vfall = vf[ibotp1, :, :]  # (NBIN, NGROUP) [cm/s]
    else:
        ibot = -1  # top of Fortran z-array = last Python index
        ibotp1 = -1
        vfall = -vf[ibotp1, :, :] * zmetl[ibotp1]  # convert to cm/s

    rhoa_surf = rhoa[ibot] / zmet[ibot]  # [g/cm^3]
    t_surf = t[ibot]
    rmu_surf = rmu[ibot]
    r_surf = r_wet[ibot, :, :]
    bpm_surf = bpm[ibot, :, :]

    vd = jnp.zeros((nbin, ngroup), dtype=DTYPE)

    for ig in range(ngroup):
        if not bool(grp_do_drydep[ig]):
            vd = vd.at[:, ig].set(vfall[:, ig])
            continue

        v_sum = jnp.zeros(nbin, dtype=DTYPE)

        if lndfrac > 0.0:
            rs = _calc_rs(DTYPE(lndfv), t_surf, r_surf[:, ig], bpm_surf[:, ig],
                          vfall[:, ig], rmu_surf, rhoa_surf, landidx=1)
            vd_lnd = vfall[:, ig] + DTYPE(1.0) / (DTYPE(lndram) + rs)
            v_sum = v_sum + DTYPE(lndfrac) * vd_lnd

        if ocnfrac > 0.0:
            rs = _calc_rs(DTYPE(ocnfv), t_surf, r_surf[:, ig], bpm_surf[:, ig],
                          vfall[:, ig], rmu_surf, rhoa_surf, landidx=2)
            vd_ocn = vfall[:, ig] + DTYPE(1.0) / (DTYPE(ocnram) + rs)
            v_sum = v_sum + DTYPE(ocnfrac) * vd_ocn

        if icefrac > 0.0:
            rs = _calc_rs(DTYPE(icefv), t_surf, r_surf[:, ig], bpm_surf[:, ig],
                          vfall[:, ig], rmu_surf, rhoa_surf, landidx=3)
            vd_ice = vfall[:, ig] + DTYPE(1.0) / (DTYPE(iceram) + rs)
            v_sum = v_sum + DTYPE(icefrac) * vd_ice

        vd = vd.at[:, ig].set(v_sum)

    if igridv != GridType.I_CART:
        vd = -vd / zmetl[-1]

    return vd
