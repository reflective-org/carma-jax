"""Dry deposition velocity for CARMA-JAX.

Computes vd(NBIN, NGROUP) [cm/s] = vf_surface + 1/(ra + rs)
weighted over land/ocean/ice fractions. rs uses Zhang (2001)
resistance model with Brownian + impaction + interception.

JIT-compiled, vectorized over (NBIN, NGROUP).

Ported from: setupvdry.F90, calcrs.F90
"""

from functools import partial

import jax
import jax.numpy as jnp

from carma.constants import BK, GRAV, PI
from carma.enums import GridType
from carma.precision import DTYPE


_EPS0 = DTYPE(3.0)  # Zhang 2001 empirical constant


def _calc_rs(ustar, tmp, radi, cc, vfall, rmu_surf, rhoa_surf,
             lam, with_interception):
    """Surface resistance rs [s/cm] — Zhang 2001, vectorized.

    Args:
        ustar: Friction velocity [cm/s] (scalar).
        tmp: Surface temperature [K] (scalar).
        radi: Wet radius at surface [cm] (NBIN, NGROUP).
        cc: Slip correction bpm at surface (NBIN, NGROUP).
        vfall: Fall velocity at surface [cm/s] (NBIN, NGROUP).
        rmu_surf, rhoa_surf: Surface-layer viscosity / dry density (scalars).
        lam: Schmidt exponent (2/3 for land, 1/2 for ocean/ice).
        with_interception: If True, include interception term (land only).
    """
    eta = rmu_surf / rhoa_surf

    db = (BK * tmp * cc) / (DTYPE(6.0) * PI * rmu_surf * radi)
    sc = eta / db
    ebrn = sc ** (-lam)

    st = vfall * ustar ** 2 / (GRAV * eta)
    eimp = (st / (DTYPE(0.8) + st)) ** 2

    eint_full = DTYPE(0.3) * (
        DTYPE(0.01) * radi / (radi + DTYPE(1e-3))
        + DTYPE(0.99) * radi / (radi + DTYPE(8e-2))
    )
    eint = eint_full if with_interception else jnp.zeros_like(radi)

    rs = jnp.where(
        ustar > DTYPE(0.0),
        DTYPE(1.0) / (_EPS0 * ustar * (ebrn + eimp + eint)),
        DTYPE(0.0),
    )
    return rs


@partial(jax.jit, static_argnames=("igridv",))
def setup_vdry(vf, r_wet, bpm, t, rmu, rhoa, zmet, zmetl,
               lndfv, ocnfv, icefv, lndram, ocnram, iceram,
               lndfrac, ocnfrac, icefrac,
               grp_do_drydep, igridv):
    """Compute dry deposition velocities.

    vd(ibin, igroup) = lndfrac*vd_lnd + ocnfrac*vd_ocn + icefrac*vd_ice
    per-surface vd = vfall + 1/(ra + rs). Zero-fraction terms contribute
    zero — always computed for JIT friendliness.

    Returns vd (NBIN, NGROUP) [cm/s].
    """
    if igridv == GridType.I_CART:
        ibot = 0
        vfall = vf[0]  # (NBIN, NGROUP)
    else:
        ibot = -1
        vfall = -vf[-1] * zmetl[-1]

    rhoa_surf = rhoa[ibot] / zmet[ibot]
    t_surf = t[ibot]
    rmu_surf = rmu[ibot]
    r_surf = r_wet[ibot]
    bpm_surf = bpm[ibot]

    rs_lnd = _calc_rs(DTYPE(lndfv), t_surf, r_surf, bpm_surf, vfall,
                      rmu_surf, rhoa_surf,
                      lam=DTYPE(2.0 / 3.0), with_interception=True)
    rs_ocn = _calc_rs(DTYPE(ocnfv), t_surf, r_surf, bpm_surf, vfall,
                      rmu_surf, rhoa_surf,
                      lam=DTYPE(0.5), with_interception=False)
    rs_ice = _calc_rs(DTYPE(icefv), t_surf, r_surf, bpm_surf, vfall,
                      rmu_surf, rhoa_surf,
                      lam=DTYPE(0.5), with_interception=False)

    vd_lnd = vfall + DTYPE(1.0) / (DTYPE(lndram) + rs_lnd)
    vd_ocn = vfall + DTYPE(1.0) / (DTYPE(ocnram) + rs_ocn)
    vd_ice = vfall + DTYPE(1.0) / (DTYPE(iceram) + rs_ice)

    vd_drydep = (DTYPE(lndfrac) * vd_lnd
                 + DTYPE(ocnfrac) * vd_ocn
                 + DTYPE(icefrac) * vd_ice)

    grp_do_drydep_j = jnp.asarray(grp_do_drydep)
    vd = jnp.where(grp_do_drydep_j[None, :], vd_drydep, vfall)

    if igridv != GridType.I_CART:
        vd = -vd / zmetl[-1]

    return vd
