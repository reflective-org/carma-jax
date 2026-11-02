"""Tests for the ``fuchs_delta_sqrt2_whole`` option of ``setup_ckern_jit``.

Published CARMA (setupckern.F90:251-254) applies sqrt(2) to the bracket of
the Fuchs transition distance only:
    delta = sqrt(2)/(6 r l) [(2r+l)^3 - (4r^2+l^2)^1.5] - 2r.
With ``fuchs_delta_sqrt2_whole=True`` sqrt(2) multiplies the whole distance:
    delta = sqrt(2) {[(2r+l)^3 - (4r^2+l^2)^1.5]/(6 r l) - 2r}.

The Brownian kernel is checked against an independent NumPy transcription of
the Fuchs (1964) interpolation with fall velocities set to zero, so the
gravitational collection term vanishes.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from carma.constants import BK, GRAV, PI, R_AIR, RPA2CGS
from carma.enums import GridType
from carma.setup_atm import setup_atm
from carma.setup_ckern import setup_ckern_jit
from carma.setup_vf import setup_vf_jit

jax.config.update("jax_enable_x64", True)

T_K = 220.0
P_PA = 5000.0
RHO = 1.77                                       # g/cm^3
RADII_CM = np.array([12.5e-7, 50e-7, 100e-7, 500e-7, 2.5e-4])


def _inputs():
    t = jnp.asarray([T_K])
    p = jnp.asarray([P_PA * float(RPA2CGS)])
    rho_air = p[0] / (R_AIR * t[0])
    pl = jnp.stack([p[0] + 500 * rho_air * GRAV, p[0] - 500 * rho_air * GRAV])
    rhoa, _, zmet, _, rmu, _, _ = setup_atm(
        t, p, pl, jnp.asarray([1e5]), jnp.asarray([99500.0, 100500.0]),
        GridType.I_CART)
    r = jnp.asarray(RADII_CM)[None, :, None]
    ones = jnp.ones((len(RADII_CM), 1))
    _, re, bpm = setup_vf_jit(t, rhoa, zmet, rmu, r, jnp.full_like(ones, RHO),
                              ones, ones)
    rmass = jnp.asarray(4.0 / 3.0 * np.pi * RADII_CM**3 * RHO)[:, None]
    vf = jnp.zeros((2, len(RADII_CM), 1))       # no gravitational collection
    return (t, rhoa, zmet, rmu, r, ones, ones, bpm, rmass, re, vf, 1.0)


def _numpy_brownian(args, whole):
    t, _, _, rmu, r, _, _, bpm, rmass, _, _, _ = (np.asarray(a) for a in args)
    r = r[0, :, 0]
    kt = BK * t[0]
    diff = kt * bpm[0, :, 0] / (6.0 * PI * rmu[0] * r)
    g = np.sqrt(8.0 * kt / (PI * rmass[:, 0]))
    lb = 8.0 * diff / (PI * g)
    bracket = ((2 * r + lb) ** 3 - (4 * r**2 + lb**2) ** 1.5) / (6 * r * lb)
    delta = np.sqrt(2.0) * (bracket - 2 * r) if whole else np.sqrt(2.0) * bracket - 2 * r
    rp = r[:, None] + r[None, :]
    dp = diff[:, None] + diff[None, :]
    gg = np.sqrt(g[:, None] ** 2 + g[None, :] ** 2)
    dl = np.sqrt(delta[:, None] ** 2 + delta[None, :] ** 2)
    return 4 * PI * rp * dp / (rp / (rp + dl) + 4 * dp / (gg * rp))


def _kernel(args, **kw):
    return np.asarray(setup_ckern_jit(*args, **kw))[0, :, :, 0, 0]


def test_default_is_published_form_bit_for_bit():
    args = _inputs()
    np.testing.assert_array_equal(
        _kernel(args), _kernel(args, fuchs_delta_sqrt2_whole=False))
    np.testing.assert_array_equal(
        _kernel(args, use_vw=jnp.asarray([[True]])),
        _kernel(args, use_vw=jnp.asarray([[True]]), fuchs_delta_sqrt2_whole=False))


def test_published_form_matches_numpy():
    args = _inputs()
    npt.assert_allclose(_kernel(args), _numpy_brownian(args, whole=False), rtol=1e-12)


def test_whole_form_matches_numpy():
    args = _inputs()
    npt.assert_allclose(_kernel(args, fuchs_delta_sqrt2_whole=True),
                        _numpy_brownian(args, whole=True), rtol=1e-12)


def test_whole_form_changes_transition_regime_only():
    args = _inputs()
    ratio = (_kernel(args, fuchs_delta_sqrt2_whole=True) / _kernel(args)).diagonal()
    # 25 nm diameter is near free-molecular at 50 hPa: delta ~ sqrt(2) l either way.
    assert abs(ratio[0] - 1.0) < 0.02
    # Scaling -2r by sqrt(2) lowers delta by 2r(sqrt(2)-1), which raises the
    # r/(r+delta) term and so lowers the kernel in the transition regime.
    assert np.all(ratio[1:4] < 1.0)
    assert not np.allclose(ratio, 1.0, rtol=1e-3)
