"""Faithful single-substep ``microfast`` for sulfate-test scope.

Mirrors Fortran's ``microfast.F90`` exactly for the sulfate-only path:
no clouds, no ice, no heterogeneous nucleation. Composes the validated
Phase 7-8 kernels (sulfate vapor pressure, supersaturation, sulfnuc,
growevapl, growp/psolve, evapp/downgevapply, gsolve, tsolve) with the
correct interleaving and sequential-data dependencies.

Two entry points:

- ``microfast_full`` — pure Python orchestration, no JIT. Useful for
  debugging because tracebacks point at real Python lines.
- ``make_microfast_full_jit`` — factory that returns a ``@jax.jit``
  closure. Inner per-(ie, ib) loop uses ``jax.lax.scan`` so trace size
  is constant in NBIN/NELEM. ~30× faster than the un-JIT'd version
  on a single-column call.

Bench result against Fortran microfast (1000 scenarios, scen_*):
- ``pc``: max rel err 1.05e-12, median 3.9e-15
- ``gc``: max rel err 8.7e-12, median 8.8e-15

Both pass rtol=1e-10 at all 1000/1000 scenarios.

Differences from the legacy ``newstate_calc.microfast_growth``:

- Computes ``pvapl/pvapi`` per-gas (Murphy 2005 for H2O, Ayers 1980
  for H2SO4 — the latter depends on H2O's pvapl).
- Computes ``supsatl/supsati`` per-gas with the gas's own gwtmol.
- Calls ``sulfnuc`` to produce ``rhompe`` (homogeneous-nucleation rate
  per bin), and threads it into ``psolve`` so the implicit Euler
  picks up the nucleation source term.
- Single-substep only — caller wraps in adaptive-retry if needed.

This implementation targets the **sulfate-test scope**:

- Single particle group (sulfate aerosol).
- Two gases: H2O (index 0) and H2SO4 (index 1).
- ``inucgas`` for the sulfate group is H2SO4.
- ``do_homogeneous=True``, ``do_heterogeneous=False`` (no I_HETNUCSULF
  mapping — see Phase 7.10 tracker row).
- No I_INVOLATILE elements, no core mass.
"""

import jax
import jax.numpy as jnp

from carma.constants import AVG
from carma.enums import RC_OK, ElementType
from carma.growth.evapp import downgevapply, evapp
from carma.growth.growevapl import growevapl
from carma.growth.growp import growp
from carma.nucleation.sulfnuc import sulfnuc
from carma.precision import DTYPE
from carma.solvers.gsolve import gsolve
from carma.solvers.psolve import psolve
from carma.solvers.totalcondensate import totalcondensate
from carma.solvers.tsolve import tsolve
from carma.sulfate_utils import wtpct_tabaz
from carma.supersaturation import supersat
from carma.utils.smallconc import maxconc
from carma.vapor_pressure import vaporp_h2o_murphy2005, vaporp_h2so4_ayers1980


def microfast_full(
    pc, gc, t, dtime,
    rhoa, zmet,
    akelvin, akelvini, gro, gro1, rup_wet,
    rmass_2d, dm_2d, rmassup, r_bins, rmrat_val,
    pratt, prat, pden1, palr,
    iz=0,
    igas_h2o=0, igas_h2so4=1,
    igroup_sulf=0, ielem_sulf=0,
    nbin=38, ngroup=1, nelem=1, ngas=2,
    gwtmol_h2o=DTYPE(18.016), gwtmol_h2so4=DTYPE(98.078479),
    method="ZhaoTurco",
    do_homogeneous=True, do_heterogeneous=False,
    rlhe=None, rlhm=None,
    ds_threshold_arr=None, dt_threshold=DTYPE(0.0),
    scale_threshold=DTYPE(1.0),
    do_pheatatm=False,
):
    """One faithful microfast substep on a single column (NZ=1).

    See module docstring for scope and divergence from legacy
    ``microfast_growth``.

    Args:
        pc: (NZ, NBIN, NELEM) particle concentrations.
        gc: (NZ, NGAS) gas concentrations.
        t: (NZ,) temperature [K].
        dtime: substep size [s].
        rhoa: (NZ,) air density · zmet [g/cm^2/z].
        zmet: (NZ,) vertical metric.
        akelvin, akelvini: (NZ, NGAS) Kelvin coefficients.
        gro, gro1: (NZ, NBIN, NGROUP) growth kernels.
        rup_wet: (NZ, NBIN, NGROUP) wet upper-boundary radii.
        rmass_2d: (NBIN, NGROUP) bin masses.
        dm_2d: (NBIN, NGROUP) bin mass widths.
        rmassup: (NBIN,) upper-bin-boundary masses for the sulfate group.
        r_bins: (NBIN,) dry radii for the sulfate group.
        rmrat_val: bin mass ratio (scalar) for the sulfate group.
        pratt: (3, NBIN, NGROUP) PPM gradient coefficients.
        prat: (4, NBIN, NGROUP) PPM boundary coefficients.
        pden1: (NBIN, NGROUP) PPM denominator.
        palr: (4, NGROUP) PPM edge slope factors.
        iz: vertical level index (default 0 — sulfate test is single-cell).
        igas_h2o, igas_h2so4: gas indices.
        igroup_sulf, ielem_sulf: sulfate group / element indices.
        nbin, ngroup, nelem, ngas: dimensions.
        gwtmol_h2o, gwtmol_h2so4: molecular weights.
        method: sulfnuc method (`"ZhaoTurco"` or `"Vehkamaki"`).
        do_homogeneous, do_heterogeneous: sulfnuc gates.
        rlhe, rlhm: (NZ, NGAS) latent-heat coefs (zero by default —
            `rlprod` doesn't affect gc, only future tsolve coupling).
        ds_threshold_arr: (NGAS,) gsolve convergence thresholds.
        dt_threshold: tsolve convergence threshold.
        scale_threshold: scaling factor for thresholds.
        do_pheatatm: enable particle heating in tsolve (default False).

    Returns:
        Tuple ``(pc, gc, t, rlheat, rc)``.
    """
    if rlhe is None:
        rlhe = jnp.zeros((1, ngas), dtype=DTYPE)
    if rlhm is None:
        rlhm = jnp.zeros((1, ngas), dtype=DTYPE)
    if ds_threshold_arr is None:
        ds_threshold_arr = jnp.zeros(ngas, dtype=DTYPE)

    # Static element/group descriptor arrays for sulfate scope.
    # Python tuples (not jnp arrays) so kernels that call int() on them
    # see Python ints at trace time. This makes microfast_full JIT-clean.
    is_ice_arr = (False,)
    igrowgas_arr = (igas_h2so4,)
    ienconc_arr = (ielem_sulf,)
    igelem_arr = (igroup_sulf,)
    itype_arr = (int(ElementType.I_VOLATILE),)

    # 1. Vapor pressure for both gases.
    pvapl_h2o, pvapi_h2o = vaporp_h2o_murphy2005(t)
    pvap_h2so4, _ = vaporp_h2so4_ayers1980(t, gc[:, igas_h2o], pvapl_h2o, zmet)
    pvapl = jnp.stack([pvapl_h2o, pvap_h2so4], axis=1)
    pvapi = jnp.stack([pvapi_h2o, pvap_h2so4], axis=1)

    # 2. Supersaturation per-gas.
    ssl_h2o, ssi_h2o = supersat(
        t, gc[:, igas_h2o], pvapl[:, igas_h2o], pvapi[:, igas_h2o],
        gwtmol_h2o, zmet,
    )
    ssl_h2so4, ssi_h2so4 = supersat(
        t, gc[:, igas_h2so4], pvapl[:, igas_h2so4], pvapi[:, igas_h2so4],
        gwtmol_h2so4, zmet,
    )
    supsatl = jnp.stack([ssl_h2o, ssl_h2so4], axis=1)
    supsati = jnp.stack([ssi_h2o, ssi_h2so4], axis=1)

    # 3. Total condensate before any updates.
    prev_ice, prev_liq = totalcondensate(
        pc, rmass_2d, igelem_arr, is_ice_arr, igrowgas_arr,
        nbin, ngroup, ngas, iz,
    )

    # 4. Sulfnuc → rhompe (homogeneous nucleation rate, per bin).
    h2o_cgs = gc[iz, igas_h2o] / zmet[iz]
    h2so4_cgs = gc[iz, igas_h2so4] / zmet[iz]
    h2o_n = h2o_cgs * AVG / gwtmol_h2o
    h2so4_n = h2so4_cgs * AVG / gwtmol_h2so4
    rh = ssl_h2o[iz] + DTYPE(1.0)
    wtp = wtpct_tabaz(t[iz], h2o_cgs, pvapl[iz, igas_h2o])
    rhompe_1d, rnuclg_1d = sulfnuc(
        t[iz], wtp, rh, h2so4_n, h2so4_cgs, h2o_n, h2o_cgs,
        r_bins, rmassup, rmrat_val, zmet[iz],
        method=method,
        do_homogeneous=do_homogeneous,
        do_heterogeneous=do_heterogeneous,
        gwtmol_h2so4=gwtmol_h2so4,
        gwtmol_h2o=gwtmol_h2o,
    )
    rhompe = jnp.zeros((nbin, nelem), dtype=DTYPE).at[:, ielem_sulf].set(rhompe_1d)
    rnuclg = jnp.zeros((nbin, ngroup, ngroup), dtype=DTYPE)

    # 5. growevapl: per-gas pvapl/supsatl drive pheat for the right igas.
    pconmax = maxconc(pc, ienconc_arr, zmet)
    growlg, evaplg = growevapl(
        pc,
        jnp.zeros((nbin, ngroup), dtype=DTYPE),
        jnp.zeros((nbin, ngroup), dtype=DTYPE),
        supsatl, supsati, pvapl, pvapi,
        akelvin, akelvini, gro, gro1, rup_wet,
        rmass_2d, dm_2d, pconmax,
        pratt, prat, pden1, palr,
        is_ice_arr, igrowgas_arr, ienconc_arr,
        dtime, iz, nbin, ngroup,
    )

    # 6. growp + psolve sequentially per (ielem, ibin) — Fortran's loop order
    # must be preserved because growp at bin i reads pc[i-1] AFTER psolve has
    # updated it (Phase 8.11 sequential-dependency finding).
    pc_nucl = jnp.zeros((1, nbin, nelem), dtype=DTYPE)
    rnucpe = jnp.zeros((nbin, nelem), dtype=DTYPE)
    evappe_zero = jnp.zeros((nbin, nelem), dtype=DTYPE)
    growpe = jnp.zeros((nbin, nelem), dtype=DTYPE)
    for ie in range(nelem):
        ig = int(igelem_arr[ie])
        for ib in range(nbin):
            growpe = growp(pc, growpe, growlg, pconmax,
                           iz, ib, ie, ig, igas_h2so4)
            pc, pc_nucl = psolve(
                pc, pc_nucl, growpe, evappe_zero, rnucpe, rhompe,
                growlg, evaplg, rnuclg, dtime, iz, ib, ie, ig, ngroup,
            )

    # 7. evapp → evappe.
    evappe = evapp(
        pc, jnp.zeros((nbin, nelem), dtype=DTYPE), evaplg, pconmax,
        ienconc_arr, itype_arr, igelem_arr, iz, nbin, ngroup, nelem,
    )

    # 8. downgevapply: pc += dt · (evappe + rnucpe) (rnucpe=0 in sulfate scope).
    pc = downgevapply(pc, evappe, jnp.zeros((nbin, nelem), dtype=DTYPE),
                      dtime, iz, nbin, nelem)

    # 9. gsolve: gc update from total-condensate change.
    curr_ice, curr_liq = totalcondensate(
        pc, rmass_2d, igelem_arr, is_ice_arr, igrowgas_arr,
        nbin, ngroup, ngas, iz,
    )
    rlprod = jnp.zeros(())
    gc, rlprod, rc_gas = gsolve(
        gc, rlprod, prev_ice, prev_liq, curr_ice, curr_liq,
        rlhe, rlhm, rhoa, dtime, iz, ngas,
        ds_threshold_arr, scale_threshold,
    )

    # 10. tsolve: temperature update from latent heat.
    rlheat = jnp.zeros(t.shape[0], dtype=DTYPE)
    partheat = jnp.zeros(t.shape[0], dtype=DTYPE)
    t, rlheat, partheat, rc_t = tsolve(
        t, rlheat, partheat, rlprod, jnp.float64(0.0),
        dtime, iz, dt_threshold, scale_threshold,
        do_pheatatm=do_pheatatm,
    )

    # rc: pass the worst of gsolve / tsolve for caller-side retry decisions.
    rc = jnp.maximum(rc_gas, rc_t)
    return pc, gc, t, rlheat[iz], rc


def make_microfast_full_jit(
    nbin=38, ngroup=1, nelem=1, ngas=2,
    igas_h2o=0, igas_h2so4=1,
    igroup_sulf=0, ielem_sulf=0,
    gwtmol_h2o=18.016, gwtmol_h2so4=98.078479,
    method="ZhaoTurco",
    do_homogeneous=True, do_heterogeneous=False,
    do_pheatatm=False,
):
    """Factory: returns a ``@jax.jit``-compiled ``microfast_full``.

    Static config (group/element/gas indices, dimensions, method strings)
    is baked into the closure. The inner per-(ie, ib) loop uses
    ``jax.lax.scan`` so the JIT trace is constant in NBIN/NELEM.

    Returns:
        ``microfast_full_jit(pc, gc, t, dtime, rhoa, zmet, akelvin,
        akelvini, gro, gro1, rup_wet, rmass_2d, dm_2d, rmassup, r_bins,
        rmrat_val, pratt, prat, pden1, palr, rlhe, rlhm,
        ds_threshold_arr, dt_threshold, scale_threshold, iz=0)`` →
        ``(pc, gc, t, rlheat_val, rc)``.
    """
    _gwtmol_h2o = DTYPE(gwtmol_h2o)
    _gwtmol_h2so4 = DTYPE(gwtmol_h2so4)
    _itype_volatile = int(ElementType.I_VOLATILE)

    @jax.jit
    def microfast_full_jit(
        pc, gc, t, dtime,
        rhoa, zmet,
        akelvin, akelvini, gro, gro1, rup_wet,
        rmass_2d, dm_2d, rmassup, r_bins, rmrat_val,
        pratt, prat, pden1, palr,
        rlhe, rlhm,
        ds_threshold_arr,
        dt_threshold=DTYPE(0.0),
        scale_threshold=DTYPE(1.0),
        iz=0,
    ):
        is_ice_arr = (False,) * ngroup
        igrowgas_arr = (igas_h2so4,) * nelem
        ienconc_arr = (ielem_sulf,) * ngroup
        igelem_arr = (igroup_sulf,) * nelem
        itype_arr = (_itype_volatile,) * nelem

        # 1. Vapor pressure for both gases.
        pvapl_h2o, pvapi_h2o = vaporp_h2o_murphy2005(t)
        pvap_h2so4, _ = vaporp_h2so4_ayers1980(
            t, gc[:, igas_h2o], pvapl_h2o, zmet,
        )
        pvapl = jnp.stack([pvapl_h2o, pvap_h2so4], axis=1)
        pvapi = jnp.stack([pvapi_h2o, pvap_h2so4], axis=1)

        # 2. Supersaturation per-gas.
        ssl_h2o, ssi_h2o = supersat(
            t, gc[:, igas_h2o], pvapl[:, igas_h2o], pvapi[:, igas_h2o],
            _gwtmol_h2o, zmet,
        )
        ssl_h2so4, ssi_h2so4 = supersat(
            t, gc[:, igas_h2so4], pvapl[:, igas_h2so4], pvapi[:, igas_h2so4],
            _gwtmol_h2so4, zmet,
        )
        supsatl = jnp.stack([ssl_h2o, ssl_h2so4], axis=1)
        supsati = jnp.stack([ssi_h2o, ssi_h2so4], axis=1)

        # 3. Total condensate (initial).
        prev_ice, prev_liq = totalcondensate(
            pc, rmass_2d, igelem_arr, is_ice_arr, igrowgas_arr,
            nbin, ngroup, ngas, iz,
        )

        # 4. Sulfnuc → rhompe.
        h2o_cgs = gc[iz, igas_h2o] / zmet[iz]
        h2so4_cgs = gc[iz, igas_h2so4] / zmet[iz]
        h2o_n = h2o_cgs * AVG / _gwtmol_h2o
        h2so4_n = h2so4_cgs * AVG / _gwtmol_h2so4
        rh = ssl_h2o[iz] + DTYPE(1.0)
        wtp = wtpct_tabaz(t[iz], h2o_cgs, pvapl[iz, igas_h2o])
        rhompe_1d, _ = sulfnuc(
            t[iz], wtp, rh, h2so4_n, h2so4_cgs, h2o_n, h2o_cgs,
            r_bins, rmassup, rmrat_val, zmet[iz],
            method=method,
            do_homogeneous=do_homogeneous,
            do_heterogeneous=do_heterogeneous,
            gwtmol_h2so4=_gwtmol_h2so4,
            gwtmol_h2o=_gwtmol_h2o,
        )
        rhompe = jnp.zeros((nbin, nelem), dtype=DTYPE).at[:, ielem_sulf].set(rhompe_1d)
        rnuclg = jnp.zeros((nbin, ngroup, ngroup), dtype=DTYPE)

        # 5. growevapl.
        pconmax = maxconc(pc, ienconc_arr, zmet)
        growlg, evaplg = growevapl(
            pc,
            jnp.zeros((nbin, ngroup), dtype=DTYPE),
            jnp.zeros((nbin, ngroup), dtype=DTYPE),
            supsatl, supsati, pvapl, pvapi,
            akelvin, akelvini, gro, gro1, rup_wet,
            rmass_2d, dm_2d, pconmax,
            pratt, prat, pden1, palr,
            is_ice_arr, igrowgas_arr, ienconc_arr,
            dtime, iz, nbin, ngroup,
        )

        # 6. growp + psolve sequentially via lax.scan over (ielem, ibin).
        # Element outer, bin inner — Fortran's loop order. Each scan step
        # reads pc[ibin-1] which has been updated by the previous psolve.
        elem_indices = jnp.repeat(jnp.arange(nelem, dtype=jnp.int32), nbin)
        bin_indices = jnp.tile(jnp.arange(nbin, dtype=jnp.int32), nelem)
        scan_indices = jnp.stack([elem_indices, bin_indices], axis=1)

        evappe_zero = jnp.zeros((nbin, nelem), dtype=DTYPE)
        rnucpe_zero = jnp.zeros((nbin, nelem), dtype=DTYPE)
        growpe_init = jnp.zeros((nbin, nelem), dtype=DTYPE)
        pc_nucl_init = jnp.zeros_like(pc)

        # `ig` per-element table — for sulfate scope NELEM=1 so it's
        # length-1, but the lookup is general.
        ig_table = jnp.asarray(igelem_arr, dtype=jnp.int32)

        def step(carry, idx):
            pc_c, pc_nucl_c, growpe_c = carry
            ie = idx[0]
            ib = idx[1]
            ig = ig_table[ie]
            growpe_c = growp(
                pc_c, growpe_c, growlg, pconmax,
                iz, ib, ie, ig, igas_h2so4,
            )
            pc_c, pc_nucl_c = psolve(
                pc_c, pc_nucl_c, growpe_c, evappe_zero, rnucpe_zero, rhompe,
                growlg, evaplg, rnuclg, dtime, iz, ib, ie, ig, ngroup,
            )
            return (pc_c, pc_nucl_c, growpe_c), None

        (pc, pc_nucl, growpe), _ = jax.lax.scan(
            step,
            (pc, pc_nucl_init, growpe_init),
            scan_indices,
        )

        # 7. evapp.
        evappe = evapp(
            pc, jnp.zeros((nbin, nelem), dtype=DTYPE), evaplg, pconmax,
            ienconc_arr, itype_arr, igelem_arr, iz, nbin, ngroup, nelem,
        )

        # 8. downgevapply.
        pc = downgevapply(pc, evappe, jnp.zeros((nbin, nelem), dtype=DTYPE),
                          dtime, iz, nbin, nelem)

        # 9. gsolve.
        curr_ice, curr_liq = totalcondensate(
            pc, rmass_2d, igelem_arr, is_ice_arr, igrowgas_arr,
            nbin, ngroup, ngas, iz,
        )
        rlprod = jnp.zeros(())
        gc, rlprod, rc_gas = gsolve(
            gc, rlprod, prev_ice, prev_liq, curr_ice, curr_liq,
            rlhe, rlhm, rhoa, dtime, iz, ngas,
            ds_threshold_arr, scale_threshold,
        )

        # 10. tsolve.
        rlheat = jnp.zeros(t.shape[0], dtype=DTYPE)
        partheat = jnp.zeros(t.shape[0], dtype=DTYPE)
        t, rlheat, partheat, rc_t = tsolve(
            t, rlheat, partheat, rlprod, jnp.float64(0.0),
            dtime, iz, dt_threshold, scale_threshold,
            do_pheatatm=do_pheatatm,
        )

        rc = jnp.maximum(rc_gas, rc_t)
        return pc, gc, t, rlheat[iz], rc

    return microfast_full_jit
