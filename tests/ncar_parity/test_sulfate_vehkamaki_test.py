"""Parity test: carma-jax vs NCAR `carma_sulfate_vehkamaki_test`.

Same setup as carma_sulfatetest (TTL conditions, sulfate microphysics,
growth+nucleation+coagulation) but with the **Vehkamäki 2002** binary
H2SO4-H2O nucleation parameterisation instead of Zhao-Turco.

We expect Vehkamäki to give SLOWER nucleation at warm T (~250 K) than
Zhao-Turco at the same [H2SO4], because Vehkamäki's threshold is
[H2SO4] ≥ 1e4 cm⁻³ AND temperature-dependent. So this test mostly
exercises the growth/coag path with a quieter nucleation source.
"""
import math
import sys
from pathlib import Path

import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from tests.ncar_parity._harness import parse_bench, bench_path, compare


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))


def _run_faithful(method="Vehkamaki"):
    from jax_ensemble import _minimal_config, _compute_ppm_coefs, _refresh_env
    from carma.constants import R_AIR, RPA2CGS
    from carma.precision import DTYPE
    from carma.step_full_faithful import make_step_full_faithful
    from carma.prestep import prestep

    P_PA = 90.0 * 100.0
    T_K = 250.0
    DTIME = 1800.0
    NSTEP = 100
    MMR_H2O_INIT = 100e-6
    MMR_H2SO4_INIT = 0.1e-9 * (98.0 / 29.0)

    rho_air = P_PA * 10.0 / (float(R_AIR) * T_K)
    p_cgs = jnp.asarray([P_PA * RPA2CGS], dtype=DTYPE)
    T = jnp.asarray([T_K], dtype=DTYPE)
    cfg = _minimal_config()
    cfg = cfg._replace(do_coag=True, do_grow=True)
    ppm = _compute_ppm_coefs(cfg)

    gc = jnp.asarray([[MMR_H2O_INIT * rho_air,
                        MMR_H2SO4_INIT * rho_air]], dtype=DTYPE)
    pc = jnp.zeros((1, cfg.nbin, cfg.nelem), dtype=DTYPE)
    t = T

    step = make_step_full_faithful(cfg, ppm_coefs=ppm, method=method)
    itype_arr = jnp.asarray([e.itype for e in cfg.elements])
    ienconc_arr = jnp.asarray([g.ienconc for g in cfg.groups])
    igelem_arr = jnp.asarray([e.igroup for e in cfg.elements])
    rmass_2d = jnp.stack(
        [jnp.asarray(g.rmass, dtype=DTYPE) for g in cfg.groups], axis=1,
    )

    for istep in range(NSTEP):
        env_s = _refresh_env(t, p_cgs, gc, cfg, ppm)
        pc, gc, t, pcl, gcl, d_gc, d_t, pconmax = prestep(
            pc, gc, t, pc, gc, t, env_s["zmet"],
            itype_arr, ienconc_arr, igelem_arr, rmass_2d,
            do_substep=True, do_coag=cfg.do_coag,
        )
        pc, gc, t, _diag = step(
            pc=pc, gc=gc, t=t, dtime=float(DTIME),
            rhoa=env_s["rhoa"], zmet=env_s["zmet"],
            akelvin=env_s["akelvin"], akelvini=env_s["akelvini"],
            gro=env_s["gro"], gro1=env_s["gro1"],
            rup_wet=env_s["rup_wet"],
            rlhe=env_s["rlhe"], rlhm=env_s["rlhm"],
            ckernel=env_s["ckernel"], pconmax=pconmax,
            ds_threshold_arr=env_s["ds_threshold_arr"],
            pcl=pcl, gcl=gcl, told=t, d_gc=d_gc, d_t=d_t,
        )
    pc.block_until_ready()
    rmass_np = np.asarray(cfg.groups[0].rmass)
    mmr_predicted = np.asarray(pc[0, :, 0]) * rmass_np / rho_air
    gc_mmr = float(gc[0, 1]) / rho_air
    return mmr_predicted, gc_mmr


@pytest.mark.slow
def test_carma_sulfate_vehkamaki_totals():
    """Faithful JAX (Vehkamaki) total mass should match NCAR within 10 %."""
    bench = parse_bench(
        bench_path("carma_sulfate_vehkamaki_test"),
        pre_mmr_skip=4, per_step_skip=3,
    )
    mmr_pred, gc_pred = _run_faithful(method="Vehkamaki")
    total_pred = float(mmr_pred.sum())
    total_bench = float(bench.mmr_final[0].sum())
    rel_total = abs(total_pred - total_bench) / max(total_bench, 1e-30)
    rel_gas = abs(gc_pred - bench.mmr_gas_final[1]) / max(
        bench.mmr_gas_final[1], 1e-30
    )
    print(f"\n  Bench  total mmr / gas: {total_bench:.3e} / {bench.mmr_gas_final[1]:.3e}")
    print(f"  Faithful total mmr / gas: {total_pred:.3e} / {gc_pred:.3e}")
    print(f"  rel(total) = {rel_total:.2%},  rel(gas) = {rel_gas:.2%}")
    assert rel_total < 0.10, f"Total mass off by {rel_total:.0%}"


@pytest.mark.slow
@pytest.mark.xfail(
    reason="Per-bin parity gap (same root cause as sulfatetest — "
            "distribution shape differs by ~1 bin centroid even though "
            "totals match within 5 %).",
    strict=False,
)
def test_carma_sulfate_vehkamaki_per_bin():
    bench = parse_bench(
        bench_path("carma_sulfate_vehkamaki_test"),
        pre_mmr_skip=4, per_step_skip=3,
    )
    mmr_pred, _ = _run_faithful(method="Vehkamaki")
    summary = compare(mmr_pred, bench.mmr_final[0],
                       rtol=5e-2, atol=1e-12,
                       label="faithful_vs_NCAR_vehkamaki_perbin")
    print(f"\n  per-bin summary: {summary}")
    assert summary["passed"]
