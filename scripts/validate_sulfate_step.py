"""Validation figures for make_step_sulfate (Phase 7.6b).

1. ``fig1_time_evolution.png`` — H2SO4 gas concentration + total
   particle number vs time at a stratospheric reference point.
2. ``fig2_size_distribution_evolution.png`` — particle size
   distribution at t = 0, 1 h, 6 h, 24 h.
3. ``fig3_mass_conservation.png`` — total H2SO4 mass (gas + particle)
   vs time; should be flat.
4. ``fig4_T_sweep.png`` — gas consumption rate vs T, quasi-steady
   diagnostic.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from carma.config import (
    CarmaConfig, ElementConfig, GroupConfig, GasConfig, SoluteConfig,
)
from carma.constants import AVG, BK, WTMOL_H2O
from carma.sulfate_step import make_step_sulfate
from carma.vapor_pressure import vaporp_h2o_murphy2005


OUTDIR = Path(__file__).parent.parent / "plots" / "phase7_sulfate_factory"

_GWTMOL_H2SO4 = 98.0
_GWTMOL_H2O = float(WTMOL_H2O)


def _make_config(nbin=20, rmin_cm=1e-7, rmrat=2.0, rho=1.8):
    vmin = (4.0 / 3.0) * np.pi * rmin_cm**3 * rho
    rmass = vmin * rmrat ** np.arange(nbin)
    rmassup = rmass * rmrat**0.5
    r = (3.0 * rmass / (4.0 * np.pi * rho)) ** (1.0 / 3.0)

    group = GroupConfig(
        name="sulfate", ishape=1, ienconc=0,
        is_ice=False, is_cloud=False, is_sulfate=True,
        do_vtran=False, do_drydep=False,
        ifallrtn=1, irhswell=3, rmrat=rmrat, eshape=1.0, rmin=rmin_cm,
        r=jnp.asarray(r), rmass=jnp.asarray(rmass),
        vol=jnp.asarray(rmass / rho),
        dr=jnp.asarray(r * 0.1), dm=jnp.asarray(rmass * 0.1),
        rmassup=jnp.asarray(rmassup),
        rup=jnp.asarray(r * 1.2), rlow=jnp.asarray(r * 0.8),
        rrat=jnp.ones(nbin), rprat=jnp.ones(nbin), arat=jnp.ones(nbin),
    )
    element = ElementConfig(
        name="sulfate_num", rho=jnp.full((nbin,), rho), igroup=0,
        itype=2, icomposition=0, isolute=0, kappa=0.65,
    )
    gas_h2o = GasConfig(
        name="H2O", wtmol=_GWTMOL_H2O, ivaprtn=2, icomposition=1,
        dgc_threshold=0.0, ds_threshold=0.0,
    )
    gas_h2so4 = GasConfig(
        name="H2SO4", wtmol=_GWTMOL_H2SO4, ivaprtn=4, icomposition=2,
        dgc_threshold=0.0, ds_threshold=0.0,
    )
    solute = SoluteConfig(name="sulfate", ions=3, wtmol=_GWTMOL_H2SO4, rho=1.8)

    cfg = CarmaConfig(
        nbin=nbin, nelem=1, ngroup=1, ngas=2, nsolute=1,
        elements=(element,), groups=(group,),
        gases=(gas_h2o, gas_h2so4), solutes=(solute,),
        coag=None,
        do_coag=False, do_grow=False, do_vtran=False, do_vdiff=False,
        do_thermo=False, do_substep=False, do_explised=False,
        do_incloud=False, do_clearsky=True, do_detrain=False,
        do_pheat=False, do_pheatatm=False, do_cnst_rlh=False,
        itbnd_pc=1, ibbnd_pc=1,
        maxsubsteps=32, minsubsteps=1, maxretries=4, conmax=1e-4,
        cstick=1.0, gsticki=1.0, gstickl=1.0, tstick=1.0, dt_threshold=0.0,
        igash2o=0, igash2so4=1, igasso2=-1,
    )
    return cfg, r


def _strat_state(T=220.0, rh=0.5, h2so4_ppbv=1.0, p_hpa=50.0, zmet=1.0):
    pvapl = float(vaporp_h2o_murphy2005(jnp.array(T))[0])
    p_cgs = p_hpa * 1e3
    n_air = p_cgs / (float(BK) * T)
    h2so4_num = h2so4_ppbv * 1e-9 * n_air
    h2so4_cgs = h2so4_num * _GWTMOL_H2SO4 / float(AVG)
    h2o_cgs = rh * pvapl * _GWTMOL_H2O / (float(BK) * T * float(AVG))
    gc = jnp.asarray([h2o_cgs * zmet, h2so4_cgs * zmet])
    return gc, pvapl, T


def fig_time_evolution():
    cfg, r = _make_config()
    step = make_step_sulfate(cfg, ntsubsteps=4)
    gc, pvapl, T = _strat_state()

    nstep = 360
    dtime = 60.0
    times = np.arange(nstep + 1) * dtime

    pc = jnp.zeros(cfg.nbin)
    pcs = [float(jnp.sum(pc))]
    gcs = [float(gc[1])]

    for _ in range(nstep):
        pc, gc, _ = step(pc, gc, T, pvapl, 1.0, dtime)
        pcs.append(float(jnp.sum(pc)))
        gcs.append(float(gc[1]))

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax2 = ax1.twinx()
    l1 = ax1.plot(times / 3600, np.array(gcs) / float(gc[1] + 1e-40),
                  "C0-", lw=2, label="H$_2$SO$_4$ / H$_2$SO$_4$(0)")
    l2 = ax2.plot(times / 3600, pcs, "C3--", lw=2, label="total pc [#/cm$^3$]")
    ax1.set_xlabel("time [hours]")
    ax1.set_ylabel("H$_2$SO$_4$ / initial", color="C0")
    ax2.set_ylabel("total particle number [#/cm$^3$]", color="C3")
    ax1.set_yscale("log")
    ax2.set_yscale("log")
    lines = l1 + l2
    ax1.legend(lines, [l.get_label() for l in lines], loc="lower left")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.set_title("Sulfate factory evolution (T=220K, RH=50%, 1 ppbv H2SO4)")
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig1_time_evolution.png", dpi=140)
    plt.close(fig)


def fig_size_distribution_evolution():
    cfg, r = _make_config()
    step = make_step_sulfate(cfg, ntsubsteps=4)
    gc, pvapl, T = _strat_state()

    snap_hours = [0.0, 1.0, 6.0, 24.0]
    pc = jnp.zeros(cfg.nbin)
    dtime = 60.0
    snapshots = {}
    for hr in snap_hours:
        n_steps_needed = int(hr * 3600 / dtime)
        steps_done = 0
        for _ in range(n_steps_needed - (0 if 0.0 not in snapshots else 0)):
            pc, gc, _ = step(pc, gc, T, pvapl, 1.0, dtime)
            steps_done += 1
        snapshots[hr] = np.asarray(pc).copy()

    r_nm = np.asarray(r) * 1e7
    fig, ax = plt.subplots(figsize=(7, 5))
    for hr, color in zip(snap_hours, ["C0", "C1", "C2", "C3"]):
        y = np.maximum(snapshots[hr], 1e-10)
        ax.plot(r_nm, y, "o-", color=color, lw=1.5, ms=4, label=f"t = {hr} h")
    ax.set_xlabel("bin radius [nm]")
    ax.set_ylabel("pc [#/cm$^3$]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Size distribution evolution (T=220K, RH=50%, 1 ppbv)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig2_size_distribution_evolution.png", dpi=140)
    plt.close(fig)


def fig_mass_conservation():
    cfg, r = _make_config()
    step = make_step_sulfate(cfg, ntsubsteps=4)
    gc, pvapl, T = _strat_state()

    rmass = cfg.groups[0].rmass
    pc = jnp.zeros(cfg.nbin).at[5].set(1e3)
    total0 = float(gc[1]) + float(jnp.sum(pc * rmass))

    nstep = 360
    dtime = 60.0
    times = np.arange(nstep + 1) * dtime / 3600
    errs = [0.0]

    for _ in range(nstep):
        pc, gc, _ = step(pc, gc, T, pvapl, 1.0, dtime)
        total = float(gc[1]) + float(jnp.sum(pc * rmass))
        errs.append(abs(total - total0) / total0)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(times, errs, "C0-", lw=1.5)
    ax.set_xlabel("time [hours]")
    ax.set_ylabel("|mass error| / initial")
    ax.set_yscale("log")
    ax.set_title("H$_2$SO$_4$ mass conservation (gas + particle), 6-hour run")
    ax.grid(True, which="both", alpha=0.3)
    ax.axhline(1e-4, color="red", ls="--", alpha=0.5, label="1e-4 tolerance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig3_mass_conservation.png", dpi=140)
    plt.close(fig)


def fig_T_sweep():
    cfg, r = _make_config()
    step = make_step_sulfate(cfg, ntsubsteps=4)

    T_grid = np.linspace(200.0, 290.0, 30)
    loss_rates = []
    for T in T_grid:
        gc, pvapl, _ = _strat_state(T=T)
        pc = jnp.zeros(cfg.nbin).at[5].set(1e3)
        gc_before = float(gc[1])
        pc, gc, _ = step(pc, gc, T, pvapl, 1.0, 1.0)    # 1-s step
        loss_rate = (gc_before - float(gc[1]))           # g/cm³ consumed
        loss_rates.append(max(loss_rate, 1e-40))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(T_grid, loss_rates, "C0-o", lw=2, ms=4)
    ax.set_xlabel("temperature [K]")
    ax.set_ylabel("H$_2$SO$_4$ consumed in 1 s [g/cm$^3$]")
    ax.set_yscale("log")
    ax.set_title("Sulfate factory: H2SO4 consumption in one step vs T")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig4_T_sweep.png", dpi=140)
    plt.close(fig)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig_time_evolution()
    fig_size_distribution_evolution()
    fig_mass_conservation()
    fig_T_sweep()
    print(f"Figures saved in {OUTDIR}")


if __name__ == "__main__":
    main()
