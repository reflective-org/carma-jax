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


def _run_with_source(cfg, gc0, pvapl, T, source_h2so4_per_s,
                      schedule, ntsubsteps=1):
    """Integrate under a prescribed H2SO4 production rate.

    ``schedule`` is a list of ``(dt_s, n_steps)`` pairs; the source
    rate is applied as an explicit increment to gc before each step.
    Returns logged (times, gc_h2so4, total_pc) per step.
    """
    step = make_step_sulfate(cfg, ntsubsteps=ntsubsteps)
    pc = jnp.zeros(cfg.nbin)
    gc = gc0
    t_cur = 0.0
    times = [0.0]
    gc_log = [float(gc[1])]
    pc_log = [0.0]
    for dt, n in schedule:
        for _ in range(n):
            # Inject fresh H2SO4 (SO2 + OH photochemistry surrogate)
            gc = gc.at[1].add(source_h2so4_per_s * dt)
            pc, gc, _ = step(pc, gc, T, pvapl, 1.0, dt)
            t_cur += dt
            times.append(t_cur)
            gc_log.append(float(gc[1]))
            pc_log.append(float(jnp.sum(pc)))
    return np.asarray(times), np.asarray(gc_log), np.asarray(pc_log)


def fig_time_evolution():
    """Two-panel time evolution.

    Left: 'puff' scenario — initial H2SO4, no source. Depletion is
        effectively instantaneous on the integrator's time scale
        (<<1 μs at saturated conditions) so we resolve it with a
        very fine dt and show the curve on a log-time axis.
    Right: 'continuous source' scenario — fresh H2SO4 produced at
        a constant rate (stratospheric SO2 + OH analogue, ~1e5
        molec/cm³/s). Shows the quasi-steady-state where nucleation
        consumes incoming gas.
    """
    cfg, r = _make_config()
    gc0, pvapl, T = _strat_state()
    gc0_h2so4 = float(gc0[1])

    # --- Left panel: puff + no source, ultra-fine dt to resolve ---
    schedule_puff = [
        (1e-7, 100),    # 0–10 μs at 0.1-μs dt
        (1e-6, 90),     # 10–100 μs at 1-μs dt
        (1e-5, 90),     # 0.1–1 ms at 10-μs dt
        (1e-4, 90),     # 1–10 ms at 100-μs dt
        (1e-3, 90),     # 10–100 ms at 1-ms dt
    ]
    t_puff, gc_puff, pc_puff = _run_with_source(
        cfg, gc0, pvapl, T, 0.0, schedule_puff,
    )

    # --- Right panel: continuous source, ~stratospheric production ---
    # 1e5 molec/cm³/s ≈ stratospheric SO2+OH flux
    src_molec_per_s = 1e5
    src_g_per_s = src_molec_per_s * _GWTMOL_H2SO4 / float(AVG)
    gc_start = jnp.asarray([gc0[0], 0.0])  # start with no H2SO4
    schedule_source = [
        (0.01, 100),   # 0–1 s
        (0.1, 90),     # 1–10 s
        (1.0, 90),     # 10–100 s
        (10.0, 90),    # 100–1000 s
    ]
    t_src, gc_src, pc_src = _run_with_source(
        cfg, gc_start, pvapl, T, src_g_per_s, schedule_source,
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # LEFT: puff depletion
    ax = axes[0]
    ax2 = ax.twinx()
    l1 = ax.plot(np.maximum(t_puff, 1e-9), gc_puff / gc0_h2so4,
                  "C0-", lw=2, label="H$_2$SO$_4$ / H$_2$SO$_4$(0)")
    l2 = ax2.plot(np.maximum(t_puff, 1e-9), np.maximum(pc_puff, 1e-10),
                   "C3--", lw=2, label="total pc [#/cm$^3$]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax2.set_yscale("log")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("H$_2$SO$_4$ / initial", color="C0")
    ax2.set_ylabel("total particle number [#/cm$^3$]", color="C3")
    lines = l1 + l2
    ax.legend(lines, [l.get_label() for l in lines], loc="center right")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title("Puff: instantaneous depletion at saturated conditions")

    # RIGHT: continuous source
    ax = axes[1]
    ax2 = ax.twinx()
    l1 = ax.plot(np.maximum(t_src, 1e-3), np.maximum(gc_src, 1e-30),
                  "C0-", lw=2, label="H$_2$SO$_4$ [g/cm$^3$]")
    l2 = ax2.plot(np.maximum(t_src, 1e-3), np.maximum(pc_src, 1e-10),
                   "C3--", lw=2, label="total pc [#/cm$^3$]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax2.set_yscale("log")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("H$_2$SO$_4$ [g/cm$^3$]", color="C0")
    ax2.set_ylabel("total particle number [#/cm$^3$]", color="C3")
    lines = l1 + l2
    ax.legend(lines, [l.get_label() for l in lines], loc="center right")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title(r"Continuous source (10$^5$ molec/cm$^3$/s) — quasi-steady")

    fig.suptitle(
        "Sulfate factory evolution at T = 220 K, RH = 50%")
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig1_time_evolution.png", dpi=140)
    plt.close(fig)


def fig_size_distribution_evolution():
    """Snapshots of pc(bin) during the depletion episode and the
    subsequent plateau. Uses the same ms-scale resolution as fig 1."""
    cfg, r = _make_config()
    step = make_step_sulfate(cfg, ntsubsteps=1)
    gc, pvapl, T = _strat_state()

    # Snapshot times (seconds): early (in-depletion), mid, late
    snap_times_s = [0.0, 0.005, 0.05, 1.0, 60.0]
    pc = jnp.zeros(cfg.nbin)
    t_cur = 0.0
    snapshots = {0.0: np.asarray(pc).copy()}

    # Use variable dt: tiny during the fast phase, coarse afterward
    schedule = [
        (0.0001, 50),   # 0–5 ms at 0.1-ms dt
        (0.001, 45),    # 5–50 ms at 1-ms dt
        (0.01, 95),     # 50 ms – 1 s at 10-ms dt
        (0.5, 118),     # 1–60 s at 0.5-s dt
    ]
    for dt, n in schedule:
        for _ in range(n):
            pc, gc, _ = step(pc, gc, T, pvapl, 1.0, dt)
            t_cur += dt
            for st in snap_times_s:
                if st not in snapshots and t_cur >= st:
                    snapshots[st] = np.asarray(pc).copy()

    r_nm = np.asarray(r) * 1e7
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["C0", "C1", "C2", "C3", "C4"]
    for (t_s, color) in zip(snap_times_s, colors):
        y = np.maximum(snapshots.get(t_s, snapshots[0.0]), 1e-10)
        label = f"t = {t_s*1000:.1f} ms" if t_s < 1 else f"t = {t_s:.0f} s"
        ax.plot(r_nm, y, "o-", color=color, lw=1.5, ms=4, label=label)
    ax.set_xlabel("bin radius [nm]")
    ax.set_ylabel("pc [#/cm$^3$]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Size distribution during H2SO4 depletion + plateau")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig2_size_distribution_evolution.png", dpi=140)
    plt.close(fig)


def fig_mass_conservation():
    """Mass-conservation error across the depletion episode.
    Uses the same ms-scale dt schedule as fig 2 so the error is
    measured during active nucleation, not on a post-depletion
    plateau where nothing happens."""
    cfg, r = _make_config()
    step = make_step_sulfate(cfg, ntsubsteps=1)
    gc, pvapl, T = _strat_state()

    rmass = cfg.groups[0].rmass
    pc = jnp.zeros(cfg.nbin).at[5].set(1e3)
    total0 = float(gc[1]) + float(jnp.sum(pc * rmass))

    t_cur = 0.0
    times = [0.0]
    errs = [0.0]

    schedule = [
        (0.0001, 50),
        (0.001, 45),
        (0.01, 95),
        (0.5, 118),
    ]
    for dt, n in schedule:
        for _ in range(n):
            pc, gc, _ = step(pc, gc, T, pvapl, 1.0, dt)
            t_cur += dt
            total = float(gc[1]) + float(jnp.sum(pc * rmass))
            times.append(t_cur)
            errs.append(abs(total - total0) / total0)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(times, errs, "C0-", lw=1.5)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("|mass error| / initial")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(
        "H$_2$SO$_4$ mass conservation across depletion (gas + particle)")
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
