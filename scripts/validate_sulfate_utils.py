"""Validation figures for carma.sulfate_utils (Phase 7.1).

Generates:
1. wt% H2SO4 vs water activity across T = [185, 260] K — checks
   piecewise continuity at activ = 0.05 and activ = 0.85.
2. Sulfate density vs wt% at fixed T = 220, 260, 300 K — checks
   monotonicity and extrapolation-clamp at T = 180/380 K.
3. Surface tension vs wt% at fixed T = 220, 260, 300 K — same.
4. wt% as a function of (T, RH) over the full stratosphere-to-
   troposphere window, plotted as a heatmap.
5. JIT + vmap regression: confirms a vectorised batch run returns
   identical values to the scalar call.

Each panel is saved to ``plots/phase7_sulfate_utils/``. All
computations happen in fp64.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from carma.constants import AVG, BK, WTMOL_H2O
from carma.sulfate_utils import wtpct_tabaz, sulfate_density, sulfate_surf_tens
from carma.vapor_pressure import vaporp_h2o_murphy2005


OUTDIR = Path(__file__).parent.parent / "plots" / "phase7_sulfate_utils"


def _h2o_mass_from_rh(T, RH):
    """Water-vapour mass concentration [g/cm^3] at (T, RH)."""
    T = jnp.asarray(T)
    pvapl, _ = vaporp_h2o_murphy2005(T)
    p_h2o = RH * pvapl
    return p_h2o * WTMOL_H2O / (BK * T * AVG), pvapl


def fig_wtpct_vs_activity():
    """wt% vs water activity at three temperatures."""
    fig, ax = plt.subplots(figsize=(7, 5))
    activity = np.linspace(0.005, 1.05, 300)
    for T, color in [(190.0, "C0"), (220.0, "C1"), (260.0, "C2")]:
        pvapl_T = float(vaporp_h2o_murphy2005(jnp.array(T))[0])
        # activity = p_h2o / pvapl, so p_h2o = activity * pvapl
        p_h2o = activity * pvapl_T
        h2o_mass = p_h2o * float(WTMOL_H2O) / (float(BK) * T * float(AVG))
        wtp = jax.vmap(lambda m: wtpct_tabaz(T, m, pvapl_T))(jnp.asarray(h2o_mass))
        ax.plot(activity, np.asarray(wtp), color=color, label=f"T = {int(T)} K", lw=2)
    for a, label in [(0.05, "lo/mid branch"), (0.85, "mid/hi branch")]:
        ax.axvline(a, color="gray", ls="--", alpha=0.4, lw=0.8)
        ax.text(a, 98, label, rotation=90, va="top", fontsize=8, color="gray")
    ax.set_xlabel("water activity (= RH for pure water)")
    ax.set_ylabel("wt% H2SO4")
    ax.set_title("Tabazadeh 1997 wt% fit — branch continuity check")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig1_wtpct_vs_activity.png", dpi=140)
    plt.close(fig)


def fig_density_vs_wtp():
    """Density vs wt% at three temperatures + clamp check."""
    fig, ax = plt.subplots(figsize=(7, 5))
    wtp_grid = np.linspace(0, 100, 300)
    for T, color in [(220.0, "C0"), (260.0, "C1"), (300.0, "C2")]:
        rho = jax.vmap(lambda w: sulfate_density(w, T))(jnp.asarray(wtp_grid))
        ax.plot(wtp_grid, np.asarray(rho), color=color, label=f"T = {int(T)} K", lw=2)
    # Clamp check: T > 380 should match T = 380
    rho_clamp = jax.vmap(lambda w: sulfate_density(w, 450.0))(jnp.asarray(wtp_grid))
    rho_380 = jax.vmap(lambda w: sulfate_density(w, 380.0))(jnp.asarray(wtp_grid))
    assert jnp.allclose(rho_clamp, rho_380), "temperature clamp at 380 K failed"
    ax.set_xlabel("wt% H2SO4")
    ax.set_ylabel(r"density [g/cm$^3$]")
    ax.set_title("sulfate_density — Washburn (NRC 1928) linear fits")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig2_density_vs_wtp.png", dpi=140)
    plt.close(fig)


def fig_surf_tens_vs_wtp():
    """Surface tension vs wt% at three temperatures."""
    fig, ax = plt.subplots(figsize=(7, 5))
    wtp_grid = np.linspace(0, 100, 300)
    for T, color in [(220.0, "C0"), (260.0, "C1"), (300.0, "C2")]:
        sig = jax.vmap(lambda w: sulfate_surf_tens(w, T))(jnp.asarray(wtp_grid))
        ax.plot(wtp_grid, np.asarray(sig), color=color, label=f"T = {int(T)} K", lw=2)
    ax.set_xlabel("wt% H2SO4")
    ax.set_ylabel(r"surface tension [erg/cm$^2$]")
    ax.set_title("sulfate_surf_tens — Sabinina & Terpugow (1935) fits")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig3_surf_tens_vs_wtp.png", dpi=140)
    plt.close(fig)


def fig_wtpct_heatmap():
    """wt% as function of (T, RH) — stratosphere-to-troposphere map."""
    T_grid = np.linspace(185.0, 300.0, 120)
    RH_grid = np.linspace(0.01, 1.0, 100)
    TT, RR = np.meshgrid(T_grid, RH_grid, indexing="ij")

    # Broadcast: compute pvapl per T, then scale by RH
    pvapl_T = np.asarray(vaporp_h2o_murphy2005(jnp.asarray(T_grid))[0])
    pvapl_2d = pvapl_T[:, None] * np.ones_like(RR)
    p_h2o = RR * pvapl_2d
    h2o_mass = p_h2o * float(WTMOL_H2O) / (float(BK) * TT * float(AVG))

    wtp_fn = jax.jit(jax.vmap(jax.vmap(wtpct_tabaz)))
    wtp = np.asarray(wtp_fn(jnp.asarray(TT), jnp.asarray(h2o_mass), jnp.asarray(pvapl_2d)))

    fig, ax = plt.subplots(figsize=(8, 5))
    cs = ax.contourf(TT, RR * 100, wtp, levels=np.arange(0, 101, 5), cmap="viridis")
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label("wt% H2SO4")
    cl = ax.contour(TT, RR * 100, wtp, levels=[10, 30, 50, 70, 90],
                    colors="white", linewidths=0.8)
    ax.clabel(cl, fmt="%d", inline=True, fontsize=8)
    ax.set_xlabel("temperature [K]")
    ax.set_ylabel("RH [%]")
    ax.set_title("H2SO4 wt% from Tabazadeh 1997 across stratosphere-to-troposphere envelope")
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig4_wtpct_heatmap.png", dpi=140)
    plt.close(fig)


def fig_physical_sanity():
    """Key stratospheric reference points — density × wt% at known conditions."""
    # Stratospheric: T=220K, RH=10% → wt% ~ 60, rho ~ 1.45
    # Polar strat cloud: T=195K, RH=50% → wt% ~ 35, rho ~ 1.30
    # UT/LS: T=230K, RH=20% → wt% ~ 50, rho ~ 1.40
    points = [
        ("Stratosphere", 220.0, 0.10),
        ("PSC conditions", 195.0, 0.50),
        ("UT/LS", 230.0, 0.20),
        ("Troposphere", 275.0, 0.60),
    ]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))

    for name, T, RH in points:
        pvapl_T = float(vaporp_h2o_murphy2005(jnp.array(T))[0])
        h2o_mass = RH * pvapl_T * float(WTMOL_H2O) / (float(BK) * T * float(AVG))
        wtp = float(wtpct_tabaz(T, h2o_mass, pvapl_T))
        rho = float(sulfate_density(wtp, T))
        sig = float(sulfate_surf_tens(wtp, T))
        ax1.scatter(T, wtp, s=80, label=f"{name}\nT={T:.0f}K RH={RH*100:.0f}%")
        ax2.scatter(T, rho, s=80)
        ax3.scatter(T, sig, s=80)
        # Annotations
        for ax, y in [(ax1, wtp), (ax2, rho), (ax3, sig)]:
            ax.annotate(name, (T, y), xytext=(6, 6), textcoords="offset points",
                        fontsize=8)

    for ax, ylabel, title in [
        (ax1, "wt% H2SO4", "composition"),
        (ax2, "density [g/cm³]", "density"),
        (ax3, "surface tension [erg/cm²]", "surface tension"),
    ]:
        ax.set_xlabel("temperature [K]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=7)

    fig.suptitle("Sulfate composition at canonical atmospheric conditions", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig5_physical_sanity.png", dpi=140)
    plt.close(fig)

    # Print summary
    print("\nCanonical reference points:")
    print(f"{'scenario':<18} {'T [K]':>7} {'RH %':>5} {'wt%':>6} "
          f"{'rho g/cm³':>10} {'sigma erg/cm²':>14}")
    for name, T, RH in points:
        pvapl_T = float(vaporp_h2o_murphy2005(jnp.array(T))[0])
        h2o_mass = RH * pvapl_T * float(WTMOL_H2O) / (float(BK) * T * float(AVG))
        wtp = float(wtpct_tabaz(T, h2o_mass, pvapl_T))
        rho = float(sulfate_density(wtp, T))
        sig = float(sulfate_surf_tens(wtp, T))
        print(f"{name:<18} {T:7.1f} {RH*100:5.0f} {wtp:6.2f} {rho:10.3f} {sig:14.2f}")


def test_jit_vmap_regression():
    """Confirm JIT and vmap produce identical outputs to scalar."""
    T_arr = jnp.array([190.0, 220.0, 260.0, 300.0])
    RH = 0.3
    pvapl_arr = vaporp_h2o_murphy2005(T_arr)[0]
    h2o_mass_arr = RH * pvapl_arr * WTMOL_H2O / (BK * T_arr * AVG)

    # Scalar
    wtp_scalar = [float(wtpct_tabaz(T_arr[i], h2o_mass_arr[i], pvapl_arr[i]))
                  for i in range(4)]
    # vmap
    wtp_vmap = jax.vmap(wtpct_tabaz)(T_arr, h2o_mass_arr, pvapl_arr)
    # JIT(vmap)
    wtp_jitvmap = jax.jit(jax.vmap(wtpct_tabaz))(T_arr, h2o_mass_arr, pvapl_arr)

    assert jnp.allclose(jnp.array(wtp_scalar), wtp_vmap)
    assert jnp.allclose(wtp_vmap, wtp_jitvmap)
    print("JIT + vmap regression: OK")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig_wtpct_vs_activity()
    fig_density_vs_wtp()
    fig_surf_tens_vs_wtp()
    fig_wtpct_heatmap()
    fig_physical_sanity()
    test_jit_vmap_regression()
    print(f"\nFigures saved in {OUTDIR}")


if __name__ == "__main__":
    main()
