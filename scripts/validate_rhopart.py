"""Validation figures for carma.rhopart (Phase 7.3).

Generates:
1. ``fig1_rhop_vs_core_fraction.png`` — rhop vs core mass fraction for
   three shell/core combinations (sulfate/BC, sulfate/dust, ice/dust).
   Compared against analytic volume-mixing formula.
2. ``fig2_rhop_three_elements.png`` — bulk density as a function of
   two independent core-mass fractions (shell + 2 cores). Heatmap.
3. ``fig3_core_exceeds_total_repair.png`` — timeline of the "m_core
   > m_total" safety repair: plot pc_num before/after vs synthetic
   core-mass inflation factor.

All plots saved to ``plots/phase7_rhopart/``.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from carma.rhopart import rhopart


OUTDIR = Path(__file__).parent.parent / "plots" / "phase7_rhopart"


def _analytic_two_element(core_frac, rho_shell, rho_core):
    """Closed-form volume-mixing density for shell + one core."""
    # m_total = 1 (scaled). m_core = f. m_shell = 1 - f.
    v = (1 - core_frac) / rho_shell + core_frac / rho_core
    return 1.0 / v


def fig_rhop_vs_core_fraction():
    """Shell + one core — density vs core mass fraction."""
    NZ, NBIN, NELEM, NGROUP = 1, 1, 2, 1
    rmass_val = 1e-15
    rmass = jnp.array([[rmass_val]])
    ienconc = jnp.array([0])
    igelem = jnp.array([0, 0])
    icorelem = jnp.array([[1]])
    ncore = jnp.array([1])

    core_frac = np.linspace(0.0, 1.0, 100)

    fig, ax = plt.subplots(figsize=(7, 5))
    for rho_shell, rho_core, label, color in [
        (1.8, 1.0, r"shell H$_2$SO$_4$ ($\rho$=1.8), core BC ($\rho$=1.0)", "C0"),
        (1.8, 2.6, r"shell H$_2$SO$_4$ ($\rho$=1.8), core dust ($\rho$=2.6)", "C1"),
        (0.92, 2.6, r"shell ice ($\rho$=0.92), core dust ($\rho$=2.6)", "C2"),
    ]:
        rhoelem = jnp.array([[rho_shell, rho_core]])
        rhop_jax = []
        for f in core_frac:
            pc_num = 1.0
            m_total = pc_num * rmass_val
            pc = jnp.array([[[pc_num, f * m_total]]])
            r, _ = rhopart(pc, rmass, rhoelem, ienconc, igelem, icorelem, ncore)
            rhop_jax.append(float(r[0, 0, 0]))
        ax.plot(core_frac, rhop_jax, color=color, lw=2, label=label)
        # Analytic
        rhop_ref = _analytic_two_element(core_frac, rho_shell, rho_core)
        ax.plot(core_frac, rhop_ref, "--", color=color, lw=1, alpha=0.5)
    ax.set_xlabel("core mass fraction")
    ax.set_ylabel(r"bulk density $\rho_{bulk}$ [g/cm$^3$]")
    ax.set_title("rhopart — shell + core volume mixing (JAX solid, analytic dashed)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig1_rhop_vs_core_fraction.png", dpi=140)
    plt.close(fig)


def fig_rhop_three_elements():
    """Shell + 2 cores — heatmap of density vs (core1_frac, core2_frac)."""
    NZ, NBIN, NELEM, NGROUP = 1, 1, 3, 1
    rmass_val = 1e-15
    rmass = jnp.array([[rmass_val]])
    rhoelem = jnp.array([[1.8, 1.0, 2.6]])  # shell, BC-like, dust-like
    ienconc = jnp.array([0])
    igelem = jnp.array([0, 0, 0])
    icorelem = jnp.array([[1], [2]])
    ncore = jnp.array([2])

    N = 50
    f1 = np.linspace(0.0, 0.8, N)  # core 1 mass fraction
    f2 = np.linspace(0.0, 0.8, N)  # core 2 mass fraction
    F1, F2 = np.meshgrid(f1, f2, indexing="ij")

    # Keep f1+f2 ≤ 1 (rest is shell mass)
    rhop_grid = np.full_like(F1, np.nan)
    for i in range(N):
        for j in range(N):
            if F1[i, j] + F2[i, j] > 1.0:
                continue
            pc_num = 1.0
            m_total = pc_num * rmass_val
            pc = jnp.array([[[pc_num, F1[i, j] * m_total, F2[i, j] * m_total]]])
            r, _ = rhopart(pc, rmass, rhoelem, ienconc, igelem, icorelem, ncore)
            rhop_grid[i, j] = float(r[0, 0, 0])

    fig, ax = plt.subplots(figsize=(8, 6))
    cs = ax.contourf(F1, F2, rhop_grid, levels=30, cmap="viridis")
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label(r"bulk $\rho$ [g/cm$^3$]")
    cl = ax.contour(F1, F2, rhop_grid, levels=[1.2, 1.4, 1.6, 1.8, 2.0, 2.2],
                    colors="white", linewidths=0.7)
    ax.clabel(cl, fmt="%.1f", inline=True, fontsize=8)
    ax.plot([0, 1], [1, 0], "k--", lw=0.7)
    ax.text(0.5, 0.52, "f₁+f₂=1 (no shell)", rotation=-45,
            ha="center", fontsize=8)
    ax.set_xlabel(r"core 1 mass fraction (BC, $\rho$=1.0)")
    ax.set_ylabel(r"core 2 mass fraction (dust, $\rho$=2.6)")
    ax.set_title("rhopart — 3-element bin (shell H$_2$SO$_4$ + 2 cores)")
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig2_rhop_three_elements.png", dpi=140)
    plt.close(fig)


def fig_core_exceeds_total_repair():
    """Show the numerical-diffusion safety: as synthetic core-mass inflation
    pushes m_core past m_total, rhop saturates at the pure-core density and
    pc_num is repaired."""
    NZ, NBIN, NELEM, NGROUP = 1, 1, 2, 1
    rmass_val = 1e-15
    rmass = jnp.array([[rmass_val]])
    rhoelem = jnp.array([[1.8, 1.0]])
    ienconc = jnp.array([0])
    igelem = jnp.array([0, 0])
    icorelem = jnp.array([[1]])
    ncore = jnp.array([1])

    # pc_num = 1, m_total = 1e-15. Vary core mass from 0 to 3e-15 (3x total).
    inflation = np.linspace(0.0, 3.0, 200)   # core_mass / m_total
    rhop_list, pc_before, pc_after = [], [], []
    for infl in inflation:
        pc_num = 1.0
        m_total = pc_num * rmass_val
        m_core = infl * m_total
        pc = jnp.array([[[pc_num, m_core]]])
        r, pc_new = rhopart(pc, rmass, rhoelem, ienconc, igelem, icorelem, ncore)
        rhop_list.append(float(r[0, 0, 0]))
        pc_before.append(pc_num)
        pc_after.append(float(pc_new[0, 0, 0]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(inflation, rhop_list, "b-", lw=2)
    ax1.axvline(1.0, color="red", ls="--", alpha=0.6)
    ax1.text(1.02, 1.3, "m_core = m_total\n(safety kicks in)", color="red",
             fontsize=9)
    ax1.set_ylabel(r"bulk $\rho$ [g/cm$^3$]")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("rhopart safety clamp under core-mass inflation")

    ax2.plot(inflation, pc_before, "k--", lw=1, label="pc_num (before)")
    ax2.plot(inflation, pc_after, "g-", lw=2, label="pc_num (after repair)")
    ax2.axvline(1.0, color="red", ls="--", alpha=0.6)
    ax2.set_xlabel("core mass / total mass")
    ax2.set_ylabel("pc_num")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(OUTDIR / "fig3_core_exceeds_total_repair.png", dpi=140)
    plt.close(fig)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig_rhop_vs_core_fraction()
    fig_rhop_three_elements()
    fig_core_exceeds_total_repair()
    print(f"Figures saved in {OUTDIR}")


if __name__ == "__main__":
    main()
