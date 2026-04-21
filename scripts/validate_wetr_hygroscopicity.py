"""Validation figures for carma.wetr + carma.hygroscopicity (Phase 7.2).

Generates:
1. ``fig1_kohler_growth_factors.png`` — rwet/rdry vs RH for κ values
   spanning pure water (κ=1.2), ammonium sulfate (κ=0.6), organics
   (κ=0.1), and insoluble dust (κ=0). Compared against the analytic
   Petters-Kreidenweis 2007 κ-Köhler form.
2. ``fig2_petters_lowT_rescale.png`` — growth factor at fixed RH = 90%
   as a function of temperature 170-298 K, showing the Yu 2015
   low-T correction smoothly blending at T = 190 K.
3. ``fig3_wtpct_sulfate_rwet.png`` — sulfate I_WTPCT_H2SO4 wet radius
   over the stratosphere-to-troposphere envelope, plotted as
   (rwet/rdry) vs T × RH.
4. ``fig4_hygroscopicity_bulk_kappa.png`` — bulk κ as a function of
   core mass fraction for a two-element (shell + core) bin, at three
   (κ_shell, κ_core) combinations.

Each panel saved to ``plots/phase7_wetr_hygroscopicity/``. All
computations happen in fp64.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from carma.constants import AVG, BK, WTMOL_H2O
from carma.enums import SwellMethod
from carma.hygroscopicity import hygroscopicity
from carma.vapor_pressure import vaporp_h2o_murphy2005
from carma.wetr import get_wetr


OUTDIR = Path(__file__).parent.parent / "plots" / "phase7_wetr_hygroscopicity"


def _analytic_kohler(rdry, rh, kappa):
    """Closed-form PK07 Eq. 6 for verification."""
    return rdry * (1.0 + rh * kappa / (1.0 - rh)) ** (1.0 / 3.0)


def fig_kohler_growth_factors():
    """Growth factor vs RH for a sweep of κ; compare to analytic PK07."""
    rdry = 1e-5  # 0.1 μm
    rhopdry = 1.77
    T = 298.0
    rh_grid = np.linspace(0.1, 0.995, 200)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, 5))
    for kappa, color in zip([0.0, 0.1, 0.3, 0.6, 1.2], colors):
        # JAX (vmap over RH)
        rwet = np.asarray(jax.vmap(lambda rh: get_wetr(
            rdry, rhopdry, rh, T,
            int(SwellMethod.I_PETTERS), kappa=kappa)[0])(jnp.asarray(rh_grid)))
        gf_jax = rwet / rdry
        # Analytic
        gf_ref = _analytic_kohler(rdry, rh_grid, kappa) / rdry
        # Note: κ=0 → growth factor should be 1 exactly
        ax.plot(rh_grid, gf_jax, color=color, lw=2, label=f"κ = {kappa}")
        ax.plot(rh_grid, gf_ref, "--", color=color, lw=1, alpha=0.6)
    ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("RH")
    ax.set_ylabel(r"growth factor $r_{wet}/r_{dry}$")
    ax.set_title("κ-Köhler growth factor — JAX (solid) vs analytic PK07 Eq. 6 (dashed)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Hygroscopicity", fontsize=9)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig1_kohler_growth_factors.png", dpi=140)
    plt.close(fig)


def fig_petters_lowT_rescale():
    """Low-T rescale blending at T = 190 K (Yu 2015)."""
    rdry = 1e-5
    rhopdry = 1.77
    rh = 0.9
    T_grid = np.linspace(170.0, 298.0, 150)

    fig, ax = plt.subplots(figsize=(7, 5))
    for kappa, color in [(0.1, "C0"), (0.3, "C1"), (0.6, "C2")]:
        rwet = np.asarray(jax.vmap(lambda T: get_wetr(
            rdry, rhopdry, rh, T,
            int(SwellMethod.I_PETTERS), kappa=kappa)[0])(jnp.asarray(T_grid)))
        ax.plot(T_grid, rwet / rdry, color=color, lw=2, label=f"κ = {kappa}")
    ax.axvline(190.0, color="gray", ls="--", alpha=0.5)
    ax.text(190.5, ax.get_ylim()[1] * 0.95,
            "T = 190 K\nYu 2015 switchover", fontsize=9, va="top", color="gray")
    ax.set_xlabel("temperature [K]")
    ax.set_ylabel(r"growth factor at RH=90%")
    ax.set_title("Low-T rescale: Yu 2015 blends below 190 K to avoid κ-Köhler pathology")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig2_petters_lowT_rescale.png", dpi=140)
    plt.close(fig)


def fig_wtpct_sulfate_rwet():
    """Binary H2SO4/H2O growth factor across the strat-to-tropo envelope."""
    rdry = 1e-5
    rhopdry_map = {220.0: 1.547, 250.0: 1.447, 280.0: 1.330, 300.0: 1.260}
    rh_grid = np.linspace(0.05, 0.95, 60)

    fig, ax = plt.subplots(figsize=(7, 5))
    for T, color in [(220.0, "C0"), (250.0, "C1"), (280.0, "C2"), (300.0, "C3")]:
        pvapl = float(vaporp_h2o_murphy2005(jnp.array(T))[0])
        rhopdry = rhopdry_map[T]
        rwet = []
        for rh in rh_grid:
            h2o_mass = rh * pvapl * float(WTMOL_H2O) / (float(BK) * T * float(AVG))
            r, _ = get_wetr(rdry, rhopdry, rh, T,
                            int(SwellMethod.I_WTPCT_H2SO4),
                            h2o_mass=h2o_mass, h2o_vp=pvapl)
            rwet.append(float(r))
        ax.plot(rh_grid, np.array(rwet) / rdry, color=color, lw=2, label=f"T = {int(T)} K")
    ax.set_xlabel("RH")
    ax.set_ylabel(r"growth factor $r_{wet}/r_{dry}$")
    ax.set_title("Binary H2SO4/H2O wet radius — I_WTPCT_H2SO4 with Kelvin correction")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig3_wtpct_sulfate_rwet.png", dpi=140)
    plt.close(fig)


def fig_hygroscopicity_bulk_kappa():
    """Bulk κ vs core mass fraction for three (κ_shell, κ_core) combos."""
    # Synthetic scenario: 1 level, 1 bin, 2 elements (shell=0, core=1).
    NZ, NBIN, NELEM, NGROUP = 1, 1, 2, 1
    rmass_2d = jnp.array([[1e-15]])  # bin mass 1e-15 g
    ienconc = jnp.array([0])
    igelem = jnp.array([0, 0])
    icorelem = jnp.array([[1]])  # core is element 1
    ncore = jnp.array([1])

    core_frac = np.linspace(0.0, 1.0, 100)

    fig, ax = plt.subplots(figsize=(7, 5))
    for k_sh, k_co, label, color in [
        (0.6, 0.1, r"shell NH$_4$SO$_4$ ($\kappa$=0.6), core BC ($\kappa$=0.1)", "C0"),
        (0.9, 0.0, r"shell H$_2$SO$_4$ ($\kappa$=0.9), core dust ($\kappa$=0)", "C1"),
        (0.3, 0.5, r"shell organic ($\kappa$=0.3), core salt ($\kappa$=0.5)", "C2"),
    ]:
        kappa_elem = jnp.array([k_sh, k_co])
        kappa_bulk = []
        for f in core_frac:
            total_mass = 1e-15
            core_mass = f * total_mass
            # pc_num such that pc_num * rmass = total_mass
            pc_num = total_mass / 1e-15
            pc_core = core_mass
            pc_shell_dummy_ielem = pc_num  # element 0 is number
            pc = jnp.array([[[pc_num, pc_core]]])
            k = hygroscopicity(pc, rmass_2d, kappa_elem,
                               ienconc, igelem, icorelem, ncore)
            kappa_bulk.append(float(k[0, 0, 0]))
        ax.plot(core_frac, kappa_bulk, color=color, lw=2, label=label)
    ax.set_xlabel("core mass fraction")
    ax.set_ylabel(r"bulk $\kappa$")
    ax.set_title("Mass-weighted bulk κ as core mass fraction varies")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig4_hygroscopicity_bulk_kappa.png", dpi=140)
    plt.close(fig)


def print_reference_points():
    """Print a few canonical points to stdout."""
    rdry = 1e-5
    rhopdry = 1.77
    print("\nReference points (rwet / rdry = growth factor):")
    print(f"{'scenario':<40} {'T':>6} {'RH':>5} {'κ':>5} {'GF':>7}")
    points = [
        ("pure water (κ=1.2), warm troposphere", 298.0, 0.90, 1.2),
        ("ammonium sulfate (κ=0.6), 90% RH", 298.0, 0.90, 0.6),
        ("organic (κ=0.1), 90% RH", 298.0, 0.90, 0.1),
        ("insoluble dust (κ=0)", 298.0, 0.90, 0.0),
        ("upper strat sulfate κ≈0.7 at 220 K / 40% RH", 220.0, 0.40, 0.7),
    ]
    for name, T, RH, kappa in points:
        rwet, rhop = get_wetr(rdry, rhopdry, RH, T,
                               int(SwellMethod.I_PETTERS), kappa=kappa)
        print(f"{name:<40} {T:6.0f} {RH:5.2f} {kappa:5.2f} {float(rwet)/rdry:7.3f}")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig_kohler_growth_factors()
    fig_petters_lowT_rescale()
    fig_wtpct_sulfate_rwet()
    fig_hygroscopicity_bulk_kappa()
    print_reference_points()
    print(f"\nFigures saved in {OUTDIR}")


if __name__ == "__main__":
    main()
