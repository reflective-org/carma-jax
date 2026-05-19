"""Validation figure for versub — compare explicit substepping
(versub) against the implicit Thomas solver (versol) on a pulse that
sediments under gravity.

Shows:

1. Pulse propagation traces at several times for versub.
2. Comparison of versub vs versol at a single snapshot; versub is
   more diffusive at high CFL (expected).
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from carma.enums import BoundaryCondition, GridType
from carma.transport.versol import versol
from carma.transport.versub import versub


OUTDIR = Path(__file__).parent.parent / "plots" / "phase8_versub"


def _column(nz=40, dz_cm=1e4):
    dz = jnp.full((nz,), dz_cm)
    cvert = jnp.zeros(nz).at[nz // 2].set(1.0)
    pcmax = cvert
    return dz, cvert, pcmax


def _run(solver, cvert, pcmax, dz, vfall_cm_s, dtime, nz):
    """Generic solver call. ``solver in {'versub', 'versol'}``."""
    vertadvu = jnp.zeros(nz + 1)
    vertadvd = jnp.full((nz + 1,), vfall_cm_s)
    vertdifu = jnp.zeros(nz + 1)
    vertdifd = jnp.zeros(nz + 1)
    common = dict(
        dz=dz, dtime=dtime,
        itbnd=int(BoundaryCondition.I_FIXED_CONC),
        ibbnd=int(BoundaryCondition.I_FIXED_CONC),
        ftop=0.0, fbot=0.0,
        cvert_tbnd=0.0, cvert_bbnd=0.0,
        vertadvu=vertadvu, vertadvd=vertadvd,
        vertdifu=vertdifu, vertdifd=vertdifd,
        igridv=int(GridType.I_CART),
    )
    if solver == "versub":
        return versub(cvert=cvert, pcmax=pcmax, **common)
    else:
        return versol(cvert=cvert, **common)


def fig_pulse_propagation():
    nz = 40
    dz, cvert, _ = _column(nz=nz)
    vfall = 50.0
    # CFL = vfall · dt / dz. For dz=1e4 cm, CFL=0.5 at dt=100 s.

    fig, ax = plt.subplots(figsize=(8, 5))
    cv = cvert
    z = np.arange(nz)
    ax.plot(cv, z, "k--", lw=1.5, label="t = 0")

    for t_hr, color in [(1, "C0"), (3, "C1"), (6, "C2"), (12, "C3")]:
        cv = cvert
        nsteps = int(t_hr * 3600 / 100)
        for _ in range(nsteps):
            cv = _run("versub", cv, cv, dz, vfall, 100.0, nz)
        ax.plot(np.asarray(cv), z, "-", color=color, lw=1.5,
                label=f"t = {t_hr} h")

    ax.set_xlabel("concentration (relative)")
    ax.set_ylabel("layer index (0 = bottom)")
    ax.set_title(
        "versub: pulse sedimentation (v_fall = 50 cm/s, dz = 100 m)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig1_pulse_propagation.png", dpi=140)
    plt.close(fig)


def fig_versus_versol():
    """Same initial pulse under same velocity and dt. versub is
    more diffusive at high CFL — but the overall pulse motion should
    match versol at low CFL.

    Integration time is tuned so the pulse traverses a similar
    distance in both panels (a few layers), highlighting the
    diffusion difference rather than boundary exits."""
    nz = 40
    dz, cvert, _ = _column(nz=nz)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, vfall, dtime, nsteps, title in [
        (axes[0], 10.0, 100.0, 30, "Low CFL (0.1), total 3000 s"),
        (axes[1], 200.0, 100.0, 4, "High CFL (2.0), total 400 s"),
    ]:
        cv_sub = cvert
        cv_sol = cvert
        for _ in range(nsteps):
            cv_sub = _run("versub", cv_sub, cv_sub, dz, vfall, dtime, nz)
            cv_sol = _run("versol", cv_sol, cv_sol, dz, vfall, dtime, nz)

        z = np.arange(nz)
        ax.plot(np.asarray(cvert), z, "k--", lw=1.2, alpha=0.4, label="t = 0")
        ax.plot(np.asarray(cv_sub), z, "C0-", lw=2, label="versub")
        ax.plot(np.asarray(cv_sol), z, "C3--", lw=2, label="versol")
        ax.set_xlabel("concentration")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("layer index")
    fig.suptitle(
        "versub (explicit substepping) vs versol (implicit Thomas)")
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig2_versub_vs_versol.png", dpi=140)
    plt.close(fig)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig_pulse_propagation()
    fig_versus_versol()
    print(f"Figures saved in {OUTDIR}")


if __name__ == "__main__":
    main()
