"""Validation figure for fixcorecol (Phase 8.1).

Sets up a synthetic column with alternating surplus / deficit of
concentration-element mass (the kind of imbalance PPM advection
creates) and shows the before/after pc profile alongside the
column-total conservation proof.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from carma.utils.fixcorecol import fixcorecol


OUTDIR = Path(__file__).parent.parent / "plots" / "phase8_fixcorecol"


def _build_column(nz=30):
    """Lognormal 'core' profile plus a sinusoidal 'advective' imbalance
    that drives concgas_md negative in several layers."""
    z = np.linspace(0, 10, nz)
    core = 5e-20 * np.exp(-((z - 5) ** 2) / 6.0)          # core mass per particle
    # Number density: physically would be core/rmass + headroom, but
    # the imbalance knocks it below core/rmass in some layers.
    rmass = 1e-20
    num_ideal = core / rmass + 2.0
    # Sinusoidal advective wiggle (what PPM might do)
    wiggle = 4.0 * np.sin(2 * np.pi * z / 4)
    num = np.clip(num_ideal + wiggle, 0.1, None)

    # Assemble into (nz, nbin=1, nelem=2): elem 0 = number, elem 1 = core
    pc = np.zeros((nz, 1, 2))
    pc[:, 0, 0] = num
    pc[:, 0, 1] = core
    return z, jnp.asarray(pc), rmass


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    z, pc_before, rmass = _build_column()
    nz = z.size

    rmass_2d = jnp.asarray([[rmass]])
    dz = jnp.ones(nz)
    icorelem = np.asarray([[1]])
    ienconc = np.asarray([0])
    ncore = np.asarray([1])

    pc_after = fixcorecol(pc_before, rmass_2d, dz, icorelem, ienconc, ncore)

    # Diagnostics
    num_before = np.asarray(pc_before[:, 0, 0])
    num_after = np.asarray(pc_after[:, 0, 0])
    core = np.asarray(pc_before[:, 0, 1])
    concgas_before = num_before * rmass - core
    concgas_after = num_after * rmass - core

    total_before = float(np.sum(num_before * rmass))
    total_after = float(np.sum(num_after * rmass))
    rel_err = abs(total_after - total_before) / total_before

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.plot(num_before, z, "C0-o", lw=1.5, ms=3, label="before")
    ax.plot(num_after, z, "C3-o", lw=1.5, ms=3, label="after fix")
    ax.plot(core / rmass, z, "k--", lw=1, alpha=0.6,
            label="core / rmass (floor)")
    ax.set_xlabel("pc[number]")
    ax.set_ylabel("z [arb.]")
    ax.set_title("Number concentration: before and after fix")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.axvline(0, color="k", lw=0.8, alpha=0.5)
    ax.plot(concgas_before, z, "C0-o", lw=1.5, ms=3, label="before")
    ax.plot(concgas_after, z, "C3-o", lw=1.5, ms=3, label="after fix")
    ax.set_xlabel("concgas_md = pc·rmass − core")
    ax.set_ylabel("z [arb.]")
    ax.set_title("Free mass per level (must be ≥ 0 after fix)")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.suptitle(
        f"fixcorecol: column total preserved, rel err = {rel_err:.2e}")
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig1_fixcorecol_profile.png", dpi=140)
    plt.close(fig)
    print(f"Figures saved in {OUTDIR}")
    print(f"column total rel err = {rel_err:.3e}")


if __name__ == "__main__":
    main()
