"""Validation figure for evap_mono + evap_poly (Phase 8.2).

Shows how a single total-evaporation event scatters into the target
CN grid under the three ``evap_mono`` regimes (too_small, interior,
too_big) and the polydisperse distribution for three ``coresig``
values. All scatters are zero-sum at the number level: every bar
sums to ``evdrop = 1``.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from carma.growth.evap_mono import evap_mono
from carma.growth.evap_poly import evap_poly


OUTDIR = Path(__file__).parent.parent / "plots" / "phase8_evap_mono_poly"


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    nbin = 12
    rho = 1.8
    vmin = (4.0 / 3.0) * np.pi * (1e-7) ** 3 * rho
    rmrat = 2.0
    rmass_1d = vmin * rmrat ** np.arange(nbin)
    dm_1d = rmass_1d * (rmrat - 1.0) / 2.0
    rmass = jnp.asarray(rmass_1d[:, None])
    dm = jnp.asarray(dm_1d[:, None])

    tgt = rmass[:, :, None, None]
    src = rmass[None, None, :, :]
    diffmass = tgt - src

    icorelem = np.asarray([[0], [1]])
    ievp2elem = np.asarray([0, 1])
    bin_idx = np.arange(nbin)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ------ evap_mono panel ------
    ax = axes[0]
    evdrop = 1.0
    evcore = jnp.asarray([0.0, 1.0])

    # Three regimes. Pick coreavg values so factor = coreavg/rmass[jbin]
    # is close to 1 in each boundary case; that way the three panels
    # are visually comparable. For an oversized/undersized coreavg the
    # factor would scale particle number by that factor (Fortran
    # conserves core mass, not number, at the grid boundaries).
    mono_cases = [
        ("too_small (coreavg = 0.7·rmass[0])", 0,
         dict(too_small=True, too_big=False, nuc_small=False),
         0.7 * float(rmass[0, 0])),
        ("interior (iavg=6)", 6,
         dict(too_small=False, too_big=False, nuc_small=False),
         0.5 * (float(rmass[5, 0]) + float(rmass[6, 0]))),
        ("too_big (coreavg = 1.3·rmass[nbin-1])", nbin - 1,
         dict(too_small=False, too_big=True, nuc_small=False),
         1.3 * float(rmass[-1, 0])),
    ]
    width = 0.28
    totals = []
    for k, (label, iavg, flags, coreavg) in enumerate(mono_cases):
        delta = evap_mono(
            evdrop=evdrop, evcore=evcore, coreavg=coreavg, iavg=iavg,
            ieto=0, igto=0, **flags,
            ncore_ig=2, icorelem=icorelem, ievp2elem=ievp2elem, ig=0,
            rmass=rmass, diffmass=diffmass, nbin=nbin, nelem=2,
        )
        ax.bar(bin_idx + (k - 1) * width, np.asarray(delta[:, 0]),
               width=width, label=label)
        totals.append(float(jnp.sum(delta[:, 0])))
    ax.set_xlabel("target bin")
    ax.set_ylabel("number scatter [per evdrop]")
    ax.set_title(
        f"evap_mono — Σ number = [{totals[0]:.2f}, {totals[1]:.2f}, {totals[2]:.2f}]"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    # ------ evap_poly panel ------
    ax = axes[1]
    iavg = 6
    coreavg = float(rmass[iavg, 0])
    for k, (coresig, color) in enumerate(zip([0.5, 1.0, 2.0],
                                              ["C0", "C1", "C2"])):
        delta = evap_poly(
            evdrop=evdrop, evcore=evcore,
            coreavg=coreavg, coresig=coresig,
            iavg=iavg, ieto=0, igto=0,
            ncore_ig=2, icorelem=icorelem, ievp2elem=ievp2elem, ig=0,
            rmass=rmass, dm=dm, nbin=nbin, nelem=2,
        )
        ax.bar(bin_idx + (k - 1) * width, np.asarray(delta[:, 0]),
               width=width, color=color, label=f"coresig = {coresig}")
    ax.axvline(iavg - 0.5, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("target bin")
    ax.set_ylabel("number scatter")
    ax.set_title("evap_poly: log-normal spread (iavg = 6)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.suptitle(
        "Phase 8.2: total-evaporation scatters into CN target grid")
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig1_mono_poly_scatter.png", dpi=140)
    plt.close(fig)
    print(f"Figures saved in {OUTDIR}")


if __name__ == "__main__":
    main()
