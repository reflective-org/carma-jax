"""Bench JAX freezglaerl_murray2010 against gfortran standalone.

Murray 2010 glassy-aerosol heterogeneous freezing: produces a uniform
per-bin rate (formula is fraction-of-aerosols-nucleated, not radius-
dependent), so the per-bin plot is degenerate. The y=x scatter still
shows JAX vs Fortran agreement across the active parameter regime.

Output:
  data/freezglaerl_murray2010_bench.npz
  plots/diff/phase11/freezglaerl_murray2010_jax_vs_fortran.png
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from carma.nucleation.freezglaerl_murray2010 import freezglaerl_murray2010

DEFAULT_BIN = (ROOT.parent / "original-carma" / "CARMA" / "build_standalone"
               / "freezglaerl_murray2010_standalone")
NBIN = 16


def sample_scenarios(n, rng):
    """T < 212 K (glassy regime); ssi in [0, 1] (parameterization is
    saturated above ssmax = 0.7 but stays valid); ssi_old anywhere
    below ssi to exercise the rising-supersaturation branch."""
    ssi = rng.uniform(0.0, 1.0, n)
    return dict(
        T          = rng.uniform(150.0, 220.0, n),
        ssi        = ssi,
        ssi_old    = ssi * rng.uniform(0.0, 1.0, n),  # ssi_old < ssi
        pconmax    = 10.0 ** rng.uniform(-2.0, 4.0, n),
        dtime      = rng.uniform(1.0, 1800.0, n),
    )


def run_fortran(binary, scen, work_dir):
    in_path = work_dir / "in.txt"
    out_path = work_dir / "out.bin"
    with open(in_path, "w") as f:
        f.write(f"{scen['T']!r} {scen['ssi']!r} {scen['ssi_old']!r} "
                f"{scen['pconmax']!r} {scen['dtime']!r} {NBIN}\n")
    subprocess.run([str(binary), str(in_path), str(out_path)], check=True)
    return np.fromfile(out_path, dtype=np.float64)


def run_jax(scen):
    return np.asarray(freezglaerl_murray2010(
        t_val=jnp.float64(scen['T']),
        supsati_val=jnp.float64(scen['ssi']),
        supsati_old_val=jnp.float64(scen['ssi_old']),
        pconmax_val=jnp.float64(scen['pconmax']),
        dtime=jnp.float64(scen['dtime']),
        nbin=NBIN,
    ))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--binary", type=Path, default=DEFAULT_BIN)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if not args.binary.exists():
        print(f"ERROR: build with build_freezglaerl_murray2010_standalone.sh first",
              file=sys.stderr); sys.exit(1)

    rng = np.random.default_rng(args.seed)
    scens = sample_scenarios(args.n, rng)
    rnuclg_F = np.zeros((args.n, NBIN))
    rnuclg_J = np.zeros((args.n, NBIN))
    print(f"Phase 11.8: {args.n} scenarios through Murray 2010 Fortran + JAX",
          flush=True)
    with tempfile.TemporaryDirectory(prefix="phase118_") as work_root:
        work_root = Path(work_root)
        for i in range(args.n):
            scen = {k: float(v[i]) for k, v in scens.items()}
            rnuclg_F[i] = run_fortran(args.binary, scen, work_root)
            rnuclg_J[i] = run_jax(scen)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{args.n}", flush=True)

    np.savez_compressed(
        ROOT / "data" / "freezglaerl_murray2010_bench.npz",
        rnuclg_F=rnuclg_F, rnuclg_J=rnuclg_J, **scens)

    # Per-bin rate is uniform across bins — collapse to scenario-level.
    rate_F = rnuclg_F[:, 0]
    rate_J = rnuclg_J[:, 0]
    denom = np.maximum(np.abs(rate_F), np.abs(rate_J))
    denom = np.where(denom > 1e-300, denom, 1.0)
    rel = np.abs(rate_F - rate_J) / denom
    nonzero = rate_F != 0
    print(f"\nrel err (per-scenario, since rate is bin-uniform) "
          f"over {args.n} scenarios ({int(nonzero.sum())} nonzero):")
    if nonzero.any():
        nz = rel[nonzero]
        for q in (50, 90, 95, 99, 100):
            print(f"  P{q}: {np.percentile(nz, q):.3e}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    if nonzero.any():
        ax.plot(rate_F[nonzero], rate_J[nonzero], "k.", ms=3, alpha=0.5)
        lo, hi = rate_F[nonzero].min(), rate_F[nonzero].max()
        if lo > 0:
            ax.set_xscale("log"); ax.set_yscale("log")
            xx = np.geomspace(lo, hi, 5)
        else:
            xx = np.linspace(lo, hi, 5)
        ax.plot(xx, xx, "r--", lw=1, label="y = x")
    ax.set_xlabel("Fortran rnuclg [s⁻¹]")
    ax.set_ylabel("JAX rnuclg [s⁻¹]")
    ax.set_title(f"Phase 11.8: Murray 2010 glassy-aerosol freezing — "
                 f"JAX vs Fortran  ({int(nonzero.sum())} of {args.n} active)")
    ax.grid(True, alpha=0.3, which="both"); ax.legend()

    # Activity diagnostic: dfice (gate-dependent fraction nucleated)
    # rather than rate. The rate carries an extra 1 / dtime factor that
    # obscures the physics when dtime is randomized — colouring by dfice
    # isolates the formula's (T, ssi, ssi_old) gate behaviour.
    KICE1 = 7.7211e-5; KICE2 = 9.2688e-3
    SSMIN = 0.21; SSMAX = 0.70; TGLASS = 212.0; FGLASS = 0.5
    ssi_arr  = np.asarray(scens['ssi'])
    sso_arr  = np.asarray(scens['ssi_old'])
    T_arr    = np.asarray(scens['T'])
    pcm_arr  = np.asarray(scens['pconmax'])
    active = ((T_arr <= TGLASS)
              & (pcm_arr > 1e-44)             # FEW_PC
              & (ssi_arr >= SSMIN)
              & (ssi_arr > sso_arr))
    ssi_c = np.minimum(ssi_arr, SSMAX)
    sso_c = np.minimum(sso_arr, SSMAX)
    dfice = KICE1 * (1.0 + ssi_c) * 100.0 - KICE2
    dfice -= np.where(sso_arr >= SSMIN,
                       KICE1 * (1.0 + sso_c) * 100.0 - KICE2, 0.0)
    dfice_active = np.where(active, FGLASS * dfice, 0.0)

    ax2 = axes[1]
    # Inactive scenarios: small gray dots in the background.
    ax2.scatter(T_arr[~active], ssi_arr[~active],
                 c="0.85", s=6, alpha=0.5, edgecolors="none",
                 zorder=1, label=f"gated off ({(~active).sum()})")
    # Active scenarios: coloured by Murray 2010 fraction (dimensionless).
    sc = ax2.scatter(T_arr[active], ssi_arr[active],
                     c=dfice_active[active], cmap="viridis",
                     s=18, edgecolors="black", linewidths=0.3,
                     zorder=3, label=f"active ({active.sum()})")
    ax2.axhline(SSMIN, color='red',    ls='--', lw=1.5, alpha=0.7,
                label=f"ssmin = {SSMIN}")
    ax2.axhline(SSMAX, color='orange', ls='--', lw=1.5, alpha=0.7,
                label=f"ssmax = {SSMAX}")
    ax2.axvline(TGLASS, color='blue',  ls='--', lw=1.5, alpha=0.7,
                label=f"tglass = {TGLASS:.0f} K")
    ax2.set_xlabel("T [K]"); ax2.set_ylabel("ssi")
    ax2.set_title("Gate activity — colour = f_glass · dfice (rate × dtime, "
                  "dtime-independent fraction)")
    plt.colorbar(sc, ax=ax2, label="0.5 · dfice  [dimensionless]")
    ax2.legend(loc="upper right", fontsize=8)

    out_dir = ROOT / "plots" / "diff" / "phase11"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "freezglaerl_murray2010_jax_vs_fortran.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
