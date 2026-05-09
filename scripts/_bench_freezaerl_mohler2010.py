"""Bench JAX freezaerl_mohler2010 against the standalone Fortran kernel.

Generates random scenarios spanning the parameter regime where the
Möhler 2010 dust ice nucleation rate is non-trivial, calls both the
JAX port and the compiled Fortran driver, and produces:

  data/freezaerl_mohler2010_bench.npz       — raw inputs / outputs
  plots/diff/phase11/freezaerl_mohler2010_jax_vs_fortran.png
                                             — 1:1 scatter + per-bin
                                               rel-err CDF / box plot
"""
import argparse
import math
import shutil
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

from carma.nucleation.freezaerl_mohler2010 import freezaerl_mohler2010

DEFAULT_BIN = (ROOT.parent / "original-carma" / "CARMA" / "build_standalone"
               / "freezaerl_mohler2010_standalone")


def bin_grid(nbin=16, rmin_cm=1e-7, rmrat=4.0, rho=1.78):
    """carma_nuc2test 'Sulfate IN' geometry: rmin = 0.1 µm, rmrat = 4."""
    vmin = (4.0 / 3.0) * math.pi * rmin_cm**3 * rho
    rmass = vmin * rmrat ** np.arange(nbin)
    vol = rmass / rho
    r = (3.0 * rmass / (4.0 * math.pi * rho)) ** (1.0 / 3.0)
    return r, vol


def sample_scenarios(n, rng):
    """Random sample over the parameter regime where the kernel is active."""
    return dict(
        T          = rng.uniform(180.0,  240.0,  n),
        ssi        = rng.uniform(0.30,    1.00,  n),
        ssl        = rng.uniform(-0.99,   0.05,  n),
        akelvin    = rng.uniform(1.0e-7,  3.0e-7, n),
        akelvini   = rng.uniform(1.0e-7,  3.0e-7, n),
        rhosol     = rng.uniform(1.50,    2.00,  n),
        pconmax    = 10.0 ** rng.uniform(-2.0,    4.0,  n),  # log-uniform 1e-2..1e4
    )


def run_fortran(binary, scen, r_bins, vol_bins, work_dir):
    """Invoke the standalone Fortran binary on one scenario, return rnuclg."""
    nbin = r_bins.shape[0]
    in_path = work_dir / "in.txt"
    out_path = work_dir / "out.bin"
    with open(in_path, "w") as f:
        f.write(f"{scen['T']!r}  {scen['ssi']!r}  {scen['ssl']!r}  "
                f"{scen['akelvin']!r}  {scen['akelvini']!r}  "
                f"{scen['rhosol']!r}  {scen['pconmax']!r}  {nbin}\n")
        f.write(" ".join(repr(float(x)) for x in r_bins) + "\n")
        f.write(" ".join(repr(float(x)) for x in vol_bins) + "\n")
    subprocess.run([str(binary), str(in_path), str(out_path)], check=True)
    return np.fromfile(out_path, dtype=np.float64)


def run_jax(scen, r_bins, vol_bins):
    return np.asarray(freezaerl_mohler2010(
        t_val=jnp.float64(scen['T']),
        supsati_val=jnp.float64(scen['ssi']),
        supsatl_val=jnp.float64(scen['ssl']),
        akelvin_val=jnp.float64(scen['akelvin']),
        akelvini_val=jnp.float64(scen['akelvini']),
        r_bins=jnp.asarray(r_bins),
        vol_bins=jnp.asarray(vol_bins),
        rhosol_val=jnp.float64(scen['rhosol']),
        pconmax_val=jnp.float64(scen['pconmax']),
    ))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=1000, help="number of scenarios")
    p.add_argument("--binary", type=Path, default=DEFAULT_BIN)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.binary.exists():
        print(f"ERROR: Fortran binary not found at {args.binary}\n"
              f"Build with scripts/fortran_patch/build_freezaerl_mohler2010_standalone.sh",
              file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    r_bins, vol_bins = bin_grid()
    nbin = r_bins.shape[0]
    scens = sample_scenarios(args.n, rng)

    rnuclg_F = np.zeros((args.n, nbin))
    rnuclg_J = np.zeros((args.n, nbin))
    print(f"Running {args.n} scenarios through Fortran + JAX...", flush=True)
    with tempfile.TemporaryDirectory(prefix="phase11_") as work:
        work = Path(work)
        for i in range(args.n):
            scen = {k: float(v[i]) for k, v in scens.items()}
            rnuclg_F[i] = run_fortran(args.binary, scen, r_bins, vol_bins, work)
            rnuclg_J[i] = run_jax(scen, r_bins, vol_bins)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{args.n}", flush=True)

    # Save raw outputs
    np.savez_compressed(
        ROOT / "data" / "freezaerl_mohler2010_bench.npz",
        rnuclg_F=rnuclg_F, rnuclg_J=rnuclg_J,
        r_bins=r_bins, vol_bins=vol_bins, **scens)

    # Stats
    denom = np.maximum(np.abs(rnuclg_F), np.abs(rnuclg_J))
    denom = np.where(denom > 1e-300, denom, 1.0)
    rel = np.abs(rnuclg_F - rnuclg_J) / denom
    nonzero = rnuclg_F > 0  # restrict stats to bins where the kernel actually fires
    print(f"\nrel err over {args.n} scenarios × {nbin} bins"
          f" ({nonzero.sum()} non-zero entries):")
    if nonzero.any():
        nz = rel[nonzero]
        for q in (50, 90, 95, 99, 100):
            print(f"  P{q}: {np.percentile(nz, q):.3e}")

    # Plot: 1:1 scatter + per-bin rel-err box
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    F_flat = rnuclg_F[nonzero]
    J_flat = rnuclg_J[nonzero]
    ax.loglog(F_flat, J_flat, "k.", ms=2, alpha=0.4)
    if F_flat.size:
        lo = max(F_flat.min(), 1e-30)
        hi = F_flat.max()
        xx = np.geomspace(lo, hi, 5)
        ax.loglog(xx, xx, "r--", lw=1, label="y = x")
    ax.set_xlabel("Fortran rnuclg [s⁻¹]")
    ax.set_ylabel("JAX rnuclg [s⁻¹]")
    ax.set_title(f"Phase 11.1: freezaerl_mohler2010 — JAX vs Fortran "
                 f"({args.n} scenarios × {nbin} bins, "
                 f"{int(nonzero.sum())} non-zero pts)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()

    ax2 = axes[1]
    d_nm = 2.0 * r_bins * 1e7
    nz_per_bin = [rel[:, b][nonzero[:, b]] for b in range(nbin)]
    nz_for_plot = [
        np.where(x > 1e-16, x, 1e-16) if x.size else np.array([1e-16])
        for x in nz_per_bin
    ]

    # Jittered scatter dots behind the boxes — one dot per scenario per bin.
    rng_jitter = np.random.default_rng(0)
    for b in range(nbin):
        ys = nz_for_plot[b]
        if ys.size == 0:
            continue
        jitter = rng_jitter.normal(loc=0.0, scale=0.07, size=ys.shape)
        xs = d_nm[b] * np.exp(jitter)
        ax2.scatter(xs, ys, s=3, alpha=0.18, color="steelblue",
                    edgecolors="none", zorder=1)

    ax2.boxplot(
        nz_for_plot,
        positions=d_nm, widths=d_nm * 0.18,
        showfliers=False, patch_artist=True,
        medianprops=dict(color="crimson", lw=1.5),
        boxprops=dict(facecolor="white", alpha=0.7, edgecolor="black", lw=1.0),
        whiskerprops=dict(color="black", lw=0.9), zorder=3,
    )
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.axhline(1e-12, color="green", ls="--", lw=1, alpha=0.7,
                label="rtol = 1e-12 (unit-test gate)")
    ax2.set_xlabel("Bin median diameter [nm]")
    ax2.set_ylabel("|J - F| / max(|J|, |F|)  per bin")
    ax2.set_title(f"Per-bin rel-err — dot = one of {args.n} scenarios, "
                  f"box = cross-scenario P25/median/P75")
    ax2.grid(True, alpha=0.3, which="both")
    ax2.legend(loc="lower left", fontsize=9)

    out_dir = ROOT / "plots" / "diff" / "phase11"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "freezaerl_mohler2010_jax_vs_fortran.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
