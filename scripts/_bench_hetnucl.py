"""Bench JAX hetnucl against the standalone Fortran kernel.

PMC-regime parameter sweep (mesospheric — T 100-180 K, p < 1 hPa,
high supsati).

Output: data/hetnucl_bench.npz +
        plots/diff/phase11/hetnucl_jax_vs_fortran.png
"""
import argparse
import math
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

from carma.nucleation.hetnucl import hetnucl

DEFAULT_BIN = (ROOT.parent / "original-carma" / "CARMA" / "build_standalone"
               / "hetnucl_standalone")


def bin_grid(nbin=16, rmin_cm=1e-7, rmrat=4.0, rho=1.78):
    vmin = (4.0 / 3.0) * math.pi * rmin_cm**3 * rho
    rmass = vmin * rmrat ** np.arange(nbin)
    r = (3.0 * rmass / (4.0 * math.pi * rho)) ** (1.0 / 3.0)
    return r


def sample_scenarios(n, rng):
    """PMC mesospheric regime: very cold, very low pressure, high ssi.
    Sample uniformly over each axis to stress-test the kernel."""
    return dict(
        T          = rng.uniform(100.0,  180.0,  n),    # mesopause
        p_dyn      = 10.0 ** rng.uniform(np.log10(1e-1), np.log10(9.9e2), n),
        ssi        = 10.0 ** rng.uniform(np.log10(0.01), np.log10(10.0), n),
        gc_h2o     = 10.0 ** rng.uniform(-15.0, -10.0, n),
        gwtmol     = np.full(n, 18.016),
        surfctia   = rng.uniform(80.0,   130.0,  n),    # ice-air σ
        pconmax    = 10.0 ** rng.uniform(-2.0,   4.0,  n),
    )


def run_fortran(binary, scen, r_bins, work_dir):
    nbin = r_bins.shape[0]
    in_path = work_dir / "in.txt"
    out_path = work_dir / "out.bin"
    with open(in_path, "w") as f:
        f.write(f"{scen['T']!r} {scen['p_dyn']!r} {scen['ssi']!r} "
                f"{scen['gc_h2o']!r} {scen['gwtmol']!r} {scen['surfctia']!r} "
                f"{scen['pconmax']!r} {nbin}\n")
        f.write(" ".join(repr(float(x)) for x in r_bins) + "\n")
    subprocess.run([str(binary), str(in_path), str(out_path)], check=True)
    return np.fromfile(out_path, dtype=np.float64)


def run_jax(scen, r_bins):
    return np.asarray(hetnucl(
        t_val=jnp.float64(scen['T']),
        p_val=jnp.float64(scen['p_dyn']),
        supsati_val=jnp.float64(scen['ssi']),
        gc_h2o=jnp.float64(scen['gc_h2o']),
        gwtmol_h2o=jnp.float64(scen['gwtmol']),
        surfctia_val=jnp.float64(scen['surfctia']),
        r_bins=jnp.asarray(r_bins),
        pconmax_val=jnp.float64(scen['pconmax']),
    ))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--binary", type=Path, default=DEFAULT_BIN)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if not args.binary.exists():
        print(f"ERROR: build with build_hetnucl_standalone.sh first",
              file=sys.stderr); sys.exit(1)

    rng = np.random.default_rng(args.seed)
    r_bins = bin_grid()
    nbin = r_bins.shape[0]
    scens = sample_scenarios(args.n, rng)

    rnuclg_F = np.zeros((args.n, nbin))
    rnuclg_J = np.zeros((args.n, nbin))
    print(f"Phase 11.7: {args.n} scenarios through hetnucl Fortran + JAX",
          flush=True)
    with tempfile.TemporaryDirectory(prefix="phase117_") as work_root:
        work_root = Path(work_root)
        for i in range(args.n):
            scen = {k: float(v[i]) for k, v in scens.items()}
            rnuclg_F[i] = run_fortran(args.binary, scen, r_bins, work_root)
            rnuclg_J[i] = run_jax(scen, r_bins)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{args.n}", flush=True)

    np.savez_compressed(
        ROOT / "data" / "hetnucl_bench.npz",
        rnuclg_F=rnuclg_F, rnuclg_J=rnuclg_J, r_bins=r_bins, **scens)

    denom = np.maximum(np.abs(rnuclg_F), np.abs(rnuclg_J))
    denom = np.where(denom > 1e-300, denom, 1.0)
    rel = np.abs(rnuclg_F - rnuclg_J) / denom
    nonzero = rnuclg_F > 0
    print(f"\nrel err over {args.n} scen × {nbin} bins ({int(nonzero.sum())} nonzero):")
    if nonzero.any():
        nz = rel[nonzero]
        for q in (50, 90, 95, 99, 100):
            print(f"  P{q}: {np.percentile(nz, q):.3e}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    F_flat = rnuclg_F[nonzero]; J_flat = rnuclg_J[nonzero]
    ax.loglog(F_flat, J_flat, "k.", ms=2, alpha=0.4)
    if F_flat.size:
        xx = np.geomspace(max(F_flat.min(), 1e-30), F_flat.max(), 5)
        ax.loglog(xx, xx, "r--", lw=1, label="y = x")
    ax.set_xlabel("Fortran rnuclg [s⁻¹]")
    ax.set_ylabel("JAX rnuclg [s⁻¹]")
    ax.set_title(f"Phase 11.7: hetnucl — JAX vs Fortran "
                 f"({args.n} scen × {nbin} bins, {int(nonzero.sum())} nonzero)")
    ax.grid(True, alpha=0.3, which="both"); ax.legend()

    ax2 = axes[1]
    d_nm = 2.0 * r_bins * 1e7
    nz_per_bin = [rel[:, b][nonzero[:, b]] for b in range(nbin)]
    nz_for_plot = [
        np.where(x > 1e-16, x, 1e-16) if x.size else np.array([1e-16])
        for x in nz_per_bin
    ]
    rng_jit = np.random.default_rng(0)
    for b in range(nbin):
        ys = nz_for_plot[b]
        if ys.size == 0: continue
        jitter = rng_jit.normal(loc=0.0, scale=0.07, size=ys.shape)
        xs = d_nm[b] * np.exp(jitter)
        ax2.scatter(xs, ys, s=3, alpha=0.18, color="steelblue",
                    edgecolors="none", zorder=1)
    ax2.boxplot(nz_for_plot,
                positions=d_nm, widths=d_nm * 0.18,
                showfliers=False, patch_artist=True,
                medianprops=dict(color="crimson", lw=1.5),
                boxprops=dict(facecolor="white", alpha=0.7, edgecolor="black"),
                whiskerprops=dict(color="black", lw=0.9), zorder=3)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.axhline(1e-12, color="green", ls="--", lw=1, alpha=0.7,
                label="rtol = 1e-12")
    ax2.set_xlabel("Bin median diameter [nm]")
    ax2.set_ylabel("|J - F| / max(|J|, |F|)  per bin")
    ax2.set_title(f"Per-bin rel-err — dot = one of {args.n} scenarios, "
                  f"box = cross-scenario P25/median/P75")
    ax2.grid(True, alpha=0.3, which="both"); ax2.legend(loc="lower left", fontsize=9)

    out_dir = ROOT / "plots" / "diff" / "phase11"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hetnucl_jax_vs_fortran.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
