"""Overlay dashboard: fp64 vs fp32 vs Fortran.

Reads the per-test .npz files produced by compare_precision.py (run
once in fp64 and once in fp32) and emits a single comparison image
alongside a wide summary table.

Usage:
    python scripts/compare_precision.py                     # creates *_fp64.npz
    CARMA_DTYPE=fp32 python scripts/compare_precision.py    # creates *_fp32.npz
    python scripts/plot_precision_overlay.py                # reads both, plots
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = (Path(__file__).parent.parent
            / "plots" / "precision_comparison" / "data")
OUT_PATH = (Path(__file__).parent.parent
            / "plots" / "precision_comparison" / "dashboard_overlay.png")

TESTS = [
    ("coagtest",   "radius [cm]",   "number density [cm⁻³]"),
    ("falltest",   "altitude [km]", "MMR [g/g]"),
    ("vdiftest",   "altitude [km]", "MMR [g/g]"),
    ("drydeptest", "altitude [km]", "MMR [g/g]"),
    ("growtest",   "radius [cm]",   "bin MMR [g/g]"),
]


def load(test, tag):
    path = DATA_DIR / f"{test}_{tag}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run "
            f"`{'CARMA_DTYPE=' + tag + ' ' if tag == 'fp32' else ''}"
            f"python scripts/compare_precision.py` first.")
    z = dict(np.load(path, allow_pickle=True))
    return z


def rel_err(jax_arr, fort_arr):
    mask = np.asarray(fort_arr) > 1e-30
    if not mask.any():
        return mask, np.array([])
    r = np.abs(np.asarray(jax_arr)[mask]
               - np.asarray(fort_arr)[mask]) / np.asarray(fort_arr)[mask]
    return mask, r


def safe_float(x, default=np.nan):
    try:
        v = float(np.asarray(x).item())
        return v if np.isfinite(v) else default
    except Exception:
        return default


def main():
    rows = []
    for name, xaxis, ylabel in TESTS:
        d64 = load(name, "fp64")
        try:
            d32 = load(name, "fp32")
        except FileNotFoundError as err:
            print(err)
            return
        rows.append((name, xaxis, ylabel, d64, d32))

    n = len(rows)
    fig = plt.figure(figsize=(5 * n, 12))

    for i, (name, xaxis, ylabel, d64, d32) in enumerate(rows):
        xvals = np.asarray(d64["xvals"])
        fort = np.asarray(d64["fortran_final"])
        jax64 = np.asarray(d64["jax_final"])
        jax32 = np.asarray(d32["jax_final"])

        # --- Top row: profile overlay ---
        ax = fig.add_subplot(3, n, i + 1)
        ax.plot(xvals, np.maximum(fort, 1e-50), "k--", lw=1.5, alpha=0.7, label="Fortran")
        ax.plot(xvals, np.maximum(jax64, 1e-50), "b-", lw=2, label="JAX fp64")
        ax.plot(xvals, np.maximum(jax32, 1e-50), "r-", lw=1.5, alpha=0.8, label="JAX fp32")
        ax.set_yscale("log")
        ax.set_xlabel(xaxis)
        ax.set_ylabel(ylabel)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        # --- Middle row: rel err vs Fortran, both precisions ---
        ax2 = fig.add_subplot(3, n, n + i + 1)
        for jax_arr, color, label in [(jax64, "b", "fp64"), (jax32, "r", "fp32")]:
            mask, re = rel_err(jax_arr, fort)
            if re.size:
                re = np.clip(re * 100, 1e-6, 1e20)
                ax2.plot(xvals[mask], re, "o-", color=color, ms=2, lw=0.8,
                         label=label, alpha=0.75)
        ax2.set_yscale("log")
        ax2.axhline(1.0, color="gray", ls="--", alpha=0.5)
        ax2.axhline(0.1, color="green", ls="--", alpha=0.5)
        ax2.set_xlabel(xaxis)
        ax2.set_ylabel("|JAX - Fortran| / Fortran  [%]")
        ax2.set_title("rel err vs Fortran")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)

        # --- Bottom row: fp64 vs fp32 delta (pure precision cost) ---
        ax3 = fig.add_subplot(3, n, 2 * n + i + 1)
        mask = (np.abs(jax64) > 1e-30) & np.isfinite(jax32)
        if mask.any():
            delta = np.abs(jax32[mask] - jax64[mask]) / np.abs(jax64[mask])
            delta = np.clip(delta * 100, 1e-6, 1e20)
            ax3.plot(xvals[mask], delta, "o-", color="m", ms=2, lw=0.8)
        ax3.set_yscale("log")
        ax3.axhline(1.0, color="gray", ls="--", alpha=0.5)
        ax3.set_xlabel(xaxis)
        ax3.set_ylabel("|fp32 - fp64| / fp64  [%]")
        ax3.set_title("precision cost")
        ax3.grid(True, alpha=0.3)

    fig.suptitle("CARMA-JAX precision comparison — fp64 vs fp32 vs Fortran",
                 fontweight="bold", fontsize=14)
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=130)
    plt.close(fig)
    print(f"Overlay dashboard: {OUT_PATH}")

    # --- Summary table ---
    print("\n" + "=" * 110)
    print(f"{'benchmark':<12} | "
          f"{'fp64 med %':>10} {'fp64 max %':>10} | "
          f"{'fp32 med %':>10} {'fp32 max %':>10} | "
          f"{'fp32/fp64 med %':>16} {'fp32/fp64 max %':>16}")
    print("-" * 110)
    for name, _, _, d64, d32 in rows:
        def fmt(x):
            if not np.isfinite(x):
                return "   nan/blow"
            return f"{x:10.4f}"
        med64 = safe_float(d64.get("rel_err_median")) * 100
        max64 = safe_float(d64.get("rel_err_max")) * 100
        med32 = safe_float(d32.get("rel_err_median")) * 100
        max32 = safe_float(d32.get("rel_err_max")) * 100

        fort = np.asarray(d64["fortran_final"])
        j64 = np.asarray(d64["jax_final"])
        j32 = np.asarray(d32["jax_final"])
        mask = (np.abs(j64) > 1e-30) & np.isfinite(j32)
        if mask.any():
            delta = np.abs(j32[mask] - j64[mask]) / np.abs(j64[mask]) * 100
            pd_med = float(np.median(delta))
            pd_max = float(np.max(delta))
        else:
            pd_med, pd_max = np.nan, np.nan
        print(f"{name:<12} | {fmt(med64)} {fmt(max64)} | "
              f"{fmt(med32)} {fmt(max32)} | "
              f"{fmt(pd_med):>16} {fmt(pd_max):>16}")
    print("=" * 110)


if __name__ == "__main__":
    main()
