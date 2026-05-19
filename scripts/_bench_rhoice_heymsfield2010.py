"""Bench JAX rhoice_heymsfield2010 against gfortran standalone.

Setup-time kernel: per-bin ice density (Heymsfield 2010) and area ratio
(Schmitt & Heymsfield 2009). No traced state — only the regime selector
("deep"|"conv"|"cold"|"avg"|"synp"|"warm") and bin grid drive output.

The interesting sweep is across regimes (six `a` coefficients) for a
fixed CARMA ice bin grid. We also stress a range of rmrat / rmassmin
to confirm the formula is bit-exact at both the bulk-ice clip
(small-D end) and the aratelem branch crossover (200 µm).

Output:
  data/rhoice_heymsfield2010_bench.npz
  plots/diff/phase11/rhoice_heymsfield2010_jax_vs_fortran.png
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

from carma.constants import PI, RHO_I
from carma.rhoice_heymsfield2010 import rhoice_heymsfield2010

DEFAULT_BIN = (ROOT.parent / "original-carma" / "CARMA" / "build_standalone"
               / "rhoice_heymsfield2010_standalone")
REGIMES = ["deep", "conv", "cold", "avg", "synp", "warm"]


def run_fortran(binary, regime, rhoice, rmassmin, rmrat, nbin, work_dir):
    in_path = work_dir / f"in_{regime}.txt"
    out_path = work_dir / f"out_{regime}.bin"
    with open(in_path, "w") as f:
        f.write(f"{regime} {rhoice!r} {rmassmin!r} {rmrat!r} {nbin}\n")
    subprocess.run([str(binary), str(in_path), str(out_path)], check=True)
    arr = np.fromfile(out_path, dtype=np.float64)
    return arr[:nbin], arr[nbin:]


def run_jax(regime, rhoice, rmassmin, rmrat, nbin):
    rho, ar = rhoice_heymsfield2010(rhoice, rmassmin, rmrat, regime, nbin)
    return np.asarray(rho), np.asarray(ar)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nbin", type=int, default=28)
    p.add_argument("--rmin_um", type=float, default=5.0,
                   help="smallest bin radius [µm]")
    p.add_argument("--rmrat", type=float, default=4.0)
    p.add_argument("--rhoice", type=float, default=float(RHO_I))
    p.add_argument("--binary", type=Path, default=DEFAULT_BIN)
    args = p.parse_args()
    if not args.binary.exists():
        print("ERROR: build with build_rhoice_heymsfield2010_standalone.sh first",
              file=sys.stderr); sys.exit(1)

    rmin_cm = args.rmin_um * 1e-4
    rmassmin = (4.0 / 3.0) * math.pi * rmin_cm**3 * args.rhoice
    nbin = args.nbin

    rho_F = np.zeros((len(REGIMES), nbin))
    rho_J = np.zeros_like(rho_F)
    ar_F = np.zeros_like(rho_F)
    ar_J = np.zeros_like(rho_F)

    print(f"Phase 11.10: rhoice_heymsfield2010 — nbin={nbin}, "
          f"rmin={args.rmin_um}µm, rmrat={args.rmrat}, "
          f"rhoice={args.rhoice}", flush=True)
    with tempfile.TemporaryDirectory(prefix="phase1110_rhoice_") as work_root:
        work_root = Path(work_root)
        for ir, regime in enumerate(REGIMES):
            rho_F[ir], ar_F[ir] = run_fortran(
                args.binary, regime, args.rhoice, rmassmin, args.rmrat,
                nbin, work_root)
            rho_J[ir], ar_J[ir] = run_jax(
                regime, args.rhoice, rmassmin, args.rmrat, nbin)
            print(f"  {regime}: done", flush=True)

    rho_denom = np.maximum(np.abs(rho_F), 1e-300)
    ar_denom = np.maximum(np.abs(ar_F), 1e-300)
    rho_rel = np.abs(rho_J - rho_F) / rho_denom
    ar_rel = np.abs(ar_J - ar_F) / ar_denom

    print(f"\nrel err over {len(REGIMES) * nbin} (regime × bin) points:")
    for q in (50, 90, 95, 99, 100):
        print(f"  rho       P{q}: {np.percentile(rho_rel, q):.3e}    "
              f"aratelem P{q}: {np.percentile(ar_rel, q):.3e}")

    # bin diameters (cm → µm) — geometric grid from rmin
    ibin = np.arange(nbin)
    rbin_cm = ((rmassmin * args.rmrat ** ibin / args.rhoice)
               * 3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
    dbin_um = 2.0 * rbin_cm * 1e4

    np.savez_compressed(
        ROOT / "data" / "rhoice_heymsfield2010_bench.npz",
        rho_F=rho_F, rho_J=rho_J, ar_F=ar_F, ar_J=ar_J,
        regimes=np.array(REGIMES), dbin_um=dbin_um,
        rmassmin=rmassmin, rmrat=args.rmrat, rhoice=args.rhoice)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(REGIMES)))

    ax = axes[0]
    for ir, regime in enumerate(REGIMES):
        ax.plot(dbin_um, rho_F[ir], "-", lw=2.0, color=cmap[ir],
                label=f"{regime}  (F)")
        ax.plot(dbin_um, rho_J[ir], "x", ms=5, color=cmap[ir], alpha=0.7)
    ax.axhline(args.rhoice, color="red", ls=":", lw=1, alpha=0.6,
               label=f"ρ_ice clip = {args.rhoice}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("D (equivalent-sphere) [µm]")
    ax.set_ylabel("ρ_eff [g/cm³]")
    ax.set_title(
        f"Phase 11.10: rhoice_heymsfield2010 — "
        f"JAX (×) vs Fortran (—), max rel err {rho_rel.max():.1e}")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="lower left", fontsize=8, ncol=2)

    ax2 = axes[1]
    for ir, regime in enumerate(REGIMES):
        ax2.plot(dbin_um, ar_F[ir], "-", lw=2.0, color=cmap[ir],
                 label=f"{regime}  (F)")
        ax2.plot(dbin_um, ar_J[ir], "x", ms=5, color=cmap[ir], alpha=0.7)
    ax2.axvline(200.0, color="red", ls=":", lw=1, alpha=0.6,
                label="branch: exp(−38·D) | 0.16·D^(−0.27)")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("D [µm]"); ax2.set_ylabel("aratelem (projected area ratio)")
    ax2.set_title(
        f"aratelem (Schmitt & Heymsfield 2009) — "
        f"max rel err {ar_rel.max():.1e}")
    ax2.grid(True, alpha=0.3, which="both")
    ax2.legend(loc="lower left", fontsize=8, ncol=2)

    out_dir = ROOT / "plots" / "diff" / "phase11"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rhoice_heymsfield2010_jax_vs_fortran.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
