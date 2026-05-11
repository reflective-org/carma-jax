"""Bench JAX setup_vf_heymsfield2010 against gfortran standalone.

Heymsfield & Westbrook (2010) fall velocity for ice particles. Single
ice regime; bins from ~5 µm to several cm. The interesting axis is
(altitude, bin) — vf scales with bin size (gravity / drag) and with
altitude (lower ρ_air → larger Re for the same particle).

Sweep: a realistic mid-latitude column (NZ=32, T 200→260 K with
height) × NBIN=28 ice bins. Random per-altitude jitter prevents
trivial dataset gating.

Output:
  data/setup_vf_heymsfield2010_bench.npz
  plots/diff/phase11/setup_vf_heymsfield2010_jax_vs_fortran.png
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

from carma.constants import RHO_I
from carma.rhoice_heymsfield2010 import rhoice_heymsfield2010
from carma.setup_vf_heymsfield2010 import setup_vf_heymsfield2010

DEFAULT_BIN = (ROOT.parent / "original-carma" / "CARMA" / "build_standalone"
               / "setup_vf_heymsfield2010_standalone")


def build_column(nz, rng):
    """Realistic tropospheric/lower-stratospheric column."""
    z_km = np.linspace(0.0, 20.0, nz)
    # Linear T fit to US Standard Atmosphere up to ~15 km, then isothermal.
    t = np.where(z_km < 11.0, 288.15 - 6.5 * z_km, 216.65).astype(np.float64)
    # Tiny jitter so no two layers are identical (~0.1 K)
    t = t + rng.normal(0, 0.1, nz)

    p_hpa = 1013.25 * np.exp(-z_km / 8.0)
    # CGS air density via ideal gas: ρ = p / (R_specific · T)
    R_specific = 287.05 * 1e4         # J/kg/K → erg/g/K
    rho_g_per_cm3 = (p_hpa * 1e3) / (R_specific * t)   # p[dyne/cm²] / R · T
    zmet = np.full(nz, 1.0e5)         # 1 km layers in cm
    rhoa = rho_g_per_cm3 * zmet       # column density g/cm² per layer

    # Sutherland: μ = μ₀ · (T/T₀)^1.5 · (T₀ + S) / (T + S),  T₀=273, S=110.4
    mu0_g_cm_s = 1.716e-4
    rmu = mu0_g_cm_s * (t / 273.0) ** 1.5 * (273.0 + 110.4) / (t + 110.4)
    return t, rhoa, zmet, rmu


def build_bins(nbin, rmin_um=5.0, rmrat=4.0, regime="avg"):
    """Ice bins using rhoice_heymsfield2010 — same setup CARMA itself uses."""
    rhoice = float(RHO_I)
    rmin_cm = rmin_um * 1e-4
    rmassmin = (4.0 / 3.0) * math.pi * rmin_cm**3 * rhoice
    rmass = rmassmin * rmrat ** np.arange(nbin)
    rho_eff, arat = rhoice_heymsfield2010(rhoice, rmassmin, rmrat, regime, nbin)
    rho_eff = np.asarray(rho_eff); arat = np.asarray(arat)
    r = (rmass / ((4.0 / 3.0) * math.pi * rho_eff)) ** (1.0 / 3.0)
    rrat = np.ones(nbin)
    d_um = 2.0 * r * 1e4
    return r, rrat, rmass, arat, d_um


def run_fortran(binary, t, rhoa, zmet, rmu, r_wet, rrat, rmass, arat, work):
    nz, nbin = r_wet.shape
    in_path = work / "in.txt"
    out_path = work / "out.bin"
    with open(in_path, "w") as f:
        f.write(f"{nz} {nbin}\n")
        f.write(" ".join(repr(float(x)) for x in t) + "\n")
        f.write(" ".join(repr(float(x)) for x in rhoa) + "\n")
        f.write(" ".join(repr(float(x)) for x in zmet) + "\n")
        f.write(" ".join(repr(float(x)) for x in rmu) + "\n")
        for k in range(nz):
            f.write(" ".join(repr(float(x)) for x in r_wet[k]) + "\n")
        f.write(" ".join(repr(float(x)) for x in rrat) + "\n")
        f.write(" ".join(repr(float(x)) for x in rmass) + "\n")
        f.write(" ".join(repr(float(x)) for x in arat) + "\n")
    subprocess.run([str(binary), str(in_path), str(out_path)], check=True)
    arr = np.fromfile(out_path, dtype=np.float64)
    n = nz * nbin
    return (arr[:n].reshape(nz, nbin),
            arr[n:2 * n].reshape(nz, nbin),
            arr[2 * n:3 * n].reshape(nz, nbin))


def run_jax(t, rhoa, zmet, rmu, r_wet, rrat, rmass, arat):
    vf, re, bpm = setup_vf_heymsfield2010(
        t=jnp.asarray(t), rhoa=jnp.asarray(rhoa), zmet=jnp.asarray(zmet),
        rmu=jnp.asarray(rmu),
        r_wet=jnp.asarray(r_wet[:, :, None]),
        rrat=jnp.asarray(rrat[:, None]),
        rmass=jnp.asarray(rmass[:, None]),
        arat=jnp.asarray(arat[:, None]))
    return (np.asarray(vf)[..., 0],
            np.asarray(re)[..., 0],
            np.asarray(bpm)[..., 0])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nz", type=int, default=32)
    p.add_argument("--nbin", type=int, default=28)
    p.add_argument("--rmin_um", type=float, default=5.0)
    p.add_argument("--rmrat", type=float, default=4.0)
    p.add_argument("--regime", default="avg")
    p.add_argument("--binary", type=Path, default=DEFAULT_BIN)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if not args.binary.exists():
        print("ERROR: build with build_setup_vf_heymsfield2010_standalone.sh first",
              file=sys.stderr); sys.exit(1)

    rng = np.random.default_rng(args.seed)
    t, rhoa, zmet, rmu = build_column(args.nz, rng)
    r, rrat, rmass, arat, d_um = build_bins(
        args.nbin, args.rmin_um, args.rmrat, args.regime)
    r_wet = np.broadcast_to(r[None, :], (args.nz, args.nbin)).copy()

    print(f"Phase 11.10: setup_vf_heymsfield2010 — "
          f"NZ={args.nz}, NBIN={args.nbin}, regime={args.regime}", flush=True)
    with tempfile.TemporaryDirectory(prefix="phase1110_vf_") as work_root:
        work_root = Path(work_root)
        vf_F, re_F, bpm_F = run_fortran(
            args.binary, t, rhoa, zmet, rmu, r_wet, rrat, rmass, arat, work_root)
    vf_J, re_J, bpm_J = run_jax(t, rhoa, zmet, rmu, r_wet, rrat, rmass, arat)

    np.savez_compressed(
        ROOT / "data" / "setup_vf_heymsfield2010_bench.npz",
        vf_F=vf_F, vf_J=vf_J, re_F=re_F, re_J=re_J, bpm_F=bpm_F, bpm_J=bpm_J,
        t=t, rhoa=rhoa, zmet=zmet, rmu=rmu, d_um=d_um)

    print(f"\nrel err over {args.nz * args.nbin} (k × bin) points:")
    for name, F, J in (("vf", vf_F, vf_J),
                       ("re", re_F, re_J),
                       ("bpm", bpm_F, bpm_J)):
        denom = np.maximum(np.abs(F), 1e-300)
        rel = np.abs(J - F) / denom
        print(f"  {name:>3}:  P50 {np.percentile(rel, 50):.2e}    "
              f"P95 {np.percentile(rel, 95):.2e}    "
              f"P100 {rel.max():.2e}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    z_km = np.arange(args.nz)

    # vf curves coloured by altitude
    ax = axes[0]
    cmap = plt.cm.viridis(np.linspace(0, 1, args.nz))
    for k in range(args.nz):
        ax.plot(d_um, vf_F[k], "-", lw=1.5, color=cmap[k], alpha=0.8)
        ax.plot(d_um, vf_J[k], "x", ms=3.0, color=cmap[k], alpha=0.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("D [µm]"); ax.set_ylabel("vf [cm/s]")
    denom = np.maximum(np.abs(vf_F), 1e-300)
    rel = np.abs(vf_J - vf_F) / denom
    ax.set_title(
        f"Phase 11.10: vf_heymsfield2010 "
        f"(— Fortran, × JAX)  —  max rel err {rel.max():.1e}")
    ax.grid(True, alpha=0.3, which="both")
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                                norm=plt.Normalize(vmin=0, vmax=args.nz - 1))
    plt.colorbar(sm, ax=ax, label="altitude index (0=surface)")

    # Reynolds number scatter
    ax = axes[1]
    ax.scatter(re_F.ravel(), re_J.ravel(), c=np.tile(z_km, args.nbin),
                cmap="viridis", s=8, alpha=0.6)
    rr = np.geomspace(max(re_F.min(), 1e-12), re_F.max(), 5)
    ax.plot(rr, rr, "r--", lw=1, label="y = x")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Reynolds F"); ax.set_ylabel("Reynolds J")
    rel = np.abs(re_J - re_F) / np.maximum(np.abs(re_F), 1e-300)
    ax.set_title(f"Reynolds — max rel err {rel.max():.1e}")
    ax.grid(True, alpha=0.3, which="both"); ax.legend()

    # bpm scatter (slip correction)
    ax = axes[2]
    ax.scatter(bpm_F.ravel(), bpm_J.ravel(),
                c=np.tile(z_km, args.nbin), cmap="viridis", s=8, alpha=0.6)
    bb = np.geomspace(max(bpm_F.min(), 1e-12), bpm_F.max(), 5)
    ax.plot(bb, bb, "r--", lw=1, label="y = x")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("bpm F"); ax.set_ylabel("bpm J")
    rel = np.abs(bpm_J - bpm_F) / np.maximum(np.abs(bpm_F), 1e-300)
    ax.set_title(f"slip correction — max rel err {rel.max():.1e}")
    ax.grid(True, alpha=0.3, which="both"); ax.legend()

    out_dir = ROOT / "plots" / "diff" / "phase11"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "setup_vf_heymsfield2010_jax_vs_fortran.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
