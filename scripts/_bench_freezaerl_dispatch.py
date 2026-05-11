"""Phase 11.5: in-dispatch bench for the three freezaerl_* kernels.

Routes the same parameter sweep through Fortran's full upstream
subroutine dispatch wrapper (cstate-resolved rhosol from the configured
solute, group/element loops, inucproc gating). Drives the patched
`carma_nuc2test_diagnostic` Fortran binary with the chosen kernel
selector and compares the dumped rnuclg to JAX.

Usage:
  python scripts/_bench_freezaerl_dispatch.py --kernel mohler    --n 200
  python scripts/_bench_freezaerl_dispatch.py --kernel tabazadeh --n 200
  python scripts/_bench_freezaerl_dispatch.py --kernel koop      --n 200

Output:
  data/freezaerl_<kernel>_dispatch_bench.npz
  plots/diff/phase11/freezaerl_<kernel>_dispatch_jax_vs_fortran.png
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
from carma.nucleation.freezaerl_tabazadeh2000 import freezaerl_tabazadeh2000
from carma.nucleation.freezaerl_koop2000 import freezaerl_koop2000

DEFAULT_BIN = (ROOT.parent / "original-carma" / "CARMA" / "build"
               / "test_nuc2test_diagnostic")

# carma_nuc2test geometry (Sulfate IN group)
NBIN = 16
NGROUP = 2
RMIN_CM, RMRAT, RHO_PARTICLE = 1.0e-7, 4.0, 1.78
# Density the Fortran kernel actually uses (cstate-resolved solute density,
# from CARMASOLUTE_Create(carma, 1, "Sulfuric Acid", 2, 98._f, 1.38_f, rc)).
RHO_SOL = 1.38

# Probe filename + Python kernel function — keyed by selector.
KERNELS = {
    "mohler":    ("freezaerl_mohler_probe.bin",    freezaerl_mohler2010),
    "tabazadeh": ("freezaerl_tabazadeh_probe.bin", freezaerl_tabazadeh2000),
    "koop":      ("freezaerl_koop_probe.bin",      freezaerl_koop2000),
}


def bin_grid():
    vmin = (4.0 / 3.0) * math.pi * RMIN_CM**3 * RHO_PARTICLE
    rmass = vmin * RMRAT ** np.arange(NBIN)
    vol = rmass / RHO_PARTICLE
    r = (3.0 * rmass / (4.0 * math.pi * RHO_PARTICLE)) ** (1.0 / 3.0)
    return r, vol


def sample_scenarios(n, rng, kernel):
    """Parameter sweep matched to each kernel's standalone bench.

    Tabazadeh has no T<240 gate; Möhler / Koop both gate at T ≤ 240.
    """
    if kernel == "tabazadeh":
        T = rng.uniform(180.0, 270.0, n)
    else:
        T = rng.uniform(180.0, 240.0, n)
    return dict(
        T          = T,
        ssi        = rng.uniform(0.31,    1.00,  n),
        ssl        = rng.uniform(-0.99,   0.05,  n),
        akelvin    = rng.uniform(1.0e-7,  3.0e-7, n),
        akelvini   = rng.uniform(1.0e-7,  3.0e-7, n),
        pconmax    = 10.0 ** rng.uniform(-2.0,    4.0,  n),
    )


def run_fortran(binary, kernel, scen, work_dir, base_T=210.0,
                 base_p_hPa=200.0, base_rh=0.95, base_n=100.0,
                 base_mu_cm=2.5e-6, base_sig=1.5):
    """Run the Fortran diagnostic and return (rnuclg, p_actual_cgs).

    p_actual_cgs is read from the cstate dump — we pass this back so
    the JAX side uses the *exact* pressure the kernel saw (which may
    differ from base_p_hPa due to atmosphere-helper unit conversions).
    """
    scen_path = work_dir / "scen.txt"
    out_dir = work_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(scen_path, "w") as f:
        f.write(
            f"{base_T} {base_p_hPa} {base_rh} {base_n} {base_mu_cm} {base_sig}  "
            f"{scen['T']!r} {scen['ssi']!r} {scen['ssl']!r}  "
            f"{scen['akelvin']!r} {scen['akelvini']!r} {scen['pconmax']!r}\n"
        )
    subprocess.run([str(binary), str(scen_path), str(out_dir), "1", kernel],
                    check=True, capture_output=True)
    probe_name, _ = KERNELS[kernel]
    arr = np.fromfile(out_dir / f"substep_0001_{probe_name}", dtype=np.float64)
    arr = arr.reshape((NBIN, NGROUP, NGROUP), order='F')
    p_actual = float(np.fromfile(out_dir / "substep_0001_p.bin",
                                   dtype=np.float64)[0])
    return arr[:, 0, 1], p_actual


def run_jax(kernel, scen, r_bins, vol_bins, p_cgs):
    if kernel == "mohler":
        return np.asarray(freezaerl_mohler2010(
            t_val=jnp.float64(scen['T']),
            supsati_val=jnp.float64(scen['ssi']),
            supsatl_val=jnp.float64(scen['ssl']),
            akelvin_val=jnp.float64(scen['akelvin']),
            akelvini_val=jnp.float64(scen['akelvini']),
            r_bins=jnp.asarray(r_bins), vol_bins=jnp.asarray(vol_bins),
            rhosol_val=jnp.float64(RHO_SOL),
            pconmax_val=jnp.float64(scen['pconmax']),
        ))
    if kernel == "tabazadeh":
        return np.asarray(freezaerl_tabazadeh2000(
            t_val=jnp.float64(scen['T']),
            supsati_val=jnp.float64(scen['ssi']),
            supsatl_val=jnp.float64(scen['ssl']),
            akelvin_val=jnp.float64(scen['akelvin']),
            r_bins=jnp.asarray(r_bins), vol_bins=jnp.asarray(vol_bins),
            rhosol_val=jnp.float64(RHO_SOL),
            gwtmol_val=jnp.float64(18.016),
            pconmax_val=jnp.float64(scen['pconmax']),
        ))
    if kernel == "koop":
        # Koop reads pressure from cstate%f_p — pass the dumped value
        # so JAX sees exactly what the upstream subroutine saw.
        return np.asarray(freezaerl_koop2000(
            t_val=jnp.float64(scen['T']),
            p_val=jnp.float64(p_cgs),
            supsati_val=jnp.float64(scen['ssi']),
            supsatl_val=jnp.float64(scen['ssl']),
            akelvin_val=jnp.float64(scen['akelvin']),
            r_bins=jnp.asarray(r_bins), vol_bins=jnp.asarray(vol_bins),
            rhosol_val=jnp.float64(RHO_SOL),
            pconmax_val=jnp.float64(scen['pconmax']),
            nbin=NBIN,
        ))
    raise ValueError(f"unknown kernel: {kernel}")


def make_plot(kernel, n, rnuclg_F, rnuclg_J, r_bins):
    denom = np.maximum(np.abs(rnuclg_F), np.abs(rnuclg_J))
    denom = np.where(denom > 1e-300, denom, 1.0)
    rel = np.abs(rnuclg_F - rnuclg_J) / denom
    nonzero = rnuclg_F > 0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    F_flat = rnuclg_F[nonzero]; J_flat = rnuclg_J[nonzero]
    ax.loglog(F_flat, J_flat, "k.", ms=2, alpha=0.4)
    if F_flat.size:
        xx = np.geomspace(max(F_flat.min(), 1e-30), F_flat.max(), 5)
        ax.loglog(xx, xx, "r--", lw=1, label="y = x")
    ax.set_xlabel("Fortran rnuclg [s⁻¹]   (upstream subroutine, full dispatch)")
    ax.set_ylabel("JAX rnuclg [s⁻¹]")
    label = {"mohler": "Möhler 2010", "tabazadeh": "Tabazadeh 2000",
             "koop": "Koop 2000"}[kernel]
    ax.set_title(f"Phase 11.5: {label} — JAX vs Fortran-in-dispatch "
                 f"({n} scen × {NBIN} bins, {int(nonzero.sum())} nonzero)")
    ax.grid(True, alpha=0.3, which="both"); ax.legend()

    ax2 = axes[1]
    d_nm = 2.0 * r_bins * 1e7
    nz_per_bin = [rel[:, b][nonzero[:, b]] for b in range(NBIN)]
    nz_for_plot = [
        np.where(x > 1e-16, x, 1e-16) if x.size else np.array([1e-16])
        for x in nz_per_bin
    ]
    rng_jit = np.random.default_rng(0)
    for b in range(NBIN):
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
    ax2.set_title(f"Per-bin rel-err — dot = one of {n} scenarios, "
                  f"box = cross-scenario P25/median/P75")
    ax2.grid(True, alpha=0.3, which="both"); ax2.legend(loc="lower left", fontsize=9)

    out_dir = ROOT / "plots" / "diff" / "phase11"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"freezaerl_{kernel}_dispatch_jax_vs_fortran.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"\nsaved {out_path}")
    return rel, nonzero


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kernel", choices=list(KERNELS.keys()), required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--binary", type=Path, default=DEFAULT_BIN)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.binary.exists():
        print(f"ERROR: build with apply_nuc2_diagnostic_patch.sh first",
              file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    r_bins, vol_bins = bin_grid()
    scens = sample_scenarios(args.n, rng, args.kernel)

    rnuclg_F = np.zeros((args.n, NBIN))
    rnuclg_J = np.zeros((args.n, NBIN))
    print(f"Phase 11.5: {args.kernel} — {args.n} scenarios via Fortran dispatch + JAX",
          flush=True)
    with tempfile.TemporaryDirectory(prefix="phase115_") as work_root:
        work_root = Path(work_root)
        for i in range(args.n):
            scen = {k: float(v[i]) for k, v in scens.items()}
            work_dir = work_root / f"s{i:05d}"
            work_dir.mkdir()
            rnuclg_F[i], p_actual = run_fortran(
                args.binary, args.kernel, scen, work_dir)
            rnuclg_J[i] = run_jax(args.kernel, scen, r_bins, vol_bins, p_actual)
            shutil.rmtree(work_dir, ignore_errors=True)
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{args.n}", flush=True)

    np.savez_compressed(
        ROOT / "data" / f"freezaerl_{args.kernel}_dispatch_bench.npz",
        rnuclg_F=rnuclg_F, rnuclg_J=rnuclg_J,
        r_bins=r_bins, vol_bins=vol_bins, **scens)

    rel, nonzero = make_plot(args.kernel, args.n, rnuclg_F, rnuclg_J, r_bins)
    print(f"\nrel err over {args.n} scen × {NBIN} bins ({int(nonzero.sum())} nonzero):")
    if nonzero.any():
        nz = rel[nonzero]
        for q in (50, 90, 95, 99, 100):
            print(f"  P{q}: {np.percentile(nz, q):.3e}")


if __name__ == "__main__":
    main()
