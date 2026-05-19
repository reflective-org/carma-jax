"""Phase 11.5 + 11.11: in-dispatch bench for nucleation kernels.

Routes each JAX nucleation kernel's inputs through Fortran's full upstream
subroutine dispatch wrapper (cstate-resolved solute density / gas conc /
surface tension / pressure, group/element loops, `inucproc` gating). Drives
the patched `carma_nuc2test_diagnostic` Fortran binary with the chosen
kernel selector and compares the dumped `rnuclg` to JAX.

Phase 11.5 covered: mohler, tabazadeh, koop.
Phase 11.11 adds:   murray, hetnucl, melticel.

Usage:
  python scripts/_bench_freezaerl_dispatch.py --kernel mohler    --n 200
  python scripts/_bench_freezaerl_dispatch.py --kernel hetnucl   --n 200
  ...etc...

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
from carma.nucleation.freezglaerl_murray2010 import freezglaerl_murray2010
from carma.nucleation.hetnucl import hetnucl
from carma.nucleation.melticel import melticel

DEFAULT_BIN = (ROOT.parent / "original-carma" / "CARMA" / "build"
               / "test_nuc2test_diagnostic")

# carma_nuc2test geometry (Sulfate IN group)
NBIN = 16
NGROUP = 2
RMIN_CM, RMRAT, RHO_PARTICLE = 1.0e-7, 4.0, 1.78
RHO_SOL = 1.38                     # solute density read by upstream subroutines
GWTMOL_H2O = 18.016


def bin_grid():
    vmin = (4.0 / 3.0) * math.pi * RMIN_CM**3 * RHO_PARTICLE
    rmass = vmin * RMRAT ** np.arange(NBIN)
    vol = rmass / RHO_PARTICLE
    r = (3.0 * rmass / (4.0 * math.pi * RHO_PARTICLE)) ** (1.0 / 3.0)
    return r, vol


# Per-kernel config: probe filename, rnuclg index slice, scenario sampler,
# scenario→Fortran fields, JAX call. Each kernel gets a distinct param sweep
# matching its physical activation regime.
def _sample_freezaerl(n, rng, T_lo=180.0, T_hi=240.0):
    """Shared sampling for mohler/tabazadeh/koop."""
    return dict(
        T          = rng.uniform(T_lo, T_hi, n),
        ssi        = rng.uniform(0.31,    1.00, n),
        ssl        = rng.uniform(-0.99,   0.05, n),
        akelvin    = rng.uniform(1.0e-7,  3.0e-7, n),
        akelvini   = rng.uniform(1.0e-7,  3.0e-7, n),
        pconmax    = 10.0 ** rng.uniform(-2.0, 4.0, n),
        ssi_old    = np.zeros(n),
        p_over     = -np.ones(n),          # sentinel: don't override cstate.p
        gc_h2o     = -np.ones(n),          # sentinel: don't override gc
    )


def _sample_murray(n, rng):
    """Murray 2010 glassy aerosol freezing.

    T ≤ 212, ssi > ssi_old, ssi ≥ 0.21."""
    ssi = rng.uniform(0.21, 0.95, n)
    ssi_old = ssi * rng.uniform(0.0, 0.95, n)        # always < ssi
    return dict(
        T          = rng.uniform(150.0, 212.0, n),
        ssi        = ssi,
        ssl        = rng.uniform(-0.99, 0.05, n),
        akelvin    = rng.uniform(1.0e-7, 3.0e-7, n),
        akelvini   = rng.uniform(1.0e-7, 3.0e-7, n),
        pconmax    = 10.0 ** rng.uniform(-2.0, 4.0, n),
        ssi_old    = ssi_old,
        p_over     = -np.ones(n),
        gc_h2o     = -np.ones(n),
    )


def _sample_hetnucl(n, rng):
    """PMC heterogeneous ice nucleation.

    p < 1 hPa (= 1000 dyne/cm²), ssi > 0, gc > 0."""
    return dict(
        T          = rng.uniform(140.0, 230.0, n),
        ssi        = rng.uniform(0.05, 5.0, n),
        ssl        = rng.uniform(-0.99, 0.05, n),
        akelvin    = rng.uniform(1.0e-7, 3.0e-7, n),
        akelvini   = rng.uniform(1.0e-7, 3.0e-7, n),
        pconmax    = 10.0 ** rng.uniform(-2.0, 4.0, n),
        ssi_old    = np.zeros(n),
        # PMC pressures: 0.001 to 1 hPa → 1 to 1000 dyne/cm². Stay below the gate.
        p_over     = 10.0 ** rng.uniform(0.0, 2.5, n),
        gc_h2o     = 10.0 ** rng.uniform(-12.0, -7.0, n),
    )


def _sample_melticel(n, rng):
    """Ice melting: T > T0 = 273.16, pconmax > FEW_PC."""
    return dict(
        T          = rng.uniform(273.5, 320.0, n),
        ssi        = rng.uniform(-0.5, 0.5, n),
        ssl        = rng.uniform(-0.99, 0.05, n),
        akelvin    = rng.uniform(1.0e-7, 3.0e-7, n),
        akelvini   = rng.uniform(1.0e-7, 3.0e-7, n),
        pconmax    = 10.0 ** rng.uniform(-2.0, 4.0, n),
        ssi_old    = np.zeros(n),
        p_over     = -np.ones(n),
        gc_h2o     = -np.ones(n),
    )


def _jax_mohler(scen, r_bins, vol_bins, surfctia, p_cgs):
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


def _jax_tabazadeh(scen, r_bins, vol_bins, surfctia, p_cgs):
    return np.asarray(freezaerl_tabazadeh2000(
        t_val=jnp.float64(scen['T']),
        supsati_val=jnp.float64(scen['ssi']),
        supsatl_val=jnp.float64(scen['ssl']),
        akelvin_val=jnp.float64(scen['akelvin']),
        r_bins=jnp.asarray(r_bins), vol_bins=jnp.asarray(vol_bins),
        rhosol_val=jnp.float64(RHO_SOL),
        gwtmol_val=jnp.float64(GWTMOL_H2O),
        pconmax_val=jnp.float64(scen['pconmax']),
    ))


def _jax_koop(scen, r_bins, vol_bins, surfctia, p_cgs):
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


def _jax_murray(scen, r_bins, vol_bins, surfctia, p_cgs):
    # Diagnostic binary uses dtime = 1.0
    return np.asarray(freezglaerl_murray2010(
        t_val=jnp.float64(scen['T']),
        supsati_val=jnp.float64(scen['ssi']),
        supsati_old_val=jnp.float64(scen['ssi_old']),
        pconmax_val=jnp.float64(scen['pconmax']),
        dtime=jnp.float64(1.0),
        nbin=NBIN,
    ))


def _jax_hetnucl(scen, r_bins, vol_bins, surfctia, p_cgs):
    return np.asarray(hetnucl(
        t_val=jnp.float64(scen['T']),
        p_val=jnp.float64(p_cgs),
        supsati_val=jnp.float64(scen['ssi']),
        gc_h2o=jnp.float64(scen['gc_h2o']),
        gwtmol_h2o=jnp.float64(GWTMOL_H2O),
        surfctia_val=jnp.float64(surfctia),
        r_bins=jnp.asarray(r_bins),
        pconmax_val=jnp.float64(scen['pconmax']),
    ))


def _jax_melticel(scen, r_bins, vol_bins, surfctia, p_cgs):
    return np.asarray(melticel(
        t_val=jnp.float64(scen['T']),
        pconmax_val=jnp.float64(scen['pconmax']),
        nbin=NBIN,
    ))


KERNELS = {
    # name -> (probe_filename, slice_for_rnuclg, sampler, jax_callable, label)
    "mohler":    ("freezaerl_mohler_probe.bin",    (slice(None), 0, 1),
                  _sample_freezaerl, _jax_mohler,    "Möhler 2010"),
    "tabazadeh": ("freezaerl_tabazadeh_probe.bin", (slice(None), 0, 1),
                  lambda n, r: _sample_freezaerl(n, r, T_lo=180.0, T_hi=270.0),
                  _jax_tabazadeh, "Tabazadeh 2000"),
    "koop":      ("freezaerl_koop_probe.bin",      (slice(None), 0, 1),
                  _sample_freezaerl, _jax_koop,      "Koop 2000"),
    "murray":    ("freezglaerl_murray_probe.bin",  (slice(None), 0, 1),
                  _sample_murray, _jax_murray,      "Murray 2010 (glassy)"),
    "hetnucl":   ("hetnucl_probe.bin",             (slice(None), 0, 1),
                  _sample_hetnucl, _jax_hetnucl,    "PMC hetnucl"),
    "melticel":  ("melticel_probe.bin",            (slice(None), 1, 0),
                  _sample_melticel, _jax_melticel,  "Ice melting"),
}


def run_fortran(binary, kernel, scen, work_dir, base_T=210.0,
                 base_p_hPa=200.0, base_rh=0.95, base_n=100.0,
                 base_mu_cm=2.5e-6, base_sig=1.5):
    """Run the Fortran diagnostic and return (rnuclg, surfctia, p_cgs, gc_h2o).

    The post-inject fields are dumped so JAX uses the exact values the
    Fortran kernel saw, defending against any routing surprise (cf. the
    rhosol / p_actual fixes in Phase 11.4 / 11.5).
    """
    scen_path = work_dir / "scen.txt"
    out_dir = work_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(scen_path, "w") as f:
        f.write(
            f"{base_T} {base_p_hPa} {base_rh} {base_n} {base_mu_cm} {base_sig}  "
            f"{scen['T']!r} {scen['ssi']!r} {scen['ssl']!r}  "
            f"{scen['akelvin']!r} {scen['akelvini']!r} {scen['pconmax']!r}  "
            f"{scen['ssi_old']!r} {scen['p_over']!r} {scen['gc_h2o']!r}\n"
        )
    subprocess.run([str(binary), str(scen_path), str(out_dir), "1", kernel],
                    check=True, capture_output=True)
    probe_name, _, _, _, _ = KERNELS[kernel]
    arr = np.fromfile(out_dir / f"substep_0001_{probe_name}", dtype=np.float64)
    arr = arr.reshape((NBIN, NGROUP, NGROUP), order='F')
    p_actual = float(np.fromfile(out_dir / "substep_0001_p.bin",
                                   dtype=np.float64)[0])
    gc_actual = float(np.fromfile(out_dir / "substep_0001_gc.bin",
                                    dtype=np.float64)[0])
    surfctia_path = out_dir / "substep_0001_surfctia.bin"
    surfctia = (float(np.fromfile(surfctia_path, dtype=np.float64)[0])
                if surfctia_path.exists() else float("nan"))
    slc = KERNELS[kernel][1]
    return arr[slc], surfctia, p_actual, gc_actual


def make_plot(kernel, n, rnuclg_F, rnuclg_J, r_bins):
    _, _, _, _, label = KERNELS[kernel]
    denom = np.maximum(np.abs(rnuclg_F), np.abs(rnuclg_J))
    denom = np.where(denom > 1e-300, denom, 1.0)
    rel = np.abs(rnuclg_F - rnuclg_J) / denom
    nonzero = rnuclg_F > 0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    F_flat = rnuclg_F[nonzero]; J_flat = rnuclg_J[nonzero]
    if F_flat.size:
        ax.loglog(F_flat, J_flat, "k.", ms=2, alpha=0.4)
        xx = np.geomspace(max(F_flat.min(), 1e-30), F_flat.max(), 5)
        ax.loglog(xx, xx, "r--", lw=1, label="y = x")
    ax.set_xlabel("Fortran rnuclg [s⁻¹]   (upstream subroutine, full dispatch)")
    ax.set_ylabel("JAX rnuclg [s⁻¹]")
    ax.set_title(f"Phase 11 dispatch: {label} — JAX vs Fortran "
                 f"({n} scen × {NBIN} bins, {int(nonzero.sum())} nonzero)")
    ax.grid(True, alpha=0.3, which="both")
    if F_flat.size: ax.legend()

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
    _, _, sampler, jax_call, _ = KERNELS[args.kernel]
    scens = sampler(args.n, rng)

    rnuclg_F = np.zeros((args.n, NBIN))
    rnuclg_J = np.zeros((args.n, NBIN))
    surfctias = np.zeros(args.n)
    p_actuals = np.zeros(args.n)
    gc_actuals = np.zeros(args.n)
    print(f"Phase 11 dispatch: {args.kernel} — {args.n} scenarios via Fortran dispatch + JAX",
          flush=True)
    with tempfile.TemporaryDirectory(prefix="phase11_disp_") as work_root:
        work_root = Path(work_root)
        for i in range(args.n):
            scen = {k: float(v[i]) for k, v in scens.items()}
            work_dir = work_root / f"s{i:05d}"
            work_dir.mkdir()
            rnuclg_F[i], surfctias[i], p_actuals[i], gc_actuals[i] = run_fortran(
                args.binary, args.kernel, scen, work_dir)
            # Inject post-Step actuals into the scen dict before calling JAX
            scen_with_actuals = dict(scen, gc_h2o=gc_actuals[i])
            rnuclg_J[i] = jax_call(
                scen_with_actuals, r_bins, vol_bins, surfctias[i], p_actuals[i])
            shutil.rmtree(work_dir, ignore_errors=True)
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{args.n}", flush=True)

    np.savez_compressed(
        ROOT / "data" / f"freezaerl_{args.kernel}_dispatch_bench.npz",
        rnuclg_F=rnuclg_F, rnuclg_J=rnuclg_J,
        r_bins=r_bins, vol_bins=vol_bins,
        surfctias=surfctias, p_actuals=p_actuals, gc_actuals=gc_actuals,
        **scens)

    rel, nonzero = make_plot(args.kernel, args.n, rnuclg_F, rnuclg_J, r_bins)
    print(f"\nrel err over {args.n} scen × {NBIN} bins ({int(nonzero.sum())} nonzero):")
    if nonzero.any():
        nz = rel[nonzero]
        for q in (50, 90, 95, 99, 100):
            print(f"  P{q}: {np.percentile(nz, q):.3e}")
    else:
        print("  WARNING: no nonzero scenarios — sweep didn't activate the kernel")


if __name__ == "__main__":
    main()
