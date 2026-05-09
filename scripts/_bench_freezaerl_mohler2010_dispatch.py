"""Phase 11.4 (option B): integrated dispatch-context bench for
freezaerl_mohler2010.

Runs the patched `test_nuc2test_diagnostic` Fortran binary across many
scenarios. For each: the binary sets up a real CARMA cstate (NGROUP=2,
NELEM=3, NGAS=1), runs CARMASTATE_Step once to fully populate the
state, then injects the scenario inputs (T, ssi, ssl, akelvin, akelvini,
pconmax) into cstate and calls `freezaerl_mohler2010(carma, cstate, iz,
rc)` directly. The probe output (cstate%f_rnuclg) is dumped and
compared to the JAX port.

Difference vs the Phase 11.1 standalone bench: that one inlines the
kernel formula in a no-CARMA Fortran driver. This bench routes the
exact same inputs through the upstream subroutine's full dispatch
wrapper (group/element loops, inucproc gating, igas/ienucto resolution
from the carma_state config). Confirms the dispatch wrapper matches
what JAX is doing externally to the kernel formula.

Output: data/freezaerl_mohler2010_dispatch_bench.npz +
        plots/diff/phase11/freezaerl_mohler2010_dispatch_jax_vs_fortran.png
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

DEFAULT_BIN = (ROOT.parent / "original-carma" / "CARMA" / "build"
               / "test_nuc2test_diagnostic")

# carma_nuc2test 'Sulfate IN' group geometry: 16 bins, rmin = 0.1 µm,
# rmrat = 4. Same as the standalone bench so the comparison is direct.
NBIN = 16
NGROUP = 2     # Sulfate IN, Ice Crystal
NELEM = 3
RMIN_CM = 1.0e-7
RMRAT = 4.0
RHO_SULF = 1.78    # Sulfate IN ELEMENT density — used for bin geometry
                    # (rmass = vol × element_rho).
RHO_SOL = 1.38      # Sulfate SOLUTE density — what the kernel reads
                    # from cstate%f_solute(isol)%f_rho. The upstream
                    # subroutine's volrat formula uses *solute* density,
                    # not particle/element density.


def bin_grid():
    vmin = (4.0 / 3.0) * math.pi * RMIN_CM**3 * RHO_SULF
    rmass = vmin * RMRAT ** np.arange(NBIN)
    vol = rmass / RHO_SULF
    r = (3.0 * rmass / (4.0 * math.pi * RHO_SULF)) ** (1.0 / 3.0)
    return r, vol


def sample_scenarios(n, rng):
    """Same parameter coverage as scripts/_bench_freezaerl_mohler2010.py."""
    return dict(
        T          = rng.uniform(180.0,  240.0,  n),
        ssi        = rng.uniform(0.30,    1.00,  n),
        ssl        = rng.uniform(-0.99,   0.05,  n),
        akelvin    = rng.uniform(1.0e-7,  3.0e-7, n),
        akelvini   = rng.uniform(1.0e-7,  3.0e-7, n),
        pconmax    = 10.0 ** rng.uniform(-2.0,    4.0,  n),
    )


def run_fortran(binary, scen, work_dir, base_T=210.0, base_p_hPa=200.0,
                base_rh=0.95, base_n=100.0, base_mu_cm=2.5e-6, base_sig=1.5):
    """Invoke Fortran diagnostic for one scenario, return rnuclg[:].

    The base fields drive the nuc2test column initialisation; they only
    need to be self-consistent enough that CARMASTATE_Step doesn't crash.
    The override fields are what actually drive the Möhler probe.
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
    res = subprocess.run([str(binary), str(scen_path), str(out_dir), "1"],
                         check=True, capture_output=True)
    arr = np.fromfile(out_dir / "substep_0001_freezaerl_mohler_probe.bin",
                       dtype=np.float64).reshape((NBIN, NGROUP, NGROUP), order='F')
    # Möhler nucleates Sulfate IN (group 1, idx 0) → Ice Crystal (group 2, idx 1).
    return arr[:, 0, 1]


def run_jax(scen, r_bins, vol_bins):
    return np.asarray(freezaerl_mohler2010(
        t_val=jnp.float64(scen['T']),
        supsati_val=jnp.float64(scen['ssi']),
        supsatl_val=jnp.float64(scen['ssl']),
        akelvin_val=jnp.float64(scen['akelvin']),
        akelvini_val=jnp.float64(scen['akelvini']),
        r_bins=jnp.asarray(r_bins),
        vol_bins=jnp.asarray(vol_bins),
        rhosol_val=jnp.float64(RHO_SOL),  # SOLUTE density, not particle
        pconmax_val=jnp.float64(scen['pconmax']),
    ))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=200,
                    help="number of scenarios; smaller default than standalone "
                         "since each scenario invokes a full CARMA Step")
    p.add_argument("--binary", type=Path, default=DEFAULT_BIN)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.binary.exists():
        print(f"ERROR: build with apply_nuc2_diagnostic_patch.sh first",
              file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    r_bins, vol_bins = bin_grid()
    scens = sample_scenarios(args.n, rng)

    rnuclg_F = np.zeros((args.n, NBIN))
    rnuclg_J = np.zeros((args.n, NBIN))
    print(f"Running {args.n} scenarios through Fortran-dispatch + JAX...",
          flush=True)
    with tempfile.TemporaryDirectory(prefix="phase114_") as work_root:
        work_root = Path(work_root)
        for i in range(args.n):
            scen = {k: float(v[i]) for k, v in scens.items()}
            work_dir = work_root / f"s{i:05d}"
            work_dir.mkdir()
            rnuclg_F[i] = run_fortran(args.binary, scen, work_dir)
            rnuclg_J[i] = run_jax(scen, r_bins, vol_bins)
            shutil.rmtree(work_dir, ignore_errors=True)
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{args.n}", flush=True)

    np.savez_compressed(
        ROOT / "data" / "freezaerl_mohler2010_dispatch_bench.npz",
        rnuclg_F=rnuclg_F, rnuclg_J=rnuclg_J,
        r_bins=r_bins, vol_bins=vol_bins, **scens)

    denom = np.maximum(np.abs(rnuclg_F), np.abs(rnuclg_J))
    denom = np.where(denom > 1e-300, denom, 1.0)
    rel = np.abs(rnuclg_F - rnuclg_J) / denom
    nonzero = rnuclg_F > 0
    print(f"\nrel err over {args.n} scenarios × {NBIN} bins"
          f" ({nonzero.sum()} non-zero entries):")
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
    ax.set_xlabel("Fortran rnuclg [s⁻¹]   (upstream subroutine, full dispatch)")
    ax.set_ylabel("JAX rnuclg [s⁻¹]")
    ax.set_title(f"Phase 11.4: Möhler 2010 — JAX vs Fortran-in-dispatch "
                 f"({args.n} scen × {NBIN} bins, {int(nonzero.sum())} nonzero)")
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
    ax2.set_title(f"Per-bin rel-err — dot = one of {args.n} scenarios, "
                  f"box = cross-scenario P25/median/P75")
    ax2.grid(True, alpha=0.3, which="both"); ax2.legend(loc="lower left", fontsize=9)

    out_dir = ROOT / "plots" / "diff" / "phase11"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "freezaerl_mohler2010_dispatch_jax_vs_fortran.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
