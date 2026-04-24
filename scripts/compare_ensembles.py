"""Phase 10.4 parity comparison — JAX vs Fortran ensemble outputs.

Loads the NPZs produced by ``scripts/jax_ensemble.py`` and
``scripts/fortran_orchestrator.py`` (both adhere to the same
schema: ``T_final``, ``gc_h2so4_final``, ``pc_final``,
``nstep_ran``) and produces:

1. Per-scenario relative-error summaries for the three
   scalar outputs (T_final, gc_h2so4_final) and the vector output
   (pc_final, reduced via median-over-bins).
2. Error CDF figures.
3. A gate-check table summarizing ≤1%, ≤5%, and ≤10% median-error
   pass rates, per the Phase 10 plan.
4. A worst-case list (top 10 scenarios by error) linking back to
   the scenario parameters that drove them.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))
from generate_sulfate_scenarios import load_scenarios


OUTDIR = _ROOT / "plots" / "phase10_parity"


def _rel_err(a, b, eps=1e-30):
    """Relative error |a - b| / max(|a|, |b|, eps)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), eps)
    return np.abs(a - b) / denom


def _pc_median_err(pc_jax, pc_fort):
    """Per-scenario median bin-wise relative error on pc_final."""
    rel = _rel_err(pc_jax, pc_fort)       # (n, nbin)
    return np.median(rel, axis=1)


def _summarize(arr, label):
    p50 = float(np.percentile(arr, 50))
    p90 = float(np.percentile(arr, 90))
    p99 = float(np.percentile(arr, 99))
    mx = float(np.max(arr))
    pass_1 = float(np.mean(arr <= 0.01) * 100)
    pass_5 = float(np.mean(arr <= 0.05) * 100)
    pass_10 = float(np.mean(arr <= 0.10) * 100)
    print(f"  {label:25s}: p50 {p50:.3e}  p90 {p90:.3e}  p99 {p99:.3e}"
          f"  max {mx:.3e}")
    print(f"  {'  pass rates':25s}: ≤1% {pass_1:5.1f}%   "
          f"≤5% {pass_5:5.1f}%   ≤10% {pass_10:5.1f}%")
    return dict(p50=p50, p90=p90, p99=p99, max=mx,
                pass_1=pass_1, pass_5=pass_5, pass_10=pass_10)


def _plot_cdf(errors_dict, outpath):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, err in errors_dict.items():
        x = np.sort(err)
        y = np.arange(1, len(x) + 1) / len(x) * 100
        ax.plot(x, y, lw=2, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("relative error")
    ax.set_ylabel("cumulative % of scenarios ≤ error")
    ax.axvline(0.01, ls="--", color="gray", alpha=0.6, label="1% gate")
    ax.axvline(0.05, ls=":",  color="gray", alpha=0.6, label="5% gate")
    ax.set_xlim(1e-10, 10.0)
    ax.set_ylim(0, 100)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("JAX vs Fortran — relative-error CDFs (1000 scenarios)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jax", type=Path,
                        default=Path("data/sulfate_jax_outputs.npz"))
    parser.add_argument("--fortran", type=Path,
                        default=Path("data/sulfate_fortran_outputs.npz"))
    parser.add_argument("--scenarios", type=Path,
                        default=Path("data/sulfate_scenarios_1000.npz"))
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)

    jax = np.load(args.jax)
    fort = np.load(args.fortran)
    scen = load_scenarios(args.scenarios)
    n_j = jax["T_final"].shape[0]
    n_f = fort["T_final"].shape[0]
    n = min(n_j, n_f)
    if n_j != n_f:
        print(f"WARNING: scenario counts differ ({n_j} vs {n_f}); "
              f"comparing first {n}")

    # --- compute errors ---
    err_T   = _rel_err(jax["T_final"][:n], fort["T_final"][:n])
    err_gc  = _rel_err(jax["gc_h2so4_final"][:n], fort["gc_h2so4_final"][:n])
    err_pc  = _pc_median_err(jax["pc_final"][:n], fort["pc_final"][:n])

    print(f"\n=== Phase 10.4 parity: JAX vs Fortran (n = {n} scenarios) ===\n")
    s_T  = _summarize(err_T,  "T_final")
    s_gc = _summarize(err_gc, "gc_h2so4_final")
    s_pc = _summarize(err_pc, "pc_final (median-bin)")

    # --- gate ---
    # Plan gate: ≥95% of scenarios ≤ 1% median; no scenario > 5%.
    pass_gate = (s_pc["pass_1"] >= 95.0) and (s_pc["max"] <= 0.05)
    print(f"\n=== Gate check (pc_final median-bin error) ===")
    print(f"  ≥95% ≤1%: {'PASS' if s_pc['pass_1'] >= 95 else 'FAIL'} "
          f"({s_pc['pass_1']:.1f}%)")
    print(f"  max ≤5%:  {'PASS' if s_pc['max'] <= 0.05 else 'FAIL'} "
          f"(max = {s_pc['max']:.3e})")
    print(f"  overall:  {'PASS' if pass_gate else 'FAIL'}")

    # --- worst-case ---
    print(f"\n=== Top 10 worst-case scenarios (by pc median-bin error) ===")
    worst = np.argsort(err_pc)[-10:][::-1]
    print(f"  {'idx':>5s}  {'err':>10s}  "
          f"{'T [K]':>7s} {'p [hPa]':>9s} {'RH':>5s}"
          f" {'H2SO4':>10s} {'μ [nm]':>8s} {'σg':>5s}")
    for idx in worst:
        print(f"  {idx:5d}  {err_pc[idx]:10.3e}  "
              f"{scen['T'][idx]:7.1f} {scen['p'][idx]:9.2f} "
              f"{scen['rh'][idx]:5.2f} {scen['h2so4_pptv'][idx]:10.3e} "
              f"{scen['aerosol_mu_nm'][idx]:8.2f} "
              f"{scen['aerosol_sigma_g'][idx]:5.2f}")

    # --- plot CDFs ---
    _plot_cdf(
        {"T_final": err_T,
         "gc_h2so4_final": err_gc,
         "pc_final (median-bin)": err_pc},
        OUTDIR / "fig1_error_cdfs.png",
    )
    print(f"\nFigures saved to {OUTDIR}")

    # --- save summary NPZ for downstream ---
    np.savez_compressed(
        OUTDIR.parent.parent / "data" / "sulfate_parity_summary.npz",
        err_T=err_T, err_gc=err_gc, err_pc=err_pc,
    )


if __name__ == "__main__":
    main()
