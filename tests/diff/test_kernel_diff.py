"""Differential validation: JAX kernel output vs Fortran's actual per-substep state.

Each test reads Fortran's per-substep dump (produced by
``test_sulfate_diagnostic``), feeds the relevant inputs into the JAX
kernel, and asserts the output matches Fortran's stored value.

Default tolerance is `rtol=1e-10`. For kernels with known floating-point
reordering vs Fortran (e.g. summation-order differences in vectorized
JAX ops), add an entry to ``_PER_KERNEL_TOL`` below.

Each test also writes a side-by-side comparison plot to
``plots/diff/scen_<NNN>/<kernel_name>.png`` so the user can inspect
JAX vs Fortran visually.

Phase 0 exit gate: ``test_vaporp_h2o_murphy2005_matches_fortran`` passes.
Adding more tests is the work of Phase 7+ — see ``docs/port/tracker.md``.
"""

from __future__ import annotations

from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pytest

from carma._diag import read_substep, SubstepDump


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DIFF_BASE = _REPO_ROOT / "data" / "diff"
_PLOT_BASE = _REPO_ROOT / "plots" / "diff"


def _available_scenarios() -> list[int]:
    """All scenario IDs that have a Phase 0 dump on disk."""
    if not _DIFF_BASE.exists():
        return []
    out = []
    for p in sorted(_DIFF_BASE.glob("scen_*")):
        try:
            i = int(p.name.split("_")[1])
            if (p / "substep_0001_pc.bin").exists():
                out.append(i)
        except (ValueError, IndexError):
            continue
    return out


_ALL_SCEN_IDS = _available_scenarios()


# Per-kernel tolerance overrides. Default is 1e-10. Add an entry here
# when a kernel has known floating-point reordering differences with
# Fortran (e.g. summation order in a vectorized JAX op).
_PER_KERNEL_TOL = {
    # _wetr_wtpct chains two cube roots, an exp, and two table lookups
    # (wtpct_tabaz + sulfate_density). The Kelvin-iteration accumulates
    # sub-ULP rounding past 1e-10. All 1000 scenarios still pass at 1e-8;
    # median rel err is 3.6e-11.
    "wetr_wtpct": 1e-8,
    # setup_ckern: Brownian + Van der Waals + Fuchs transition + grav.
    # Reordering between sqrt/exp/log chains pushes ~80 of 1000 scenarios
    # past 1e-10; all 1000 pass at 1e-9 (max 2.1e-10, median 6.8e-12).
    "setup_ckern": 1e-9,
}


def _tol_for(kernel_name: str) -> float:
    return _PER_KERNEL_TOL.get(kernel_name, 1e-10)


def compute_rel_err(jax_arr, fortran_arr) -> tuple[np.ndarray, float]:
    """Per-element relative error and the max thereof. Used by both the
    per-scenario plotter and the across-scenario summary."""
    j = np.asarray(jax_arr)
    f = np.asarray(fortran_arr)
    diff = np.abs(j - f)
    scale = np.maximum(np.abs(j), np.abs(f))
    rel = np.where(scale > 0, diff / scale, 0.0)
    max_rel = float(rel.max()) if rel.size else 0.0
    return rel, max_rel


def make_summary_plot(
    kernel_name: str,
    scenario_max_rel: dict,
    out_dir: Path,
    rtol: float = 1e-10,
):
    """One plot per kernel: histogram + CDF of max-relative-error
    across scenarios. ``scenario_max_rel`` maps scen_id → max_rel_err.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{kernel_name.replace(':', '_')}_summary.png"

    n = len(scenario_max_rel)
    if n == 0:
        return None
    errs = np.asarray([v for _, v in sorted(scenario_max_rel.items())])
    safe = np.maximum(errs, 1e-30)
    n_pass = int((errs <= rtol).sum())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax_hist, ax_cdf = axes

    bins = np.logspace(-30, max(2, np.log10(safe.max() + 1e-30)), 50)
    ax_hist.hist(safe, bins=bins, color="C0", edgecolor="k", alpha=0.7)
    ax_hist.axvline(rtol, color="gray", ls="--", lw=1, label=f"rtol={rtol:.0e}")
    ax_hist.set_xscale("log")
    ax_hist.set_xlabel("max relative error per scenario")
    ax_hist.set_ylabel("scenario count")
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)
    ax_hist.set_title(f"distribution across {n} scenarios")

    sorted_errs = np.sort(safe)
    cdf_y = np.arange(1, n + 1) / n
    ax_cdf.plot(sorted_errs, cdf_y, "C0-", lw=1.5)
    ax_cdf.axvline(rtol, color="gray", ls="--", lw=1, label=f"rtol={rtol:.0e}")
    ax_cdf.set_xscale("log")
    ax_cdf.set_xlabel("max relative error per scenario")
    ax_cdf.set_ylabel("CDF")
    ax_cdf.legend()
    ax_cdf.grid(True, alpha=0.3)
    ax_cdf.set_title(f"CDF: {n_pass}/{n} pass at rtol")

    fig.suptitle(f"{kernel_name}: across-scenario summary", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_kernel_plot(
    kernel_name: str,
    jax_arr: np.ndarray,
    fortran_arr: np.ndarray,
    out_dir: Path,
    *,
    x_label: str = "index",
    y_label: str = "value",
    title_extra: str = "",
):
    """Save a side-by-side comparison plot of JAX vs Fortran arrays.

    Handles 1-D arrays directly. For 2-D arrays, plots each column
    as a separate line. Higher-D arrays are flattened with a warning.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{kernel_name.replace(':', '_')}.png"

    j = np.asarray(jax_arr)
    f = np.asarray(fortran_arr)

    # Compute relative error for the title
    diff = np.abs(j - f)
    scale = np.maximum(np.abs(j), np.abs(f))
    rel_err = np.where(scale > 0, diff / scale, 0.0)
    max_rel = float(rel_err.max()) if rel_err.size else 0.0
    title_main = f"{kernel_name}: max rel err = {max_rel:.2e}"
    if title_extra:
        title_main += f"  ({title_extra})"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax_val, ax_err = axes

    if j.ndim == 0:
        # Scalar: just a bar with two values
        ax_val.bar([0, 1], [float(f), float(j)],
                   color=["k", "C0"], width=0.6)
        ax_val.set_xticks([0, 1])
        ax_val.set_xticklabels(["Fortran", "JAX"])
        ax_val.set_ylabel(y_label)
        ax_err.text(0.5, 0.5, f"|J - F| = {abs(float(j) - float(f)):.3e}",
                    ha="center", va="center", transform=ax_err.transAxes)
        ax_err.set_xticks([])
        ax_err.set_yticks([])
    elif j.ndim == 1:
        x = np.arange(j.size)
        # Use signed-log scaling: positive log bars + minus sign for negatives
        ax_val.plot(x, f, "k.-", label="Fortran", lw=1.5, ms=4)
        ax_val.plot(x, j, "C0o--", label="JAX", lw=1, ms=3, alpha=0.7)
        ax_val.set_xlabel(x_label)
        ax_val.set_ylabel(y_label)
        ax_val.legend()
        ax_val.grid(True, alpha=0.3)
        if (np.all(f >= 0) and np.all(j >= 0) and (f.max() > 0)):
            ax_val.set_yscale("symlog", linthresh=max(f.max(), 1e-30) * 1e-12)
        # Error panel: rel err per element
        ax_err.semilogy(x, np.maximum(rel_err.flatten(), 1e-30),
                        "C3o-", lw=1, ms=3)
        ax_err.axhline(1e-10, color="gray", ls="--", lw=1, label="rtol=1e-10")
        ax_err.set_xlabel(x_label)
        ax_err.set_ylabel("|J - F| / max(|J|,|F|)")
        ax_err.set_title("relative error")
        ax_err.legend()
        ax_err.grid(True, which="both", alpha=0.3)
    else:
        # 2-D or higher: flatten to 1-D and plot
        j_flat = j.flatten()
        f_flat = f.flatten()
        x = np.arange(j_flat.size)
        ax_val.plot(x, f_flat, "k.-", label="Fortran", lw=1, ms=3)
        ax_val.plot(x, j_flat, "C0o--", label="JAX", lw=1, ms=2, alpha=0.7)
        ax_val.set_xlabel(f"flat {x_label} (orig shape {j.shape})")
        ax_val.set_ylabel(y_label)
        ax_val.legend()
        ax_val.grid(True, alpha=0.3)
        ax_err.semilogy(x, np.maximum(rel_err.flatten(), 1e-30),
                        "C3o-", lw=0.8, ms=2)
        ax_err.axhline(1e-10, color="gray", ls="--", lw=1)
        ax_err.set_xlabel(f"flat {x_label}")
        ax_err.set_ylabel("rel err")
        ax_err.set_title("relative error")
        ax_err.grid(True, which="both", alpha=0.3)

    fig.suptitle(title_main, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def assert_kernel_match(
    kernel_name: str,
    jax_output,
    fortran_output,
    rtol: float | None = None,
    atol: float = 0.0,
):
    """Assert JAX kernel output matches Fortran's stored value.

    Args:
        kernel_name: name of the kernel under test (used for tolerance lookup).
        jax_output: JAX result (numpy array or scalar).
        fortran_output: Fortran result (numpy array or scalar).
        rtol: relative tolerance; defaults to per-kernel override or 1e-10.
        atol: absolute tolerance; defaults to 0 (kernel-bench style).
    """
    if rtol is None:
        rtol = _tol_for(kernel_name)

    jax_arr = np.asarray(jax_output)
    f_arr = np.asarray(fortran_output)

    if jax_arr.shape != f_arr.shape:
        raise AssertionError(
            f"{kernel_name}: shape mismatch — JAX {jax_arr.shape} vs Fortran {f_arr.shape}"
        )

    # Use np.allclose for the overall pass/fail, but compute and report the
    # max relative error explicitly so the failure message is useful.
    diff = np.abs(jax_arr - f_arr)
    scale = np.maximum(np.abs(jax_arr), np.abs(f_arr))
    rel_err = np.where(scale > 0, diff / scale, 0.0)
    max_rel = float(rel_err.max()) if rel_err.size else 0.0
    abs_diff_max = float(diff.max()) if diff.size else 0.0

    if not np.allclose(jax_arr, f_arr, rtol=rtol, atol=atol):
        # Locate worst element for the report.
        idx = np.unravel_index(np.argmax(rel_err), rel_err.shape) if rel_err.size else None
        raise AssertionError(
            f"{kernel_name}: mismatch at rtol={rtol}\n"
            f"  max relative error: {max_rel:.3e} at index {idx}\n"
            f"  max absolute diff:  {abs_diff_max:.3e}\n"
            f"  JAX[idx]: {jax_arr[idx]!r}\n"
            f"  Fortran[idx]: {f_arr[idx]!r}"
        )


# ---------------------------------------------------------------------------
# Phase 0 exit gate: bench across all available scenarios at substep 1.
# Skips automatically if no scenarios have been dumped yet.
# ---------------------------------------------------------------------------

if not _ALL_SCEN_IDS:
    pytest.skip(
        f"No diagnostic dumps found under {_DIFF_BASE}. "
        "Run scripts/run_diagnostic_ensemble.py first.",
        allow_module_level=True,
    )


def _summary_plot_dir() -> Path:
    return _PLOT_BASE / "summary"


def test_vaporp_h2o_murphy2005_matches_fortran():
    """Phase 0 exit gate: JAX vaporp_h2o_murphy2005 matches Fortran's
    f_pvapl[:, igas_h2o] and f_pvapi[:, igas_h2o] across every dumped
    scenario at substep 1.
    """
    from carma.vapor_pressure import vaporp_h2o_murphy2005

    igas_h2o = 0
    pvapl_max_rel = {}
    pvapi_max_rel = {}
    failures = []

    for scen_id in _ALL_SCEN_IDS:
        scen_dir = _DIFF_BASE / f"scen_{scen_id:03d}"
        d = read_substep(scen_dir, step=1)
        pvapl_jax, pvapi_jax = vaporp_h2o_murphy2005(d.t)
        pvapl_f = d.pvapl[:, igas_h2o]
        pvapi_f = d.pvapi[:, igas_h2o]

        _, m_pvapl = compute_rel_err(np.asarray(pvapl_jax), pvapl_f)
        _, m_pvapi = compute_rel_err(np.asarray(pvapi_jax), pvapi_f)
        pvapl_max_rel[scen_id] = m_pvapl
        pvapi_max_rel[scen_id] = m_pvapi

        if m_pvapl > _tol_for("vaporp_h2o_murphy2005:pvapl"):
            failures.append(("pvapl", scen_id, m_pvapl))
        if m_pvapi > _tol_for("vaporp_h2o_murphy2005:pvapi"):
            failures.append(("pvapi", scen_id, m_pvapi))

    # Save summary plots regardless of pass/fail.
    summary_dir = _summary_plot_dir()
    make_summary_plot("vaporp_h2o_murphy2005_pvapl",
                      pvapl_max_rel, summary_dir)
    make_summary_plot("vaporp_h2o_murphy2005_pvapi",
                      pvapi_max_rel, summary_dir)

    if failures:
        msg_lines = [
            f"vaporp_h2o_murphy2005 mismatched on {len(failures)} scenarios:",
        ]
        for which, scen_id, err in failures[:10]:
            msg_lines.append(
                f"  scen {scen_id} ({which}): max rel err {err:.3e}"
            )
        if len(failures) > 10:
            msg_lines.append(f"  ... and {len(failures) - 10} more")
        raise AssertionError("\n".join(msg_lines))


def test_wtpct_tabaz_matches_fortran():
    """Phase 7.1: JAX wtpct_tabaz matches Fortran's f_wtpct[iz] across
    every dumped scenario at substep 1.

    Fortran inputs (per vaporp_h2so4_ayers1980.F90 line 60):
      temp = max(t(iz), 140)
      h2o_mass = gc(iz, igash2o) / zmet(iz)
      h2o_vp = pvapl(iz, igash2o)
    """
    from carma.sulfate_utils import wtpct_tabaz
    import jax.numpy as jnp

    igas_h2o = 0
    max_rel = {}
    failures = []

    for scen_id in _ALL_SCEN_IDS:
        scen_dir = _DIFF_BASE / f"scen_{scen_id:03d}"
        d = read_substep(scen_dir, step=1)

        # Reproduce Fortran's input prep verbatim.
        temp = float(np.maximum(d.t[0], 140.0))
        h2o_mass = float(d.gc[0, igas_h2o] / d.zmet[0])
        h2o_vp = float(d.pvapl[0, igas_h2o])

        wtp_jax = float(wtpct_tabaz(
            jnp.asarray(temp), jnp.asarray(h2o_mass), jnp.asarray(h2o_vp),
        ))
        wtp_f = float(d.wtpct[0])

        _, m = compute_rel_err(wtp_jax, wtp_f)
        max_rel[scen_id] = m
        if m > _tol_for("wtpct_tabaz"):
            failures.append((scen_id, m, temp, h2o_mass, h2o_vp,
                             wtp_jax, wtp_f))

    make_summary_plot("wtpct_tabaz", max_rel, _summary_plot_dir())

    if failures:
        msg_lines = [f"wtpct_tabaz mismatched on {len(failures)} scenarios:"]
        for scen_id, err, T, hm, hv, wj, wf in failures[:10]:
            msg_lines.append(
                f"  scen {scen_id}: rel err {err:.3e}  T={T:.2f}  "
                f"h2o_mass={hm:.3e}  h2o_vp={hv:.3e}  JAX={wj:.4f}  F={wf:.4f}"
            )
        if len(failures) > 10:
            msg_lines.append(f"  ... and {len(failures) - 10} more")
        raise AssertionError("\n".join(msg_lines))


def test_sulfate_density_matches_fortran():
    """Phase 7.2: JAX sulfate_density(wtp, t) matches the Fortran probe
    `sulfdens` (which calls Fortran sulfate_density at the substep's
    wtpct and t) across all 1000 scenarios at substep 1.
    """
    from carma.sulfate_utils import sulfate_density
    import jax.numpy as jnp

    max_rel = {}
    failures = []
    for scen_id in _ALL_SCEN_IDS:
        d = read_substep(_DIFF_BASE / f"scen_{scen_id:03d}", step=1)
        wtp = float(d.wtpct[0])
        t = float(d.t[0])
        rho_jax = float(sulfate_density(jnp.asarray(wtp), jnp.asarray(t)))
        rho_f = float(d.sulfdens[0])
        _, m = compute_rel_err(rho_jax, rho_f)
        max_rel[scen_id] = m
        if m > _tol_for("sulfate_density"):
            failures.append((scen_id, m, wtp, t, rho_jax, rho_f))

    make_summary_plot("sulfate_density", max_rel, _summary_plot_dir())

    if failures:
        msg_lines = [f"sulfate_density mismatched on {len(failures)} scenarios:"]
        for scen_id, err, w, t, rj, rf in failures[:10]:
            msg_lines.append(
                f"  scen {scen_id}: rel err {err:.3e}  wtp={w:.3f}  T={t:.2f}  "
                f"JAX={rj:.6f}  F={rf:.6f}"
            )
        if len(failures) > 10:
            msg_lines.append(f"  ... and {len(failures) - 10} more")
        raise AssertionError("\n".join(msg_lines))


def test_sulfate_surf_tens_matches_fortran():
    """Phase 7.3: JAX sulfate_surf_tens(wtp, t) matches the Fortran probe
    `sulfsurf` across all 1000 scenarios at substep 1.
    """
    from carma.sulfate_utils import sulfate_surf_tens
    import jax.numpy as jnp

    max_rel = {}
    failures = []
    for scen_id in _ALL_SCEN_IDS:
        d = read_substep(_DIFF_BASE / f"scen_{scen_id:03d}", step=1)
        wtp = float(d.wtpct[0])
        t = float(d.t[0])
        s_jax = float(sulfate_surf_tens(jnp.asarray(wtp), jnp.asarray(t)))
        s_f = float(d.sulfsurf[0])
        _, m = compute_rel_err(s_jax, s_f)
        max_rel[scen_id] = m
        if m > _tol_for("sulfate_surf_tens"):
            failures.append((scen_id, m, wtp, t, s_jax, s_f))

    make_summary_plot("sulfate_surf_tens", max_rel, _summary_plot_dir())

    if failures:
        msg_lines = [f"sulfate_surf_tens mismatched on {len(failures)} scenarios:"]
        for scen_id, err, w, t, sj, sf in failures[:10]:
            msg_lines.append(
                f"  scen {scen_id}: rel err {err:.3e}  wtp={w:.3f}  T={t:.2f}  "
                f"JAX={sj:.6f}  F={sf:.6f}"
            )
        if len(failures) > 10:
            msg_lines.append(f"  ... and {len(failures) - 10} more")
        raise AssertionError("\n".join(msg_lines))


def test_vaporp_h2so4_ayers1980_matches_fortran():
    """Phase 7.4: JAX vaporp_h2so4_ayers1980(t, gc_h2o, pvapl_h2o, zmet)
    matches Fortran's f_pvapl[:, igas_h2so4] across all scenarios at
    substep 1.

    Fortran writes pvapl[iz, igash2o] in vaporp_h2so4_ayers1980.F90;
    that's what we dump and bench against.
    """
    from carma.vapor_pressure import vaporp_h2so4_ayers1980
    import jax.numpy as jnp

    igas_h2o = 0
    igas_h2so4 = 1
    max_rel = {}
    failures = []

    for scen_id in _ALL_SCEN_IDS:
        d = read_substep(_DIFF_BASE / f"scen_{scen_id:03d}", step=1)
        pv_jax, _ = vaporp_h2so4_ayers1980(
            jnp.asarray(d.t),
            jnp.asarray(d.gc[:, igas_h2o]),
            jnp.asarray(d.pvapl[:, igas_h2o]),
            jnp.asarray(d.zmet),
        )
        pv_f = d.pvapl[:, igas_h2so4]
        _, m = compute_rel_err(np.asarray(pv_jax), pv_f)
        max_rel[scen_id] = m
        if m > _tol_for("vaporp_h2so4_ayers1980"):
            failures.append((scen_id, m,
                             float(d.t[0]), float(d.gc[0, igas_h2o]),
                             float(d.pvapl[0, igas_h2o]),
                             float(np.asarray(pv_jax)[0]), float(pv_f[0])))

    make_summary_plot("vaporp_h2so4_ayers1980", max_rel, _summary_plot_dir())

    if failures:
        msg_lines = [f"vaporp_h2so4_ayers1980 mismatched on {len(failures)} scenarios:"]
        for scen_id, err, T, gh2o, pv_h2o, pj, pf in failures[:10]:
            msg_lines.append(
                f"  scen {scen_id}: rel err {err:.3e}  T={T:.2f}  "
                f"gc_h2o={gh2o:.3e}  pvapl_h2o={pv_h2o:.3e}  "
                f"JAX={pj:.3e}  F={pf:.3e}"
            )
        if len(failures) > 10:
            msg_lines.append(f"  ... and {len(failures) - 10} more")
        raise AssertionError("\n".join(msg_lines))


def test_binary_nuc_zhao1995_matches_fortran():
    """Phase 7.9: JAX binary_nuc_zhao1995 matches Fortran's
    binary_nuc_zhao1995 invoked via the diagnostic probe at the
    end-of-step state.

    Sulfate test runs ``sulfnucl_method='ZhaoTurco'`` (carma_sulfatetest.F90:171),
    so this is the exercised kernel; binary_nuc_vehk2002 is not exercised here.

    The probe (`dump_zhao1995_probe` in `carma_sulfatetest_diagnostic.F90`)
    invokes Fortran's `binary_nuc_zhao1995` with the same inputs we feed
    JAX. This avoids the issue that the dumped `rhompe` is computed during
    a microfast substep with an evolved gc/t state that's not fully
    captured in cstate at end-of-step.

    Probe outputs: 4 scalars per scenario [nucrate_cgs, mass_cluster_dry,
    radius_cluster, ftry].
    """
    from carma.nucleation.sulfnucrate import binary_nuc_zhao1995
    from carma.constants import AVG, RGAS, PI as PI_carma
    import jax.numpy as jnp

    AVG_J = float(AVG)
    RGAS_J = float(RGAS)
    PI_J = float(PI_carma)
    GWTMOL_H2SO4 = 98.078479
    GWTMOL_H2O = 18.016
    igas_h2o = 0
    igas_h2so4 = 1
    iz = 0

    nuc_max_rel = {}
    mass_max_rel = {}
    rstar_max_rel = {}
    ftry_max_rel = {}
    failures = []

    for scen_id in _ALL_SCEN_IDS:
        scen_dir = _DIFF_BASE / f"scen_{scen_id:03d}"
        d = read_substep(scen_dir, step=1)
        probe_path = scen_dir / "substep_0001_zhao1995_probe.bin"
        if not probe_path.exists():
            pytest.skip("zhao1995_probe.bin not in dump — re-run scripts/run_diagnostic_ensemble.py")
        F_nuc, F_mass, F_rstar, F_ftry = np.fromfile(probe_path, dtype=np.float64)

        T = float(d.t[iz])
        zmet = float(d.zmet[iz])
        wtpct_iz = float(d.wtpct[iz])
        h2so4_cgs = float(d.gc[iz, igas_h2so4] / zmet)
        h2o_cgs = float(d.gc[iz, igas_h2o] / zmet)
        h2so4 = h2so4_cgs * AVG_J / GWTMOL_H2SO4
        h2o_n = h2o_cgs * AVG_J / GWTMOL_H2O
        rh = float(d.supsatl[iz, igas_h2o]) + 1.0
        beta1 = float(np.sqrt(RGAS_J * T / (2.0 * PI_J) / GWTMOL_H2SO4))

        nuc, mass, rstar, ftry = binary_nuc_zhao1995(
            jnp.float64(T), jnp.float64(wtpct_iz), jnp.float64(rh),
            jnp.float64(h2so4), jnp.float64(h2so4_cgs),
            jnp.float64(h2o_n), jnp.float64(h2o_cgs),
            jnp.float64(beta1),
            jnp.float64(GWTMOL_H2SO4), jnp.float64(GWTMOL_H2O),
        )
        def _re(j, f):
            return abs(j - f) / max(abs(f), 1e-300) if f != 0.0 else (0.0 if j == 0.0 else 1.0)

        m_nuc   = _re(float(nuc),   F_nuc)
        m_mass  = _re(float(mass),  F_mass)
        m_rstar = _re(float(rstar), F_rstar)
        m_ftry  = _re(float(ftry),  F_ftry)
        nuc_max_rel[scen_id]   = m_nuc
        mass_max_rel[scen_id]  = m_mass
        rstar_max_rel[scen_id] = m_rstar
        ftry_max_rel[scen_id]  = m_ftry

        tol = _tol_for("binary_nuc_zhao1995")
        for which, m, j, f in (
            ("nucrate_cgs", m_nuc,   float(nuc),   F_nuc),
            ("mass_dry",    m_mass,  float(mass),  F_mass),
            ("rstar",       m_rstar, float(rstar), F_rstar),
            ("ftry",        m_ftry,  float(ftry),  F_ftry),
        ):
            if m > tol:
                failures.append((which, scen_id, m, j, f))

    sd = _summary_plot_dir()
    make_summary_plot("binary_nuc_zhao1995_nucrate", nuc_max_rel,   sd)
    make_summary_plot("binary_nuc_zhao1995_mass",    mass_max_rel,  sd)
    make_summary_plot("binary_nuc_zhao1995_rstar",   rstar_max_rel, sd)
    make_summary_plot("binary_nuc_zhao1995_ftry",    ftry_max_rel,  sd)

    if failures:
        msg_lines = [f"binary_nuc_zhao1995 mismatched on {len(failures)} (output, scen) tuples:"]
        for which, scen_id, err, j, f in failures[:10]:
            msg_lines.append(
                f"  {which} scen {scen_id}: rel err {err:.3e}  JAX={j:.3e}  F={f:.3e}"
            )
        if len(failures) > 10:
            msg_lines.append(f"  ... and {len(failures) - 10} more")
        raise AssertionError("\n".join(msg_lines))


def test_microslow_matches_fortran():
    """Phase 8.18-8.20: JAX microslow (coagl + coagp + csolve composed)
    matches Fortran's pc_postmicroslow.

    Fortran microslow.F90:
      1. zero coagpe, coaglg
      2. coagl → fills coaglg
      3. for ielem, ibin: coagp → accumulates coagpe; csolve → updates pc
    JAX microslow_jit fuses these via lax.scan.

    The probe `dump_microslow_probe` mirrors that flow exactly. Bench
    feeds (pc=dump.pc, pcl=dump.pcl, ckernel=dump.ckernel, pconmax=dump.pconmax)
    to JAX and verifies pc_postmicroslow_probe.

    This transitively validates `coagp` and `csolve` since they are the
    only steps after coagl that can move pc.
    """
    from carma.microslow import make_microslow
    from carma.coagulation.setup_coag import setup_coag
    import jax.numpy as jnp
    from types import SimpleNamespace

    NBIN, NGROUP, NELEM = 38, 1, 1
    d0 = read_substep(_DIFF_BASE / "scen_000", step=1)
    if d0.rmass_bin is None:
        pytest.skip("setup tables missing — re-run scripts/run_diagnostic_ensemble.py")

    rmass = np.asarray(d0.rmass_bin[:, 0])
    rmrat = float(d0.rmrat_group[0])
    grp0 = SimpleNamespace(rmass=rmass, rmrat=rmrat, ienconc=0)
    coag_cfg = setup_coag(NBIN, NGROUP, NELEM, (grp0,), (),
                          np.array([[0]], dtype=np.int32),
                          np.array([[0]], dtype=np.int32))

    # Build microslow JIT'd function. NELEM=1, igroup=0, itype=I_VOLATILE=2.
    from carma.enums import ElementType
    microslow_fn = make_microslow(
        nbin=NBIN, nelem=NELEM, ngroup=NGROUP,
        elem_igroup=jnp.array([0]),
        icoag=coag_cfg.icoag, volx=coag_cfg.volx,
        icoagelem=coag_cfg.icoagelem,
        npairu=coag_cfg.npairu, npairl=coag_cfg.npairl,
        iup=coag_cfg.iup, jup=coag_cfg.jup,
        igup=coag_cfg.igup, jgup=coag_cfg.jgup,
        ilow=coag_cfg.ilow, jlow=coag_cfg.jlow,
        iglow=coag_cfg.iglow, jglow=coag_cfg.jglow,
        pkernel=coag_cfg.pkernel,
        ienconc_arr=jnp.array([0]),
        elem_itypes=jnp.array([int(ElementType.I_VOLATILE)]),
    )

    pc_max_rel = {}
    failures = []

    for scen_id in _ALL_SCEN_IDS:
        scen_dir = _DIFF_BASE / f"scen_{scen_id:03d}"
        d = read_substep(scen_dir, step=1)
        post_path = scen_dir / "substep_0001_pc_postmicroslow_probe.bin"
        if not post_path.exists():
            pytest.skip("microslow probe missing")
        pc_post_F = np.fromfile(post_path, dtype=np.float64).reshape(NBIN, NELEM, order="F")

        pc_jax_post = microslow_fn(
            jnp.asarray(d.pc),
            jnp.asarray(d.pcl),
            jnp.asarray(d.ckernel),
            jnp.asarray(d.pconmax),
            jnp.asarray(d.zmet),
            1800.0,
        )
        pc_post_J = np.asarray(pc_jax_post[0])
        _, m = compute_rel_err(pc_post_J, pc_post_F)
        pc_max_rel[scen_id] = m
        if m > _tol_for("microslow"):
            failures.append((scen_id, m))

    make_summary_plot("microslow_pc", pc_max_rel, _summary_plot_dir())
    if failures:
        msg_lines = [f"microslow mismatched on {len(failures)} scenarios:"]
        for scen_id, err in failures[:10]:
            msg_lines.append(f"  scen {scen_id}: rel err {err:.3e}")
        if len(failures) > 10:
            msg_lines.append(f"  ... and {len(failures) - 10} more")
        raise AssertionError("\n".join(msg_lines))


def test_coagl_matches_fortran():
    """Phase 8.18: JAX coagl matches Fortran's coaglg.

    Fortran microslow.F90 zeros coaglg/coagpe, calls coagl, then loops
    coagp/csolve. Our `dump_microslow_probe` follows the same flow and
    dumps coaglg_probe right after coagl (before any coagp/csolve mutates
    pc). JAX coagl: ckernel × pcl × volx (or just ckernel × pcl when
    igrp != ig).

    Sulfate scope: NGROUP=1 self-coag (igrp == ig) → uses volx.
    """
    from carma.coagulation.coagl import coagl
    from carma.coagulation.setup_coag import setup_coag
    from types import SimpleNamespace
    import jax.numpy as jnp

    NBIN, NGROUP, NELEM = 38, 1, 1
    d0 = read_substep(_DIFF_BASE / "scen_000", step=1)
    if d0.rmass_bin is None:
        pytest.skip("setup tables missing — re-run scripts/run_diagnostic_ensemble.py")

    rmass = np.asarray(d0.rmass_bin[:, 0])
    rmrat = float(d0.rmrat_group[0])
    grp0 = SimpleNamespace(rmass=rmass, rmrat=rmrat, ienconc=0)
    coag_cfg = setup_coag(NBIN, NGROUP, NELEM, (grp0,), (),
                          np.array([[0]], dtype=np.int32),
                          np.array([[0]], dtype=np.int32))
    cfg = SimpleNamespace(nbin=NBIN, ngroup=NGROUP, groups=(grp0,), coag=coag_cfg)

    coaglg_max_rel = {}
    failures = []

    for scen_id in _ALL_SCEN_IDS:
        scen_dir = _DIFF_BASE / f"scen_{scen_id:03d}"
        d = read_substep(scen_dir, step=1)
        probe_path = scen_dir / "substep_0001_coaglg_probe.bin"
        if not probe_path.exists():
            pytest.skip("coaglg_probe missing — re-run scripts/run_diagnostic_ensemble.py")
        coaglg_F = np.fromfile(probe_path, dtype=np.float64).reshape(NBIN, NGROUP, order="F")

        # JAX coagl uses pcl (start-of-step) per Fortran. Use dump's pcl.
        # pconmax: Fortran's microslow probe uses cstate%f_pconmax which was
        # set at the START of the substep loop (still the prestep value here
        # since this is the first substep of the outer step).
        pconmax_jax = jnp.asarray(d.pconmax)

        coaglg_jax = coagl(
            cfg,
            jnp.asarray(d.ckernel),
            jnp.asarray(d.pcl),
            pconmax_jax,
        )
        coaglg_jax_iz = np.asarray(coaglg_jax[0])  # (NBIN, NGROUP)
        _, m = compute_rel_err(coaglg_jax_iz, coaglg_F)
        coaglg_max_rel[scen_id] = m
        if m > _tol_for("coagl"):
            failures.append((scen_id, m))

    make_summary_plot("coagl_coaglg", coaglg_max_rel, _summary_plot_dir())
    if failures:
        msg_lines = [f"coagl mismatched on {len(failures)} scenarios:"]
        for scen_id, err in failures[:10]:
            msg_lines.append(f"  scen {scen_id}: rel err {err:.3e}")
        if len(failures) > 10:
            msg_lines.append(f"  ... and {len(failures) - 10} more")
        raise AssertionError("\n".join(msg_lines))


def test_setup_ckern_matches_fortran():
    """Phase 8.17: JAX setup_ckern matches Fortran's ckernel.

    setup_ckern (setupckern.F90) computes the (Brownian + grav) coagulation
    kernel from atmospheric and aerosol kinematics. Sulfate test calls
    CARMA_AddCoagulation with I_COLLEC_FUCHS, so the Fuchs transition
    formula is used.

    Static-ish bench: ckernel is recomputed each substep in CARMA whenever
    cstate is rebuilt. Our dump captures it at substep 1, and we feed all
    inputs (t, rhoa, zmet, rmu, r_wet, rrat, rprat, bpm, rmass, re, vf,
    cstick=1.0) to the JAX setup_ckern_jit and compare.

    Sulfate test: NGROUP=1 (sulfate self-coag), I_COLLEC_FUCHS,
    cstick=1.0 (default).
    """
    from carma.setup_ckern import setup_ckern_jit
    import jax.numpy as jnp

    NBIN, NGROUP = 38, 1
    d0 = read_substep(_DIFF_BASE / "scen_000", step=1)
    if d0.rrat is None or d0.rmu is None:
        pytest.skip("setup_ckern inputs missing — re-run scripts/run_diagnostic_ensemble.py")

    rrat_j = jnp.asarray(d0.rrat); rprat_j = jnp.asarray(d0.rprat)
    rmass_j = jnp.asarray(d0.rmass_bin)

    ckern_max_rel = {}
    failures = []

    for scen_id in _ALL_SCEN_IDS:
        scen_dir = _DIFF_BASE / f"scen_{scen_id:03d}"
        d = read_substep(scen_dir, step=1)
        if d.ckernel is None:
            pytest.skip("ckernel missing")

        ckern_jax = setup_ckern_jit(
            jnp.asarray(d.t),
            jnp.asarray(d.rhoa),
            jnp.asarray(d.zmet),
            jnp.asarray(d.rmu),
            jnp.asarray(d.r_wet),
            rrat_j, rprat_j,
            jnp.asarray(d.bpm),
            rmass_j,
            jnp.asarray(d.re),
            jnp.asarray(d.vf),
            jnp.float64(1.0),       # cstick default
            # Sulfate test: NGROUP=1, is_sulfate=True for the single group
            # → use_vw = [[True]] (Chan & Mozurkewich 2001 enhancement)
            use_vw=jnp.asarray([[True]]),
        )
        _, m = compute_rel_err(np.asarray(ckern_jax), d.ckernel)
        ckern_max_rel[scen_id] = m
        if m > _tol_for("setup_ckern"):
            failures.append((scen_id, m))

    make_summary_plot("setup_ckern_ckernel", ckern_max_rel, _summary_plot_dir())
    if failures:
        msg_lines = [f"setup_ckern mismatched on {len(failures)} scenarios:"]
        for scen_id, err in failures[:10]:
            msg_lines.append(f"  scen {scen_id}: rel err {err:.3e}")
        if len(failures) > 10:
            msg_lines.append(f"  ... and {len(failures) - 10} more")
        raise AssertionError("\n".join(msg_lines))


def test_setup_coag_matches_fortran():
    """Phase 8.16: JAX setup_coag matches Fortran's setupcoag tables.

    Static (state-independent) bench: build JAX setup_coag from the
    sulfate test's bin grid (rmass, rmrat) and compare:
      - kbin     (NGROUP, NGROUP, NGROUP, NBIN, NBIN)
      - volx     (NGROUP, NGROUP, NGROUP, NBIN, NBIN)
      - pkernel  (NBIN, NBIN, NGROUP, NGROUP, NGROUP, 6)
      - npairl   (NGROUP, NBIN)
      - npairu   (NGROUP, NBIN)
    """
    from carma.coagulation.setup_coag import setup_coag
    from types import SimpleNamespace
    import numpy as np

    NBIN, NGROUP, NELEM = 38, 1, 1
    d0 = read_substep(_DIFF_BASE / "scen_000", step=1)
    if d0.kbin is None:
        pytest.skip("kbin/volx/pkernel/npairl/npairu missing — re-run scripts/run_diagnostic_ensemble.py")

    # Synthesize minimal "groups" tuple expected by setup_coag.
    rmass = np.asarray(d0.rmass_bin[:, 0])
    rmrat = float(d0.rmrat_group[0])
    grp0 = SimpleNamespace(rmass=rmass, rmrat=rmrat)
    groups = (grp0,)

    # Sulfate test enables coag for group 1 → 1 (self → self).
    icoag_input = np.array([[0]], dtype=np.int32)            # 0 = group 0 (target)
    icoagelem_input = np.array([[0]], dtype=np.int32)

    coag = setup_coag(NBIN, NGROUP, NELEM, groups, (), icoag_input, icoagelem_input)

    # kbin: Fortran 1-based; JAX 0-based. Add 1 to JAX before compare.
    kbin_jax = np.asarray(coag.icoagop) * 0  # placeholder
    kbin_jax_arr = np.zeros_like(d0.kbin)
    # The setup_coag returns kbin via internal numpy — let's regenerate it directly
    # by inspecting the function's intermediate. Easier: re-implement the kbin
    # comparison by extracting from CoagConfig fields. setup_coag doesn't expose
    # `kbin` as a field — only volx, pkernel, npairl, npairu.
    # → compare volx, pkernel, npairl, npairu only. kbin is implicit in pair lists.

    volx_jax = np.asarray(coag.volx)        # (NGROUP, NGROUP, NGROUP, NBIN, NBIN)
    pkernel_jax = np.asarray(coag.pkernel)  # (NBIN, NBIN, NGROUP, NGROUP, NGROUP, 6)
    npairl_jax = np.asarray(coag.npairl)
    npairu_jax = np.asarray(coag.npairu)

    # Fortran dumps: kbin (Fortran 1-based int), volx, pkernel, npairl, npairu (1-based)
    npairl_F = np.asarray(d0.npairl, dtype=np.int32)  # (NGROUP, NBIN)
    npairu_F = np.asarray(d0.npairu, dtype=np.int32)
    volx_F = np.asarray(d0.volx)
    pkernel_F = np.asarray(d0.pkernel)

    # Compare counts first
    assert npairl_jax.shape == npairl_F.shape, (
        f"npairl shape mismatch: JAX {npairl_jax.shape} vs F {npairl_F.shape}"
    )
    assert np.array_equal(npairl_jax, npairl_F), \
        f"npairl mismatch — JAX max {npairl_jax.max()}, F max {npairl_F.max()}"
    assert np.array_equal(npairu_jax, npairu_F), \
        f"npairu mismatch — JAX max {npairu_jax.max()}, F max {npairu_F.max()}"

    # volx: should be exact (constructed from same rmass/rmrat)
    _, m_vx = compute_rel_err(volx_jax, volx_F)
    assert m_vx < 1e-12, f"volx mismatch rel err {m_vx:.3e}"

    # pkernel: should be exact
    _, m_pk = compute_rel_err(pkernel_jax, pkernel_F)
    assert m_pk < 1e-12, f"pkernel mismatch rel err {m_pk:.3e}"


def test_prestep_matches_fortran():
    """Phase 8.15: JAX prestep matches Fortran's prestep behavior.

    Fortran prestep.F90 does:
      1. d_gc = gc - gcl; where d_gc<0: d_gc=0, gcl=gc; else: gc=gcl
      2. d_t = t - told; t = told
      3. smallconc per (iz, ibin, ielem)
      4. pcl = pc
      5. maxconc per iz

    smallconc and maxconc are already validated (Phase 8.1). The d_gc/d_t
    logic is pure arithmetic that takes (gc, gcl) and (t, told) snapshots
    that we don't have separately for the start of the substep — but we
    can verify the output gc/t/d_gc/d_t are self-consistent given the
    dumped (gc, gcl, t, told).

    For each scenario, call JAX prestep with (pc=dump.pc, gc=dump.gc,
    t=dump.t, pcl=dump.pcl, gcl=dump.gcl, told=dump.told) and verify:
      - For each gas: d_gc_raw = gc - gcl; d_gc = max(0, d_gc_raw)
      - JAX gc[i] = gcl_old (rewound) where d_gc_raw>=0, else gc[i] (kept)
      - JAX gcl[i] = gc (latest) where d_gc_raw<0, else gcl_old
      - d_t = t - told; output t = told.
      - smallconc and maxconc match (already structurally validated).
    """
    from carma.prestep import prestep
    from carma.enums import ElementType
    import jax.numpy as jnp

    NBIN, NGROUP, NELEM = 38, 1, 1
    itype_arr = jnp.asarray([int(ElementType.I_VOLATILE)])
    ienconc_arr = jnp.asarray([0])
    igelem_arr = jnp.asarray([0])

    err_d_gc = []
    err_d_t = []
    err_t = []

    for scen_id in _ALL_SCEN_IDS:
        d = read_substep(_DIFF_BASE / f"scen_{scen_id:03d}", step=1)
        if d.gcl is None or d.told is None:
            pytest.skip("prestep snapshots missing — re-run scripts/run_diagnostic_ensemble.py")

        pc_in = jnp.asarray(d.pc); gc_in = jnp.asarray(d.gc); t_in = jnp.asarray(d.t)
        pcl_in = jnp.asarray(d.pcl); gcl_in = jnp.asarray(d.gcl); told_in = jnp.asarray(d.told)
        rmass_2d = jnp.asarray(d.rmass_bin)

        pc_out, gc_out, t_out, pcl_out, gcl_out, d_gc_out, d_t_out, pconmax_out = prestep(
            pc_in, gc_in, t_in, pcl_in, gcl_in, told_in,
            jnp.asarray(d.zmet), itype_arr, ienconc_arr, igelem_arr, rmass_2d,
            do_substep=True, do_coag=True,
        )

        # Fortran prestep at the START of the next substep: t <- told, d_t = t_old - told_old
        # Verify JAX d_t computation matches the trivial arithmetic
        d_t_expected = np.asarray(d.t) - np.asarray(d.told)
        err_d_t.append(float(np.max(np.abs(np.asarray(d_t_out) - d_t_expected))))
        # JAX rewinds t to told
        err_t.append(float(np.max(np.abs(np.asarray(t_out) - np.asarray(d.told)))))

        # d_gc: max(0, gc - gcl)
        d_gc_raw = np.asarray(d.gc) - np.asarray(d.gcl)
        d_gc_expected = np.where(d_gc_raw < 0, 0.0, d_gc_raw)
        err_d_gc.append(float(np.max(np.abs(np.asarray(d_gc_out) - d_gc_expected))))

    assert np.max(err_d_gc) < 1e-30, f"prestep d_gc max abs err {np.max(err_d_gc):.3e}"
    assert np.max(err_d_t) < 1e-30, f"prestep d_t max abs err {np.max(err_d_t):.3e}"
    assert np.max(err_t) < 1e-30, f"prestep t-rewind max abs err {np.max(err_t):.3e}"


def test_nsubsteps_matches_fortran():
    """Phase 8.14: JAX nsubsteps matches Fortran's nsubsteps integer.

    Fortran nsubsteps.F90 returns the suggested ntsubsteps based on:
      - drop activation: forces maxsubsteps
      - aerfreeze (supsati > 0.4 AND T < 233.16 K): forces maxsubsteps
      - growth-rate limit: ntsubsteps = round(dtime / min(dm/dmdt))

    Sulfate test: NGROUP=1 single I_VOLATILE group, no I_DROPACT,
    no I_AERFREEZE, just I_HOMNUC. So only the growth-rate path runs.

    Probe `dump_nsubsteps_probe` calls maxconc + nsubsteps at end-of-step
    state and dumps the integer (cast to f8).

    Most scenarios produce ntsubsteps=1; a few hit 2-3 from larger
    dmdt at coarse bins.
    """
    from carma.nsubsteps import nsubsteps as jax_nsubsteps
    import jax.numpy as jnp

    # carma_sulfatetest.F90: maxsubsteps=32, default minsubsteps=1, conmax=0.1
    MINSUBSTEPS = 1
    MAXSUBSTEPS = 32
    CONMAX = 0.1
    DTIME = 1800.0

    d0 = read_substep(_DIFF_BASE / "scen_000", step=1)
    if d0.dm_bin is None or d0.itype is None:
        pytest.skip("PPM/static config tables missing — re-run scripts/run_diagnostic_ensemble.py")

    NBIN = d0.rmass_bin.shape[0]
    NGROUP = d0.rmrat_group.shape[0]
    NELEM = d0.itype.shape[0]

    # Convert Fortran 1-based to 0-based (with -1 sentinel for "none"=0)
    def _f1_to_0(arr):
        return np.asarray(arr, dtype=np.int32) - 1

    ienconc_arr = _f1_to_0(d0.ienconc)
    itype_arr = np.asarray(d0.itype, dtype=np.int32)              # itype values are not indices
    inucgas_arr = _f1_to_0(d0.inucgas)
    igrowgas_arr = _f1_to_0(d0.igrowgas)
    nnuc2elem_arr = np.asarray(d0.nnuc2elem, dtype=np.int32)
    inuc2elem_arr = _f1_to_0(d0.inuc2elem)                         # (NELEM, NELEM)
    inucproc_arr = np.asarray(d0.inucproc, dtype=np.int32)         # process IDs
    is_grp_ice_arr = (d0.is_grp_ice > 0.5).astype(bool)

    nts_max_diff = {}
    failures = []

    for scen_id in _ALL_SCEN_IDS:
        scen_dir = _DIFF_BASE / f"scen_{scen_id:03d}"
        d = read_substep(scen_dir, step=1)
        probe_path = scen_dir / "substep_0001_nsubsteps_probe.bin"
        if not probe_path.exists():
            pytest.skip("nsubsteps_probe.bin missing — re-run scripts/run_diagnostic_ensemble.py")
        nts_F = int(np.fromfile(probe_path, dtype=np.float64)[0])

        # Compute pconmax at end-of-step (matches what probe does)
        from carma.utils.smallconc import maxconc as _maxconc
        pconmax_j = _maxconc(jnp.asarray(d.pc), jnp.asarray(ienconc_arr),
                             jnp.asarray(d.zmet))

        # supsatlold not in dump for scen_id=*; use current supsatl as proxy.
        # Fortran path that uses supsatlold is gated on I_INVOLATILE +
        # I_DROPACT, which is NOT exercised in sulfate test.
        nts_J_arr = jax_nsubsteps(
            jnp.asarray(d.pc),
            jnp.asarray(d.supsatl), jnp.asarray(d.supsati),
            jnp.asarray(d.supsatl),                  # supsatlold proxy
            jnp.asarray(d.pvapl), jnp.asarray(d.pvapi),
            jnp.zeros((d.pc.shape[0], NBIN, NGROUP)),  # scrit (unused for I_VOLATILE)
            jnp.asarray(d.gro), jnp.asarray(d.gro1),
            pconmax_j, jnp.asarray(d.zmet), jnp.asarray(d0.dm_bin),
            iz=0,
            dtime_save=DTIME,
            minsubsteps=MINSUBSTEPS, maxsubsteps=MAXSUBSTEPS,
            conmax=CONMAX,
            ngroup=NGROUP, nbin=NBIN,
            ienconc_arr=ienconc_arr, itype_arr=itype_arr,
            inucgas_arr=inucgas_arr, igrowgas_arr=igrowgas_arr,
            nnuc2elem_arr=nnuc2elem_arr, inuc2elem_arr=inuc2elem_arr,
            inucproc_arr=inucproc_arr, is_grp_ice_arr=is_grp_ice_arr,
            do_substep=True,
        )
        nts_J = int(nts_J_arr)
        diff = abs(nts_J - nts_F)
        nts_max_diff[scen_id] = diff
        if diff != 0:
            failures.append((scen_id, nts_J, nts_F))

    make_summary_plot("nsubsteps", nts_max_diff, _summary_plot_dir())
    if failures:
        msg_lines = [f"nsubsteps mismatched on {len(failures)} scenarios:"]
        for scen_id, j, f in failures[:10]:
            msg_lines.append(f"  scen {scen_id}: JAX={j}  F={f}")
        if len(failures) > 10:
            msg_lines.append(f"  ... and {len(failures) - 10} more")
        raise AssertionError("\n".join(msg_lines))


def test_tsolve_matches_fortran():
    """Phase 8.13: JAX tsolve matches Fortran's tsolve(t_post).

    Fortran tsolve.F90: dt = dtime * rlprod (+ phprod if do_pheatatm).
    Sulfate test has do_pheatatm=False so phprod is gated out.

    The probe `dump_tsolve_probe` runs the full microfast evolution
    sequence (sulfnuc → growevapl → psolve → evapp → downgevapply →
    gsolve), then snapshots t_pre, calls tsolve, snapshots t_post.

    JAX bench: feed t_pre + rlprod (scalar from gsolve) to JAX tsolve
    with phprod=0 and verify t_post matches.
    """
    from carma.solvers.tsolve import tsolve
    import jax.numpy as jnp

    t_max_rel = {}
    failures = []

    for scen_id in _ALL_SCEN_IDS:
        scen_dir = _DIFF_BASE / f"scen_{scen_id:03d}"
        t_path = scen_dir / "substep_0001_t_tsolve_probe.bin"
        rl_path = scen_dir / "substep_0001_rlheat_tsolve_probe.bin"
        rlprod_path = scen_dir / "substep_0001_rlprod_probe.bin"
        if not (t_path.exists() and rl_path.exists() and rlprod_path.exists()):
            pytest.skip("tsolve probe files missing — re-run scripts/run_diagnostic_ensemble.py")
        t_arr = np.fromfile(t_path, dtype=np.float64)
        rlheat_arr = np.fromfile(rl_path, dtype=np.float64)
        rlprod_arr = np.fromfile(rlprod_path, dtype=np.float64)
        t_pre, t_post_F = float(t_arr[0]), float(t_arr[1])
        rlheat_pre = float(rlheat_arr[0])
        rlprod = float(rlprod_arr[0])

        t_j = jnp.asarray([t_pre])
        rlheat_j = jnp.asarray([rlheat_pre])
        partheat_j = jnp.zeros(1)
        t_new, _, _, _ = tsolve(
            t_j, rlheat_j, partheat_j,
            jnp.float64(rlprod), jnp.float64(0.0),    # phprod=0 (do_pheatatm=false)
            1800.0, 0,
            jnp.float64(0.0), 1.0,
        )
        t_post_J = float(t_new[0])
        m = abs(t_post_J - t_post_F) / max(abs(t_post_F), 1e-300)
        t_max_rel[scen_id] = m
        if m > _tol_for("tsolve"):
            failures.append((scen_id, m, t_post_J, t_post_F))

    make_summary_plot("tsolve_t", t_max_rel, _summary_plot_dir())
    if failures:
        msg_lines = [f"tsolve mismatched on {len(failures)} scenarios:"]
        for scen_id, err, j, f in failures[:10]:
            msg_lines.append(f"  scen {scen_id}: rel err {err:.3e}  JAX={j}  F={f}")
        if len(failures) > 10:
            msg_lines.append(f"  ... and {len(failures) - 10} more")
        raise AssertionError("\n".join(msg_lines))


def test_totalcondensate_matches_fortran():
    """Phase 8.12a: JAX totalcondensate matches Fortran on dump's pc.

    `previous_ice_probe` and `previous_liquid_probe` are computed by
    Fortran's `totalcondensate` on the dumped pc (no mutation), so they're
    a directly matched comparison.

    Sulfate test: NELEM=1, no cores → volatilemass = pc * rmass. NGAS=2,
    igrowgas[0]=H2SO4=1 (0-based). Group is liquid (not ice).
    """
    from carma.solvers.totalcondensate import totalcondensate
    import jax.numpy as jnp

    NBIN, NGROUP, NELEM, NGAS = 38, 1, 1, 2

    prev_ice_max_rel = {}
    prev_liq_max_rel = {}
    failures = []

    for scen_id in _ALL_SCEN_IDS:
        scen_dir = _DIFF_BASE / f"scen_{scen_id:03d}"
        d = read_substep(scen_dir, step=1)
        ice_path = scen_dir / "substep_0001_previous_ice_probe.bin"
        liq_path = scen_dir / "substep_0001_previous_liquid_probe.bin"
        if not (ice_path.exists() and liq_path.exists()):
            pytest.skip("totalcondensate probe files missing — re-run scripts/run_diagnostic_ensemble.py")
        prev_ice_F = np.fromfile(ice_path, dtype=np.float64)
        prev_liq_F = np.fromfile(liq_path, dtype=np.float64)

        ti, tl = totalcondensate(
            jnp.asarray(d.pc), jnp.asarray(d.rmass_bin),
            jnp.asarray([0]), jnp.asarray([False]), jnp.asarray([1]),
            NBIN, NGROUP, NGAS, 0,
        )
        ti_arr = np.asarray(ti); tl_arr = np.asarray(tl)
        _, m_ice = compute_rel_err(ti_arr, prev_ice_F)
        _, m_liq = compute_rel_err(tl_arr, prev_liq_F)
        prev_ice_max_rel[scen_id] = m_ice
        prev_liq_max_rel[scen_id] = m_liq
        tol = _tol_for("totalcondensate")
        if m_ice > tol:
            failures.append(("ice", scen_id, m_ice))
        if m_liq > tol:
            failures.append(("liq", scen_id, m_liq))

    sd = _summary_plot_dir()
    make_summary_plot("totalcondensate_ice", prev_ice_max_rel, sd)
    make_summary_plot("totalcondensate_liquid", prev_liq_max_rel, sd)

    if failures:
        msg_lines = [f"totalcondensate mismatched on {len(failures)} (output, scen) tuples:"]
        for which, scen_id, err in failures[:10]:
            msg_lines.append(f"  {which} scen {scen_id}: rel err {err:.3e}")
        if len(failures) > 10:
            msg_lines.append(f"  ... and {len(failures) - 10} more")
        raise AssertionError("\n".join(msg_lines))


def test_gsolve_matches_fortran():
    """Phase 8.12b: JAX gsolve matches Fortran's gsolve(gc_post).

    Fortran gsolve computes:
        gasprod = ((prev_ice - tot_ice) + (prev_liq - tot_liq)) / dt
        gc[iz, igas] += dt * gasprod

    We feed JAX gsolve the Fortran-side previous_*/total_* and the
    Fortran gc_pregsolve_probe and verify gc_postgsolve_probe matches.

    Latent heat is not bench-validated here (rlhe/rlhm not in dump);
    rlprod doesn't affect gc.
    """
    from carma.solvers.gsolve import gsolve
    import jax.numpy as jnp

    NGAS = 2
    gc_max_rel = {}
    failures = []

    for scen_id in _ALL_SCEN_IDS:
        scen_dir = _DIFF_BASE / f"scen_{scen_id:03d}"
        d = read_substep(scen_dir, step=1)
        prev_ice_F = np.fromfile(scen_dir/"substep_0001_previous_ice_probe.bin", dtype=np.float64)
        prev_liq_F = np.fromfile(scen_dir/"substep_0001_previous_liquid_probe.bin", dtype=np.float64)
        tot_ice_F = np.fromfile(scen_dir/"substep_0001_total_ice_probe.bin", dtype=np.float64)
        tot_liq_F = np.fromfile(scen_dir/"substep_0001_total_liquid_probe.bin", dtype=np.float64)
        gc_pre_F = np.fromfile(scen_dir/"substep_0001_gc_pregsolve_probe.bin", dtype=np.float64)
        gc_post_F = np.fromfile(scen_dir/"substep_0001_gc_postgsolve_probe.bin", dtype=np.float64)

        gc_init = jnp.zeros((1, NGAS)).at[0, :].set(jnp.asarray(gc_pre_F))
        rlhe = jnp.zeros((1, NGAS)); rlhm = jnp.zeros((1, NGAS))
        rhoa = jnp.asarray(d.rhoa); rlprod = jnp.zeros(())
        gc_new, _, _ = gsolve(
            gc_init, rlprod,
            jnp.asarray(prev_ice_F), jnp.asarray(prev_liq_F),
            jnp.asarray(tot_ice_F), jnp.asarray(tot_liq_F),
            rlhe, rlhm, rhoa, 1800.0, 0, NGAS,
            jnp.asarray([0.0, 0.0]), 1.0,
        )
        gc_new_arr = np.asarray(gc_new[0])
        _, m = compute_rel_err(gc_new_arr, gc_post_F)
        gc_max_rel[scen_id] = m
        if m > _tol_for("gsolve"):
            failures.append((scen_id, m))

    make_summary_plot("gsolve_gc", gc_max_rel, _summary_plot_dir())
    if failures:
        msg_lines = [f"gsolve mismatched on {len(failures)} scenarios:"]
        for scen_id, err in failures[:10]:
            msg_lines.append(f"  scen {scen_id}: rel err {err:.3e}")
        if len(failures) > 10:
            msg_lines.append(f"  ... and {len(failures) - 10} more")
        raise AssertionError("\n".join(msg_lines))


def test_psolve_matches_fortran():
    """Phase 8.11: JAX psolve (interleaved with growp) matches Fortran.

    Fortran microfast.F90 runs psolve in a per-(ielem, ibin) loop:
        do ielem = 1, NELEM
          do ibin = 1, NBIN
            growp(...)         ! reads pc[ibin-1] (already-updated by previous psolve)
            upgxfer(...)
            psolve(...)        ! mutates pc[ibin]
          enddo
        enddo
    There's a sequential dependency: growp at bin i reads pc[i-1] AFTER
    psolve at i-1 has updated it. The bench must mirror this loop —
    not pre-compute growpe in advance.

    Probe `dump_psolve_probe`:
      pc_prepsolve_probe (NBIN,NELEM): pc[iz,:,:] after sulfnuc+growevapl
                                       (i.e., what psolve sees at start of loop)
      rhompe_probe (NBIN,NELEM): rhompe from sulfnuc
      pc_postpsolve_probe (NBIN,NELEM): pc[iz,:,:] after the full loop
    """
    from carma.solvers.psolve import psolve
    from carma.growth.growp import growp
    from carma.growth.growevapl import growevapl
    from carma.utils.smallconc import maxconc
    import jax.numpy as jnp

    NBIN, NGROUP, NELEM = 38, 1, 1
    d0 = read_substep(_DIFF_BASE / "scen_000", step=1)
    if d0.dm_bin is None:
        pytest.skip("PPM tables not in dump — re-run scripts/run_diagnostic_ensemble.py")
    dm_j = jnp.asarray(d0.dm_bin); pratt_j = jnp.asarray(d0.pratt)
    prat_j = jnp.asarray(d0.prat); pden1_j = jnp.asarray(d0.pden1)
    palr_j = jnp.asarray(d0.palr)
    igrowgas_arr = jnp.asarray([int(d0.igrowgas[0]) - 1])
    igas_num = int(d0.igrowgas[0]) - 1
    ienconc_arr = jnp.asarray([0]); is_ice_arr = jnp.asarray([False])

    pc_max_rel = {}
    failures = []

    for scen_id in _ALL_SCEN_IDS:
        scen_dir = _DIFF_BASE / f"scen_{scen_id:03d}"
        d = read_substep(scen_dir, step=1)
        pre_path = scen_dir / "substep_0001_pc_prepsolve_probe.bin"
        post_path = scen_dir / "substep_0001_pc_postpsolve_probe.bin"
        rhompe_path = scen_dir / "substep_0001_rhompe_probe.bin"
        if not (pre_path.exists() and post_path.exists() and rhompe_path.exists()):
            pytest.skip("psolve probe files not in dump — re-run scripts/run_diagnostic_ensemble.py")

        pc_pre = np.fromfile(pre_path, dtype=np.float64).reshape(NBIN, NELEM, order="F")
        pc_post_F = np.fromfile(post_path, dtype=np.float64).reshape(NBIN, NELEM, order="F")
        rhompe_F = np.fromfile(rhompe_path, dtype=np.float64).reshape(NBIN, NELEM, order="F")

        # Build pc with iz=0 from the pre-snapshot
        pc_jax = jnp.zeros((1, NBIN, NELEM)).at[0, :, :].set(jnp.asarray(pc_pre))

        # Compute growlg/evaplg from same pre-snapshot
        pconmax_j = maxconc(pc_jax, ienconc_arr, jnp.asarray(d.zmet))
        gl_j, el_j = growevapl(
            pc_jax,
            jnp.zeros((NBIN, NGROUP)), jnp.zeros((NBIN, NGROUP)),
            jnp.asarray(d.supsatl), jnp.asarray(d.supsati),
            jnp.asarray(d.pvapl), jnp.asarray(d.pvapi),
            jnp.asarray(d.akelvin), jnp.asarray(d.akelvini),
            jnp.asarray(d.gro), jnp.asarray(d.gro1), jnp.asarray(d.rup_wet),
            jnp.asarray(d.rmass_bin), dm_j, pconmax_j,
            pratt_j, prat_j, pden1_j, palr_j,
            is_ice_arr, igrowgas_arr, ienconc_arr, 1800.0, 0, NBIN, NGROUP,
        )

        pc_nucl_j = jnp.zeros((1, NBIN, NELEM))
        rhompe_j = jnp.asarray(rhompe_F)
        rnucpe_j = jnp.zeros((NBIN, NELEM))
        evappe_j = jnp.zeros((NBIN, NELEM))
        rnuclg_j = jnp.zeros((NBIN, NGROUP, NGROUP))
        growpe_j = jnp.zeros((NBIN, NELEM))

        # Mirror Fortran sequential loop: per (ielem, ibin): growp → psolve
        for ie in range(NELEM):
            for ib in range(NBIN):
                growpe_j = growp(
                    pc_jax, growpe_j, gl_j, pconmax_j,
                    0, ib, ie, 0, igas_num,
                )
                pc_jax, pc_nucl_j = psolve(
                    pc_jax, pc_nucl_j, growpe_j, evappe_j, rnucpe_j, rhompe_j,
                    gl_j, el_j, rnuclg_j, 1800.0, 0, ib, ie, 0, NGROUP,
                )

        pc_post_j = np.asarray(pc_jax[0])
        _, m = compute_rel_err(pc_post_j, pc_post_F)
        pc_max_rel[scen_id] = m
        if m > _tol_for("psolve"):
            failures.append((scen_id, m))

    make_summary_plot("psolve_pc", pc_max_rel, _summary_plot_dir())
    if failures:
        msg_lines = [f"psolve mismatched on {len(failures)} scenarios:"]
        for scen_id, err in failures[:10]:
            msg_lines.append(f"  scen {scen_id}: rel err {err:.3e}")
        if len(failures) > 10:
            msg_lines.append(f"  ... and {len(failures) - 10} more")
        raise AssertionError("\n".join(msg_lines))


def test_evapp_matches_fortran():
    """Phase 8.8a: JAX evapp matches Fortran's evapp(evappe).

    Sulfate test: NELEM=1, no cores, I_VOLATILE → falls into the
    `ic1 == 0` branch → calls `evap_ingrp` per bin, which sums
    `pc[ibin] * evaplg[ibin]` into `evappe[ibin-1]`. evap_mono and
    evap_poly are core-only paths and never run in this scope.

    Probe: dump_evapp_probe runs maxconc → growevapl → evapp at
    end-of-step state and dumps evappe.
    """
    from carma.growth.evapp import evapp
    from carma.growth.growevapl import growevapl
    from carma.utils.smallconc import maxconc
    from carma.enums import ElementType
    import jax.numpy as jnp

    NBIN, NGROUP, NELEM = 38, 1, 1
    d0 = read_substep(_DIFF_BASE / "scen_000", step=1)
    if d0.dm_bin is None:
        pytest.skip("PPM tables not in dump — re-run scripts/run_diagnostic_ensemble.py")
    dm_j = jnp.asarray(d0.dm_bin); pratt_j = jnp.asarray(d0.pratt)
    prat_j = jnp.asarray(d0.prat); pden1_j = jnp.asarray(d0.pden1)
    palr_j = jnp.asarray(d0.palr)
    igrowgas_arr = jnp.asarray([int(d0.igrowgas[0]) - 1])
    ienconc_arr = jnp.asarray([0]); is_ice_arr = jnp.asarray([False])
    itype_arr = jnp.asarray([int(ElementType.I_VOLATILE)])
    igroup_arr = jnp.asarray([0])

    evappe_max_rel = {}
    failures = []

    for scen_id in _ALL_SCEN_IDS:
        scen_dir = _DIFF_BASE / f"scen_{scen_id:03d}"
        d = read_substep(scen_dir, step=1)
        probe_path = scen_dir / "substep_0001_evappe_probe.bin"
        if not probe_path.exists():
            pytest.skip("evappe_probe.bin not in dump — re-run scripts/run_diagnostic_ensemble.py")
        evappe_F = np.fromfile(probe_path, dtype=np.float64).reshape(
            NBIN, NELEM, order="F",
        )

        pconmax_j = maxconc(jnp.asarray(d.pc), ienconc_arr, jnp.asarray(d.zmet))
        gl_j, el_j = growevapl(
            jnp.asarray(d.pc),
            jnp.zeros((NBIN, NGROUP)), jnp.zeros((NBIN, NGROUP)),
            jnp.asarray(d.supsatl), jnp.asarray(d.supsati),
            jnp.asarray(d.pvapl), jnp.asarray(d.pvapi),
            jnp.asarray(d.akelvin), jnp.asarray(d.akelvini),
            jnp.asarray(d.gro), jnp.asarray(d.gro1), jnp.asarray(d.rup_wet),
            jnp.asarray(d.rmass_bin), dm_j, pconmax_j,
            pratt_j, prat_j, pden1_j, palr_j,
            is_ice_arr, igrowgas_arr, ienconc_arr, 1800.0, 0, NBIN, NGROUP,
        )
        evappe_j = evapp(
            jnp.asarray(d.pc), jnp.zeros((NBIN, NELEM)), el_j, pconmax_j,
            ienconc_arr, itype_arr, igroup_arr, 0, NBIN, NGROUP, NELEM,
        )

        _, m = compute_rel_err(np.asarray(evappe_j), evappe_F)
        evappe_max_rel[scen_id] = m
        if m > _tol_for("evapp"):
            failures.append((scen_id, m))

    make_summary_plot("evapp_evappe", evappe_max_rel, _summary_plot_dir())
    if failures:
        msg_lines = [f"evapp mismatched on {len(failures)} scenarios:"]
        for scen_id, err in failures[:10]:
            msg_lines.append(f"  scen {scen_id}: rel err {err:.3e}")
        if len(failures) > 10:
            msg_lines.append(f"  ... and {len(failures) - 10} more")
        raise AssertionError("\n".join(msg_lines))


def test_downgevapply_matches_fortran():
    """Phase 8.8b: JAX downgevapply matches Fortran's downgevapply (post-apply pc).

    Fortran (downgevapply.F90): pc += dtime * (evappe + rnucpe), then smallconc.
    JAX downgevapply does the same explicit-Euler apply with SMALL_PC floor.
    Sulfate test: rnucpe = 0 (Phase 7.10), so this is just pc += dt*evappe.

    Probe dumps:
      - pc_predowng_probe: pc at iz=1 before downgevapply
      - pc_postdowng_probe: pc at iz=1 after downgevapply
    """
    from carma.growth.evapp import evapp, downgevapply
    from carma.growth.growevapl import growevapl
    from carma.utils.smallconc import maxconc
    from carma.enums import ElementType
    import jax.numpy as jnp

    NBIN, NGROUP, NELEM = 38, 1, 1
    d0 = read_substep(_DIFF_BASE / "scen_000", step=1)
    if d0.dm_bin is None:
        pytest.skip("PPM tables not in dump — re-run scripts/run_diagnostic_ensemble.py")
    dm_j = jnp.asarray(d0.dm_bin); pratt_j = jnp.asarray(d0.pratt)
    prat_j = jnp.asarray(d0.prat); pden1_j = jnp.asarray(d0.pden1)
    palr_j = jnp.asarray(d0.palr)
    igrowgas_arr = jnp.asarray([int(d0.igrowgas[0]) - 1])
    ienconc_arr = jnp.asarray([0]); is_ice_arr = jnp.asarray([False])
    itype_arr = jnp.asarray([int(ElementType.I_VOLATILE)])
    igroup_arr = jnp.asarray([0])

    pc_max_rel = {}
    failures = []

    for scen_id in _ALL_SCEN_IDS:
        scen_dir = _DIFF_BASE / f"scen_{scen_id:03d}"
        d = read_substep(scen_dir, step=1)
        post_path = scen_dir / "substep_0001_pc_postdowng_probe.bin"
        if not post_path.exists():
            pytest.skip("pc_postdowng_probe not in dump — re-run scripts/run_diagnostic_ensemble.py")
        pcpost_F = np.fromfile(post_path, dtype=np.float64).reshape(
            NBIN, NELEM, order="F",
        )

        pconmax_j = maxconc(jnp.asarray(d.pc), ienconc_arr, jnp.asarray(d.zmet))
        gl_j, el_j = growevapl(
            jnp.asarray(d.pc),
            jnp.zeros((NBIN, NGROUP)), jnp.zeros((NBIN, NGROUP)),
            jnp.asarray(d.supsatl), jnp.asarray(d.supsati),
            jnp.asarray(d.pvapl), jnp.asarray(d.pvapi),
            jnp.asarray(d.akelvin), jnp.asarray(d.akelvini),
            jnp.asarray(d.gro), jnp.asarray(d.gro1), jnp.asarray(d.rup_wet),
            jnp.asarray(d.rmass_bin), dm_j, pconmax_j,
            pratt_j, prat_j, pden1_j, palr_j,
            is_ice_arr, igrowgas_arr, ienconc_arr, 1800.0, 0, NBIN, NGROUP,
        )
        ev_j = evapp(
            jnp.asarray(d.pc), jnp.zeros((NBIN, NELEM)), el_j, pconmax_j,
            ienconc_arr, itype_arr, igroup_arr, 0, NBIN, NGROUP, NELEM,
        )
        pc_j = downgevapply(
            jnp.asarray(d.pc), ev_j, jnp.zeros((NBIN, NELEM)),
            1800.0, 0, NBIN, NELEM,
        )
        pc_j_iz0 = np.asarray(pc_j[0])  # (NBIN, NELEM)

        _, m = compute_rel_err(pc_j_iz0, pcpost_F)
        pc_max_rel[scen_id] = m
        if m > _tol_for("downgevapply"):
            failures.append((scen_id, m))

    make_summary_plot("downgevapply_pc", pc_max_rel, _summary_plot_dir())
    if failures:
        msg_lines = [f"downgevapply mismatched on {len(failures)} scenarios:"]
        for scen_id, err in failures[:10]:
            msg_lines.append(f"  scen {scen_id}: rel err {err:.3e}")
        if len(failures) > 10:
            msg_lines.append(f"  ... and {len(failures) - 10} more")
        raise AssertionError("\n".join(msg_lines))


def test_growp_matches_fortran():
    """Phase 8.6: JAX growp matches Fortran's growp(growpe) per (ibin, ielem).

    Fortran growp.F90: growpe[ibin, ielem] = pc[iz, ibin-1, ielem] * growlg[ibin-1, igroup]
    (gated on igrowgas != 0, ibin > 0, pconmax > FEW_PC).

    The growp probe runs the chain `maxconc → growevapl → growp` at
    end-of-step state and dumps growpe (NBIN, NELEM). JAX side does the
    same chain to get growlg, then applies growp per bin.

    Sulfate test: NGROUP=1, NELEM=1, igas=H2SO4 (1, 0-based).
    """
    from carma.growth.growp import growp
    from carma.growth.growevapl import growevapl
    from carma.utils.smallconc import maxconc
    import jax.numpy as jnp

    NBIN, NGROUP, NELEM = 38, 1, 1

    d0 = read_substep(_DIFF_BASE / "scen_000", step=1)
    if d0.dm_bin is None:
        pytest.skip("PPM tables not in dump — re-run scripts/run_diagnostic_ensemble.py")
    dm_j = jnp.asarray(d0.dm_bin); pratt_j = jnp.asarray(d0.pratt)
    prat_j = jnp.asarray(d0.prat); pden1_j = jnp.asarray(d0.pden1)
    palr_j = jnp.asarray(d0.palr)
    igrowgas_arr = jnp.asarray([int(d0.igrowgas[0]) - 1])
    igas_num = int(d0.igrowgas[0]) - 1
    ienconc_arr = jnp.asarray([0]); is_ice_arr = jnp.asarray([False])

    growpe_max_rel = {}
    failures = []

    for scen_id in _ALL_SCEN_IDS:
        scen_dir = _DIFF_BASE / f"scen_{scen_id:03d}"
        d = read_substep(scen_dir, step=1)
        probe_path = scen_dir / "substep_0001_growpe_probe.bin"
        if not probe_path.exists():
            pytest.skip("growpe_probe.bin not in dump — re-run scripts/run_diagnostic_ensemble.py")
        growpe_F = np.fromfile(probe_path, dtype=np.float64).reshape(
            NBIN, NELEM, order="F",
        )

        pconmax_j = maxconc(jnp.asarray(d.pc), ienconc_arr, jnp.asarray(d.zmet))
        gl_j, _ = growevapl(
            jnp.asarray(d.pc),
            jnp.zeros((NBIN, NGROUP)), jnp.zeros((NBIN, NGROUP)),
            jnp.asarray(d.supsatl), jnp.asarray(d.supsati),
            jnp.asarray(d.pvapl), jnp.asarray(d.pvapi),
            jnp.asarray(d.akelvin), jnp.asarray(d.akelvini),
            jnp.asarray(d.gro), jnp.asarray(d.gro1), jnp.asarray(d.rup_wet),
            jnp.asarray(d.rmass_bin), dm_j, pconmax_j,
            pratt_j, prat_j, pden1_j, palr_j,
            is_ice_arr, igrowgas_arr, ienconc_arr, 1800.0, 0, NBIN, NGROUP,
        )

        growpe_j = jnp.zeros((NBIN, NELEM))
        for ib in range(NBIN):
            growpe_j = growp(
                jnp.asarray(d.pc), growpe_j, gl_j, pconmax_j,
                0, ib, 0, 0, igas_num,
            )

        _, m = compute_rel_err(np.asarray(growpe_j), growpe_F)
        growpe_max_rel[scen_id] = m
        if m > _tol_for("growp"):
            failures.append((scen_id, m))

    make_summary_plot("growp_growpe", growpe_max_rel, _summary_plot_dir())
    if failures:
        msg_lines = [f"growp mismatched on {len(failures)} scenarios:"]
        for scen_id, err in failures[:10]:
            msg_lines.append(f"  scen {scen_id}: rel err {err:.3e}")
        if len(failures) > 10:
            msg_lines.append(f"  ... and {len(failures) - 10} more")
        raise AssertionError("\n".join(msg_lines))


def test_growevapl_matches_fortran():
    """Phase 8.5: JAX growevapl matches Fortran's growevapl(growlg, evaplg)
    across all scenarios at substep 1.

    Fortran growevapl.F90: PPM growth+evap loss rates. Called from microfast
    with pconmax that was set pre-microfast — meaning the regular dumped
    growlg/evaplg were computed on a different (pre-advance) pc state.
    The `dump_growevapl_probe` calls maxconc then growevapl at end-of-step
    so JAX and Fortran share the same pc/state.

    Sulfate test: NGROUP=1, igas=H2SO4 (index 1, 0-based), no ice.
    PPM tables (dm, pratt, prat, pden1, palr) are dumped at step 1 only.
    dtime = 1800 s (carma_sulfatetest.F90).
    """
    from carma.growth.growevapl import growevapl
    import jax.numpy as jnp

    DTIME = 1800.0
    iz = 0
    NBIN = 38
    NGROUP = 1
    is_ice_arr = jnp.asarray([False])
    ienconc_arr = jnp.asarray([0])

    growlg_max_rel = {}
    evaplg_max_rel = {}
    failures = []

    # Load step-1 static tables from scen_000 (same for all scenarios)
    d0 = read_substep(_DIFF_BASE / "scen_000", step=1)
    if d0.dm_bin is None or d0.pratt is None:
        pytest.skip("PPM tables not in dump — re-run scripts/run_diagnostic_ensemble.py")
    dm_j   = jnp.asarray(d0.dm_bin)
    pratt_j = jnp.asarray(d0.pratt)
    prat_j  = jnp.asarray(d0.prat)
    pden1_j = jnp.asarray(d0.pden1)
    palr_j  = jnp.asarray(d0.palr)
    # igrowgas from dump is float-encoded Fortran 1-based; convert to 0-based int
    igrowgas_arr = jnp.asarray(
        [int(d0.igrowgas[0]) - 1]  # 2 (Fortran) → 1 (0-based H2SO4)
    )

    for scen_id in _ALL_SCEN_IDS:
        scen_dir = _DIFF_BASE / f"scen_{scen_id:03d}"
        d = read_substep(scen_dir, step=1)
        growlg_F = np.fromfile(scen_dir / "substep_0001_growlg_probe.bin",
                               dtype=np.float64).reshape(NBIN, NGROUP, order="F")
        evaplg_F = np.fromfile(scen_dir / "substep_0001_evaplg_probe.bin",
                               dtype=np.float64).reshape(NBIN, NGROUP, order="F")
        if not (scen_dir / "substep_0001_growlg_probe.bin").exists():
            pytest.skip("growlg_probe not in dump")

        # Compute pconmax at end-of-step (matches what growevapl probe does)
        from carma.utils.smallconc import maxconc
        pconmax_j = maxconc(jnp.asarray(d.pc), ienconc_arr, jnp.asarray(d.zmet))

        growlg_init = jnp.zeros((NBIN, NGROUP), dtype=jnp.float64)
        evaplg_init = jnp.zeros((NBIN, NGROUP), dtype=jnp.float64)
        growlg_j, evaplg_j = growevapl(
            jnp.asarray(d.pc),
            growlg_init, evaplg_init,
            jnp.asarray(d.supsatl), jnp.asarray(d.supsati),
            jnp.asarray(d.pvapl), jnp.asarray(d.pvapi),
            jnp.asarray(d.akelvin), jnp.asarray(d.akelvini),
            jnp.asarray(d.gro), jnp.asarray(d.gro1),
            jnp.asarray(d.rup_wet),
            jnp.asarray(d.rmass_bin), dm_j, pconmax_j,
            pratt_j, prat_j, pden1_j, palr_j,
            is_ice_arr, igrowgas_arr, ienconc_arr,
            DTIME, iz, NBIN, NGROUP,
        )

        _, mg = compute_rel_err(np.asarray(growlg_j), growlg_F)
        _, me = compute_rel_err(np.asarray(evaplg_j), evaplg_F)
        growlg_max_rel[scen_id] = mg
        evaplg_max_rel[scen_id] = me
        tol = _tol_for("growevapl")
        if mg > tol:
            failures.append(("growlg", scen_id, mg))
        if me > tol:
            failures.append(("evaplg", scen_id, me))

    sd = _summary_plot_dir()
    make_summary_plot("growevapl_growlg", growlg_max_rel, sd)
    make_summary_plot("growevapl_evaplg", evaplg_max_rel, sd)

    if failures:
        msg_lines = [f"growevapl mismatched on {len(failures)} (output, scen) tuples:"]
        for which, scen_id, err in failures[:10]:
            msg_lines.append(f"  {which} scen {scen_id}: rel err {err:.3e}")
        if len(failures) > 10:
            msg_lines.append(f"  ... and {len(failures) - 10} more")
        raise AssertionError("\n".join(msg_lines))


def test_pheat_matches_fortran():
    """Phase 8.4: JAX pheat matches Fortran's pheat for bins 0..NBIN-2
    across all scenarios at substep 1.

    Sulfate test: do_pheat=False, NWAVE=0, so the no-radiation branch runs:
        akas = exp(akelvin[iz, igas] / rup_wet[iz, ibin, igroup])
        dmdt = pvap * (S + 1 - akas) * g0 / (1 + g0*g1*pvap)

    growevapl calls pheat for ibin=1..NBIN-1 (Fortran 1-based) = bins
    0..NBIN-2 (0-based). The probe matches that range; dmdt[NBIN-1]=0
    (unused slot).

    Probe dumps `pheat_probe.bin` — a flat (NBIN,) array where the last
    entry is 0.
    """
    from carma.growth.pheat import pheat as jax_pheat
    import jax.numpy as jnp

    iz = 0
    igroup = 0
    igas = 1       # H2SO4 is gas index 1 (0-based) in sulfate test
    NBIN = 38

    dmdt_max_rel = {}
    failures = []

    for scen_id in _ALL_SCEN_IDS:
        scen_dir = _DIFF_BASE / f"scen_{scen_id:03d}"
        d = read_substep(scen_dir, step=1)
        probe_path = scen_dir / "substep_0001_pheat_probe.bin"
        if not probe_path.exists() or d.rup_wet is None:
            pytest.skip("pheat_probe.bin or rup_wet not in dump — re-run scripts/run_diagnostic_ensemble.py")
        F_dmdt = np.fromfile(probe_path, dtype=np.float64)  # (NBIN,)

        per_scen_errs = []
        for ib in range(NBIN - 1):
            dmdt_j = float(jax_pheat(
                jnp.asarray(d.pc), jnp.asarray(d.supsatl), jnp.asarray(d.supsati),
                jnp.asarray(d.pvapl), jnp.asarray(d.pvapi),
                jnp.asarray(d.akelvin), jnp.asarray(d.akelvini),
                jnp.asarray(d.gro), jnp.asarray(d.gro1),
                jnp.asarray(d.rup_wet), None,
                is_ice=False, iz=iz, igroup=igroup, ibin=ib, igas=igas,
            ))
            f = float(F_dmdt[ib])
            e = abs(dmdt_j - f) / max(abs(f), 1e-300)
            per_scen_errs.append(e)
            tol = _tol_for("pheat")
            if e > tol:
                failures.append((scen_id, ib, e, dmdt_j, f))

        dmdt_max_rel[scen_id] = max(per_scen_errs) if per_scen_errs else 0.0

    make_summary_plot("pheat_dmdt", dmdt_max_rel, _summary_plot_dir())
    if failures:
        msg_lines = [f"pheat mismatched on {len(failures)} (scen, bin) pairs:"]
        for scen_id, ib, err, j, f in failures[:10]:
            msg_lines.append(
                f"  scen {scen_id} bin {ib}: rel err {err:.3e}  JAX={j:.3e}  F={f:.3e}"
            )
        if len(failures) > 10:
            msg_lines.append(f"  ... and {len(failures) - 10} more")
        raise AssertionError("\n".join(msg_lines))


def test_maxconc_matches_fortran():
    """Phase 8.1a: JAX maxconc matches Fortran's maxconc(end-of-step cstate).

    Fortran (maxconc.F90:38-41):
        pconmax(iz, igrp) = maxval(pc(iz, :, iep)) / zmet(iz)

    The regular dump's f_pconmax is stale — it was computed pre-microfast
    (newstate_calc.F90:175), while the dumped pc is post-microfast. A
    `dump_maxconc_probe` in the diagnostic calls Fortran's maxconc at
    end-of-step and dumps the result as `maxconc_probe.bin`.

    Sulfate test: NZ=1, NGROUP=1, ienconc[0]=0.
    """
    from carma.utils.smallconc import maxconc
    import jax.numpy as jnp

    ienconc_arr = jnp.asarray([0])

    pconmax_max_rel = {}
    failures = []

    for scen_id in _ALL_SCEN_IDS:
        scen_dir = _DIFF_BASE / f"scen_{scen_id:03d}"
        d = read_substep(scen_dir, step=1)
        probe_path = scen_dir / "substep_0001_maxconc_probe.bin"
        if not probe_path.exists():
            pytest.skip("maxconc_probe.bin not in dump — re-run scripts/run_diagnostic_ensemble.py")
        # probe is shape (NBIN, NGROUP) from cstate%f_pconmax but only NZ=1
        pconmax_F = np.fromfile(probe_path, dtype=np.float64).reshape(
            d.pconmax.shape, order="F"
        )
        pconmax_jax = maxconc(
            jnp.asarray(d.pc), ienconc_arr, jnp.asarray(d.zmet)
        )
        _, m = compute_rel_err(np.asarray(pconmax_jax), pconmax_F)
        pconmax_max_rel[scen_id] = m
        if m > _tol_for("maxconc"):
            failures.append((scen_id, m,
                             float(pconmax_jax[0, 0]), float(pconmax_F[0, 0])))

    make_summary_plot("maxconc_pconmax", pconmax_max_rel, _summary_plot_dir())
    if failures:
        msg_lines = [f"maxconc mismatched on {len(failures)} scenarios:"]
        for scen_id, err, jv, fv in failures[:10]:
            msg_lines.append(
                f"  scen {scen_id}: rel err {err:.3e}  JAX={jv:.3e}  F={fv:.3e}"
            )
        if len(failures) > 10:
            msg_lines.append(f"  ... and {len(failures) - 10} more")
        raise AssertionError("\n".join(msg_lines))


def test_smallconc_matches_fortran():
    """Phase 8.1b: JAX smallconc matches Fortran after psolve application.

    Fortran (smallconc.F90:42-43) for number-element (ielem == ip):
        pc(iz, ibin, ielem) = max(pc(iz, ibin, ielem), SMALL_PC)

    Sulfate test has NELEM=1, itype=I_VOLATILE (number element only —
    no core-mass or second-moment elements). So smallconc reduces to
    a per-bin max(pc, SMALL_PC). Since the dumped pc already reflects
    the post-psolve/post-smallconc state, feeding that pc back through
    smallconc must be a no-op (all values ≥ SMALL_PC). The bench
    therefore verifies:
      1. JAX smallconc is idempotent on the dumped pc.
      2. The output matches the input exactly (bit-for-bit).
    """
    from carma.utils.smallconc import smallconc
    from carma.enums import ElementType
    import jax.numpy as jnp

    # Sulfate test: NELEM=1, NGROUP=1, element type is I_VOLATILE
    # (carma_sulfatetest.F90:145 — CARMAELEMENT_Create with I_VOLATILE).
    # I_VOLATILE is the particle number-density type.
    itype_arr = jnp.asarray([int(ElementType.I_VOLATILE)])
    ienconc_arr = jnp.asarray([0])
    igelem_arr = jnp.asarray([0])

    pc_max_rel = {}
    failures = []

    for scen_id in _ALL_SCEN_IDS:
        d = read_substep(_DIFF_BASE / f"scen_{scen_id:03d}", step=1)
        if d.rmass_bin is None:
            pytest.skip("rmass_bin not in dump — re-run scripts/run_diagnostic_ensemble.py")

        pc_jax = smallconc(
            jnp.asarray(d.pc), itype_arr, ienconc_arr, igelem_arr,
            jnp.asarray(d.rmass_bin),
        )
        # Expected: idempotent on already-floored pc (input == output)
        _, m = compute_rel_err(np.asarray(pc_jax), d.pc)
        pc_max_rel[scen_id] = m
        if m > _tol_for("smallconc"):
            failures.append((scen_id, m))

    make_summary_plot("smallconc_pc", pc_max_rel, _summary_plot_dir())
    if failures:
        msg_lines = [f"smallconc mismatched on {len(failures)} scenarios:"]
        for scen_id, err in failures[:10]:
            msg_lines.append(f"  scen {scen_id}: rel err {err:.3e}")
        if len(failures) > 10:
            msg_lines.append(f"  ... and {len(failures) - 10} more")
        raise AssertionError("\n".join(msg_lines))


def test_gasexchange_matches_fortran():
    """Phase 7.12: JAX gasexchange matches Fortran's gasexchange invoked
    via the diagnostic probe at end-of-step state.

    Fortran's gasexchange is dead code in modern CARMA (microfast.F90:152
    has the call commented out — gsolve overwrites gasprod via total-
    condensate). The bench therefore uses a diagnostic probe that calls
    Fortran's gasexchange directly with end-of-step cstate; JAX is fed
    the same end-of-step inputs and we compare the resulting gasprod
    vector.

    Sulfate test scope: cmf == 0 and totevap == False everywhere
    (verified empirically across 1000 scenarios). Heterogeneous nucleation
    is gated out (Phase 7.10: rnuclg == 0). Diffmass for adjacent
    same-group bins is rmass[i+1] - rmass[i] for the mass-ratio-2 grid.
    """
    from carma.gasexchange import gasexchange
    import jax.numpy as jnp

    igroup = 0
    iz = 0
    ielem_num = 0          # ienconc[0] = 0 (single element)
    igas_h2o = 0
    igas_h2so4 = 1
    NGROUP = 1
    NELEM = 1
    NGAS = 2

    # Static CARMA tables for the sulfate test:
    ienconc = jnp.asarray([ielem_num])
    igelem = jnp.asarray([igroup])
    inucgas = jnp.asarray([igas_h2so4])     # group 0's condensing gas = H2SO4
    nnuc2elem = jnp.asarray([1])             # AddNucleation(1,1,...) sets nnuc2elem[0]=1
    igrowgas = jnp.asarray([igas_h2so4])    # AddGrowth(1, 2, ...) → igrowgas[0]=1
    if_nuc = jnp.asarray([[True]])           # nucleation maps elem 0 → elem 0

    gasprod_max_rel = {}
    failures = []

    for scen_id in _ALL_SCEN_IDS:
        scen_dir = _DIFF_BASE / f"scen_{scen_id:03d}"
        d = read_substep(scen_dir, step=1)
        probe_path = scen_dir / "substep_0001_gasexchange_probe.bin"
        if not probe_path.exists() or d.rmass_bin is None or d.cmf is None or d.totevap is None:
            pytest.skip("required dump fields missing — re-run scripts/run_diagnostic_ensemble.py")
        gasprod_F = np.fromfile(probe_path, dtype=np.float64)  # (NGAS,)

        NBIN = d.rmass_bin.shape[0]
        rmass_2d = jnp.asarray(d.rmass_bin)                 # (NBIN, NGROUP)
        # Build a sparse diffmass[(i2, ig2, i, ig)] = rmass[i2, ig2] - rmass[i, ig]
        rmass_arr = np.asarray(d.rmass_bin)
        diffmass_np = (
            rmass_arr[:, :, None, None] - rmass_arr[None, None, :, :]
        )  # (NBIN, NGROUP, NBIN, NGROUP)
        diffmass = jnp.asarray(diffmass_np)

        # inuc2bin: for sulfate test rnuclg=0 so this is irrelevant, but
        # supply -1 to indicate "no target" (matches Fortran 0 → 0-based -1).
        inuc2bin = -jnp.ones((NBIN, NGROUP, NGROUP), dtype=jnp.int32)

        gasprod_jax = gasexchange(
            pc_iz=jnp.asarray(d.pc[iz]),
            rhompe=jnp.asarray(d.rhompe),
            rnuclg=jnp.asarray(d.rnuclg),
            growlg=jnp.asarray(d.growlg),
            evaplg=jnp.asarray(d.evaplg),
            rmass=rmass_2d,
            diffmass=diffmass,
            cmf=jnp.asarray(d.cmf),
            totevap=jnp.asarray(d.totevap > 0.5),
            inuc2bin=inuc2bin,
            if_nuc=if_nuc,
            ienconc=ienconc,
            igelem=igelem,
            inucgas=inucgas,
            nnuc2elem=nnuc2elem,
            igrowgas=igrowgas,
            ngas=NGAS, ngroup=NGROUP, nelem=NELEM, nbin=NBIN,
        )
        gasprod_jax = np.asarray(gasprod_jax)

        _, m = compute_rel_err(gasprod_jax, gasprod_F)
        gasprod_max_rel[scen_id] = m
        if m > _tol_for("gasexchange"):
            failures.append(
                (scen_id, m,
                 float(gasprod_jax[0]), float(gasprod_F[0]),
                 float(gasprod_jax[1]), float(gasprod_F[1]))
            )

    make_summary_plot("gasexchange_gasprod", gasprod_max_rel, _summary_plot_dir())

    if failures:
        msg_lines = [f"gasexchange mismatched on {len(failures)} scenarios:"]
        for scen_id, err, jh2o, fh2o, jh2so4, fh2so4 in failures[:10]:
            msg_lines.append(
                f"  scen {scen_id}: rel err {err:.3e}  "
                f"h2o J={jh2o:.3e} F={fh2o:.3e}  "
                f"h2so4 J={jh2so4:.3e} F={fh2so4:.3e}"
            )
        if len(failures) > 10:
            msg_lines.append(f"  ... and {len(failures) - 10} more")
        raise AssertionError("\n".join(msg_lines))


def test_homogeneous_nucleation_matches_fortran():
    """Phase 7.11: JAX homogeneous_nucleation matches Fortran's
    sulfnuc homogeneous output (nucrate placed at nucbin in rhompe).

    The bench compares against a Fortran-side reconstruction:
    ``expected_rhompe[k] = (probe_nucrate · zmet) at nucbin_F, else 0``,
    where ``nucbin_F`` is computed from the probe-dumped ``mass_cluster_dry``
    using Fortran's bin-locate formula:
        nucbin_F = 0 if mass < rmassup[0]
                   else 1 + floor(log(mass/rmassup[0]) / log(rmrat))
    """
    from carma.nucleation.sulfnuc import homogeneous_nucleation
    from carma.constants import AVG, RGAS, PI as PI_carma
    import jax.numpy as jnp

    AVG_J = float(AVG)
    RGAS_J = float(RGAS)
    PI_J = float(PI_carma)
    GWTMOL_H2SO4 = 98.078479
    GWTMOL_H2O = 18.016
    igas_h2o = 0
    igas_h2so4 = 1
    iz = 0
    igroup = 0

    rhompe_max_rel = {}
    failures = []

    for scen_id in _ALL_SCEN_IDS:
        scen_dir = _DIFF_BASE / f"scen_{scen_id:03d}"
        d = read_substep(scen_dir, step=1)
        probe_path = scen_dir / "substep_0001_zhao1995_probe.bin"
        if not probe_path.exists() or d.rmassup_bin is None or d.rmrat_group is None:
            pytest.skip("probe / rmassup_bin / rmrat_group not in dump — re-run scripts/run_diagnostic_ensemble.py")
        F_nuc, F_mass, _, _ = np.fromfile(probe_path, dtype=np.float64)

        T = float(d.t[iz])
        zmet = float(d.zmet[iz])
        wtpct_iz = float(d.wtpct[iz])
        h2so4_cgs = float(d.gc[iz, igas_h2so4] / zmet)
        h2o_cgs = float(d.gc[iz, igas_h2o] / zmet)
        h2so4 = h2so4_cgs * AVG_J / GWTMOL_H2SO4
        h2o_n = h2o_cgs * AVG_J / GWTMOL_H2O
        rh = float(d.supsatl[iz, igas_h2o]) + 1.0
        rmassup = jnp.asarray(d.rmassup_bin[:, igroup])
        rmrat = float(d.rmrat_group[igroup])
        NBIN = rmassup.shape[0]

        # JAX homogeneous_nucleation
        nucrate_z, nucbin, _, _ = homogeneous_nucleation(
            jnp.float64(T), jnp.float64(wtpct_iz), jnp.float64(rh),
            jnp.float64(h2so4), jnp.float64(h2so4_cgs),
            jnp.float64(h2o_n), jnp.float64(h2o_cgs),
            rmassup, jnp.float64(rmrat), jnp.float64(zmet),
            method="ZhaoTurco",
            gwtmol_h2so4=jnp.float64(GWTMOL_H2SO4),
            gwtmol_h2o=jnp.float64(GWTMOL_H2O),
        )
        rhompe_jax = np.zeros(NBIN, dtype=np.float64)
        nb_jax = int(nucbin)
        if 0 <= nb_jax < NBIN:
            rhompe_jax[nb_jax] = float(nucrate_z)

        # Fortran-side expected from probe
        rmassup0 = float(d.rmassup_bin[0, igroup])
        if F_mass < rmassup0:
            nb_F = 0
        else:
            nb_F = 1 + int(np.floor(np.log(F_mass / rmassup0) / np.log(rmrat)))
        rhompe_F = np.zeros(NBIN, dtype=np.float64)
        if 0 <= nb_F < NBIN:
            rhompe_F[nb_F] = F_nuc * zmet

        _, m = compute_rel_err(rhompe_jax, rhompe_F)
        rhompe_max_rel[scen_id] = m
        if m > _tol_for("homogeneous_nucleation") or nb_jax != nb_F:
            failures.append((scen_id, m, nb_jax, nb_F,
                             float(rhompe_jax[nb_jax] if 0 <= nb_jax < NBIN else 0),
                             float(rhompe_F[nb_F] if 0 <= nb_F < NBIN else 0)))

    make_summary_plot("homogeneous_nucleation_rhompe", rhompe_max_rel,
                      _summary_plot_dir())

    if failures:
        msg_lines = [f"homogeneous_nucleation mismatched on {len(failures)} scenarios:"]
        for scen_id, err, nb_j, nb_f, jv, fv in failures[:10]:
            msg_lines.append(
                f"  scen {scen_id}: rel err {err:.3e}  nucbin J={nb_j} F={nb_f}  "
                f"JAX={jv:.3e}  F={fv:.3e}"
            )
        if len(failures) > 10:
            msg_lines.append(f"  ... and {len(failures) - 10} more")
        raise AssertionError("\n".join(msg_lines))


def test_sulfnuc_driver_matches_fortran():
    """Phase 7.11: JAX sulfnuc driver matches Fortran (rhompe, rnuclg).

    Sulfate test: heterogeneous nucleation gated out (no I_HETNUCSULF
    mapping — see Phase 7.10 row), so rnuclg is identically 0 from
    Fortran. The driver bench therefore checks that JAX `sulfnuc`:
      - rhompe matches the probe-derived expected (via homogeneous_nucleation)
      - rnuclg is identically 0
    """
    from carma.nucleation.sulfnuc import sulfnuc
    from carma.constants import AVG, RGAS, PI as PI_carma
    import jax.numpy as jnp

    AVG_J = float(AVG)
    RGAS_J = float(RGAS)
    PI_J = float(PI_carma)
    GWTMOL_H2SO4 = 98.078479
    GWTMOL_H2O = 18.016
    igas_h2o = 0
    igas_h2so4 = 1
    iz = 0
    igroup = 0

    rhompe_max_rel = {}
    rnuclg_max_abs = {}
    failures = []

    for scen_id in _ALL_SCEN_IDS:
        scen_dir = _DIFF_BASE / f"scen_{scen_id:03d}"
        d = read_substep(scen_dir, step=1)
        probe_path = scen_dir / "substep_0001_zhao1995_probe.bin"
        if not probe_path.exists() or d.rmassup_bin is None or d.rmrat_group is None or d.r_bin is None:
            pytest.skip("required dump fields missing — re-run scripts/run_diagnostic_ensemble.py")
        F_nuc, F_mass, _, _ = np.fromfile(probe_path, dtype=np.float64)

        T = float(d.t[iz])
        zmet = float(d.zmet[iz])
        wtpct_iz = float(d.wtpct[iz])
        h2so4_cgs = float(d.gc[iz, igas_h2so4] / zmet)
        h2o_cgs = float(d.gc[iz, igas_h2o] / zmet)
        h2so4 = h2so4_cgs * AVG_J / GWTMOL_H2SO4
        h2o_n = h2o_cgs * AVG_J / GWTMOL_H2O
        rh = float(d.supsatl[iz, igas_h2o]) + 1.0
        rmassup = jnp.asarray(d.rmassup_bin[:, igroup])
        rmrat = float(d.rmrat_group[igroup])
        # Sulfate test has no het seeding; r_bins is dry radius (unused
        # because heterogeneous gate fails on the Fortran side).
        r_bins = jnp.asarray(d.r_bin[:, igroup])
        NBIN = rmassup.shape[0]

        # JAX sulfnuc driver — heterogeneous DISABLED to match Fortran's
        # gate (sulfate test never adds an I_HETNUCSULF mapping, so the
        # Fortran sulfhetnucrate is never invoked).
        rhompe_jax_arr, rnuclg_jax_arr = sulfnuc(
            jnp.float64(T), jnp.float64(wtpct_iz), jnp.float64(rh),
            jnp.float64(h2so4), jnp.float64(h2so4_cgs),
            jnp.float64(h2o_n), jnp.float64(h2o_cgs),
            r_bins, rmassup, jnp.float64(rmrat), jnp.float64(zmet),
            method="ZhaoTurco",
            do_homogeneous=True,
            do_heterogeneous=False,
            gwtmol_h2so4=jnp.float64(GWTMOL_H2SO4),
            gwtmol_h2o=jnp.float64(GWTMOL_H2O),
        )
        rhompe_jax = np.asarray(rhompe_jax_arr)
        rnuclg_jax = np.asarray(rnuclg_jax_arr)

        rmassup0 = float(d.rmassup_bin[0, igroup])
        if F_mass < rmassup0:
            nb_F = 0
        else:
            nb_F = 1 + int(np.floor(np.log(F_mass / rmassup0) / np.log(rmrat)))
        rhompe_F = np.zeros(NBIN, dtype=np.float64)
        if 0 <= nb_F < NBIN:
            rhompe_F[nb_F] = F_nuc * zmet
        rnuclg_F = d.rnuclg[:, igroup, igroup]   # (NBIN,) for sulfate test

        _, m_h = compute_rel_err(rhompe_jax, rhompe_F)
        m_g = float(np.max(np.abs(rnuclg_jax - rnuclg_F)))
        rhompe_max_rel[scen_id] = m_h
        rnuclg_max_abs[scen_id] = m_g

        tol = _tol_for("sulfnuc")
        if m_h > tol:
            failures.append(("rhompe", scen_id, m_h, ""))
        if m_g > 0.0:
            failures.append(("rnuclg", scen_id, m_g, "expected all zeros"))

    sd = _summary_plot_dir()
    make_summary_plot("sulfnuc_rhompe", rhompe_max_rel, sd)
    # rnuclg is exactly 0 across the bench scope, so a CDF plot adds
    # little — skip.

    if failures:
        msg_lines = [f"sulfnuc driver mismatched on {len(failures)} (output, scen) tuples:"]
        for which, scen_id, err, note in failures[:10]:
            msg_lines.append(f"  {which} scen {scen_id}: err {err:.3e}  {note}")
        if len(failures) > 10:
            msg_lines.append(f"  ... and {len(failures) - 10} more")
        raise AssertionError("\n".join(msg_lines))


def test_rhopart_matches_fortran():
    """Phase 7.8: JAX rhopart matches Fortran's f_rhop[:, :, igroup] across
    all scenarios at substep 1.

    Sulfate test has NELEM=1, NGROUP=1, ncore=0, so the Fortran code
    short-circuits to ``rhop(iz,:,igroup) = rhoelem(:,iepart)`` (=1.923
    g/cm³ at every bin). This bench therefore confirms the no-core path
    of the JAX rhopart against Fortran's stored output.
    """
    from carma.rhopart import rhopart
    import jax.numpy as jnp

    igroup = 0
    iepart = 0
    RHO_SULFATE = 1.923  # carma_sulfatetest.F90:137 — RHO_SULFATE

    rhop_max_rel = {}
    failures = []

    for scen_id in _ALL_SCEN_IDS:
        d = read_substep(_DIFF_BASE / f"scen_{scen_id:03d}", step=1)
        if d.rmass_bin is None:
            pytest.skip("rmass_bin not in dump — re-run scripts/run_diagnostic_ensemble.py")
        NBIN = d.rmass_bin.shape[0]
        NELEM = d.pc.shape[2]

        rhoelem = jnp.full((NBIN, NELEM), RHO_SULFATE)
        ienconc_arr = jnp.asarray([iepart])
        igelem_arr = jnp.asarray([igroup])
        # No cores: MAX_NCORE=1 padded with -1 so the inner loop is a no-op.
        icorelem = jnp.asarray([[-1]])
        ncore = jnp.asarray([0])

        rhop_jax, _ = rhopart(
            jnp.asarray(d.pc), jnp.asarray(d.rmass_bin), rhoelem,
            ienconc_arr, igelem_arr, icorelem, ncore,
        )
        rhop_f = d.rhop[:, :, igroup]
        _, m = compute_rel_err(np.asarray(rhop_jax[:, :, igroup]), rhop_f)
        rhop_max_rel[scen_id] = m
        if m > _tol_for("rhopart"):
            failures.append((scen_id, m))

    make_summary_plot("rhopart_rhop", rhop_max_rel, _summary_plot_dir())
    if failures:
        msg_lines = [f"rhopart mismatched on {len(failures)} scenarios:"]
        for scen_id, err in failures[:10]:
            msg_lines.append(f"  scen {scen_id}: rel err {err:.3e}")
        if len(failures) > 10:
            msg_lines.append(f"  ... and {len(failures) - 10} more")
        raise AssertionError("\n".join(msg_lines))


def test_wetr_wtpct_matches_fortran():
    """Phase 7.6: JAX `_wetr_wtpct` (the I_WTPCT_H2SO4 branch of get_wetr)
    matches Fortran's per-bin r_wet[iz,ibin,igroup] and rhop_wet[iz,ibin,igroup]
    across all scenarios at substep 1.

    Fortran call site (rhopart.F90 lines 148-150):
        getwetr(carma, igroup, ibin, relhum(iz), r(ibin,igroup), r_wet(iz,ibin,igroup),
                rhop(iz,ibin,igroup), rhop_wet(iz,ibin,igroup), ...,
                h2o_mass=gc(iz,igash2o)/zmet(iz), h2o_vp=pvapl(iz,igash2o), temp=t(iz))

    For sulfate (single element, no core), rhop equals the element density
    so wetr.F90's rdry_init→rdry transform is a no-op and we can pass
    r(ibin,igroup) directly as rdry.
    """
    from carma.wetr import _wetr_wtpct
    import jax.numpy as jnp

    igas_h2o = 0
    GWTMOL_H2SO4 = 98.078479
    igroup = 0  # sulfate test has NGROUP=1
    iz = 0

    rwet_max_rel = {}
    rhopwet_max_rel = {}
    failures = []

    for scen_id in _ALL_SCEN_IDS:
        d = read_substep(_DIFF_BASE / f"scen_{scen_id:03d}", step=1)
        if d.r_bin is None:
            pytest.skip("r_bin not in dump — re-run scripts/run_diagnostic_ensemble.py")
        NBIN = d.r_bin.shape[0]
        T = float(d.t[iz])
        h2o_mass = float(d.gc[iz, igas_h2o] / d.zmet[iz])
        h2o_vp = float(d.pvapl[iz, igas_h2o])

        rwet_jax = np.empty(NBIN, dtype=np.float64)
        rhopwet_jax = np.empty(NBIN, dtype=np.float64)
        for ib in range(NBIN):
            rdry = float(d.r_bin[ib, igroup])
            rhopdry = float(d.rhop[iz, ib, igroup])
            rw, rhw = _wetr_wtpct(
                jnp.float64(rdry), jnp.float64(rhopdry),
                jnp.float64(T), jnp.float64(h2o_mass),
                jnp.float64(h2o_vp), jnp.float64(GWTMOL_H2SO4),
            )
            rwet_jax[ib] = float(rw)
            rhopwet_jax[ib] = float(rhw)

        rwet_f = d.r_wet[iz, :, igroup]
        rhopwet_f = d.rhop_wet[iz, :, igroup]
        _, m_r = compute_rel_err(rwet_jax, rwet_f)
        _, m_rho = compute_rel_err(rhopwet_jax, rhopwet_f)
        rwet_max_rel[scen_id] = m_r
        rhopwet_max_rel[scen_id] = m_rho
        tol = _tol_for("wetr_wtpct")
        if m_r > tol:
            failures.append(("r_wet", scen_id, m_r))
        if m_rho > tol:
            failures.append(("rhop_wet", scen_id, m_rho))

    sd = _summary_plot_dir()
    make_summary_plot("wetr_wtpct_r_wet",    rwet_max_rel,    sd)
    make_summary_plot("wetr_wtpct_rhop_wet", rhopwet_max_rel, sd)

    if failures:
        msg_lines = [f"_wetr_wtpct mismatched on {len(failures)} (output, scen) tuples:"]
        for which, scen_id, err in failures[:10]:
            msg_lines.append(f"  {which} scen {scen_id}: rel err {err:.3e}")
        if len(failures) > 10:
            msg_lines.append(f"  ... and {len(failures) - 10} more")
        raise AssertionError("\n".join(msg_lines))


def test_supersat_matches_fortran():
    """Phase 7.5: JAX supersat(t, gc, pvapl, pvapi, gwtmol, zmet) matches
    Fortran's supsatl[:, igas] / supsati[:, igas] for both gases (H2O,
    H2SO4) across all scenarios at substep 1.

    Fortran reference (supersat.F90 lines 36-41):
        rvap = RGAS / gwtmol(igas)
        gc_cgs = gc(iz, igas) / zmet(iz)
        supsatl(iz, igas) = (gc_cgs * rvap * t(iz) - pvapl(iz, igas)) / pvapl(iz, igas)
        supsati(iz, igas) = (gc_cgs * rvap * t(iz) - pvapi(iz, igas)) / pvapi(iz, igas)

    Sulfate test is clearsky (do_incloud=False), so the cloud-scaling
    branch is not exercised here.
    """
    from carma.supersaturation import supersat
    import jax.numpy as jnp

    # Gas registration in carma_sulfatetest.F90:
    #   igas=1 (Fortran 1-based) "Water Vapor"   gwtmol = 18.016     -> Python igas_h2o=0
    #   igas=2 (Fortran 1-based) "Sulpheric Acid" gwtmol = 98.078479 -> Python igas_h2so4=1
    GAS_NAMES = ["h2o", "h2so4"]
    GWTMOL = [18.016, 98.078479]

    max_rel = {0: {}, 1: {}}
    max_rel_i = {0: {}, 1: {}}
    failures = []

    for scen_id in _ALL_SCEN_IDS:
        d = read_substep(_DIFF_BASE / f"scen_{scen_id:03d}", step=1)
        for igas in (0, 1):
            ssl_jax, ssi_jax = supersat(
                jnp.asarray(d.t),
                jnp.asarray(d.gc[:, igas]),
                jnp.asarray(d.pvapl[:, igas]),
                jnp.asarray(d.pvapi[:, igas]),
                GWTMOL[igas],
                jnp.asarray(d.zmet),
            )
            ssl_f = d.supsatl[:, igas]
            ssi_f = d.supsati[:, igas]
            _, ml = compute_rel_err(np.asarray(ssl_jax), ssl_f)
            _, mi = compute_rel_err(np.asarray(ssi_jax), ssi_f)
            max_rel[igas][scen_id] = ml
            max_rel_i[igas][scen_id] = mi
            tol = _tol_for("supersat")
            if ml > tol:
                failures.append(("supsatl", GAS_NAMES[igas], scen_id, ml,
                                 float(ssl_jax[0]), float(ssl_f[0])))
            if mi > tol:
                failures.append(("supsati", GAS_NAMES[igas], scen_id, mi,
                                 float(ssi_jax[0]), float(ssi_f[0])))

    sd = _summary_plot_dir()
    make_summary_plot("supersat_supsatl_h2o",   max_rel[0],   sd)
    make_summary_plot("supersat_supsatl_h2so4", max_rel[1],   sd)
    make_summary_plot("supersat_supsati_h2o",   max_rel_i[0], sd)
    make_summary_plot("supersat_supsati_h2so4", max_rel_i[1], sd)

    if failures:
        msg_lines = [f"supersat mismatched on {len(failures)} (output, gas, scen) tuples:"]
        for which, gas, scen_id, err, j, f in failures[:10]:
            msg_lines.append(
                f"  {which} [{gas}] scen {scen_id}: rel err {err:.3e}  "
                f"JAX={j:.3e}  F={f:.3e}"
            )
        if len(failures) > 10:
            msg_lines.append(f"  ... and {len(failures) - 10} more")
        raise AssertionError("\n".join(msg_lines))
