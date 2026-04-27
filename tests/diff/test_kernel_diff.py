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
_DIFF_DIR_SCEN_24 = _REPO_ROOT / "data" / "diff" / "scen_024"
_PLOT_DIR_SCEN_24 = _REPO_ROOT / "plots" / "diff" / "scen_024"


# Per-kernel tolerance overrides. Default is 1e-10. Add an entry here
# when a kernel has known floating-point reordering differences with
# Fortran (e.g. summation order in a vectorized JAX op).
_PER_KERNEL_TOL = {
    # "growevapl": 1e-6,  # example
}


def _tol_for(kernel_name: str) -> float:
    return _PER_KERNEL_TOL.get(kernel_name, 1e-10)


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
# Phase 0 exit gate: trivial test confirming the harness wiring works
# end-to-end on a kernel that's known to be correctly ported.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scen24_step1() -> SubstepDump:
    """Load substep 1 of scenario 24 from the diagnostic dump.

    Skips the tests if the dump hasn't been produced yet (see
    ``scripts/fortran_patch/apply_diagnostic_patch.sh`` to build and
    run the diagnostic binary).
    """
    if not _DIFF_DIR_SCEN_24.exists():
        pytest.skip(
            f"Diagnostic dump missing: {_DIFF_DIR_SCEN_24}. "
            "Run apply_diagnostic_patch.sh + test_sulfate_diagnostic first."
        )
    return read_substep(_DIFF_DIR_SCEN_24, step=1)


def test_vaporp_h2o_murphy2005_matches_fortran(scen24_step1):
    """Phase 0 exit gate: JAX vaporp_h2o_murphy2005 matches Fortran's
    f_pvapl[:, igas_h2o] and f_pvapi[:, igas_h2o] for scen 24 substep 1.
    """
    from carma.vapor_pressure import vaporp_h2o_murphy2005

    # In the diagnostic binary's CARMA setup, igas_h2o = 1 (1-based) → 0 (0-based).
    igas_h2o = 0
    t = scen24_step1.t                       # (NZ,)
    pvapl_jax, pvapi_jax = vaporp_h2o_murphy2005(t)
    pvapl_fortran = scen24_step1.pvapl[:, igas_h2o]
    pvapi_fortran = scen24_step1.pvapi[:, igas_h2o]

    make_kernel_plot(
        "vaporp_h2o_murphy2005_pvapl",
        np.asarray(pvapl_jax), pvapl_fortran,
        _PLOT_DIR_SCEN_24,
        x_label="z-level", y_label="pvapl over liquid water [dyne/cm²]",
        title_extra=f"T={float(t[0]):.2f}K, scen 24 step 1",
    )
    make_kernel_plot(
        "vaporp_h2o_murphy2005_pvapi",
        np.asarray(pvapi_jax), pvapi_fortran,
        _PLOT_DIR_SCEN_24,
        x_label="z-level", y_label="pvapi over ice [dyne/cm²]",
        title_extra=f"T={float(t[0]):.2f}K, scen 24 step 1",
    )

    assert_kernel_match("vaporp_h2o_murphy2005:pvapl",
                         pvapl_jax, pvapl_fortran)
    assert_kernel_match("vaporp_h2o_murphy2005:pvapi",
                         pvapi_jax, pvapi_fortran)
