"""Per-scenario diff between JAX env construction and Fortran dumps.

Builds the JAX env (same code path as jax_ensemble.py) for each
scenario, loads the corresponding Fortran probe dump from
data/diff/scen_<NNN>/, and reports max relative error per array.

This isolates env-builder bugs from the (already-bench-validated)
microfast / microslow kernels.
"""
import math
import sys
from pathlib import Path

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from jax_ensemble import (
    _fortran_matching_config, _compute_ppm_coefs, _env_for_scenario,
)
from carma._diag.fortran_reader import read_substep


def rel_err(a, b):
    a = np.asarray(a).astype(np.float64).ravel()
    b = np.asarray(b).astype(np.float64).ravel()
    if a.shape != b.shape:
        return f"SHAPE_MISMATCH a={a.shape} b={b.shape}"
    denom = np.maximum(np.abs(a), np.abs(b))
    denom = np.where(denom > 1e-300, denom, 1.0)
    return float(np.max(np.abs(a - b) / denom))


def diff_one(i, scens, cfg, ppm_coefs):
    env = _env_for_scenario(
        scens["T"][i], scens["p"][i], scens["rh"][i],
        scens["h2so4_pptv"][i], scens["aerosol_mu_nm"][i],
        scens["aerosol_sigma_g"][i], cfg, ppm_coefs,
    )
    scen_dir = ROOT / "data" / "diff" / f"scen_{i:03d}"
    if not scen_dir.exists():
        return None
    d = read_substep(scen_dir, step=1)

    # JAX env arrays (squeezed to match Fortran's NZ=1)
    out = {}
    out["t"]       = rel_err(np.asarray(env["t"]),       d.t)
    out["rhoa"]    = rel_err(np.asarray(env["rhoa"]),    d.rhoa)
    out["zmet"]    = rel_err(np.asarray(env["zmet"]),    d.zmet)
    out["akelvin"] = rel_err(np.asarray(env["akelvin"]), d.akelvin)
    out["akelvini"]= rel_err(np.asarray(env["akelvini"]),d.akelvini)
    out["gro"]     = rel_err(np.asarray(env["gro"]),     d.gro)
    out["gro1"]    = rel_err(np.asarray(env["gro1"]),    d.gro1)
    out["rup_wet"] = rel_err(np.asarray(env["rup_wet"]), d.rup_wet)
    out["rlhe"]    = rel_err(np.asarray(env["rlhe"]),
                             np.fromfile(scen_dir / "substep_0001_rlhe_probe.bin",
                                         dtype=np.float64).reshape(1, 2))
    out["rlhm"]    = rel_err(np.asarray(env["rlhm"]),
                             np.fromfile(scen_dir / "substep_0001_rlhm_probe.bin",
                                         dtype=np.float64).reshape(1, 2))
    out["ckernel"] = rel_err(np.asarray(env["ckernel"]), d.ckernel)
    # Initial pc / gc parity (against Fortran's pcl/gcl which is the
    # state at start of step 1's substep loop, i.e., after prestep)
    out["pc_init"] = rel_err(np.asarray(env["pc"]).ravel(), d.pcl.ravel())
    out["gc_init"] = rel_err(np.asarray(env["gc"]).ravel(), d.gcl.ravel())
    # r_wet (per bin centre, for setup_vf and setup_ckern). The dump's
    # r_wet has shape (NZ, NBIN, NGROUP); env stores rup_wet not r_wet —
    # we sneak r_wet in directly:
    return out


def main():
    cfg = _fortran_matching_config()
    ppm_coefs = _compute_ppm_coefs(cfg)
    scens = np.load(ROOT / "data" / "sulfate_scenarios_realistic_1000.npz")

    n_check = 25
    print(f"Diffing JAX env vs Fortran dump on first {n_check} scenarios...\n")

    fields = ["t", "rhoa", "zmet", "akelvin", "akelvini", "gro", "gro1",
              "rup_wet", "rlhe", "rlhm", "ckernel", "pc_init", "gc_init"]
    print(f"{'scen':>5}", *(f"{f:>10}" for f in fields))
    all_errs = {f: [] for f in fields}
    for i in range(n_check):
        result = diff_one(i, scens, cfg, ppm_coefs)
        if result is None:
            print(f"{i:>5}  (no dump)")
            continue
        row = [f"{i:>5}"]
        for f in fields:
            v = result[f]
            row.append(f"{v:>10.2e}" if isinstance(v, float) else f"{v:>10s}")
            if isinstance(v, float):
                all_errs[f].append(v)
        print(*row)

    print("\nSummary (max over scenarios):")
    for f in fields:
        if all_errs[f]:
            mx = max(all_errs[f])
            print(f"  {f:<10}: max={mx:.3e}")


if __name__ == "__main__":
    main()
