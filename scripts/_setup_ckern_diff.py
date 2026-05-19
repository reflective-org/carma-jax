"""Bench JAX setup_ckern using Fortran-dumped inputs."""
import sys
from pathlib import Path
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from carma._diag.fortran_reader import read_substep
from carma.setup_ckern import setup_ckern_jit
from carma.precision import DTYPE


def rel(a, b):
    a, b = np.ravel(a), np.ravel(b)
    denom = np.maximum(np.abs(a), np.abs(b))
    denom = np.where(denom > 1e-300, denom, 1.0)
    return float(np.max(np.abs(a - b) / denom))


def main():
    NBIN, NGROUP = 38, 1
    rrat_2d = jnp.ones((NBIN, NGROUP))
    rprat_2d = jnp.ones((NBIN, NGROUP))
    cstick = DTYPE(1.0)

    errs = []
    for i in range(25):
        scen_dir = ROOT / "data" / "diff" / f"scen_{i:03d}"
        if not scen_dir.exists():
            continue
        d = read_substep(scen_dir, step=1)
        rmass_2d = jnp.asarray(d.rmass_bin)
        ck_J = setup_ckern_jit(
            jnp.asarray(d.t), jnp.asarray(d.rhoa), jnp.asarray(d.zmet),
            jnp.asarray(d.rmu), jnp.asarray(d.r_wet),
            rrat_2d, rprat_2d, jnp.asarray(d.bpm),
            rmass_2d, jnp.asarray(d.re), jnp.asarray(d.vf),
            cstick,
        )
        e = rel(ck_J, d.ckernel)
        errs.append(e)
        if i < 3 or e > 1e-3:
            print(f"scen {i}: ckernel rel err = {e:.3e}")

    print(f"\nSummary: max={max(errs):.3e}  median={np.median(errs):.3e}  min={min(errs):.3e}")


if __name__ == "__main__":
    main()
