"""Run JAX coag-only on realistic scenarios (matches Fortran's H2SO4=0 setup)."""
import sys, time
from pathlib import Path
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests" / "unit"))
from generate_sulfate_scenarios import load_scenarios
from jax_ensemble import _fortran_matching_config, _env_for_scenario, _compute_ppm_coefs
from carma.step import make_step_coag

def main():
    cfg = _fortran_matching_config()
    ppm = _compute_ppm_coefs(cfg)
    step_coag = make_step_coag(cfg)
    sc = load_scenarios(ROOT / "data" / "sulfate_scenarios_realistic_1000.npz")
    n = int(sc["_n"])
    rmass = np.asarray(cfg.groups[0].rmass)
    pc_all = np.zeros((n, 38), dtype=np.float64)

    print(f"Running JAX coag-only on {n} scenarios (1800s × 100, H2SO4=0)...")
    t0 = time.perf_counter()
    for i in range(n):
        # Force H2SO4=0 by zeroing gc[1], pass through other params
        env = _env_for_scenario(sc["T"][i], sc["p"][i], sc["rh"][i],
                                  0.0, sc["aerosol_mu_nm"][i], sc["aerosol_sigma_g"][i],
                                  cfg, ppm)
        pc = env["pc"]
        gc = env["gc"]
        t = env["t"]
        pcl = pc; gcl = gc; told = t
        ckernel = env["ckernel"]
        zmet = env["zmet"]
        for _ in range(100):
            pc, gc, t, pcl, gcl, _ = step_coag(pc, gc, t, pcl, gcl, told, zmet, ckernel, 1800.0)
            told = t
        rhoa = float(env["rhoa"][0])
        pc_all[i] = np.asarray(pc[0, :, 0]) * rmass / rhoa
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n}", flush=True)
    wall = time.perf_counter() - t0
    print(f"Wall: {wall:.1f}s")
    np.savez_compressed(ROOT / "data" / "sulfate_jax_coagonly.npz", pc_final=pc_all)
    print(f"Saved data/sulfate_jax_coagonly.npz")

if __name__ == "__main__":
    main()
