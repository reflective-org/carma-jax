"""Inspect the actual values for one scenario where rel err is large."""
import sys
from pathlib import Path

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from jax_ensemble import (
    _fortran_matching_config, _compute_ppm_coefs, _env_for_scenario,
)
from carma._diag.fortran_reader import read_substep

cfg = _fortran_matching_config()
ppm_coefs = _compute_ppm_coefs(cfg)
scens = np.load(ROOT / "data" / "sulfate_scenarios_realistic_1000.npz")

i = 0
env = _env_for_scenario(
    scens["T"][i], scens["p"][i], scens["rh"][i],
    scens["h2so4_pptv"][i], scens["aerosol_mu_nm"][i],
    scens["aerosol_sigma_g"][i], cfg, ppm_coefs,
)
d = read_substep(ROOT / "data" / "diff" / f"scen_{i:03d}", step=1)

print(f"Scenario {i}: T={scens['T'][i]:.2f}K, p={scens['p'][i]:.2f}hPa, "
      f"rh={scens['rh'][i]:.3f}, h2so4={scens['h2so4_pptv'][i]:.2f}pptv\n")

print("=" * 70)
print("rhoa (JAX vs Fortran):")
print(f"  JAX:     {float(env['rhoa'][0]):.10e}")
print(f"  Fortran: {float(d.rhoa[0]):.10e}")
print(f"  rel: {abs(float(env['rhoa'][0]) - float(d.rhoa[0])) / float(d.rhoa[0]):.2e}")

print("\n" + "=" * 70)
print("pc_init (JAX env['pc'] vs Fortran d.pcl): mass per bin (g/cm^3/z)")
print(f"  shape JAX: {np.asarray(env['pc']).shape}, F: {d.pcl.shape}")
print(f"  JAX sum:     {float(np.sum(env['pc'])):.4e}")
print(f"  Fortran sum: {float(np.sum(d.pcl)):.4e}")
print(f"  ratio J/F: {float(np.sum(env['pc'])) / float(np.sum(d.pcl)):.3f}")
print(f"  bin    JAX            Fortran        rel-err")
pc_J = np.asarray(env['pc']).ravel()[:38]
pc_F = d.pcl.ravel()[:38]
for ib in range(38):
    rel = abs(pc_J[ib] - pc_F[ib]) / max(abs(pc_F[ib]), abs(pc_J[ib]), 1e-300)
    flag = " *" if rel > 1e-3 else ""
    print(f"  {ib:3d}  {pc_J[ib]:.4e}    {pc_F[ib]:.4e}    {rel:.2e}{flag}")

print("\n" + "=" * 70)
print("akelvin (JAX vs Fortran): NGAS=2 → [H2O, H2SO4]")
ak_J = np.asarray(env['akelvin']).ravel()
ak_F = d.akelvin.ravel()
print(f"  JAX:     {ak_J}")
print(f"  Fortran: {ak_F}")
print(f"  rel:     {(ak_J - ak_F) / ak_F}")

print("\n" + "=" * 70)
print("akelvini (JAX vs Fortran):")
ak_J = np.asarray(env['akelvini']).ravel()
ak_F = d.akelvini.ravel()
print(f"  JAX:     {ak_J}")
print(f"  Fortran: {ak_F}")
print(f"  rel:     {(ak_J - ak_F) / np.where(np.abs(ak_F)>1e-300, ak_F, 1.0)}")

print("\n" + "=" * 70)
print("rup_wet (JAX vs Fortran): per bin (cm)")
ru_J = np.asarray(env['rup_wet']).ravel()
ru_F = d.rup_wet.ravel()
print(f"  bin    JAX            Fortran        rel-err")
for ib in range(0, 38, 4):
    rel = abs(ru_J[ib] - ru_F[ib]) / max(abs(ru_F[ib]), 1e-300)
    print(f"  {ib:3d}  {ru_J[ib]:.4e}    {ru_F[ib]:.4e}    {rel:.2e}")

print("\n" + "=" * 70)
print("rlhm (JAX vs Fortran): NGAS=2")
rh_J = np.asarray(env['rlhm']).ravel()
rh_F = np.fromfile(ROOT / "data" / "diff" / f"scen_{i:03d}" / "substep_0001_rlhm_probe.bin",
                    dtype=np.float64)
print(f"  JAX:     {rh_J}")
print(f"  Fortran: {rh_F}")

print("\n" + "=" * 70)
print("gro at peak bin (10):")
gro_J = np.asarray(env['gro']).ravel()
gro_F = d.gro.ravel()
print(f"  shape JAX: {np.asarray(env['gro']).shape}, F: {d.gro.shape}")
for ib in [0, 5, 10, 15, 20, 30, 37]:
    print(f"  bin {ib}: JAX={gro_J[ib]:.4e}  F={gro_F[ib]:.4e}  rel={abs(gro_J[ib]-gro_F[ib])/max(abs(gro_F[ib]),1e-300):.2e}")
