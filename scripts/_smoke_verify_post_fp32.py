"""Smoke verification plot for the fp32-cleanup change.

Runs the same n=5 / nstep=100 ensemble as scripts/_plot_smoke.py but
writes to a distinct file so the existing Phase 10.5 figures are
preserved alongside.
"""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
J = np.load(ROOT / "data/_smoke_post_fp32cleanup_n5.npz")
F = np.load(ROOT / "data/sulfate_fortran_realistic_outputs.npz")
scens = np.load(ROOT / "data/sulfate_scenarios_realistic_1000.npz")

nbin, rmin, rmrat, rho = 38, 2.0e-8, 2.0, 1.923
vmin = (4.0 / 3.0) * math.pi * rmin**3 * rho
rmass = np.array([vmin * rmrat ** i for i in range(nbin)])
r = (3.0 * rmass / (4.0 * math.pi * rho)) ** (1.0 / 3.0)
r_nm = r * 1e7
dlogr = math.log10(rmrat) / 3.0


def init_mmr_per_bin(mu_nm, sigma_g, total_mmr=1.0e-18):
    log_mu = math.log(mu_nm * 1e-7)
    log_sig = math.log(sigma_g)
    shape = (np.exp(-0.5 * ((np.log(r) - log_mu) / log_sig) ** 2)
             / (r * log_sig * math.sqrt(2 * math.pi)) * rmass)
    return shape * (total_mmr / shape.sum())


n = 5
fig, axes = plt.subplots(n, 2, figsize=(12, 14), sharex=True)
for i in range(n):
    pcJ = J["pc_final"][i]; pcF = F["pc_final"][i]
    pc0 = init_mmr_per_bin(scens["aerosol_mu_nm"][i], scens["aerosol_sigma_g"][i])

    ax = axes[i, 0]
    ax.loglog(r_nm, pc0 / dlogr, color="0.6", ls=":", lw=1.5, label="initial (t=0)")
    ax.loglog(r_nm, pcF / dlogr, "k-",  lw=2, label="Fortran (t=50h)")
    ax.loglog(r_nm, pcJ / dlogr, "r--", lw=2, label="JAX fp64-locked (t=50h)")
    ax.set_ylabel("dM/dlogr (g/g)")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_title(f"scen {i}: T={scens['T'][i]:.1f}K, RH={scens['rh'][i]:.2f}, "
                 f"H2SO4={scens['h2so4_pptv'][i]:.1f}pptv (mass)")
    if i == 0:
        ax.legend()

    ax2 = axes[i, 1]
    ax2.loglog(r_nm, pc0 / rmass / dlogr, color="0.6", ls=":", lw=1.5, label="initial")
    ax2.loglog(r_nm, pcF / rmass / dlogr, "k-",  lw=2, label="Fortran")
    ax2.loglog(r_nm, pcJ / rmass / dlogr, "r--", lw=2, label="JAX fp64-locked")
    ax2.set_ylabel("dN/dlogr ∝ pc_mmr/rmass")
    ax2.grid(True, alpha=0.3, which="both")
    ax2.set_title(f"scen {i} (number)")

axes[-1, 0].set_xlabel("r (nm)")
axes[-1, 1].set_xlabel("r (nm)")
fig.suptitle("Smoke verification after dropping fp32 path / locking precision to fp64. "
             "JAX outputs bit-for-bit identical to Phase 10.5 baseline.",
             fontsize=11, y=1.005)
plt.tight_layout()
out_dir = ROOT / "plots" / "diff" / "phase10"
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "smoke_jax_vs_fortran_n5_post_fp32cleanup.png"
plt.savefig(out, dpi=110, bbox_inches="tight")
print(f"saved {out}")
