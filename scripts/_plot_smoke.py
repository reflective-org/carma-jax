"""Quick plot: per-bin mass and number distributions, JAX vs Fortran.

Reads the n=5 smoke JAX output and the full Fortran ensemble output,
plots dN/dlogr and dM/dlogr per bin for the first N scenarios.
"""
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent

J = np.load(ROOT / "data/_smoke_jax_n5_s100.npz")
F = np.load(ROOT / "data/sulfate_fortran_realistic_outputs.npz")
scens = np.load(ROOT / "data/sulfate_scenarios_realistic_1000.npz")

# Fortran-matching bin grid (same constants jax_ensemble uses)
nbin, rmin, rmrat, rho = 38, 2.0e-8, 2.0, 1.923
vmin = (4.0 / 3.0) * math.pi * rmin**3 * rho
rmass = np.array([vmin * rmrat**i for i in range(nbin)])
r = (3.0 * rmass / (4.0 * math.pi * rho)) ** (1.0 / 3.0)    # cm
r_nm = r * 1e7
dlogr = np.log10(rmrat) / 3.0           # constant per bin

def init_mmr_per_bin(mu_nm, sigma_g, total_mmr=1.0e-18):
    """Initial sulfate aerosol mmr per bin (Fortran's seed convention,
    `carma_sulfatetest_ensemble.F90:194-200`): mass-weighted lognormal
    in r, normalised so Σ_bin mmr = total_mmr."""
    log_mu = math.log(mu_nm * 1e-7)
    log_sig = math.log(sigma_g)
    shape = (np.exp(-0.5 * ((np.log(r) - log_mu) / log_sig) ** 2)
             / (r * log_sig * math.sqrt(2 * math.pi)) * rmass)
    return shape * (total_mmr / shape.sum())


n = 5
fig, axes = plt.subplots(n, 2, figsize=(12, 14), sharex=True)
for i in range(n):
    pcJ = J["pc_final"][i]            # mmr per bin (g/g)
    pcF = F["pc_final"][i]
    pc0 = init_mmr_per_bin(scens["aerosol_mu_nm"][i],
                           scens["aerosol_sigma_g"][i])
    # mass per bin: dM/dlogr in g/g per decade
    mass_J = pcJ / dlogr
    mass_F = pcF / dlogr
    mass_0 = pc0 / dlogr
    # number per bin: pc_mmr / rmass × rhoa_air, but we don't have rhoa in
    # the saved output. Fortran outputs are MMR per bin so we plot
    # MMR/rmass which is proportional to number; up to a constant rhoa
    # factor the *shape* is what matters.
    num_J = pcJ / rmass / dlogr
    num_F = pcF / rmass / dlogr
    num_0 = pc0 / rmass / dlogr

    ax = axes[i, 0]
    ax.loglog(r_nm, mass_0, color="0.6", ls=":", lw=1.5, label="initial (t=0)")
    ax.loglog(r_nm, mass_F, "k-", lw=2, label="Fortran (t=50h)")
    ax.loglog(r_nm, mass_J, "r--", lw=2, label="JAX (t=50h)")
    ax.set_ylabel("dM/dlogr (g/g)")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_title(f"scen {i}: T={scens['T'][i]:.1f}K, RH={scens['rh'][i]:.2f}, "
                 f"H2SO4={scens['h2so4_pptv'][i]:.1f}pptv (mass)")
    if i == 0:
        ax.legend()

    ax2 = axes[i, 1]
    ax2.loglog(r_nm, num_0, color="0.6", ls=":", lw=1.5, label="initial (t=0)")
    ax2.loglog(r_nm, num_F, "k-", lw=2, label="Fortran (t=50h)")
    ax2.loglog(r_nm, num_J, "r--", lw=2, label="JAX (t=50h)")
    ax2.set_ylabel("dN/dlogr  ∝  pc_mmr/rmass")
    ax2.grid(True, alpha=0.3, which="both")
    ax2.set_title(f"scen {i} (number)")

axes[-1, 0].set_xlabel("r (nm)")
axes[-1, 1].set_xlabel("r (nm)")
plt.tight_layout()
out_dir = ROOT / "plots" / "diff" / "phase10"
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "smoke_jax_vs_fortran_n5.png"
plt.savefig(out, dpi=110, bbox_inches="tight")
print(f"saved {out}")
