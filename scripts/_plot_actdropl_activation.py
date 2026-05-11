"""Activation diagram for actdropl.

Shows where the constant-rate kludge fires across (bin radius, supsatl)
space at a warm temperature. The activation boundary is set by the
per-bin critical Köhler supersaturation `scrit`, which decreases with
bin size (larger droplets activate at lower supersaturation).

Since JAX bit-matches the gfortran standalone (10/10 unit tests, no
FP arithmetic), the informative figure here is the activation pattern
itself — analogous to the gate diagram for freezdropl/melticel.
"""
import math
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from carma.nucleation.actdropl import actdropl

ROOT = Path(__file__).parent.parent

NBIN = 16
RMIN_CM, RMRAT, RHO = 1.0e-7, 4.0, 1.78
vmin = (4.0 / 3.0) * math.pi * RMIN_CM**3 * RHO
rmass = vmin * RMRAT ** np.arange(NBIN)
r_bins = (3.0 * rmass / (4.0 * math.pi * RHO)) ** (1.0 / 3.0)
d_nm = 2.0 * r_bins * 1e7

# Realistic per-bin scrit: decreases with size (Köhler theory — larger
# particles activate at lower critical supersaturation).
scrit = np.geomspace(1e-2, 1e-5, NBIN)
pc = np.full(NBIN, 1e3)
target_evap = np.zeros(NBIN)

supsatl_range = np.logspace(-5, -1, 100)        # 10⁻⁵ to 0.1
activation = np.zeros((supsatl_range.shape[0], NBIN))

for i, s in enumerate(supsatl_range):
    out = actdropl(
        t_val=jnp.float64(280.0),                  # warm
        supsatl_val=jnp.float64(s),
        pconmax_val=jnp.float64(1e3),
        scrit_bins=jnp.asarray(scrit),
        pc_bins=jnp.asarray(pc),
        evappe_target_bins=jnp.asarray(target_evap),
    )
    activation[i, :] = np.asarray(out)

fig, ax = plt.subplots(1, 1, figsize=(11, 5))
# Map shows: pixel = (supsatl, bin); value = rnuclg (0 or 1000)
im = ax.pcolormesh(d_nm, supsatl_range, activation,
                    cmap="Greens", vmin=0, vmax=1000, shading="auto")
# Overlay scrit line
ax.plot(d_nm, scrit, "r-", lw=2, label="scrit per bin (Köhler critical)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Bin median diameter [nm]")
ax.set_ylabel("supsatl")
ax.set_title("Phase 11.9: actdropl activation map @ T=280K  "
             "(green = rate fires; red line = per-bin scrit)\n"
             "JAX bit-exact match to gfortran standalone (10/10 unit tests)")
plt.colorbar(im, ax=ax, label="rnuclg [s⁻¹]")
ax.legend(loc="upper left", fontsize=10)
ax.grid(True, alpha=0.3, which="both")

plt.tight_layout()
out_dir = ROOT / "plots" / "diff" / "phase11"
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "actdropl_activation.png"
plt.savefig(out, dpi=110, bbox_inches="tight")
print(f"saved {out}")
