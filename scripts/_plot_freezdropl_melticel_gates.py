"""Gate-activation diagram for freezdropl + melticel.

These are placeholder "constant rate when T-gate fires" kernels — the
upstream Fortran sources self-identify as "temporary simple kludges".
Bit-exact JAX/Fortran agreement is verified by the unit tests; the
informative figure here is the gate pattern itself.
"""
import math
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from carma.constants import T0
from carma.nucleation.freezdropl import freezdropl
from carma.nucleation.melticel import melticel

ROOT = Path(__file__).parent.parent
T_vals = np.linspace(200.0, 320.0, 601)

# freezdropl: per-bin pc gate; sweep T with a dense bin.
pc_dense = jnp.array([1e3])
rate_freez = np.array([float(freezdropl(jnp.float64(T), pc_dense)[0])
                        for T in T_vals])

# melticel: per-group pconmax gate.
rate_melt = np.array([float(melticel(jnp.float64(T), jnp.float64(1e3), 1)[0])
                       for T in T_vals])

fig, ax = plt.subplots(1, 1, figsize=(11, 5))
ax.plot(T_vals, rate_freez, "b-", lw=2, label="freezdropl  (T < T₀ − 40)")
ax.plot(T_vals, rate_melt,  "r-", lw=2, label="melticel    (T > T₀)")
ax.axvline(float(T0) - 40, color="b", ls=":", lw=1, alpha=0.6)
ax.axvline(float(T0),       color="r", ls=":", lw=1, alpha=0.6)
ax.set_xlabel("T [K]")
ax.set_ylabel("rnuclg per bin [s⁻¹]")
ax.set_title("Phase 11.6: freezdropl + melticel — constant-rate gate kernels\n"
             "(JAX exact-match to gfortran-compiled standalone; 14/14 unit tests pass)")
ax.grid(True, alpha=0.3)
ax.legend(loc="center left", fontsize=10)

# Annotate the gate boundaries
ax.annotate(f"T₀ − 40 = {float(T0) - 40:.2f} K",
             xy=(float(T0) - 40, 50), xytext=(float(T0) - 70, 70),
             fontsize=9, color="b",
             arrowprops=dict(arrowstyle="->", color="b", alpha=0.5))
ax.annotate(f"T₀ = {float(T0):.2f} K",
             xy=(float(T0), 50), xytext=(float(T0) + 10, 30),
             fontsize=9, color="r",
             arrowprops=dict(arrowstyle="->", color="r", alpha=0.5))

plt.tight_layout()
out_dir = ROOT / "plots" / "diff" / "phase11"
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "freezdropl_melticel_gates.png"
plt.savefig(out, dpi=110, bbox_inches="tight")
print(f"saved {out}")
