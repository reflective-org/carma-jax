"""Plot Fortran vs JAX bin distributions at dt=60s and dt=1800s.

Shows per-bin median across n=100 scenarios for:
  - Fortran 60s (3000 steps)
  - Fortran 1800s (100 steps)  
  - JAX 60s (3000 steps)
  - JAX 1800s (100 steps)
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path('/Users/ali/Documents/GitHub/aerosol-microphysics/carma-app/carma-jax')
N = 100  # JAX 60s only ran 100 scenarios

f60 = np.load(ROOT / 'data' / 'sulfate_fortran_realistic_60s.npz')
f1800 = np.load(ROOT / 'data' / 'sulfate_fortran_realistic_outputs.npz')
j60 = np.load(ROOT / 'data' / 'sulfate_jax_realistic_60s_n100.npz')
j1800 = np.load(ROOT / 'data' / 'sulfate_jax_realistic_outputs.npz')

pcF60 = f60['pc_final'][:N]
pcF1800 = f1800['pc_final'][:N]
pcJ60 = j60['pc_final']
pcJ1800 = j1800['pc_final'][:N]

# Bin radii in nm
nbin = 38
rho = 1.923; rmin = 2e-8; rmrat = 2.0
vmin = (4/3)*np.pi*rmin**3*rho
rmass = vmin * rmrat**np.arange(nbin)
r_nm = (3*rmass/(4*np.pi*rho))**(1/3) * 1e7

# Plot per-bin median
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

# Left: dt=1800s
ax = axes[0]
ax.semilogy(r_nm, np.median(pcF1800, axis=0), 'k-', lw=2, label='Fortran 1800s')
ax.semilogy(r_nm, np.median(pcJ1800, axis=0), 'r--', lw=2, label='JAX 1800s')
ax.set_xscale('log')
ax.set_xlabel('Bin radius (nm)')
ax.set_ylabel('mmr per bin (g/g) — median over n=100')
ax.set_title('dt = 1800 s × 100 steps (CARMA default)')
ax.legend(loc='lower center')
ax.grid(True, alpha=0.3)
ax.set_ylim(1e-30, 1e-10)

# Right: dt=60s
ax = axes[1]
ax.semilogy(r_nm, np.median(pcF60, axis=0), 'k-', lw=2, label='Fortran 60s')
ax.semilogy(r_nm, np.median(pcJ60, axis=0), 'r--', lw=2, label='JAX 60s')
ax.set_xscale('log')
ax.set_xlabel('Bin radius (nm)')
ax.set_title('dt = 60 s × 3000 steps')
ax.legend(loc='lower center')
ax.grid(True, alpha=0.3)
ax.set_ylim(1e-30, 1e-10)

fig.suptitle('Median bin distribution — Fortran is dt-invariant, JAX collapses to top bin at fine dt',
             fontsize=12, y=1.02)
fig.tight_layout()
out = ROOT / 'plots' / 'phase10_parity' / 'fig5_dt_comparison.png'
fig.savefig(out, dpi=130, bbox_inches='tight')
print(f'Saved: {out}')

# Also a 4-line overlay
fig2, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(r_nm, np.median(pcF1800, axis=0), 'k-', lw=2.5, label='Fortran 1800s')
ax.semilogy(r_nm, np.median(pcF60, axis=0), 'k:', lw=2, label='Fortran 60s (overlaid)')
ax.semilogy(r_nm, np.median(pcJ1800, axis=0), 'b-', lw=2, label='JAX 1800s')
ax.semilogy(r_nm, np.median(pcJ60, axis=0), 'r--', lw=2, label='JAX 60s')
ax.set_xscale('log')
ax.set_xlabel('Bin radius (nm)', fontsize=11)
ax.set_ylabel('mmr per bin (g/g) — median over n=100 realistic scenarios', fontsize=11)
ax.set_title('dt sensitivity: Fortran identical at 60s/1800s; JAX diverges at 60s')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim(1e-30, 1e-10)
ax.axvline(r_nm[37], color='gray', ls='--', alpha=0.5)
ax.text(r_nm[37]*0.7, 1e-12, 'top bin\n(1 µm)', ha='right', fontsize=9, color='gray')
out2 = ROOT / 'plots' / 'phase10_parity' / 'fig6_dt_overlay.png'
fig2.savefig(out2, dpi=130, bbox_inches='tight')
print(f'Saved: {out2}')

# Per-scenario heatmap-style: how much each scenario shifts to top
fig3, axes = plt.subplots(2, 2, figsize=(14, 9))
configs = [
    ('Fortran 1800s', pcF1800, axes[0, 0]),
    ('Fortran 60s', pcF60, axes[0, 1]),
    ('JAX 1800s', pcJ1800, axes[1, 0]),
    ('JAX 60s', pcJ60, axes[1, 1]),
]
for name, pc, ax in configs:
    # Normalize each scenario to its total (so we see shape, not magnitude)
    safe_tot = np.maximum(pc.sum(axis=1, keepdims=True), 1e-50)
    norm = pc / safe_tot
    im = ax.imshow(np.log10(np.maximum(norm, 1e-10)),
                    aspect='auto', origin='lower', cmap='viridis',
                    vmin=-6, vmax=0)
    ax.set_title(name, fontsize=11)
    ax.set_xlabel('Bin index')
    ax.set_ylabel('Scenario')
fig3.colorbar(im, ax=axes, shrink=0.7, label='log10(mass fraction in bin)')
fig3.suptitle('Per-scenario bin distributions (normalized to total)', fontsize=12)
out3 = ROOT / 'plots' / 'phase10_parity' / 'fig7_per_scenario_heatmap.png'
fig3.savefig(out3, dpi=130, bbox_inches='tight')
print(f'Saved: {out3}')
