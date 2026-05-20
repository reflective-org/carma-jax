"""Condensation-only validation: fixed [H2SO4] = 1e7 molec/cm^3.

Textbook condensation problem (per Pierce/Adams class-style):
  Aerosol:      N0   = 1.0e4 #/cm^3
                GMD  = 20 nm
                GSD  = 1.60
  Environment:  T    = 298 K
                p    = 101325 Pa
                RH   = 30 %
  Gas:          [H2SO4] = 1e7 molec/cm^3 — HELD CONSTANT
  Mode:         Condensation only (no nucleation, no coagulation,
                                    no gas-phase evolution)
  Run:          dt = 60 s × 1440 steps = 24 h

Expected behaviour:
  - Particles in the kinetic regime (Dp < ~50 nm) grow by ~20 nm in 24 h.
  - Larger particles (~100 nm, entering transition regime) grow somewhat less.
  - Total particle number is conserved (no nucleation, no coag).
  - The whole size distribution shifts to the right.

The RHS is a stripped-down sulfate diffrax variant:
  - Upwind growth/evap in mass space (same as production)
  - Bin-0 evap-out sink (same as production)
  - **No nucleation**
  - **dgc/dt = 0** (gas held fixed)
  - **dT/dt = 0** (isothermal — latent-heat coupling off; negligible at
                    these particle masses anyway)

Output:
  benchmark_final/plots/condensation_only/validation.png
"""
import math
import sys
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import diffrax
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "scripts"))


# ------------------------------------------------------------
# Constants for the controlled scenario
# ------------------------------------------------------------
T_K = 298.0
P_Pa = 101325.0
RH = 0.30
H2SO4_MOL_CM3 = 1.0e7      # held constant
N0 = 1.0e4                 # #/cm^3 total particle number
GMD_NM = 20.0
GSD = 1.60
DTIME = 60.0
NSTEP = 1440
GWTMOL_H2SO4 = 98.078479
GWTMOL_H2O = 18.01528
AVG = 6.02252e23
RHO_SULF = 1.923           # g/cm^3
CP_AIR = 1.004e7           # erg/g/K (CGS)
RPA2CGS = 10.0


def build_scenario():
    """Return cfg, ppm, env, pc0, gc_fixed, T0 for the scenario."""
    from jax_ensemble import _minimal_config, _compute_ppm_coefs, _refresh_env
    from carma.precision import DTYPE

    cfg = _minimal_config()
    ppm = _compute_ppm_coefs(cfg)
    grp = cfg.groups[0]

    p_cgs_val = P_Pa * RPA2CGS                # dyn/cm^2 in CGS
    rhoa_g_cm3 = p_cgs_val / (8.314e7 * T_K / 29.0)   # ~ 1.185e-3 g/cm^3
    p_cgs = jnp.asarray([p_cgs_val], dtype=DTYPE)
    T_arr = jnp.asarray([T_K], dtype=DTYPE)

    # Convert [H2SO4] = 1e7 molec/cm^3 → mass conc → "gc state" (mmr × rhoa)
    h2so4_g_cm3 = H2SO4_MOL_CM3 * GWTMOL_H2SO4 / AVG
    # H2O at 30% RH (background, also held fixed in our RHS but seeded properly)
    # Murphy 2005 vapor pressure at 298 K:
    pvapl_Pa = math.exp(54.842763 - 6763.22 / T_K
                         - 4.210 * math.log(T_K) + 0.000367 * T_K)
    h2o_mmr = RH * pvapl_Pa * GWTMOL_H2O / (29.0 * P_Pa)
    h2o_g_cm3 = h2o_mmr * rhoa_g_cm3
    gc = jnp.asarray([[h2o_g_cm3, h2so4_g_cm3]], dtype=DTYPE)

    # Build a frozen env at this state
    env_d = _refresh_env(T_arr, p_cgs, gc, cfg, ppm)

    # Initial lognormal seed: N=N0 #/cm^3, GMD=20nm, GSD=1.6
    r = np.asarray(grp.r)
    rmass_np = np.asarray(grp.rmass)
    rmassup_np = np.asarray(grp.rmassup)
    rmasslow_np = np.concatenate([[rmass_np[0]/(grp.rmrat**0.5)], rmassup_np[:-1]])
    dm = jnp.asarray(rmassup_np - rmasslow_np, dtype=DTYPE)

    log_mu_cm = math.log(GMD_NM * 1e-7)
    log_sigma = math.log(GSD)
    # pc[bin] = (N0 / ln(rmrat)) × pdf (in log-D space) per bin
    # Simpler: pc[bin] = N0 × (fraction of total in bin)
    # Use dlogD integration:
    d_nm = 2.0 * (3.0 * rmass_np / (4*np.pi*RHO_SULF))**(1/3) * 1e7
    dlog10_d = math.log10(grp.rmrat) / 3.0   # since D ∝ m^(1/3)
    # dN/dlog10(D) for lognormal: N0/(sqrt(2π) log10(GSD)) × exp(-((log10(D)-log10(GMD))^2)/(2 log10(GSD)^2))
    log10_gsd = math.log10(GSD)
    log10_gmd = math.log10(GMD_NM)
    dN_dlogD = (N0 / (math.sqrt(2*math.pi) * log10_gsd)
                * np.exp(-((np.log10(d_nm) - log10_gmd) ** 2)
                          / (2 * log10_gsd ** 2)))
    pc_per_bin = dN_dlogD * dlog10_d         # #/cm^3 per bin
    # Verify total ≈ N0
    print(f"  built initial pc sum = {pc_per_bin.sum():.3e}  (target {N0:.3e})")

    pc0 = jnp.asarray(pc_per_bin, dtype=DTYPE)[:, None]   # (nbin, nelem=1)

    return cfg, ppm, env_d, dm, pc0, gc[0], T_arr[0]


def make_condensation_rhs(env, dm, shape):
    """RHS variant: condensation only, gc and T held fixed."""
    from carma.growth.pheat import pheat
    from carma.supersaturation import supersat
    from carma.vapor_pressure import vaporp_h2o_murphy2005, vaporp_h2so4_ayers1980
    from carma.precision import DTYPE

    nbin = shape.nbin
    ig = 0       # only sulfate group
    iz = 0

    @jax.jit
    def rhs(t, y, _args):
        del t, _args
        pc_flat = y[:shape.n_pc]
        pc = pc_flat.reshape(nbin, shape.nelem)
        gc = y[shape.n_pc:shape.n_pc + shape.ngas]
        T_scalar = y[shape.n_pc + shape.ngas]
        T = jnp.atleast_1d(T_scalar)
        gc_2d = gc[None, :]
        pc_3d = pc[None, :, :]

        # Vapor pressure + supersat from CURRENT (fixed) state.
        pvapl_h2o, pvapi_h2o = vaporp_h2o_murphy2005(T)
        pvap_h2so4, _ = vaporp_h2so4_ayers1980(
            T, gc_2d[:, 0], pvapl_h2o, env["zmet"],
        )
        pvapl = jnp.stack([pvapl_h2o, pvap_h2so4], axis=1)
        pvapi = jnp.stack([pvapi_h2o, pvap_h2so4], axis=1)
        ssl_h2o, _ = supersat(T, gc_2d[:, 0], pvapl[:, 0], pvapi[:, 0],
                                DTYPE(GWTMOL_H2O), env["zmet"])
        ssl_h2so4, ssi_h2so4 = supersat(T, gc_2d[:, 1], pvapl[:, 1], pvapi[:, 1],
                                          DTYPE(GWTMOL_H2SO4), env["zmet"])
        supsatl = jnp.stack([ssl_h2o, ssl_h2so4], axis=1)
        supsati = jnp.stack([jnp.zeros_like(ssl_h2o), ssi_h2so4], axis=1)

        # Per-boundary dmdt
        def _dmdt_at(ibin):
            return pheat(
                pc_3d, supsatl, supsati, pvapl, pvapi,
                env["akelvin"], env["akelvini"], env["gro"], env["gro1"],
                env["rup_wet"], jnp.asarray(env["rmass_2d"]),
                False, iz, ig, ibin, 1,   # is_ice=False, igas=H2SO4
            )
        dmdt_inner = jax.vmap(_dmdt_at)(jnp.arange(nbin - 1))
        zero = jnp.zeros((1,), dtype=DTYPE)
        dmdt = jnp.concatenate([dmdt_inner, zero])

        # Upwind transfer
        dm_g = dm
        pc_elem = pc[:, 0]
        pc_above = jnp.concatenate([pc_elem[1:], zero])
        dm_above = jnp.concatenate([dm_g[1:], jnp.ones((1,), dtype=DTYPE)])
        n_at = jnp.where(dmdt > 0, pc_elem / dm_g, pc_above / dm_above)
        F_n = dmdt * n_at
        F_n_below = jnp.concatenate([zero, F_n[:-1]])
        dpc_dt = F_n_below - F_n
        # Bin-0 sink
        evap_bin0 = jnp.where(
            dmdt[0] < 0, -dmdt[0]/dm_g[0] * pc_elem[0], DTYPE(0.0),
        )
        dpc_dt = dpc_dt.at[0].add(-evap_bin0)

        # Pack: dpc/dt for pc_elem, zero for gc and T
        dpc_dt_2d = dpc_dt[:, None]
        dgc_dt = jnp.zeros(shape.ngas, dtype=DTYPE)
        dT_dt = DTYPE(0.0)

        return jnp.concatenate([dpc_dt_2d.reshape(-1), dgc_dt,
                                  jnp.atleast_1d(dT_dt)])

    return rhs


def main():
    from carma.precision import DTYPE
    from carma_diffrax.state import StateShape, pack, unpack
    OUT_DIR = ROOT / "plots" / "condensation_only"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg, ppm, env_d, dm, pc0, gc0, T0 = build_scenario()
    shape = StateShape(nbin=cfg.nbin, nelem=1, ngas=2)
    grp = cfg.groups[0]
    rmass_np = np.asarray(grp.rmass)
    d_nm = 2.0 * (3.0 * rmass_np / (4*np.pi*RHO_SULF))**(1/3) * 1e7
    dlog10_d = math.log10(grp.rmrat) / 3.0

    # Wrap env arrays for the RHS — env_d is the dict from _refresh_env;
    # rmass_2d isn't there, build it.
    env = dict(env_d)
    env["rmass_2d"] = np.asarray(grp.rmass)[:, None]
    rhs = make_condensation_rhs(env, dm, shape)

    # Integrate with diffrax: 1440 outer steps of 60 s each.
    print("Running 1440 outer × 60 s = 24 h condensation-only ...")
    term = diffrax.ODETerm(rhs)
    solver = diffrax.Kvaerno5()
    controller = diffrax.PIDController(
        rtol=1e-7, atol=1e-7,
        pcoeff=0.3, icoeff=0.3, factormin=0.5, factormax=5.0, safety=0.7,
    )

    @jax.jit
    def outer_step(pc, gc, T):
        y0 = pack(pc, gc, T)
        sol = diffrax.diffeqsolve(
            term, solver, t0=0.0, t1=DTIME, dt0=None, y0=y0,
            args=None, stepsize_controller=controller,
            max_steps=10_000, saveat=diffrax.SaveAt(t1=True),
        )
        pc_new, gc_new, T_new = unpack(sol.ys[-1], shape)
        return pc_new, gc_new, T_new

    pc = pc0; gc = gc0; T_scalar = T0
    # Save snapshots at t = 0, 6 h, 12 h, 18 h, 24 h
    snapshot_steps = {0: 0, 360: 6, 720: 12, 1080: 18, 1440: 24}
    snapshots = {0: np.asarray(pc[:, 0])}

    t_start = time.perf_counter()
    for istep in range(1, NSTEP + 1):
        pc, gc, T_scalar = outer_step(pc, gc, T_scalar)
        if istep in snapshot_steps:
            snapshots[istep] = np.asarray(pc[:, 0])
            elapsed = time.perf_counter() - t_start
            n_tot = float(jnp.sum(pc[:, 0]))
            mass_g = float(jnp.sum(pc[:, 0] * jnp.asarray(rmass_np)))
            gmd_b = _gmd_from_pc(np.asarray(pc[:, 0]), d_nm, dlog10_d)
            print(f"  t = {snapshot_steps[istep]:2d} h:  "
                   f"N={n_tot:.2e} #/cm³, mass={mass_g:.2e} g/cm³, "
                   f"GMD={gmd_b:.1f} nm,  elapsed {elapsed:.0f}s")
    pc.block_until_ready()
    total_wall = time.perf_counter() - t_start
    print(f"\nWall time: {total_wall:.0f} s")

    # ---------- Plot ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                              gridspec_kw=dict(wspace=0.30))
    ax = axes[0]
    cmap = plt.cm.viridis
    colors = [cmap(x) for x in np.linspace(0, 0.85, len(snapshots))]
    for c, (step, pc_snap) in zip(colors, snapshots.items()):
        t_h = snapshot_steps[step]
        dN_dlogD = pc_snap / dlog10_d
        ax.plot(d_nm, dN_dlogD, "o-", color=c, lw=2, ms=4,
                 label=f"t = {t_h:>2d} h, GMD={_gmd_from_pc(pc_snap, d_nm, dlog10_d):.1f} nm")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Diameter [nm]")
    ax.set_ylabel("dN/dlog₁₀(D) [#/cm³]")
    ax.set_title(f"Condensation-only @ [H₂SO₄] = {H2SO4_MOL_CM3:.0e} cm⁻³, T={T_K} K, RH={RH:.0%}")
    ax.set_ylim(1e1, 5e4)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3, which="both")
    ax.axvline(GMD_NM, color="gray", ls=":", lw=1, alpha=0.6)
    ax.axvline(GMD_NM + 20, color="green", ls=":", lw=1, alpha=0.6,
                label=f"expected: GMD₀ + 20 = {GMD_NM+20:.0f} nm")
    ax.legend(fontsize=8, loc="upper right")

    # GMD vs time
    ax2 = axes[1]
    steps_sorted = sorted(snapshots.keys())
    gmd_t = [_gmd_from_pc(snapshots[s], d_nm, dlog10_d) for s in steps_sorted]
    t_h_arr = [snapshot_steps[s] for s in steps_sorted]
    ax2.plot(t_h_arr, gmd_t, "o-", color="steelblue", lw=2, ms=8)
    ax2.axhline(GMD_NM, color="gray", ls=":", lw=1, alpha=0.6,
                 label=f"initial GMD = {GMD_NM} nm")
    ax2.axhline(GMD_NM + 20, color="green", ls=":", lw=1, alpha=0.6,
                 label=f"expected GMD after 24h ≈ {GMD_NM+20} nm")
    ax2.set_xlabel("Time [h]")
    ax2.set_ylabel("GMD [nm]")
    ax2.set_title(f"GMD vs time (final {gmd_t[-1]:.1f} nm)")
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    fig.suptitle(
        f"Diffrax condensation-only validation — "
        f"N₀={N0:.0e}, GMD₀={GMD_NM} nm, GSD={GSD}, "
        f"[H₂SO₄]={H2SO4_MOL_CM3:.0e} cm⁻³ fixed",
        fontsize=12, fontweight="bold", y=1.02,
    )
    out = OUT_DIR / "validation.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")


def _gmd_from_pc(pc_arr, d_nm, dlog10_d):
    """Compute GMD (geometric mean diameter) from per-bin number array."""
    pos = pc_arr > 0
    if not pos.any():
        return float("nan")
    weights = pc_arr[pos]
    logs = np.log(d_nm[pos])
    mean_log = (weights * logs).sum() / weights.sum()
    return float(np.exp(mean_log))


if __name__ == "__main__":
    main()
