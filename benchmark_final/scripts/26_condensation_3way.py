"""Condensation-only validation across all three solvers.

Compares Fortran, faithful JAX, and diffrax on four controlled scenarios:

  GMD₀ ∈ {20 nm, 100 nm}  ×  [H₂SO₄] ∈ {1e7, 1e8} molec/cm³

Common setup (per the user's class-style condensation problem):
  N₀ = 1.0e4 #/cm³
  GSD = 1.60
  T = 298 K, p = 101325 Pa, RH = 30 %
  Run: dt = 60 s × 1440 steps = 24 h
  Mode: condensation only (no nucleation, no coagulation)
  [H₂SO₄] held fixed throughout

For Fortran: uses the new `--fixed-h2so4` argv option (resets gc and T
each step in the F90).
For faithful JAX: wraps step_full_faithful with gc/T reset between
outer-step calls.
For diffrax: uses the dedicated condensation-only RHS from script #25.

All three are run with do_coag=False. Output: 2×2 plot grid.
"""
import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "scripts"))


# ---------------- Constants ----------------
T_K = 298.0
P_HPA = 1013.25
P_PA = P_HPA * 100.0
RH = 0.30
N0 = 1.0e4
GSD = 1.60
# Coarser outer-step than the gas-depletion timescale would let Fortran
# and faithful JAX see gas depletion *between* resets, biasing growth
# downward. Pick dt small enough that gas barely depletes per step.
# At [H2SO4]=1e7 cm^-3, N=1e4, the depletion timescale is ~25 s. We use
# dt=1s for Fortran/faithful so gas is reset before noticeable depletion.
# Diffrax holds dgc/dt=0 internally, so its dt doesn't matter for this.
DTIME = 1.0
NSTEP = 86400
M_H2SO4 = 98.078479
AVG = 6.02252e23
RPA2CGS = 10.0
RHO_SULF = 1.923

SCENARIOS = [
    ("20nm_1e7", 20.0, 1e7, "GMD₀=20 nm, [H₂SO₄]=1e7"),
    ("20nm_1e8", 20.0, 1e8, "GMD₀=20 nm, [H₂SO₄]=1e8"),
    ("100nm_1e7", 100.0, 1e7, "GMD₀=100 nm, [H₂SO₄]=1e7"),
    ("100nm_1e8", 100.0, 1e8, "GMD₀=100 nm, [H₂SO₄]=1e8"),
]

FORTRAN_BIN = REPO.parent / "original-carma" / "CARMA" / "build" / "test_sulfate_realistic"
OUT_DIR = ROOT / "outputs" / "condensation_3way"
PLOT_DIR = ROOT / "plots" / "condensation_3way"


def initial_seed(gmd_nm, gsd, n0=N0, rho_air_g_cm3=None):
    """Build the initial per-bin distribution from a true lognormal in
    log10(D) space, normalised to total number N0 particles.

    Returns the per-bin pc array that all three solvers use as the
    canonical initial state. To make Fortran start from the SAME pc,
    write this array to a binary file and feed Fortran's
    ``--pc-init-file`` argv option (see run_fortran).

    Returns: pc_per_bin [#/cm^3 of air], d_nm, dlog10_d, rmass.
    """
    from jax_ensemble import _minimal_config

    cfg = _minimal_config()
    grp = cfg.groups[0]
    rmass = np.asarray(grp.rmass)
    d_nm = 2.0 * (3.0 * rmass / (4 * np.pi * RHO_SULF)) ** (1/3) * 1e7
    dlog10_d = math.log10(grp.rmrat) / 3.0

    log10_gsd = math.log10(gsd)
    log10_gmd = math.log10(gmd_nm)
    dN_dlogD = (n0 / (math.sqrt(2 * math.pi) * log10_gsd)
                 * np.exp(-((np.log10(d_nm) - log10_gmd) ** 2)
                            / (2 * log10_gsd ** 2)))
    pc_per_bin = dN_dlogD * dlog10_d
    return pc_per_bin, d_nm, dlog10_d, rmass


def gmd_from_pc(pc, d_nm):
    pos = pc > 0
    if not pos.any():
        return float("nan")
    w = pc[pos]
    return float(np.exp((w * np.log(d_nm[pos])).sum() / w.sum()))


# ---------------- Fortran ----------------
def run_fortran(gmd_nm, h2so4_mol_cm3, scenario_dir):
    """Call the patched Fortran binary in fixed-H2SO4 mode.

    Writes the canonical initial-pc array (built by initial_seed) to a
    temp binary file and points Fortran at it via CARMA_PC_INIT_FILE
    env var. This guarantees Fortran starts from the exact same
    per-bin distribution as the JAX paths.
    """
    import os as _os
    from carma.constants import R_AIR
    rho_air_g_cm3 = P_PA * 10.0 / (float(R_AIR) * T_K)
    pc_init, d_nm, dlog10_d, rmass = initial_seed(gmd_nm, GSD)
    mmr_per_bin = pc_init * rmass / rho_air_g_cm3   # g/g per bin
    # Sanity: compute equivalent M_ug_m3 for the print-out
    M_ug_m3 = mmr_per_bin.sum() * rho_air_g_cm3 * 1e12
    print(f"    Fortran: feeding pc_init via file; total N={pc_init.sum():.3e}, "
          f"M={M_ug_m3:.3e} µg/m³")

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        scen_path = d / "scen.txt"
        out_path = d / "out.json"
        pc_init_path = d / "pc_init.bin"
        # Write per-bin MMR (38 × float64, little-endian native).
        mmr_per_bin.astype(np.float64).tofile(pc_init_path)
        scen_line = (f"{T_K!r} {P_HPA!r} {RH!r} 0.0 "
                      f"{float(M_ug_m3)!r} {float(gmd_nm)!r} {GSD!r}\n")
        scen_path.write_text(scen_line)
        cmd = [str(FORTRAN_BIN), str(scen_path), str(out_path),
                "0", "1", str(DTIME), str(NSTEP),
                str(h2so4_mol_cm3)]
        env = {**_os.environ, "CARMA_PC_INIT_FILE": str(pc_init_path)}
        t0 = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=3600, env=env)
        wall = time.perf_counter() - t0
        if r.returncode != 0:
            raise RuntimeError(f"Fortran failed: {r.stderr[:500]}")
        out = json.loads(out_path.read_text())
        pc_final_mmr = np.asarray(out["pc_final"])
        # Read history if present
        hist_path = Path(str(out_path) + ".history.bin")
        history = None
        if hist_path.exists():
            raw = np.fromfile(hist_path, dtype=np.float64)
            i = 0
            t_hist = raw[i:i+NSTEP]; i += NSTEP
            gc_hist = raw[i:i+NSTEP*2].reshape((2, NSTEP)).T; i += NSTEP*2
            pc_hist_mmr = raw[i:i+NSTEP*38].reshape((38, NSTEP)).T
            history = dict(t=t_hist, gc=gc_hist, pc_mmr=pc_hist_mmr)
    pc_final = pc_final_mmr * rho_air_g_cm3 / rmass
    return dict(pc_final=pc_final, pc_init=pc_init, d_nm=d_nm,
                  wall=wall, history=history, rho_air=rho_air_g_cm3,
                  rmass=rmass)


# ---------------- Faithful JAX ----------------
def run_faithful(gmd_nm, h2so4_mol_cm3):
    """Loop step_full_faithful with gc/T reset between calls."""
    from jax_ensemble import _minimal_config, _compute_ppm_coefs, _refresh_env
    from carma.constants import AVG as _AVG, R_AIR
    from carma.precision import DTYPE
    from carma.step_full_faithful import make_step_full_faithful
    from carma.prestep import prestep

    cfg = _minimal_config()
    cfg = cfg._replace(do_coag=False, do_grow=True)
    ppm = _compute_ppm_coefs(cfg)
    step = make_step_full_faithful(cfg, ppm_coefs=ppm)
    itype_arr = jnp.asarray([e.itype for e in cfg.elements])
    ienconc_arr = jnp.asarray([g.ienconc for g in cfg.groups])
    igelem_arr = jnp.asarray([e.igroup for e in cfg.elements])
    rmass_2d = jnp.stack(
        [jnp.asarray(g.rmass, dtype=DTYPE) for g in cfg.groups], axis=1,
    )

    rho_air_g_cm3 = P_PA * 10.0 / (float(R_AIR) * T_K)
    p_cgs = jnp.asarray([P_PA * RPA2CGS], dtype=DTYPE)
    T0 = jnp.asarray([T_K], dtype=DTYPE)

    pc_init, d_nm, dlog10_d, rmass = initial_seed(gmd_nm, GSD)
    pc = jnp.zeros((1, cfg.nbin, cfg.nelem), dtype=DTYPE)
    pc = pc.at[0, :, 0].set(jnp.asarray(pc_init, dtype=DTYPE))

    # gc[H2SO4] fixed value in g/cm^3
    h2so4_g_cm3 = h2so4_mol_cm3 * M_H2SO4 / AVG
    # H2O mmr at 30% RH
    pvapl_Pa = math.exp(54.842763 - 6763.22 / T_K
                         - 4.210 * math.log(T_K) + 0.000367 * T_K)
    h2o_mmr = RH * pvapl_Pa * 18.0 / (29.0 * P_PA)
    h2o_g_cm3 = h2o_mmr * rho_air_g_cm3
    gc_fixed = jnp.asarray([[h2o_g_cm3, h2so4_g_cm3]], dtype=DTYPE)

    t = T0
    t_start = time.perf_counter()
    for istep in range(NSTEP):
        # Reset gc[H2SO4] and T at start of each outer step.
        gc = gc_fixed
        t = T0
        env_s = _refresh_env(t, p_cgs, gc, cfg, ppm)
        pc, gc2, t2, pcl, gcl, d_gc, d_t, pconmax = prestep(
            pc, gc, t, pc, gc, t, env_s["zmet"],
            itype_arr, ienconc_arr, igelem_arr, rmass_2d,
            do_substep=True, do_coag=cfg.do_coag,
        )
        pc, gc3, t3, _diag = step(
            pc=pc, gc=gc2, t=t2, dtime=float(DTIME),
            rhoa=env_s["rhoa"], zmet=env_s["zmet"],
            akelvin=env_s["akelvin"], akelvini=env_s["akelvini"],
            gro=env_s["gro"], gro1=env_s["gro1"],
            rup_wet=env_s["rup_wet"],
            rlhe=env_s["rlhe"], rlhm=env_s["rlhm"],
            ckernel=env_s["ckernel"], pconmax=pconmax,
            ds_threshold_arr=env_s["ds_threshold_arr"],
            pcl=pcl, gcl=gcl, told=t2, d_gc=d_gc, d_t=d_t,
        )
    pc.block_until_ready()
    wall = time.perf_counter() - t_start
    pc_final = np.asarray(pc[0, :, 0])
    return dict(pc_final=pc_final, pc_init=pc_init, d_nm=d_nm,
                  wall=wall, rho_air=rho_air_g_cm3, rmass=rmass)


# ---------------- Diffrax (condensation-only RHS) ----------------
def run_diffrax(gmd_nm, h2so4_mol_cm3):
    from jax_ensemble import _minimal_config, _compute_ppm_coefs, _refresh_env
    from carma.precision import DTYPE
    from carma.growth.pheat import pheat
    from carma.supersaturation import supersat
    from carma.vapor_pressure import (
        vaporp_h2o_murphy2005, vaporp_h2so4_ayers1980,
    )
    from carma.constants import R_AIR
    from carma_diffrax.state import StateShape, pack, unpack

    cfg = _minimal_config()
    ppm = _compute_ppm_coefs(cfg)
    shape = StateShape(nbin=cfg.nbin, nelem=1, ngas=2)
    grp = cfg.groups[0]
    rmass = np.asarray(grp.rmass)
    rmassup = np.asarray(grp.rmassup)
    rmasslow = np.concatenate([[rmass[0] / (grp.rmrat ** 0.5)], rmassup[:-1]])
    dm = jnp.asarray(rmassup - rmasslow, dtype=DTYPE)

    rho_air_g_cm3 = P_PA * 10.0 / (float(R_AIR) * T_K)
    p_cgs = jnp.asarray([P_PA * RPA2CGS], dtype=DTYPE)
    T0 = jnp.asarray([T_K], dtype=DTYPE)
    h2so4_g_cm3 = h2so4_mol_cm3 * M_H2SO4 / AVG
    pvapl_Pa = math.exp(54.842763 - 6763.22 / T_K
                         - 4.210 * math.log(T_K) + 0.000367 * T_K)
    h2o_mmr = RH * pvapl_Pa * 18.0 / (29.0 * P_PA)
    h2o_g_cm3 = h2o_mmr * rho_air_g_cm3
    gc_fixed = jnp.asarray([[h2o_g_cm3, h2so4_g_cm3]], dtype=DTYPE)

    env_d = _refresh_env(T0, p_cgs, gc_fixed, cfg, ppm)
    env = dict(env_d)
    env["rmass_2d"] = jnp.asarray(rmass, dtype=DTYPE)[:, None]

    nbin = shape.nbin

    @jax.jit
    def rhs(t, y, _args):
        del t, _args
        pc_flat = y[:shape.n_pc]
        pc = pc_flat.reshape(nbin, shape.nelem)
        gc = y[shape.n_pc:shape.n_pc + shape.ngas]
        T_scalar = y[shape.n_pc + shape.ngas]
        T_arr = jnp.atleast_1d(T_scalar)
        gc_2d = gc[None, :]
        pc_3d = pc[None, :, :]
        pvapl_h2o, pvapi_h2o = vaporp_h2o_murphy2005(T_arr)
        pvap_h2so4, _ = vaporp_h2so4_ayers1980(T_arr, gc_2d[:, 0],
                                                 pvapl_h2o, env["zmet"])
        pvapl = jnp.stack([pvapl_h2o, pvap_h2so4], axis=1)
        pvapi = jnp.stack([pvapi_h2o, pvap_h2so4], axis=1)
        ssl_h2o, _ = supersat(T_arr, gc_2d[:, 0], pvapl[:, 0], pvapi[:, 0],
                                DTYPE(18.01528), env["zmet"])
        ssl_h2so4, ssi_h2so4 = supersat(
            T_arr, gc_2d[:, 1], pvapl[:, 1], pvapi[:, 1],
            DTYPE(M_H2SO4), env["zmet"],
        )
        supsatl = jnp.stack([ssl_h2o, ssl_h2so4], axis=1)
        supsati = jnp.stack([jnp.zeros_like(ssl_h2o), ssi_h2so4], axis=1)

        def _dmdt_at(ibin):
            return pheat(pc_3d, supsatl, supsati, pvapl, pvapi,
                          env["akelvin"], env["akelvini"], env["gro"],
                          env["gro1"], env["rup_wet"], env["rmass_2d"],
                          False, 0, 0, ibin, 1)
        dmdt_inner = jax.vmap(_dmdt_at)(jnp.arange(nbin - 1))
        zero = jnp.zeros((1,), dtype=DTYPE)
        dmdt = jnp.concatenate([dmdt_inner, zero])
        pc_elem = pc[:, 0]
        pc_above = jnp.concatenate([pc_elem[1:], zero])
        dm_above = jnp.concatenate([dm[1:], jnp.ones((1,), dtype=DTYPE)])
        n_at = jnp.where(dmdt > 0, pc_elem / dm, pc_above / dm_above)
        F_n = dmdt * n_at
        F_n_below = jnp.concatenate([zero, F_n[:-1]])
        dpc_dt = F_n_below - F_n
        evap_bin0 = jnp.where(dmdt[0] < 0,
                               -dmdt[0] / dm[0] * pc_elem[0], DTYPE(0.0))
        dpc_dt = dpc_dt.at[0].add(-evap_bin0)
        dpc_dt_2d = dpc_dt[:, None]
        return jnp.concatenate([dpc_dt_2d.reshape(-1),
                                  jnp.zeros(shape.ngas, dtype=DTYPE),
                                  jnp.atleast_1d(DTYPE(0.0))])

    pc_init, d_nm, dlog10_d, _ = initial_seed(gmd_nm, GSD)
    pc = jnp.asarray(pc_init, dtype=DTYPE)[:, None]
    term = diffrax.ODETerm(rhs)
    solver = diffrax.Kvaerno5()
    controller = diffrax.PIDController(
        rtol=1e-7, atol=1e-7,
        pcoeff=0.3, icoeff=0.3, factormin=0.5, factormax=5.0, safety=0.7,
    )

    @jax.jit
    def outer(pc_in, gc_in, T_in):
        y0 = pack(pc_in, gc_in, T_in)
        sol = diffrax.diffeqsolve(term, solver, t0=0.0, t1=DTIME,
                                    dt0=None, y0=y0, args=None,
                                    stepsize_controller=controller,
                                    max_steps=10_000,
                                    saveat=diffrax.SaveAt(t1=True))
        return unpack(sol.ys[-1], shape)

    pc_cur = pc; gc_cur = gc_fixed[0]; T_cur = T0[0]
    t_start = time.perf_counter()
    for istep in range(NSTEP):
        pc_cur, gc_cur, T_cur = outer(pc_cur, gc_cur, T_cur)
    pc_cur.block_until_ready()
    wall = time.perf_counter() - t_start
    return dict(pc_final=np.asarray(pc_cur[:, 0]),
                  pc_init=pc_init, d_nm=d_nm, wall=wall,
                  rho_air=rho_air_g_cm3, rmass=rmass)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenarios", nargs="*", default=None,
                    help="subset of scenario keys (default: all 4)")
    p.add_argument("--skip", nargs="*", default=[],
                    choices=["fortran", "faithful", "diffrax"])
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    scens = SCENARIOS
    if args.scenarios:
        scens = [s for s in SCENARIOS if s[0] in args.scenarios]

    all_results = {}
    for key, gmd, h2so4, label in scens:
        print(f"\n=== {key}: {label} ===")
        scen_dir = OUT_DIR / key
        scen_dir.mkdir(exist_ok=True)
        results = {}

        if "fortran" not in args.skip:
            print(f"  Fortran ...", flush=True)
            results["fortran"] = run_fortran(gmd, h2so4, scen_dir)
            print(f"    wall = {results['fortran']['wall']:.1f}s")
        if "faithful" not in args.skip:
            print(f"  Faithful JAX ...", flush=True)
            results["faithful"] = run_faithful(gmd, h2so4)
            print(f"    wall = {results['faithful']['wall']:.1f}s")
        if "diffrax" not in args.skip:
            print(f"  Diffrax ...", flush=True)
            results["diffrax"] = run_diffrax(gmd, h2so4)
            print(f"    wall = {results['diffrax']['wall']:.1f}s")

        # Report GMDs
        if results:
            r0 = next(iter(results.values()))
            init_gmd = gmd_from_pc(r0["pc_init"], r0["d_nm"])
            print(f"  GMDs: init={init_gmd:.1f} nm")
            for solver, res in results.items():
                final_gmd = gmd_from_pc(res["pc_final"], res["d_nm"])
                print(f"    {solver:10s}: GMD_final = {final_gmd:.1f} nm "
                       f"(Δ = {final_gmd - init_gmd:+.1f})")

        all_results[key] = dict(label=label, gmd_init=gmd, h2so4=h2so4,
                                  results=results)
        # Save raw arrays
        np.savez_compressed(
            scen_dir / "results.npz",
            **{
                f"{s}_pc_final": r["pc_final"] for s, r in results.items()
            },
            **{
                f"{s}_wall": np.array([r["wall"]]) for s, r in results.items()
            },
            d_nm=next(iter(results.values()))["d_nm"],
            pc_init=next(iter(results.values()))["pc_init"],
        )

    # ---------- 2×2 plot ----------
    fig, axes = plt.subplots(2, 2, figsize=(15, 10),
                              gridspec_kw=dict(hspace=0.35, wspace=0.30))
    solver_styles = dict(
        fortran=dict(color="tab:red", lw=2.2, ls="-", marker=None,
                      label="Fortran (semi-implicit Euler)"),
        faithful=dict(color="tab:orange", lw=1.8, ls="--", marker=None,
                       label="Faithful JAX (same algo)"),
        diffrax=dict(color="tab:blue", lw=1.8, ls="-.", marker=None,
                      label="Diffrax (Kvaerno5+upwind)"),
    )

    for ax, (key, info) in zip(axes.flat, all_results.items()):
        rs = info["results"]
        if not rs:
            continue
        r0 = next(iter(rs.values()))
        d_nm = r0["d_nm"]
        dlog10_d = math.log10(2.0) / 3.0
        # Initial
        init = r0["pc_init"] / dlog10_d
        ax.plot(d_nm, init, "k-", lw=2.5, alpha=0.5,
                 label=f"initial (GMD={gmd_from_pc(r0['pc_init'], d_nm):.1f} nm)")
        for solver, res in rs.items():
            y = res["pc_final"] / dlog10_d
            gmd_f = gmd_from_pc(res["pc_final"], d_nm)
            kw = dict(solver_styles[solver])
            kw["label"] = (
                f"{solver_styles[solver]['label']}: "
                f"GMD={gmd_f:.1f} nm  ({res['wall']:.0f}s)"
            )
            ax.plot(d_nm, y, **kw)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Diameter [nm]")
        ax.set_ylabel("dN/dlog₁₀(D) [#/cm³]")
        ax.set_title(info["label"], fontweight="bold")
        ax.set_ylim(1e1, 5e4)
        ax.legend(fontsize=8.5, loc="lower left")
        ax.grid(alpha=0.3, which="both")

    fig.suptitle(
        f"Condensation-only validation — 3 solvers × 4 scenarios "
        f"(N₀={N0:.0e} #/cm³, GSD={GSD}, T={T_K} K, RH={RH:.0%}, 24 h)",
        fontsize=13, fontweight="bold", y=0.995,
    )
    out = PLOT_DIR / "compare_4scenarios.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
