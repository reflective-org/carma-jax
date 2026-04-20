"""Precision-comparison harness for CARMA-JAX.

Runs all five single-column benchmarks (coagtest, falltest, vdiftest,
drydeptest, growtest), captures the final state + metrics vs the
Fortran reference, and writes both per-test data (.npz) and a
one-page overview dashboard.

Precision is selected via the ``CARMA_DTYPE`` env var (see
``carma.precision``):

    python scripts/compare_precision.py                     # fp64 baseline
    CARMA_DTYPE=fp32 python scripts/compare_precision.py    # fp32 run

Output files and the dashboard image are suffixed with the active
dtype. Running both produces data that ``plot_precision_overlay.py``
stitches into an fp64-vs-fp32-vs-Fortran overlay dashboard.

Known pre-existing deltas reflected here (not introduced by this
harness):
  - growtest outer-tail delta: size-distribution tail under-resolution
    documented in docs/PROGRESS.md deferred item 6.
  - drydeptest max ~108%: single outlier at a depleted tail bin
    (Fortran ~1e-25) — median still < 0.07%.
"""

import time
from pathlib import Path

# Precision (x64 flag) is set inside carma.precision from CARMA_DTYPE.
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from carma.precision import DTYPE
from carma.constants import RM2CGS, RPA2CGS, SMALL_PC, BK, RHO_I, R_AIR, WTMOL_H2O
from carma.bins import setup_bins
from carma.atmosphere_std import get_standard_atmosphere
from carma.setup_atm import setup_atm
from carma.setup_vf import setup_vf_jit
from carma.setup_bdif import setup_bdif
from carma.setup_vdry import setup_vdry
from carma.setup_grow import setup_grow
from carma.setup_gkern import setup_gkern
from carma.supersaturation import supersat
from carma.vapor_pressure import vaporp_h2o_murphy2005
from carma.enums import ElementType, GridType, BoundaryCondition, VaporPressureRoutine
from carma.coagulation.setup_coag import setup_coag
from carma.config import ElementConfig, GroupConfig, GasConfig, CarmaConfig
from carma.step import make_step_coag, make_step_transport, make_step_microfast


BENCH_DIR = (Path(__file__).parent.parent.parent
             / "original-carma" / "CARMA" / "tests" / "bench")
OUTDIR = Path(__file__).parent.parent / "plots" / "precision_comparison"
DATADIR = OUTDIR / "data"


# ---------------------------------------------------------------------------
# Bench-file parsers (lifted from the existing validate_*.py scripts)
# ---------------------------------------------------------------------------

def _parse_time_series(lines, nz, nelem=1):
    """Generic parser for `time` followed by NZ×NELEM `idx val [val2]` blocks."""
    times, data = [], []
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        try:
            t_val = float(line.split()[0])
        except (ValueError, IndexError):
            idx += 1
            continue
        times.append(t_val)
        idx += 1
        block = np.zeros((nelem, nz))
        for _ in range(nz * nelem):
            parts = lines[idx].split()
            if nelem == 1:
                iz = int(parts[0]) - 1
                block[0, iz] = float(parts[1])
            else:
                ie = int(parts[0]) - 1
                iz = int(parts[1]) - 1
                block[ie, iz] = float(parts[2])
            idx += 1
        data.append(block[0] if nelem == 1 else block)
    return np.array(times), np.array(data)


def parse_fall_or_vdif(path):
    lines = Path(path).read_text().splitlines()
    nz = int(lines[0].strip())
    zc = np.zeros(nz)
    dz = np.zeros(nz)
    for i in range(nz):
        parts = lines[1 + i].split()
        zc[i] = float(parts[1])
        dz[i] = float(parts[2])
    times, mmr = _parse_time_series(lines[1 + nz:], nz, nelem=1)
    return {"nz": nz, "zc": zc, "dz": dz, "times": times, "mmr": mmr}


def parse_drydep(path):
    lines = Path(path).read_text().splitlines()
    parts0 = lines[0].split()
    nz, nelem = int(parts0[0]), int(parts0[1])
    zc = np.zeros(nz)
    dz = np.zeros(nz)
    for i in range(nz):
        p = lines[1 + i].split()
        zc[i] = float(p[1])
        dz[i] = float(p[2])
    times, mmr = _parse_time_series(lines[1 + nz:], nz, nelem=nelem)
    return {"nz": nz, "nelem": nelem, "zc": zc, "dz": dz,
            "times": times, "mmr": mmr}


def parse_coag(path):
    lines = Path(path).read_text().splitlines()
    nbin = int(lines[0].split()[0])
    radii = np.zeros(nbin)
    dr = np.zeros(nbin)
    for i in range(nbin):
        p = lines[1 + i].split()
        radii[i] = float(p[1])
        dr[i] = float(p[2])
    times, nd = [], []
    idx = 1 + nbin
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1; continue
        try:
            t_val = float(line)
        except ValueError:
            idx += 1; continue
        times.append(t_val); idx += 1
        vals = np.zeros(nbin)
        for i in range(nbin):
            vals[i] = float(lines[idx].split()[1]); idx += 1
        nd.append(vals)
    return {"nbin": nbin, "radii": radii, "dr": dr,
            "times": np.array(times), "nd": np.array(nd)}


def parse_grow(path):
    """carma_growtest.txt: bin info then time × (bin_mmr, bin_n, gas_mmr)."""
    lines = Path(path).read_text().splitlines()
    nbin = int(lines[0].split()[0])
    radii = np.zeros(nbin)
    for i in range(nbin):
        radii[i] = float(lines[1 + i].split()[1])
    idx = 1 + nbin
    times, bin_mmrs, gas_mmrs, t_changes = [], [], [], []
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1; continue
        tokens = line.split()
        if len(tokens) < 2:
            idx += 1; continue
        try:
            t_val = float(tokens[0])
            t_change = float(tokens[1]) if len(tokens) > 1 else 0.0
            gas_mmr = float(tokens[2]) if len(tokens) > 2 else 0.0
        except ValueError:
            idx += 1; continue
        times.append(t_val); t_changes.append(t_change); gas_mmrs.append(gas_mmr)
        idx += 1
        mmr = np.zeros(nbin)
        for i in range(nbin):
            mmr[i] = float(lines[idx].split()[1]); idx += 1
        bin_mmrs.append(mmr)
    return {"nbin": nbin, "radii": radii,
            "times": np.array(times),
            "mmr": np.array(bin_mmrs),
            "gas_mmr": np.array(gas_mmrs),
            "t_change": np.array(t_changes)}


# ---------------------------------------------------------------------------
# Individual benchmark runners — each returns a result dict
# ---------------------------------------------------------------------------

def run_coagtest():
    fort = parse_coag(BENCH_DIR / "carma_coagtest.txt")
    NBIN, NELEM, NGROUP, NZ = 20, 1, 1, 80
    r, rmass, vol, dr, dm, rup, rlow, rmassup = setup_bins(3e-7, 2.0, NBIN, 2.0)
    groups = (GroupConfig(
        name='dust', ishape=1, ienconc=0, is_ice=False, is_cloud=False,
        is_sulfate=False, do_vtran=False, do_drydep=False, ifallrtn=1,
        irhswell=0, rmrat=2.0, eshape=1.0, rmin=3e-7,
        r=r, rmass=rmass, vol=vol, dr=dr, dm=dm, rmassup=rmassup,
        rup=rup, rlow=rlow, rrat=jnp.ones(NBIN), rprat=jnp.ones(NBIN),
        arat=jnp.ones(NBIN)),)
    elements = (ElementConfig(
        name='dust', rho=jnp.full(NBIN, 2.0), igroup=0,
        itype=int(ElementType.I_INVOLATILE), icomposition=0, isolute=-1, kappa=0.0),)
    coag = setup_coag(NBIN, NGROUP, NELEM, groups, elements,
                     np.array([[0]], dtype=np.int32), np.array([[0]], dtype=np.int32))
    cfg = CarmaConfig(
        nbin=NBIN, nelem=NELEM, ngroup=NGROUP, ngas=0, nsolute=0,
        elements=elements, groups=groups, gases=(), solutes=(), coag=coag,
        do_coag=True, do_grow=False, do_vtran=False, do_vdiff=False,
        do_thermo=False, do_substep=False, do_explised=False,
        do_incloud=False, do_clearsky=False, do_detrain=False,
        do_pheat=False, do_pheatatm=False, do_cnst_rlh=False,
        itbnd_pc=1, ibbnd_pc=1, maxsubsteps=1, minsubsteps=1, maxretries=5,
        conmax=0.0, cstick=1.0, gsticki=1.0, gstickl=1.0, tstick=1.0,
        dt_threshold=0.0, igash2o=-1, igash2so4=-1, igasso2=-1)

    deltaz = 100.0 * RM2CGS
    zc = jnp.arange(0.5, NZ) * deltaz
    zl = jnp.arange(0.0, NZ + 1) * deltaz
    p_pa, t_k = get_standard_atmosphere(zc / RM2CGS)
    pl_pa, _ = get_standard_atmosphere(zl / RM2CGS)
    rhoa, dz, zmet, zmetl, rmu, thcond, rhoa_wet = setup_atm(
        t_k, p_pa * RPA2CGS, pl_pa * RPA2CGS, zc, zl, GridType.I_CART)

    ck0 = DTYPE(8.0) * BK * DTYPE(298.0) / (DTYPE(3.0) * DTYPE(1.85e-4))
    ckernel = jnp.full((NZ, NBIN, NBIN, NGROUP, NGROUP), ck0)
    pc = jnp.full((NZ, NBIN, NELEM), SMALL_PC)
    pc = pc.at[0, 0, 0].set(DTYPE(1e6))

    step = make_step_coag(cfg)
    gc = jnp.zeros((NZ, 0)); gcl = jnp.zeros((NZ, 0)); told = t_k; pcl = pc
    dtime, nstep = 600.0, 72

    t0 = time.time()
    nd_hist = [np.array(pc[0, :, 0])]
    for _ in range(nstep):
        pc, gc, t_k, pcl, gcl, _ = step(
            pc, gc, t_k, pcl, gcl, told, zmet, ckernel, DTYPE(dtime))
        nd_hist.append(np.array(pc[0, :, 0]))
    loop_s = time.time() - t0

    jax_final = nd_hist[-1]
    fort_final = fort["nd"][-1]
    mask = fort_final > 1e-10
    rel_err = np.abs(jax_final[mask] - fort_final[mask]) / fort_final[mask]
    m_init = np.sum(nd_hist[0] * rmass)
    m_final = np.sum(nd_hist[-1] * rmass)

    return {
        "name": "coagtest", "xaxis": "radius [cm]", "xvals": np.array(r),
        "jax_final": jax_final, "fortran_final": fort_final,
        "times": fort["times"], "loop_s": loop_s,
        "rel_err_median": float(np.median(rel_err)),
        "rel_err_max": float(rel_err.max()),
        "mass_cons": float((m_final - m_init) / m_init),
        "ylabel": "number density [cm⁻³]",
    }


def _atm_pair(zc_m, zl_m):
    p_pa, t_k = get_standard_atmosphere(jnp.array(zc_m, dtype=DTYPE))
    pl_pa, _ = get_standard_atmosphere(jnp.array(zl_m, dtype=DTYPE))
    rhoa, dz, zmet, zmetl, rmu, thcond, rhoa_wet = setup_atm(
        t_k, p_pa * RPA2CGS, pl_pa * RPA2CGS,
        jnp.array(zc_m) * RM2CGS, jnp.array(zl_m) * RM2CGS, GridType.I_CART)
    return (p_pa, t_k, rhoa, dz, zmet, zmetl, rmu, rhoa_wet)


def _transport_config(NBIN, NELEM, NGROUP, rmin, rmrat, rho, do_drydep):
    r, rmass, vol, dr, dm, rup, rlow, rmassup = setup_bins(rmin, rmrat, NBIN, rho)
    groups = tuple(GroupConfig(
        name=f'g{ig}', ishape=1, ienconc=0, is_ice=False, is_cloud=False,
        is_sulfate=False, do_vtran=True, do_drydep=bool(do_drydep[ig]),
        ifallrtn=1, irhswell=0, rmrat=rmrat, eshape=1.0, rmin=rmin,
        r=r, rmass=rmass, vol=vol, dr=dr, dm=dm, rmassup=rmassup,
        rup=rup, rlow=rlow, rrat=jnp.ones(NBIN), rprat=jnp.ones(NBIN),
        arat=jnp.ones(NBIN)) for ig in range(NGROUP))
    elements = tuple(ElementConfig(
        name=f'e{ie}', rho=jnp.full(NBIN, rho), igroup=ie,
        itype=int(ElementType.I_INVOLATILE), icomposition=0, isolute=-1, kappa=0.0
    ) for ie in range(NELEM))
    icoag = np.zeros((NGROUP, NGROUP), dtype=np.int32)
    icoagelem = np.zeros((NELEM, NGROUP), dtype=np.int32)
    coag = setup_coag(NBIN, NGROUP, NELEM, groups, elements, icoag, icoagelem)
    cfg = CarmaConfig(
        nbin=NBIN, nelem=NELEM, ngroup=NGROUP, ngas=0, nsolute=0,
        elements=elements, groups=groups, gases=(), solutes=(), coag=coag,
        do_coag=False, do_grow=False, do_vtran=True, do_vdiff=False,
        do_thermo=False, do_substep=False, do_explised=False,
        do_incloud=False, do_clearsky=False, do_detrain=False,
        do_pheat=False, do_pheatatm=False, do_cnst_rlh=False,
        itbnd_pc=int(BoundaryCondition.I_FIXED_CONC),
        ibbnd_pc=int(BoundaryCondition.I_FIXED_CONC),
        maxsubsteps=1, minsubsteps=1, maxretries=5, conmax=0.0,
        cstick=1.0, gsticki=1.0, gstickl=1.0, tstick=1.0, dt_threshold=0.0,
        igash2o=-1, igash2so4=-1, igasso2=-1)
    return cfg, r, rmass


def run_falltest():
    fort = parse_fall_or_vdif(BENCH_DIR / "carma_falltest.txt")
    NZ, NBIN, NELEM, NGROUP = 110, 8, 1, 1
    dtime, nstep, deltaz_m = 1000.0, 600, 100.0
    cfg, r, rmass = _transport_config(NBIN, NELEM, NGROUP, 7.5e-4, 2.0, 2.65, [False])
    zc_m = np.arange(NZ) * deltaz_m + deltaz_m / 2
    zl_m = np.arange(NZ + 1) * deltaz_m
    p_pa, t_k, rhoa, dz, zmet, zmetl, rmu, rhoa_wet = _atm_pair(zc_m, zl_m)

    vf = jnp.full((NZ + 1, NBIN, NGROUP), DTYPE(2.0))
    dkz = jnp.zeros((NZ + 1, NBIN, NGROUP))
    vd = jnp.zeros((NBIN, NGROUP))
    pc = jnp.zeros((NZ, NBIN, NELEM))
    mmr_init = 1e-10 * np.exp(-((zc_m - 8000.0) / 3000.0) ** 2) * 287.0 * np.array(t_k) / np.array(p_pa)
    pc = pc.at[:, 0, 0].set(jnp.array(mmr_init * np.array(rhoa_wet), dtype=DTYPE))

    step = make_step_transport(cfg)
    gc = jnp.zeros((NZ, 0)); gcl = jnp.zeros((NZ, 0)); told = t_k; pcl = pc
    zeros_bc = jnp.zeros((NBIN, NELEM))

    t0 = time.time()
    for _ in range(nstep):
        pc, gc, t_k, pcl, gcl, _, _ = step(
            pc, gc, t_k, pcl, gcl, told, zmet,
            vf, dkz, vd, dz, jnp.array(zc_m) * RM2CGS, jnp.array(zl_m) * RM2CGS,
            rhoa, DTYPE(dtime),
            zeros_bc, zeros_bc, zeros_bc, zeros_bc)
    loop_s = time.time() - t0

    jax_mmr = np.array(pc[:, 0, 0]) / np.array(rhoa_wet)
    fort_mmr = fort["mmr"][-1]
    mask = fort_mmr > 1e-30
    rel_err = np.abs(jax_mmr[mask] - fort_mmr[mask]) / fort_mmr[mask]

    return {
        "name": "falltest", "xaxis": "altitude [km]", "xvals": zc_m / 1000,
        "jax_final": jax_mmr, "fortran_final": fort_mmr,
        "times": fort["times"], "loop_s": loop_s,
        "rel_err_median": float(np.median(rel_err)),
        "rel_err_max": float(rel_err.max()),
        "mass_cons": np.nan,  # column mass not meaningful (fixed_conc BC absorbs)
        "ylabel": "MMR [g/g]",
    }


def run_vdiftest():
    fort = parse_fall_or_vdif(BENCH_DIR / "carma_vdiftest.txt")
    NZ, NBIN, NELEM, NGROUP = 240, 1, 1, 1
    dtime, nstep, deltaz_m, zmin_m = 1000.0, 600, 100.0, 80000.0
    cfg, r, rmass = _transport_config(NBIN, NELEM, NGROUP, 2e-8, 2.0, 2.65, [False])
    zc_m = np.arange(NZ) * deltaz_m + zmin_m + deltaz_m / 2
    zl_m = np.arange(NZ + 1) * deltaz_m + zmin_m
    p_pa, t_k, rhoa, dz, zmet, zmetl, rmu, rhoa_wet = _atm_pair(zc_m, zl_m)

    r_wet = jnp.broadcast_to(jnp.array(r)[None, :, None], (NZ, NBIN, NGROUP))
    rhop_wet = jnp.full((NZ, NBIN, NGROUP), DTYPE(2.65))
    rrat = jnp.ones((NBIN, NGROUP)); rprat = jnp.ones((NBIN, NGROUP))
    vf_computed, _, bpm = setup_vf_jit(t_k, rhoa, zmet, rmu, r_wet, rhop_wet, rrat, rprat)
    vf = jnp.full((NZ + 1, NBIN, NGROUP), DTYPE(1e-10))
    dkz = setup_bdif(t_k, rmu, r_wet, bpm, rprat, zmetl, GridType.I_CART)
    vd = jnp.zeros((NBIN, NGROUP))

    mmr_init = 1e-10 * np.exp(-5.0 * ((zc_m - 90000.0) / 3000.0) ** 2) * 287.0 * np.array(t_k) / np.array(p_pa)
    pc = jnp.zeros((NZ, NBIN, NELEM))
    pc = pc.at[:, 0, 0].set(jnp.array(mmr_init * np.array(rhoa_wet), dtype=DTYPE))

    step = make_step_transport(cfg)
    gc = jnp.zeros((NZ, 0)); gcl = jnp.zeros((NZ, 0)); told = t_k; pcl = pc
    zeros_bc = jnp.zeros((NBIN, NELEM))

    t0 = time.time()
    for _ in range(nstep):
        pc, gc, t_k, pcl, gcl, _, _ = step(
            pc, gc, t_k, pcl, gcl, told, zmet,
            vf, dkz, vd, dz, jnp.array(zc_m) * RM2CGS, jnp.array(zl_m) * RM2CGS,
            rhoa, DTYPE(dtime),
            zeros_bc, zeros_bc, zeros_bc, zeros_bc)
    loop_s = time.time() - t0

    jax_mmr = np.array(pc[:, 0, 0]) / np.array(rhoa_wet)
    fort_mmr = fort["mmr"][-1]
    mask = fort_mmr > 1e-30
    rel_err = np.abs(jax_mmr[mask] - fort_mmr[mask]) / fort_mmr[mask]

    return {
        "name": "vdiftest", "xaxis": "altitude [km]", "xvals": zc_m / 1000,
        "jax_final": jax_mmr, "fortran_final": fort_mmr,
        "times": fort["times"], "loop_s": loop_s,
        "rel_err_median": float(np.median(rel_err)),
        "rel_err_max": float(rel_err.max()),
        "mass_cons": np.nan,
        "ylabel": "MMR [g/g]",
    }


def run_drydeptest():
    fort = parse_drydep(BENCH_DIR / "carma_drydeptest.txt")
    NZ, NBIN, NELEM, NGROUP = 150, 16, 2, 2
    OUTBIN = 13
    dtime, nstep, deltaz_m = 1000.0, 600, 100.0
    cfg, r, rmass = _transport_config(NBIN, NELEM, NGROUP, 1e-6, 4.32, 2.65, [True, False])
    zc_m = np.arange(NZ) * deltaz_m + deltaz_m / 2
    zl_m = np.arange(NZ + 1) * deltaz_m
    p_pa, t_k, rhoa, dz, zmet, zmetl, rmu, rhoa_wet = _atm_pair(zc_m, zl_m)

    r_wet = jnp.broadcast_to(jnp.array(r)[None, :, None], (NZ, NBIN, NGROUP))
    rhop_wet = jnp.full((NZ, NBIN, NGROUP), DTYPE(2.65))
    rrat = jnp.ones((NBIN, NGROUP)); rprat = jnp.ones((NBIN, NGROUP))
    vf, _, bpm = setup_vf_jit(t_k, rhoa, zmet, rmu, r_wet, rhop_wet, rrat, rprat)
    dkz = jnp.zeros((NZ + 1, NBIN, NGROUP))
    vd = setup_vdry(
        vf, r_wet, bpm, t_k, rmu, rhoa, zmet, zmetl,
        1.5, 2.0, 2.5, 60.0, 40.0, 20.0, 0.0, 1.0, 0.0,
        grp_do_drydep=np.array([True, False]), igridv=GridType.I_CART)

    mmr_profile = 1e-10 * np.exp(-((zc_m - 8000.0) / 3000.0) ** 2) * 287.0 * np.array(t_k) / np.array(p_pa)
    pc = jnp.zeros((NZ, NBIN, NELEM))
    for ib in range(NBIN):
        for ie in range(NELEM):
            pc = pc.at[:, ib, ie].set(
                jnp.array(mmr_profile * np.array(rhoa_wet), dtype=DTYPE))

    step = make_step_transport(cfg)
    gc = jnp.zeros((NZ, 0)); gcl = jnp.zeros((NZ, 0)); told = t_k; pcl = pc
    zeros_bc = jnp.zeros((NBIN, NELEM))

    t0 = time.time()
    for _ in range(nstep):
        pc, gc, t_k, pcl, gcl, _, _ = step(
            pc, gc, t_k, pcl, gcl, told, zmet,
            vf, dkz, vd, dz, jnp.array(zc_m) * RM2CGS, jnp.array(zl_m) * RM2CGS,
            rhoa, DTYPE(dtime),
            zeros_bc, zeros_bc, zeros_bc, zeros_bc)
    loop_s = time.time() - t0

    # Only compare element 0 bin 14 (what Fortran writes)
    jax_mmr = np.array(pc[:, OUTBIN, 0]) / np.array(rhoa_wet)
    fort_mmr = fort["mmr"][-1, 0, :]
    mask = fort_mmr > 1e-30
    rel_err = np.abs(jax_mmr[mask] - fort_mmr[mask]) / fort_mmr[mask]

    return {
        "name": "drydeptest", "xaxis": "altitude [km]", "xvals": zc_m / 1000,
        "jax_final": jax_mmr, "fortran_final": fort_mmr,
        "times": fort["times"], "loop_s": loop_s,
        "rel_err_median": float(np.median(rel_err)),
        "rel_err_max": float(rel_err.max()),
        "mass_cons": np.nan,
        "ylabel": "MMR [g/g]",
    }


def run_growtest():
    """Growtest — reuse existing validate_growtest machinery."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from validate_growtest import run_jax_growtest, parse_growtest_bench

    t0 = time.time()
    jax_res = run_jax_growtest()
    loop_s = time.time() - t0

    fort = parse_growtest_bench(BENCH_DIR / "carma_growtest.txt")
    jax_mmr = jax_res["mmr_bins"][-1]
    fort_mmr = fort["mmr"][-1] if fort["mmr"].size else jax_mmr
    mask = fort_mmr > 1e-30
    rel_err = (np.abs(jax_mmr[mask] - fort_mmr[mask]) / fort_mmr[mask]
               if mask.any() else np.array([0.0]))

    return {
        "name": "growtest", "xaxis": "radius [cm]", "xvals": jax_res["radii"],
        "jax_final": jax_mmr, "fortran_final": fort_mmr,
        "times": jax_res["times"], "loop_s": loop_s,
        "rel_err_median": float(np.median(rel_err)),
        "rel_err_max": float(rel_err.max()),
        "mass_cons": np.nan,
        "ylabel": "bin MMR [g/g]",
    }


# ---------------------------------------------------------------------------
# Dashboard plot
# ---------------------------------------------------------------------------

def _dtype_tag():
    return "fp64" if jnp.dtype(DTYPE).itemsize == 8 else "fp32"


def make_dashboard(results, outdir, title, tag):
    outdir.mkdir(parents=True, exist_ok=True)
    n = len(results)
    fig = plt.figure(figsize=(5 * n, 10))
    for i, res in enumerate(results):
        # Top row: profile overlay
        ax = fig.add_subplot(2, n, i + 1)
        ax.plot(res["xvals"], np.maximum(res["jax_final"], 1e-50), "-", lw=2, label="JAX")
        ax.plot(res["xvals"], np.maximum(res["fortran_final"], 1e-50), "--", lw=1.5, alpha=0.8, label="Fortran")
        ax.set_yscale("log")
        ax.set_xlabel(res["xaxis"])
        ax.set_ylabel(res["ylabel"])
        ax.set_title(res["name"])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        # Bottom row: relative error
        ax2 = fig.add_subplot(2, n, n + i + 1)
        mask = res["fortran_final"] > 1e-30
        if mask.any():
            rel_err = np.abs(res["jax_final"][mask] - res["fortran_final"][mask]) / res["fortran_final"][mask]
            x_active = res["xvals"][mask]
            ax2.plot(x_active, rel_err * 100, "ro-", ms=3, lw=1)
        ax2.set_yscale("log")
        ax2.set_xlabel(res["xaxis"])
        ax2.set_ylabel("rel err [%]")
        ax2.set_title(f"med={res['rel_err_median']*100:.3f}% max={res['rel_err_max']*100:.2f}%")
        ax2.axhline(1.0, color="gray", ls="--", alpha=0.5)
        ax2.axhline(0.1, color="green", ls="--", alpha=0.5)
        ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontweight="bold", fontsize=14)
    fig.tight_layout()
    out_path = outdir / f"dashboard_{tag}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Dashboard: {out_path}")


def make_summary_table(results):
    print("\n" + "=" * 85)
    print(f"{'benchmark':<12} {'rel_med %':>12} {'rel_max %':>12} "
          f"{'mass_cons':>14} {'loop_s':>10} {'dtype':>8}")
    print("-" * 85)
    for r in results:
        mc_str = f"{r['mass_cons']:.2e}" if np.isfinite(r["mass_cons"]) else "n/a"
        print(f"{r['name']:<12} "
              f"{r['rel_err_median']*100:>12.4f} "
              f"{r['rel_err_max']*100:>12.4f} "
              f"{mc_str:>14} "
              f"{r['loop_s']:>10.2f} "
              f"{str(DTYPE.dtype if hasattr(DTYPE, 'dtype') else DTYPE):>8}")
    print("=" * 85)


def main():
    DATADIR.mkdir(parents=True, exist_ok=True)
    tag = _dtype_tag()
    print(f"Running compare_precision in {tag} mode.")
    runs = [
        ("coagtest", run_coagtest),
        ("falltest", run_falltest),
        ("vdiftest", run_vdiftest),
        ("drydeptest", run_drydeptest),
        ("growtest", run_growtest),
    ]
    results = []
    for name, fn in runs:
        print(f"\n── {name} ──")
        res = fn()
        results.append(res)
        np.savez(
            DATADIR / f"{name}_{tag}.npz",
            **{k: np.asarray(v) for k, v in res.items()
               if k not in ("name", "xaxis", "ylabel")},
        )
    make_summary_table(results)
    title = {"fp64": "CARMA-JAX float64 baseline",
             "fp32": "CARMA-JAX float32 run"}[tag]
    make_dashboard(results, OUTDIR, title, tag)


if __name__ == "__main__":
    main()
