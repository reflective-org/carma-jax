"""Reader for Fortran per-substep diagnostic dumps.

The instrumented binary (``test_sulfate_diagnostic``) writes one Fortran
unformatted-stream file per (substep, array) into ``<dir>/substep_<NNNN>_<name>.bin``.
This module loads those files into a typed dict.

Usage:
    >>> from carma._diag import read_substep
    >>> dump = read_substep("data/diff/scen_024", 1)
    >>> dump.pc.shape
    (1, 38, 1)
    >>> dump.t
    array([213.2])

Array shapes (in numpy, after Fortran→C order conversion):

    pc          (NZ, NBIN, NELEM)
    gc          (NZ, NGAS)
    t           (NZ,)
    p           (NZ,)
    rhoa        (NZ,)
    zmet        (NZ,)
    wtpct       (NZ,)        — sulfate wt% from wtpct_tabaz
    sulfdens    (NZ,)        — sulfate_density(wtpct, t) Fortran probe
    sulfsurf    (NZ,)        — sulfate_surf_tens(wtpct, t) Fortran probe
    pvapl       (NZ, NGAS)
    pvapi       (NZ, NGAS)
    supsatl     (NZ, NGAS)
    supsati     (NZ, NGAS)
    akelvin     (NZ, NGAS)
    akelvini    (NZ, NGAS)
    gro         (NZ, NBIN, NGROUP)
    gro1        (NZ, NBIN, NGROUP)
    rhompe      (NBIN, NELEM)
    rnuclg      (NBIN, NGROUP, NGROUP)
    rnucpe      (NBIN, NELEM)
    growpe      (NBIN, NELEM)
    evappe      (NBIN, NELEM)
    growlg      (NBIN, NGROUP)
    evaplg      (NBIN, NGROUP)
    gasprod     (NGAS,)
    rlheat      (NZ,)
    pconmax     (NZ, NGROUP)
    coaglg      (NZ, NBIN, NGROUP)
    coagpe      (NZ, NBIN, NELEM)
    rmu         (NZ,)                                 — dynamic viscosity
    bpm         (NZ, NBIN, NGROUP)                    — Cunningham slip correction
    re          (NZ, NBIN, NGROUP)                    — Reynolds number
    vf          (NZP1, NBIN, NGROUP)                  — fall velocity
    pcl         (NZ, NBIN, NELEM)                     — prestep pc snapshot
    gcl         (NZ, NGAS)                            — prestep gc snapshot
    told        (NZ,)                                 — prestep t snapshot
    d_gc        (NZ, NGAS)                            — substep d_gc bookkeeping
    d_t         (NZ,)                                 — substep d_t bookkeeping
    zsubsteps   (NZ,)                                 — substeps used per layer
    nretries    (1,)                                  — retries used (scalar→1-arr)
    relhum      (NZ,)                                 — RH for swelling
    rhop        (NZ, NBIN, NGROUP)                    — dry particle density
    r_wet       (NZ, NBIN, NGROUP)                    — wet radius (getwetr)
    rhop_wet    (NZ, NBIN, NGROUP)                    — wet density (getwetr)
    cmf         (NBIN, NGROUP)                        — core mass fraction
    totevap     (NBIN, NGROUP)                        — total-evap flag (0/1 float)
    rup_wet     (NZ, NBIN, NGROUP)                    — wet upper-boundary radius
    ckernel     (NZ, NBIN, NBIN, NGROUP, NGROUP)      — only present at step 1
    r_bin       (NBIN, NGROUP)                        — only present at step 1 (static)
    rmass_bin   (NBIN, NGROUP)                        — only present at step 1 (static)
    rmassup_bin (NBIN, NGROUP)                        — only present at step 1 (static)
    rmrat_group (NGROUP,)                             — only present at step 1 (static)
    rrat        (NBIN, NGROUP)                        — only present at step 1 (static)
    rprat       (NBIN, NGROUP)                        — only present at step 1 (static)
    dm_bin      (NBIN, NGROUP)                        — only present at step 1 (static)
    pratt       (3, NBIN, NGROUP)                     — only present at step 1 (static)
    prat        (4, NBIN, NGROUP)                     — only present at step 1 (static)
    pden1       (NBIN, NGROUP)                        — only present at step 1 (static)
    palr        (4, NGROUP)                           — only present at step 1 (static)
    igrowgas    (NELEM,)                              — only present at step 1 (static, float-encoded int)
    inucgas     (NGROUP,)                             — only present at step 1 (static, float-encoded int)
    nnuc2elem   (NELEM,)                              — only present at step 1 (static, float-encoded int)
    ienconc     (NGROUP,)                             — only present at step 1 (static, float-encoded int)
    itype       (NELEM,)                              — only present at step 1 (static, float-encoded int)
    igelem      (NELEM,)                              — only present at step 1 (static, float-encoded int)
    is_grp_ice  (NGROUP,)                             — only present at step 1 (static, 0/1 float)
    inuc2elem   (NELEM, NELEM)                        — only present at step 1 (static, float-encoded int)
    inucproc    (NELEM, NELEM)                        — only present at step 1 (static, float-encoded int)
    kbin        (NGROUP, NGROUP, NGROUP, NBIN, NBIN)  — only at step 1 (static, float-encoded int)
    volx        (NGROUP, NGROUP, NGROUP, NBIN, NBIN)  — only at step 1 (static)
    pkernel     (NBIN, NBIN, NGROUP, NGROUP, NGROUP, 6) — only at step 1 (static)
    npairl      (NGROUP, NBIN)                        — only at step 1 (static, float-encoded int)
    npairu      (NGROUP, NBIN)                        — only at step 1 (static, float-encoded int)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np


# Default sulfate-test dimensions. These match the diagnostic binary's
# hard-coded constants (NZ=1, NBIN=38, NELEM=1, NGROUP=1, NGAS=2).
# If we ever instrument a multi-bin/multi-element test, expose these as
# parameters of read_substep().
_DEFAULT_DIMS = dict(NZ=1, NBIN=38, NELEM=1, NGROUP=1, NGAS=2)


def _shape_map(dims):
    NZ, NBIN, NELEM, NGROUP, NGAS = (
        dims["NZ"], dims["NBIN"], dims["NELEM"], dims["NGROUP"], dims["NGAS"],
    )
    NZP1 = NZ + 1
    return {
        "pc":         (NZ, NBIN, NELEM),
        "gc":         (NZ, NGAS),
        "t":          (NZ,),
        "p":          (NZ,),
        "rhoa":       (NZ,),
        "zmet":       (NZ,),
        "wtpct":      (NZ,),
        "sulfdens":   (NZ,),
        "sulfsurf":   (NZ,),
        "pvapl":      (NZ, NGAS),
        "pvapi":      (NZ, NGAS),
        "supsatl":    (NZ, NGAS),
        "supsati":    (NZ, NGAS),
        "akelvin":    (NZ, NGAS),
        "akelvini":   (NZ, NGAS),
        "gro":        (NZ, NBIN, NGROUP),
        "gro1":       (NZ, NBIN, NGROUP),
        "rhompe":     (NBIN, NELEM),
        "rnuclg":     (NBIN, NGROUP, NGROUP),
        "rnucpe":     (NBIN, NELEM),
        "growpe":     (NBIN, NELEM),
        "evappe":     (NBIN, NELEM),
        "growlg":     (NBIN, NGROUP),
        "evaplg":     (NBIN, NGROUP),
        "gasprod":    (NGAS,),
        "rlheat":     (NZ,),
        "pconmax":    (NZ, NGROUP),
        "coaglg":     (NZ, NBIN, NGROUP),
        "coagpe":     (NZ, NBIN, NELEM),
        "pcl":        (NZ, NBIN, NELEM),
        "gcl":        (NZ, NGAS),
        "told":       (NZ,),
        "d_gc":       (NZ, NGAS),
        "d_t":        (NZ,),
        "zsubsteps":  (NZ,),
        "nretries":   (1,),
        "relhum":     (NZ,),
        "rhop":       (NZ, NBIN, NGROUP),
        "r_wet":      (NZ, NBIN, NGROUP),
        "rhop_wet":   (NZ, NBIN, NGROUP),
        "cmf":        (NBIN, NGROUP),
        "totevap":    (NBIN, NGROUP),
        "rup_wet":    (NZ, NBIN, NGROUP),
        "rmu":        (NZ,),
        "bpm":        (NZ, NBIN, NGROUP),
        "re":         (NZ, NBIN, NGROUP),
        "vf":         (NZP1, NBIN, NGROUP),
        "ckernel":    (NZ, NBIN, NBIN, NGROUP, NGROUP),
        "r_bin":       (NBIN, NGROUP),
        "rmass_bin":   (NBIN, NGROUP),
        "rmassup_bin": (NBIN, NGROUP),
        "rmrat_group": (NGROUP,),
        "rrat":        (NBIN, NGROUP),
        "rprat":       (NBIN, NGROUP),
        "dm_bin":      (NBIN, NGROUP),
        "pratt":       (3, NBIN, NGROUP),
        "prat":        (4, NBIN, NGROUP),
        "pden1":       (NBIN, NGROUP),
        "palr":        (4, NGROUP),
        "igrowgas":    (NELEM,),
        "inucgas":     (NGROUP,),
        "nnuc2elem":   (NELEM,),
        "ienconc":     (NGROUP,),
        "itype":       (NELEM,),
        "igelem":      (NELEM,),
        "is_grp_ice":  (NGROUP,),
        "inuc2elem":   (NELEM, NELEM),
        "inucproc":    (NELEM, NELEM),
        "kbin":        (NGROUP, NGROUP, NGROUP, NBIN, NBIN),
        "volx":        (NGROUP, NGROUP, NGROUP, NBIN, NBIN),
        "pkernel":     (NBIN, NBIN, NGROUP, NGROUP, NGROUP, 6),
        "npairl":      (NGROUP, NBIN),
        "npairu":      (NGROUP, NBIN),
    }


@dataclass
class SubstepDump:
    """Typed container for one Fortran substep dump.

    Each attribute is a numpy float64 array reshaped from the Fortran
    column-major raw bytes. Missing arrays (e.g. ckernel for substeps
    after the first) are ``None``.
    """
    step: int
    pc:        np.ndarray
    gc:        np.ndarray
    t:         np.ndarray
    p:         np.ndarray
    rhoa:      np.ndarray
    zmet:      np.ndarray
    wtpct:     np.ndarray
    sulfdens:  np.ndarray
    sulfsurf:  np.ndarray
    pvapl:     np.ndarray
    pvapi:     np.ndarray
    supsatl:   np.ndarray
    supsati:   np.ndarray
    akelvin:   np.ndarray
    akelvini:  np.ndarray
    gro:       np.ndarray
    gro1:      np.ndarray
    rhompe:    np.ndarray
    rnuclg:    np.ndarray
    rnucpe:    np.ndarray
    growpe:    np.ndarray
    evappe:    np.ndarray
    growlg:    np.ndarray
    evaplg:    np.ndarray
    gasprod:   np.ndarray
    rlheat:    np.ndarray
    pconmax:   np.ndarray
    coaglg:    np.ndarray
    coagpe:    np.ndarray
    pcl:       np.ndarray
    gcl:       np.ndarray
    told:      np.ndarray
    d_gc:      np.ndarray
    d_t:       np.ndarray
    zsubsteps: np.ndarray
    nretries:  np.ndarray
    relhum:    np.ndarray
    rhop:      np.ndarray
    r_wet:     np.ndarray
    rhop_wet:  np.ndarray
    cmf:       np.ndarray
    totevap:   np.ndarray
    rup_wet:   np.ndarray
    rmu:       np.ndarray
    bpm:       np.ndarray
    re:        np.ndarray
    vf:        np.ndarray
    ckernel:   Optional[np.ndarray] = None
    r_bin:       Optional[np.ndarray] = None
    rmass_bin:   Optional[np.ndarray] = None
    rmassup_bin: Optional[np.ndarray] = None
    rmrat_group: Optional[np.ndarray] = None
    rrat:        Optional[np.ndarray] = None
    rprat:       Optional[np.ndarray] = None
    dm_bin:      Optional[np.ndarray] = None
    pratt:       Optional[np.ndarray] = None
    prat:        Optional[np.ndarray] = None
    pden1:       Optional[np.ndarray] = None
    palr:        Optional[np.ndarray] = None
    igrowgas:    Optional[np.ndarray] = None
    inucgas:     Optional[np.ndarray] = None
    nnuc2elem:   Optional[np.ndarray] = None
    ienconc:     Optional[np.ndarray] = None
    itype:       Optional[np.ndarray] = None
    igelem:      Optional[np.ndarray] = None
    is_grp_ice:  Optional[np.ndarray] = None
    inuc2elem:   Optional[np.ndarray] = None
    inucproc:    Optional[np.ndarray] = None
    kbin:        Optional[np.ndarray] = None
    volx:        Optional[np.ndarray] = None
    pkernel:     Optional[np.ndarray] = None
    npairl:      Optional[np.ndarray] = None
    npairu:      Optional[np.ndarray] = None


def _load_one(path: Path, expected_shape: tuple) -> np.ndarray:
    """Load a Fortran unformatted-stream binary as a column-major
    ndarray, reshape to `expected_shape`, and assert size match."""
    arr = np.fromfile(path, dtype=np.float64)
    n_expected = int(np.prod(expected_shape))
    if arr.size != n_expected:
        raise ValueError(
            f"{path.name}: file has {arr.size} elements, expected {n_expected} "
            f"for shape {expected_shape}"
        )
    return arr.reshape(expected_shape, order="F")


def read_substep(out_dir, step: int, dims=None) -> SubstepDump:
    """Read all per-array dumps for one substep into a SubstepDump.

    Args:
        out_dir: directory containing ``substep_<NNNN>_<name>.bin`` files.
        step: substep number (1-based, matches Fortran istep).
        dims: optional dimension overrides; defaults to NZ=1, NBIN=38,
            NELEM=1, NGROUP=1, NGAS=2 (the sulfate-test dimensions).

    Returns:
        SubstepDump with all numeric arrays populated. ``ckernel`` is
        ``None`` for any substep > 1 (the Fortran binary only dumps it
        at step 1 since it's static).
    """
    out_dir = Path(out_dir)
    if dims is None:
        dims = _DEFAULT_DIMS
    shapes = _shape_map(dims)

    def _path(name):
        return out_dir / f"substep_{step:04d}_{name}.bin"

    def _opt(name):
        p = _path(name)
        return _load_one(p, shapes[name]) if p.exists() else None

    return SubstepDump(
        step=step,
        pc=_load_one(_path("pc"), shapes["pc"]),
        gc=_load_one(_path("gc"), shapes["gc"]),
        t=_load_one(_path("t"), shapes["t"]),
        p=_load_one(_path("p"), shapes["p"]),
        rhoa=_load_one(_path("rhoa"), shapes["rhoa"]),
        zmet=_load_one(_path("zmet"), shapes["zmet"]),
        wtpct=_load_one(_path("wtpct"), shapes["wtpct"]),
        sulfdens=_load_one(_path("sulfdens"), shapes["sulfdens"]),
        sulfsurf=_load_one(_path("sulfsurf"), shapes["sulfsurf"]),
        pvapl=_load_one(_path("pvapl"), shapes["pvapl"]),
        pvapi=_load_one(_path("pvapi"), shapes["pvapi"]),
        supsatl=_load_one(_path("supsatl"), shapes["supsatl"]),
        supsati=_load_one(_path("supsati"), shapes["supsati"]),
        akelvin=_load_one(_path("akelvin"), shapes["akelvin"]),
        akelvini=_load_one(_path("akelvini"), shapes["akelvini"]),
        gro=_load_one(_path("gro"), shapes["gro"]),
        gro1=_load_one(_path("gro1"), shapes["gro1"]),
        rhompe=_load_one(_path("rhompe"), shapes["rhompe"]),
        rnuclg=_load_one(_path("rnuclg"), shapes["rnuclg"]),
        rnucpe=_load_one(_path("rnucpe"), shapes["rnucpe"]),
        growpe=_load_one(_path("growpe"), shapes["growpe"]),
        evappe=_load_one(_path("evappe"), shapes["evappe"]),
        growlg=_load_one(_path("growlg"), shapes["growlg"]),
        evaplg=_load_one(_path("evaplg"), shapes["evaplg"]),
        gasprod=_load_one(_path("gasprod"), shapes["gasprod"]),
        rlheat=_load_one(_path("rlheat"), shapes["rlheat"]),
        pconmax=_load_one(_path("pconmax"), shapes["pconmax"]),
        coaglg=_load_one(_path("coaglg"), shapes["coaglg"]),
        coagpe=_load_one(_path("coagpe"), shapes["coagpe"]),
        pcl=_load_one(_path("pcl"), shapes["pcl"]),
        gcl=_load_one(_path("gcl"), shapes["gcl"]),
        told=_load_one(_path("told"), shapes["told"]),
        d_gc=_load_one(_path("d_gc"), shapes["d_gc"]),
        d_t=_load_one(_path("d_t"), shapes["d_t"]),
        zsubsteps=_load_one(_path("zsubsteps"), shapes["zsubsteps"]),
        nretries=_load_one(_path("nretries"), shapes["nretries"]),
        relhum=_load_one(_path("relhum"), shapes["relhum"]),
        rhop=_load_one(_path("rhop"), shapes["rhop"]),
        r_wet=_load_one(_path("r_wet"), shapes["r_wet"]),
        rhop_wet=_load_one(_path("rhop_wet"), shapes["rhop_wet"]),
        cmf=_load_one(_path("cmf"), shapes["cmf"]),
        totevap=_load_one(_path("totevap"), shapes["totevap"]),
        rup_wet=_load_one(_path("rup_wet"), shapes["rup_wet"]),
        rmu=_load_one(_path("rmu"), shapes["rmu"]),
        bpm=_load_one(_path("bpm"), shapes["bpm"]),
        re=_load_one(_path("re"), shapes["re"]),
        vf=_load_one(_path("vf"), shapes["vf"]),
        ckernel=_opt("ckernel"),
        r_bin=_opt("r_bin"),
        rmass_bin=_opt("rmass_bin"),
        rmassup_bin=_opt("rmassup_bin"),
        rmrat_group=_opt("rmrat_group"),
        rrat=_opt("rrat"),
        rprat=_opt("rprat"),
        dm_bin=_opt("dm_bin"),
        pratt=_opt("pratt"),
        prat=_opt("prat"),
        pden1=_opt("pden1"),
        palr=_opt("palr"),
        igrowgas=_opt("igrowgas"),
        inucgas=_opt("inucgas"),
        nnuc2elem=_opt("nnuc2elem"),
        ienconc=_opt("ienconc"),
        itype=_opt("itype"),
        igelem=_opt("igelem"),
        is_grp_ice=_opt("is_grp_ice"),
        inuc2elem=_opt("inuc2elem"),
        inucproc=_opt("inucproc"),
        kbin=_opt("kbin"),
        volx=_opt("volx"),
        pkernel=_opt("pkernel"),
        npairl=_opt("npairl"),
        npairu=_opt("npairu"),
    )


_STEP_RE = re.compile(r"substep_(\d{4})_pc\.bin$")


def iter_substeps(out_dir, dims=None) -> Iterator[SubstepDump]:
    """Iterate over all dumped substeps in chronological order."""
    out_dir = Path(out_dir)
    steps = sorted({
        int(m.group(1))
        for m in (_STEP_RE.search(p.name) for p in out_dir.glob("substep_*_pc.bin"))
        if m
    })
    for s in steps:
        yield read_substep(out_dir, s, dims=dims)
