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
    ckernel     (NZ, NBIN, NBIN, NGROUP, NGROUP)  — only present at step 1
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
    return {
        "pc":         (NZ, NBIN, NELEM),
        "gc":         (NZ, NGAS),
        "t":          (NZ,),
        "p":          (NZ,),
        "rhoa":       (NZ,),
        "zmet":       (NZ,),
        "wtpct":      (NZ,),
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
        "ckernel":    (NZ, NBIN, NBIN, NGROUP, NGROUP),
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
    ckernel:   Optional[np.ndarray] = None


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
        ckernel=_opt("ckernel"),
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
