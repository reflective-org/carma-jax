"""Unit tests for gasexchange (nucleation + growth gas flux)."""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from carma.gasexchange import gasexchange


def _sulfate_single_group():
    """Minimal single-group, single-gas sulfate setup:
       1 group with 1 element (number conc) that nucleates into itself.
       1 gas (H2SO4) that is both inucgas and igrowgas."""
    nbin = 4
    ngroup = 1
    nelem = 1
    ngas = 1

    rmass = jnp.asarray([[1e-20], [2e-20], [4e-20], [8e-20]])   # (nbin, ngroup)
    rmrat = 2.0

    # diffmass[target_bin, target_grp, src_bin, src_grp] = rmass[tgt] - rmass[src]
    # Simple absolute difference; Fortran uses this for het nucleation + growth.
    rmass_tgt = rmass[:, :, None, None]             # (nbin, ngroup, 1, 1)
    rmass_src = rmass[None, None, :, :]             # (1, 1, nbin, ngroup)
    diffmass = jnp.broadcast_to(
        rmass_tgt - rmass_src, (nbin, ngroup, nbin, ngroup)
    )

    # inuc2bin[src_bin, src_grp, tgt_grp]: het nucleation grows from bin i to i+1
    i2_map = np.arange(nbin) + 1
    i2_map[-1] = -1          # top bin has no target
    inuc2bin = jnp.asarray(i2_map.reshape(nbin, 1, 1))

    # Config indices stay as Python/numpy arrays (resolved at trace time)
    if_nuc = np.asarray([[True]])                   # sulfate → sulfate
    ienconc = np.asarray([0])
    igelem = np.asarray([0])
    inucgas = np.asarray([0])
    nnuc2elem = np.asarray([1])                     # ielem=0 nucleates 1 element
    igrowgas = np.asarray([0])

    return dict(
        nbin=nbin, ngroup=ngroup, nelem=nelem, ngas=ngas,
        rmass=rmass, diffmass=diffmass, inuc2bin=inuc2bin,
        if_nuc=if_nuc, ienconc=ienconc, igelem=igelem,
        inucgas=inucgas, nnuc2elem=nnuc2elem, igrowgas=igrowgas,
    )


# --- shape + no-flux ---

def test_output_shape():
    cfg = _sulfate_single_group()
    zeros_nbin = jnp.zeros((cfg["nbin"], cfg["nelem"]))
    zeros_rng = jnp.zeros((cfg["nbin"], cfg["ngroup"]))
    zeros_het = jnp.zeros((cfg["nbin"], cfg["ngroup"], cfg["ngroup"]))
    cmf = jnp.zeros((cfg["nbin"], cfg["ngroup"]))
    totevap = jnp.zeros((cfg["nbin"], cfg["ngroup"]), dtype=bool)
    out = gasexchange(
        pc_iz=zeros_nbin, rhompe=zeros_nbin, rnuclg=zeros_het,
        growlg=zeros_rng, evaplg=zeros_rng,
        rmass=cfg["rmass"], diffmass=cfg["diffmass"],
        cmf=cmf, totevap=totevap, inuc2bin=cfg["inuc2bin"],
        if_nuc=cfg["if_nuc"], ienconc=cfg["ienconc"], igelem=cfg["igelem"],
        inucgas=cfg["inucgas"], nnuc2elem=cfg["nnuc2elem"],
        igrowgas=cfg["igrowgas"],
        ngas=cfg["ngas"], ngroup=cfg["ngroup"],
        nelem=cfg["nelem"], nbin=cfg["nbin"],
    )
    assert out.shape == (1,)
    assert float(out[0]) == 0.0


# --- homogeneous-only ---

def test_homogeneous_only_consumes_gas():
    """rhompe=1 in bin 2 → gas loss = rhompe · rmass[2]."""
    cfg = _sulfate_single_group()
    nbin = cfg["nbin"]
    rhompe = jnp.zeros((nbin, 1)).at[2, 0].set(5.0)
    zeros_rng = jnp.zeros((nbin, 1))
    zeros_het = jnp.zeros((nbin, 1, 1))
    cmf = jnp.zeros((nbin, 1))
    totevap = jnp.zeros((nbin, 1), dtype=bool)

    out = gasexchange(
        pc_iz=jnp.zeros((nbin, 1)), rhompe=rhompe, rnuclg=zeros_het,
        growlg=zeros_rng, evaplg=zeros_rng,
        rmass=cfg["rmass"], diffmass=cfg["diffmass"],
        cmf=cmf, totevap=totevap, inuc2bin=cfg["inuc2bin"],
        if_nuc=cfg["if_nuc"], ienconc=cfg["ienconc"], igelem=cfg["igelem"],
        inucgas=cfg["inucgas"], nnuc2elem=cfg["nnuc2elem"],
        igrowgas=cfg["igrowgas"],
        ngas=cfg["ngas"], ngroup=cfg["ngroup"],
        nelem=cfg["nelem"], nbin=cfg["nbin"],
    )
    expected = -5.0 * 4e-20
    assert abs(float(out[0]) - expected) < 1e-30


# --- heterogeneous-only ---

def test_heterogeneous_only_consumes_gas():
    """rnuclg at bin 1 with pc=10 grows into bin 2: gas loss =
    10 · rnuclg · (rmass[2] - rmass[1])."""
    cfg = _sulfate_single_group()
    nbin = cfg["nbin"]
    pc_iz = jnp.zeros((nbin, 1)).at[1, 0].set(10.0)
    rnuclg = jnp.zeros((nbin, 1, 1)).at[1, 0, 0].set(0.5)
    zeros_nbin = jnp.zeros((nbin, 1))
    zeros_rng = jnp.zeros((nbin, 1))
    cmf = jnp.zeros((nbin, 1))
    totevap = jnp.zeros((nbin, 1), dtype=bool)

    out = gasexchange(
        pc_iz=pc_iz, rhompe=zeros_nbin, rnuclg=rnuclg,
        growlg=zeros_rng, evaplg=zeros_rng,
        rmass=cfg["rmass"], diffmass=cfg["diffmass"],
        cmf=cmf, totevap=totevap, inuc2bin=cfg["inuc2bin"],
        if_nuc=cfg["if_nuc"], ienconc=cfg["ienconc"], igelem=cfg["igelem"],
        inucgas=cfg["inucgas"], nnuc2elem=cfg["nnuc2elem"],
        igrowgas=cfg["igrowgas"],
        ngas=cfg["ngas"], ngroup=cfg["ngroup"],
        nelem=cfg["nelem"], nbin=cfg["nbin"],
    )
    # source bin = 1, target = 2; diffmass = rmass[2]-rmass[1] = 4e-20 - 2e-20
    expected = -10.0 * 0.5 * (4e-20 - 2e-20)
    assert abs(float(out[0]) - expected) < 1e-30


def test_heterogeneous_top_bin_no_target():
    """inuc2bin[nbin-1] = -1 → that bin contributes zero."""
    cfg = _sulfate_single_group()
    nbin = cfg["nbin"]
    pc_iz = jnp.ones((nbin, 1))
    rnuclg = jnp.ones((nbin, 1, 1))         # uniform nuclg
    zeros_nbin = jnp.zeros((nbin, 1))
    zeros_rng = jnp.zeros((nbin, 1))
    cmf = jnp.zeros((nbin, 1))
    totevap = jnp.zeros((nbin, 1), dtype=bool)

    out = gasexchange(
        pc_iz=pc_iz, rhompe=zeros_nbin, rnuclg=rnuclg,
        growlg=zeros_rng, evaplg=zeros_rng,
        rmass=cfg["rmass"], diffmass=cfg["diffmass"],
        cmf=cmf, totevap=totevap, inuc2bin=cfg["inuc2bin"],
        if_nuc=cfg["if_nuc"], ienconc=cfg["ienconc"], igelem=cfg["igelem"],
        inucgas=cfg["inucgas"], nnuc2elem=cfg["nnuc2elem"],
        igrowgas=cfg["igrowgas"],
        ngas=cfg["ngas"], ngroup=cfg["ngroup"],
        nelem=cfg["nelem"], nbin=cfg["nbin"],
    )
    # Only bins 0..2 contribute: sum of (rmass[i+1] - rmass[i]) for i in [0,1,2]
    dm = np.asarray(cfg["rmass"])[:, 0]
    expected = -sum(dm[i + 1] - dm[i] for i in range(3))
    assert abs(float(out[0]) - expected) < 1e-30


# --- growth / evap ---

def test_growth_consumes_gas():
    """Growth from bin 0→1 with pc=3, growlg=0.5 → gas loss =
    0.5 · 3 · (rmass[1] - rmass[0])."""
    cfg = _sulfate_single_group()
    nbin = cfg["nbin"]
    pc_iz = jnp.zeros((nbin, 1)).at[0, 0].set(3.0)
    growlg = jnp.zeros((nbin, 1)).at[0, 0].set(0.5)
    zeros_nbin = jnp.zeros((nbin, 1))
    zeros_rng = jnp.zeros((nbin, 1))
    zeros_het = jnp.zeros((nbin, 1, 1))
    cmf = jnp.zeros((nbin, 1))
    totevap = jnp.zeros((nbin, 1), dtype=bool)
    # Use if_nuc=False so only growth path runs
    if_nuc_off = np.asarray([[False]])

    out = gasexchange(
        pc_iz=pc_iz, rhompe=zeros_nbin, rnuclg=zeros_het,
        growlg=growlg, evaplg=zeros_rng,
        rmass=cfg["rmass"], diffmass=cfg["diffmass"],
        cmf=cmf, totevap=totevap, inuc2bin=cfg["inuc2bin"],
        if_nuc=if_nuc_off, ienconc=cfg["ienconc"], igelem=cfg["igelem"],
        inucgas=cfg["inucgas"], nnuc2elem=cfg["nnuc2elem"],
        igrowgas=cfg["igrowgas"],
        ngas=cfg["ngas"], ngroup=cfg["ngroup"],
        nelem=cfg["nelem"], nbin=cfg["nbin"],
    )
    expected = -0.5 * 3.0 * (2e-20 - 1e-20)
    assert abs(float(out[0]) - expected) < 1e-30


def test_evap_gains_gas_partial():
    """Partial evap from bin 2 → bin 1, cmf=0, no totevap:
    gain = evaplg[2] · pc[2] · diffmass[2, 1] = 0.5 · 4 · (4e-20 - 2e-20)."""
    cfg = _sulfate_single_group()
    nbin = cfg["nbin"]
    pc_iz = jnp.zeros((nbin, 1)).at[2, 0].set(4.0)
    evaplg = jnp.zeros((nbin, 1)).at[2, 0].set(0.5)
    zeros_nbin = jnp.zeros((nbin, 1))
    zeros_rng = jnp.zeros((nbin, 1))
    zeros_het = jnp.zeros((nbin, 1, 1))
    cmf = jnp.zeros((nbin, 1))
    totevap = jnp.zeros((nbin, 1), dtype=bool)
    if_nuc_off = np.asarray([[False]])

    out = gasexchange(
        pc_iz=pc_iz, rhompe=zeros_nbin, rnuclg=zeros_het,
        growlg=zeros_rng, evaplg=evaplg,
        rmass=cfg["rmass"], diffmass=cfg["diffmass"],
        cmf=cmf, totevap=totevap, inuc2bin=cfg["inuc2bin"],
        if_nuc=if_nuc_off, ienconc=cfg["ienconc"], igelem=cfg["igelem"],
        inucgas=cfg["inucgas"], nnuc2elem=cfg["nnuc2elem"],
        igrowgas=cfg["igrowgas"],
        ngas=cfg["ngas"], ngroup=cfg["ngroup"],
        nelem=cfg["nelem"], nbin=cfg["nbin"],
    )
    # evap_term: evaplg[2] · pc[2] · (rmass[2]-rmass[1])
    expected = 0.5 * 4.0 * (4e-20 - 2e-20)
    assert abs(float(out[0]) - expected) < 1e-30


def test_totevap_out_of_bin_0():
    """evaplg[0] · pc[0] · (1 - cmf[0]) · rmass[0] — always a gain."""
    cfg = _sulfate_single_group()
    nbin = cfg["nbin"]
    pc_iz = jnp.zeros((nbin, 1)).at[0, 0].set(2.0)
    evaplg = jnp.zeros((nbin, 1)).at[0, 0].set(0.7)
    zeros_nbin = jnp.zeros((nbin, 1))
    zeros_rng = jnp.zeros((nbin, 1))
    zeros_het = jnp.zeros((nbin, 1, 1))
    cmf = jnp.zeros((nbin, 1)).at[0, 0].set(0.2)  # 20% is core mass
    totevap = jnp.zeros((nbin, 1), dtype=bool)
    if_nuc_off = np.asarray([[False]])

    out = gasexchange(
        pc_iz=pc_iz, rhompe=zeros_nbin, rnuclg=zeros_het,
        growlg=zeros_rng, evaplg=evaplg,
        rmass=cfg["rmass"], diffmass=cfg["diffmass"],
        cmf=cmf, totevap=totevap, inuc2bin=cfg["inuc2bin"],
        if_nuc=if_nuc_off, ienconc=cfg["ienconc"], igelem=cfg["igelem"],
        inucgas=cfg["inucgas"], nnuc2elem=cfg["nnuc2elem"],
        igrowgas=cfg["igrowgas"],
        ngas=cfg["ngas"], ngroup=cfg["ngroup"],
        nelem=cfg["nelem"], nbin=cfg["nbin"],
    )
    # Only the "total evap out of bin 0" term fires because evaplg[1:] == 0
    expected = 0.7 * 2.0 * (1.0 - 0.2) * 1e-20
    assert abs(float(out[0]) - expected) < 1e-30


def test_totevap_flag_uses_rmass_not_diffmass():
    """When totevap[i+1] is True, gasgain = (1-cmf)·rmass, not diffmass."""
    cfg = _sulfate_single_group()
    nbin = cfg["nbin"]
    pc_iz = jnp.zeros((nbin, 1)).at[1, 0].set(6.0)
    evaplg = jnp.zeros((nbin, 1)).at[1, 0].set(0.3)
    zeros_nbin = jnp.zeros((nbin, 1))
    zeros_rng = jnp.zeros((nbin, 1))
    zeros_het = jnp.zeros((nbin, 1, 1))
    cmf = jnp.zeros((nbin, 1)).at[1, 0].set(0.1)
    totevap = jnp.zeros((nbin, 1), dtype=bool).at[1, 0].set(True)
    if_nuc_off = np.asarray([[False]])

    out = gasexchange(
        pc_iz=pc_iz, rhompe=zeros_nbin, rnuclg=zeros_het,
        growlg=zeros_rng, evaplg=evaplg,
        rmass=cfg["rmass"], diffmass=cfg["diffmass"],
        cmf=cmf, totevap=totevap, inuc2bin=cfg["inuc2bin"],
        if_nuc=if_nuc_off, ienconc=cfg["ienconc"], igelem=cfg["igelem"],
        inucgas=cfg["inucgas"], nnuc2elem=cfg["nnuc2elem"],
        igrowgas=cfg["igrowgas"],
        ngas=cfg["ngas"], ngroup=cfg["ngroup"],
        nelem=cfg["nelem"], nbin=cfg["nbin"],
    )
    # evap_term (i+1=1): evaplg[1] · pc[1] · (1-cmf[1]) · rmass[1]  (not diffmass)
    expected = 0.3 * 6.0 * (1.0 - 0.1) * 2e-20
    assert abs(float(out[0]) - expected) < 1e-30


# --- full combination ---

def test_full_combined_path():
    """Turn on homogeneous, heterogeneous, growth and evap together; result
    must equal the sum of the individual-path outputs."""
    cfg = _sulfate_single_group()
    nbin = cfg["nbin"]

    rhompe = jnp.zeros((nbin, 1)).at[2, 0].set(5.0)
    rnuclg = jnp.zeros((nbin, 1, 1)).at[1, 0, 0].set(0.5)
    growlg = jnp.zeros((nbin, 1)).at[0, 0].set(0.5)
    evaplg = jnp.zeros((nbin, 1)).at[2, 0].set(0.5)
    pc_iz = jnp.zeros((nbin, 1))
    pc_iz = pc_iz.at[0, 0].set(3.0).at[1, 0].set(10.0).at[2, 0].set(4.0)
    cmf = jnp.zeros((nbin, 1))
    totevap = jnp.zeros((nbin, 1), dtype=bool)

    out = gasexchange(
        pc_iz=pc_iz, rhompe=rhompe, rnuclg=rnuclg,
        growlg=growlg, evaplg=evaplg,
        rmass=cfg["rmass"], diffmass=cfg["diffmass"],
        cmf=cmf, totevap=totevap, inuc2bin=cfg["inuc2bin"],
        if_nuc=cfg["if_nuc"], ienconc=cfg["ienconc"], igelem=cfg["igelem"],
        inucgas=cfg["inucgas"], nnuc2elem=cfg["nnuc2elem"],
        igrowgas=cfg["igrowgas"],
        ngas=cfg["ngas"], ngroup=cfg["ngroup"],
        nelem=cfg["nelem"], nbin=cfg["nbin"],
    )

    hom = -5.0 * 4e-20
    het = -10.0 * 0.5 * (4e-20 - 2e-20)
    growth = -0.5 * 3.0 * (2e-20 - 1e-20)
    evap = 0.5 * 4.0 * (4e-20 - 2e-20)
    expected = hom + het + growth + evap
    assert abs(float(out[0]) - expected) < 1e-30


# --- JIT ---

def test_gasexchange_jits():
    """gasexchange JITs when config arrays are captured via closure
    (they aren't meant to be traced — they drive Python-level loops)."""
    cfg = _sulfate_single_group()
    nbin = cfg["nbin"]
    rhompe = jnp.zeros((nbin, 1)).at[2, 0].set(5.0)
    rnuclg = jnp.zeros((nbin, 1, 1))
    growlg = jnp.zeros((nbin, 1))
    evaplg = jnp.zeros((nbin, 1))
    pc_iz = jnp.zeros((nbin, 1))
    cmf = jnp.zeros((nbin, 1))
    totevap = jnp.zeros((nbin, 1), dtype=bool)

    # Close over static config params — Python-level ints/arrays
    def _wrap(pc_iz, rhompe, rnuclg, growlg, evaplg, cmf, totevap):
        return gasexchange(
            pc_iz=pc_iz, rhompe=rhompe, rnuclg=rnuclg,
            growlg=growlg, evaplg=evaplg,
            rmass=cfg["rmass"], diffmass=cfg["diffmass"],
            cmf=cmf, totevap=totevap, inuc2bin=cfg["inuc2bin"],
            if_nuc=cfg["if_nuc"], ienconc=cfg["ienconc"], igelem=cfg["igelem"],
            inucgas=cfg["inucgas"], nnuc2elem=cfg["nnuc2elem"],
            igrowgas=cfg["igrowgas"],
            ngas=cfg["ngas"], ngroup=cfg["ngroup"],
            nelem=cfg["nelem"], nbin=cfg["nbin"],
        )

    eager = _wrap(pc_iz, rhompe, rnuclg, growlg, evaplg, cmf, totevap)
    jit_out = jax.jit(_wrap)(pc_iz, rhompe, rnuclg, growlg, evaplg, cmf, totevap)
    assert jnp.allclose(eager, jit_out)
