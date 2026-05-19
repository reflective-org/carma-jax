"""Full microfast driver for CARMA-JAX.

Orchestrates all fast microphysics processes in the correct order:
nucleation → growth rates → growp + upgxfer + psolve → evapp → gsolve → tsolve.
Handles multi-group configurations.
Ported from: microfast.F90
"""

import jax.numpy as jnp

from carma.constants import SMALL_PC, FEW_PC, CP, RGAS, RHO_W
from carma.enums import ElementType, RC_OK
from carma.precision import DTYPE, POWMAX
from carma.solvers.psolve import psolve
from carma.solvers.totalcondensate import totalcondensate
from carma.growth.growp import growp
from carma.growth.evapp import evapp, downgevapply
from carma.growth.upgxfer import upgxfer


def microfast_step(pc, gc, t, iz, dtime,
                    # Atmospheric state
                    rhoa, zmet, p,
                    # Vapor pressure / supersaturation
                    pvapl, pvapi, supsatl, supsati,
                    # Growth kernels
                    growlg, evaplg, gro, gro1,
                    rup_wet, rmass_2d, dm_2d,
                    pratt, prat, pden1, palr,
                    # Latent heats
                    rlhe, rlhm,
                    # Nucleation arrays
                    rnuclg,  # (NBIN, NGROUP, NGROUP)
                    # Config
                    is_ice_arr, igrowgas_arr, ienconc_arr,
                    igroup_arr, itype_arr, gwtmol_arr,
                    # Nucleation config
                    nnucelem, inucelem, nnucbin, inucbin,
                    # Dimensions
                    nbin, ngroup, ngas, nelem,
                    # Thresholds
                    dt_threshold=0.0, scale_threshold=1.0):
    """Execute one microfast step at one level.

    Full call sequence matching Fortran microfast.F90:
    1. Save condensate
    2. [Nucleation routines set rnuclg externally before calling this]
    3. growevapl (growth/evaporation loss rates)
    4. For each elem/bin: growp + upgxfer + psolve
    5. evapp + downgevapply
    6. gsolve (gas conservation)
    7. tsolve (temperature)

    Args:
        ... (extensive, matching full CARMA state)

    Returns:
        Tuple of (pc, gc, t, rlheat_val).
    """
    # Save condensate before microphysics
    prev_ice, prev_liq = totalcondensate(
        pc, rmass_2d, igroup_arr, is_ice_arr, igrowgas_arr,
        nbin, ngroup, ngas, iz,
    )

    # Zero production arrays
    growpe_arr = jnp.zeros((nbin, nelem), dtype=DTYPE)
    evappe_arr = jnp.zeros((nbin, nelem), dtype=DTYPE)
    rnucpe_arr = jnp.zeros((nbin, nelem), dtype=DTYPE)
    rhompe_arr = jnp.zeros((nbin, nelem), dtype=DTYPE)
    pc_nucl = jnp.zeros_like(pc)
    pconmax = jnp.zeros((pc.shape[0], ngroup), dtype=DTYPE)

    # Compute pconmax for each group
    for ig in range(ngroup):
        ie = int(ienconc_arr[ig])
        pconmax = pconmax.at[:, ig].set(
            jnp.max(pc[:, :, ie], axis=1) / zmet
        )

    # Growth production + nucleation production + solve (per elem, per bin)
    for ielem in range(nelem):
        ig = int(igroup_arr[ielem])
        # Use group-level growth gas (all elements in a growing group participate)
        iepart = int(ienconc_arr[ig])
        igrow = int(igrowgas_arr[iepart])
        for ibin in range(nbin):
            # Growth production (from bin i-1 growing into bin i)
            growpe_arr = growp(pc, growpe_arr, growlg, pconmax, iz, ibin, ielem, ig, igrow)

            # Nucleation production (from other groups nucleating into this bin)
            rnucpe_arr = upgxfer(
                rnucpe_arr, rnuclg, pc, rmass_2d, pconmax,
                ielem, ibin, iz,
                nnucelem, inucelem, nnucbin, inucbin,
                igroup_arr, itype_arr, nbin, ngroup,
            )

            # Solve for new particle concentration
            pc, pc_nucl = psolve(
                pc, pc_nucl, growpe_arr, evappe_arr, rnucpe_arr, rhompe_arr,
                growlg, evaplg, rnuclg, dtime, iz, ibin, ielem, ig, ngroup,
            )

    # Evaporation production
    evappe_arr = jnp.zeros((nbin, nelem), dtype=DTYPE)
    evappe_arr = evapp(
        pc, evappe_arr, evaplg, pconmax, ienconc_arr, itype_arr,
        igroup_arr, iz, nbin, ngroup, nelem,
    )

    # Apply evaporation
    rnucpe_zero = jnp.zeros((nbin, nelem), dtype=DTYPE)
    pc = downgevapply(pc, evappe_arr, rnucpe_zero, dtime, iz, nbin, nelem)

    # Gas solver: conserve mass
    curr_ice, curr_liq = totalcondensate(
        pc, rmass_2d, igroup_arr, is_ice_arr, igrowgas_arr,
        nbin, ngroup, ngas, iz,
    )

    rlprod = DTYPE(0.0)
    for igas in range(ngas):
        gasprod = (
            (prev_ice[igas] - curr_ice[igas])
            + (prev_liq[igas] - curr_liq[igas])
        ) / dtime

        ice_change = prev_ice[igas] - curr_ice[igas]
        liq_change = prev_liq[igas] - curr_liq[igas]
        rlprod = rlprod - (
            ice_change * (rlhe[iz, igas] + rlhm[iz, igas])
            + liq_change * rlhe[iz, igas]
        ) / (CP * rhoa[iz] * dtime)

        gc = gc.at[iz, igas].add(dtime * gasprod)

    # Temperature solver
    dt_val = dtime * rlprod
    t = t.at[iz].add(dt_val)
    rlheat_val = rlprod * dtime

    return pc, gc, t, rlheat_val
