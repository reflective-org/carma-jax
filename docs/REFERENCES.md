# CARMA-JAX References and Sources

## Fortran Source Code

- **CARMA source**: `../original-carma/CARMA/source/base/` (~90 F90 files)
- **CARMA tests**: `../original-carma/CARMA/tests/` (32+ test programs)
- **Benchmark data**: `../original-carma/CARMA/tests/bench/` (text output files)
- **CARMA documentation**: `../original-carma/CARMA/doc/`
- **CARMA repository**: https://github.com/ESCOMP/CARMA

## Papers by Phase

### Phase 7: H₂SO₄ sulfate physics

- Tabazadeh, A., R.P. Turco, M.Z. Jacobson (1997). "A model for studying the composition and chemical effects of stratospheric aerosols." J. Geophys. Res. 99(D6), 12897–12914.
  - Wt% H₂SO₄ piecewise activity fits, valid T = 185–260 K, activity = 0.01–1
  - Used by `wtpct_tabaz` in `src/carma/sulfate_utils.py`

- Washburn, E.W. ed. (1928). "International Critical Tables." NRC.
  - Tabulated H₂SO₄/H₂O solution densities at 0–100 °C
  - Used by `sulfate_density` in `src/carma/sulfate_utils.py`

- Beyer, K.D., A.R. Ravishankara, E.R. Lovejoy (1996). "Measurements of the surface tension and density of sulfuric acid solutions at low temperatures." J. Geophys. Res. 101(D9), 14519–14524.
  - Validates linear-in-T extrapolation of Washburn density to 180–380 K

- Sabinina, L. and L. Terpugow (1935). "Die Oberflächenspannung des Systems Schwefelsäure-Wasser." Z. phys. Chem. A 173, 237–241.
  - Surface tension fits used by `sulfate_surf_tens` in `src/carma/sulfate_utils.py`

- Mills, M.J. (1996). "Stratospheric Sulfate Aerosol: A Microphysical Model." Ph.D. Thesis, Univ. of Colorado.
  - Source of the surface-tension table coefficients (`stwtp`, `stc0`, `stc1`)

### Phase 1: Coagulation

- Fuchs, N.A. (1964). "The Mechanics of Aerosols." Pergamon Press.
  - Brownian coagulation kernel derivation
  - Referenced in: `setupckern.F90`

- Jacobson, M.Z., Turco, R.P., Jensen, E.J., Toon, O.B. (1994). "Modeling coagulation among particles of different composition and size." Atmos. Environ., 28, 1327-1338.
  - Coagulation validation benchmark (Fig 2 used in coagtest)
  - Referenced in: `carma_coagtest.F90`

### Phase 2: Thermodynamics & Growth Setup

- Buck, A.L. (1981). "New equations for computing vapor pressure and enhancement factor." J. Appl. Meteorol., 20, 1527-1532.
  - H2O vapor pressure parameterization
  - Referenced in: `vaporp_h2o_buck1981.F90`

- Murphy, D.M. and Koop, T. (2005). "Review of the vapour pressures of ice and supercooled water for atmospheric applications." Q. J. R. Meteorol. Soc., 131, 1539-1565.
  - Alternative H2O vapor pressure
  - Referenced in: `vaporp_h2o_murphy2005.F90`

- Ayers, G.P., Gillett, R.W., Gras, J.L. (1980). "On the vapor pressure of sulfuric acid." Geophys. Res. Lett., 7, 433-436.
  - H2SO4 vapor pressure parameterization
  - Referenced in: `vaporp_h2so4_ayers1980.F90`

- Heymsfield, A.J. and Westbrook, C.D. (2010). "Advances in the estimation of ice particle fall speeds using laboratory and field measurements." J. Atmos. Sci., 67, 2469-2482.
  - Ice crystal fall velocity parameterization
  - Referenced in: `setupvf_heymsfield2010.F90`

- Pruppacher, H.R. and Klett, J.D. (1997). "Microphysics of Clouds and Precipitation." Kluwer Academic Publishers.
  - Condensational growth equation, ventilation factors, fall velocity regimes
  - Referenced in: `setupgkern.F90`, `setupvf_std.F90`, `growevapl.F90`

### Phase 3: Nucleation (most critical for papers)

- Zhao, J. and Turco, R.P. (1995). "Nucleation simulations in the wake of a jet aircraft in stratospheric conditions." J. Aerosol Sci., 26, 779-795.
  - Homogeneous H2SO4-H2O binary nucleation (default method)
  - Referenced in: `sulfnucrate.F90` (binary_nuc_zhao1995)

- Vehkamaki, H., Kulmala, M., Napari, I., Lehtinen, K.E.J., Timmreck, C., Noppel, M., Laaksonen, A. (2002). "An improved parameterization for sulfuric acid-water nucleation rates for tropospheric and stratospheric conditions." J. Geophys. Res., 107, 4622.
  - Alternative nucleation parameterization (polynomial fits)
  - Referenced in: `sulfnucrate.F90` (binary_nuc_vehk2002)

- Tabazadeh, A., Martin, S.T., Lin, J.S. (2000). "The effect of particle size and nitric acid uptake on the homogeneous freezing of aqueous sulfuric acid particles." Geophys. Res. Lett., 27, 1111-1114.
  - Sulfuric acid aerosol freezing parameterization
  - Referenced in: `freezaerl_tabazadeh2000.F90`

- Koop, T., Luo, B., Tsias, A., Peter, T. (2000). "Water activity as the determinant for homogeneous ice nucleation in aqueous solutions." Nature, 406, 611-614.
  - Water activity-based homogeneous ice nucleation
  - Referenced in: `freezaerl_koop2000.F90`

- Mohler, O., et al. (2010). Heterogeneous ice nucleation parameterization.
  - Referenced in: `freezaerl_mohler2010.F90`

- Murray, B.J., et al. (2010). "Heterogeneous nucleation of ice particles on glassy aerosols under cirrus conditions." Nature Geosci., 3, 233-237.
  - Glassy aerosol freezing
  - Referenced in: `freezglaerl_murray2010.F90`

- Rapp, M. and Thomas, G.E. (2006). "Modeling the microphysics of mesospheric ice particles." J. Atmos. Sol.-Terr. Phys., 68, 715-744.
  - Heterogeneous deposition nucleation
  - Referenced in: `hetnucl.F90`

### Phase 4: Vertical Transport

- Colella, P. and Woodward, P.R. (1984). "The Piecewise Parabolic Method (PPM) for gas-dynamical simulations." J. Comput. Phys., 54, 174-201.
  - PPM advection scheme used in `vertadv.F90` and `versol.F90`

- Zhang, L., Gong, S., Padro, J., Barrie, L. (2001). "A size-segregated particle dry deposition scheme for an atmospheric aerosol module." Atmos. Environ., 35, 549-560.
  - Dry deposition parameterization
  - Referenced in: `setupvdry.F90`

- Seinfeld, J.H. and Pandis, S.N. (1998). "Atmospheric Chemistry and Physics." John Wiley & Sons.
  - Brownian diffusion coefficient (Eq. 8.73)
  - Referenced in: `setupbdif.F90`

### Background / CARMA Formulation

- Turco, R.P., Hamill, P., Toon, O.B., Whitten, R.C., Kiang, C.S. (1979). "A one-dimensional model describing aerosol formation and evolution in the stratosphere." J. Phys. Chem., 83, 2210-2223.
  - Original CARMA formulation

- Toon, O.B., Turco, R.P., Westphal, D., Malone, R., Liu, M.S. (1988). "A multidimensional model for aerosols: Description of computational analogs." J. Atmos. Sci., 45, 2123-2143.
  - Original CARMA formulation

- Bardeen, C.G., Toon, O.B., Jensen, E.J., Marsh, D.R., Harvey, V.L. (2008). "Numerical simulations of the three-dimensional distribution of meteoric dust in the mesosphere and upper stratosphere." J. Geophys. Res., 113, D17202.
  - CARMA in WACCM application
