"""Physical constants for CARMA-JAX in CGS units.

Ported from: carma_constants_mod.F90
"""

from carma.precision import DTYPE

# Temperature
T0 = DTYPE(273.16)  # Triple-point temperature [K]

# Mathematical
PI = DTYPE(3.14159265358979)
TWOPI = DTYPE(2.0) * PI
DEG2RAD = PI / DTYPE(180.0)
RAD2DEG = DTYPE(180.0) / PI

# Gravity and Earth
GRAV = DTYPE(980.6)  # Gravitational acceleration [cm/s^2]
REARTH = DTYPE(6.37e8)  # Earth radius [cm]

# Molecular
AVG = DTYPE(6.02252e23)  # Avogadro's number [particles/mole]
BK = DTYPE(1.38054e-16)  # Boltzmann constant [erg/K]
ALOS = DTYPE(2.68719e19)  # Loschmidt's number [mole/cm^3 at STP]

# Air properties
WTMOL_AIR = DTYPE(28.966)  # Molecular weight of dry air [g/mole]
WTMOL_H2O = DTYPE(18.016)  # Molecular weight of water [g/mole]
RGAS = DTYPE(8.31430e7)  # Universal gas constant [erg/K/mole]
R_AIR = RGAS / WTMOL_AIR  # Gas constant for dry air [erg/K/g]
CP = DTYPE(1.004e7)  # Specific heat at constant pressure [cm^2/s^2/K]
RKAPPA = R_AIR / CP  # R_AIR / CP ratio

# Reference pressure
PREF = DTYPE(1000.0e3)  # Reference pressure [dyne/cm^2]

# Unit conversions (to CGS)
RMB2CGS = DTYPE(1000.0)  # millibar to dyne/cm^2
RPA2CGS = DTYPE(10.0)  # Pascal to dyne/cm^2
RM2CGS = DTYPE(100.0)  # meter to cm

# Time
SCDAY = DTYPE(86400.0)  # Seconds per day

# Water and ice densities
RHO_W = DTYPE(1.0)  # Liquid water density [g/cm^3]
RHO_I = DTYPE(0.93)  # Ice density [g/cm^3]

# Latent heats
RLHE_CNST = DTYPE(2.501e10)  # Latent heat of evaporation [cm^2/s^2]
RLHM_CNST = DTYPE(3.337e9)  # Latent heat of ice melting [cm^2/s^2]

# Threshold constants
SMALL_PC = DTYPE(1e-50)  # Minimum particle concentration [#/cm^3/z]
FEW_PC = SMALL_PC * DTYPE(1e6)  # "Few particles" threshold [#/cm^3/z]
FIX_COREF = DTYPE(0.1)  # Core mass fraction for smallconc fixes
CLDFRC_MIN = DTYPE(1e-4)  # Minimum cloud fraction
CLDFRC_INCLOUD = DTYPE(0.10)  # In-cloud cloud fraction threshold

# Integer parameters
IT = 1  # Dimension parameter
CARMA_NAME_LEN = 255
CARMA_SHORT_NAME_LEN = 6
CAM_FILL = -999
