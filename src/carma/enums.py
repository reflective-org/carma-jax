"""Enumeration flags for CARMA-JAX physics modes.

Ported from: carma_enums_mod.F90
"""

from enum import IntEnum

# Return codes
RC_OK = 0
RC_ERROR = -1
RC_WARNING = 1
RC_WARNING_RETRY = 2


class BoundaryCondition(IntEnum):
    """Vertical transport boundary condition flags."""
    I_FIXED_CONC = 1
    I_FLUX_SPEC = 2


class ElementType(IntEnum):
    """Particle element type (itype)."""
    I_INVOLATILE = 1
    I_VOLATILE = 2
    I_COREMASS = 3
    I_VOLCORE = 4
    I_CORE2MOM = 5


class NucProcess(IntEnum):
    """Nucleation process flags (bit mask values for inucproc)."""
    I_AF_TABAZADEH_2000 = 1
    I_AF_KOOP_2000 = 2
    I_AF_MOHLER_2010 = 4
    I_AF_MURRAY_2010 = 8
    I_DROPACT = 256
    I_AERFREEZE = 512
    I_DROPFREEZE = 1024
    I_ICEMELT = 2048
    I_HETNUC = 4096
    I_HOMNUC = 8192
    I_HETNUCSULF = 16384


class CollectionProcess(IntEnum):
    """Collection efficiency method."""
    I_COLLEC_CONST = 1
    I_COLLEC_FUCHS = 2
    I_COLLEC_DATA = 3


class CoagOperation(IntEnum):
    """Coagulation kernel operation mode."""
    I_COAGOP_CONST = 1
    I_COAGOP_CALC = 2


class Shape(IntEnum):
    """Particle shape."""
    I_SPHERE = 1
    I_HEXAGON = 2
    I_CYLINDER = 3


class SwellMethod(IntEnum):
    """Particle swelling parameterization."""
    I_NO_SWELLING = 0
    I_FITZGERALD = 1
    I_GERBER = 2
    I_WTPCT_H2SO4 = 3
    I_PETTERS = 4


class SwellFitzgerald(IntEnum):
    """Swelling composition for Fitzgerald method."""
    I_SWF_NH42SO4 = 1
    I_SWF_NH4NO3 = 2
    I_SWF_NANO3 = 3
    I_SWF_NH4CL = 4
    I_SWF_CACL2 = 5
    I_SWF_NABR = 6
    I_SWF_NACL = 7
    I_SWF_MGCL2 = 8
    I_SWF_LICL = 9


class SwellGerber(IntEnum):
    """Swelling composition for Gerber method."""
    I_SWG_NH42SO4 = 11
    I_SWG_SEA_SALT = 12
    I_SWG_URBAN = 13
    I_SWG_RURAL = 14


class VaporPressureRoutine(IntEnum):
    """Vapor pressure calculation method."""
    I_VAPRTN_H2O_BUCK1981 = 1
    I_VAPRTN_H2O_MURPHY2005 = 2
    I_VAPRTN_H2O_GOFF1946 = 3
    I_VAPRTN_H2SO4_AYERS1980 = 4


class FallRoutine(IntEnum):
    """Fall velocity calculation method."""
    I_FALLRTN_STD = 1
    I_FALLRTN_STD_SHAPE = 2
    I_FALLRTN_HEYMSFIELD2010 = 3


class MieRoutine(IntEnum):
    """Mie optical properties calculation method."""
    I_MIERTN_TOON1981 = 1
    I_MIERTN_BOHREN1983 = 2
    I_MIERTN_BOTET1997 = 3


class GasComposition(IntEnum):
    """Gas species composition identifier."""
    I_GCOMP_H2O = 1
    I_GCOMP_H2SO4 = 2
    I_GCOMP_SO2 = 3


class ConstituentType(IntEnum):
    """Model constituent type."""
    I_CNSTTYPE_PROGNOSTIC = 1
    I_CNSTTYPE_DIAGNOSTIC = 2


class OpticsType(IntEnum):
    """Optical properties calculation method."""
    I_OPTICS_FIXED = 1
    I_OPTICS_MIXED_YU2015 = 2
    I_OPTICS_SULFATE_YU2015 = 3
    I_OPTICS_MIXED_YU_H2O = 4
    I_OPTICS_MIXED_CORESHELL = 5
    I_OPTICS_MIXED_VOLUME = 6
    I_OPTICS_MIXED_MAXWELL = 7
    I_OPTICS_SULFATE = 8


class GridType(IntEnum):
    """Coordinate grid type."""
    I_CART = 1
    I_SIG = 2
    I_LL = 3
    I_LC = 4
    I_PS = 5
    I_ME = 6
    I_HYBRID = 7
