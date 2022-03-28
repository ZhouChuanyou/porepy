""" Composite submodule in PorePy. Contains classes representing components and phases.
Define compositional flow models using available substances.
"""

""" List of components to be included for in the testing framework. 
Include the name of your Component child class here to have it tested by the porepy unit tests and other.
"""
_COMPONENT_TEST_ARRAY = [ "SimpleFLuid",
                          "UnitSolid"
                          ]

from .compositional_domain import CompositionalDomain

from .phase import PhaseField, PhysicalState
from .substance import Substance, FluidSubstance, SolidSubstance

from ._composite_utils import (
    STATES_OF_MATTER,
    COMPUTATIONAL_VARIABLES,
    IDEAL_GAS_CONSTANT,
    create_merged_variable,
    create_merged_mortar_variable
    )

from .unit_substance import SimpleFluid, UnitSolid

from .material_subdomain import MaterialSubdomain


#------------------------------------------------------------------------------
### IMPORT concrete Component children below
#------------------------------------------------------------------------------