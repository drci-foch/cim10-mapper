
"""
Foch CIM-10 Mapper - A Python package for mapping French medical text to CIM-10 codes
"""

__version__ = "0.1.0"
__author__ = "Foch Hospital Team"
__email__ = "contact@foch.fr"

from .core.mapper import FochCIM10Mapper
from .core.negation import NegationDetector
from .exceptions import FochCIM10Error, ModelNotFoundError, DataNotFoundError

__all__ = [
    "FochCIM10Mapper", 
    "NegationDetector",
    "FochCIM10Error", 
    "ModelNotFoundError", 
    "DataNotFoundError",
    "__version__"
]