
"""
Custom exceptions for Foch CIM-10 Mapper
"""

class FochCIM10Error(Exception):
    """Base exception for Foch CIM-10 Mapper"""
    pass

class ModelNotFoundError(FochCIM10Error):
    """Raised when a required model cannot be found or loaded"""
    pass

class DataNotFoundError(FochCIM10Error):
    """Raised when required data files cannot be found"""
    pass

class EmbeddingError(FochCIM10Error):
    """Raised when there's an issue with embeddings creation or loading"""
    pass

class NegationError(FochCIM10Error):
    """Raised when there's an issue with negation detection"""
    pass