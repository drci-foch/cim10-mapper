
"""
Configuration settings for Foch CIM-10 Mapper
"""

import os
from pathlib import Path
from typing import Dict, Any

# Default configuration
DEFAULT_CONFIG = {
    "embedding_model": "FremyCompany/BioLORD-2023-M",
    "ner_model": "almanach/camembert-bio-gliner-v0.1",
    "use_gpu": True,
    "gpu_batch_size": 64,
    "cpu_batch_size": 16,
    "similarity_threshold": 0.5,
    "ner_threshold": 0.3,
    "top_k": 3,
    "cache_dir": str(Path.home() / ".cache" / "foch_cim10_mapper"),
    "data_encoding": "utf-8",
    "csv_separator": ";",
}

# CIM-10 CSV column mappings
CIM10_COLUMNS = {
    "code_column": 7,      # Column index for CIM-10 code
    "label_column_1": 9,   # First label column
    "label_column_2": 10,  # Second label column  
    "label_column_3": 21,  # Third label column (synonyms/alternatives)
}

# Negation detection labels
NEGATION_LABELS = ["Maladie"]

def get_config() -> Dict[str, Any]:
    """Get the current configuration"""
    return DEFAULT_CONFIG.copy()

def update_config(**kwargs) -> None:
    """Update configuration with new values"""
    DEFAULT_CONFIG.update(kwargs)
