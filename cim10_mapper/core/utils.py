
"""
Utility functions for Foch CIM-10 Mapper
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Optional
import logging

from ..config import CIM10_COLUMNS, DEFAULT_CONFIG
from ..exceptions import DataNotFoundError

logger = logging.getLogger(__name__)

def detect_encoding(file_path: str) -> str:
    """
    Detect the encoding of a CSV file
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Detected encoding
    """
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            pd.read_csv(file_path, sep=';', header=None, nrows=3, encoding=encoding, dtype=str)
            logger.info(f"Detected encoding: {encoding}")
            return encoding
        except:
            continue
    
    logger.warning("Could not detect encoding, defaulting to utf-8")
    return 'utf-8'

def load_cim10_data(file_path: str, encoding: Optional[str] = None) -> pd.DataFrame:
    """
    Load and preprocess CIM-10 data from CSV file
    
    Args:
        file_path: Path to the CIM-10 CSV file
        encoding: File encoding (auto-detected if None)
        
    Returns:
        Processed DataFrame with code_cim10 and libelle columns
        
    Raises:
        DataNotFoundError: If file cannot be found or loaded
    """
    if not os.path.exists(file_path):
        raise DataNotFoundError(f"CIM-10 data file not found: {file_path}")
    
    try:
        # Auto-detect encoding if not provided
        if encoding is None:
            encoding = detect_encoding(file_path)
        
        # Load the data
        df = pd.read_csv(file_path, sep=';', header=None, dtype=str, encoding=encoding)
        logger.info(f"Loaded CIM-10 data: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Assign column names
        df.columns = [f'col{i}' for i in range(df.shape[1])]
        
        # Extract CIM-10 code (column 8, but 0-indexed as column 7)
        df['code_cim10'] = df.iloc[:, CIM10_COLUMNS["code_column"]].astype(str).str.strip()
        
        # Concatenate label columns
        label_parts = []
        for i in range(len(df)):
            parts = []
            
            # Add synonyms/alternatives first if available
            if CIM10_COLUMNS["label_column_3"] < df.shape[1]:
                col3_val = str(df.iloc[i, CIM10_COLUMNS["label_column_3"]]).strip()
                if col3_val and col3_val != 'nan':
                    parts.append(col3_val)
            
            # Add main labels
            col1_val = str(df.iloc[i, CIM10_COLUMNS["label_column_1"]]).strip()
            if col1_val and col1_val != 'nan':
                parts.append(col1_val)
                
            col2_val = str(df.iloc[i, CIM10_COLUMNS["label_column_2"]]).strip()
            if col2_val and col2_val != 'nan':
                parts.append(col2_val)
            
            label_parts.append(' - '.join(parts) if parts else '')
        
        df['libelle'] = label_parts
        
        # Clean the data
        original_length = len(df)
        df = df[(df['code_cim10'] != '') & (df['libelle'] != '')]
        df = df[df['libelle'].str.len() > 2]
        df = df.reset_index(drop=True)
        
        cleaned_length = len(df)
        logger.info(f"Cleaned data: {cleaned_length} valid entries (removed {original_length - cleaned_length})")
        
        return df[['code_cim10', 'libelle']]
        
    except Exception as e:
        raise DataNotFoundError(f"Error loading CIM-10 data: {str(e)}")

def create_embeddings_filename(model_name: str, device_type: str) -> str:
    """
    Create a standardized filename for embeddings cache
    
    Args:
        model_name: Name of the embedding model
        device_type: Type of device ('cpu' or 'gpu')
        
    Returns:
        Filename for embeddings cache
    """
    safe_model_name = model_name.replace('/', '_').replace('-', '_')
    return f"cim10_embeddings_{safe_model_name}_{device_type}.pkl"

def setup_cache_directory(cache_dir: Optional[str] = None) -> Path:
    """
    Setup cache directory for embeddings and models
    
    Args:
        cache_dir: Custom cache directory path
        
    Returns:
        Path to cache directory
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CONFIG["cache_dir"]
    
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    return cache_path

def validate_similarity_threshold(threshold: float) -> float:
    """
    Validate and normalize similarity threshold
    
    Args:
        threshold: Similarity threshold value
        
    Returns:
        Validated threshold
        
    Raises:
        ValueError: If threshold is not in valid range
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Similarity threshold must be between 0.0 and 1.0, got {threshold}")
    
    return float(threshold)

def format_similarity_score(score: float) -> str:
    """
    Format similarity score for display
    
    Args:
        score: Similarity score
        
    Returns:
        Formatted score with emoji
    """
    if score > 0.8:
        return f"🟢 {score:.3f}"
    elif score > 0.6:
        return f"🟡 {score:.3f}"
    else:
        return f"🟠 {score:.3f}"