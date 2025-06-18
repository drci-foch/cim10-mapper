
"""
Tests for the main FochCIM10Mapper class
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from foch_cim10_mapper import FochCIM10Mapper, FochCIM10Error, DataNotFoundError
from foch_cim10_mapper.core.utils import load_cim10_data, create_embeddings_filename

class TestFochCIM10Mapper:
    
    @pytest.fixture
    def sample_cim10_data(self):
        """Create sample CIM-10 data for testing"""
        data = {
            'col7': ['E10', 'E11', 'J45', 'I10'],
            'col8': ['E10', 'E11', 'J45', 'I10'], 
            'col9': ['Diabète sucré, type 1', 'Diabète sucré, type 2', 'Asthme', 'Hypertension essentielle'],
            'col10': ['Avec coma', 'Sans complication', 'Sans précision', 'Sans précision'],
            'col21': ['DID', 'DNID', 'Asthme bronchique', 'HTA']
        }
        
        # Add extra columns to match expected format
        for i in range(22):
            if f'col{i}' not in data:
                data[f'col{i}'] = [''] * 4
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def temp_cim10_file(self, sample_cim10_data):
        """Create temporary CIM-10 CSV file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            sample_cim10_data.to_csv(f, sep=';', header=False, index=False)
            return f.name
    
    def test_load_cim10_data(self, temp_cim10_file):
        """Test loading CIM-10 data"""
        df = load_cim10_data(temp_cim10_file)
        
        assert len(df) > 0
        assert 'code_cim10' in df.columns
        assert 'libelle' in df.columns
        assert df['code_cim10'].notna().all()
        assert df['libelle'].notna().all()
        
        # Clean up
        os.unlink(temp_cim10_file)
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises error"""
        with pytest.raises(DataNotFoundError):
            load_cim10_data("nonexistent_file.csv")
    
    def test_create_embeddings_filename(self):
        """Test embeddings filename creation"""
        filename = create_embeddings_filename("model/name", "gpu")
        assert "model_name" in filename
        assert "gpu" in filename
        assert filename.endswith(".pkl")
    
    @pytest.mark.slow
    def test_mapper_initialization(self, temp_cim10_file):
        """Test mapper initialization with minimal models"""
        # This test may take time due to model loading
        mapper = FochCIM10Mapper(
            temp_cim10_file,
            use_gpu=False,  # Use CPU for testing
            gpu_batch_size=2  # Small batch size
        )
        
        assert mapper.cim10_df is not None
        assert len(mapper.cim10_df) > 0
        assert mapper.cim10_embeddings is not None
        assert mapper.sentence_model is not None
        assert mapper.ner_model is not None
        
        # Clean up
        os.unlink(temp_cim10_file)
    
    @pytest.mark.slow 
    def test_extract_entities(self, temp_cim10_file):
        """Test entity extraction"""
        mapper = FochCIM10Mapper(temp_cim10_file, use_gpu=False)
        
        text = "Patient avec asthme et diabète"
        entities = mapper.extract_entities(text, threshold=0.1)  # Low threshold for testing
        
        assert isinstance(entities, list)
        # Should detect at least some entities
        
        # Clean up
        os.unlink(temp_cim10_file)
    
    @pytest.mark.slow
    def test_process_text(self, temp_cim10_file):
        """Test complete text processing"""
        mapper = FochCIM10Mapper(temp_cim10_file, use_gpu=False)
        
        text = "Patient présente un diabète de type 2"
        results = mapper.process_text(text, ner_threshold=0.1, top_k=2)
        
        assert 'text' in results
        assert 'entities_found' in results
        assert 'mappings' in results
        assert isinstance(results['mappings'], list)
        
        # Clean up
        os.unlink(temp_cim10_file)
    
    def test_system_info(self, temp_cim10_file):
        """Test system info retrieval"""
        # Mock a minimal mapper for system info testing
        mapper = FochCIM10Mapper.__new__(FochCIM10Mapper)
        mapper.config = {'embedding_model': 'test', 'ner_model': 'test'}
        mapper.device = 'cpu'
        mapper.cim10_df = pd.DataFrame({'code': ['E10'], 'libelle': ['test']})
        mapper.cim10_embeddings = np.array([[1, 2, 3]])
        mapper.cache_dir = Path('/tmp')
        
        info = mapper.get_system_info()
        
        assert 'device' in info
        assert 'embedding_model' in info
        assert 'ner_model' in info
        assert 'cim10_entries' in info
        
        # Clean up
        os.unlink(temp_cim10_file)

class TestNegationDetector:
    
    def test_negation_initialization(self):
        """Test negation detector initialization"""
        from foch_cim10_mapper.core.negation import NegationDetector
        
        detector = NegationDetector()
        assert detector.nlp is not None
    
    def test_check_entities_negation(self):
        """Test negation detection"""
        from foch_cim10_mapper.core.negation import NegationDetector
        
        detector = NegationDetector()
        text = "Patient sans diabète mais avec hypertension"
        entities = ["diabète", "hypertension"]
        
        results = detector.check_entities_negation(text, entities)
        
        assert len(results) == 2
        assert all('is_negated' in result for result in results)
        assert all('entity' in result for result in results)

# =============================================================================
# tests/conftest.py
# =============================================================================

"""
Pytest configuration and fixtures
"""

import pytest

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )