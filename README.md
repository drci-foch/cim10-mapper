# Foch CIM-10 Mapper

A Python package for mapping French medical text to CIM-10 codes using Named Entity Recognition (NER) and semantic similarity.

## Features

- 🏥 **Medical NER**: Extract medical entities from French clinical text
- 🔍 **CIM-10 Mapping**: Map entities to standardized CIM-10 codes using semantic similarity
- ❌ **Negation Detection**: Identify negated medical conditions using EDS-NLP
- 🚀 **GPU Support**: Accelerated processing with GPU support
- 💾 **Caching**: Intelligent caching of embeddings and models
- 🎯 **High Accuracy**: Uses state-of-the-art French biomedical models

## Installation

### From PyPI (when published)
```bash
pip install foch-cim10-mapper
```

### From Source
```bash
git clone https://github.com/foch-hospital/foch-cim10-mapper.git
cd foch-cim10-mapper
pip install -e .
```

### With GPU Support
```bash
pip install foch-cim10-mapper[gpu]
```

## Quick Start

```python
from foch_cim10_mapper import FochCIM10Mapper

# Initialize the mapper
mapper = FochCIM10Mapper(
    cim10_file_path="path/to/cim10_data.csv",
    use_gpu=True  # Enable GPU acceleration
)

# Analyze medical text
text = """
Patient présente une hypertension artérielle et un diabète de type 2.
Pas d'antécédent de maladie cardiaque.
"""

results = mapper.process_text(text)
mapper.print_results(results)
```

## Command Line Usage

```bash
# Basic usage
foch-cim10 cim10_data.csv --text "Patient avec asthme et allergie"

# From file with GPU
foch-cim10 cim10_data.csv --input-file medical_note.txt --gpu --output-file results.json

# Advanced options
foch-cim10 cim10_data.csv \
    --text "Suspicion d'infarctus du myocarde" \
    --top-k 5 \
    --similarity-threshold 0.7 \
    --ner-threshold 0.4 \
    --gpu
```

## Configuration

### Default Models
- **Embedding Model**: `FremyCompany/BioLORD-2023-M` (French biomedical embeddings)
- **NER Model**: `almanach/camembert-bio-gliner-v0.1` (French medical NER)

### Custom Configuration
```python
mapper = FochCIM10Mapper(
    cim10_file_path="cim10_data.csv",
    embedding_model="custom/model",
    ner_model="custom/ner-model",
    use_gpu=True,
    gpu_batch_size=64,
    cache_dir="/custom/cache/path"
)
```

## API Reference

### FochCIM10Mapper

Main class for CIM-10 mapping functionality.

#### Methods

##### `__init__(cim10_file_path, **kwargs)`
Initialize the mapper with CIM-10 data.

**Parameters:**
- `cim10_file_path` (str): Path to CIM-10 CSV file
- `embedding_model` (str): Embedding model name
- `ner_model` (str): NER model name  
- `use_gpu` (bool): Enable GPU acceleration
- `gpu_batch_size` (int): Batch size for GPU processing
- `cache_dir` (str): Cache directory path

##### `process_text(text, **kwargs)`
Process text and return CIM-10 mappings with negation detection.

**Parameters:**
- `text` (str): Input medical text
- `ner_threshold` (float): NER confidence threshold (default: 0.3)
- `top_k` (int): Number of top matches (default: 3)
- `similarity_threshold` (float): Minimum similarity score
- `include_negation` (bool): Include negation detection (default: True)

**Returns:**
```python
{
    'text': str,
    'entities_found': int,
    'mappings': [
        {
            'entity': str,
            'cim10_matches': [
                {
                    'code_cim10': str,
                    'libelle': str,
                    'similarity_score': float,
                    'rank': int
                }
            ],
            'negation': {
                'is_negated': bool,
                'negated_tokens': List[str]
            }
        }
    ],
    'negation_analysis': List[Dict]
}
```

##### `extract_entities(text, threshold=0.3)`
Extract medical entities from text.

##### `find_similar_cim10(entity_text, top_k=3)`
Find similar CIM-10 codes for an entity.

##### `print_results(results)`
Print analysis results in a readable format.

### NegationDetector

Handle negation detection for medical entities.

#### Methods

##### `check_entities_negation(text, entities)`
Check if entities are negated in the given text.

##### `analyze_text_negation(text, verbose=False)`
Perform comprehensive negation analysis.

## Data Format

### CIM-10 CSV Format
The CIM-10 CSV file should have the following structure:
- Column 8: CIM-10 codes
- Column 9: Primary labels
- Column 10: Secondary labels  
- Column 21: Synonyms/alternatives (optional)

Example:
```csv
...,"E10","Diabète sucré, type 1","Avec coma","DID, Diabète insulino-dépendant",...
```

## Examples

### Basic Entity Extraction
```python
mapper = FochCIM10Mapper("cim10_data.csv")

text = "Patient avec asthme et rhinite allergique"
entities = mapper.extract_entities(text)
print(entities)  # ['asthme', 'rhinite allergique']
```

### CIM-10 Mapping
```python
results = mapper.find_similar_cim10("asthme", top_k=3)
for match in results:
    print(f"{match['code_cim10']}: {match['libelle']} ({match['similarity_score']:.3f})")
```

### Negation Detection
```python
text = "Pas d'antécédent d'infarctus du myocarde"
results = mapper.process_text(text, include_negation=True)

for mapping in results['mappings']:
    entity = mapping['entity']
    is_negated = mapping['negation']['is_negated']
    status = "NEGATED" if is_negated else "AFFIRMED"
    print(f"{entity}: {status}")
```

### Batch Processing
```python
texts = [
    "Patient avec hypertension",
    "Diabète de type 2 bien équilibré", 
    "Absence de maladie cardiaque"
]

for text in texts:
    results = mapper.process_text(text)
    print(f"\nText: {text}")
    mapper.print_results(results)
```

## Performance

### Benchmarks
- **Entity Extraction**: ~100ms per document
- **CIM-10 Mapping**: ~50ms per entity
- **GPU Acceleration**: 3-5x speedup for large batches

### Memory Usage
- **CPU**: ~2GB RAM for standard models
- **GPU**: ~4GB VRAM for standard models with FP16

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
git clone https://github.com/foch-hospital/foch-cim10-mapper.git
cd foch-cim10-mapper
pip install -e .[dev]
pytest tests/
```


## Support

- 📧 Email: s.ben-yahia@hopital-foch.com
- 🐛 Issues: [GitHub Issues](https://github.com/drci-foch/cim10-mapper/issues)

---
