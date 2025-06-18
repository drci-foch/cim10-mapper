"""
Main mapper class for Foch CIM-10 mapping
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from gliner import GLiNER
import pickle
import os
import torch
import time
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

from .utils import load_cim10_data, create_embeddings_filename, setup_cache_directory, validate_similarity_threshold
from .negation import NegationDetector
from ..config import DEFAULT_CONFIG, NEGATION_LABELS
from ..exceptions import ModelNotFoundError, EmbeddingError, DataNotFoundError

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class FochCIM10Mapper:
    """
    Main class for mapping French medical text to CIM-10 codes
    
    This class combines Named Entity Recognition (NER) with semantic similarity
    to map free text medical descriptions to standardized CIM-10 codes.
    """
    
    def __init__(
        self,
        cim10_file_path: str,
        embedding_model: str = None,
        ner_model: str = None,
        use_gpu: bool = None,
        gpu_batch_size: int = None,
        cache_dir: str = None,
        **kwargs
    ):
        """
        Initialize the Foch CIM-10 Mapper
        
        Args:
            cim10_file_path: Path to the CIM-10 CSV file
            embedding_model: Name of the embedding model to use
            ner_model: Name of the NER model to use
            use_gpu: Whether to use GPU if available
            gpu_batch_size: Batch size for GPU processing
            cache_dir: Directory for caching embeddings and models
            **kwargs: Additional configuration options
        """
        # Load configuration
        self.config = DEFAULT_CONFIG.copy()
        self.config.update(kwargs)
        
        # Override with provided parameters
        if embedding_model is not None:
            self.config["embedding_model"] = embedding_model
        if ner_model is not None:
            self.config["ner_model"] = ner_model
        if use_gpu is not None:
            self.config["use_gpu"] = use_gpu
        if gpu_batch_size is not None:
            self.config["gpu_batch_size"] = gpu_batch_size
        if cache_dir is not None:
            self.config["cache_dir"] = cache_dir
            
        self.cim10_file_path = cim10_file_path
        
        logger.info("🚀 Initializing Foch CIM-10 Mapper")
        logger.info("=" * 60)
        
        # Setup cache directory
        self.cache_dir = setup_cache_directory(self.config["cache_dir"])
        
        # Setup GPU/CPU
        self._setup_device()
        
        # Load embedding model
        self.sentence_model = self._load_embedding_model()
        
        # Load CIM-10 data
        self.cim10_df = load_cim10_data(self.cim10_file_path)
        
        # Load or create embeddings
        self.cim10_embeddings = self._load_or_create_embeddings()
        
        # Load NER model
        self.ner_model = self._load_ner_model()
        
        # Initialize negation detector
        self.negation_detector = NegationDetector()
        
        logger.info("✅ Foch CIM-10 Mapper initialized successfully!")
        logger.info("=" * 60)
    
    def _setup_device(self) -> None:
        """Setup GPU/CPU device configuration"""
        if self.config["use_gpu"] and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"🚀 GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # Enable optimizations
            if hasattr(torch.backends.cudnn, 'benchmark'):
                torch.backends.cudnn.benchmark = True
                
        else:
            self.device = torch.device("cpu")
            if self.config["use_gpu"]:
                logger.warning("⚠️  GPU requested but not available, using CPU")
            else:
                logger.info("💻 Using CPU")
            
            # Adjust batch size for CPU
            self.config["gpu_batch_size"] = min(self.config["gpu_batch_size"], self.config["cpu_batch_size"])
    
    def _load_embedding_model(self) -> SentenceTransformer:
        """Load the sentence transformer model"""
        logger.info(f"📦 Loading embedding model: {self.config['embedding_model']}")
        
        try:
            model = SentenceTransformer(self.config["embedding_model"], device=self.device)
            
            # Enable half precision for GPU if supported
            if self.device.type == "cuda":
                try:
                    model = model.half()
                    logger.info("✨ Half precision (FP16) enabled")
                except:
                    logger.warning("⚠️  Half precision not supported, using FP32")
            
            logger.info(f"✅ Embedding model loaded on: {self.device}")
            return model
            
        except Exception as e:
            raise ModelNotFoundError(f"Failed to load embedding model: {str(e)}")
    
    def _load_ner_model(self) -> GLiNER:
        """Load the NER model"""
        logger.info(f"📦 Loading NER model: {self.config['ner_model']}")
        
        try:
            model = GLiNER.from_pretrained(self.config["ner_model"])
            logger.info("✅ NER model loaded successfully")
            return model
            
        except Exception as e:
            raise ModelNotFoundError(f"Failed to load NER model: {str(e)}")
    
    def _load_or_create_embeddings(self) -> np.ndarray:
        """Load existing embeddings or create new ones"""
        device_suffix = "gpu" if self.device.type == "cuda" else "cpu"
        embeddings_filename = create_embeddings_filename(
            self.config["embedding_model"], 
            device_suffix
        )
        embeddings_path = self.cache_dir / embeddings_filename
        
        if embeddings_path.exists():
            logger.info(f"📂 Loading cached embeddings: {embeddings_filename}")
            try:
                with open(embeddings_path, 'rb') as f:
                    embeddings = pickle.load(f)
                logger.info(f"✅ Embeddings loaded: {embeddings.shape}")
                return embeddings
            except Exception as e:
                logger.warning(f"⚠️  Failed to load cached embeddings: {e}")
                logger.info("🔄 Creating new embeddings...")
                
        return self._create_embeddings(embeddings_path)
    
    def _create_embeddings(self, save_path: Path) -> np.ndarray:
        """Create embeddings for all CIM-10 labels"""
        libelles = self.cim10_df['libelle'].tolist()
        total_items = len(libelles)
        
        logger.info(f"🔄 Creating embeddings for {total_items} labels...")
        logger.info(f"📊 Device: {self.device}, Batch size: {self.config['gpu_batch_size']}")
        
        # Optimize batch size based on GPU memory
        batch_size = self.config["gpu_batch_size"]
        if self.device.type == "cuda":
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory_gb < 8:
                batch_size = min(batch_size, 32)
            elif gpu_memory_gb >= 16:
                batch_size = min(batch_size, 128)
        
        embeddings = []
        start_time = time.time()
        
        try:
            for i in range(0, total_items, batch_size):
                batch = libelles[i:i+batch_size]
                batch_end = min(i + batch_size, total_items)
                
                # Clear GPU cache between batches
                if self.device.type == "cuda" and i > 0:
                    torch.cuda.empty_cache()
                
                # Create embeddings
                batch_embeddings = self.sentence_model.encode(
                    batch,
                    show_progress_bar=False,
                    convert_to_tensor=False,
                    normalize_embeddings=True,
                    batch_size=min(batch_size, len(batch))
                )
                embeddings.extend(batch_embeddings)
                
                # Progress reporting
                if i % (batch_size * 5) == 0 or batch_end == total_items:
                    progress = (batch_end / total_items) * 100
                    elapsed = time.time() - start_time
                    logger.info(f"   Progress: {progress:.1f}% ({batch_end}/{total_items}) - {elapsed:.1f}s")
            
            embeddings = np.array(embeddings)
            
            # Save embeddings
            logger.info(f"💾 Saving embeddings: {save_path.name}")
            with open(save_path, 'wb') as f:
                pickle.dump(embeddings, f)
            
            total_time = time.time() - start_time
            logger.info(f"✅ Embeddings created in {total_time:.1f}s: {embeddings.shape}")
            
            return embeddings
            
        except Exception as e:
            raise EmbeddingError(f"Failed to create embeddings: {str(e)}")
        finally:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
    
    def extract_entities(self, text: str, threshold: float = None) -> List[str]:
        """
        Extract medical entities from text using NER
        
        Args:
            text: Input text
            threshold: NER confidence threshold
            
        Returns:
            List of extracted entity texts
        """
        if threshold is None:
            threshold = self.config["ner_threshold"]
        
        entities = self.ner_model.predict_entities(text, NEGATION_LABELS, threshold=threshold)
        return [entity["text"] for entity in entities]
    
    def find_similar_cim10(
        self, 
        entity_text: str, 
        top_k: int = None, 
        similarity_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        Find the most similar CIM-10 codes for an entity
        
        Args:
            entity_text: Medical entity text
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of similar CIM-10 codes with metadata
        """
        if top_k is None:
            top_k = self.config["top_k"]
        if similarity_threshold is not None:
            similarity_threshold = validate_similarity_threshold(similarity_threshold)
        
        # Create embedding for entity
        entity_embedding = self.sentence_model.encode(
            [entity_text],
            convert_to_tensor=False,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # Calculate similarities
        similarities = cosine_similarity(entity_embedding, self.cim10_embeddings)[0]
        
        # Get top results
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        
        results = []
        for i, idx in enumerate(top_indices):
            score = similarities[idx]
            
            # Apply threshold if specified
            if similarity_threshold is not None and score < similarity_threshold:
                break
            
            results.append({
                'code_cim10': self.cim10_df.iloc[idx]['code_cim10'],
                'libelle': self.cim10_df.iloc[idx]['libelle'],
                'similarity_score': float(score),
                'rank': i + 1
            })
        
        return results
    
    def process_text(
        self,
        text: str,
        ner_threshold: float = None,
        top_k: int = None,
        similarity_threshold: float = None,
        include_negation: bool = True
    ) -> Dict[str, Any]:
        """
        Process a complete text: NER extraction + CIM-10 mapping + negation detection
        
        Args:
            text: Input medical text
            ner_threshold: NER confidence threshold
            top_k: Number of top CIM-10 matches per entity
            similarity_threshold: Minimum similarity score
            include_negation: Whether to include negation detection
            
        Returns:
            Complete analysis results
        """
        # Extract entities
        entities = self.extract_entities(text, ner_threshold)
        
        # Detect negation if requested
        negation_results = None
        if include_negation and entities:
            negation_results = self.negation_detector.check_entities_negation(text, entities)
        
        results = {
            'text': text,
            'entities_found': len(entities),
            'mappings': [],
            'negation_analysis': negation_results
        }
        
        # Map each entity to CIM-10
        for entity in entities:
            similar_codes = self.find_similar_cim10(
                entity,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
            
            # Add negation status if available
            negation_status = None
            if negation_results:
                for neg_result in negation_results:
                    if neg_result['entity'] == entity:
                        negation_status = {
                            'is_negated': neg_result['is_negated'],
                            'negated_tokens': neg_result.get('negated_tokens', [])
                        }
                        break
            
            results['mappings'].append({
                'entity': entity,
                'cim10_matches': similar_codes,
                'negation': negation_status
            })
        
        return results
    
    def print_results(self, results: Dict[str, Any], show_negation: bool = True) -> None:
        """
        Print analysis results in a readable format
        
        Args:
            results: Results from process_text
            show_negation: Whether to show negation information
        """
        print(f"\n🔍 FOCH CIM-10 ANALYSIS")
        print("=" * 60)
        print(f"Text: {results['text'][:100]}{'...' if len(results['text']) > 100 else ''}")
        print(f"Entities found: {results['entities_found']}")
        
        if results['entities_found'] == 0:
            print("❌ No medical entities detected")
            return
        
        print(f"\n📋 CIM-10 MAPPINGS")
        print("=" * 60)
        
        for i, mapping in enumerate(results['mappings']):
            # Entity header with negation status
            entity_status = ""
            if show_negation and mapping.get('negation'):
                if mapping['negation']['is_negated']:
                    entity_status = " ❌ [NEGATED]"
                else:
                    entity_status = " ✅ [AFFIRMED]"
            
            print(f"\n🎯 Entity #{i+1}: '{mapping['entity']}'{entity_status}")
            
            if not mapping['cim10_matches']:
                print("   ❌ No matches found with specified threshold")
                continue
            
            for match in mapping['cim10_matches']:
                score_emoji = "🟢" if match['similarity_score'] > 0.8 else "🟡" if match['similarity_score'] > 0.6 else "🟠"
                print(f"   {score_emoji} {match['rank']}. {match['code_cim10']} - {match['libelle']}")
                print(f"      Similarity: {match['similarity_score']:.3f}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and statistics"""
        info = {
            'device': str(self.device),
            'embedding_model': self.config['embedding_model'],
            'ner_model': self.config['ner_model'],
            'cim10_entries': len(self.cim10_df),
            'embeddings_shape': self.cim10_embeddings.shape,
            'cache_dir': str(self.cache_dir),
            'configuration': self.config
        }
        
        if self.device.type == "cuda":
            info['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1e9
            info['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1e9
        
        return info
    
    def print_system_info(self) -> None:
        """Print system information"""
        info = self.get_system_info()
        
        print(f"\n📊 SYSTEM INFORMATION")
        print("=" * 60)
        print(f"Device: {info['device']}")
        print(f"Embedding model: {info['embedding_model']}")
        print(f"NER model: {info['ner_model']}")
        print(f"CIM-10 entries: {info['cim10_entries']}")
        print(f"Embeddings shape: {info['embeddings_shape']}")
        print(f"Cache directory: {info['cache_dir']}")
        
        if 'gpu_memory_allocated' in info:
            print(f"GPU memory: {info['gpu_memory_allocated']:.2f}GB allocated / {info['gpu_memory_reserved']:.2f}GB reserved")
