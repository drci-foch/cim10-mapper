
"""
Negation detection for Foch CIM-10 Mapper using EDS-NLP
"""

import edsnlp
import edsnlp.pipes as eds
from typing import List, Dict, Any, Optional
import logging

from ..exceptions import NegationError

logger = logging.getLogger(__name__)

class NegationDetector:
    """
    Handles negation detection for medical entities using EDS-NLP
    """
    
    def __init__(self):
        """Initialize the negation detection pipeline"""
        try:
            self.nlp = self._setup_pipeline()
            logger.info("✅ Negation detector initialized")
        except Exception as e:
            raise NegationError(f"Failed to initialize negation detector: {str(e)}")
    
    def _setup_pipeline(self):
        """Setup EDS-NLP pipeline with negation detection"""
        nlp = edsnlp.blank("eds")
        nlp.add_pipe(eds.sentences())
        nlp.add_pipe(eds.negation())
        return nlp
    
    def check_entities_negation(self, text: str, entities: List[str]) -> List[Dict[str, Any]]:
        """
        Check if extracted entities are negated in the given text
        
        Args:
            text: Input medical text
            entities: List of entity texts to check
            
        Returns:
            List of dictionaries with negation status for each entity
        """
        try:
            # Process text with EDS-NLP
            doc = self.nlp(text)
            
            entity_results = []
            
            for entity_text in entities:
                # Find entity position in text
                start_idx = text.lower().find(entity_text.lower())
                if start_idx == -1:
                    # Entity not found, skip
                    entity_results.append({
                        'entity': entity_text,
                        'is_negated': False,
                        'negated_tokens': [],
                        'entity_tokens': [],
                        'context': '',
                        'found_in_text': False
                    })
                    continue
                
                end_idx = start_idx + len(entity_text)
                
                # Find tokens that overlap with this entity
                entity_tokens = []
                negated_tokens = []
                
                for token in doc:
                    token_start = token.idx
                    token_end = token.idx + len(token.text)
                    
                    # Check if token overlaps with entity span
                    if token_start < end_idx and token_end > start_idx:
                        entity_tokens.append(token)
                        
                        # Check if this token is negated
                        if hasattr(token._, 'negation') and token._.negation:
                            negated_tokens.append(token.text)
                
                # Determine if entity is negated
                is_negated = len(negated_tokens) > 0
                
                # Extract context around entity
                context_start = max(0, start_idx - 50)
                context_end = min(len(text), end_idx + 50)
                context = text[context_start:context_end].strip()
                
                entity_results.append({
                    'entity': entity_text,
                    'start_char': start_idx,
                    'end_char': end_idx,
                    'is_negated': is_negated,
                    'negated_tokens': negated_tokens,
                    'entity_tokens': [t.text for t in entity_tokens],
                    'context': context,
                    'found_in_text': True
                })
            
            return entity_results
            
        except Exception as e:
            logger.error(f"Error in negation detection: {str(e)}")
            # Return default results if negation detection fails
            return [{
                'entity': entity,
                'is_negated': False,
                'negated_tokens': [],
                'entity_tokens': [],
                'context': '',
                'error': str(e)
            } for entity in entities]
    
    def analyze_text_negation(self, text: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Perform comprehensive negation analysis on a text
        
        Args:
            text: Input text
            verbose: Whether to include detailed token information
            
        Returns:
            Complete negation analysis
        """
        try:
            doc = self.nlp(text)
            
            # Find all negated tokens
            negated_tokens = []
            all_tokens = []
            
            for token in doc:
                token_info = {
                    'text': token.text,
                    'position': token.idx,
                    'index': token.i,
                    'is_negated': hasattr(token._, 'negation') and token._.negation
                }
                
                all_tokens.append(token_info)
                
                if token_info['is_negated']:
                    negated_tokens.append(token_info)
            
            result = {
                'text': text,
                'total_tokens': len(all_tokens),
                'negated_tokens_count': len(negated_tokens),
                'negated_tokens': negated_tokens
            }
            
            if verbose:
                result['all_tokens'] = all_tokens
            
            return result
            
        except Exception as e:
            raise NegationError(f"Error in text negation analysis: {str(e)}")
    
    def print_negation_analysis(self, results: List[Dict[str, Any]]) -> None:
        """
        Print negation analysis results in a readable format
        
        Args:
            results: Results from check_entities_negation
        """
        print("\n🔍 NEGATION ANALYSIS")
        print("=" * 60)
        
        for result in results:
            if not result.get('found_in_text', True):
                print(f"❓ Entity '{result['entity']}' not found in text")
                continue
            
            status = "❌ NEGATED" if result['is_negated'] else "✅ AFFIRMED"
            print(f"{status} | {result['entity']}")
            
            if result['entity_tokens']:
                print(f"   Entity tokens: {result['entity_tokens']}")
            
            if result['negated_tokens']:
                print(f"   Negated tokens: {result['negated_tokens']}")
            
            if result['context']:
                print(f"   Context: ...{result['context']}...")
            
            if 'error' in result:
                print(f"   ⚠️  Error: {result['error']}")
            
            print()

