
"""
Command line interface for Foch CIM-10 Mapper
"""

import argparse
import json
import sys
from pathlib import Path
import logging

from .core.mapper import FochCIM10Mapper
from .exceptions import FochCIM10Error

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Foch CIM-10 Mapper - Map French medical text to CIM-10 codes"
    )
    
    parser.add_argument(
        "cim10_file",
        help="Path to CIM-10 CSV file"
    )
    
    parser.add_argument(
        "--text",
        help="Text to analyze (alternative to --input-file)"
    )
    
    parser.add_argument(
        "--input-file",
        help="Input file containing text to analyze"
    )
    
    parser.add_argument(
        "--output-file",
        help="Output file for results (JSON format)"
    )
    
    parser.add_argument(
        "--embedding-model",
        default="FremyCompany/BioLORD-2023-M",
        help="Embedding model to use"
    )
    
    parser.add_argument(
        "--ner-model", 
        default="almanach/camembert-bio-gliner-v0.1",
        help="NER model to use"
    )
    
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU if available"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top CIM-10 matches to return"
    )
    
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        help="Minimum similarity threshold"
    )
    
    parser.add_argument(
        "--ner-threshold",
        type=float,
        default=0.3,
        help="NER confidence threshold"
    )
    
    parser.add_argument(
        "--no-negation",
        action="store_true",
        help="Disable negation detection"
    )
    
    parser.add_argument(
        "--cache-dir",
        help="Cache directory for models and embeddings"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Validate arguments
    if not args.text and not args.input_file:
        print("Error: Either --text or --input-file must be provided")
        sys.exit(1)
    
    if not Path(args.cim10_file).exists():
        print(f"Error: CIM-10 file not found: {args.cim10_file}")
        sys.exit(1)
    
    try:
        # Initialize mapper
        mapper = FochCIM10Mapper(
            cim10_file_path=args.cim10_file,
            embedding_model=args.embedding_model,
            ner_model=args.ner_model,
            use_gpu=args.gpu,
            cache_dir=args.cache_dir
        )
        
        # Get input text
        if args.text:
            text = args.text
        else:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                text = f.read()
        
        # Process text
        results = mapper.process_text(
            text,
            ner_threshold=args.ner_threshold,
            top_k=args.top_k,
            similarity_threshold=args.similarity_threshold,
            include_negation=not args.no_negation
        )
        
        # Output results
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Results saved to: {args.output_file}")
        else:
            mapper.print_results(results)
        
        if args.verbose:
            mapper.print_system_info()
            
    except FochCIM10Error as e:
        print(f"Foch CIM-10 Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()