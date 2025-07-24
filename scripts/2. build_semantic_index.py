#!/usr/bin/env python3
"""
Script to build semantic indexes from structured transcript JSONs.

This script processes structured transcript JSONs, extracts individual paragraphs,
and builds FAISS indexes for semantic search.

Usage:
    # Build index for SP500, processing max 10 files
    python scripts/build_semantic_index.py SP500 --max-files 10
    
    # Build index for STOXX600, processing all files
    python scripts/build_semantic_index.py STOXX600
    
    # Build separate indexes for both SP500 and STOXX600
    python scripts/build_semantic_index.py --all --max-files 5
    
    # Use custom model
    python scripts/build_semantic_index.py SP500 --model all-mpnet-base-v2 --max-files 20

Author: Marleen de Jonge
Date: 2025
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.config import (
    get_final_output_folder, 
    SUPPORTED_INDICES,
    OUTPUTS_DIR,
    LOGS_DIR
)
from src.analysis.semantic_indexer import SemanticIndexBuilder, build_separate_indexes
from src.analysis.paragraph_extractor import estimate_paragraph_counts, preview_paragraphs


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    # Create logs directory
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOGS_DIR / 'semantic_indexing.log', mode='a')
        ]
    )


def preview_data(input_folder: Path, stock_index: str):
    """Preview the data before processing."""
    print(f"\n📊 Previewing data for {stock_index}")
    print(f"Input folder: {input_folder}")
    
    if not input_folder.exists():
        print(f"❌ Input folder does not exist: {input_folder}")
        return
    
    # Get estimates
    estimates = estimate_paragraph_counts(input_folder)
    print(f"\n📈 Paragraph estimates:")
    print(f"  Files found: {estimates['total_files']}")
    print(f"  Estimated paragraphs: {estimates['estimated_total_paragraphs']:,}")
    print(f"  Avg paragraphs per file: {estimates['avg_paragraphs_per_file']}")
    
    # Show examples
    examples = preview_paragraphs(input_folder, 3)
    if examples:
        print(f"\n📝 Example paragraphs:")
        for i, ex in enumerate(examples, 1):
            print(f"  {i}. {ex['company_name']} ({ex['section']})")
            print(f"     Speaker: {ex['speaker']} - {ex['profession']}")
            print(f"     Text: {ex['text'][:100]}...")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Build semantic indexes from structured transcript JSONs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Build index for SP500, processing max 10 files
    python scripts/build_semantic_index.py SP500 --max-files 10
    
    # Build index for STOXX600, processing all files
    python scripts/build_semantic_index.py STOXX600
    
    # Build separate indexes for both indices, max 5 files each
    python scripts/build_semantic_index.py --all --max-files 5
    
    # Preview data without building index
    python scripts/build_semantic_index.py SP500 --preview-only
    
    # Use custom model and batch size
    python scripts/build_semantic_index.py SP500 --model all-mpnet-base-v2 --batch-size 32
        """
    )
    
    parser.add_argument(
        'stock_index',
        nargs='?',
        choices=SUPPORTED_INDICES,
        help='Stock index to process (SP500 or STOXX600)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all available stock indices (SP500 and STOXX600)'
    )
    
    parser.add_argument(
        '--max-files',
        type=int,
        help='Maximum number of JSON files to process per index (default: all files)'
    )
    
    parser.add_argument(
        '--model',
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='Sentence transformer model to use (default: all-MiniLM-L6-v2)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for encoding (default: 64)'
    )
    
    parser.add_argument(
        '--output-folder',
        type=Path,
        help='Custom output folder (default: data/semantic_indexes)'
    )
    
    parser.add_argument(
        '--preview-only',
        action='store_true',
        help='Only preview the data, do not build indexes'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.all and not args.stock_index:
        parser.error("Must specify either a stock index or --all")
    
    if args.all and args.stock_index:
        parser.error("Cannot specify both --all and a specific stock index")
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Set up paths
    base_input_folder = OUTPUTS_DIR / "processed_transcripts"
    base_output_folder = args.output_folder or Path("data/semantic_indexes")
    
    logger.info(f"Starting semantic index building")
    logger.info(f"Input folder: {base_input_folder}")
    logger.info(f"Output folder: {base_output_folder}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Max files per index: {args.max_files or 'All files'}")
    
    try:
        if args.all:
            # Process all indices
            stock_indices = SUPPORTED_INDICES
            
            # Preview data for all indices
            for stock_index in stock_indices:
                input_folder = base_input_folder / stock_index
                preview_data(input_folder, stock_index)
            
            if args.preview_only:
                print("✅ Preview completed. Use without --preview-only to build indexes.")
                return
            
            # Build indexes
            print("🚀 Building separate indexes for all stock indices...")
            builder = SemanticIndexBuilder(args.model, args.batch_size)
            
            results = build_separate_indexes(
                base_input_folder=base_input_folder,
                base_output_folder=base_output_folder,
                stock_indices=stock_indices,
                max_files_per_index=args.max_files
            )
            
            # Print results summary
            print("\n📊 Build Results Summary:")
            for stock_index, result in results.items():
                if 'error' in result:
                    print(f"  ❌ {stock_index}: {result['error']}")
                else:
                    print(f"  ✅ {stock_index}: {result['num_paragraphs']:,} paragraphs indexed")
                    print(f"     Files processed: {result['files_processed']}")
                    print(f"     Index saved to: {Path(result['index_path']).parent}")
        
        else:
            # Process single index
            stock_index = args.stock_index
            input_folder = base_input_folder / stock_index
            output_folder = base_output_folder / stock_index
            
            # Preview data
            preview_data(input_folder, stock_index)
            
            if args.preview_only:
                print("✅ Preview completed. Use without --preview-only to build index.")
                return
            
            if not input_folder.exists():
                print(f"❌ Input folder does not exist: {input_folder}")
                return
            
            # Build index
            print(f"🚀 Building semantic index for {stock_index}...")
            builder = SemanticIndexBuilder(args.model, args.batch_size)
            
            result = builder.build_index_from_folder(
                structured_json_folder=input_folder,
                output_folder=output_folder,
                stock_index=stock_index,
                max_files=args.max_files
            )
            
            # Print results
            print("\n📊 Build Results:")
            print(f"  ✅ {stock_index}: {result['num_paragraphs']:,} paragraphs indexed")
            print(f"  📁 Files processed: {result['files_processed']}")
            print(f"  💾 Index saved to: {output_folder}")
            print(f"  🔍 Index name: {result['index_name']}")
    
    except KeyboardInterrupt:
        print("\n🛑 Process interrupted by user")
        return
    except Exception as e:
        logger.error(f"❌ Error during processing: {e}")
        print(f"❌ Error: {e}")
        return
    
    print("\n✅ Semantic indexing completed!")


if __name__ == "__main__":
    main()