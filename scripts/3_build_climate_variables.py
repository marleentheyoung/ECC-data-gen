#!/usr/bin/env python3
"""
Script to build semantic climate exposure variables from pre-extracted climate snippets.

This script processes climate-relevant snippets that have already been extracted
from earnings calls and constructs firm-quarter panel variables for econometric analysis.

Usage:
    # Build variables for both SP500 and STOXX600
    python scripts/3_build_climate_variables.py --all
    
    # Build variables for SP500 only
    python scripts/3_build_climate_variables.py SP500
    
    # Build with custom year range
    python scripts/3_build_climate_variables.py --all --start-year 2010 --end-year 2023
    
    # Export to Stata format
    python scripts/3_build_climate_variables.py SP500 --export-format stata

Author: Marleen de Jonge
Date: 2025
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.config import SUPPORTED_INDICES, LOGS_DIR
from src.analysis.climate_variables.climate_searcher import ClimateSemanticSearcher
from src.analysis.climate_variables.semantic_constructor import SemanticClimateVariableConstructor


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOGS_DIR / 'climate_variables.log', mode='a')
        ]
    )


def check_data_availability(base_data_path: Path, stock_indices: list, use_original: bool = False) -> dict:
    """Check if climate snippet data is available for the specified indices."""
    availability = {}
    
    for stock_index in stock_indices:
        index_path = base_data_path / stock_index
        if index_path.exists():
            if use_original:
                # Look for original climate segment files
                json_files = list(index_path.glob("climate_segments_*.json"))
            else:
                # Look for enhanced climate segment files
                json_files = list(index_path.glob("enhanced_climate_segments_*.json"))
            
            availability[stock_index] = {
                'available': len(json_files) > 0,
                'files_found': len(json_files),
                'path': str(index_path),
                'file_type': 'original' if use_original else 'enhanced'
            }
        else:
            availability[stock_index] = {
                'available': False,
                'files_found': 0,
                'path': str(index_path),
                'file_type': 'original' if use_original else 'enhanced'
            }
    
    return availability


def preview_data(searcher: ClimateSemanticSearcher, stock_indices: list):
    """Preview the climate snippet data."""
    print(f"\n📊 Data Preview")
    print("=" * 50)
    
    stats = searcher.get_statistics()
    
    print(f"Total climate snippets: {stats.get('total_snippets', 0):,}")
    print(f"Unique companies: {stats.get('unique_companies', 0)}")
    print(f"Year range: {stats.get('year_range', 'Unknown')}")
    
    # Check if sentence ratio data is available
    has_sentence_ratios = any(
        hasattr(snippet, 'climate_sentence_ratio') and snippet.climate_sentence_ratio is not None
        for snippet in searcher.snippets[:10]  # Check first 10 snippets
    )
    
    if has_sentence_ratios:
        print(f"✅ Enhanced data with sentence ratios detected")
        
        # Show sentence ratio statistics
        ratios = [
            snippet.climate_sentence_ratio for snippet in searcher.snippets
            if hasattr(snippet, 'climate_sentence_ratio') and snippet.climate_sentence_ratio is not None
        ]
        
        if ratios:
            import numpy as np
            print(f"📏 Sentence ratio statistics:")
            print(f"  Mean ratio: {np.mean(ratios):.4f}")
            print(f"  Median ratio: {np.median(ratios):.4f}")
            print(f"  Range: {np.min(ratios):.4f} - {np.max(ratios):.4f}")
    else:
        print(f"⚠️  Original data without sentence ratios")
        print(f"    Run scripts/2.5_calculate_sentence_ratios.py first for normalized attention metrics")
    
    print(f"\nTop 10 companies by snippet count:")
    top_companies = stats.get('top_companies', {})
    for i, (ticker, count) in enumerate(list(top_companies.items())[:10], 1):
        print(f"  {i:2d}. {ticker}: {count:,} snippets")
    
    print(f"\nYear distribution:")
    year_dist = stats.get('year_distribution', {})
    for year, count in sorted(year_dist.items()):
        if isinstance(year, int) and year >= 2009:  # Only show recent years
            print(f"  {year}: {count:,} snippets")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Build semantic climate exposure variables from pre-extracted snippets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Build variables for both indices
    python scripts/3_build_climate_variables.py --all
    
    # Build for SP500 only, years 2010-2023
    python scripts/3_build_climate_variables.py SP500 --start-year 2010 --end-year 2023
    
    # Preview data without building variables
    python scripts/3_build_climate_variables.py --all --preview-only
    
    # Build and export to Stata format
    python scripts/3_build_climate_variables.py STOXX600 --export-format stata
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
        help='Process all available stock indices'
    )
    
    parser.add_argument(
        '--start-year',
        type=int,
        default=2009,
        help='Start year for panel construction (default: 2009)'
    )
    
    parser.add_argument(
        '--end-year',
        type=int,
        default=2024,
        help='End year for panel construction (default: 2024)'
    )
    
    parser.add_argument(
        '--data-path',
        type=Path,
        default=Path("outputs/enhanced_climate_snippets"),
        help='Path to enhanced climate snippets data with sentence ratios (default: outputs/enhanced_climate_snippets)'
    )
    
    parser.add_argument(
        '--use-original-snippets',
        action='store_true',
        help='Use original climate snippets instead of enhanced ones (no sentence ratios available)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path("outputs/climate_variables"),
        help='Output directory for climate variables (default: outputs/climate_variables)'
    )
    
    parser.add_argument(
        '--model',
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='Sentence transformer model for semantic search (default: all-MiniLM-L6-v2)'
    )
    
    parser.add_argument(
        '--export-format',
        choices=['csv', 'stata', 'r'],
        default='csv',
        help='Export format for econometric analysis (default: csv)'
    )
    
    parser.add_argument(
        '--preview-only',
        action='store_true',
        help='Only preview the data, do not build variables'
    )
    
    parser.add_argument(
        '--save-index',
        action='store_true',
        help='Save the semantic search index for later use'
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
    
    # Determine stock indices to process
    stock_indices = SUPPORTED_INDICES if args.all else [args.stock_index]
    
    print("🌍 Climate Variables Construction")
    print("=" * 50)
    print(f"Stock indices: {', '.join(stock_indices)}")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Year range: {args.start_year}-{args.end_year}")
    print(f"Model: {args.model}")
    
    # Check data availability
    print(f"\n🔍 Checking data availability...")
    availability = check_data_availability(args.data_path, stock_indices, args.use_original_snippets)
    
    available_indices = []
    for stock_index, info in availability.items():
        if info['available']:
            file_type = info['file_type']
            print(f"  ✅ {stock_index}: {info['files_found']} {file_type} files found")
            available_indices.append(stock_index)
        else:
            file_type = info['file_type']
            print(f"  ❌ {stock_index}: No {file_type} data found at {info['path']}")
    
    if not available_indices:
        file_type = "enhanced" if not args.use_original_snippets else "original"
        print(f"\n❌ No {file_type} climate snippet data found for any index.")
        print(f"Expected data structure:")
        if args.use_original_snippets:
            print(f"  {args.data_path}/")
            print(f"  ├── SP500/")
            print(f"  │   ├── climate_segments_1.json")
            print(f"  │   └── climate_segments_2.json")
            print(f"  └── STOXX600/")
            print(f"      ├── climate_segments_1.json")
            print(f"      └── climate_segments_2.json")
        else:
            print(f"  {args.data_path}/")
            print(f"  ├── SP500/")
            print(f"  │   ├── enhanced_climate_segments_1.json")
            print(f"  │   └── enhanced_climate_segments_2.json")
            print(f"  └── STOXX600/")
            print(f"      ├── enhanced_climate_segments_1.json")
            print(f"      └── enhanced_climate_segments_2.json")
            print(f"\n💡 Tip: Run scripts/2.5_calculate_sentence_ratios.py first to create enhanced files")
        return
    
    try:
        # Initialize the semantic searcher
        print(f"\n🔧 Initializing semantic searcher...")
        searcher = ClimateSemanticSearcher(
            base_data_path=str(args.data_path),
            use_enhanced_files=not args.use_original_snippets
        )
        
        # Load climate data
        print(f"📥 Loading climate snippet data...")
        searcher.load_climate_data(available_indices)
        
        # Preview data
        preview_data(searcher, available_indices)
        
        if args.preview_only:
            print("✅ Preview completed. Use without --preview-only to build variables.")
            return
        
        # Build semantic index
        print(f"🧠 Building semantic search index...")
        searcher.build_semantic_index(model_name=args.model)
        
        # Save index if requested
        if args.save_index:
            index_path = args.output_dir / "semantic_index"
            searcher.save_search_index(str(index_path))
            print(f"💾 Semantic index saved to: {index_path}")
        
        # Initialize variable constructor
        print(f"🏗️ Initializing variable constructor...")
        constructor = SemanticClimateVariableConstructor(
            searcher=searcher,
            output_dir=str(args.output_dir)
        )
        
        # Build climate variables
        print(f"🚀 Building semantic climate variables...")
        df = constructor.construct_climate_variables(
            start_year=args.start_year,
            end_year=args.end_year
        )
        
        print(f"\n📊 Construction Results:")
        print(f"  Total firm-quarter observations: {len(df):,}")
        print(f"  Observations with climate content: {(df['has_climate_content'] == True).sum():,}")
        print(f"  Unique firms: {df['ticker'].nunique()}")
        print(f"  Year range: {df['year'].min()}-{df['year'].max()}")
        print(f"  Total climate snippets: {df['total_climate_snippets'].sum():,}")
        
        # Show sentence ratio statistics if available
        if 'climate_sentence_ratio' in df.columns:
            valid_ratios = df[df['climate_sentence_ratio'].notna()]
            if len(valid_ratios) > 0:
                print(f"  Sentence ratio coverage: {len(valid_ratios):,} firm-quarters ({len(valid_ratios)/len(df):.1%})")
                print(f"  Mean climate sentence ratio: {valid_ratios['climate_sentence_ratio'].mean():.4f}")
        
        # Show sample climate exposure statistics
        print(f"\n🎯 Sample Climate Exposure Statistics:")
        exposure_vars = [col for col in df.columns if col.startswith('semantic_') and col.endswith('_exposure')]
        for var in exposure_vars[:5]:  # Show first 5
            topic = var.replace('semantic_', '').replace('_exposure', '')
            mean_exp = df[var].mean()
            coverage = (df[var] > 0).mean()
            print(f"  {topic.title()}: {mean_exp:.4f} mean, {coverage:.1%} coverage")
        
        # Show normalized attention statistics if available
        if 'normalized_climate_attention' in df.columns:
            valid_attention = df[df['normalized_climate_attention'].notna()]
            if len(valid_attention) > 0:
                print(f"\n📏 Normalized Attention Statistics:")
                print(f"  Coverage: {len(valid_attention):,} firm-quarters ({len(valid_attention)/len(df):.1%})")
                print(f"  Mean attention: {valid_attention['normalized_climate_attention'].mean():.4f}")
                print(f"  Median attention: {valid_attention['normalized_climate_attention'].median():.4f}")
                print(f"  95th percentile: {valid_attention['normalized_climate_attention'].quantile(0.95):.4f}")
        
        # Create firm-level summary
        print(f"📈 Creating firm-level summary...")
        firm_summary = constructor.create_firm_level_summary(df)
        print(f"  Firm-level summary saved with {len(firm_summary)} firms")
        
        # Export for analysis
        if args.export_format != 'csv':
            print(f"📤 Exporting to {args.export_format.upper()} format...")
            constructor.export_for_analysis(df, format=args.export_format)
        
        print(f"\n✅ Climate variables construction completed!")
        print(f"📁 Output files saved to: {args.output_dir}")
        print(f"\nMain output files:")
        print(f"  • semantic_climate_panel.csv - Main panel dataset")
        print(f"  • firm_level_summary.csv - Firm-level aggregations")
        print(f"  • methodology.json - Documentation")
        print(f"  • summary_statistics.json - Variable summaries")
        print(f"  • quality_report.json - Data quality metrics")
        
        if args.export_format == 'stata':
            print(f"  • semantic_climate_panel.dta - Stata format")
        elif args.export_format == 'r':
            print(f"  • semantic_climate_panel_for_r.csv - R format")
            print(f"  • analysis_template.R - R analysis template")
    
    except KeyboardInterrupt:
        print("\n🛑 Process interrupted by user")
        return
    except Exception as e:
        logger.error(f"❌ Error during processing: {e}")
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return


if __name__ == "__main__":
    main()