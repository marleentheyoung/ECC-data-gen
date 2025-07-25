#!/usr/bin/env python3
"""
Quick prototype script to calculate basic climate attention variables.

This script creates a simple firm-quarter panel with overall climate attention
measures without complex semantic topic matching - much faster for prototyping.

Usage:
    # Quick prototype for STOXX600
    python scripts/3_quick_climate_prototype.py STOXX600
    
    # Both indices with custom year range
    python scripts/3_quick_climate_prototype.py --all --start-year 2010 --end-year 2019

Author: Marleen de Jonge
Date: 2025
"""

import argparse
import json
import logging
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
from tqdm import tqdm

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.config import SUPPORTED_INDICES, LOGS_DIR


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOGS_DIR / 'quick_climate_prototype.log', mode='a')
        ]
    )


def load_climate_snippets_simple(data_path: Path, stock_indices: List[str], 
                                use_enhanced: bool = True) -> List[Dict[str, Any]]:
    """
    Load climate snippets without complex processing - just count them.
    
    Args:
        data_path: Path to climate snippets data
        stock_indices: List of stock indices to process
        use_enhanced: Whether to use enhanced files with sentence ratios
        
    Returns:
        List of simple climate records
    """
    all_climate_records = []
    
    for stock_index in stock_indices:
        index_path = data_path / stock_index
        if not index_path.exists():
            print(f"⚠️ Path not found: {index_path}")
            continue
        
        # Find JSON files
        if use_enhanced:
            json_files = list(index_path.glob("enhanced_climate_segments_*.json"))
        else:
            json_files = list(index_path.glob("climate_segments_*.json"))
        
        if not json_files:
            print(f"⚠️ No climate files found for {stock_index}")
            continue
        
        print(f"📥 Loading {len(json_files)} files for {stock_index}...")
        
        for json_file in tqdm(json_files, desc=f"Loading {stock_index}"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for transcript in data:
                    # Extract basic info
                    company_name = transcript.get('company_name', '')
                    ticker = transcript.get('ticker', '')
                    year = transcript.get('year')
                    quarter = transcript.get('quarter', '')
                    date = transcript.get('date', '')
                    
                    # Count climate snippets
                    climate_texts = transcript.get('texts', [])
                    climate_snippet_count = len(climate_texts)
                    
                    # Calculate total words in climate snippets
                    total_climate_words = sum(
                        len(text.get('text', '').split()) 
                        for text in climate_texts
                    )
                    
                    # Get sentence ratio info if available (enhanced files)
                    climate_sentence_count = transcript.get('climate_sentence_count', 0)
                    total_sentences_in_call = transcript.get('total_sentences_in_call')
                    climate_sentence_ratio = transcript.get('climate_sentence_ratio')
                    
                    # Create simple record
                    record = {
                        'stock_index': stock_index,
                        'company_name': company_name,
                        'ticker': ticker,
                        'year': int(year) if year else None,
                        'quarter': quarter,
                        'date': date,
                        'climate_snippet_count': climate_snippet_count,
                        'total_climate_words': total_climate_words,
                        'climate_sentence_count': climate_sentence_count,
                        'total_sentences_in_call': total_sentences_in_call,
                        'climate_sentence_ratio': climate_sentence_ratio,
                        'has_climate_content': climate_snippet_count > 0
                    }
                    
                    all_climate_records.append(record)
                    
            except Exception as e:
                print(f"⚠️ Error loading {json_file}: {e}")
                continue
    
    print(f"✅ Loaded {len(all_climate_records)} climate records")
    return all_climate_records


def create_firm_quarter_panel(climate_records: List[Dict[str, Any]], 
                             start_year: int, end_year: int) -> pd.DataFrame:
    """
    Create a simple firm-quarter panel with climate attention measures.
    
    Args:
        climate_records: List of climate records from transcripts
        start_year: Start year for panel
        end_year: End year for panel
        
    Returns:
        DataFrame with firm-quarter climate variables
    """
    print(f"🏗️ Creating firm-quarter panel ({start_year}-{end_year})...")
    
    # Convert to DataFrame
    df_climate = pd.DataFrame(climate_records)
    
    # Clean data
    df_climate = df_climate.dropna(subset=['year', 'ticker'])
    df_climate = df_climate[(df_climate['year'] >= start_year) & (df_climate['year'] <= end_year)]
    
    # Get all firms
    firms = df_climate['ticker'].unique()
    
    # Create complete panel structure
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    years = range(start_year, end_year + 1)
    
    panel_data = []
    
    print(f"📊 Creating panel for {len(firms)} firms...")
    
    for firm in tqdm(firms):
        # Get firm info
        firm_data = df_climate[df_climate['ticker'] == firm].iloc[0]
        company_name = firm_data['company_name']
        stock_index = firm_data['stock_index']
        
        for year in years:
            for quarter in quarters:
                # Find matching climate record
                matching_records = df_climate[
                    (df_climate['ticker'] == firm) & 
                    (df_climate['year'] == year) & 
                    (df_climate['quarter'] == quarter)
                ]
                
                if len(matching_records) > 0:
                    # Has earnings call with climate data
                    record = matching_records.iloc[0]
                    
                    # Calculate normalized climate attention
                    if (record['total_sentences_in_call'] and 
                        record['total_sentences_in_call'] > 0 and 
                        record['climate_sentence_count']):
                        normalized_attention = record['climate_sentence_count'] / record['total_sentences_in_call']
                    else:
                        normalized_attention = record['climate_sentence_ratio']
                    
                    panel_row = {
                        'ticker': firm,
                        'company_name': company_name,
                        'stock_index': stock_index,
                        'year': year,
                        'quarter': quarter,
                        'date': record['date'],
                        'has_earnings_call': True,
                        'has_climate_content': record['has_climate_content'],
                        'climate_snippet_count': record['climate_snippet_count'],
                        'total_climate_words': record['total_climate_words'],
                        'climate_sentence_count': record['climate_sentence_count'],
                        'total_sentences_in_call': record['total_sentences_in_call'],
                        'climate_sentence_ratio': record['climate_sentence_ratio'],
                        'normalized_climate_attention': normalized_attention,
                        # Simple binary indicators
                        'has_any_climate_discussion': 1 if record['has_climate_content'] else 0,
                        'high_climate_attention': 1 if (normalized_attention and normalized_attention > 0.05) else 0
                    }
                else:
                    # No earnings call this quarter
                    panel_row = {
                        'ticker': firm,
                        'company_name': company_name,
                        'stock_index': stock_index,
                        'year': year,
                        'quarter': quarter,
                        'date': None,
                        'has_earnings_call': False,
                        'has_climate_content': False,
                        'climate_snippet_count': 0,
                        'total_climate_words': 0,
                        'climate_sentence_count': 0,
                        'total_sentences_in_call': None,
                        'climate_sentence_ratio': None,
                        'normalized_climate_attention': None,
                        'has_any_climate_discussion': 0,
                        'high_climate_attention': 0
                    }
                
                panel_data.append(panel_row)
    
    # Convert to DataFrame
    df_panel = pd.DataFrame(panel_data)
    
    # Add quarter-year identifier
    df_panel['quarter_year'] = df_panel['year'].astype(str) + df_panel['quarter']
    
    # Add firm and time identifiers for regression analysis
    df_panel['firm_id'] = pd.Categorical(df_panel['ticker']).codes
    df_panel['time_id'] = pd.Categorical(df_panel['quarter_year']).codes
    
    print(f"✅ Created panel: {len(df_panel):,} firm-quarter observations")
    return df_panel


def calculate_simple_statistics(df_panel: pd.DataFrame) -> Dict[str, Any]:
    """Calculate simple summary statistics."""
    
    stats = {
        'panel_overview': {
            'total_observations': len(df_panel),
            'unique_firms': df_panel['ticker'].nunique(),
            'time_periods': df_panel['quarter_year'].nunique(),
            'observations_with_calls': (df_panel['has_earnings_call'] == True).sum(),
            'observations_with_climate': (df_panel['has_climate_content'] == True).sum(),
            'coverage_rate': (df_panel['has_earnings_call'] == True).mean(),
            'climate_discussion_rate': (df_panel['has_climate_content'] == True).mean()
        }
    }
    
    # Statistics for calls with climate content
    climate_obs = df_panel[df_panel['has_climate_content'] == True]
    
    if len(climate_obs) > 0:
        stats['climate_attention'] = {
            'mean_snippets_per_call': climate_obs['climate_snippet_count'].mean(),
            'mean_climate_words': climate_obs['total_climate_words'].mean(),
            'mean_normalized_attention': climate_obs['normalized_climate_attention'].mean(),
            'median_normalized_attention': climate_obs['normalized_climate_attention'].median(),
            'p90_normalized_attention': climate_obs['normalized_climate_attention'].quantile(0.9),
            'high_attention_rate': (climate_obs['normalized_climate_attention'] > 0.05).mean()
        }
    
    # Temporal trends
    yearly_stats = df_panel.groupby('year').agg({
        'has_earnings_call': 'sum',
        'has_climate_content': 'sum',
        'ticker': 'nunique'
    }).rename(columns={'ticker': 'unique_firms'})
    
    stats['temporal_trends'] = yearly_stats.to_dict('index')
    
    return stats


def save_quick_results(df_panel: pd.DataFrame, stats: Dict[str, Any], output_dir: Path):
    """Save the quick prototype results."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save main panel
    df_panel.to_csv(output_dir / 'quick_climate_panel.csv', index=False)
    df_panel.to_parquet(output_dir / 'quick_climate_panel.parquet')
    
    # Save statistics
    with open(output_dir / 'quick_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    # Create simple Stata file if possible
    try:
        df_panel.to_stata(
            output_dir / 'quick_climate_panel.dta', 
            write_index=False,
            version=117
        )
    except:
        # Fallback to CSV for Stata
        df_panel.to_csv(output_dir / 'quick_climate_panel_for_stata.csv', index=False)
    
    print(f"💾 Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Quick prototype for basic climate attention variables',
        formatter_class=argparse.RawDescriptionHelpFormatter
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
        default=2010,
        help='Start year for panel (default: 2010)'
    )
    
    parser.add_argument(
        '--end-year',
        type=int,
        default=2019,
        help='End year for panel (default: 2019)'
    )
    
    parser.add_argument(
        '--data-path',
        type=Path,
        default=Path("outputs/enhanced_climate_snippets"),
        help='Path to climate snippets data'
    )
    
    parser.add_argument(
        '--use-original',
        action='store_true',
        help='Use original climate segments (not enhanced)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path("outputs/quick_climate_prototype"),
        help='Output directory'
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
    
    setup_logging(args.verbose)
    
    # Determine stock indices
    stock_indices = SUPPORTED_INDICES if args.all else [args.stock_index]
    
    print("⚡ Quick Climate Attention Prototype")
    print("=" * 50)
    print(f"Stock indices: {', '.join(stock_indices)}")
    print(f"Year range: {args.start_year}-{args.end_year}")
    print(f"Data path: {args.data_path}")
    
    try:
        # Load climate records (fast - no semantic processing)
        climate_records = load_climate_snippets_simple(
            args.data_path, 
            stock_indices, 
            use_enhanced=not args.use_original
        )
        
        if not climate_records:
            print("❌ No climate records found!")
            return
        
        # Create panel (fast - just counting and aggregating)
        df_panel = create_firm_quarter_panel(
            climate_records, 
            args.start_year, 
            args.end_year
        )
        
        # Quick statistics
        stats = calculate_simple_statistics(df_panel)
        
        # Save results
        save_quick_results(df_panel, stats, args.output_dir)
        
        # Print summary
        print(f"\n📊 Quick Prototype Results:")
        overview = stats['panel_overview']
        print(f"  Total observations: {overview['total_observations']:,}")
        print(f"  Unique firms: {overview['unique_firms']:,}")
        print(f"  Coverage rate: {overview['coverage_rate']:.1%}")
        print(f"  Climate discussion rate: {overview['climate_discussion_rate']:.1%}")
        
        if 'climate_attention' in stats:
            climate_stats = stats['climate_attention']
            print(f"  Mean climate attention: {climate_stats['mean_normalized_attention']:.4f}")
            print(f"  High attention rate: {climate_stats['high_attention_rate']:.1%}")
        
        print(f"\n✅ Quick prototype completed!")
        print(f"📁 Files saved:")
        print(f"  • quick_climate_panel.csv - Main dataset")
        print(f"  • quick_climate_panel.dta - Stata format")
        print(f"  • quick_statistics.json - Summary stats")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()