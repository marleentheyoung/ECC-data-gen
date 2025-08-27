#!/usr/bin/env python3
"""
Firm-level climate attention calculator for enhanced climate snippets.

This script calculates firm-quarter climate attention measures by computing the ratio
of climate-related sentences to total sentences for individual earnings calls.
Creates panel data suitable for econometric analysis and event studies.

Usage:
    python scripts/3_agg_variables/firm_level_climate_attention.py

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
from typing import Dict, List, Optional
from tqdm import tqdm
from datetime import datetime
import re

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

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
            logging.FileHandler(LOGS_DIR / 'firm_level_climate_attention.log', mode='a')
        ]
    )


def parse_date_to_month(date_str: str) -> Optional[str]:
    """Parse various date formats to YYYY-MM format."""
    if not date_str or pd.isna(date_str):
        return None
    
    date_str = str(date_str).strip()
    
    # Try different date formats
    date_formats = [
        '%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d',
        '%d %B %Y', '%B %d, %Y', '%d %b %Y', '%b %d, %Y'
    ]
    
    for fmt in date_formats:
        try:
            parsed_date = datetime.strptime(date_str, fmt)
            return parsed_date.strftime('%Y-%m')
        except ValueError:
            continue
    
    # Try pandas date parser as fallback
    try:
        parsed_date = pd.to_datetime(date_str, dayfirst=True)
        return parsed_date.strftime('%Y-%m')
    except:
        return None


def count_sentences_in_text(text: str) -> int:
    """Count sentences in text using robust heuristics."""
    if not text or pd.isna(text):
        return 0
    
    text = str(text).strip()
    sentences = re.split(r'[.!?]+(?=\s+[A-Z]|\s*$)', text)
    valid_sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    return len(valid_sentences)


def create_firm_level_climate_panel(enhanced_snippets_path: Path, 
                                   structured_path: Path,
                                   stock_indices: List[str]) -> pd.DataFrame:
    """
    Create firm-level climate attention panel dataset.
    
    Returns:
        DataFrame with firm-quarter climate attention measures
    """
    logger = logging.getLogger(__name__)
    
    all_firm_records = []
    
    for stock_index in stock_indices:
        logger.info(f"Processing {stock_index}...")
        
        # Load enhanced climate snippets
        enhanced_path = enhanced_snippets_path / stock_index
        if not enhanced_path.exists():
            logger.warning(f"Enhanced snippets path not found: {enhanced_path}")
            continue
        
        json_files = list(enhanced_path.glob("enhanced_climate_segments_*.json"))
        
        for json_file in tqdm(json_files, desc=f"Processing {stock_index}"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for transcript in data:
                    # Extract metadata
                    company_name = transcript.get('company_name', '')
                    ticker = transcript.get('ticker', '')
                    year = transcript.get('year')
                    quarter = transcript.get('quarter', '')
                    date_str = transcript.get('date', '')
                    month = parse_date_to_month(date_str)

                    # Count climate sentences
                    climate_sentence_count = 0
                    climate_texts = transcript.get('texts', [])
                    
                    for text_snippet in climate_texts:
                        snippet_sentences = text_snippet.get('sentence_count', 0)
                        if snippet_sentences:
                            climate_sentence_count += snippet_sentences
                        else:
                            snippet_text = text_snippet.get('text', '')
                            climate_sentence_count += count_sentences_in_text(snippet_text)
                    
                    # Get total sentences (use enhanced data if available)
                    total_sentences = transcript.get('total_sentences_in_call')
                    
                    # Calculate climate attention ratio
                    climate_attention_ratio = 0.0
                    if total_sentences and total_sentences > 0:
                        climate_attention_ratio = climate_sentence_count / total_sentences
                    
                    # Create firm-quarter record
                    record = {
                        # Identifiers
                        'stock_index': stock_index,
                        'region': 'US' if stock_index == 'SP500' else 'EU',
                        'ticker': ticker,
                        'company_name': company_name,
                        'year': int(year) if year else None,
                        'quarter': quarter,
                        'date': date_str,
                        'month': month,
                        
                        # Climate attention measures
                        'climate_attention_ratio': climate_attention_ratio,
                        'climate_sentence_count': climate_sentence_count,
                        'total_sentences_in_call': total_sentences,
                        'climate_snippet_count': len(climate_texts),
                        'has_climate_content': len(climate_texts) > 0,
                        
                        # Coverage indicators
                        'climate_coverage_binary': 1 if len(climate_texts) > 0 else 0,
                        'log_climate_sentences': np.log(climate_sentence_count + 1),
                        'log_total_sentences': np.log(total_sentences + 1) if total_sentences else None
                    }
                    
                    all_firm_records.append(record)
                    
            except Exception as e:
                logger.error(f"Error processing {json_file}: {e}")
                continue

    
    # # After collecting all records, deduplicate
    # seen_transcripts = {}
    # deduplicated_records = []
    
    # for record in all_firm_records:
    #     # Create unique key
    #     key = f"{record['ticker']}_{record['year']}_{record['quarter']}"
        
    #     if key in seen_transcripts:
    #         # Keep the record with more total sentences (likely more complete)
    #         existing = seen_transcripts[key]
    #         if record['total_sentences_in_call'] and existing['total_sentences_in_call']:
    #             if record['total_sentences_in_call'] > existing['total_sentences_in_call']:
    #                 # Replace with better record
    #                 deduplicated_records = [r for r in deduplicated_records if f"{r['ticker']}_{r['year']}_{r['quarter']}" != key]
    #                 deduplicated_records.append(record)
    #                 seen_transcripts[key] = record
    #                 print(f"🔄 Replaced duplicate {key}: {existing['total_sentences_in_call']} → {record['total_sentences_in_call']} sentences")
    #             else:
    #                 print(f"⏭️ Skipped duplicate {key}: keeping better record")
    #         else:
    #             # If one record has None total_sentences, prefer the one with actual data
    #             if record['total_sentences_in_call'] and not existing['total_sentences_in_call']:
    #                 deduplicated_records = [r for r in deduplicated_records if f"{r['ticker']}_{r['year']}_{r['quarter']}" != key]
    #                 deduplicated_records.append(record)
    #                 seen_transcripts[key] = record
    #                 print(f"🔄 Replaced duplicate {key}: None → {record['total_sentences_in_call']} sentences")
    #             else:
    #                 print(f"⏭️ Skipped duplicate {key}: keeping existing record")
    #     else:
    #         deduplicated_records.append(record)
    #         seen_transcripts[key] = record
    
    # all_firm_records = deduplicated_records
    # print(f"✅ Deduplication: {len(all_firm_records)} records after removing duplicates")
    
    # # After collecting all records, deduplicate
    # seen_transcripts = {}
    # deduplicated_records = []
    
    # for record in all_firm_records:
    #     # Create unique key
    #     key = f"{record['ticker']}_{record['year']}_{record['quarter']}"
        
    #     if key in seen_transcripts:
    #         # Keep the record with more total sentences (likely more complete)
    #         existing = seen_transcripts[key]
    #         if record['total_sentences_in_call'] and existing['total_sentences_in_call']:
    #             if record['total_sentences_in_call'] > existing['total_sentences_in_call']:
    #                 # Replace with better record
    #                 deduplicated_records = [r for r in deduplicated_records if f"{r['ticker']}_{r['year']}_{r['quarter']}" != key]
    #                 deduplicated_records.append(record)
    #                 seen_transcripts[key] = record
    #                 print(f"🔄 Replaced duplicate {key}: {existing['total_sentences_in_call']} → {record['total_sentences_in_call']} sentences")
    #             else:
    #                 print(f"⏭️ Skipped duplicate {key}: keeping better record")
    #         else:
    #             # If one record has None total_sentences, prefer the one with actual data
    #             if record['total_sentences_in_call'] and not existing['total_sentences_in_call']:
    #                 deduplicated_records = [r for r in deduplicated_records if f"{r['ticker']}_{r['year']}_{r['quarter']}" != key]
    #                 deduplicated_records.append(record)
    #                 seen_transcripts[key] = record
    #                 print(f"🔄 Replaced duplicate {key}: None → {record['total_sentences_in_call']} sentences")
    #             else:
    #                 print(f"⏭️ Skipped duplicate {key}: keeping existing record")
    #     else:
    #         deduplicated_records.append(record)
    #         seen_transcripts[key] = record
    
    # all_firm_records = deduplicated_records
    # print(f"✅ Basic deduplication: {len(all_firm_records)} records after removing duplicates")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_firm_records)
    
    if df.empty:
        logger.error("No firm records created!")
        return df
    
    # Data quality filters
    df = df[df['ticker'].notna() & (df['ticker'] != '')]
    df = df[df['year'].notna()]
    
    # # Strategy: Remove EU records that also exist as US records
    # # 1. Find EU companies without dash that also exist as US companies
    # eu_no_dash = df[(df['region'] == 'EU') & 
    #                 (df['stock_index'] == 'STOXX600') & 
    #                 (~df['ticker'].str.contains('-', na=False))]
    
    # us_companies = df[(df['region'] == 'US') & 
    #                   (df['stock_index'] == 'SP500')]
    
    # # Find overlapping tickers
    # overlapping_tickers = set(eu_no_dash['ticker'].unique()) & set(us_companies['ticker'].unique())
    
    # if overlapping_tickers:
    #     print(f"🚨 Found {len(overlapping_tickers)} companies in both EU and US indices:")
    #     print(f"   Overlapping tickers: {', '.join(sorted(overlapping_tickers))}")
        
    #     # Remove EU records for overlapping tickers
    #     before_count = len(df)
    #     removal_mask = ((df['region'] == 'EU') & 
    #                    (df['stock_index'] == 'STOXX600') & 
    #                    (df['ticker'].isin(overlapping_tickers)))
        
    #     removed_count = removal_mask.sum()
    #     df = df[~removal_mask]
        
    #     print(f"🗑️ Removed {removed_count} EU duplicate records")
    #     print(f"📊 Dataset: {before_count} → {len(df)} records")
    # else:
    #     print("✅ No overlapping companies found between EU and US indices")
    
    # Create extended ticker column with -US suffix for US companies
    df['ISSUER_TICKER'] = df['ticker']
    us_mask = df['region'] == 'US'
    df.loc[us_mask, 'ISSUER_TICKER'] = df.loc[us_mask, 'ticker'] + '-US'
    
    # Create quarterly identifier
    df['quarter_id'] = df['year'].astype(str) + df['quarter'].astype(str)
    
    # Sort by firm and time
    df = df.sort_values(['ticker', 'year', 'quarter']).reset_index(drop=True)
    
    logger.info(f"✅ Created panel with {len(df)} firm-quarter observations")
    logger.info(f"   {df['ticker'].nunique()} unique firms")
    logger.info(f"   {df['quarter_id'].nunique()} unique quarters")
    
    return df


def add_lagged_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged climate attention variables for econometric analysis."""
    
    df = df.sort_values(['ticker', 'year', 'quarter'])
    
    # Create lagged variables (useful for event studies)
    for lag in [1, 2, 4]:  # 1, 2, and 4 quarters
        df[f'climate_attention_ratio_lag{lag}'] = df.groupby('ticker')['climate_attention_ratio'].shift(lag)
        df[f'has_climate_content_lag{lag}'] = df.groupby('ticker')['has_climate_content'].shift(lag)
    
    # Create moving averages (useful for smoothing)
    for window in [2, 4]:
        df[f'climate_attention_ratio_ma{window}'] = (
            df.groupby('ticker')['climate_attention_ratio']
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
    
    return df


def create_summary_statistics(df: pd.DataFrame) -> Dict:
    """Create comprehensive summary statistics."""
    
    summary_stats = {
        'panel_structure': {
            'total_observations': len(df),
            'unique_firms': df['ticker'].nunique(),
            'unique_quarters': df['quarter_id'].nunique(),
            'time_span': {
                'start_year': int(df['year'].min()),
                'end_year': int(df['year'].max()),
                'start_quarter': df['quarter_id'].min(),
                'end_quarter': df['quarter_id'].max()
            }
        },
        
        'climate_attention_distribution': {
            'mean': float(df['climate_attention_ratio'].mean()),
            'median': float(df['climate_attention_ratio'].median()),
            'std': float(df['climate_attention_ratio'].std()),
            'min': float(df['climate_attention_ratio'].min()),
            'max': float(df['climate_attention_ratio'].max()),
            'p25': float(df['climate_attention_ratio'].quantile(0.25)),
            'p75': float(df['climate_attention_ratio'].quantile(0.75)),
            'p90': float(df['climate_attention_ratio'].quantile(0.90)),
            'p99': float(df['climate_attention_ratio'].quantile(0.99))
        },
        
        'coverage_statistics': {
            'pct_calls_with_climate': float(df['has_climate_content'].mean() * 100),
            'avg_climate_sentences_per_call': float(df['climate_sentence_count'].mean()),
            'avg_total_sentences_per_call': float(df['total_sentences_in_call'].mean()),
            'avg_climate_snippets_per_call': float(df['climate_snippet_count'].mean())
        }
    }
    
    # Regional breakdown
    for region in ['US', 'EU']:
        region_data = df[df['region'] == region]
        if len(region_data) > 0:
            summary_stats[f'{region}_statistics'] = {
                'observations': len(region_data),
                'firms': region_data['ticker'].nunique(),
                'avg_climate_attention': float(region_data['climate_attention_ratio'].mean()),
                'climate_coverage_pct': float(region_data['has_climate_content'].mean() * 100)
            }
    
    return summary_stats


def save_firm_level_results(df: pd.DataFrame, summary_stats: Dict, output_dir: Path):
    """Save firm-level results in multiple formats."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Main panel dataset
    df.to_csv(output_dir / 'general_climate' / 'firm_level_climate_attention.csv', index=False)
    df.to_parquet(output_dir / 'general_climate' / 'firm_level_climate_attention.parquet')
    
    # Summary statistics
    with open(output_dir / 'firm_level_summary_statistics.json', 'w') as f:
        json.dump(summary_stats, f, indent=2, default=str)
    
    # Regional subsets
    for region in ['US', 'EU']:
        region_data = df[df['region'] == region]
        if len(region_data) > 0:
            region_data.to_csv(output_dir / 'general_climate' / f'firm_level_climate_attention_{region.lower()}.csv', index=False)
    
    # Stata format for econometric analysis
    try:
        # Create Stata-friendly variable names
        df_stata = df.copy()
        df_stata.columns = [col.replace('_', '').lower()[:32] for col in df_stata.columns]
        
        df_stata.to_stata(
            output_dir / 'firm_level_climate_attention.dta',
            write_index=False,
            version=117
        )
    except Exception as e:
        # Fallback
        df.to_csv(output_dir / 'general_climate' / 'firm_level_climate_attention_for_stata.csv', index=False)
    
    # Create balanced panel indicator
    firm_quarters = df.groupby('ticker').size()
    max_quarters = firm_quarters.max()
    balanced_firms = firm_quarters[firm_quarters == max_quarters].index
    
    df['balanced_panel'] = df['ticker'].isin(balanced_firms)
    balanced_df = df[df['balanced_panel']].copy()
    
    if len(balanced_df) > 0:
        balanced_df.to_csv(output_dir / 'general_climate' / 'firm_level_climate_attention_balanced.csv', index=False)
    
    print(f"💾 Results saved to: {output_dir}")
    print(f"📊 Panel structure:")
    print(f"   • {len(df):,} firm-quarter observations")
    print(f"   • {df['ticker'].nunique():,} unique firms")
    print(f"   • {df['quarter_id'].nunique()} unique quarters")
    print(f"   • {len(balanced_df):,} observations in balanced panel")


def main():
    parser = argparse.ArgumentParser(
        description='Create firm-level climate attention panel dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--enhanced-snippets-path',
        type=Path,
        default=Path("data/enhanced_climate_snippets"),
        help='Path to enhanced climate snippets data'
    )
    
    parser.add_argument(
        '--structured-transcripts-path',
        type=Path,
        default=Path("data/processed/structured_jsons"),
        help='Path to structured transcripts data'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path("outputs/firm_level_panel"),
        help='Output directory for firm-level panel data'
    )
    
    parser.add_argument(
        '--stock-indices',
        nargs='+',
        default=SUPPORTED_INDICES,
        choices=SUPPORTED_INDICES,
        help='Stock indices to process'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    print("🏢 Firm-Level Climate Attention Panel Creator")
    print("=" * 50)
    
    try:
        # Create firm-level panel
        df = create_firm_level_climate_panel(
            args.enhanced_snippets_path,
            args.structured_transcripts_path,
            args.stock_indices
        )
        
        if df.empty:
            print("❌ No firm-level data created!")
            return
        
        # Add lagged variables
        df = add_lagged_variables(df)
        
        # Create summary statistics
        summary_stats = create_summary_statistics(df)
        
        # Save results
        save_firm_level_results(df, summary_stats, args.output_dir)
        
        print(f"\n✅ Firm-level climate attention panel completed!")
        print(f"📁 Key output files:")
        print(f"   • firm_level_climate_attention.csv - Main panel dataset")
        print(f"   • firm_level_climate_attention.dta - Stata format")
        print(f"   • firm_level_climate_attention_balanced.csv - Balanced panel")
        print(f"   • firm_level_summary_statistics.json - Summary stats")
        
    except Exception as e:
        logger.error(f"❌ Error during processing: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()