#!/usr/bin/env python3
"""
Monthly climate attention aggregator for enhanced climate snippets.

This script calculates monthly overall climate attention by computing the ratio
of climate-related paragraphs to total paragraphs in all earnings calls for each month.
Creates separate aggregations for overall, EU (STOXX600), and US (SP500) markets.

Usage:
    python scripts/3_agg_variables/monthly_climate_attention.py
    
    # With custom paths
    python scripts/3_agg_variables/monthly_climate_attention.py --enhanced-snippets-path /custom/path --structured-transcripts-path /custom/path

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
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
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
            logging.FileHandler(LOGS_DIR / 'monthly_climate_attention.log', mode='a')
        ]
    )


def parse_date_to_month(date_str: str) -> Optional[str]:
    """
    Parse various date formats to YYYY-MM format.
    
    Args:
        date_str: Date string in various formats
        
    Returns:
        Date in YYYY-MM format or None if parsing fails
    """
    if not date_str or pd.isna(date_str):
        return None
    
    date_str = str(date_str).strip()
    
    # Try different date formats
    date_formats = [
        '%Y-%m-%d',
        '%d-%m-%Y', 
        '%m/%d/%Y',
        '%d/%m/%Y',
        '%Y/%m/%d',
        '%d %B %Y',
        '%B %d, %Y',
        '%d %b %Y',
        '%b %d, %Y'
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
    """
    Count sentences in text using robust heuristics.
    
    Args:
        text: Input text
        
    Returns:
        Number of sentences
    """
    if not text or pd.isna(text):
        return 0
    
    text = str(text).strip()
    
    # Split on sentence endings, but be careful with abbreviations
    sentences = re.split(r'[.!?]+(?=\s+[A-Z]|\s*$)', text)
    
    # Filter empty strings and very short "sentences"
    valid_sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    return len(valid_sentences)


def load_enhanced_climate_snippets(enhanced_snippets_path: Path, 
                                 stock_indices: List[str]) -> List[Dict]:
    """
    Load enhanced climate snippets data.
    
    Args:
        enhanced_snippets_path: Path to enhanced climate snippets
        stock_indices: List of stock indices to process
        
    Returns:
        List of climate snippet records with metadata
    """
    logger = logging.getLogger(__name__)
    
    all_climate_records = []
    
    for stock_index in stock_indices:
        index_path = enhanced_snippets_path / stock_index
        if not index_path.exists():
            logger.warning(f"Enhanced snippets path not found: {index_path}")
            continue
        
        # Find enhanced climate segment files
        json_files = list(index_path.glob("enhanced_climate_segments_*.json"))
        
        if not json_files:
            logger.warning(f"No enhanced climate segment files found for {stock_index}")
            continue
        
        logger.info(f"Loading {len(json_files)} enhanced files for {stock_index}")
        
        for json_file in tqdm(json_files, desc=f"Loading {stock_index} climate data"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for transcript in data:
                    # Extract basic metadata
                    company_name = transcript.get('company_name', '')
                    ticker = transcript.get('ticker', '')
                    year = transcript.get('year')
                    quarter = transcript.get('quarter', '')
                    date_str = transcript.get('date', '')
                    
                    # Parse date to month
                    month = parse_date_to_month(date_str)
                    
                    # Count climate-related sentences
                    climate_sentence_count = 0
                    climate_texts = transcript.get('texts', [])
                    
                    for text_snippet in climate_texts:
                        snippet_sentences = text_snippet.get('sentence_count', 0)
                        if snippet_sentences:
                            climate_sentence_count += snippet_sentences
                        else:
                            # Fallback to counting sentences in text
                            snippet_text = text_snippet.get('text', '')
                            climate_sentence_count += count_sentences_in_text(snippet_text)
                    
                    # Get total sentences from enhanced data
                    total_sentences = transcript.get('total_sentences_in_call')
                    
                    # Create record
                    record = {
                        'stock_index': stock_index,
                        'ticker': ticker,
                        'company_name': company_name,
                        'year': int(year) if year else None,
                        'quarter': quarter,
                        'date': date_str,
                        'month': month,
                        'climate_sentence_count': climate_sentence_count,
                        'total_sentences_in_call': total_sentences,
                        'climate_snippet_count': len(climate_texts),
                        'has_climate_content': len(climate_texts) > 0
                    }
                    
                    all_climate_records.append(record)
                    
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
                continue
    
    logger.info(f"✅ Loaded {len(all_climate_records)} climate records")
    return all_climate_records


def load_structured_transcripts(structured_path: Path, 
                              stock_indices: List[str]) -> List[Dict]:
    """
    Load structured transcript data to get total paragraph counts.
    
    Args:
        structured_path: Path to structured transcripts
        stock_indices: List of stock indices to process
        
    Returns:
        List of transcript records with paragraph counts
    """
    logger = logging.getLogger(__name__)
    
    all_transcript_records = []
    
    for stock_index in stock_indices:
        index_path = structured_path / stock_index
        if not index_path.exists():
            logger.warning(f"Structured transcripts path not found: {index_path}")
            continue
        
        # Find structured call files
        json_files = list(index_path.glob("structured_calls_*.json"))
        
        if not json_files:
            logger.warning(f"No structured transcript files found for {stock_index}")
            continue
        
        logger.info(f"Loading {len(json_files)} structured files for {stock_index}")
        
        for json_file in tqdm(json_files, desc=f"Loading {stock_index} transcripts"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for transcript in data:
                    # Extract metadata
                    ticker = transcript.get('ticker', '')
                    year = transcript.get('year')
                    quarter = transcript.get('quarter', '')
                    date_str = transcript.get('date', '')
                    
                    # Parse date to month
                    month = parse_date_to_month(date_str)
                    
                    # Count total paragraphs
                    mgmt_paragraphs = transcript.get('management_paragraphs', [])
                    qa_paragraphs = transcript.get('qa_paragraphs', [])
                    total_paragraphs = len(mgmt_paragraphs) + len(qa_paragraphs)
                    
                    # Count total sentences (if not available from enhanced data)
                    mgmt_segments = transcript.get('speaker_segments_management', [])
                    qa_segments = transcript.get('speaker_segments_qa', [])
                    
                    total_sentences = 0
                    for segment in mgmt_segments + qa_segments:
                        for paragraph in segment.get('paragraphs', []):
                            total_sentences += count_sentences_in_text(paragraph)
                    
                    record = {
                        'stock_index': stock_index,
                        'ticker': ticker,
                        'year': int(year) if year else None,
                        'quarter': quarter,
                        'date': date_str,
                        'month': month,
                        'total_paragraphs': total_paragraphs,
                        'total_sentences_estimated': total_sentences
                    }
                    
                    all_transcript_records.append(record)
                    
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
                continue
    
    logger.info(f"✅ Loaded {len(all_transcript_records)} transcript records")
    return all_transcript_records


def calculate_monthly_climate_attention(climate_records: List[Dict], 
                                       transcript_records: List[Dict]) -> pd.DataFrame:
    """
    Calculate monthly climate attention aggregated by region.
    
    Args:
        climate_records: Climate snippet records
        transcript_records: Full transcript records
        
    Returns:
        DataFrame with monthly climate attention by region
    """
    logger = logging.getLogger(__name__)
    
    # Convert to DataFrames for easier processing
    df_climate = pd.DataFrame(climate_records)
    df_transcripts = pd.DataFrame(transcript_records)
    
    # Filter out records without valid months
    df_climate = df_climate[df_climate['month'].notna()]
    df_transcripts = df_transcripts[df_transcripts['month'].notna()]
    
    if len(df_climate) == 0 or len(df_transcripts) == 0:
        logger.error("No valid records with parseable dates found")
        return pd.DataFrame()
    
    logger.info(f"Processing {len(df_climate)} climate records and {len(df_transcripts)} transcript records")
    
    # Create region mapping
    region_mapping = {
        'SP500': 'US',
        'STOXX600': 'EU'
    }
    
    df_climate['region'] = df_climate['stock_index'].map(region_mapping)
    df_transcripts['region'] = df_transcripts['stock_index'].map(region_mapping)
    
    # Merge climate and transcript data
    merged_df = pd.merge(
        df_climate[['stock_index', 'ticker', 'month', 'region', 'climate_sentence_count', 
                   'total_sentences_in_call', 'climate_snippet_count', 'has_climate_content']],
        df_transcripts[['stock_index', 'ticker', 'month', 'total_paragraphs', 'total_sentences_estimated']],
        on=['stock_index', 'ticker', 'month'],
        how='outer'
    )
    
    # Fill missing values
    merged_df['climate_sentence_count'] = merged_df['climate_sentence_count'].fillna(0)
    merged_df['climate_snippet_count'] = merged_df['climate_snippet_count'].fillna(0)
    merged_df['has_climate_content'] = merged_df['has_climate_content'].fillna(False)
    
    # Use enhanced total sentences if available, otherwise fall back to estimated
    merged_df['total_sentences_final'] = merged_df['total_sentences_in_call'].combine_first(
        merged_df['total_sentences_estimated']
    )
    
    # Calculate individual call climate attention ratios
    merged_df['call_climate_ratio'] = 0.0
    valid_calls_mask = (merged_df['total_sentences_final'].notna()) & (merged_df['total_sentences_final'] > 0)
    merged_df.loc[valid_calls_mask, 'call_climate_ratio'] = (
        merged_df.loc[valid_calls_mask, 'climate_sentence_count'] / 
        merged_df.loc[valid_calls_mask, 'total_sentences_final']
    )
    
    # Calculate monthly aggregations
    monthly_agg_results = []
    
    # Overall aggregation - average of individual call ratios
    overall_monthly = merged_df.groupby('month').agg({
        'call_climate_ratio': 'mean',  # Average of individual call ratios
        'climate_sentence_count': 'sum',
        'total_sentences_final': 'sum', 
        'total_paragraphs': 'sum',
        'climate_snippet_count': 'sum',
        'ticker': 'count',  # Number of earnings calls
        'has_climate_content': 'sum'  # Number of calls with climate content
    }).reset_index()
    
    overall_monthly['region'] = 'Overall'
    overall_monthly['climate_attention_ratio'] = overall_monthly['call_climate_ratio']
    
    monthly_agg_results.append(overall_monthly)
    
    # Regional aggregations
    for region in ['US', 'EU']:
        region_data = merged_df[merged_df['region'] == region]
        
        if len(region_data) > 0:
            region_monthly = region_data.groupby('month').agg({
                'call_climate_ratio': 'mean',  # Average of individual call ratios
                'climate_sentence_count': 'sum',
                'total_sentences_final': 'sum',
                'total_paragraphs': 'sum',
                'climate_snippet_count': 'sum',
                'ticker': 'count',
                'has_climate_content': 'sum'
            }).reset_index()
            
            region_monthly['region'] = region
            region_monthly['climate_attention_ratio'] = region_monthly['call_climate_ratio']
            
            monthly_agg_results.append(region_monthly)
    
    # Combine all results
    final_df = pd.concat(monthly_agg_results, ignore_index=True)
    
    # Rename columns for clarity
    final_df = final_df.rename(columns={
        'ticker': 'earnings_calls_count',
        'climate_sentence_count': 'total_climate_sentences',
        'total_sentences_final': 'total_sentences',
        'climate_snippet_count': 'total_climate_snippets',
        'has_climate_content': 'calls_with_climate_content'
    })
    
    # Calculate coverage rate correctly
    final_df['climate_coverage_rate'] = final_df['calls_with_climate_content'] / final_df['earnings_calls_count']
    
    # Sort by month and region
    final_df = final_df.sort_values(['month', 'region']).reset_index(drop=True)
    
    logger.info(f"✅ Calculated monthly climate attention for {len(final_df)} month-region combinations")
    
    return final_df


def create_summary_statistics(monthly_df: pd.DataFrame) -> Dict:
    """Create summary statistics for the monthly climate attention data."""
    
    summary_stats = {}
    
    for region in monthly_df['region'].unique():
        region_data = monthly_df[monthly_df['region'] == region]
        
        summary_stats[region] = {
            'time_period': {
                'start_month': region_data['month'].min(),
                'end_month': region_data['month'].max(),
                'total_months': len(region_data)
            },
            'climate_attention': {
                'mean_ratio': float(region_data['climate_attention_ratio'].mean()),
                'median_ratio': float(region_data['climate_attention_ratio'].median()),
                'std_ratio': float(region_data['climate_attention_ratio'].std()),
                'min_ratio': float(region_data['climate_attention_ratio'].min()),
                'max_ratio': float(region_data['climate_attention_ratio'].max()),
                'p25_ratio': float(region_data['climate_attention_ratio'].quantile(0.25)),
                'p75_ratio': float(region_data['climate_attention_ratio'].quantile(0.75))
            },
            'volume_statistics': {
                'total_earnings_calls': int(region_data['earnings_calls_count'].sum()),
                'avg_calls_per_month': float(region_data['earnings_calls_count'].mean()),
                'total_climate_sentences': int(region_data['total_climate_sentences'].sum()),
                'total_sentences': int(region_data['total_sentences'].sum()),
                'avg_climate_coverage_rate': float(region_data['climate_coverage_rate'].mean())
            }
        }
    
    return summary_stats


def save_results(monthly_df: pd.DataFrame, summary_stats: Dict, output_dir: Path):
    """Save results to files."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save main monthly data
    monthly_df.to_csv(output_dir / 'monthly_climate_attention.csv', index=False)
    monthly_df.to_parquet(output_dir / 'monthly_climate_attention.parquet')
    
    # Save summary statistics
    with open(output_dir / 'monthly_climate_attention_summary.json', 'w') as f:
        json.dump(summary_stats, f, indent=2, default=str)
    
    # Create separate files by region for easy analysis
    for region in monthly_df['region'].unique():
        region_data = monthly_df[monthly_df['region'] == region]
        region_filename = f'monthly_climate_attention_{region.lower()}.csv'
        region_data.to_csv(output_dir / region_filename, index=False)
    
    # Create Stata-compatible file
    try:
        monthly_df.to_stata(
            output_dir / 'monthly_climate_attention.dta',
            write_index=False,
            version=117
        )
    except:
        # Fallback to CSV for Stata
        monthly_df.to_csv(output_dir / 'monthly_climate_attention_for_stata.csv', index=False)
    
    print(f"💾 Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Calculate monthly climate attention from enhanced climate snippets',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--enhanced-snippets-path',
        type=Path,
        default=Path("outputs/enhanced_climate_snippets"),
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
        default=Path("outputs/monthly_aggregates"),
        help='Output directory for monthly aggregates'
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
    
    print("📊 Monthly Climate Attention Aggregator")
    print("=" * 50)
    print(f"Enhanced snippets path: {args.enhanced_snippets_path}")
    print(f"Structured transcripts path: {args.structured_transcripts_path}")
    print(f"Stock indices: {', '.join(args.stock_indices)}")
    print(f"Output directory: {args.output_dir}")
    
    try:
        # Load enhanced climate snippets
        print(f"\n📥 Loading enhanced climate snippets...")
        climate_records = load_enhanced_climate_snippets(
            args.enhanced_snippets_path, 
            args.stock_indices
        )
        
        if not climate_records:
            print("❌ No climate records found!")
            return
        
        # Load structured transcripts
        print(f"📥 Loading structured transcripts...")
        transcript_records = load_structured_transcripts(
            args.structured_transcripts_path,
            args.stock_indices
        )
        
        if not transcript_records:
            print("❌ No transcript records found!")
            return
        
        # Calculate monthly climate attention
        print(f"🧮 Calculating monthly climate attention...")
        monthly_df = calculate_monthly_climate_attention(
            climate_records, 
            transcript_records
        )
        
        if monthly_df.empty:
            print("❌ No monthly data could be calculated!")
            return
        
        # Create summary statistics
        summary_stats = create_summary_statistics(monthly_df)
        
        # Save results
        save_results(monthly_df, summary_stats, args.output_dir)
        
        # Print summary
        print(f"\n📊 Monthly Climate Attention Results:")
        print(f"  Total month-region observations: {len(monthly_df)}")
        print(f"  Date range: {monthly_df['month'].min()} to {monthly_df['month'].max()}")
        
        for region in ['Overall', 'US', 'EU']:
            if region in monthly_df['region'].values:
                region_data = monthly_df[monthly_df['region'] == region]
                avg_attention = region_data['climate_attention_ratio'].mean()
                total_calls = region_data['earnings_calls_count'].sum()
                print(f"  {region}: {avg_attention:.4f} avg attention, {total_calls:,} total calls")
        
        print(f"\n✅ Monthly climate attention calculation completed!")
        print(f"📁 Output files:")
        print(f"  • monthly_climate_attention.csv - Main dataset")
        print(f"  • monthly_climate_attention_overall.csv - Overall aggregation")
        print(f"  • monthly_climate_attention_us.csv - US (SP500) data")
        print(f"  • monthly_climate_attention_eu.csv - EU (STOXX600) data")
        print(f"  • monthly_climate_attention_summary.json - Summary statistics")
        print(f"  • monthly_climate_attention.dta - Stata format")
        
    except Exception as e:
        logger.error(f"❌ Error during processing: {e}")
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()