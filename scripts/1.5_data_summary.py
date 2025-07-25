#!/usr/bin/env python3
"""
Data summary script for ECC transcript analysis.

This script analyzes structured transcript JSONs and generates comprehensive
descriptive statistics about dataset coverage, temporal distribution, and
transcript characteristics.

Usage:
    # Analyze both SP500 and STOXX600
    python scripts/1.5_data_summary.py --all
    
    # Analyze SP500 only
    python scripts/1.5_data_summary.py SP500
    
    # Generate plots and detailed analysis
    python scripts/1.5_data_summary.py --all --create-plots --detailed
    
    # Export summary to CSV
    python scripts/1.5_data_summary.py STOXX600 --export-csv

Author: Marleen de Jonge
Date: 2025
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from tqdm import tqdm
import re

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.config import SUPPORTED_INDICES, LOGS_DIR, get_structured_json_folder


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOGS_DIR / 'data_summary.log', mode='a')
        ]
    )


def count_words(text: str) -> int:
    """Count words in text."""
    if not text or pd.isna(text):
        return 0
    return len(str(text).split())


def count_sentences(text: str) -> int:
    """Count sentences in text using robust heuristics."""
    if not text or pd.isna(text):
        return 0
    
    text = str(text).strip()
    
    # Split on sentence endings, but be careful with abbreviations
    sentences = re.split(r'[.!?]+(?=\s+[A-Z]|\s*$)', text)
    
    # Filter empty strings and very short "sentences"
    valid_sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    return len(valid_sentences)


def extract_speaker_statistics(speaker_segments: List[Dict]) -> Dict:
    """Extract statistics from speaker segments."""
    if not speaker_segments:
        return {
            'speaker_count': 0,
            'total_paragraphs': 0,
            'avg_paragraphs_per_speaker': 0,
            'speaker_roles': {},
            'total_words': 0
        }
    
    speaker_counts = Counter()
    role_counts = Counter()
    total_paragraphs = 0
    total_words = 0
    
    for segment in speaker_segments:
        speaker = segment.get('speaker', 'Unknown')
        profession = segment.get('profession', 'Unknown')
        paragraphs = segment.get('paragraphs', [])
        
        speaker_counts[speaker] += 1
        role_counts[profession] += 1
        total_paragraphs += len(paragraphs)
        
        # Count words in paragraphs
        for para in paragraphs:
            total_words += count_words(para)
    
    return {
        'speaker_count': len(speaker_counts),
        'total_paragraphs': total_paragraphs,
        'avg_paragraphs_per_speaker': total_paragraphs / len(speaker_counts) if speaker_counts else 0,
        'speaker_roles': dict(role_counts),
        'total_words': total_words,
        'most_active_speakers': dict(speaker_counts.most_common(5))
    }


def analyze_single_transcript(transcript: Dict) -> Dict:
    """Analyze a single transcript and return statistics."""
    
    # Basic metadata - handle None values
    company_name = transcript.get('company_name') or 'Unknown'
    ticker = transcript.get('ticker') or 'Unknown'
    year = transcript.get('year')
    quarter = transcript.get('quarter') or 'Unknown'
    date = transcript.get('date') or 'Unknown'
    filename = transcript.get('filename') or transcript.get('file') or 'Unknown'
    
    # Convert year to int if possible, otherwise None
    if year is not None:
        try:
            year = int(year)
        except (ValueError, TypeError):
            year = None
    
    # Management section analysis
    mgmt_segments = transcript.get('speaker_segments_management', [])
    mgmt_paragraphs = transcript.get('management_paragraphs', [])
    mgmt_full_text = transcript.get('management_discussion_full', '')
    
    mgmt_stats = extract_speaker_statistics(mgmt_segments)
    mgmt_word_count = count_words(mgmt_full_text) if mgmt_full_text else sum(count_words(p) for p in mgmt_paragraphs)
    mgmt_sentence_count = count_sentences(mgmt_full_text) if mgmt_full_text else sum(count_sentences(p) for p in mgmt_paragraphs)
    
    # Q&A section analysis
    qa_segments = transcript.get('speaker_segments_qa', [])
    qa_paragraphs = transcript.get('qa_paragraphs', [])
    qa_full_text = transcript.get('qa_section_full', '')
    
    qa_stats = extract_speaker_statistics(qa_segments)
    qa_word_count = count_words(qa_full_text) if qa_full_text else sum(count_words(p) for p in qa_paragraphs)
    qa_sentence_count = count_sentences(qa_full_text) if qa_full_text else sum(count_sentences(p) for p in qa_paragraphs)
    
    # Overall statistics
    total_word_count = mgmt_word_count + qa_word_count
    total_sentence_count = mgmt_sentence_count + qa_sentence_count
    total_speakers = mgmt_stats['speaker_count'] + qa_stats['speaker_count']
    
    return {
        # Metadata
        'filename': filename,
        'company_name': company_name,
        'ticker': ticker,
        'year': year,
        'quarter': quarter,
        'date': date,
        
        # Management section
        'mgmt_word_count': mgmt_word_count,
        'mgmt_sentence_count': mgmt_sentence_count,
        'mgmt_paragraph_count': len(mgmt_paragraphs),
        'mgmt_speaker_count': mgmt_stats['speaker_count'],
        'mgmt_speaker_roles': mgmt_stats['speaker_roles'],
        
        # Q&A section
        'qa_word_count': qa_word_count,
        'qa_sentence_count': qa_sentence_count,
        'qa_paragraph_count': len(qa_paragraphs),
        'qa_speaker_count': qa_stats['speaker_count'],
        'qa_speaker_roles': qa_stats['speaker_roles'],
        
        # Overall
        'total_word_count': total_word_count,
        'total_sentence_count': total_sentence_count,
        'total_paragraph_count': len(mgmt_paragraphs) + len(qa_paragraphs),
        'total_speaker_count': total_speakers,
        
        # Ratios
        'mgmt_word_ratio': mgmt_word_count / total_word_count if total_word_count > 0 else 0,
        'qa_word_ratio': qa_word_count / total_word_count if total_word_count > 0 else 0,
        'words_per_sentence': total_word_count / total_sentence_count if total_sentence_count > 0 else 0,
        'sentences_per_paragraph': total_sentence_count / (len(mgmt_paragraphs) + len(qa_paragraphs)) if (len(mgmt_paragraphs) + len(qa_paragraphs)) > 0 else 0
    }


def load_and_analyze_stock_index(stock_index: str) -> Tuple[List[Dict], Dict]:
    """
    Load and analyze all transcripts for a stock index.
    
    Args:
        stock_index: Stock index to analyze
        
    Returns:
        Tuple of (transcript_stats_list, summary_stats)
    """
    logger = logging.getLogger(__name__)
    
    structured_folder = get_structured_json_folder(stock_index)
    
    if not structured_folder.exists():
        raise FileNotFoundError(f"Structured JSON folder not found: {structured_folder}")
    
    # Find all structured JSON files
    json_files = list(structured_folder.glob("structured_calls_*.json"))
    
    if not json_files:
        raise FileNotFoundError(f"No structured JSON files found in: {structured_folder}")
    
    logger.info(f"Found {len(json_files)} structured JSON files for {stock_index}")
    
    all_transcript_stats = []
    
    for json_file in tqdm(json_files, desc=f"Analyzing {stock_index}"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                transcripts = json.load(f)
            
            for transcript in transcripts:
                stats = analyze_single_transcript(transcript)
                stats['source_file'] = json_file.name
                stats['stock_index'] = stock_index
                all_transcript_stats.append(stats)
                
        except Exception as e:
            logger.error(f"Error processing {json_file}: {e}")
            continue
    
    # Calculate summary statistics
    summary_stats = calculate_summary_statistics(all_transcript_stats, stock_index)
    
    logger.info(f"✅ Analyzed {len(all_transcript_stats)} transcripts for {stock_index}")
    
    return all_transcript_stats, summary_stats


def calculate_summary_statistics(transcript_stats: List[Dict], stock_index: str) -> Dict:
    """Calculate comprehensive summary statistics."""
    
    if not transcript_stats:
        return {'error': 'No transcript data available'}
    
    df = pd.DataFrame(transcript_stats)
    
    # Clean up None values that could cause comparison issues
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['quarter'] = df['quarter'].fillna('Unknown')
    df['ticker'] = df['ticker'].fillna('Unknown')
    df['company_name'] = df['company_name'].fillna('Unknown')
    
    # Drop rows with invalid years
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)
    
    if len(df) == 0:
        return {'error': 'No valid transcript data after cleaning'}
    
    # Basic coverage statistics
    total_transcripts = len(df)
    unique_firms = df['ticker'].nunique()
    unique_years = df['year'].nunique()
    date_range = (df['year'].min(), df['year'].max())
    
    # Temporal distribution
    transcripts_by_year = df.groupby('year').size().to_dict()
    transcripts_by_quarter = df.groupby('quarter').size().to_dict()
    firms_by_year = df.groupby('year')['ticker'].nunique().to_dict()
    
    # Company coverage
    transcripts_per_firm = df.groupby('ticker').size()
    company_stats = {
        'mean_transcripts_per_firm': transcripts_per_firm.mean(),
        'median_transcripts_per_firm': transcripts_per_firm.median(),
        'min_transcripts_per_firm': transcripts_per_firm.min(),
        'max_transcripts_per_firm': transcripts_per_firm.max(),
        'firms_with_1_transcript': (transcripts_per_firm == 1).sum(),
        'firms_with_5plus_transcripts': (transcripts_per_firm >= 5).sum(),
        'firms_with_10plus_transcripts': (transcripts_per_firm >= 10).sum()
    }
    
    # Length statistics (word counts)
    length_stats = {
        'total_words': {
            'mean': df['total_word_count'].mean(),
            'median': df['total_word_count'].median(),
            'std': df['total_word_count'].std(),
            'min': df['total_word_count'].min(),
            'max': df['total_word_count'].max(),
            'p25': df['total_word_count'].quantile(0.25),
            'p75': df['total_word_count'].quantile(0.75),
            'p90': df['total_word_count'].quantile(0.90)
        },
        'management_words': {
            'mean': df['mgmt_word_count'].mean(),
            'median': df['mgmt_word_count'].median(),
            'std': df['mgmt_word_count'].std(),
            'min': df['mgmt_word_count'].min(),
            'max': df['mgmt_word_count'].max()
        },
        'qa_words': {
            'mean': df['qa_word_count'].mean(),
            'median': df['qa_word_count'].median(),
            'std': df['qa_word_count'].std(),
            'min': df['qa_word_count'].min(),
            'max': df['qa_word_count'].max()
        }
    }
    
    # Sentence statistics
    sentence_stats = {
        'total_sentences': {
            'mean': df['total_sentence_count'].mean(),
            'median': df['total_sentence_count'].median(),
            'std': df['total_sentence_count'].std()
        },
        'management_sentences': {
            'mean': df['mgmt_sentence_count'].mean(),
            'median': df['mgmt_sentence_count'].median(),
            'std': df['mgmt_sentence_count'].std()
        },
        'qa_sentences': {
            'mean': df['qa_sentence_count'].mean(),
            'median': df['qa_sentence_count'].median(),
            'std': df['qa_sentence_count'].std()
        }
    }
    
    # Speaker statistics
    speaker_stats = {
        'management_speakers': {
            'mean': df['mgmt_speaker_count'].mean(),
            'median': df['mgmt_speaker_count'].median(),
            'std': df['mgmt_speaker_count'].std()
        },
        'qa_speakers': {
            'mean': df['qa_speaker_count'].mean(),
            'median': df['qa_speaker_count'].median(),
            'std': df['qa_speaker_count'].std()
        },
        'total_speakers': {
            'mean': df['total_speaker_count'].mean(),
            'median': df['total_speaker_count'].median(),
            'std': df['total_speaker_count'].std()
        }
    }
    
    # Section balance
    section_balance = {
        'avg_mgmt_word_ratio': df['mgmt_word_ratio'].mean(),
        'avg_qa_word_ratio': df['qa_word_ratio'].mean(),
        'median_mgmt_word_ratio': df['mgmt_word_ratio'].median(),
        'median_qa_word_ratio': df['qa_word_ratio'].median()
    }
    
    # Top companies by transcript count
    top_companies_series = df.groupby(['ticker', 'company_name']).size().nlargest(10)
    top_companies_formatted = [
        {'ticker': k[0], 'company_name': k[1], 'transcript_count': v}
        for k, v in top_companies_series.items()
    ]
    
    # Yearly progression
    yearly_progression = []
    valid_years = sorted([y for y in df['year'].unique() if pd.notna(y)])
    
    for year in valid_years:
        year_data = df[df['year'] == year]
        yearly_progression.append({
            'year': int(year),
            'transcript_count': len(year_data),
            'unique_firms': year_data['ticker'].nunique(),
            'avg_total_words': float(year_data['total_word_count'].mean()),
            'avg_mgmt_words': float(year_data['mgmt_word_count'].mean()),
            'avg_qa_words': float(year_data['qa_word_count'].mean())
        })
    
    return {
        'stock_index': stock_index,
        'dataset_overview': {
            'total_transcripts': total_transcripts,
            'unique_firms': unique_firms,
            'unique_years': unique_years,
            'date_range': date_range,
            'avg_transcripts_per_year': total_transcripts / unique_years if unique_years > 0 else 0
        },
        'temporal_distribution': {
            'transcripts_by_year': transcripts_by_year,
            'transcripts_by_quarter': transcripts_by_quarter,
            'firms_by_year': firms_by_year
        },
        'company_coverage': company_stats,
        'length_statistics': length_stats,
        'sentence_statistics': sentence_stats,
        'speaker_statistics': speaker_stats,
        'section_balance': section_balance,
        'top_companies': top_companies_formatted,
        'yearly_progression': yearly_progression
    }


def create_summary_report(all_stats: Dict[str, Tuple[List[Dict], Dict]], 
                         output_dir: Path) -> None:
    """Create comprehensive summary report."""
    
    # Combine data from all indices
    combined_transcript_stats = []
    combined_summary = {}
    
    for stock_index, stats_tuple in all_stats.items():
        transcript_stats, summary_stats = stats_tuple
        combined_transcript_stats.extend(transcript_stats)
        combined_summary[stock_index] = summary_stats
    
    # Create combined DataFrame
    df_all = pd.DataFrame(combined_transcript_stats)
    
    # Save detailed transcript-level data
    df_all.to_csv(output_dir / 'transcript_level_statistics.csv', index=False)
    df_all.to_parquet(output_dir / 'transcript_level_statistics.parquet')
    
    # Save summary statistics
    with open(output_dir / 'summary_statistics.json', 'w', encoding='utf-8') as f:
        json.dump(combined_summary, f, indent=2, default=str)
    
    # Create aggregated summary across all indices
    overall_summary = create_overall_summary(df_all)
    with open(output_dir / 'overall_summary.json', 'w', encoding='utf-8') as f:
        json.dump(overall_summary, f, indent=2, default=str)
    
    # Create human-readable report
    create_readable_report(combined_summary, overall_summary, output_dir)


def create_overall_summary(df_all: pd.DataFrame) -> Dict:
    """Create overall summary across all stock indices."""
    
    # Clean data first
    df_all = df_all.copy()
    df_all['year'] = pd.to_numeric(df_all['year'], errors='coerce')
    df_all = df_all.dropna(subset=['year'])
    df_all['year'] = df_all['year'].astype(int)
    
    if len(df_all) == 0:
        return {'error': 'No valid data after cleaning'}
    
    return {
        'dataset_overview': {
            'total_transcripts': len(df_all),
            'unique_firms': df_all['ticker'].nunique(),
            'stock_indices': df_all['stock_index'].unique().tolist(),
            'date_range': [int(df_all['year'].min()), int(df_all['year'].max())],
            'total_years': df_all['year'].nunique(),
            'quarters_covered': sorted([q for q in df_all['quarter'].unique() if pd.notna(q) and q != 'Unknown'])
        },
        'content_statistics': {
            'total_words_across_all_transcripts': int(df_all['total_word_count'].sum()),
            'avg_words_per_transcript': float(df_all['total_word_count'].mean()),
            'median_words_per_transcript': float(df_all['total_word_count'].median()),
            'avg_sentences_per_transcript': float(df_all['total_sentence_count'].mean()),
            'avg_paragraphs_per_transcript': float(df_all['total_paragraph_count'].mean())
        },
        'section_breakdown': {
            'avg_management_words': float(df_all['mgmt_word_count'].mean()),
            'avg_qa_words': float(df_all['qa_word_count'].mean()),
            'management_word_share': float(df_all['mgmt_word_ratio'].mean()),
            'qa_word_share': float(df_all['qa_word_ratio'].mean())
        },
        'temporal_trends': {
            'transcripts_per_year': {int(k): int(v) for k, v in df_all.groupby('year').size().items()},
            'avg_length_by_year': {int(k): int(v) for k, v in df_all.groupby('year')['total_word_count'].mean().round().items()},
            'firms_per_year': {int(k): int(v) for k, v in df_all.groupby('year')['ticker'].nunique().items()}
        },
        'top_companies_overall': [
            {'ticker': k[0], 'company_name': k[1], 'transcript_count': int(v)}
            for k, v in df_all.groupby(['ticker', 'company_name']).size().nlargest(15).items()
        ]
    }


def create_readable_report(summary_stats: Dict, overall_summary: Dict, 
                          output_dir: Path) -> None:
    """Create human-readable summary report."""
    
    report_lines = []
    
    # Header
    report_lines.append("=" * 80)
    report_lines.append("ECC TRANSCRIPT DATASET SUMMARY REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Check if overall_summary has error
    if 'error' in overall_summary:
        report_lines.append(f"❌ Error in overall summary: {overall_summary['error']}")
        return
    
    # Overall statistics
    overview = overall_summary['dataset_overview']
    content_stats = overall_summary['content_statistics']
    
    report_lines.append("📊 DATASET OVERVIEW")
    report_lines.append("-" * 30)
    report_lines.append(f"Total Transcripts: {overview['total_transcripts']:,}")
    report_lines.append(f"Unique Firms: {overview['unique_firms']:,}")
    report_lines.append(f"Stock Indices: {', '.join(overview['stock_indices'])}")
    report_lines.append(f"Time Period: {overview['date_range'][0]}-{overview['date_range'][1]} ({overview['total_years']} years)")
    report_lines.append(f"Quarters Covered: {', '.join(sorted(overview['quarters_covered']))}")
    report_lines.append("")
    
    # Content statistics
    report_lines.append("📝 CONTENT STATISTICS")
    report_lines.append("-" * 30)
    report_lines.append(f"Total Words (All Transcripts): {content_stats['total_words_across_all_transcripts']:,}")
    report_lines.append(f"Average Words per Transcript: {content_stats['avg_words_per_transcript']:,.0f}")
    report_lines.append(f"Median Words per Transcript: {content_stats['median_words_per_transcript']:,.0f}")
    report_lines.append(f"Average Sentences per Transcript: {content_stats['avg_sentences_per_transcript']:.0f}")
    report_lines.append(f"Average Paragraphs per Transcript: {content_stats['avg_paragraphs_per_transcript']:.0f}")
    report_lines.append("")
    
    # Section breakdown
    section_stats = overall_summary['section_breakdown']
    report_lines.append("🗣️ SECTION BREAKDOWN")
    report_lines.append("-" * 30)
    report_lines.append(f"Average Management Section Words: {section_stats['avg_management_words']:,.0f}")
    report_lines.append(f"Average Q&A Section Words: {section_stats['avg_qa_words']:,.0f}")
    report_lines.append(f"Management Section Share: {section_stats['management_word_share']:.1%}")
    report_lines.append(f"Q&A Section Share: {section_stats['qa_word_share']:.1%}")
    report_lines.append("")
    
    # By stock index
    for stock_index, index_summary in summary_stats.items():
        if 'error' in index_summary:
            continue
            
        report_lines.append(f"📈 {stock_index} SPECIFIC STATISTICS")
        report_lines.append("-" * 40)
        
        overview = index_summary['dataset_overview']
        length_stats = index_summary['length_statistics']
        
        report_lines.append(f"Transcripts: {overview['total_transcripts']:,}")
        report_lines.append(f"Unique Firms: {overview['unique_firms']:,}")
        report_lines.append(f"Date Range: {overview['date_range'][0]}-{overview['date_range'][1]}")
        report_lines.append(f"Avg Transcripts per Year: {overview['avg_transcripts_per_year']:.1f}")
        report_lines.append("")
        
        # Length distribution
        total_words = length_stats['total_words']
        report_lines.append(f"Word Count Distribution:")
        report_lines.append(f"  Mean: {total_words['mean']:,.0f}")
        report_lines.append(f"  Median: {total_words['median']:,.0f}")
        report_lines.append(f"  25th percentile: {total_words['p25']:,.0f}")
        report_lines.append(f"  75th percentile: {total_words['p75']:,.0f}")
        report_lines.append(f"  90th percentile: {total_words['p90']:,.0f}")
        report_lines.append(f"  Min: {total_words['min']:,.0f}")
        report_lines.append(f"  Max: {total_words['max']:,.0f}")
        report_lines.append("")
        
        # Top companies
        report_lines.append(f"Top 5 Companies by Transcript Count:")
        for i, company in enumerate(index_summary['top_companies'][:5], 1):
            report_lines.append(f"  {i}. {company['ticker']} ({company['company_name']}): {company['transcript_count']} transcripts")
        report_lines.append("")
    
    # Temporal trends
    temporal = overall_summary['temporal_trends']
    report_lines.append("📅 TEMPORAL TRENDS")
    report_lines.append("-" * 30)
    
    # Show yearly progression
    report_lines.append("Transcripts by Year:")
    for year in sorted(temporal['transcripts_per_year'].keys()):
        transcript_count = temporal['transcripts_per_year'][year]
        firm_count = temporal['firms_per_year'][year]
        avg_length = temporal['avg_length_by_year'][year]
        report_lines.append(f"  {year}: {transcript_count:,} transcripts, {firm_count} firms, {avg_length:,.0f} avg words")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # Save report
    with open(output_dir / 'summary_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))


def create_visualizations(all_stats: Dict[str, Tuple[List[Dict], Dict]], 
                         output_dir: Path) -> None:
    """Create data visualizations."""
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create plots directory
        plots_dir = output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Combine all transcript data
        combined_data = []
        for stock_index, stats_tuple in all_stats.items():
            transcript_stats, summary_stats = stats_tuple
            if 'error' not in summary_stats:
                combined_data.extend(transcript_stats)
        
        if not combined_data:
            print("⚠️ No valid data for visualization")
            return
        
        df = pd.DataFrame(combined_data)
        
        # 1. Transcripts by year and stock index
        plt.figure(figsize=(12, 6))
        yearly_counts = df.groupby(['year', 'stock_index']).size().unstack(fill_value=0)
        yearly_counts.plot(kind='bar', stacked=True)
        plt.title('Number of Transcripts by Year and Stock Index')
        plt.xlabel('Year')
        plt.ylabel('Number of Transcripts')
        plt.legend(title='Stock Index')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plots_dir / 'transcripts_by_year.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Word count distribution
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.hist(df['total_word_count'], bins=50, alpha=0.7, color='skyblue')
        plt.title('Total Word Count Distribution')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 3, 2)
        plt.hist(df['mgmt_word_count'], bins=50, alpha=0.7, color='lightgreen')
        plt.title('Management Section Word Count')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 3, 3)
        plt.hist(df['qa_word_count'], bins=50, alpha=0.7, color='lightcoral')
        plt.title('Q&A Section Word Count')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'word_count_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Management vs Q&A word ratio
        plt.figure(figsize=(10, 6))
        plt.scatter(df['mgmt_word_count'], df['qa_word_count'], alpha=0.6, c=df['stock_index'].astype('category').cat.codes)
        plt.xlabel('Management Section Word Count')
        plt.ylabel('Q&A Section Word Count')
        plt.title('Management vs Q&A Section Lengths')
        
        # Add diagonal line
        max_val = max(df['mgmt_word_count'].max(), df['qa_word_count'].max())
        plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Equal Length')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / 'mgmt_vs_qa_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Average transcript length over time
        plt.figure(figsize=(12, 6))
        yearly_avg_length = df.groupby(['year', 'stock_index'])['total_word_count'].mean().unstack()
        yearly_avg_length.plot(kind='line', marker='o')
        plt.title('Average Transcript Length Over Time')
        plt.xlabel('Year')
        plt.ylabel('Average Word Count')
        plt.legend(title='Stock Index')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'avg_length_over_time.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Number of firms over time
        plt.figure(figsize=(12, 6))
        firms_by_year = df.groupby(['year', 'stock_index'])['ticker'].nunique().unstack(fill_value=0)
        firms_by_year.plot(kind='line', marker='s')
        plt.title('Number of Unique Firms by Year')
        plt.xlabel('Year')
        plt.ylabel('Number of Firms')
        plt.legend(title='Stock Index')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'firms_by_year.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Created {len([f for f in plots_dir.glob('*.png')])} visualization plots in: {plots_dir}")
        
    except ImportError:
        print("⚠️ Matplotlib/Seaborn not available - skipping plot creation")
    except Exception as e:
        print(f"⚠️ Error creating plots: {e}")


def export_summary_csv(all_stats: Dict[str, Tuple[List[Dict], Dict]], 
                      output_dir: Path) -> None:
    """Export summary statistics to CSV format for easy analysis."""
    
    # Create yearly summary for each stock index
    yearly_summaries = []
    
    for stock_index, stats_tuple in all_stats.items():
        transcript_stats, summary_stats = stats_tuple
        if 'error' in summary_stats:
            continue
        
        yearly_progression = summary_stats['yearly_progression']
        
        for year_data in yearly_progression:
            year_data['stock_index'] = stock_index
            yearly_summaries.append(year_data)
    
    # Save yearly summary
    if yearly_summaries:
        df_yearly = pd.DataFrame(yearly_summaries)
        df_yearly.to_csv(output_dir / 'yearly_summary.csv', index=False)
    
    # Create firm-level summary
    firm_summaries = []
    
    for stock_index, stats_tuple in all_stats.items():
        transcript_stats, summary_stats = stats_tuple
        if 'error' in summary_stats:
            continue
            
        df_transcripts = pd.DataFrame(transcript_stats)
        
        # Aggregate by firm
        firm_agg = df_transcripts.groupby(['ticker', 'company_name']).agg({
            'total_word_count': ['count', 'mean', 'std', 'min', 'max'],
            'mgmt_word_count': 'mean',
            'qa_word_count': 'mean',
            'total_sentence_count': 'mean',
            'year': ['min', 'max'],
            'stock_index': 'first'
        }).round(2)
        
        # Flatten column names
        firm_agg.columns = ['_'.join(col).strip() for col in firm_agg.columns.values]
        firm_agg = firm_agg.reset_index()
        
        # Add to overall list
        for _, row in firm_agg.iterrows():
            firm_summaries.append(row.to_dict())
    
    # Save firm summary
    df_firms = pd.DataFrame(firm_summaries)
    df_firms.to_csv(output_dir / 'firm_level_summary.csv', index=False)
    
    print(f"📤 Exported CSV summaries to {output_dir}")


def print_summary_report(all_stats: Dict[str, Tuple[List[Dict], Dict]]) -> None:
    """Print a concise summary report to console."""
    
    print("\n" + "=" * 80)
    print("📊 ECC TRANSCRIPT DATASET SUMMARY")
    print("=" * 80)
    
    # Overall statistics
    total_transcripts = 0
    total_firms = set()
    all_years = set()
    
    for stock_index, stats_tuple in all_stats.items():
        transcript_stats, summary_stats = stats_tuple
        if 'error' in summary_stats:
            print(f"❌ {stock_index}: {summary_stats['error']}")
            continue
        
        overview = summary_stats['dataset_overview']
        length_stats = summary_stats['length_statistics']
        
        print(f"\n📈 {stock_index}")
        print("-" * 40)
        print(f"  Transcripts: {overview['total_transcripts']:,}")
        print(f"  Unique Firms: {overview['unique_firms']:,}")
        print(f"  Date Range: {overview['date_range'][0]}-{overview['date_range'][1]}")
        print(f"  Avg Words per Transcript: {length_stats['total_words']['mean']:,.0f}")
        print(f"  Avg Management Words: {length_stats['management_words']['mean']:,.0f}")
        print(f"  Avg Q&A Words: {length_stats['qa_words']['mean']:,.0f}")
        
        # Track overall totals
        total_transcripts += overview['total_transcripts']
        # Add firms from this index (tickers might overlap between indices)
        for stats in transcript_stats:
            total_firms.add(f"{stats['ticker']}_{stock_index}")
        all_years.update(range(overview['date_range'][0], overview['date_range'][1] + 1))
    
    print(f"\n🌍 COMBINED STATISTICS")
    print("-" * 40)
    print(f"  Total Transcripts: {total_transcripts:,}")
    print(f"  Total Firm-Index Combinations: {len(total_firms):,}")
    print(f"  Years Covered: {len(all_years)} years ({min(all_years)}-{max(all_years)})")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Generate comprehensive descriptive statistics for ECC transcript data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze both SP500 and STOXX600
    python scripts/1.5_data_summary.py --all
    
    # Analyze SP500 only with detailed output
    python scripts/1.5_data_summary.py SP500 --detailed
    
    # Create visualizations and export to CSV
    python scripts/1.5_data_summary.py --all --create-plots --export-csv
    
    # Quick summary without detailed analysis
    python scripts/1.5_data_summary.py STOXX600 --quick
        """
    )
    
    parser.add_argument(
        'stock_index',
        nargs='?',
        choices=SUPPORTED_INDICES,
        help='Stock index to analyze (SP500 or STOXX600)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Analyze all available stock indices'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path("outputs/data_summary"),
        help='Output directory for summary files (default: outputs/data_summary)'
    )
    
    parser.add_argument(
        '--create-plots',
        action='store_true',
        help='Create visualization plots (requires matplotlib/seaborn)'
    )
    
    parser.add_argument(
        '--export-csv',
        action='store_true',
        help='Export summary statistics to CSV format'
    )
    
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Generate detailed analysis with speaker statistics'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick summary without detailed analysis'
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
    
    print("📊 ECC Transcript Data Summary")
    print("=" * 50)
    print(f"Stock indices: {', '.join(stock_indices)}")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        all_stats = {}
        
        for stock_index in stock_indices:
            print(f"\n🔍 Analyzing {stock_index}...")
            
            try:
                transcript_stats, summary_stats = load_and_analyze_stock_index(stock_index)
                all_stats[stock_index] = (transcript_stats, summary_stats)
                
                overview = summary_stats['dataset_overview']
                print(f"✅ {stock_index}: {overview['total_transcripts']:,} transcripts, {overview['unique_firms']} firms")
                
            except Exception as e:
                logger.error(f"Error analyzing {stock_index}: {e}")
                print(f"❌ Failed to analyze {stock_index}: {e}")
                all_stats[stock_index] = ([], {'error': str(e)})
                continue
        
        if not any('error' not in stats[1] for stats in all_stats.values()):
            print("❌ No data could be analyzed. Check that structured JSON files exist.")
            return
        
        # Print summary to console
        print_summary_report(all_stats)
        
        if not args.quick:
            # Create comprehensive reports
            print(f"\n📝 Creating detailed summary reports...")
            create_summary_report(all_stats, args.output_dir)
            
            # Export to CSV if requested
            if args.export_csv:
                print(f"📤 Exporting to CSV format...")
                export_summary_csv(all_stats, args.output_dir)
            
            # Create visualizations if requested
            if args.create_plots:
                print(f"📊 Creating visualizations...")
                create_visualizations(all_stats, args.output_dir)
        
        print(f"\n✅ Data summary completed!")
        print(f"📁 Output files saved to: {args.output_dir}")
        
        if not args.quick:
            print(f"\nGenerated files:")
            print(f"  • summary_report.txt - Human-readable summary")
            print(f"  • transcript_level_statistics.csv - Detailed transcript data")
            print(f"  • summary_statistics.json - Complete statistics")
            print(f"  • overall_summary.json - Cross-index summary")
            
            if args.export_csv:
                print(f"  • yearly_summary.csv - Year-by-year breakdown")
                print(f"  • firm_level_summary.csv - Firm-level aggregations")
            
            if args.create_plots:
                print(f"  • plots/ - Visualization charts")
    
    except KeyboardInterrupt:
        print("\n🛑 Analysis interrupted by user")
        return
    except Exception as e:
        logger.error(f"❌ Error during analysis: {e}")
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return


if __name__ == "__main__":
    main()