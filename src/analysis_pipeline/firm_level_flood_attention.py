#!/usr/bin/env python3
"""
Firm-level climate flood attention calculator using semantic search.

This script calculates firm-quarter climate flood attention measures by 
applying semantic search to identify flood-related discussions in earnings calls.
Creates panel data suitable for econometric analysis and event studies.

Includes annotation framework for threshold validation using human expert judgment.

Usage:
    # Create annotation sample for human validation
    python scripts/3_agg_variables/firm_level_flood_attention.py --create-annotation-sample
    
    # Validate threshold after human annotation
    python scripts/3_agg_variables/firm_level_flood_attention.py --validate-threshold annotation_sample_completed.csv
    
    # Run main analysis
    python scripts/3_agg_variables/firm_level_flood_attention.py

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
import random
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import semantic search components
try:
    from src.analysis_pipeline.base_searcher import BaseSemanticSearcher, ClimateSnippet
except ImportError:
    # Fallback for development
    sys.path.append(str(Path(__file__).parent.parent / "semantic_search"))
    from base_searcher import BaseSemanticSearcher, ClimateSnippet

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
            logging.FileHandler(LOGS_DIR / 'firm_level_flood_attention.log', mode='a')
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


class FloodAnalyzer(BaseSemanticSearcher):
    """
    Semantic searcher for climate flood discussions in earnings calls.
    Enhanced with annotation and threshold validation capabilities.
    """
    
    def __init__(self, index_path: Path, config_path: Optional[str] = None):
        """Initialize flood analyzer with semantic search capabilities."""
        super().__init__(index_path, config_path)
        
        self.flood_queries = [
            # Physical risk
            "flooding storm surge heavy rainfall extreme precipitation inundation",
            "hurricane typhoon cyclone river flooding coastal flooding flash flood",
            "sea level rise tidal flooding coastal erosion water intrusion",

            # Business / operations impact
            "flood damage to facilities disruption of operations supply chain interruptions",
            "insurance costs premiums flood risk underwriting losses catastrophe exposure",
            "infrastructure damage transportation delays logistics disruption flooding",

            # Adaptation / resilience
            "flood resilience adaptation strategies flood defenses levees sea walls",
            "drainage improvements stormwater management flood-proof infrastructure",
            "relocation of assets flood risk mitigation business continuity planning",

            # Financial framing
            "catastrophe bonds insurance linked securities flood risk transfer",
            "flood-related write-downs stranded assets impaired valuations",
            "liability exposure litigation due to flooding inadequate preparation",

            # Sector-specific
            "agriculture crop losses due to flooding waterlogging harvest disruption",
            "utilities power outages water infrastructure damage flooding",
            "real estate commercial property exposure to coastal flooding",
            "ports shipping logistics hubs vulnerable to flood risk"
        ]
        
        self.similarity_threshold = 0.42
        
    def find_flood_snippets(self) -> Dict[str, List[ClimateSnippet]]:
        """Find all flood-related snippets using semantic search."""
        logger = logging.getLogger(__name__)
        
        logger.info(f"🔍 Searching for flood snippets with {len(self.flood_queries)} queries")
        logger.info(f"Using similarity threshold: {self.similarity_threshold}")
        
        # Perform semantic search
        flood_snippets = self.multi_query_search(
            queries=self.flood_queries,
            top_k=10000,  # Large number to get comprehensive results
            min_score=self.similarity_threshold,
            aggregation='max'
        )
        
        logger.info(f"Found {len(flood_snippets)} flood-related snippets")
        
        # Group snippets by earnings call
        snippets_by_call = {}
        for snippet in flood_snippets:
            call_key = f"{snippet.ticker}_{snippet.year}_{snippet.quarter}"
            if call_key not in snippets_by_call:
                snippets_by_call[call_key] = []
            snippets_by_call[call_key].append(snippet)
        
        return snippets_by_call
    
    def collect_annotation_samples(self, sample_size: int = 400) -> List[ClimateSnippet]:
        """
        Collect stratified sample of snippets for human annotation.
        Returns snippets across similarity score ranges for threshold validation.
        """
        logger = logging.getLogger(__name__)
        
        logger.info(f"🔬 Collecting annotation samples (n={sample_size})")
        
        # Get ALL snippets (not just above threshold) with very low threshold
        all_snippets = self.multi_query_search(
            queries=self.flood_queries,
            top_k=50000,  # Large number to capture everything
            min_score=0.15,  # Very low threshold to capture everything
            aggregation='max'
        )
        
        logger.info(f"Found {len(all_snippets)} total snippets for sampling")
        
        # Stratify by similarity scores
        annotation_samples = []
        
        # High confidence (clearly relevant) - 40% of sample
        high_confidence = [s for s in all_snippets if s.similarity_score >= 0.50]
        if high_confidence:
            n_high = min(len(high_confidence), int(sample_size * 0.4))
            annotation_samples.extend(random.sample(high_confidence, n_high))
            logger.info(f"Sampled {n_high} high confidence snippets (≥0.50)")
        
        # Medium confidence (around threshold) - 40% of sample  
        medium_confidence = [s for s in all_snippets if 0.35 <= s.similarity_score < 0.50]
        if medium_confidence:
            n_medium = min(len(medium_confidence), int(sample_size * 0.4))
            annotation_samples.extend(random.sample(medium_confidence, n_medium))
            logger.info(f"Sampled {n_medium} medium confidence snippets (0.35-0.50)")
        
        # Low confidence (likely irrelevant) - 20% of sample
        low_confidence = [s for s in all_snippets if 0.15 <= s.similarity_score < 0.35]
        if low_confidence:
            n_low = min(len(low_confidence), int(sample_size * 0.2))
            annotation_samples.extend(random.sample(low_confidence, n_low))
            logger.info(f"Sampled {n_low} low confidence snippets (0.15-0.35)")
        
        logger.info(f"✅ Total annotation samples collected: {len(annotation_samples)}")
        
        return annotation_samples
    
    def export_for_annotation(self, annotation_samples: List[ClimateSnippet], output_path: Path) -> pd.DataFrame:
        """Export snippets for human annotation with proper formatting."""
        logger = logging.getLogger(__name__)
        
        annotation_data = []
        
        for i, snippet in enumerate(annotation_samples):
            annotation_data.append({
                'snippet_id': f"{snippet.ticker}_{snippet.year}_{snippet.quarter}_{i:04d}",
                'text': snippet.text,
                'ticker': snippet.ticker,
                'company_name': getattr(snippet, 'company_name', ''),
                'year': snippet.year,
                'quarter': snippet.quarter,
                'similarity_score': round(snippet.similarity_score, 4),
                'model_prediction': 1 if snippet.similarity_score >= self.similarity_threshold else 0,
                'confidence_category': self._get_confidence_category(snippet.similarity_score),
                
                # Fields for human annotation (leave empty)
                'human_relevance': '',  # Instructions: 0=Not relevant, 1=Slightly relevant, 2=Relevant, 3=Highly relevant
                'human_binary': '',     # Instructions: 0=Not flood-related, 1=flood-related
                'annotator_id': '',     # Instructions: Enter your annotator ID
                'confidence': '',       # Instructions: 1=Very uncertain to 5=Very certain
                'notes': ''            # Instructions: Optional notes about borderline cases
            })
        
        df = pd.DataFrame(annotation_data)
        
        # Add instruction header row
        instructions = {
            'snippet_id': 'INSTRUCTIONS',
            'text': 'Rate each snippet: human_relevance (0-3 scale), human_binary (0/1), confidence (1-5), add notes if needed',
            'ticker': 'DELETE_THIS_ROW',
            'company_name': 'AFTER_READING',
            'year': 'THE_INSTRUCTIONS',
            'quarter': '',
            'similarity_score': 0.0,
            'model_prediction': 0,
            'confidence_category': 'EXAMPLE',
            'human_relevance': '0-3',
            'human_binary': '0/1',
            'annotator_id': 'YOUR_ID',
            'confidence': '1-5',
            'notes': 'Optional comments'
        }
        
        df_with_instructions = pd.concat([pd.DataFrame([instructions]), df], ignore_index=True)
        
        # Save with instructions
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_with_instructions.to_csv(output_path, index=False)
        
        # Also save annotation guidelines
        guidelines_path = output_path.parent / 'annotation_guidelines.txt'
        self._create_annotation_guidelines(guidelines_path)
        
        logger.info(f"💾 Annotation sample exported to: {output_path}")
        logger.info(f"📋 Annotation guidelines saved to: {guidelines_path}")
        
        return df
    
    def _get_confidence_category(self, similarity_score: float) -> str:
        """Categorize confidence based on similarity score."""
        if similarity_score >= 0.50:
            return 'high'
        elif similarity_score >= 0.35:
            return 'medium'
        else:
            return 'low'

def create_firm_level_flood_panel(enhanced_snippets_path: Path, 
                                       structured_path: Path,
                                       stock_indices: List[str],
                                       semantic_index_path: Path) -> pd.DataFrame:
    """
    Create firm-level climate flood attention panel dataset using semantic search.
    
    Returns:
        DataFrame with firm-quarter climate flood attention measures
    """
    logger = logging.getLogger(__name__)
    
    # Initialize semantic searcher
    logger.info(f"🔍 Initializing semantic search from {semantic_index_path}")
    flood_analyzer = FloodAnalyzer(semantic_index_path)
    
    # Find flood snippets using semantic search
    flood_snippets_by_call = flood_analyzer.find_flood_snippets()
    
    all_firm_records = []
    
    for stock_index in stock_indices:
        logger.info(f"Processing {stock_index}...")
        
        # Load enhanced climate snippets for base structure
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
                    
                    # Get total sentences from base transcript
                    total_sentences = transcript.get('total_sentences_in_call')
                    
                    # Check for flood content using semantic search results
                    call_key = f"{ticker}_{year}_{quarter}"
                    flood_snippets = flood_snippets_by_call.get(call_key, [])
                    
                    # Count flood sentences from semantic search
                    flood_sentence_count = len(flood_snippets)
                    
                    # Calculate similarity statistics
                    if flood_snippets:
                        similarity_scores = [s.similarity_score for s in flood_snippets]
                        max_similarity = max(similarity_scores)
                        mean_similarity = np.mean(similarity_scores)
                        min_similarity = min(similarity_scores)
                    else:
                        max_similarity = 0.0
                        mean_similarity = 0.0
                        min_similarity = 0.0
                    
                    # Calculate flood attention ratio
                    climate_flood_ratio = 0.0
                    if total_sentences and total_sentences > 0:
                        climate_flood_ratio = flood_sentence_count / total_sentences
                    
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
                        
                        # Climate flood measures
                        'climate_flood_ratio': climate_flood_ratio,
                        'flood_sentence_count': flood_sentence_count,
                        'total_sentences_in_call': total_sentences,
                        'flood_snippet_count': len(flood_snippets),
                        'has_flood_content': len(flood_snippets) > 0,
                        
                        # Similarity measures
                        'max_similarity_score': max_similarity,
                        'mean_similarity_score': mean_similarity,
                        'min_similarity_score': min_similarity,
                        'similarity_threshold': flood_analyzer.similarity_threshold,
                        
                        # Coverage indicators
                        'flood_coverage_binary': 1 if len(flood_snippets) > 0 else 0,
                        'log_flood_sentences': np.log(flood_sentence_count + 1),
                        'log_total_sentences': np.log(total_sentences + 1) if total_sentences else None
                    }
                    
                    all_firm_records.append(record)
                    
            except Exception as e:
                logger.error(f"Error processing {json_file}: {e}")
                continue
    
    # Convert to DataFrame
    df = pd.DataFrame(all_firm_records)
    
    if df.empty:
        logger.error("No firm records created!")
        return df
    
    # Apply same data quality filters as climate attention
    df = df[df['ticker'].notna() & (df['ticker'] != '')]
    df = df[df['year'].notna()]
    
    # Create extended ticker column with -US suffix for US companies
    df['ISSUER_TICKER'] = df['ticker']
    us_mask = df['region'] == 'US'
    df.loc[us_mask, 'ISSUER_TICKER'] = df.loc[us_mask, 'ticker'] + '-US'
    
    # Create quarterly identifier
    df['quarter_id'] = df['year'].astype(str) + df['quarter'].astype(str)
    
    # Sort by firm and time
    df = df.sort_values(['ticker', 'year', 'quarter']).reset_index(drop=True)
    
    logger.info(f"✅ Created flood panel with {len(df)} firm-quarter observations")
    logger.info(f"   {df['ticker'].nunique()} unique firms")
    logger.info(f"   {df['quarter_id'].nunique()} unique quarters")
    
    return df


def add_lagged_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged climate flood attention variables for econometric analysis."""
    
    df = df.sort_values(['ticker', 'year', 'quarter'])
    
    # Create lagged variables (useful for event studies)
    for lag in [1, 2, 4]:  # 1, 2, and 4 quarters
        df[f'climate_flood_ratio_lag{lag}'] = df.groupby('ticker')['climate_flood_ratio'].shift(lag)
        df[f'has_flood_content_lag{lag}'] = df.groupby('ticker')['has_flood_content'].shift(lag)
    
    # Create moving averages (useful for smoothing)
    for window in [2, 4]:
        df[f'climate_flood_ratio_ma{window}'] = (
            df.groupby('ticker')['climate_flood_ratio']
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
        
        'flood_attention_distribution': {
            'mean': float(df['climate_flood_ratio'].mean()),
            'median': float(df['climate_flood_ratio'].median()),
            'std': float(df['climate_flood_ratio'].std()),
            'min': float(df['climate_flood_ratio'].min()),
            'max': float(df['climate_flood_ratio'].max()),
            'p25': float(df['climate_flood_ratio'].quantile(0.25)),
            'p75': float(df['climate_flood_ratio'].quantile(0.75)),
            'p90': float(df['climate_flood_ratio'].quantile(0.90)),
            'p99': float(df['climate_flood_ratio'].quantile(0.99))
        },
        
        'coverage_statistics': {
            'pct_calls_with_flood': float(df['has_flood_content'].mean() * 100),
            'avg_flood_sentences_per_call': float(df['flood_sentence_count'].mean()),
            'avg_total_sentences_per_call': float(df['total_sentences_in_call'].mean()),
            'avg_flood_snippets_per_call': float(df['flood_snippet_count'].mean())
        },
        
        'similarity_statistics': {
            'mean_max_similarity': float(df['max_similarity_score'].mean()),
            'mean_avg_similarity': float(df['mean_similarity_score'].mean()),
            'similarity_threshold_used': float(df['similarity_threshold'].iloc[0])
        }
    }
    
    # Regional breakdown
    for region in ['US', 'EU']:
        region_data = df[df['region'] == region]
        if len(region_data) > 0:
            summary_stats[f'{region}_statistics'] = {
                'observations': len(region_data),
                'firms': region_data['ticker'].nunique(),
                'avg_flood_attention': float(region_data['climate_flood_ratio'].mean()),
                'flood_coverage_pct': float(region_data['has_flood_content'].mean() * 100)
            }
    
    return summary_stats


def save_firm_level_results(df: pd.DataFrame, summary_stats: Dict, output_dir: Path):
    """Save firm-level results in multiple formats."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Main panel dataset
    df.to_csv(output_dir / 'firm_level_flood_attention.csv', index=False)
    df.to_parquet(output_dir / 'firm_level_flood_attention.parquet')
    
    # Summary statistics
    with open(output_dir / 'firm_level_flood_summary_statistics.json', 'w') as f:
        json.dump(summary_stats, f, indent=2, default=str)
    
    # Regional subsets
    for region in ['US', 'EU']:
        region_data = df[df['region'] == region]
        if len(region_data) > 0:
            region_data.to_csv(output_dir / f'firm_level_flood_attention_{region.lower()}.csv', index=False)
    
    # Stata format for econometric analysis
    try:
        # Create Stata-friendly variable names
        df_stata = df.copy()
        df_stata.columns = [col.replace('_', '').lower()[:32] for col in df_stata.columns]
        
        df_stata.to_stata(
            output_dir / 'firm_level_flood_attention.dta',
            write_index=False,
            version=117
        )
    except Exception as e:
        # Fallback
        df.to_csv(output_dir / 'firm_level_flood_attention_for_stata.csv', index=False)
    
    # Create balanced panel indicator
    firm_quarters = df.groupby('ticker').size()
    max_quarters = firm_quarters.max()
    balanced_firms = firm_quarters[firm_quarters == max_quarters].index
    
    df['balanced_panel'] = df['ticker'].isin(balanced_firms)
    balanced_df = df[df['balanced_panel']].copy()
    
    if len(balanced_df) > 0:
        balanced_df.to_csv(output_dir / 'firm_level_flood_attention_balanced.csv', index=False)
    
    print(f"💾 Results saved to: {output_dir}")
    print(f"📊 Panel structure:")
    print(f"   • {len(df):,} firm-quarter observations")
    print(f"   • {df['ticker'].nunique():,} unique firms")
    print(f"   • {df['quarter_id'].nunique()} unique quarters")
    print(f"   • {len(balanced_df):,} observations in balanced panel")


def main():
    parser = argparse.ArgumentParser(
        description='Create firm-level climate flood attention panel dataset using semantic search',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create annotation sample for threshold validation
  python firm_level_flood_attention.py --create-annotation-sample
  
  # Validate threshold after human annotation
  python firm_level_flood_attention.py --validate-threshold annotation_sample_completed.csv
  
  # Run main analysis with default settings
  python firm_level_flood_attention.py
        """
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
        '--semantic-index-path',
        type=Path,
        default=Path("data/semantic_indexes/combined"),
        help='Path to semantic index for flood search'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path("outputs/firm_level_panel/flood"),
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
    
    # Annotation and validation arguments
    parser.add_argument(
        '--create-annotation-sample',
        action='store_true',
        help='Create sample for human annotation (for threshold validation)'
    )
    
    parser.add_argument(
        '--annotation-sample-size',
        type=int,
        default=400,
        help='Size of annotation sample (default: 400)'
    )
    
    parser.add_argument(
        '--validate-threshold',
        type=Path,
        help='Path to completed annotation CSV file for threshold validation'
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    print("🌱 Firm-Level Climate flood Attention Panel Creator")
    print("=" * 55)
    
    try:
        # MODE 1: Create annotation sample
        if args.create_annotation_sample:
            print("📋 Creating annotation sample for threshold validation...")
            
            # Initialize flood analyzer
            flood_analyzer = FloodAnalyzer(args.semantic_index_path)
            
            # Collect annotation samples
            annotation_samples = flood_analyzer.collect_annotation_samples(args.annotation_sample_size)
            
            # Export for annotation
            annotation_output = args.output_dir / 'annotation' / 'annotation_sample.csv'
            flood_analyzer.export_for_annotation(annotation_samples, annotation_output)
            
            print(f"\n✅ Annotation sample created successfully!")
            print(f"📁 Files created:")
            print(f"   • {annotation_output}")
            print(f"   • {annotation_output.parent / 'annotation_guidelines.txt'}")
            print(f"\n📋 Next steps:")
            print(f"   1. Review annotation guidelines")
            print(f"   2. Have experts annotate the CSV file")
            print(f"   3. Run validation: --validate-threshold annotation_sample_completed.csv")
            
            return
        
        # MODE 2: Validate threshold using human annotations
        if args.validate_threshold:
            print(f"🎯 Validating threshold using annotations from {args.validate_threshold}")
            
            # Initialize flood analyzer
            flood_analyzer = FloodAnalyzer(args.semantic_index_path)
            
            # Calculate threshold performance
            validation_output_dir = args.output_dir / 'validation'
            results_df, optimal_threshold = flood_analyzer.calculate_threshold_performance(
                args.validate_threshold, 
                validation_output_dir
            )
            
            print(f"\n✅ Threshold validation completed!")
            print(f"📁 Results saved to: {validation_output_dir}")
            print(f"📊 Key findings:")
            print(f"   • Current threshold: {flood_analyzer.similarity_threshold}")
            print(f"   • Optimal threshold: {optimal_threshold:.3f}")
            
            if abs(optimal_threshold - flood_analyzer.similarity_threshold) > 0.05:
                print(f"⚠️  Significant difference detected!")
                print(f"   Consider updating similarity_threshold in floodAnalyzer class")
            else:
                print(f"✅ Current threshold is well-calibrated")
            
            return
        
        # MODE 3: Main analysis - create firm-level panel
        print("🔍 Running main analysis to create firm-level flood panel...")
        
        # Create firm-level flood panel
        df = create_firm_level_flood_panel(
            args.enhanced_snippets_path,
            args.structured_transcripts_path,
            args.stock_indices,
            args.semantic_index_path
        )
        
        if df.empty:
            print("❌ No firm-level flood data created!")
            return
        
        # Add lagged variables
        print("📈 Adding lagged variables for econometric analysis...")
        df = add_lagged_variables(df)
        
        # Create summary statistics
        print("📊 Generating summary statistics...")
        summary_stats = create_summary_statistics(df)
        
        # Save results
        print("💾 Saving results...")
        save_firm_level_results(df, summary_stats, args.output_dir)
        
        print(f"\n✅ Firm-level climate flood attention panel completed!")
        print(f"📁 Key output files:")
        print(f"   • firm_level_flood_attention.csv - Main panel dataset")
        print(f"   • firm_level_flood_attention.dta - Stata format")
        print(f"   • firm_level_flood_attention_balanced.csv - Balanced panel")
        print(f"   • firm_level_flood_summary_statistics.json - Summary stats")
        
        # Display key statistics
        print(f"\n📊 Panel overview:")
        print(f"   • {len(df):,} firm-quarter observations")
        print(f"   • {df['ticker'].nunique():,} unique firms")
        print(f"   • {df['quarter_id'].nunique()} unique quarters")
        print(f"   • {df['has_flood_content'].mean():.1%} of calls contain flood content")
        print(f"   • Mean flood attention: {df['climate_flood_ratio'].mean():.4f}")
        
        # Recommend next steps
        print(f"\n🎯 Recommended next steps:")
        print(f"   1. Validate methodology: --create-annotation-sample")
        print(f"   2. Review summary statistics and distributions")
        print(f"   3. Run econometric analysis using the generated panel data")
        
    except Exception as e:
        logger.error(f"❌ Error during processing: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        raise


if __name__ == "__main__":
    main()