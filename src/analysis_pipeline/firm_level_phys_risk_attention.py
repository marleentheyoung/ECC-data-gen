#!/usr/bin/env python3
"""
Firm-level climate physical risk attention calculator using semantic search.

This script calculates firm-quarter climate physical risk attention measures by 
applying semantic search to identify physical risk-related discussions in earnings calls.
Creates panel data suitable for econometric analysis and event studies.

Based on Sautner et al. (2023) framework but leverages new NLP techniques such as 
transformers and semantic search to capture nuanced physical risk discussions.

Usage:
    python scripts/3_agg_variables/firm_level_physical_risk_attention.py

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
            logging.FileHandler(LOGS_DIR / 'firm_level_physical_risk_attention.log', mode='a')
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


class PhysicalRiskAnalyzer(BaseSemanticSearcher):
    """
    Semantic searcher for climate physical risk discussions in earnings calls.
    
    This analyzer focuses on both acute and chronic physical climate risks:
    - Acute risks: extreme weather events, natural disasters, supply chain disruption
    - Chronic risks: long-term climate changes, sea level rise, temperature shifts
    """
    
    def __init__(self, index_path: Path, config_path: Optional[str] = None):
        """Initialize physical risk analyzer with semantic search capabilities."""
        super().__init__(index_path, config_path)
        
        # Physical risk queries based on your research framework
        self.acute_physical_risk_queries = [
            'extreme weather climate physical risk operational disruption supply chain',
            'hurricanes floods droughts wildfires extreme weather events business operations',
            'natural disasters climate-related infrastructure damage facility disruption',
            'severe weather operational resilience climate adaptation physical assets',
            'acute physical risks climate hazards business continuity emergency',
            'weather-related supply chain disruption logistics transportation',
            'storm damage flooding wildfire operational impact infrastructure',
            'hurricane typhoon cyclone business operations facility damage',
            'drought water scarcity operational constraints production disruption',
            'wildfire smoke air quality operational health safety impacts',
            'flooding facility damage equipment infrastructure water damage',
            'extreme precipitation heavy rainfall operational flooding disruption',
            'heat waves extreme temperatures operational efficiency workforce productivity',
            'cold snaps winter storms energy disruption operational challenges'
        ]
        
        self.chronic_physical_risk_queries = [
            'rising sea levels temperature changes precipitation patterns long-term climate',
            'chronic physical risks climate change operational locations infrastructure adaptation',
            'long-term climate trends physical environment changing weather patterns',
            'climate change adaptation infrastructure resilience chronic physical risks',
            'sea level rise coastal operations long-term climate impacts',
            'temperature increases heat stress operational efficiency cooling costs',
            'changing precipitation water availability drought risk water security',
            'chronic climate change long-term business operations infrastructure',
            'climate adaptation resilience infrastructure investment long-term planning',
            'water stress drought risk long-term water availability operations',
            'coastal flooding sea level rise facility relocation adaptation',
            'ecosystem changes biodiversity loss supply chain agricultural impacts',
            'permafrost thaw arctic operations infrastructure stability risks',
            'desertification land degradation agricultural supply chain impacts'
        ]
        
        # Combined physical risk queries for comprehensive analysis
        self.physical_risk_queries = self.acute_physical_risk_queries + self.chronic_physical_risk_queries
        
        # Set similarity threshold based on your research (0.40 for physical risk)
        self.similarity_threshold = 0.40
        
    def find_physical_risk_snippets(self) -> Dict[str, List[ClimateSnippet]]:
        """Find all physical risk-related snippets using semantic search."""
        logger = logging.getLogger(__name__)
        
        logger.info(f"🌪️ Searching for physical risk snippets with {len(self.physical_risk_queries)} queries")
        logger.info(f"   - {len(self.acute_physical_risk_queries)} acute physical risk queries")
        logger.info(f"   - {len(self.chronic_physical_risk_queries)} chronic physical risk queries")
        logger.info(f"Using similarity threshold: {self.similarity_threshold}")
        
        # Perform semantic search
        physical_risk_snippets = self.multi_query_search(
            queries=self.physical_risk_queries,
            top_k=10000,  # Large number to get comprehensive results
            min_score=self.similarity_threshold,
            aggregation='max'
        )
        
        logger.info(f"Found {len(physical_risk_snippets)} physical risk-related snippets")
        
        # Group snippets by earnings call
        snippets_by_call = {}
        for snippet in physical_risk_snippets:
            call_key = f"{snippet.ticker}_{snippet.year}_{snippet.quarter}"
            if call_key not in snippets_by_call:
                snippets_by_call[call_key] = []
            snippets_by_call[call_key].append(snippet)
        
        return snippets_by_call
    
    def find_acute_physical_risk_snippets(self) -> Dict[str, List[ClimateSnippet]]:
        """Find acute physical risk snippets specifically."""
        logger = logging.getLogger(__name__)
        
        logger.info(f"⚡ Searching for acute physical risk snippets")
        
        acute_snippets = self.multi_query_search(
            queries=self.acute_physical_risk_queries,
            top_k=10000,
            min_score=self.similarity_threshold,
            aggregation='max'
        )
        
        # Group by call
        snippets_by_call = {}
        for snippet in acute_snippets:
            call_key = f"{snippet.ticker}_{snippet.year}_{snippet.quarter}"
            if call_key not in snippets_by_call:
                snippets_by_call[call_key] = []
            snippets_by_call[call_key].append(snippet)
        
        return snippets_by_call
    
    def find_chronic_physical_risk_snippets(self) -> Dict[str, List[ClimateSnippet]]:
        """Find chronic physical risk snippets specifically."""
        logger = logging.getLogger(__name__)
        
        logger.info(f"🌊 Searching for chronic physical risk snippets")
        
        chronic_snippets = self.multi_query_search(
            queries=self.chronic_physical_risk_queries,
            top_k=10000,
            min_score=self.similarity_threshold,
            aggregation='max'
        )
        
        # Group by call
        snippets_by_call = {}
        for snippet in chronic_snippets:
            call_key = f"{snippet.ticker}_{snippet.year}_{snippet.quarter}"
            if call_key not in snippets_by_call:
                snippets_by_call[call_key] = []
            snippets_by_call[call_key].append(snippet)
        
        return snippets_by_call


def create_firm_level_physical_risk_panel(enhanced_snippets_path: Path, 
                                         structured_path: Path,
                                         stock_indices: List[str],
                                         semantic_index_path: Path) -> pd.DataFrame:
    """
    Create firm-level climate physical risk attention panel dataset using semantic search.
    
    Returns:
        DataFrame with firm-quarter climate physical risk attention measures
    """
    logger = logging.getLogger(__name__)
    
    # Initialize semantic searcher
    logger.info(f"🌪️ Initializing physical risk semantic search from {semantic_index_path}")
    physical_risk_analyzer = PhysicalRiskAnalyzer(semantic_index_path)
    
    # Find different types of physical risk snippets
    all_physical_risk_snippets = physical_risk_analyzer.find_physical_risk_snippets()
    acute_risk_snippets = physical_risk_analyzer.find_acute_physical_risk_snippets()
    chronic_risk_snippets = physical_risk_analyzer.find_chronic_physical_risk_snippets()
    
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
                    
                    # Check for physical risk content using semantic search results
                    call_key = f"{ticker}_{year}_{quarter}"
                    
                    # All physical risk snippets
                    all_physical_snippets = all_physical_risk_snippets.get(call_key, [])
                    acute_snippets = acute_risk_snippets.get(call_key, [])
                    chronic_snippets = chronic_risk_snippets.get(call_key, [])
                    
                    # Count sentences for different risk types
                    total_physical_risk_sentences = len(all_physical_snippets)
                    acute_physical_risk_sentences = len(acute_snippets)
                    chronic_physical_risk_sentences = len(chronic_snippets)
                    
                    # Calculate similarity statistics for all physical risk
                    if all_physical_snippets:
                        all_similarity_scores = [s.similarity_score for s in all_physical_snippets]
                        max_similarity = max(all_similarity_scores)
                        mean_similarity = np.mean(all_similarity_scores)
                        min_similarity = min(all_similarity_scores)
                    else:
                        max_similarity = 0.0
                        mean_similarity = 0.0
                        min_similarity = 0.0
                    
                    # Calculate similarity statistics for acute risk
                    if acute_snippets:
                        acute_similarity_scores = [s.similarity_score for s in acute_snippets]
                        acute_max_similarity = max(acute_similarity_scores)
                        acute_mean_similarity = np.mean(acute_similarity_scores)
                    else:
                        acute_max_similarity = 0.0
                        acute_mean_similarity = 0.0
                    
                    # Calculate similarity statistics for chronic risk
                    if chronic_snippets:
                        chronic_similarity_scores = [s.similarity_score for s in chronic_snippets]
                        chronic_max_similarity = max(chronic_similarity_scores)
                        chronic_mean_similarity = np.mean(chronic_similarity_scores)
                    else:
                        chronic_max_similarity = 0.0
                        chronic_mean_similarity = 0.0
                    
                    # Calculate physical risk attention ratios
                    physical_risk_ratio = 0.0
                    acute_risk_ratio = 0.0
                    chronic_risk_ratio = 0.0
                    
                    if total_sentences and total_sentences > 0:
                        physical_risk_ratio = total_physical_risk_sentences / total_sentences
                        acute_risk_ratio = acute_physical_risk_sentences / total_sentences
                        chronic_risk_ratio = chronic_physical_risk_sentences / total_sentences
                    
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
                        
                        # Physical risk measures (overall)
                        'climate_physical_risk_ratio': physical_risk_ratio,
                        'physical_risk_sentence_count': total_physical_risk_sentences,
                        'physical_risk_snippet_count': len(all_physical_snippets),
                        'has_physical_risk_content': len(all_physical_snippets) > 0,
                        
                        # Acute physical risk measures
                        'acute_physical_risk_ratio': acute_risk_ratio,
                        'acute_risk_sentence_count': acute_physical_risk_sentences,
                        'acute_risk_snippet_count': len(acute_snippets),
                        'has_acute_risk_content': len(acute_snippets) > 0,
                        
                        # Chronic physical risk measures  
                        'chronic_physical_risk_ratio': chronic_risk_ratio,
                        'chronic_risk_sentence_count': chronic_physical_risk_sentences,
                        'chronic_risk_snippet_count': len(chronic_snippets),
                        'has_chronic_risk_content': len(chronic_snippets) > 0,
                        
                        # Base transcript information
                        'total_sentences_in_call': total_sentences,
                        
                        # Similarity measures (overall physical risk)
                        'max_similarity_score': max_similarity,
                        'mean_similarity_score': mean_similarity,
                        'min_similarity_score': min_similarity,
                        
                        # Similarity measures (acute risk)
                        'acute_max_similarity_score': acute_max_similarity,
                        'acute_mean_similarity_score': acute_mean_similarity,
                        
                        # Similarity measures (chronic risk)
                        'chronic_max_similarity_score': chronic_max_similarity,
                        'chronic_mean_similarity_score': chronic_mean_similarity,
                        
                        # Technical parameters
                        'similarity_threshold': physical_risk_analyzer.similarity_threshold,
                        
                        # Coverage indicators
                        'physical_risk_coverage_binary': 1 if len(all_physical_snippets) > 0 else 0,
                        'acute_risk_coverage_binary': 1 if len(acute_snippets) > 0 else 0,
                        'chronic_risk_coverage_binary': 1 if len(chronic_snippets) > 0 else 0,
                        
                        # Log transformations
                        'log_physical_risk_sentences': np.log(total_physical_risk_sentences + 1),
                        'log_acute_risk_sentences': np.log(acute_physical_risk_sentences + 1),
                        'log_chronic_risk_sentences': np.log(chronic_physical_risk_sentences + 1),
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
    
    # Apply same data quality filters
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
    
    logger.info(f"✅ Created physical risk panel with {len(df)} firm-quarter observations")
    logger.info(f"   {df['ticker'].nunique()} unique firms")
    logger.info(f"   {df['quarter_id'].nunique()} unique quarters")
    
    return df


def add_lagged_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged climate physical risk attention variables for econometric analysis."""
    
    df = df.sort_values(['ticker', 'year', 'quarter'])
    
    # Create lagged variables (useful for event studies)
    for lag in [1, 2, 4]:  # 1, 2, and 4 quarters
        df[f'climate_physical_risk_ratio_lag{lag}'] = df.groupby('ticker')['climate_physical_risk_ratio'].shift(lag)
        df[f'has_physical_risk_content_lag{lag}'] = df.groupby('ticker')['has_physical_risk_content'].shift(lag)
        df[f'acute_physical_risk_ratio_lag{lag}'] = df.groupby('ticker')['acute_physical_risk_ratio'].shift(lag)
        df[f'chronic_physical_risk_ratio_lag{lag}'] = df.groupby('ticker')['chronic_physical_risk_ratio'].shift(lag)
    
    # Create moving averages (useful for smoothing)
    for window in [2, 4]:
        df[f'climate_physical_risk_ratio_ma{window}'] = (
            df.groupby('ticker')['climate_physical_risk_ratio']
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        
        df[f'acute_physical_risk_ratio_ma{window}'] = (
            df.groupby('ticker')['acute_physical_risk_ratio']
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        
        df[f'chronic_physical_risk_ratio_ma{window}'] = (
            df.groupby('ticker')['chronic_physical_risk_ratio']
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
    
    return df


def create_summary_statistics(df: pd.DataFrame) -> Dict:
    """Create comprehensive summary statistics for physical risk attention."""
    
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
        
        'physical_risk_attention_distribution': {
            'overall': {
                'mean': float(df['climate_physical_risk_ratio'].mean()),
                'median': float(df['climate_physical_risk_ratio'].median()),
                'std': float(df['climate_physical_risk_ratio'].std()),
                'min': float(df['climate_physical_risk_ratio'].min()),
                'max': float(df['climate_physical_risk_ratio'].max()),
                'p25': float(df['climate_physical_risk_ratio'].quantile(0.25)),
                'p75': float(df['climate_physical_risk_ratio'].quantile(0.75)),
                'p90': float(df['climate_physical_risk_ratio'].quantile(0.90)),
                'p99': float(df['climate_physical_risk_ratio'].quantile(0.99))
            },
            'acute_risk': {
                'mean': float(df['acute_physical_risk_ratio'].mean()),
                'median': float(df['acute_physical_risk_ratio'].median()),
                'std': float(df['acute_physical_risk_ratio'].std()),
                'max': float(df['acute_physical_risk_ratio'].max())
            },
            'chronic_risk': {
                'mean': float(df['chronic_physical_risk_ratio'].mean()),
                'median': float(df['chronic_physical_risk_ratio'].median()),
                'std': float(df['chronic_physical_risk_ratio'].std()),
                'max': float(df['chronic_physical_risk_ratio'].max())
            }
        },
        
        'coverage_statistics': {
            'pct_calls_with_physical_risk': float(df['has_physical_risk_content'].mean() * 100),
            'pct_calls_with_acute_risk': float(df['has_acute_risk_content'].mean() * 100),
            'pct_calls_with_chronic_risk': float(df['has_chronic_risk_content'].mean() * 100),
            'avg_physical_risk_sentences_per_call': float(df['physical_risk_sentence_count'].mean()),
            'avg_acute_risk_sentences_per_call': float(df['acute_risk_sentence_count'].mean()),
            'avg_chronic_risk_sentences_per_call': float(df['chronic_risk_sentence_count'].mean()),
            'avg_total_sentences_per_call': float(df['total_sentences_in_call'].mean()),
            'avg_physical_risk_snippets_per_call': float(df['physical_risk_snippet_count'].mean())
        },
        
        'similarity_statistics': {
            'mean_max_similarity': float(df['max_similarity_score'].mean()),
            'mean_avg_similarity': float(df['mean_similarity_score'].mean()),
            'acute_mean_max_similarity': float(df['acute_max_similarity_score'].mean()),
            'chronic_mean_max_similarity': float(df['chronic_max_similarity_score'].mean()),
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
                'avg_physical_risk_attention': float(region_data['climate_physical_risk_ratio'].mean()),
                'avg_acute_risk_attention': float(region_data['acute_physical_risk_ratio'].mean()),
                'avg_chronic_risk_attention': float(region_data['chronic_physical_risk_ratio'].mean()),
                'physical_risk_coverage_pct': float(region_data['has_physical_risk_content'].mean() * 100),
                'acute_risk_coverage_pct': float(region_data['has_acute_risk_content'].mean() * 100),
                'chronic_risk_coverage_pct': float(region_data['has_chronic_risk_content'].mean() * 100)
            }
    
    return summary_stats


def save_firm_level_results(df: pd.DataFrame, summary_stats: Dict, output_dir: Path):
    """Save firm-level physical risk results in multiple formats."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Main panel dataset
    df.to_csv(output_dir / 'firm_level_physical_risk_attention.csv', index=False)
    df.to_parquet(output_dir / 'firm_level_physical_risk_attention.parquet')
    
    # Summary statistics
    with open(output_dir / 'firm_level_physical_risk_summary_statistics.json', 'w') as f:
        json.dump(summary_stats, f, indent=2, default=str)
    
    # Risk type subsets
    acute_risk_df = df[df['has_acute_risk_content'] == True].copy()
    chronic_risk_df = df[df['has_chronic_risk_content'] == True].copy()
    
    if len(acute_risk_df) > 0:
        acute_risk_df.to_csv(output_dir / 'firm_level_acute_physical_risk_attention.csv', index=False)
    
    if len(chronic_risk_df) > 0:
        chronic_risk_df.to_csv(output_dir / 'firm_level_chronic_physical_risk_attention.csv', index=False)
    
    # Regional subsets
    for region in ['US', 'EU']:
        region_data = df[df['region'] == region]
        if len(region_data) > 0:
            region_data.to_csv(output_dir / f'firm_level_physical_risk_attention_{region.lower()}.csv', index=False)
    
    # Stata format for econometric analysis
    try:
        # Create Stata-friendly variable names (max 32 chars, no special characters)
        df_stata = df.copy()
        stata_name_mapping = {
            'climate_physical_risk_ratio': 'physrisk_ratio',
            'acute_physical_risk_ratio': 'acuterisk_ratio', 
            'chronic_physical_risk_ratio': 'chronrisk_ratio',
            'physical_risk_sentence_count': 'physrisk_sentences',
            'acute_risk_sentence_count': 'acuterisk_sentences',
            'chronic_risk_sentence_count': 'chronrisk_sentences',
            'physical_risk_snippet_count': 'physrisk_snippets',
            'acute_risk_snippet_count': 'acuterisk_snippets',
            'chronic_risk_snippet_count': 'chronrisk_snippets',
            'has_physical_risk_content': 'has_physrisk',
            'has_acute_risk_content': 'has_acuterisk',
            'has_chronic_risk_content': 'has_chronrisk',
            'total_sentences_in_call': 'total_sentences',
            'max_similarity_score': 'max_similarity',
            'mean_similarity_score': 'mean_similarity',
            'acute_max_similarity_score': 'acute_max_sim',
            'chronic_max_similarity_score': 'chron_max_sim',
            'physical_risk_coverage_binary': 'physrisk_binary',
            'acute_risk_coverage_binary': 'acuterisk_binary',
            'chronic_risk_coverage_binary': 'chronrisk_binary'
        }
        
        # Rename columns for Stata compatibility
        for old_name, new_name in stata_name_mapping.items():
            if old_name in df_stata.columns:
                df_stata = df_stata.rename(columns={old_name: new_name})
        
        # Ensure remaining column names are Stata-friendly
        df_stata.columns = [col.replace('_', '').lower()[:32] for col in df_stata.columns]
        
        df_stata.to_stata(
            output_dir / 'firm_level_physical_risk_attention.dta',
            write_index=False,
            version=117
        )
    except Exception as e:
        # Fallback to CSV if Stata export fails
        df.to_csv(output_dir / 'firm_level_physical_risk_attention_for_stata.csv', index=False)
        print(f"⚠️ Stata export failed, saved CSV instead: {e}")
    
    # Create balanced panel indicator
    firm_quarters = df.groupby('ticker').size()
    max_quarters = firm_quarters.max()
    balanced_firms = firm_quarters[firm_quarters == max_quarters].index
    
    df['balanced_panel'] = df['ticker'].isin(balanced_firms)
    balanced_df = df[df['balanced_panel']].copy()
    
    if len(balanced_df) > 0:
        balanced_df.to_csv(output_dir / 'firm_level_physical_risk_attention_balanced.csv', index=False)
    
    print(f"💾 Results saved to: {output_dir}")
    print(f"📊 Panel structure:")
    print(f"   • {len(df):,} firm-quarter observations")
    print(f"   • {df['ticker'].nunique():,} unique firms")
    print(f"   • {df['quarter_id'].nunique()} unique quarters")
    print(f"   • {len(balanced_df):,} observations in balanced panel")
    print(f"🌪️ Physical risk coverage:")
    print(f"   • {df['has_physical_risk_content'].sum():,} calls with any physical risk content ({df['has_physical_risk_content'].mean()*100:.1f}%)")
    print(f"   • {df['has_acute_risk_content'].sum():,} calls with acute risk content ({df['has_acute_risk_content'].mean()*100:.1f}%)")
    print(f"   • {df['has_chronic_risk_content'].sum():,} calls with chronic risk content ({df['has_chronic_risk_content'].mean()*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Create firm-level climate physical risk attention panel dataset using semantic search',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with default paths
    python firm_level_physical_risk_attention.py
    
    # Specify custom paths and indices
    python firm_level_physical_risk_attention.py \\
        --enhanced-snippets-path data/enhanced_climate_snippets \\
        --semantic-index-path data/semantic_indexes/combined \\
        --output-dir outputs/firm_level_panel/physical_risk \\
        --stock-indices SP500 STOXX600 --verbose
    
    # Process only S&P 500 firms
    python firm_level_physical_risk_attention.py \\
        --stock-indices SP500 --verbose

This script creates comprehensive physical risk attention measures including:
- Overall physical risk attention (acute + chronic)
- Acute physical risk attention (extreme weather, disasters)
- Chronic physical risk attention (long-term climate changes)
- Similarity scores and coverage statistics
- Lagged variables for econometric analysis
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
        help='Path to semantic index for physical risk search'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path("outputs/firm_level_panel/physical_risk"),
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
    
    print("🌪️ Firm-Level Climate Physical Risk Attention Panel Creator")
    print("=" * 60)
    print(f"📍 Based on Sautner et al. (2023) framework with semantic search enhancements")
    print(f"🔍 Analyzing both acute and chronic physical climate risks")
    print()
    
    try:
        # Validate inputs
        if not args.semantic_index_path.exists():
            raise FileNotFoundError(f"Semantic index path not found: {args.semantic_index_path}")
        
        if not args.enhanced_snippets_path.exists():
            raise FileNotFoundError(f"Enhanced snippets path not found: {args.enhanced_snippets_path}")
        
        # Create firm-level physical risk panel
        print("🔄 Creating firm-level physical risk attention panel...")
        df = create_firm_level_physical_risk_panel(
            args.enhanced_snippets_path,
            args.structured_transcripts_path,
            args.stock_indices,
            args.semantic_index_path
        )
        
        if df.empty:
            print("❌ No firm-level physical risk data created!")
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
        
        print(f"\n✅ Firm-level climate physical risk attention panel completed!")
        print(f"📁 Key output files:")
        print(f"   • firm_level_physical_risk_attention.csv - Main panel dataset")
        print(f"   • firm_level_physical_risk_attention.dta - Stata format")
        print(f"   • firm_level_physical_risk_attention_balanced.csv - Balanced panel")
        print(f"   • firm_level_acute_physical_risk_attention.csv - Acute risk subset")
        print(f"   • firm_level_chronic_physical_risk_attention.csv - Chronic risk subset")
        print(f"   • firm_level_physical_risk_summary_statistics.json - Summary stats")
        print()
        print(f"🔬 Research applications:")
        print(f"   • Event studies around extreme weather events")
        print(f"   • Cross-sectional analysis of physical risk exposure")
        print(f"   • Time-series analysis of physical risk attention evolution")
        print(f"   • Regional comparisons (US vs EU) of physical risk discourse")
        print(f"   • Validation against actual climate damages and insurance claims")
        
    except Exception as e:
        logger.error(f"❌ Error during processing: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()