#!/usr/bin/env python3
"""
Firm-level climate transition risk attention calculator using semantic search.

This script calculates firm-quarter climate transition risk attention measures by 
applying semantic search to identify transition risk-related discussions in earnings calls.
Creates panel data suitable for econometric analysis and event studies.

Based on Sautner et al. (2023) framework but leverages new NLP techniques such as 
transformers and semantic search to capture nuanced transition risk discussions.

Transition risks are categorized into three main types:
1. Policy/Regulatory transition risks (carbon pricing, climate regulations)
2. Technology transition risks (stranded assets, technological disruption)  
3. Market transition risks (consumer preferences, reputation risks)

Usage:
    python scripts/3_agg_variables/firm_level_transition_risk_attention.py

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
            logging.FileHandler(LOGS_DIR / 'firm_level_transition_risk_attention.log', mode='a')
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


class TransitionRiskAnalyzer(BaseSemanticSearcher):
    """
    Semantic searcher for climate transition risk discussions in earnings calls.
    
    This analyzer focuses on three main types of transition risks:
    1. Policy/Regulatory transition risks - carbon pricing, climate regulations, compliance costs
    2. Technology transition risks - stranded assets, technological disruption, obsolescence
    3. Market transition risks - consumer preferences, reputation risks, demand shifts
    """
    
    def __init__(self, index_path: Path, config_path: Optional[str] = None):
        """Initialize transition risk analyzer with semantic search capabilities."""
        super().__init__(index_path, config_path)
        
        # Policy/Regulatory transition risk queries based on your research framework
        self.policy_regulatory_risk_queries = [
            'climate regulations carbon pricing policy transition risk regulatory changes',
            'climate policy regulatory compliance environmental regulations transition costs',
            'carbon regulations climate legislation regulatory transition risk compliance',
            'environmental policy compliance climate regulatory landscape transition burden',
            'regulatory transition risks climate policy uncertainty regulatory costs',
            'climate regulation compliance costs policy implementation transition expense',
            'environmental regulatory changes climate transition policy compliance burden',
            'climate policy development regulatory compliance transition regulatory risk',
            'carbon pricing mechanisms regulatory transition costs carbon tax burden',
            'emissions trading system regulatory compliance transition risk costs',
            'climate disclosure requirements regulatory transition compliance costs',
            'Paris Agreement regulatory compliance transition risk policy burden',
            'green taxonomy regulatory compliance transition costs policy risk',
            'climate litigation legal risk regulatory transition compliance liability'
        ]
        
        # Technology transition risk queries
        self.technology_transition_risk_queries = [
            'clean technology renewable energy transition technological disruption competitive risk',
            'energy transition technology obsolescence stranded assets technological transition risk',
            'low-carbon technology renewable energy competitive disadvantage technology transition',
            'clean tech innovation energy transformation technological transition risk disruption',
            'technological disruption clean energy competitiveness technology obsolescence risk',
            'energy technology transition renewable energy adoption technological risk',
            'clean technology development technological obsolescence transition risk',
            'technology transition risks clean energy innovation competitive threat',
            'stranded assets carbon intensive business model technological transition risk',
            'fossil fuel asset impairment devaluation stranded technological risk',
            'business model transition technological change disruption competitive risk',
            'automation digitalization technology transition operational disruption risk',
            'electric vehicles technology transition automotive industry disruption',
            'renewable energy technology disruption fossil fuel competitive risk'
        ]
        
        # Market transition risk queries
        self.market_transition_risk_queries = [
            'consumer preferences demand shift sustainable products market transition risk',
            'market transition risk customer behavior sustainable products reputation risk',
            'changing consumer demand climate-conscious consumers market shifts reputation',
            'sustainable products market transition brand reputation climate preferences risk',
            'market demand sustainable products green consumer preferences transition risk',
            'climate-conscious consumers market transition reputation risk brand impact',
            'sustainable market trends consumer behavior climate awareness reputation risk',
            'green products market demand climate consumer preferences transition risk',
            'ESG rating reputation risk climate performance market transition risk',
            'climate reputation risk brand image market transition consumer perception',
            'sustainability reputation risk market transition consumer expectations',
            'greenwashing reputation risk market transition climate credibility',
            'climate litigation reputation risk market transition legal exposure',
            'climate activism investor pressure market transition reputation risk'
        ]
        
        # Combined transition risk queries for comprehensive analysis
        self.transition_risk_queries = (self.policy_regulatory_risk_queries + 
                                      self.technology_transition_risk_queries + 
                                      self.market_transition_risk_queries)
        
        # Set similarity threshold based on your research (0.40 for transition risk)
        self.similarity_threshold = 0.40
        
    def find_transition_risk_snippets(self) -> Dict[str, List[ClimateSnippet]]:
        """Find all transition risk-related snippets using semantic search."""
        logger = logging.getLogger(__name__)
        
        logger.info(f"⚡ Searching for transition risk snippets with {len(self.transition_risk_queries)} queries")
        logger.info(f"   - {len(self.policy_regulatory_risk_queries)} policy/regulatory risk queries")
        logger.info(f"   - {len(self.technology_transition_risk_queries)} technology transition risk queries")
        logger.info(f"   - {len(self.market_transition_risk_queries)} market transition risk queries")
        logger.info(f"Using similarity threshold: {self.similarity_threshold}")
        
        # Perform semantic search
        transition_risk_snippets = self.multi_query_search(
            queries=self.transition_risk_queries,
            top_k=10000,  # Large number to get comprehensive results
            min_score=self.similarity_threshold,
            aggregation='max'
        )
        
        logger.info(f"Found {len(transition_risk_snippets)} transition risk-related snippets")
        
        # Group snippets by earnings call
        snippets_by_call = {}
        for snippet in transition_risk_snippets:
            call_key = f"{snippet.ticker}_{snippet.year}_{snippet.quarter}"
            if call_key not in snippets_by_call:
                snippets_by_call[call_key] = []
            snippets_by_call[call_key].append(snippet)
        
        return snippets_by_call
    
    def find_policy_regulatory_risk_snippets(self) -> Dict[str, List[ClimateSnippet]]:
        """Find policy/regulatory transition risk snippets specifically."""
        logger = logging.getLogger(__name__)
        
        logger.info(f"📜 Searching for policy/regulatory transition risk snippets")
        
        policy_snippets = self.multi_query_search(
            queries=self.policy_regulatory_risk_queries,
            top_k=10000,
            min_score=self.similarity_threshold,
            aggregation='max'
        )
        
        # Group by call
        snippets_by_call = {}
        for snippet in policy_snippets:
            call_key = f"{snippet.ticker}_{snippet.year}_{snippet.quarter}"
            if call_key not in snippets_by_call:
                snippets_by_call[call_key] = []
            snippets_by_call[call_key].append(snippet)
        
        return snippets_by_call
    
    def find_technology_transition_risk_snippets(self) -> Dict[str, List[ClimateSnippet]]:
        """Find technology transition risk snippets specifically."""
        logger = logging.getLogger(__name__)
        
        logger.info(f"🔬 Searching for technology transition risk snippets")
        
        tech_snippets = self.multi_query_search(
            queries=self.technology_transition_risk_queries,
            top_k=10000,
            min_score=self.similarity_threshold,
            aggregation='max'
        )
        
        # Group by call
        snippets_by_call = {}
        for snippet in tech_snippets:
            call_key = f"{snippet.ticker}_{snippet.year}_{snippet.quarter}"
            if call_key not in snippets_by_call:
                snippets_by_call[call_key] = []
            snippets_by_call[call_key].append(snippet)
        
        return snippets_by_call
    
    def find_market_transition_risk_snippets(self) -> Dict[str, List[ClimateSnippet]]:
        """Find market transition risk snippets specifically."""
        logger = logging.getLogger(__name__)
        
        logger.info(f"🏪 Searching for market transition risk snippets")
        
        market_snippets = self.multi_query_search(
            queries=self.market_transition_risk_queries,
            top_k=10000,
            min_score=self.similarity_threshold,
            aggregation='max'
        )
        
        # Group by call
        snippets_by_call = {}
        for snippet in market_snippets:
            call_key = f"{snippet.ticker}_{snippet.year}_{snippet.quarter}"
            if call_key not in snippets_by_call:
                snippets_by_call[call_key] = []
            snippets_by_call[call_key].append(snippet)
        
        return snippets_by_call


def create_firm_level_transition_risk_panel(enhanced_snippets_path: Path, 
                                           structured_path: Path,
                                           stock_indices: List[str],
                                           semantic_index_path: Path) -> pd.DataFrame:
    """
    Create firm-level climate transition risk attention panel dataset using semantic search.
    
    Returns:
        DataFrame with firm-quarter climate transition risk attention measures
    """
    logger = logging.getLogger(__name__)
    
    # Initialize semantic searcher
    logger.info(f"⚡ Initializing transition risk semantic search from {semantic_index_path}")
    transition_risk_analyzer = TransitionRiskAnalyzer(semantic_index_path)
    
    # Find different types of transition risk snippets
    all_transition_risk_snippets = transition_risk_analyzer.find_transition_risk_snippets()
    policy_risk_snippets = transition_risk_analyzer.find_policy_regulatory_risk_snippets()
    technology_risk_snippets = transition_risk_analyzer.find_technology_transition_risk_snippets()
    market_risk_snippets = transition_risk_analyzer.find_market_transition_risk_snippets()
    
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
                    
                    # Check for transition risk content using semantic search results
                    call_key = f"{ticker}_{year}_{quarter}"
                    
                    # All transition risk snippets
                    all_transition_snippets = all_transition_risk_snippets.get(call_key, [])
                    policy_snippets = policy_risk_snippets.get(call_key, [])
                    technology_snippets = technology_risk_snippets.get(call_key, [])
                    market_snippets = market_risk_snippets.get(call_key, [])
                    
                    # Count sentences for different risk types
                    total_transition_risk_sentences = len(all_transition_snippets)
                    policy_risk_sentences = len(policy_snippets)
                    technology_risk_sentences = len(technology_snippets)
                    market_risk_sentences = len(market_snippets)
                    
                    # Calculate similarity statistics for all transition risk
                    if all_transition_snippets:
                        all_similarity_scores = [s.similarity_score for s in all_transition_snippets]
                        max_similarity = max(all_similarity_scores)
                        mean_similarity = np.mean(all_similarity_scores)
                        min_similarity = min(all_similarity_scores)
                    else:
                        max_similarity = 0.0
                        mean_similarity = 0.0
                        min_similarity = 0.0
                    
                    # Calculate similarity statistics for policy risk
                    if policy_snippets:
                        policy_similarity_scores = [s.similarity_score for s in policy_snippets]
                        policy_max_similarity = max(policy_similarity_scores)
                        policy_mean_similarity = np.mean(policy_similarity_scores)
                    else:
                        policy_max_similarity = 0.0
                        policy_mean_similarity = 0.0
                    
                    # Calculate similarity statistics for technology risk
                    if technology_snippets:
                        tech_similarity_scores = [s.similarity_score for s in technology_snippets]
                        tech_max_similarity = max(tech_similarity_scores)
                        tech_mean_similarity = np.mean(tech_similarity_scores)
                    else:
                        tech_max_similarity = 0.0
                        tech_mean_similarity = 0.0
                    
                    # Calculate similarity statistics for market risk
                    if market_snippets:
                        market_similarity_scores = [s.similarity_score for s in market_snippets]
                        market_max_similarity = max(market_similarity_scores)
                        market_mean_similarity = np.mean(market_similarity_scores)
                    else:
                        market_max_similarity = 0.0
                        market_mean_similarity = 0.0
                    
                    # Calculate transition risk attention ratios
                    transition_risk_ratio = 0.0
                    policy_risk_ratio = 0.0
                    technology_risk_ratio = 0.0
                    market_risk_ratio = 0.0
                    
                    if total_sentences and total_sentences > 0:
                        transition_risk_ratio = total_transition_risk_sentences / total_sentences
                        policy_risk_ratio = policy_risk_sentences / total_sentences
                        technology_risk_ratio = technology_risk_sentences / total_sentences
                        market_risk_ratio = market_risk_sentences / total_sentences
                    
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
                        
                        # Transition risk measures (overall)
                        'climate_transition_risk_ratio': transition_risk_ratio,
                        'transition_risk_sentence_count': total_transition_risk_sentences,
                        'transition_risk_snippet_count': len(all_transition_snippets),
                        'has_transition_risk_content': len(all_transition_snippets) > 0,
                        
                        # Policy/regulatory transition risk measures
                        'policy_transition_risk_ratio': policy_risk_ratio,
                        'policy_risk_sentence_count': policy_risk_sentences,
                        'policy_risk_snippet_count': len(policy_snippets),
                        'has_policy_risk_content': len(policy_snippets) > 0,
                        
                        # Technology transition risk measures  
                        'technology_transition_risk_ratio': technology_risk_ratio,
                        'technology_risk_sentence_count': technology_risk_sentences,
                        'technology_risk_snippet_count': len(technology_snippets),
                        'has_technology_risk_content': len(technology_snippets) > 0,
                        
                        # Market transition risk measures
                        'market_transition_risk_ratio': market_risk_ratio,
                        'market_risk_sentence_count': market_risk_sentences,
                        'market_risk_snippet_count': len(market_snippets),
                        'has_market_risk_content': len(market_snippets) > 0,
                        
                        # Base transcript information
                        'total_sentences_in_call': total_sentences,
                        
                        # Similarity measures (overall transition risk)
                        'max_similarity_score': max_similarity,
                        'mean_similarity_score': mean_similarity,
                        'min_similarity_score': min_similarity,
                        
                        # Similarity measures (policy risk)
                        'policy_max_similarity_score': policy_max_similarity,
                        'policy_mean_similarity_score': policy_mean_similarity,
                        
                        # Similarity measures (technology risk)
                        'technology_max_similarity_score': tech_max_similarity,
                        'technology_mean_similarity_score': tech_mean_similarity,
                        
                        # Similarity measures (market risk)
                        'market_max_similarity_score': market_max_similarity,
                        'market_mean_similarity_score': market_mean_similarity,
                        
                        # Technical parameters
                        'similarity_threshold': transition_risk_analyzer.similarity_threshold,
                        
                        # Coverage indicators
                        'transition_risk_coverage_binary': 1 if len(all_transition_snippets) > 0 else 0,
                        'policy_risk_coverage_binary': 1 if len(policy_snippets) > 0 else 0,
                        'technology_risk_coverage_binary': 1 if len(technology_snippets) > 0 else 0,
                        'market_risk_coverage_binary': 1 if len(market_snippets) > 0 else 0,
                        
                        # Log transformations
                        'log_transition_risk_sentences': np.log(total_transition_risk_sentences + 1),
                        'log_policy_risk_sentences': np.log(policy_risk_sentences + 1),
                        'log_technology_risk_sentences': np.log(technology_risk_sentences + 1),
                        'log_market_risk_sentences': np.log(market_risk_sentences + 1),
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
    
    logger.info(f"✅ Created transition risk panel with {len(df)} firm-quarter observations")
    logger.info(f"   {df['ticker'].nunique()} unique firms")
    logger.info(f"   {df['quarter_id'].nunique()} unique quarters")
    
    return df


def add_lagged_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged climate transition risk attention variables for econometric analysis."""
    
    df = df.sort_values(['ticker', 'year', 'quarter'])
    
    # Create lagged variables (useful for event studies)
    for lag in [1, 2, 4]:  # 1, 2, and 4 quarters
        df[f'climate_transition_risk_ratio_lag{lag}'] = df.groupby('ticker')['climate_transition_risk_ratio'].shift(lag)
        df[f'has_transition_risk_content_lag{lag}'] = df.groupby('ticker')['has_transition_risk_content'].shift(lag)
        df[f'policy_transition_risk_ratio_lag{lag}'] = df.groupby('ticker')['policy_transition_risk_ratio'].shift(lag)
        df[f'technology_transition_risk_ratio_lag{lag}'] = df.groupby('ticker')['technology_transition_risk_ratio'].shift(lag)
        df[f'market_transition_risk_ratio_lag{lag}'] = df.groupby('ticker')['market_transition_risk_ratio'].shift(lag)
    
    # Create moving averages (useful for smoothing)
    for window in [2, 4]:
        df[f'climate_transition_risk_ratio_ma{window}'] = (
            df.groupby('ticker')['climate_transition_risk_ratio']
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        
        df[f'policy_transition_risk_ratio_ma{window}'] = (
            df.groupby('ticker')['policy_transition_risk_ratio']
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        
        df[f'technology_transition_risk_ratio_ma{window}'] = (
            df.groupby('ticker')['technology_transition_risk_ratio']
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        
        df[f'market_transition_risk_ratio_ma{window}'] = (
            df.groupby('ticker')['market_transition_risk_ratio']
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
    
    return df


def create_summary_statistics(df: pd.DataFrame) -> Dict:
    """Create comprehensive summary statistics for transition risk attention."""
    
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
        
        'transition_risk_attention_distribution': {
            'overall': {
                'mean': float(df['climate_transition_risk_ratio'].mean()),
                'median': float(df['climate_transition_risk_ratio'].median()),
                'std': float(df['climate_transition_risk_ratio'].std()),
                'min': float(df['climate_transition_risk_ratio'].min()),
                'max': float(df['climate_transition_risk_ratio'].max()),
                'p25': float(df['climate_transition_risk_ratio'].quantile(0.25)),
                'p75': float(df['climate_transition_risk_ratio'].quantile(0.75)),
                'p90': float(df['climate_transition_risk_ratio'].quantile(0.90)),
                'p99': float(df['climate_transition_risk_ratio'].quantile(0.99))
            },
            'policy_risk': {
                'mean': float(df['policy_transition_risk_ratio'].mean()),
                'median': float(df['policy_transition_risk_ratio'].median()),
                'std': float(df['policy_transition_risk_ratio'].std()),
                'max': float(df['policy_transition_risk_ratio'].max())
            },
            'technology_risk': {
                'mean': float(df['technology_transition_risk_ratio'].mean()),
                'median': float(df['technology_transition_risk_ratio'].median()),
                'std': float(df['technology_transition_risk_ratio'].std()),
                'max': float(df['technology_transition_risk_ratio'].max())
            },
            'market_risk': {
                'mean': float(df['market_transition_risk_ratio'].mean()),
                'median': float(df['market_transition_risk_ratio'].median()),
                'std': float(df['market_transition_risk_ratio'].std()),
                'max': float(df['market_transition_risk_ratio'].max())
            }
        },
        
        'coverage_statistics': {
            'pct_calls_with_transition_risk': float(df['has_transition_risk_content'].mean() * 100),
            'pct_calls_with_policy_risk': float(df['has_policy_risk_content'].mean() * 100),
            'pct_calls_with_technology_risk': float(df['has_technology_risk_content'].mean() * 100),
            'pct_calls_with_market_risk': float(df['has_market_risk_content'].mean() * 100),
            'avg_transition_risk_sentences_per_call': float(df['transition_risk_sentence_count'].mean()),
            'avg_policy_risk_sentences_per_call': float(df['policy_risk_sentence_count'].mean()),
            'avg_technology_risk_sentences_per_call': float(df['technology_risk_sentence_count'].mean()),
            'avg_market_risk_sentences_per_call': float(df['market_risk_sentence_count'].mean()),
            'avg_total_sentences_per_call': float(df['total_sentences_in_call'].mean()),
            'avg_transition_risk_snippets_per_call': float(df['transition_risk_snippet_count'].mean())
        },
        
        'similarity_statistics': {
            'mean_max_similarity': float(df['max_similarity_score'].mean()),
            'mean_avg_similarity': float(df['mean_similarity_score'].mean()),
            'policy_mean_max_similarity': float(df['policy_max_similarity_score'].mean()),
            'technology_mean_max_similarity': float(df['technology_max_similarity_score'].mean()),
            'market_mean_max_similarity': float(df['market_max_similarity_score'].mean()),
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
                'avg_transition_risk_attention': float(region_data['climate_transition_risk_ratio'].mean()),
                'avg_policy_risk_attention': float(region_data['policy_transition_risk_ratio'].mean()),
                'avg_technology_risk_attention': float(region_data['technology_transition_risk_ratio'].mean()),
                'avg_market_risk_attention': float(region_data['market_transition_risk_ratio'].mean()),
                'transition_risk_coverage_pct': float(region_data['has_transition_risk_content'].mean() * 100),
                'policy_risk_coverage_pct': float(region_data['has_policy_risk_content'].mean() * 100),
                'technology_risk_coverage_pct': float(region_data['has_technology_risk_content'].mean() * 100),
                'market_risk_coverage_pct': float(region_data['has_market_risk_content'].mean() * 100)
            }
    
    return summary_stats


def save_firm_level_results(df: pd.DataFrame, summary_stats: Dict, output_dir: Path):
    """Save firm-level transition risk results in multiple formats."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Main panel dataset
    df.to_csv(output_dir / 'firm_level_transition_risk_attention.csv', index=False)
    df.to_parquet(output_dir / 'firm_level_transition_risk_attention.parquet')
    
    # Summary statistics
    with open(output_dir / 'firm_level_transition_risk_summary_statistics.json', 'w') as f:
        json.dump(summary_stats, f, indent=2, default=str)
    
    # Risk type subsets
    policy_risk_df = df[df['has_policy_risk_content'] == True].copy()
    technology_risk_df = df[df['has_technology_risk_content'] == True].copy()
    market_risk_df = df[df['has_market_risk_content'] == True].copy()
    
    if len(policy_risk_df) > 0:
        policy_risk_df.to_csv(output_dir / 'firm_level_policy_transition_risk_attention.csv', index=False)
    
    if len(technology_risk_df) > 0:
        technology_risk_df.to_csv(output_dir / 'firm_level_technology_transition_risk_attention.csv', index=False)
    
    if len(market_risk_df) > 0:
        market_risk_df.to_csv(output_dir / 'firm_level_market_transition_risk_attention.csv', index=False)
    
    # Regional subsets
    for region in ['US', 'EU']:
        region_data = df[df['region'] == region]
        if len(region_data) > 0:
            region_data.to_csv(output_dir / f'firm_level_transition_risk_attention_{region.lower()}.csv', index=False)
    
    # Stata format for econometric analysis
    try:
        # Create Stata-friendly variable names (max 32 chars, no special characters)
        df_stata = df.copy()
        stata_name_mapping = {
            'climate_transition_risk_ratio': 'transrisk_ratio',
            'policy_transition_risk_ratio': 'polrisk_ratio', 
            'technology_transition_risk_ratio': 'techrisk_ratio',
            'market_transition_risk_ratio': 'mktrisk_ratio',
            'transition_risk_sentence_count': 'transrisk_sentences',
            'policy_risk_sentence_count': 'polrisk_sentences',
            'technology_risk_sentence_count': 'techrisk_sentences',
            'market_risk_sentence_count': 'mktrisk_sentences',
            'transition_risk_snippet_count': 'transrisk_snippets',
            'policy_risk_snippet_count': 'polrisk_snippets',
            'technology_risk_snippet_count': 'techrisk_snippets',
            'market_risk_snippet_count': 'mktrisk_snippets',
            'has_transition_risk_content': 'has_transrisk',
            'has_policy_risk_content': 'has_polrisk',
            'has_technology_risk_content': 'has_techrisk',
            'has_market_risk_content': 'has_mktrisk',
            'total_sentences_in_call': 'total_sentences',
            'max_similarity_score': 'max_similarity',
            'mean_similarity_score': 'mean_similarity',
            'policy_max_similarity_score': 'pol_max_sim',
            'technology_max_similarity_score': 'tech_max_sim',
            'market_max_similarity_score': 'mkt_max_sim',
            'transition_risk_coverage_binary': 'transrisk_binary',
            'policy_risk_coverage_binary': 'polrisk_binary',
            'technology_risk_coverage_binary': 'techrisk_binary',
            'market_risk_coverage_binary': 'mktrisk_binary'
        }
        
        # Rename columns for Stata compatibility
        for old_name, new_name in stata_name_mapping.items():
            if old_name in df_stata.columns:
                df_stata = df_stata.rename(columns={old_name: new_name})
        
        # Ensure remaining column names are Stata-friendly
        df_stata.columns = [col.replace('_', '').lower()[:32] for col in df_stata.columns]
        
        df_stata.to_stata(
            output_dir / 'firm_level_transition_risk_attention.dta',
            write_index=False,
            version=117
        )
    except Exception as e:
        # Fallback to CSV if Stata export fails
        df.to_csv(output_dir / 'firm_level_transition_risk_attention_for_stata.csv', index=False)
        print(f"⚠️ Stata export failed, saved CSV instead: {e}")
    
    # Create balanced panel indicator
    firm_quarters = df.groupby('ticker').size()
    max_quarters = firm_quarters.max()
    balanced_firms = firm_quarters[firm_quarters == max_quarters].index
    
    df['balanced_panel'] = df['ticker'].isin(balanced_firms)
    balanced_df = df[df['balanced_panel']].copy()
    
    if len(balanced_df) > 0:
        balanced_df.to_csv(output_dir / 'firm_level_transition_risk_attention_balanced.csv', index=False)
    
    print(f"💾 Results saved to: {output_dir}")
    print(f"📊 Panel structure:")
    print(f"   • {len(df):,} firm-quarter observations")
    print(f"   • {df['ticker'].nunique():,} unique firms")
    print(f"   • {df['quarter_id'].nunique()} unique quarters")
    print(f"   • {len(balanced_df):,} observations in balanced panel")
    print(f"⚡ Transition risk coverage:")
    print(f"   • {df['has_transition_risk_content'].sum():,} calls with any transition risk content ({df['has_transition_risk_content'].mean()*100:.1f}%)")
    print(f"   • {df['has_policy_risk_content'].sum():,} calls with policy/regulatory risk content ({df['has_policy_risk_content'].mean()*100:.1f}%)")
    print(f"   • {df['has_technology_risk_content'].sum():,} calls with technology risk content ({df['has_technology_risk_content'].mean()*100:.1f}%)")
    print(f"   • {df['has_market_risk_content'].sum():,} calls with market risk content ({df['has_market_risk_content'].mean()*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Create firm-level climate transition risk attention panel dataset using semantic search',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with default paths
    python firm_level_transition_risk_attention.py
    
    # Specify custom paths and indices
    python firm_level_transition_risk_attention.py \\
        --enhanced-snippets-path data/enhanced_climate_snippets \\
        --semantic-index-path data/semantic_indexes/combined \\
        --output-dir outputs/firm_level_panel/transition_risk \\
        --stock-indices SP500 STOXX600 --verbose
    
    # Process only S&P 500 firms
    python firm_level_transition_risk_attention.py \\
        --stock-indices SP500 --verbose

This script creates comprehensive transition risk attention measures including:
- Overall transition risk attention (policy + technology + market)
- Policy/regulatory transition risk attention (carbon pricing, regulations)
- Technology transition risk attention (stranded assets, disruption)
- Market transition risk attention (consumer preferences, reputation)
- Similarity scores and coverage statistics
- Lagged variables for econometric analysis

Based on the three-pillar transition risk framework:
1. Policy/Regulatory Risks: Carbon pricing, climate regulations, compliance costs
2. Technology Risks: Stranded assets, technological disruption, obsolescence  
3. Market Risks: Consumer preferences, reputation risks, demand shifts
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
        help='Path to semantic index for transition risk search'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path("outputs/firm_level_panel/transition_risk"),
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
    
    print("⚡ Firm-Level Climate Transition Risk Attention Panel Creator")
    print("=" * 65)
    print(f"📍 Based on Sautner et al. (2023) framework with semantic search enhancements")
    print(f"🔍 Analyzing three types of transition risks:")
    print(f"   1. Policy/Regulatory Risks: Carbon pricing, climate regulations, compliance")
    print(f"   2. Technology Risks: Stranded assets, technological disruption, obsolescence")
    print(f"   3. Market Risks: Consumer preferences, reputation risks, demand shifts")
    print()
    
    try:
        # Validate inputs
        if not args.semantic_index_path.exists():
            raise FileNotFoundError(f"Semantic index path not found: {args.semantic_index_path}")
        
        if not args.enhanced_snippets_path.exists():
            raise FileNotFoundError(f"Enhanced snippets path not found: {args.enhanced_snippets_path}")
        
        # Create firm-level transition risk panel
        print("🔄 Creating firm-level transition risk attention panel...")
        df = create_firm_level_transition_risk_panel(
            args.enhanced_snippets_path,
            args.structured_transcripts_path,
            args.stock_indices,
            args.semantic_index_path
        )
        
        if df.empty:
            print("❌ No firm-level transition risk data created!")
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
        
        print(f"\n✅ Firm-level climate transition risk attention panel completed!")
        print(f"📁 Key output files:")
        print(f"   • firm_level_transition_risk_attention.csv - Main panel dataset")
        print(f"   • firm_level_transition_risk_attention.dta - Stata format")
        print(f"   • firm_level_transition_risk_attention_balanced.csv - Balanced panel")
        print(f"   • firm_level_policy_transition_risk_attention.csv - Policy risk subset")
        print(f"   • firm_level_technology_transition_risk_attention.csv - Technology risk subset")
        print(f"   • firm_level_market_transition_risk_attention.csv - Market risk subset")
        print(f"   • firm_level_transition_risk_summary_statistics.json - Summary stats")
        print()
        print(f"🔬 Research applications:")
        print(f"   • Event studies around major climate policy announcements")
        print(f"   • Cross-sectional analysis of transition risk exposure by industry")
        print(f"   • Time-series analysis of transition risk attention evolution")
        print(f"   • Regional comparisons (US vs EU) of transition risk discourse")
        print(f"   • Validation against carbon intensity and stranded asset measures")
        print(f"   • Policy heterogeneity analysis using causal forests")
        print(f"   • Market transition risk and ESG rating correlation studies")
        
    except Exception as e:
        logger.error(f"❌ Error during processing: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()