#!/usr/bin/env python3
"""
Firm-level climate opportunity attention calculator using semantic search.

This script calculates firm-quarter climate opportunity attention measures by 
applying semantic search to identify opportunity-related discussions in earnings calls.
Creates panel data suitable for econometric analysis and event studies.

Includes annotation framework for threshold validation using human expert judgment.

Usage:
    # Create annotation sample for human validation
    python scripts/3_agg_variables/firm_level_opportunity_attention.py --create-annotation-sample
    
    # Validate threshold after human annotation
    python scripts/3_agg_variables/firm_level_opportunity_attention.py --validate-threshold annotation_sample_completed.csv
    
    # Run main analysis
    python scripts/3_agg_variables/firm_level_opportunity_attention.py

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
            logging.FileHandler(LOGS_DIR / 'firm_level_opportunity_attention.log', mode='a')
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


class OpportunityAnalyzer(BaseSemanticSearcher):
    """
    Semantic searcher for climate opportunity discussions in earnings calls.
    Enhanced with annotation and threshold validation capabilities.
    """
    
    def __init__(self, index_path: Path, config_path: Optional[str] = None):
        """Initialize opportunity analyzer with semantic search capabilities."""
        super().__init__(index_path, config_path)
        
        # Climate opportunity queries
        self.opportunity_queries = [
            'renewable energy investments clean technology opportunities solar wind',
            'green innovation sustainable business models growth opportunities',
            'energy transition investment opportunities clean energy market',
            'sustainability competitive advantage ESG opportunities',
            'clean technology innovation renewable energy development',
            'green finance sustainable finance ESG investment opportunities',
            'carbon credits emissions trading environmental commodities',
            'electric vehicles EV battery technology clean transportation',
            'energy efficiency technology solutions optimization savings',
            'circular economy waste reduction resource efficiency business',
            'green bonds sustainability-linked financing climate finance',
            'sustainable supply chain green procurement cost savings',
            'climate adaptation resilience business opportunities',
            'green markets sustainable products consumer demand growth'
        ]
        
        self.similarity_threshold = 0.42
        
    def find_opportunity_snippets(self) -> Dict[str, List[ClimateSnippet]]:
        """Find all opportunity-related snippets using semantic search."""
        logger = logging.getLogger(__name__)
        
        logger.info(f"🔍 Searching for opportunity snippets with {len(self.opportunity_queries)} queries")
        logger.info(f"Using similarity threshold: {self.similarity_threshold}")
        
        # Perform semantic search
        opportunity_snippets = self.multi_query_search(
            queries=self.opportunity_queries,
            top_k=10000,  # Large number to get comprehensive results
            min_score=self.similarity_threshold,
            aggregation='max'
        )
        
        logger.info(f"Found {len(opportunity_snippets)} opportunity-related snippets")
        
        # Group snippets by earnings call
        snippets_by_call = {}
        for snippet in opportunity_snippets:
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
            queries=self.opportunity_queries,
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
                'human_binary': '',     # Instructions: 0=Not opportunity-related, 1=Opportunity-related
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
    
    def _create_annotation_guidelines(self, guidelines_path: Path):
        """Create comprehensive annotation guidelines file."""
        guidelines = """
CLIMATE OPPORTUNITY ANNOTATION GUIDELINES
========================================

TASK OVERVIEW:
You will evaluate whether text snippets from earnings calls discuss climate-related business opportunities.

RATING SCALES:

1. HUMAN_RELEVANCE (0-3 scale):
   0 = Not climate-related at all
   1 = Mentions climate/environment but not as business opportunity
   2 = Discusses climate-related business opportunities
   3 = Substantial discussion of climate business opportunities

2. HUMAN_BINARY (0/1):
   0 = Not a climate opportunity discussion
   1 = Is a climate opportunity discussion

3. CONFIDENCE (1-5):
   1 = Very uncertain about my rating
   2 = Somewhat uncertain
   3 = Moderately confident
   4 = Quite confident
   5 = Very confident

WHAT COUNTS AS CLIMATE OPPORTUNITIES:
✓ Renewable energy investments or growth
✓ Clean technology development or sales
✓ Green finance or ESG-linked products
✓ Sustainable product lines or services
✓ Energy efficiency improvements as business opportunity
✓ Carbon credits or environmental commodities
✓ Electric vehicles or clean transportation
✓ Circular economy business models
✓ Climate adaptation services
✓ Green supply chain advantages

WHAT DOES NOT COUNT:
✗ Compliance costs or regulatory burdens
✗ Physical climate risks (floods, droughts)
✗ General ESG reporting without business opportunity angle
✗ Climate risks or transition costs
✗ Purely operational discussions without opportunity framing

EXAMPLES:

HIGH RELEVANCE (3): "We see significant growth opportunities in our renewable energy portfolio, with solar installations driving 20% revenue growth this quarter."

MEDIUM RELEVANCE (2): "Our new sustainable product line is gaining traction with environmentally conscious consumers."

LOW RELEVANCE (1): "We comply with all environmental regulations and report our carbon emissions annually."

NOT RELEVANT (0): "Our quarterly revenue increased 5% driven by strong performance in traditional manufacturing."

TIPS:
- Focus on OPPORTUNITY language: growth, investment, market potential, competitive advantage
- Context matters: same technology can be opportunity or cost depending on framing
- When uncertain, use CONFIDENCE rating to indicate your uncertainty
- Use NOTES field for borderline cases or complex situations

ANNOTATOR ID: Please use a consistent ID (e.g., your initials + number: "MJ01")
"""
        guidelines_path.write_text(guidelines)
    
    def calculate_threshold_performance(self, annotated_data_path: Path, output_dir: Path) -> tuple[pd.DataFrame, float]:
        """
        Calculate precision, recall, F1 for different thresholds using human annotations.
        Returns results DataFrame and optimal threshold.
        """
        logger = logging.getLogger(__name__)
        
        logger.info(f"📊 Calculating threshold performance from {annotated_data_path}")
        
        # Load human annotations
        df = pd.read_csv(annotated_data_path)
        
        # Remove instruction row if it exists
        df = df[df['snippet_id'] != 'INSTRUCTIONS'].copy()
        
        # Validate annotations
        df = self._validate_annotations(df)
        
        if len(df) == 0:
            logger.error("No valid annotations found!")
            return pd.DataFrame(), self.similarity_threshold
        
        # Convert human relevance to binary (2+ = relevant for opportunities)
        df['human_binary_calc'] = (df['human_relevance'] >= 2).astype(int)
        
        # Also use explicit human binary if provided
        if 'human_binary' in df.columns:
            valid_binary = df['human_binary'].notna() & (df['human_binary'] != '')
            df.loc[valid_binary, 'human_binary_calc'] = df.loc[valid_binary, 'human_binary'].astype(int)
        
        logger.info(f"✅ Processed {len(df)} valid annotations")
        logger.info(f"   Positive examples: {df['human_binary_calc'].sum()}")
        logger.info(f"   Negative examples: {(1 - df['human_binary_calc']).sum()}")
        
        # Calculate performance across thresholds
        results = []
        thresholds = np.arange(0.25, 0.70, 0.02)
        
        for threshold in thresholds:
            # Model predictions at this threshold
            model_pred = (df['similarity_score'] >= threshold).astype(int)
            human_true = df['human_binary_calc']
            
            # Calculate metrics
            precision = precision_score(human_true, model_pred, zero_division=0)
            recall = recall_score(human_true, model_pred, zero_division=0)
            f1 = f1_score(human_true, model_pred, zero_division=0)
            accuracy = accuracy_score(human_true, model_pred)
            
            # Coverage (what % of snippets retained)
            coverage = model_pred.mean()
            
            # True/False positives and negatives for detailed analysis
            tp = ((model_pred == 1) & (human_true == 1)).sum()
            fp = ((model_pred == 1) & (human_true == 0)).sum()
            tn = ((model_pred == 0) & (human_true == 0)).sum()
            fn = ((model_pred == 0) & (human_true == 1)).sum()
            
            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'coverage': coverage,
                'n_predicted_positive': model_pred.sum(),
                'n_true_positive': human_true.sum(),
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn
            })
        
        results_df = pd.DataFrame(results)
        
        # Find optimal threshold (maximize F1)
        optimal_idx = results_df['f1'].idxmax()
        optimal_threshold = results_df.loc[optimal_idx, 'threshold']
        
        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_dir / 'threshold_validation_results.csv', index=False)
        
        # Save detailed analysis
        self._save_threshold_analysis(df, results_df, optimal_threshold, output_dir)
        
        # Print summary
        optimal_row = results_df.loc[optimal_idx]
        logger.info(f"🎯 OPTIMAL THRESHOLD ANALYSIS:")
        logger.info(f"   Threshold: {optimal_threshold:.3f}")
        logger.info(f"   F1 Score: {optimal_row['f1']:.3f}")
        logger.info(f"   Precision: {optimal_row['precision']:.3f}")
        logger.info(f"   Recall: {optimal_row['recall']:.3f}")
        logger.info(f"   Accuracy: {optimal_row['accuracy']:.3f}")
        logger.info(f"   Coverage: {optimal_row['coverage']:.1%}")
        
        return results_df, optimal_threshold
    
    def _validate_annotations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean annotation data."""
        logger = logging.getLogger(__name__)
        
        initial_count = len(df)
        
        # Remove rows with missing human_relevance
        df = df[df['human_relevance'].notna() & (df['human_relevance'] != '')].copy()
        
        # Convert to numeric and validate range
        df['human_relevance'] = pd.to_numeric(df['human_relevance'], errors='coerce')
        df = df[df['human_relevance'].between(0, 3)].copy()
        
        # Log validation results
        removed_count = initial_count - len(df)
        if removed_count > 0:
            logger.warning(f"Removed {removed_count} invalid annotations")
        
        return df
    
    def _save_threshold_analysis(self, df: pd.DataFrame, results_df: pd.DataFrame, 
                                optimal_threshold: float, output_dir: Path):
        """Save detailed threshold analysis results."""
        
        # Save annotated data with model predictions
        df_analysis = df.copy()
        df_analysis['optimal_model_pred'] = (df_analysis['similarity_score'] >= optimal_threshold).astype(int)
        df_analysis['current_model_pred'] = (df_analysis['similarity_score'] >= self.similarity_threshold).astype(int)
        df_analysis.to_csv(output_dir / 'annotated_data_with_predictions.csv', index=False)
        
        # Create summary statistics
        summary = {
            'annotation_summary': {
                'total_annotations': len(df),
                'positive_examples': int(df['human_binary_calc'].sum()),
                'negative_examples': int((1 - df['human_binary_calc']).sum()),
                'mean_similarity_positive': float(df[df['human_binary_calc'] == 1]['similarity_score'].mean()),
                'mean_similarity_negative': float(df[df['human_binary_calc'] == 0]['similarity_score'].mean())
            },
            'optimal_threshold': {
                'threshold': float(optimal_threshold),
                'f1_score': float(results_df.loc[results_df['threshold'] == optimal_threshold, 'f1'].iloc[0]),
                'precision': float(results_df.loc[results_df['threshold'] == optimal_threshold, 'precision'].iloc[0]),
                'recall': float(results_df.loc[results_df['threshold'] == optimal_threshold, 'recall'].iloc[0]),
                'coverage': float(results_df.loc[results_df['threshold'] == optimal_threshold, 'coverage'].iloc[0])
            },
            'current_threshold': {
                'threshold': float(self.similarity_threshold),
                'performance': results_df[results_df['threshold'].round(3) == round(self.similarity_threshold, 3)].to_dict('records')[0] if any(results_df['threshold'].round(3) == round(self.similarity_threshold, 3)) else {}
            }
        }
        
        with open(output_dir / 'threshold_analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Create examples of classification differences
        df_examples = df.copy()
        df_examples['optimal_pred'] = (df_examples['similarity_score'] >= optimal_threshold).astype(int)
        df_examples['current_pred'] = (df_examples['similarity_score'] >= self.similarity_threshold).astype(int)
        
        # False positives at current threshold
        false_positives = df_examples[
            (df_examples['current_pred'] == 1) & 
            (df_examples['human_binary_calc'] == 0)
        ][['text', 'similarity_score', 'human_relevance']].head(10)
        
        # False negatives at current threshold  
        false_negatives = df_examples[
            (df_examples['current_pred'] == 0) & 
            (df_examples['human_binary_calc'] == 1)
        ][['text', 'similarity_score', 'human_relevance']].head(10)
        
        # Save examples
        with open(output_dir / 'classification_examples.txt', 'w') as f:
            f.write("CLASSIFICATION EXAMPLES\n")
            f.write("=====================\n\n")
            f.write(f"Current threshold: {self.similarity_threshold}\n")
            f.write(f"Optimal threshold: {optimal_threshold}\n\n")
            
            f.write("FALSE POSITIVES (Current threshold captures but humans say not relevant):\n")
            f.write("-" * 70 + "\n")
            for idx, row in false_positives.iterrows():
                f.write(f"Similarity: {row['similarity_score']:.3f} | Human rating: {row['human_relevance']}\n")
                f.write(f"Text: {row['text'][:200]}...\n\n")
            
            f.write("FALSE NEGATIVES (Current threshold misses but humans say relevant):\n")
            f.write("-" * 70 + "\n")
            for idx, row in false_negatives.iterrows():
                f.write(f"Similarity: {row['similarity_score']:.3f} | Human rating: {row['human_relevance']}\n")
                f.write(f"Text: {row['text'][:200]}...\n\n")


def create_firm_level_opportunity_panel(enhanced_snippets_path: Path, 
                                       structured_path: Path,
                                       stock_indices: List[str],
                                       semantic_index_path: Path) -> pd.DataFrame:
    """
    Create firm-level climate opportunity attention panel dataset using semantic search.
    
    Returns:
        DataFrame with firm-quarter climate opportunity attention measures
    """
    logger = logging.getLogger(__name__)
    
    # Initialize semantic searcher
    logger.info(f"🔍 Initializing semantic search from {semantic_index_path}")
    opportunity_analyzer = OpportunityAnalyzer(semantic_index_path)
    
    # Find opportunity snippets using semantic search
    opportunity_snippets_by_call = opportunity_analyzer.find_opportunity_snippets()
    
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
                    
                    # Check for opportunity content using semantic search results
                    call_key = f"{ticker}_{year}_{quarter}"
                    opportunity_snippets = opportunity_snippets_by_call.get(call_key, [])
                    
                    # Count opportunity sentences from semantic search
                    opportunity_sentence_count = len(opportunity_snippets)
                    
                    # Calculate similarity statistics
                    if opportunity_snippets:
                        similarity_scores = [s.similarity_score for s in opportunity_snippets]
                        max_similarity = max(similarity_scores)
                        mean_similarity = np.mean(similarity_scores)
                        min_similarity = min(similarity_scores)
                    else:
                        max_similarity = 0.0
                        mean_similarity = 0.0
                        min_similarity = 0.0
                    
                    # Calculate opportunity attention ratio
                    climate_opportunity_ratio = 0.0
                    if total_sentences and total_sentences > 0:
                        climate_opportunity_ratio = opportunity_sentence_count / total_sentences
                    
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
                        
                        # Climate opportunity measures
                        'climate_opportunity_ratio': climate_opportunity_ratio,
                        'opportunity_sentence_count': opportunity_sentence_count,
                        'total_sentences_in_call': total_sentences,
                        'opportunity_snippet_count': len(opportunity_snippets),
                        'has_opportunity_content': len(opportunity_snippets) > 0,
                        
                        # Similarity measures
                        'max_similarity_score': max_similarity,
                        'mean_similarity_score': mean_similarity,
                        'min_similarity_score': min_similarity,
                        'similarity_threshold': opportunity_analyzer.similarity_threshold,
                        
                        # Coverage indicators
                        'opportunity_coverage_binary': 1 if len(opportunity_snippets) > 0 else 0,
                        'log_opportunity_sentences': np.log(opportunity_sentence_count + 1),
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
    
    logger.info(f"✅ Created opportunity panel with {len(df)} firm-quarter observations")
    logger.info(f"   {df['ticker'].nunique()} unique firms")
    logger.info(f"   {df['quarter_id'].nunique()} unique quarters")
    
    return df


def add_lagged_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged climate opportunity attention variables for econometric analysis."""
    
    df = df.sort_values(['ticker', 'year', 'quarter'])
    
    # Create lagged variables (useful for event studies)
    for lag in [1, 2, 4]:  # 1, 2, and 4 quarters
        df[f'climate_opportunity_ratio_lag{lag}'] = df.groupby('ticker')['climate_opportunity_ratio'].shift(lag)
        df[f'has_opportunity_content_lag{lag}'] = df.groupby('ticker')['has_opportunity_content'].shift(lag)
    
    # Create moving averages (useful for smoothing)
    for window in [2, 4]:
        df[f'climate_opportunity_ratio_ma{window}'] = (
            df.groupby('ticker')['climate_opportunity_ratio']
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
        
        'opportunity_attention_distribution': {
            'mean': float(df['climate_opportunity_ratio'].mean()),
            'median': float(df['climate_opportunity_ratio'].median()),
            'std': float(df['climate_opportunity_ratio'].std()),
            'min': float(df['climate_opportunity_ratio'].min()),
            'max': float(df['climate_opportunity_ratio'].max()),
            'p25': float(df['climate_opportunity_ratio'].quantile(0.25)),
            'p75': float(df['climate_opportunity_ratio'].quantile(0.75)),
            'p90': float(df['climate_opportunity_ratio'].quantile(0.90)),
            'p99': float(df['climate_opportunity_ratio'].quantile(0.99))
        },
        
        'coverage_statistics': {
            'pct_calls_with_opportunities': float(df['has_opportunity_content'].mean() * 100),
            'avg_opportunity_sentences_per_call': float(df['opportunity_sentence_count'].mean()),
            'avg_total_sentences_per_call': float(df['total_sentences_in_call'].mean()),
            'avg_opportunity_snippets_per_call': float(df['opportunity_snippet_count'].mean())
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
                'avg_opportunity_attention': float(region_data['climate_opportunity_ratio'].mean()),
                'opportunity_coverage_pct': float(region_data['has_opportunity_content'].mean() * 100)
            }
    
    return summary_stats


def save_firm_level_results(df: pd.DataFrame, summary_stats: Dict, output_dir: Path):
    """Save firm-level results in multiple formats."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Main panel dataset
    df.to_csv(output_dir / 'firm_level_opportunity_attention.csv', index=False)
    df.to_parquet(output_dir / 'firm_level_opportunity_attention.parquet')
    
    # Summary statistics
    with open(output_dir / 'firm_level_opportunity_summary_statistics.json', 'w') as f:
        json.dump(summary_stats, f, indent=2, default=str)
    
    # Regional subsets
    for region in ['US', 'EU']:
        region_data = df[df['region'] == region]
        if len(region_data) > 0:
            region_data.to_csv(output_dir / f'firm_level_opportunity_attention_{region.lower()}.csv', index=False)
    
    # Stata format for econometric analysis
    try:
        # Create Stata-friendly variable names
        df_stata = df.copy()
        df_stata.columns = [col.replace('_', '').lower()[:32] for col in df_stata.columns]
        
        df_stata.to_stata(
            output_dir / 'firm_level_opportunity_attention.dta',
            write_index=False,
            version=117
        )
    except Exception as e:
        # Fallback
        df.to_csv(output_dir / 'firm_level_opportunity_attention_for_stata.csv', index=False)
    
    # Create balanced panel indicator
    firm_quarters = df.groupby('ticker').size()
    max_quarters = firm_quarters.max()
    balanced_firms = firm_quarters[firm_quarters == max_quarters].index
    
    df['balanced_panel'] = df['ticker'].isin(balanced_firms)
    balanced_df = df[df['balanced_panel']].copy()
    
    if len(balanced_df) > 0:
        balanced_df.to_csv(output_dir / 'firm_level_opportunity_attention_balanced.csv', index=False)
    
    print(f"💾 Results saved to: {output_dir}")
    print(f"📊 Panel structure:")
    print(f"   • {len(df):,} firm-quarter observations")
    print(f"   • {df['ticker'].nunique():,} unique firms")
    print(f"   • {df['quarter_id'].nunique()} unique quarters")
    print(f"   • {len(balanced_df):,} observations in balanced panel")


def main():
    parser = argparse.ArgumentParser(
        description='Create firm-level climate opportunity attention panel dataset using semantic search',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create annotation sample for threshold validation
  python firm_level_opportunity_attention.py --create-annotation-sample
  
  # Validate threshold after human annotation
  python firm_level_opportunity_attention.py --validate-threshold annotation_sample_completed.csv
  
  # Run main analysis with default settings
  python firm_level_opportunity_attention.py
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
        help='Path to semantic index for opportunity search'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path("outputs/firm_level_panel/opportunity"),
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
    
    print("🌱 Firm-Level Climate Opportunity Attention Panel Creator")
    print("=" * 55)
    
    try:
        # MODE 1: Create annotation sample
        if args.create_annotation_sample:
            print("📋 Creating annotation sample for threshold validation...")
            
            # Initialize opportunity analyzer
            opportunity_analyzer = OpportunityAnalyzer(args.semantic_index_path)
            
            # Collect annotation samples
            annotation_samples = opportunity_analyzer.collect_annotation_samples(args.annotation_sample_size)
            
            # Export for annotation
            annotation_output = args.output_dir / 'annotation' / 'annotation_sample.csv'
            opportunity_analyzer.export_for_annotation(annotation_samples, annotation_output)
            
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
            
            # Initialize opportunity analyzer
            opportunity_analyzer = OpportunityAnalyzer(args.semantic_index_path)
            
            # Calculate threshold performance
            validation_output_dir = args.output_dir / 'validation'
            results_df, optimal_threshold = opportunity_analyzer.calculate_threshold_performance(
                args.validate_threshold, 
                validation_output_dir
            )
            
            print(f"\n✅ Threshold validation completed!")
            print(f"📁 Results saved to: {validation_output_dir}")
            print(f"📊 Key findings:")
            print(f"   • Current threshold: {opportunity_analyzer.similarity_threshold}")
            print(f"   • Optimal threshold: {optimal_threshold:.3f}")
            
            if abs(optimal_threshold - opportunity_analyzer.similarity_threshold) > 0.05:
                print(f"⚠️  Significant difference detected!")
                print(f"   Consider updating similarity_threshold in OpportunityAnalyzer class")
            else:
                print(f"✅ Current threshold is well-calibrated")
            
            return
        
        # MODE 3: Main analysis - create firm-level panel
        print("🔍 Running main analysis to create firm-level opportunity panel...")
        
        # Create firm-level opportunity panel
        df = create_firm_level_opportunity_panel(
            args.enhanced_snippets_path,
            args.structured_transcripts_path,
            args.stock_indices,
            args.semantic_index_path
        )
        
        if df.empty:
            print("❌ No firm-level opportunity data created!")
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
        
        print(f"\n✅ Firm-level climate opportunity attention panel completed!")
        print(f"📁 Key output files:")
        print(f"   • firm_level_opportunity_attention.csv - Main panel dataset")
        print(f"   • firm_level_opportunity_attention.dta - Stata format")
        print(f"   • firm_level_opportunity_attention_balanced.csv - Balanced panel")
        print(f"   • firm_level_opportunity_summary_statistics.json - Summary stats")
        
        # Display key statistics
        print(f"\n📊 Panel overview:")
        print(f"   • {len(df):,} firm-quarter observations")
        print(f"   • {df['ticker'].nunique():,} unique firms")
        print(f"   • {df['quarter_id'].nunique()} unique quarters")
        print(f"   • {df['has_opportunity_content'].mean():.1%} of calls contain opportunity content")
        print(f"   • Mean opportunity attention: {df['climate_opportunity_ratio'].mean():.4f}")
        
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