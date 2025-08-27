#!/usr/bin/env python3
"""
Simplified firm-level climate disclosure attention calculator using semantic search.

Creates a single compound measure reflecting ESG strategy and climate transparency/disclosure attention.
Includes optional ROC validation with reduced query set and duplicate removal.
"""

import argparse
import json
import logging
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Set
from tqdm import tqdm
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from src.analysis_pipeline.base_searcher import BaseSemanticSearcher, ClimateSnippet
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent / "semantic_search"))
    from base_searcher import BaseSemanticSearcher, ClimateSnippet

from src.config import SUPPORTED_INDICES, LOGS_DIR
from src.analysis_pipeline.roc_validator import SimpleROCValidator


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOGS_DIR / 'climate_disclosure_attention.log', mode='a')
        ]
    )


def parse_date_to_month(date_str: str) -> Optional[str]:
    """Parse various date formats to YYYY-MM format."""
    if not date_str or pd.isna(date_str):
        return None
    
    date_str = str(date_str).strip()
    
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
    
    try:
        parsed_date = pd.to_datetime(date_str, dayfirst=True)
        return parsed_date.strftime('%Y-%m')
    except:
        return None


class ClimateDisclosureAnalyzer(BaseSemanticSearcher):
    """Simplified semantic searcher for climate disclosure and ESG strategy discussions."""
    
    def __init__(self, index_path: Path, config_path: Optional[str] = None):
        """Initialize climate disclosure analyzer."""
        super().__init__(index_path, config_path)
        
        # Simplified query set focusing on key disclosure and strategy themes
        self.disclosure_queries = [
            # General ESG / disclosure framing
            "ESG reporting sustainability report TCFD disclosure transparency",
            "carbon emissions disclosure scope emissions footprint reporting",
            "climate governance strategy net zero targets sustainability",
            "climate risk disclosure regulatory compliance SEC",
            "sustainability metrics ESG performance environmental disclosure",
            "climate transition plan decarbonization strategy",
            "ESG ratings sustainability assessment disclosure",

            # Earnings call–specific framing
            "investor expectations ESG disclosure transparency sustainability strategy",
            "analyst questions on climate reporting disclosure transparency",
            "SEC climate rule disclosure requirements earnings call discussion",
            "clarity transparency on ESG roadmap targets disclosure commitments",
            "progress update on sustainability reporting frameworks TCFD SASB GRI",
            "alignment of climate disclosures with regulatory or investor standards",
            "transparency around carbon targets net zero pathway disclosure",
            "management commentary on ESG disclosure audit assurance",
            "stakeholder engagement disclosure around sustainability strategy",
            "consistency comparability reliability of ESG disclosures reporting"
        ]

        
        self.similarity_threshold = 0.40
        
    def find_climate_disclosure_snippets(self) -> Dict[str, List[ClimateSnippet]]:
        """Find climate disclosure snippets and remove duplicates."""
        logger = logging.getLogger(__name__)
        
        logger.info(f"Searching for climate disclosure snippets with {len(self.disclosure_queries)} queries")
        logger.info(f"Using similarity threshold: {self.similarity_threshold}")
        
        # Perform semantic search
        all_snippets = self.multi_query_search(
            queries=self.disclosure_queries,
            top_k=1000000,
            min_score=self.similarity_threshold,
            aggregation='max'
        )
        
        # Remove duplicates based on text content and metadata
        unique_snippets = self._remove_duplicates(all_snippets)
        
        logger.info(f"Found {len(all_snippets)} snippets, {len(unique_snippets)} after deduplication")
        
        # Group snippets by earnings call
        snippets_by_call = {}
        for snippet in unique_snippets:
            call_key = f"{snippet.ticker}_{snippet.year}_{snippet.quarter}"
            if call_key not in snippets_by_call:
                snippets_by_call[call_key] = []
            snippets_by_call[call_key].append(snippet)
        
        return snippets_by_call
    
    def _remove_duplicates(self, snippets: List[ClimateSnippet]) -> List[ClimateSnippet]:
        """Remove duplicates, keeping the snippet with highest similarity score."""
        snippet_dict = {}
        
        for snippet in snippets:
            snippet_key = f"{hash(snippet.text)}_{snippet.ticker}_{snippet.year}_{snippet.quarter}"
            
            if snippet_key not in snippet_dict:
                snippet_dict[snippet_key] = snippet
            else:
                # Keep the one with higher similarity score
                if snippet.similarity_score > snippet_dict[snippet_key].similarity_score:
                    snippet_dict[snippet_key] = snippet
        
        return list(snippet_dict.values())


def create_climate_disclosure_panel(enhanced_snippets_path: Path, 
                                   structured_path: Path,
                                   stock_indices: List[str],
                                   semantic_index_path: Path,
                                   run_roc_validation: bool = False,
                                   roc_output_dir: Optional[Path] = None) -> pd.DataFrame:
    """Create simplified climate disclosure attention panel dataset."""
    logger = logging.getLogger(__name__)
    
    # Initialize semantic searcher
    logger.info(f"Initializing climate disclosure analyzer from {semantic_index_path}")
    disclosure_analyzer = ClimateDisclosureAnalyzer(semantic_index_path)
    
    # Optional ROC validation
    if run_roc_validation:
        print("Running ROC validation for climate disclosure threshold optimization...")
        
        if roc_output_dir:
            roc_output_dir.mkdir(parents=True, exist_ok=True)
        
        validator = SimpleROCValidator()
        
        roc_results = validator.validate_and_plot_roc(
            searcher=disclosure_analyzer,
            queries=disclosure_analyzer.disclosure_queries,
            policy_description="climate ESG strategy transparency and disclosure discussions in earnings calls",
            sample_size=800,
            save_validation_data=True,
            output_path=str(roc_output_dir / "climate_disclosure_roc_curve.png") if roc_output_dir else "climate_disclosure_roc_curve.png"
        )
        
        # Save ROC results
        if roc_output_dir:
            with open(roc_output_dir / "roc_validation_results.json", 'w') as f:
                json.dump(roc_results, f, indent=2)
        
        print(f"ROC validation complete. AUC: {roc_results['auc']:.3f}, Optimal threshold: {roc_results['optimal_threshold']:.3f}")
        print(f"Current threshold: {disclosure_analyzer.similarity_threshold}")
    
    # Find climate disclosure snippets
    disclosure_snippets = disclosure_analyzer.find_climate_disclosure_snippets()
    
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
                    
                    # Check for disclosure content using semantic search results
                    call_key = f"{ticker}_{year}_{quarter}"
                    call_disclosure_snippets = disclosure_snippets.get(call_key, [])
                    
                    # Calculate disclosure attention measures
                    disclosure_sentence_count = len(call_disclosure_snippets)
                    
                    # Calculate similarity statistics
                    if call_disclosure_snippets:
                        similarity_scores = [s.similarity_score for s in call_disclosure_snippets]
                        max_similarity = max(similarity_scores)
                        mean_similarity = np.mean(similarity_scores)
                    else:
                        max_similarity = 0.0
                        mean_similarity = 0.0
                    
                    # Calculate disclosure attention ratio
                    disclosure_ratio = 0.0
                    if total_sentences and total_sentences > 0:
                        disclosure_ratio = disclosure_sentence_count / total_sentences
                    
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
                        
                        # Climate disclosure measures (compound measure)
                        'climate_disclosure_ratio': disclosure_ratio,
                        'disclosure_sentence_count': disclosure_sentence_count,
                        'disclosure_snippet_count': len(call_disclosure_snippets),
                        'has_disclosure_content': len(call_disclosure_snippets) > 0,
                        
                        # Base transcript information
                        'total_sentences_in_call': total_sentences,
                        
                        # Similarity measures
                        'max_similarity_score': max_similarity,
                        'mean_similarity_score': mean_similarity,
                        
                        # Technical parameters
                        'similarity_threshold': disclosure_analyzer.similarity_threshold,
                        
                        # Log transformations
                        'log_disclosure_sentences': np.log(disclosure_sentence_count + 1),
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
    
    # Apply data quality filters
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
    
    logger.info(f"Created climate disclosure panel with {len(df)} firm-quarter observations")
    logger.info(f"   {df['ticker'].nunique()} unique firms")
    logger.info(f"   {df['quarter_id'].nunique()} unique quarters")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Create simplified firm-level climate disclosure attention panel dataset'
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
        help='Path to semantic index for climate disclosure search'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path("outputs/firm_level_panel/climate_disclosure"),
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
        '--run-roc-validation',
        action='store_true',
        help='Run ROC validation to determine optimal threshold'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    print("Simplified Climate Disclosure Attention Panel Creator")
    print("=" * 55)
    
    try:
        # Validate inputs
        if not args.semantic_index_path.exists():
            raise FileNotFoundError(f"Semantic index path not found: {args.semantic_index_path}")
        
        if not args.enhanced_snippets_path.exists():
            raise FileNotFoundError(f"Enhanced snippets path not found: {args.enhanced_snippets_path}")
        
        # Create climate disclosure panel
        df = create_climate_disclosure_panel(
            args.enhanced_snippets_path,
            args.structured_transcripts_path,
            args.stock_indices,
            args.semantic_index_path,
            run_roc_validation=args.run_roc_validation,
            roc_output_dir=args.output_dir / "roc_validation" if args.run_roc_validation else None
        )
        
        if df.empty:
            print("No climate disclosure data created!")
            return
        
        # Save results
        args.output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_dir / 'climate_disclosure_attention.csv', index=False)
        
        print(f"\nClimate disclosure attention panel completed!")
        print(f"Results saved to: {args.output_dir}")
        print(f"Panel structure: {len(df)} firm-quarter observations, {df['ticker'].nunique()} unique firms")
        print(f"Coverage: {df['has_disclosure_content'].sum()} calls with disclosure content ({df['has_disclosure_content'].mean()*100:.1f}%)")
        
        if args.run_roc_validation:
            print(f"ROC validation results saved to: {args.output_dir / 'roc_validation'}")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()