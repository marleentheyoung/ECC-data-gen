#!/usr/bin/env python3
"""
Semantic Climate Change Risk Analysis

This script reads climate snippets and applies semantic search to identify 
risk-related paragraphs, then calculates firm-level climate risk exposure
as the ratio of risk-related sentences to total sentences in earnings calls.

Author: Marleen de Jonge
Date: 2025
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
from tqdm import tqdm
import re
import argparse

# NLP imports
try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(f"Required packages not installed: {e}. Run: pip install sentence-transformers")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SemanticClimateRiskAnalyzer:
    """
    Semantic climate change risk analyzer that uses semantic search to identify
    risk-related climate discussions and calculates exposure ratios.
    """
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None
        
        # Risk-focused query patterns
        # These capture various ways firms discuss climate-related risks
        self.risk_queries = [
            # Physical risks
            'extreme weather climate physical risk operational disruption',
            'supply chain disruption weather events natural disasters',
            'flooding drought wildfire climate hazards business impact',
            'climate adaptation resilience operational vulnerability',
            'weather related operational disruption facility damage',
            
            # Transition risks
            'stranded assets carbon intensive business model risk',
            'technology disruption energy transition competitive risk',
            'carbon tax transition costs regulatory burden',
            'fossil fuel asset impairment writedown risk',
            'business model transition risk technological obsolescence',
            
            # Regulatory and policy risks
            'climate policy regulatory risk compliance costs',
            'carbon pricing regulatory burden financial impact',
            'environmental regulation compliance risk penalties',
            'climate disclosure requirements regulatory pressure',
            'regulatory uncertainty climate policy changes',
            
            # Financial and market risks
            'climate risk financial performance impact',
            'carbon costs financial burden operational expenses',
            'climate related credit risk financing challenges',
            'ESG rating climate risk investor concerns',
            'climate litigation legal risk regulatory action',
            
            # Reputational and strategic risks
            'climate reputation risk stakeholder pressure',
            'sustainability expectations market pressure competitive disadvantage',
            'climate strategy risk business continuity concerns',
            'environmental reputation damage brand risk'
        ]
        
        # Similarity threshold for risk identification
        self.risk_threshold = 0.40
    
    def _safe_int(self, value):
        """Safely convert value to integer."""
        try:
            if value is None or value == '':
                return 0
            return int(float(value))  # Convert through float to handle string numbers
        except (ValueError, TypeError):
            return 0
    
    def _safe_float(self, value):
        """Safely convert value to float."""
        try:
            if value is None or value == '':
                return 0.0
            return float(value)
        except (ValueError, TypeError):
            return 0.0
        
    def normalize_ticker(self, ticker: str, source_index: str) -> str:
        """
        Normalize ticker format based on stock index.
        For STOXX600, ensure tickers have '-EU' suffix if no dash is present.
        
        Args:
            ticker: Original ticker symbol
            source_index: Stock index name ('STOXX600' or 'SP500')
            
        Returns:
            Normalized ticker string
        """
        if not ticker or not isinstance(ticker, str):
            return ticker
        
        ticker = ticker.strip().upper()
        
        # For STOXX600, add '-EU' suffix if no dash is present
        if source_index == 'STOXX600':
            if '-' not in ticker:
                ticker = f"{ticker}-EU"
        
        return ticker
    
    def load_model(self):
        """Load the sentence transformer model."""
        if self.model is None:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("✅ Model loaded successfully")
    
    def load_climate_snippets(self, data_paths: List[Path]) -> List[Dict[str, Any]]:
        """
        Load climate snippets from JSON files.
        
        Args:
            data_paths: List of paths to climate snippet directories
            
        Returns:
            List of loaded transcript data
        """
        all_transcripts = []
        
        for data_path in data_paths:
            if not data_path.exists():
                logger.warning(f"Data path does not exist: {data_path}")
                continue
                
            logger.info(f"Loading climate snippets from: {data_path}")
            
            # Find all JSON files in the directory
            json_files = list(data_path.glob("*.json"))
            if not json_files:
                logger.warning(f"No JSON files found in: {data_path}")
                continue
            
            for json_file in tqdm(json_files, desc=f"Loading from {data_path.name}"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        file_data = json.load(f)
                    
                    # Handle both single transcript and array of transcripts
                    if isinstance(file_data, list):
                        # Array of transcripts (expected format)
                        for transcript_data in file_data:
                            if isinstance(transcript_data, dict):
                                # Add source information
                                transcript_data['source_file'] = str(json_file)
                                transcript_data['source_index'] = data_path.name
                                all_transcripts.append(transcript_data)
                    elif isinstance(file_data, dict):
                        # Single transcript object
                        file_data['source_file'] = str(json_file)
                        file_data['source_index'] = data_path.name
                        all_transcripts.append(file_data)
                    else:
                        logger.warning(f"Invalid data format in {json_file}")
                        continue
                    
                except Exception as e:
                    logger.error(f"Error loading {json_file}: {e}")
                    continue
        
        logger.info(f"✅ Loaded {len(all_transcripts)} transcripts total")
        return all_transcripts
    
    def semantic_risk_search(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Perform semantic search to identify risk-related climate texts.
        
        Args:
            texts: List of climate text snippets to search
            
        Returns:
            List of risk-related texts with similarity scores
        """
        if not texts:
            return []
        
        paragraphs = [text['text'] for text in texts]
        
        # Ensure model is loaded
        self.load_model()
        
        # Encode texts and risk queries (suppress progress bars)
        text_embeddings = self.model.encode(paragraphs, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        query_embeddings = self.model.encode(self.risk_queries, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        
        risk_related_texts = []
        
        # Calculate similarities between each text and each risk query
        for i, text in enumerate(texts):
            max_similarity = 0.0
            best_query_idx = -1
            
            # Find highest similarity with any risk query
            similarities = np.dot(query_embeddings, text_embeddings[i])
            max_similarity = np.max(similarities)
            best_query_idx = np.argmax(similarities)
            
            if max_similarity >= self.risk_threshold:
                risk_related_texts.append({
                    'text_index': i,
                    'text': text,
                    'similarity_score': float(max_similarity),
                    'best_query': self.risk_queries[best_query_idx],
                    'best_query_index': best_query_idx
                })
        
        return risk_related_texts
    
    def split_text_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex-based approach.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        if not text or not isinstance(text, str):
            return []
        
        # Simple sentence splitting using regex
        # This handles common sentence endings while avoiding issues with abbreviations
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
        
        # Filter out very short sentences (likely splitting errors)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        return sentences
    
    def calculate_climate_risk_exposure(self, transcript_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate climate risk exposure using semantic search and sentence ratios.
        
        Args:
            transcript_data: Single transcript data dictionary
            
        Returns:
            Dictionary of risk exposure measures
        """
        # Extract basic information with safe type conversion
        texts = transcript_data.get('texts', [])
        total_sentences_in_call = self._safe_int(transcript_data.get('total_sentences_in_call', 0))
        
        # If no climate texts, risk exposure is 0
        if not texts or total_sentences_in_call == 0:
            return {
                'climate_risk_exposure': 0.0,
                'risk_related_texts_count': 0,
                'risk_sentences_count': 0,
                'total_climate_texts': len(texts),
                'risk_coverage_ratio': 0.0,
                'avg_risk_similarity': 0.0
            }
        
        # Apply semantic search to find risk-related texts
        risk_related_texts = self.semantic_risk_search(texts)
        
        # Count sentences in risk-related texts
        risk_sentences_count = 0
        similarity_scores = []
        
        for risk_text in risk_related_texts:
            sentences = self.split_text_into_sentences(risk_text['text'])
            risk_sentences_count += len(sentences)
            similarity_scores.append(risk_text['similarity_score'])
        
        # Calculate risk exposure ratio following Sautner et al. approach
        climate_risk_exposure = risk_sentences_count / total_sentences_in_call if total_sentences_in_call > 0 else 0.0
        
        # Calculate additional metrics
        risk_coverage_ratio = len(risk_related_texts) / len(texts) if len(texts) > 0 else 0.0
        avg_risk_similarity = np.mean(similarity_scores) if similarity_scores else 0.0
        
        return {
            'climate_risk_exposure': climate_risk_exposure,
            'risk_related_texts_count': len(risk_related_texts),
            'risk_sentences_count': risk_sentences_count,
            'total_climate_texts': len(texts),
            'risk_coverage_ratio': risk_coverage_ratio,
            'avg_risk_similarity': avg_risk_similarity
        }
    
    def process_transcripts(self, transcripts: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Process all transcripts and calculate risk exposure measures.
        
        Args:
            transcripts: List of transcript data dictionaries
            
        Returns:
            DataFrame with firm-level risk exposure measures
        """
        logger.info(f"Processing {len(transcripts)} transcripts for climate risk exposure...")
        
        results = []
        
        for transcript in tqdm(transcripts, desc="Calculating climate risk exposure"):
            try:
                source_index = transcript.get('source_index', '')
                original_ticker = transcript.get('ticker', '')

                normalized_ticker = self.normalize_ticker(original_ticker, source_index)
                
                # Basic transcript information with data type cleaning
                row = {
                    'file': transcript.get('file', ''),
                    'company_name': transcript.get('company_name', ''),
                    'ticker': normalized_ticker,
                    'quarter': transcript.get('quarter', ''),
                    'year': self._safe_int(transcript.get('year', 0)),
                    'date': transcript.get('date', ''),
                    'source_index': transcript.get('source_index', ''),
                    
                    # Original climate metrics from the data
                    'original_climate_sentence_count': self._safe_int(transcript.get('climate_sentence_count', 0)),
                    'original_climate_sentence_ratio': self._safe_float(transcript.get('climate_sentence_ratio', 0.0)),
                    'total_sentences_in_call': self._safe_int(transcript.get('total_sentences_in_call', 0)),
                    'management_sentences': self._safe_int(transcript.get('management_sentences', 0)),
                    'qa_sentences': self._safe_int(transcript.get('qa_sentences', 0)),
                    'matched_transcript_file': transcript.get('matched_transcript_file', '')
                }
                
                # Calculate climate risk exposure using semantic search
                risk_measures = self.calculate_climate_risk_exposure(transcript)
                row.update(risk_measures)
                
                results.append(row)
                
            except Exception as e:
                logger.error(f"Error processing transcript {transcript.get('file', 'unknown')}: {e}")
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        logger.info(f"✅ Processed {len(df)} transcripts successfully")
        
        return df
    
    def save_results(self, df: pd.DataFrame, output_path: Path):
        """
        Save results to CSV file.
        
        Args:
            df: Results DataFrame
            output_path: Output file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Sort by ticker, year, quarter for better organization
        df_sorted = df.sort_values(['ticker', 'year', 'quarter'], na_position='last')
        
        # Save to CSV
        df_sorted.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"✅ Results saved to: {output_path}")
        
        # Print summary statistics
        self.print_summary_statistics(df_sorted)
    
    def print_summary_statistics(self, df: pd.DataFrame):
        """Print summary statistics of the results."""
        logger.info("\n" + "="*60)
        logger.info("SEMANTIC CLIMATE RISK ANALYSIS SUMMARY")
        logger.info("="*60)
        
        logger.info(f"Total observations: {len(df):,}")
        logger.info(f"Unique firms: {df['ticker'].nunique()}")
        
        # Handle year range safely
        try:
            valid_years = df[df['year'] > 0]['year']
            if len(valid_years) > 0:
                logger.info(f"Year range: {valid_years.min()}-{valid_years.max()}")
            else:
                logger.info("Year range: No valid years found")
        except Exception as e:
            logger.info(f"Year range: Unable to determine ({e})")
            
        logger.info(f"Stock indices: {', '.join(df['source_index'].unique())}")
        
        # Climate risk exposure statistics
        logger.info(f"\nClimate Risk Exposure Statistics:")
        logger.info(f"Mean climate risk exposure ratio: {df['climate_risk_exposure'].mean():.4f}")
        logger.info(f"Median climate risk exposure ratio: {df['climate_risk_exposure'].median():.4f}")
        logger.info(f"Firms with climate risk exposure > 0: {(df['climate_risk_exposure'] > 0).mean():.1%}")
        logger.info(f"Mean risk sentences per call: {df['risk_sentences_count'].mean():.1f}")
        logger.info(f"Mean risk-related texts per call: {df['risk_related_texts_count'].mean():.1f}")
        
        # Semantic search effectiveness
        logger.info(f"\nSemantic Search Effectiveness:")
        logger.info(f"Mean risk coverage ratio (risk texts / total climate texts): {df['risk_coverage_ratio'].mean():.4f}")
        logger.info(f"Mean similarity score for risk texts: {df['avg_risk_similarity'].mean():.4f}")
        logger.info(f"Calls with identified risk content: {(df['risk_related_texts_count'] > 0).mean():.1%}")
        
        # Compare with overall climate measures if available
        if 'original_climate_sentence_ratio' in df.columns:
            try:
                # Calculate correlation between risk exposure and overall climate exposure
                correlation = df['climate_risk_exposure'].corr(df['original_climate_sentence_ratio'])
                logger.info(f"\nComparison with Overall Climate Exposure:")
                logger.info(f"Correlation with original climate ratio: {correlation:.4f}")
                logger.info(f"Risk vs. Overall climate ratio: {df['climate_risk_exposure'].mean():.4f} vs {df['original_climate_sentence_ratio'].mean():.4f}")
                
                # Risk as percentage of overall climate discussion
                non_zero_climate = df[df['original_climate_sentence_ratio'] > 0]
                if len(non_zero_climate) > 0:
                    risk_share = (non_zero_climate['climate_risk_exposure'] / non_zero_climate['original_climate_sentence_ratio']).mean()
                    logger.info(f"Risk discussions as % of climate discussions: {risk_share:.1%}")
            except Exception as e:
                logger.info(f"Could not calculate correlation: {e}")
        
        # Distribution statistics
        logger.info(f"\nDistribution Statistics:")
        logger.info(f"Min climate risk exposure: {df['climate_risk_exposure'].min():.4f}")
        logger.info(f"25th percentile: {df['climate_risk_exposure'].quantile(0.25):.4f}")
        logger.info(f"75th percentile: {df['climate_risk_exposure'].quantile(0.75):.4f}")
        logger.info(f"Max climate risk exposure: {df['climate_risk_exposure'].max():.4f}")
        logger.info(f"Standard deviation: {df['climate_risk_exposure'].std():.4f}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Semantic Climate Risk Analysis for Earnings Calls',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script.py --indices SP500                    # Process only SP500 data
  python script.py --indices STOXX600                 # Process only STOXX600 data  
  python script.py --indices SP500 STOXX600           # Process both (default)
  python script.py --model all-mpnet-base-v2          # Use different model
  python script.py --threshold 0.35                   # Lower similarity threshold
        """
    )
    
    parser.add_argument(
        '--indices', 
        nargs='+', 
        choices=['SP500', 'STOXX600'], 
        default=['SP500', 'STOXX600'],
        help='Stock indices to process (default: both SP500 and STOXX600)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='Sentence transformer model to use (default: all-MiniLM-L6-v2)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.40,
        help='Similarity threshold for risk identification (default: 0.40)'
    )
    
    parser.add_argument(
        '--base-path',
        type=str,
        default="/Users/marleendejonge/Desktop/ECC-data-generation/data/enhanced_climate_snippets",
        help='Base path to climate snippets directories'
    )
    
    parser.add_argument(
        '--output-path',
        type=str,
        default="/Users/marleendejonge/Desktop/ECC-data-generation/outputs/climate_risk_exposure.csv",
        help='Output CSV file path'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Display configuration
    logger.info("="*60)
    logger.info("SEMANTIC CLIMATE RISK ANALYSIS CONFIGURATION")
    logger.info("="*60)
    logger.info(f"Stock indices to process: {', '.join(args.indices)}")
    logger.info(f"Sentence transformer model: {args.model}")
    logger.info(f"Risk similarity threshold: {args.threshold}")
    logger.info(f"Base data path: {args.base_path}")
    logger.info(f"Output file: {args.output_path}")
    logger.info("="*60)
    
    # Define data paths based on selected indices
    base_path = Path(args.base_path)
    data_paths = []
    
    for index in args.indices:
        index_path = base_path / index
        if index_path.exists():
            data_paths.append(index_path)
            logger.info(f"✅ Found data directory for {index}: {index_path}")
        else:
            logger.warning(f"❌ Data directory not found for {index}: {index_path}")
    
    if not data_paths:
        logger.error("No valid data paths found. Please check the data directories.")
        return
    
    # Initialize analyzer with custom parameters
    analyzer = SemanticClimateRiskAnalyzer(model_name=args.model)
    analyzer.risk_threshold = args.threshold
    
    # Load climate snippets from selected indices
    transcripts = analyzer.load_climate_snippets(data_paths)
    
    if not transcripts:
        logger.error("No transcripts loaded. Please check the data files.")
        return
    
    # Process transcripts and calculate risk exposure measures
    results_df = analyzer.process_transcripts(transcripts)
    
    # Generate output filename based on selected indices
    output_path = Path(args.output_path)
    if len(args.indices) == 1:
        # If only one index selected, include it in filename
        stem = output_path.stem
        suffix = output_path.suffix
        output_path = output_path.parent / f"{stem}_{args.indices[0]}{suffix}"
    
    # Save results
    analyzer.save_results(results_df, output_path)


if __name__ == "__main__":
    main()