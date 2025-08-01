#!/usr/bin/env python3
"""
Climate Change Exposure Analysis

This script reads climate snippets and calculates firm-level climate exposure
as the ratio of climate-related sentences to total sentences in earnings calls,
following the Sautner et al. (2023) approach.

Author: Marleen de Jonge
Date: 2025
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import logging
from datetime import datetime
from tqdm import tqdm
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ClimateExposureAnalyzer:
    """
    Climate change exposure analyzer that calculates the ratio of climate-related 
    sentences to total sentences in earnings calls.
    """
    
    def __init__(self):
        pass
    
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
    
    def split_text_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex-based approach.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """

        if not text or not isinstance(text, dict):
            return []
        
        paragraph = text.get('text', [])

        if not paragraph or not isinstance(paragraph, str):
            return []

        # Simple sentence splitting using regex
        # This handles common sentence endings while avoiding issues with abbreviations
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', paragraph.strip())
        
        # Filter out very short sentences (likely splitting errors)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        return sentences
    
    def calculate_climate_exposure(self, transcript_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate climate exposure as ratio of climate sentences to total sentences.
        
        Args:
            transcript_data: Single transcript data dictionary
            
        Returns:
            Dictionary of exposure measures
        """
        # Extract basic information with safe type conversion
        texts = transcript_data.get('texts', [])
        total_sentences_in_call = self._safe_int(transcript_data.get('total_sentences_in_call', 0))

        # If no climate texts, exposure is 0
        if not texts or total_sentences_in_call == 0:
            return {
                'climate_exposure': 0.0,
                'climate_sentences_from_texts': 0,
                'total_sentences_in_call': total_sentences_in_call
            }

        # Count sentences in climate-related texts
        climate_sentences_count = 0
        for text in texts:
            sentences = self.split_text_into_sentences(text)
            climate_sentences_count += len(sentences)

        # Calculate exposure ratio following Sautner et al. approach
        climate_exposure = climate_sentences_count / total_sentences_in_call if total_sentences_in_call > 0 else 0.0
        
        return {
            'climate_exposure': climate_exposure,
            'climate_sentences_from_texts': climate_sentences_count,
            'total_sentences_in_call': total_sentences_in_call
        }
    
    def process_transcripts(self, transcripts: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Process all transcripts and calculate exposure measures.
        
        Args:
            transcripts: List of transcript data dictionaries
            
        Returns:
            DataFrame with firm-level exposure measures
        """
        logger.info(f"Processing {len(transcripts)} transcripts for climate exposure...")
        
        results = []
        
        for transcript in tqdm(transcripts, desc="Calculating climate exposure"):
            try:
                # Basic transcript information with data type cleaning
                source_index = transcript.get('source_index', '')
                original_ticker = transcript.get('ticker', '')
                
                # Normalize ticker based on source index
                normalized_ticker = self.normalize_ticker(original_ticker, source_index)
                
                row = {
                    'file': transcript.get('file', ''),
                    'company_name': transcript.get('company_name', ''),
                    'ticker': normalized_ticker,  # Use normalized ticker
                    'original_ticker': original_ticker,  # Keep original for reference
                    'quarter': transcript.get('quarter', ''),
                    'year': self._safe_int(transcript.get('year', 0)),
                    'date': transcript.get('date', ''),
                    'source_index': source_index,
                    
                    # Original climate metrics from the data
                    'original_climate_sentence_count': self._safe_int(transcript.get('climate_sentence_count', 0)),
                    'original_climate_sentence_ratio': self._safe_float(transcript.get('climate_sentence_ratio', 0.0)),
                    'management_sentences': self._safe_int(transcript.get('management_sentences', 0)),
                    'qa_sentences': self._safe_int(transcript.get('qa_sentences', 0)),
                    'matched_transcript_file': transcript.get('matched_transcript_file', '')
                }
        
                # Calculate climate exposure from texts
                exposure_measures = self.calculate_climate_exposure(transcript)
                row.update(exposure_measures)
                
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
        logger.info("CLIMATE EXPOSURE ANALYSIS SUMMARY")
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
        
        # Climate exposure statistics
        logger.info(f"\nClimate Exposure Statistics:")
        logger.info(f"Mean climate exposure ratio: {df['climate_exposure'].mean():.4f}")
        logger.info(f"Median climate exposure ratio: {df['climate_exposure'].median():.4f}")
        logger.info(f"Firms with climate exposure > 0: {(df['climate_exposure'] > 0).mean():.1%}")
        logger.info(f"Mean climate sentences per call: {df['climate_sentences_from_texts'].mean():.1f}")
        logger.info(f"Mean total sentences per call: {df['total_sentences_in_call'].mean():.1f}")
        
        # Compare with original measures if available
        if 'original_climate_sentence_ratio' in df.columns:
            try:
                correlation = df['climate_exposure'].corr(df['original_climate_sentence_ratio'])
                logger.info(f"\nComparison with original measures:")
                logger.info(f"Correlation with original climate ratio: {correlation:.4f}")
                logger.info(f"Mean original climate ratio: {df['original_climate_sentence_ratio'].mean():.4f}")
            except Exception as e:
                logger.info(f"Could not calculate correlation: {e}")
        
        # Distribution statistics
        logger.info(f"\nDistribution Statistics:")
        logger.info(f"Min climate exposure: {df['climate_exposure'].min():.4f}")
        logger.info(f"25th percentile: {df['climate_exposure'].quantile(0.25):.4f}")
        logger.info(f"75th percentile: {df['climate_exposure'].quantile(0.75):.4f}")
        logger.info(f"Max climate exposure: {df['climate_exposure'].max():.4f}")
        logger.info(f"Standard deviation: {df['climate_exposure'].std():.4f}")


def main():
    """Main execution function."""
    # Define data paths
    sp500_path = Path("/Users/marleendejonge/Desktop/ECC-data-generation/data/enhanced_climate_snippets/SP500")
    stoxx600_path = Path("/Users/marleendejonge/Desktop/ECC-data-generation/data/enhanced_climate_snippets/STOXX600")
    
    # Output path
    output_path = Path("/Users/marleendejonge/Desktop/ECC-data-generation/outputs/climate_exposure_ratios.csv")
    
    # Initialize analyzer
    analyzer = ClimateExposureAnalyzer()
    
    # Load climate snippets
    data_paths = []
    if sp500_path.exists():
        data_paths.append(sp500_path)
    if stoxx600_path.exists():
        data_paths.append(stoxx600_path)
    
    if not data_paths:
        logger.error("No valid data paths found. Please check the data directories.")
        return
    
    transcripts = analyzer.load_climate_snippets(data_paths)
    
    if not transcripts:
        logger.error("No transcripts loaded. Please check the data files.")
        return
    
    # Process transcripts and calculate exposure measures
    results_df = analyzer.process_transcripts(transcripts)
    
    # Save results
    analyzer.save_results(results_df, output_path)


if __name__ == "__main__":
    main()