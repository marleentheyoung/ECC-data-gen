#!/usr/bin/env python3
"""
OPTIMIZED Semantic Climate Change Risk Analysis for M3 MacBook Pro

This script reads climate snippets and applies semantic search to identify 
risk-related paragraphs, then calculates firm-level climate risk exposure
as the ratio of risk-related sentences to total sentences in earnings calls.

Optimizations for M3 MacBook Pro:
- Apple Silicon GPU (MPS) acceleration
- Batch processing for embeddings
- Pre-computed query embeddings
- Vectorized similarity calculations
- Memory-efficient processing
- Optimized sentence splitting

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
from functools import lru_cache

# Add this to check which part is slow
import time

# NLP imports
try:
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError as e:
    raise ImportError(f"Required packages not installed: {e}. Run: pip install sentence-transformers torch")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pre-compile regex patterns for performance
SENTENCE_PATTERN = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


class OptimizedSemanticClimateRiskAnalyzer:
    """
    OPTIMIZED semantic climate change risk analyzer with M3 MacBook Pro specific optimizations.
    """
    
    def __init__(self, model_name: str = 'sentence-transformers/all-mpnet-base-v2'):
        self.model_name = model_name
        self.model = None
        self.query_embeddings = None  # Pre-computed query embeddings
        self.device = self._get_optimal_device()
        
        # Risk-focused query patterns (optimized set)
        self.risk_queries = [
            # Physical risks
            'extreme weather climate physical risk operational disruption',
            'supply chain disruption weather events natural disasters',
            'flooding drought wildfire hurricane climate hazards business impact',
            'climate adaptation resilience operational vulnerability infrastructure',
            'weather related operational disruption facility damage costs',
            
            # Transition risks
            'stranded assets carbon intensive business model risk',
            'technology disruption energy transition competitive risk obsolescence',
            'carbon tax transition costs regulatory burden pricing',
            'fossil fuel asset impairment writedown risk devaluation',
            'business model transition risk technological change disruption',
            
            # Regulatory and policy risks
            'climate policy regulatory risk compliance costs burden',
            'carbon pricing regulatory financial impact costs',
            'environmental regulation compliance risk penalties fines',
            'climate disclosure requirements regulatory pressure mandates',
            'regulatory uncertainty climate policy changes government',
            
            # Financial and market risks
            'climate risk financial performance impact costs expenses',
            'carbon costs financial burden operational expenses pricing',
            'climate related credit risk financing challenges access',
            'ESG rating climate risk investor concerns capital',
            'climate litigation legal risk regulatory action lawsuits',
            
            # Reputational and strategic risks
            'climate reputation risk stakeholder pressure brand damage',
            'sustainability expectations market pressure competitive disadvantage',
            'climate strategy risk business continuity concerns planning',
            'environmental reputation damage brand risk consumer pressure'
        ]
        
        # Optimized parameters for M3
        self.risk_threshold = 0.40
        self.batch_size = 128  # Larger batch size for M3's unified memory
        self.max_chunk_size = 2000  # Process in chunks to manage memory
    
    def _get_optimal_device(self) -> str:
        """Determine the best device for M3 MacBook Pro."""
        # Force MPS usage if available
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            logger.info("🚀 Using Apple Silicon GPU (MPS) acceleration")
            return 'mps'
        elif torch.cuda.is_available():
            logger.info("🚀 Using CUDA GPU acceleration")
            return 'cuda'
        else:
            logger.info("⚡ Using CPU (MPS not available)")
            return 'cpu'
    
    def _safe_int(self, value):
        """Safely convert value to integer."""
        try:
            if value is None or value == '':
                return 0
            return int(float(value))
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
        """Normalize ticker format based on stock index."""
        if not ticker or not isinstance(ticker, str):
            return ticker
        
        ticker = ticker.strip().upper()
        
        # For STOXX600, add '-EU' suffix if no dash is present
        if source_index == 'STOXX600':
            if '-' not in ticker:
                ticker = f"{ticker}-EU"
        
        return ticker
    
    def load_model(self):
        """Load the sentence transformer model with M3 optimizations."""
        if self.model is None:
            logger.info(f"Loading model: {self.model_name} on device: {self.device}")
            
            # Load model and explicitly move to device
            self.model = SentenceTransformer(self.model_name)
            
            # Explicitly move model to MPS device
            if self.device == 'mps':
                try:
                    # Move the underlying PyTorch model to MPS
                    self.model = self.model.to(self.device)
                    logger.info("✅ Model successfully moved to MPS device")
                except Exception as e:
                    logger.warning(f"Failed to move model to MPS: {e}")
                    logger.info("Falling back to CPU")
                    self.device = 'cpu'
                    self.model = self.model.to('cpu')
            
            # Pre-compute query embeddings for efficiency
            logger.info("Pre-computing risk query embeddings...")
            self.query_embeddings = self.model.encode(
                self.risk_queries,
                batch_size=len(self.risk_queries),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
                device=self.device  # Ensure encoding uses correct device
            )
            
            logger.info(f"✅ Model loaded with {len(self.risk_queries)} pre-computed risk queries on {self.device}")
            
            # Verify device usage
            if hasattr(self.model, 'device'):
                logger.info(f"🔍 Model device: {self.model.device}")
            
            # Clear any cached memory
            if self.device == 'mps':
                try:
                    torch.mps.empty_cache()
                    logger.info("🧹 Cleared MPS cache")
                except:
                    pass
    
    def load_climate_snippets(self, data_paths: List[Path]) -> List[Dict[str, Any]]:
        """Load climate snippets from JSON files with optimized I/O."""
        all_transcripts = []
        
        for data_path in data_paths:
            if not data_path.exists():
                logger.warning(f"Data path does not exist: {data_path}")
                continue
                
            logger.info(f"Loading climate snippets from: {data_path}")
            
            json_files = list(data_path.glob("*.json"))
            if not json_files:
                logger.warning(f"No JSON files found in: {data_path}")
                continue
            
            # Process files with progress bar
            for json_file in tqdm(json_files, desc=f"Loading {data_path.name}", unit="files"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        file_data = json.load(f)
                    
                    # Handle both single transcript and array of transcripts
                    if isinstance(file_data, list):
                        for transcript_data in file_data:
                            if isinstance(transcript_data, dict):
                                transcript_data['source_file'] = str(json_file)
                                transcript_data['source_index'] = data_path.name
                                all_transcripts.append(transcript_data)
                    elif isinstance(file_data, dict):
                        file_data['source_file'] = str(json_file)
                        file_data['source_index'] = data_path.name
                        all_transcripts.append(file_data)
                    
                except Exception as e:
                    logger.error(f"Error loading {json_file}: {e}")
                    continue
        
        logger.info(f"✅ Loaded {len(all_transcripts)} transcripts total")
        return all_transcripts
    
    def reload_model(self):
        """Reload the model to clear all internal state and caches."""
        logger.info("🔄 Reloading model to clear memory leaks...")
        
        # Clear caches first
        if hasattr(self, 'split_text_into_sentences_cached'):
            self.split_text_into_sentences_cached.cache_clear()
            logger.info("🧹 Cleared sentence splitting cache")
        
        # Clear MPS cache
        if self.device == 'mps':
            try:
                torch.mps.empty_cache()
                logger.info("🧹 Cleared MPS cache")
            except:
                pass
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Reset model and query embeddings
        self.model = None
        self.query_embeddings = None
        
        # Reload everything
        self.load_model()
        logger.info("✅ Model reloaded successfully")
    
    def aggressive_memory_cleanup(self):
        """Perform aggressive memory cleanup without reloading model."""
        # Clear sentence splitting cache
        if hasattr(self, 'split_text_into_sentences_cached'):
            self.split_text_into_sentences_cached.cache_clear()
        
        # Clear MPS cache
        if self.device == 'mps':
            try:
                torch.mps.empty_cache()
            except:
                pass
        
        # Force garbage collection
        import gc
        gc.collect()
        
        logger.info("🧹 Aggressive memory cleanup completed")
    
    @lru_cache(maxsize=500)  # Reduced cache size to prevent memory buildup
    def split_text_into_sentences_cached(self, text: str) -> tuple:
        """Cached sentence splitting using pre-compiled regex."""
        if not text or not isinstance(text, str):
            return ()
        
        sentences = SENTENCE_PATTERN.split(text.strip())
        sentences = tuple(s.strip() for s in sentences if len(s.strip()) > 10)
        return sentences
    
    def split_text_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using optimized cached approach."""
        return list(self.split_text_into_sentences_cached(text))
        """
        OPTIMIZED batch semantic search to identify risk-related climate texts.
        Uses vectorized operations and pre-computed query embeddings.
        """
        if not texts:
            return []
        
        # Extract text content
        paragraphs = [text.get('text', '') for text in texts if text.get('text', '').strip()]
        
        if not paragraphs:
            return []
        
        # Ensure model is loaded
        self.load_model()
        
        # Batch encode all texts at once (major performance improvement)
        try:
            text_embeddings = self.model.encode(
                paragraphs,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
                device=self.device  # Explicitly specify device
            )
            
            # Log device usage for debugging
            if len(paragraphs) > 100:  # Only log for larger batches
                logger.debug(f"Encoded {len(paragraphs)} texts on {self.device}")
                
        except Exception as e:
            logger.error(f"Error during encoding: {e}")
            logger.info("Falling back to CPU encoding")
            # Fallback to CPU if MPS fails
            self.device = 'cpu'
            self.model = self.model.to('cpu')
            text_embeddings = self.model.encode(
                paragraphs,
                batch_size=min(32, self.batch_size),  # Smaller batch for CPU
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False
            )
        
        # Vectorized similarity calculation (much faster than loops)
        # Shape: (n_queries, n_texts)
        similarity_matrix = np.dot(self.query_embeddings, text_embeddings.T)
        
        # Find best matches for each text
        max_similarities = np.max(similarity_matrix, axis=0)
        best_query_indices = np.argmax(similarity_matrix, axis=0)
        
        # Filter texts above threshold
        risk_related_texts = []
        valid_text_idx = 0
        
        for i, text in enumerate(texts):
            if not text.get('text', '').strip():
                continue
                
            similarity = max_similarities[valid_text_idx]
            
            if similarity >= self.risk_threshold:
                risk_related_texts.append({
                    'text_index': i,
                    'text': text,
                    'similarity_score': float(similarity),
                    'best_query': self.risk_queries[best_query_indices[valid_text_idx]],
                    'best_query_index': int(best_query_indices[valid_text_idx])
                })
            
            valid_text_idx += 1
        
        return risk_related_texts
    
    def calculate_climate_risk_exposure(self, transcript_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate climate risk exposure with optimized processing."""
        texts = transcript_data.get('texts', [])
        total_sentences_in_call = self._safe_int(transcript_data.get('total_sentences_in_call', 0))
        
        if not texts or total_sentences_in_call == 0:
            return {
                'climate_risk_exposure': 0.0,
                'risk_related_texts_count': 0,
                'risk_sentences_count': 0,
                'total_climate_texts': len(texts),
                'risk_coverage_ratio': 0.0,
                'avg_risk_similarity': 0.0
            }
        
        # Process texts in chunks if too large
        if len(texts) > self.max_chunk_size:
            risk_related_texts = []
            for i in range(0, len(texts), self.max_chunk_size):
                chunk = texts[i:i + self.max_chunk_size]
                chunk_risks = self.semantic_risk_search_batch(chunk)
                # Adjust indices for the chunk offset
                for risk_text in chunk_risks:
                    risk_text['text_index'] += i
                risk_related_texts.extend(chunk_risks)
        else:
            risk_related_texts = self.semantic_risk_search_batch(texts)
        
        # Count sentences in risk-related texts (vectorized where possible)
        risk_sentences_count = 0
        similarity_scores = []
        
        for risk_text in risk_related_texts:
            text_content = risk_text['text'].get('text', '')
            sentences = self.split_text_into_sentences(text_content)
            risk_sentences_count += len(sentences)
            similarity_scores.append(risk_text['similarity_score'])
        
        # Calculate metrics
        climate_risk_exposure = risk_sentences_count / total_sentences_in_call if total_sentences_in_call > 0 else 0.0
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
    
    def detect_resume_point(self, output_path: Path, transcripts: List[Dict[str, Any]]) -> int:
        """
        Detect where to resume processing by checking existing output file.
        
        Args:
            output_path: Path to the output CSV file
            transcripts: List of all transcripts to process
            
        Returns:
            Index to start processing from (0 if no existing file)
        """
        if not output_path.exists():
            logger.info("📄 No existing output file found - starting from beginning")
            return 0
        
        try:
            # Read existing CSV to find processed transcripts
            existing_df = pd.read_csv(output_path, encoding='utf-8')
            
            if existing_df.empty:
                logger.info("📄 Existing file is empty - starting from beginning")
                return 0
            
            processed_files = set(existing_df['file'].tolist())
            logger.info(f"📄 Found existing file with {len(existing_df)} processed transcripts")
            
            # Find where to resume
            resume_index = 0
            for i, transcript in enumerate(transcripts):
                transcript_file = transcript.get('file', '')
                if transcript_file not in processed_files:
                    resume_index = i
                    break
            else:
                # All transcripts are already processed
                logger.info("✅ All transcripts already processed!")
                return len(transcripts)
            
            skipped_count = resume_index
            remaining_count = len(transcripts) - resume_index
            
            logger.info(f"🔄 RESUME MODE:")
            logger.info(f"   📊 Already processed: {skipped_count:,} transcripts")
            logger.info(f"   ⏭️ Will skip to index: {resume_index}")
            logger.info(f"   🚀 Remaining to process: {remaining_count:,} transcripts")
            
            return resume_index
            
        except Exception as e:
            logger.error(f"Error reading existing file: {e}")
            logger.info("Starting from beginning due to read error")
            return 0
    
    def save_batch_to_csv(self, batch_results: List[Dict], output_path: Path, is_first_batch: bool = False):
        """Save a batch of results to CSV file."""
        if not batch_results:
            return
        
        # Convert batch to DataFrame
        batch_df = pd.DataFrame(batch_results)
        
        # Sort batch for consistency
        batch_df = batch_df.sort_values(['ticker', 'year', 'quarter'], na_position='last')
        
        # Save to CSV
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if is_first_batch and not output_path.exists():
            # First batch: create new file with headers
            batch_df.to_csv(output_path, index=False, encoding='utf-8', mode='w')
            logger.info(f"💾 Created output file: {output_path}")
        else:
            # Subsequent batches or resume: append without headers
            batch_df.to_csv(output_path, index=False, encoding='utf-8', mode='a', header=False)
        
        logger.info(f"💾 Saved batch of {len(batch_results)} results to {output_path.name}")
    
    def process_transcripts_iterative(self, transcripts: List[Dict[str, Any]], output_path: Path, batch_size: int = 100, resume: bool = False) -> int:
        """
        Process transcripts with iterative saving, model reloading, and resume capability.
        
        Args:
            transcripts: List of transcript data
            output_path: Path to save results
            batch_size: Number of transcripts to process before saving
            resume: Whether to resume from existing file
            
        Returns:
            Number of successfully processed transcripts
        """
        
        # Determine starting point
        if resume:
            start_index = self.detect_resume_point(output_path, transcripts)
            if start_index >= len(transcripts):
                logger.info("🎉 All transcripts already processed!")
                return len(transcripts)
            
            # Slice transcripts to remaining ones
            remaining_transcripts = transcripts[start_index:]
            logger.info(f"🔄 Resuming from transcript {start_index + 1}")
        else:
            remaining_transcripts = transcripts
            start_index = 0
            logger.info(f"🚀 Processing all {len(transcripts)} transcripts from beginning...")
        
        logger.info(f"📊 Batch size: {batch_size} (saves every {batch_size} transcripts)")
        logger.info(f"🔄 Model reload: Every 1000 transcripts to prevent memory leaks")
        
        batch_results = []
        total_processed = 0
        cleanup_interval = 50  # Clear cache every 50 transcripts
        model_reload_interval = 1000  # Reload model every 1000 transcripts
        
        # Process remaining transcripts with progress bar
        for i, transcript in enumerate(tqdm(remaining_transcripts, desc="Processing transcripts", unit="transcripts")):
            try:
                # Calculate actual transcript number (for model reload timing)
                actual_transcript_num = start_index + i + 1
                
                # Reload model periodically to prevent memory leaks
                if actual_transcript_num % model_reload_interval == 0 and actual_transcript_num > 0:
                    self.reload_model()
                    logger.info(f"🔄 Model reloaded after {actual_transcript_num} total transcripts")
                
                source_index = transcript.get('source_index', '')
                original_ticker = transcript.get('ticker', '')
                normalized_ticker = self.normalize_ticker(original_ticker, source_index)
                
                # Basic transcript information
                row = {
                    'file': transcript.get('file', ''),
                    'company_name': transcript.get('company_name', ''),
                    'ticker': normalized_ticker,
                    'quarter': transcript.get('quarter', ''),
                    'year': self._safe_int(transcript.get('year', 0)),
                    'date': transcript.get('date', ''),
                    'source_index': source_index,
                    
                    # Original climate metrics
                    'original_climate_sentence_count': self._safe_int(transcript.get('climate_sentence_count', 0)),
                    'original_climate_sentence_ratio': self._safe_float(transcript.get('climate_sentence_ratio', 0.0)),
                    'total_sentences_in_call': self._safe_int(transcript.get('total_sentences_in_call', 0)),
                    'management_sentences': self._safe_int(transcript.get('management_sentences', 0)),
                    'qa_sentences': self._safe_int(transcript.get('qa_sentences', 0)),
                    'matched_transcript_file': transcript.get('matched_transcript_file', '')
                }
                
                # Calculate climate risk exposure
                risk_measures = self.calculate_climate_risk_exposure(transcript)
                row.update(risk_measures)
                
                batch_results.append(row)
                total_processed += 1
                
                # Save batch when it reaches batch_size
                if len(batch_results) >= batch_size:
                    # For resume mode, never treat as first batch (file already exists)
                    is_first_batch = not resume and (start_index + total_processed == len(batch_results))
                    self.save_batch_to_csv(batch_results, output_path, is_first_batch)
                    batch_results = []  # Clear batch to free memory
                    
                    # Aggressive memory cleanup after each batch
                    self.aggressive_memory_cleanup()
                    logger.info(f"🧹 Memory cleared after processing {start_index + total_processed} total transcripts")
                
                # Periodic cache clearing (more frequent than batch saving)
                elif (i + 1) % cleanup_interval == 0:
                    self.aggressive_memory_cleanup()
                
            except Exception as e:
                logger.error(f"Error processing transcript {transcript.get('file', 'unknown')}: {e}")
                continue
        
        # Save any remaining results in the final batch
        if batch_results:
            is_first_batch = not resume and (start_index + total_processed == len(batch_results))
            self.save_batch_to_csv(batch_results, output_path, is_first_batch)
            logger.info(f"💾 Saved final batch of {len(batch_results)} results")
        
        final_total = start_index + total_processed
        logger.info(f"✅ Successfully processed {total_processed} new transcripts")
        logger.info(f"📊 Total transcripts in file: {final_total}")
        logger.info(f"📁 Results saved to: {output_path}")
        
        return final_total
    
    def load_final_results_for_summary(self, output_path: Path) -> pd.DataFrame:
        """Load the complete results from CSV for summary statistics."""
        try:
            df = pd.read_csv(output_path, encoding='utf-8')
            logger.info(f"📊 Loaded {len(df)} rows from {output_path.name} for summary")
            return df
        except Exception as e:
            logger.error(f"Error loading results for summary: {e}")
            return pd.DataFrame()
    
    def save_results(self, output_path: Path):
        """Load results and print summary statistics."""
        # Load the saved results for summary
        df = self.load_final_results_for_summary(output_path)
        
        if not df.empty:
            # Print summary
            self.print_summary_statistics(df)
        else:
            logger.warning("No data available for summary statistics")
    
    def print_summary_statistics(self, df: pd.DataFrame):
        """Print comprehensive summary statistics."""
        logger.info("\n" + "="*70)
        logger.info("🎯 OPTIMIZED SEMANTIC CLIMATE RISK ANALYSIS SUMMARY")
        logger.info("="*70)
        
        logger.info(f"📊 Dataset Overview:")
        logger.info(f"   Total observations: {len(df):,}")
        logger.info(f"   Unique firms: {df['ticker'].nunique():,}")
        
        # Year range
        try:
            valid_years = df[df['year'] > 0]['year']
            if len(valid_years) > 0:
                logger.info(f"   Year range: {valid_years.min()}-{valid_years.max()}")
        except:
            logger.info("   Year range: Unable to determine")
            
        logger.info(f"   Stock indices: {', '.join(df['source_index'].unique())}")
        
        # Climate risk exposure statistics
        logger.info(f"\n🎯 Climate Risk Exposure Analysis:")
        logger.info(f"   Mean climate risk exposure: {df['climate_risk_exposure'].mean():.4f}")
        logger.info(f"   Median climate risk exposure: {df['climate_risk_exposure'].median():.4f}")
        logger.info(f"   Firms with climate risk exposure > 0: {(df['climate_risk_exposure'] > 0).mean():.1%}")
        logger.info(f"   Mean risk sentences per call: {df['risk_sentences_count'].mean():.1f}")
        logger.info(f"   Mean risk-related texts per call: {df['risk_related_texts_count'].mean():.1f}")
        
        # Semantic search effectiveness
        logger.info(f"\n🔍 Semantic Search Performance:")
        logger.info(f"   Risk coverage ratio: {df['risk_coverage_ratio'].mean():.3f}")
        logger.info(f"   Average similarity score: {df['avg_risk_similarity'].mean():.3f}")
        logger.info(f"   Calls with risk content: {(df['risk_related_texts_count'] > 0).mean():.1%}")
        
        # Comparison with overall climate measures
        if 'original_climate_sentence_ratio' in df.columns:
            try:
                correlation = df['climate_risk_exposure'].corr(df['original_climate_sentence_ratio'])
                logger.info(f"\n📈 Risk vs Overall Climate Comparison:")
                logger.info(f"   Correlation: {correlation:.3f}")
                logger.info(f"   Risk ratio: {df['climate_risk_exposure'].mean():.4f}")
                logger.info(f"   Overall climate ratio: {df['original_climate_sentence_ratio'].mean():.4f}")
                
                non_zero_climate = df[df['original_climate_sentence_ratio'] > 0]
                if len(non_zero_climate) > 0:
                    risk_share = (non_zero_climate['climate_risk_exposure'] / 
                                non_zero_climate['original_climate_sentence_ratio']).mean()
                    logger.info(f"   Risk as % of climate discussion: {risk_share:.1%}")
            except:
                pass
        
        # Distribution
        logger.info(f"\n📊 Distribution Statistics:")
        logger.info(f"   Min: {df['climate_risk_exposure'].min():.4f}")
        logger.info(f"   25th percentile: {df['climate_risk_exposure'].quantile(0.25):.4f}")
        logger.info(f"   75th percentile: {df['climate_risk_exposure'].quantile(0.75):.4f}")
        logger.info(f"   Max: {df['climate_risk_exposure'].max():.4f}")
        logger.info(f"   Std dev: {df['climate_risk_exposure'].std():.4f}")
        
        logger.info("="*70)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='OPTIMIZED Semantic Climate Risk Analysis for M3 MacBook Pro',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🚀 M3 MacBook Pro Optimizations:
  - Apple Silicon GPU (MPS) acceleration
  - Batch processing for 10-50x speed improvement
  - Pre-computed query embeddings
  - Vectorized similarity calculations
  - Memory-efficient processing

Examples:
  # Test with small subset
  python script.py --test                             # Process first 100 transcripts
  # Resume from existing file
  python script.py --resume --indices SP500              # Continue where you left off
  python script.py --resume --save-batch-size 100        # Resume with smaller batches
  
  # Regular usage
  python script.py --indices SP500                    # Process SP500 only
  python script.py --indices STOXX600                 # Process STOXX600 only  
  python script.py --indices SP500 STOXX600           # Process both (default)
  python script.py --model all-mpnet-base-v2          # Use higher quality model
  python script.py --threshold 0.35 --batch-size 256  # Custom parameters
  python script.py --save-batch-size 1000             # Save every 1000 transcripts
        """
    )
    
    parser.add_argument(
        '--indices', 
        nargs='+', 
        choices=['SP500', 'STOXX600'], 
        default=['SP500', 'STOXX600'],
        help='Stock indices to process (default: both)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='sentence-transformers/all-mpnet-base-v2',
        help='Sentence transformer model (default: all-mpnet-base-v2, optimized for M3)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.40,
        help='Similarity threshold for risk identification (default: 0.40)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size for encoding (default: 128, optimized for M3)'
    )
    
    parser.add_argument(
        '--base-path',
        type=str,
        default="outputs/enhanced_climate_snippets",
        help='Base path to enhanced climate snippets'
    )
    
    parser.add_argument(
        '--output-path',
        type=str,
        default="outputs/climate_risk_exposure.csv",
        help='Output CSV file path'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of transcripts for testing (e.g., --limit 50)'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run in test mode (process only first 100 transcripts)'
    )
    
    parser.add_argument(
        '--save-batch-size',
        type=int,
        default=500,
        help='Number of transcripts to process before saving (default: 500)'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume processing from existing output file'
    )
    
    return parser.parse_args()


def main():
    """Main execution function with M3 optimizations."""
    args = parse_arguments()
    
    # Display optimized configuration
    logger.info("🚀" + "="*69)
    logger.info("🚀 OPTIMIZED SEMANTIC CLIMATE RISK ANALYSIS (M3 MacBook Pro)")
    logger.info("🚀" + "="*69)
    logger.info(f"📊 Stock indices: {', '.join(args.indices)}")
    logger.info(f"🤖 Model: {args.model}")
    logger.info(f"🎯 Risk threshold: {args.threshold}")
    logger.info(f"⚡ Batch size: {args.batch_size}")
    logger.info(f"💾 Save every: {args.save_batch_size} transcripts")
    logger.info(f"🔄 Model reload: Every 1000 transcripts")
    logger.info(f"🔄 Resume mode: {'Enabled' if args.resume else 'Disabled'}")
    logger.info(f"📁 Data path: {args.base_path}")
    logger.info(f"💾 Output: {args.output_path}")
    logger.info("🚀" + "="*69)
    
    # Setup data paths
    base_path = Path(args.base_path)
    data_paths = []
    
    for index in args.indices:
        index_path = base_path / index
        if index_path.exists():
            data_paths.append(index_path)
            logger.info(f"✅ Found data for {index}: {index_path}")
        else:
            logger.warning(f"❌ Data not found for {index}: {index_path}")
    
    if not data_paths:
        logger.error("❌ No valid data paths found!")
        return
    
    # Initialize optimized analyzer
    analyzer = OptimizedSemanticClimateRiskAnalyzer(model_name=args.model)
    analyzer.risk_threshold = args.threshold
    analyzer.batch_size = args.batch_size
    
    # Load and process
    logger.info("📥 Loading climate snippets...")
    transcripts = analyzer.load_climate_snippets(data_paths)
    
    if not transcripts:
        logger.error("❌ No transcripts loaded!")
        return
    
    # Apply test/limit filtering
    original_count = len(transcripts)
    if args.test:
        transcripts = transcripts[:100]
        logger.info(f"🧪 TEST MODE: Processing {len(transcripts)} of {original_count:,} transcripts")
    elif args.limit:
        transcripts = transcripts[:args.limit]
        logger.info(f"🔬 LIMITED MODE: Processing {len(transcripts)} of {original_count:,} transcripts")
    else:
        logger.info(f"🚀 FULL MODE: Processing all {len(transcripts):,} transcripts")
    
    logger.info("⚡ Starting optimized iterative processing...")
    
    # Generate output filename first
    output_path = Path(args.output_path)
    
    # Add test/limit suffix to filename
    if args.test:
        stem = output_path.stem
        suffix = output_path.suffix
        output_path = output_path.parent / f"{stem}_test{suffix}"
    elif args.limit:
        stem = output_path.stem
        suffix = output_path.suffix
        output_path = output_path.parent / f"{stem}_limit{args.limit}{suffix}"
    
    # Add index suffix if only one index
    if len(args.indices) == 1:
        stem = output_path.stem
        suffix = output_path.suffix
        output_path = output_path.parent / f"{stem}_{args.indices[0]}{suffix}"
    
    # Process transcripts with iterative saving
    processed_count = analyzer.process_transcripts_iterative(
        transcripts, 
        output_path, 
        batch_size=args.save_batch_size,
        resume=args.resume
    )
    
    logger.info(f"🎉 Processing complete! Processed {processed_count} transcripts")
    
    # Generate summary statistics from saved file
    logger.info("📊 Generating summary statistics...")
    analyzer.save_results(output_path)
    
    logger.info("🎉 Optimized analysis complete!")


if __name__ == "__main__":
    main()