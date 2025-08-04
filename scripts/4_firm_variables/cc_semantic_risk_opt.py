#!/usr/bin/env python3
"""
Semantic Climate Risk Analysis Pipeline

Clean implementation of semantic search for climate risk exposure in earnings calls.
Processes enhanced climate snippets and calculates firm-level risk exposure measures.

Author: Marleen de Jonge
Date: 2025
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from tqdm import tqdm
import argparse
import gc
import hashlib
import time

try:
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError as e:
    raise ImportError(f"Required packages not installed: {e}")

# Setup minimal logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SemanticClimateRiskAnalyzer:
    """
    Semantic climate risk analyzer using sentence transformers.
    """
    
    def __init__(self, model_name: str = 'sentence-transformers/all-mpnet-base-v2', threshold: float = 0.40):
        self.model_name = model_name
        self.threshold = threshold
        self.model = None
        self.query_embeddings = None
        self.device = self._get_device()
        self.batch_size = 32  # Reduced for better memory management
        
        # Risk-focused query patterns based on your research
        self.risk_queries = [
            # Physical risks
            'extreme weather climate physical risk operational disruption',
            'supply chain disruption weather events natural disasters',
            'flooding drought wildfire hurricane climate hazards business impact',
            'weather related operational disruption facility damage costs',
            
            # Transition risks
            'stranded assets carbon intensive business model risk',
            'technology disruption energy transition competitive risk',
            'carbon tax transition costs regulatory burden pricing',
            'fossil fuel asset impairment writedown risk devaluation',
            'business model transition risk technological change',
            
            # Regulatory and policy risks
            'climate policy regulatory risk compliance costs burden',
            'carbon pricing regulatory financial impact costs',
            'environmental regulation compliance risk penalties',
            'climate disclosure requirements regulatory pressure',
            'regulatory uncertainty climate policy changes',
            
            # Financial and market risks
            'climate risk financial performance impact costs expenses',
            'carbon costs financial burden operational expenses',
            'climate related credit risk financing challenges',
            'ESG rating climate risk investor concerns',
            'climate litigation legal risk regulatory action',
            
            # Reputational and strategic risks
            'climate reputation risk stakeholder pressure brand damage',
            'sustainability expectations market pressure competitive',
            'climate strategy risk business continuity concerns'
        ]
    
    def _get_device(self) -> str:
        """Get optimal device (prioritize MPS for M3)."""
        if torch.backends.mps.is_available():
            logger.info("Using Apple Silicon GPU (MPS)")
            return 'mps'
        elif torch.cuda.is_available():
            logger.info("Using CUDA GPU")
            return 'cuda'
        else:
            logger.info("Using CPU")
            return 'cpu'
    
    def load_model(self):
        """Load sentence transformer model."""
        if self.model is None:
            logger.info(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.model = self.model.to(self.device)
            
            # Pre-compute query embeddings
            logger.info("Pre-computing risk query embeddings...")
            self.query_embeddings = self.model.encode(
                self.risk_queries,
                batch_size=len(self.risk_queries),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
                device=self.device
            )
            logger.info(f"Model loaded with {len(self.risk_queries)} risk queries")
    
    def load_climate_snippets(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load enhanced climate snippets from JSON file."""
        logger.info(f"Loading climate snippets from: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both single transcript and array formats
        if isinstance(data, list):
            transcripts = data
        else:
            transcripts = [data]
        
        # Add unique keys for each transcript
        for transcript in transcripts:
            transcript['transcript_key'] = self._create_transcript_key(transcript)
        
        logger.info(f"Loaded {len(transcripts)} transcripts")
        return transcripts
    
    def _create_transcript_key(self, transcript: Dict[str, Any]) -> str:
        """Create unique key for transcript."""
        ticker = str(transcript.get('ticker', '')).upper().strip()
        year = str(transcript.get('year', 0))
        quarter = str(transcript.get('quarter', '')).strip()
        company = str(transcript.get('company_name', '')).strip()
        
        composite_key = f"{ticker}_{year}_{quarter}_{company}"
        key_hash = hashlib.md5(composite_key.encode('utf-8')).hexdigest()[:8]
        return f"{ticker}_{year}_{quarter}_{key_hash}"
    
    def semantic_risk_search_batch(self, texts: List[Dict]) -> List[Dict[str, Any]]:
        """Batch semantic search for risk-related climate texts."""
        if not texts:
            return []
        
        # Extract text content
        paragraphs = [text.get('text', '') for text in texts if text.get('text', '').strip()]
        if not paragraphs:
            return []
        
        # Ensure model is loaded
        self.load_model()
        
        # Encode texts with no_grad for memory efficiency
        with torch.no_grad():
            try:
                text_embeddings = self.model.encode(
                    paragraphs,
                    batch_size=min(32, self.batch_size),  # Smaller batch for MPS
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    device=self.device
                )
            except Exception as e:
                logger.warning(f"GPU encoding failed, falling back to CPU: {e}")
                self.device = 'cpu'
                self.model = self.model.to('cpu')
                text_embeddings = self.model.encode(
                    paragraphs,
                    batch_size=16,  # Even smaller batch for stability
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
        
        # Calculate similarities
        similarity_matrix = np.dot(self.query_embeddings, text_embeddings.T)
        max_similarities = np.max(similarity_matrix, axis=0)
        best_query_indices = np.argmax(similarity_matrix, axis=0)
        
        # Filter above threshold
        risk_related_texts = []
        valid_text_idx = 0
        
        for i, text in enumerate(texts):
            if not text.get('text', '').strip():
                continue
            
            similarity = max_similarities[valid_text_idx]
            
            if similarity >= self.threshold:
                risk_related_texts.append({
                    'text_index': i,
                    'text': text,
                    'similarity_score': float(similarity),
                    'best_query': self.risk_queries[best_query_indices[valid_text_idx]],
                    'best_query_index': int(best_query_indices[valid_text_idx])
                })
            
            valid_text_idx += 1
        
        return risk_related_texts
    
    def split_text_into_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting."""
        if not text or not isinstance(text, str):
            return []
        
        # Basic sentence splitting on periods, exclamation marks, question marks
        import re
        sentences = re.split(r'[.!?]+', text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        return sentences
    
    def calculate_climate_risk_exposure(self, transcript_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate climate risk exposure for a transcript."""
        start_time = time.time()
    
    
    
    # ... rest of function
    
        texts = transcript_data.get('texts', [])
        print(f"Got {len(texts)} texts in {time.time() - start_time:.3f}s")

        start_search = time.time()
        
        total_sentences_in_call = transcript_data.get('total_sentences_in_call', 0)
        
        if not texts or total_sentences_in_call == 0:
            return {
                'climate_risk_exposure': 0.0,
                'risk_related_texts_count': 0,
                'risk_sentences_count': 0,
                'total_climate_texts': len(texts),
                'risk_coverage_ratio': 0.0,
                'avg_risk_similarity': 0.0
            }
        
        risk_related_texts = self.semantic_risk_search_batch(texts)
        print(f"Search took {time.time() - start_search:.3f}s")
        # Find risk-related texts
        
        # Count sentences in risk-related texts
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
    
    def process_transcripts(self, transcripts: List[Dict[str, Any]], output_path: Path, batch_size: int = 100) -> None:
        """Process transcripts and save results."""
        results = []
        
        for i, transcript in enumerate(tqdm(transcripts, desc="Processing transcripts")):
            try:
                # Basic transcript info
                row = {
                    'file': transcript.get('file', ''),
                    'company_name': transcript.get('company_name', ''),
                    'ticker': transcript.get('ticker', ''),
                    'quarter': transcript.get('quarter', ''),
                    'year': transcript.get('year', 0),
                    'date': transcript.get('date', ''),
                    'transcript_key': transcript.get('transcript_key', ''),
                    
                    # Original climate metrics
                    'original_climate_sentence_count': transcript.get('climate_sentence_count', 0),
                    'original_climate_sentence_ratio': transcript.get('climate_sentence_ratio', 0.0),
                    'total_sentences_in_call': transcript.get('total_sentences_in_call', 0),
                    'management_sentences': transcript.get('management_sentences', 0),
                    'qa_sentences': transcript.get('qa_sentences', 0)
                }
                
                # Calculate risk exposure
                risk_measures = self.calculate_climate_risk_exposure(transcript)
                row.update(risk_measures)
                
                results.append(row)
                
                # Aggressive memory cleanup every 50 transcripts
                if (i + 1) % 50 == 0:
                    if self.device == 'mps':
                        torch.mps.empty_cache()
                    gc.collect()
                
                # Save batch periodically
                if len(results) >= batch_size:
                    self._save_batch(results, output_path)
                    results = []
                    
                    # Major memory cleanup after batch save
                    if self.device == 'mps':
                        torch.mps.empty_cache()
                    gc.collect()
                    
            except Exception as e:
                logger.error(f"Error processing transcript {transcript.get('transcript_key', 'unknown')}: {e}")
                continue
        
        # Save remaining results
        if results:
            self._save_batch(results, output_path)
        
        logger.info(f"Results saved to: {output_path}")
    
    def _save_batch(self, results: List[Dict], output_path: Path):
        """Save batch to CSV."""
        df = pd.DataFrame(results)
        df = df.sort_values(['ticker', 'year', 'quarter'], na_position='last')
        
        # Create directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save (append if file exists)
        if output_path.exists():
            df.to_csv(output_path, index=False, mode='a', header=False, encoding='utf-8')
        else:
            df.to_csv(output_path, index=False, encoding='utf-8')
    
    def print_summary(self, output_path: Path):
        """Print summary statistics."""
        try:
            df = pd.read_csv(output_path, encoding='utf-8')
            
            logger.info(f"\n{'='*50}")
            logger.info("SEMANTIC CLIMATE RISK ANALYSIS SUMMARY")
            logger.info(f"{'='*50}")
            logger.info(f"Total observations: {len(df):,}")
            logger.info(f"Unique firms: {df['ticker'].nunique():,}")
            logger.info(f"Mean climate risk exposure: {df['climate_risk_exposure'].mean():.4f}")
            logger.info(f"Firms with risk exposure > 0: {(df['climate_risk_exposure'] > 0).mean():.1%}")
            logger.info(f"Mean risk sentences per call: {df['risk_sentences_count'].mean():.1f}")
            logger.info(f"Average similarity score: {df['avg_risk_similarity'].mean():.3f}")
            logger.info(f"{'='*50}")
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Semantic Climate Risk Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process specific file
  python semantic_risk.py --file enhanced_climate_segments_1.json --index SP500
  
  # Process with custom threshold
  python semantic_risk.py --file enhanced_climate_segments_2.json --index STOXX600 --threshold 0.35
  
  # Test mode
  python semantic_risk.py --file enhanced_climate_segments_1.json --index SP500 --limit 50
        """
    )
    
    parser.add_argument(
        '--file',
        type=str,
        required=True,
        help='Enhanced climate snippets filename (e.g., enhanced_climate_segments_1.json)'
    )
    
    parser.add_argument(
        '--index',
        type=str,
        choices=['SP500', 'STOXX600'],
        required=True,
        help='Stock index (SP500 or STOXX600)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.40,
        help='Similarity threshold for risk identification (default: 0.40)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='sentence-transformers/all-mpnet-base-v2',
        help='Sentence transformer model (default: all-mpnet-base-v2)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for saving results (default: 100)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of transcripts for testing'
    )
    
    parser.add_argument(
        '--base-path',
        type=str,
        default='/Users/marleendejonge/Desktop/ECC-data-generation',
        help='Base path to project directory'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Setup paths
    base_path = Path(args.base_path)
    input_path = base_path / "data" / "enhanced_climate_snippets" / args.index / args.file
    
    # Create output filename
    file_stem = Path(args.file).stem
    output_filename = f"cc_risk_{args.index}_{file_stem}.csv"
    output_path = base_path / "outputs" / "variables" / "cc_risk" / output_filename
    
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info(f"Model: {args.model}")
    
    # Check input file exists
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return
    
    # Initialize analyzer
    analyzer = SemanticClimateRiskAnalyzer(
        model_name=args.model,
        threshold=args.threshold
    )
    
    # Load transcripts
    transcripts = analyzer.load_climate_snippets(input_path)
    
    # Apply limit if specified
    if args.limit:
        transcripts = transcripts[:args.limit]
        logger.info(f"Limited to {len(transcripts)} transcripts")
    
    # Process transcripts
    logger.info("Starting processing...")
    analyzer.process_transcripts(transcripts, output_path, batch_size=args.batch_size)
    
    # Print summary
    analyzer.print_summary(output_path)
    
    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()