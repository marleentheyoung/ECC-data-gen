#!/usr/bin/env python3
"""
Optimized ClimateBERT classification script for ECC data generation pipeline.

This script processes structured transcript JSONs and identifies climate-related
paragraphs using ClimateBERT with batch processing and parallel execution.

Author: Marleen de Jonge
Date: 2025
"""

import json
import os
import sys
import atexit
import signal
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import multiprocessing as mp

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import psutil

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from src.config import BASE_DIR, SUPPORTED_INDICES


@dataclass
class ProcessingConfig:
    """Configuration for climate classification processing."""
    batch_size: int = 32
    max_length: int = 512
    min_words: int = 10
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    memory_limit_gb: float = 4.0


class OptimizedClimateBERTClassifier:
    """Optimized ClimateBERT classifier with batch processing."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.device = torch.device(config.device)
        
        # Load model and tokenizer
        self._load_model()
        
    def _load_model(self):
        """Load ClimateBERT model and tokenizer."""
        model_name = "climatebert/distilroberta-base-climate-detector"
        
        logging.info(f"Loading ClimateBERT model: {model_name}")
        logging.info(f"Device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        logging.info("✅ Model loaded successfully")
    
    def classify_batch(self, texts: List[str]) -> List[bool]:
        """
        Classify a batch of texts for climate relevance.
        
        Args:
            texts: List of text strings to classify
            
        Returns:
            List of boolean values indicating climate relevance
        """
        if not texts:
            return []
        
        # Tokenize batch
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.config.max_length
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        
        # Convert to boolean list (1 = climate-related)
        return [pred.item() == 1 for pred in predictions]
    
    def process_paragraphs(self, paragraphs_with_metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process paragraphs in batches and return climate-related ones.
        
        Args:
            paragraphs_with_metadata: List of paragraph dictionaries
            
        Returns:
            List of climate-related paragraphs
        """
        if not paragraphs_with_metadata:
            return []
        
        climate_paragraphs = []
        
        # Process in batches
        for i in range(0, len(paragraphs_with_metadata), self.config.batch_size):
            batch = paragraphs_with_metadata[i:i + self.config.batch_size]
            batch_texts = [para['text'] for para in batch]
            
            try:
                climate_predictions = self.classify_batch(batch_texts)
                
                # Keep climate-related paragraphs
                for para, is_climate in zip(batch, climate_predictions):
                    if is_climate:
                        climate_paragraphs.append(para)
                        
            except Exception as e:
                logging.warning(f"Error processing batch: {e}")
                continue
        
        return climate_paragraphs


def extract_paragraphs_from_call(call: Dict[str, Any], min_words: int = 10) -> List[Dict[str, Any]]:
    """
    Extract paragraphs with metadata from a single call.
    
    Args:
        call: Call dictionary from structured JSON
        min_words: Minimum word count for paragraphs
        
    Returns:
        List of paragraph dictionaries with speaker metadata
    """
    paragraphs_with_metadata = []
    
    # Process management segments
    for segment in call.get('speaker_segments_management', []):
        speaker = segment.get('speaker', 'Unknown')
        profession = segment.get('profession', 'Unknown')
        
        for text in segment.get('paragraphs', []):
            if len(text.split()) >= min_words:
                paragraphs_with_metadata.append({
                    'speaker': speaker,
                    'profession': profession,
                    'text': text.strip()
                })
    
    # Process Q&A segments
    for segment in call.get('speaker_segments_qa', []):
        speaker = segment.get('speaker', 'Unknown')
        profession = segment.get('profession', 'Unknown')
        
        for text in segment.get('paragraphs', []):
            if len(text.split()) >= min_words:
                paragraphs_with_metadata.append({
                    'speaker': speaker,
                    'profession': profession,
                    'text': text.strip()
                })
    
    return paragraphs_with_metadata


def process_single_file(input_path: str, output_path: str, config: ProcessingConfig) -> Dict[str, Any]:
    """
    Process a single structured JSON file.
    
    Args:
        input_path: Path to input structured JSON file
        output_path: Path to output climate segments file
        config: Processing configuration
        
    Returns:
        Dictionary with processing statistics
    """
    try:
        # Initialize classifier
        classifier = OptimizedClimateBERTClassifier(config)
        
        # Load input data
        with open(input_path, "r", encoding="utf-8") as f:
            calls = json.load(f)
        
        climate_segments = []
        total_paragraphs = 0
        climate_paragraphs = 0
        
        for call in tqdm(calls, desc=f"Processing {os.path.basename(input_path)}", unit="call", leave=False):
            # Extract paragraphs
            paragraphs_with_metadata = extract_paragraphs_from_call(call, config.min_words)
            total_paragraphs += len(paragraphs_with_metadata)
            
            # Classify paragraphs
            climate_texts = classifier.process_paragraphs(paragraphs_with_metadata)
            climate_paragraphs += len(climate_texts)
            
            # Create output structure
            climate_segment = {
                'file': call.get('file', ''),
                'company_name': call.get('company_name', ''),
                'ticker': call.get('ticker', ''),
                'quarter': call.get('quarter', ''),
                'year': call.get('year', ''),
                'date': call.get('date', ''),
                'texts': climate_texts
            }
            
            climate_segments.append(climate_segment)
        
        # Save results
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(climate_segments, f, indent=2, ensure_ascii=False)
        
        return {
            'input_file': os.path.basename(input_path),
            'output_file': os.path.basename(output_path),
            'total_calls': len(calls),
            'total_paragraphs': total_paragraphs,
            'climate_paragraphs': climate_paragraphs,
            'climate_ratio': climate_paragraphs / total_paragraphs if total_paragraphs > 0 else 0,
            'success': True
        }
        
    except Exception as e:
        return {
            'input_file': os.path.basename(input_path),
            'error': str(e),
            'success': False
        }


def get_file_pairs(stock_index: str) -> List[Tuple[str, str]]:
    """
    Get input/output file pairs for a stock index.
    
    Args:
        stock_index: Stock index (SP500 or STOXX600)
        
    Returns:
        List of (input_path, output_path) tuples
    """
    base_dir = BASE_DIR
    input_base = base_dir / "data" / "processed" / "structured_jsons" / stock_index
    output_base = base_dir / "data" / "climate_paragraphs" / stock_index
    
    # Determine file count based on stock index
    max_files = 19 if stock_index == "SP500" else 15
    
    file_pairs = []
    for i in range(1, max_files + 1):
        input_path = input_base / f"structured_calls_{i}.json"
        output_path = output_base / f"climate_segments_{i}.json"
        
        if input_path.exists():
            file_pairs.append((str(input_path), str(output_path)))
        else:
            logging.warning(f"Input file not found: {input_path}")
    
    return file_pairs


def process_stock_index_parallel(stock_index: str, config: ProcessingConfig) -> List[Dict[str, Any]]:
    """
    Process all files for a stock index using parallel processing.
    
    Args:
        stock_index: Stock index to process
        config: Processing configuration
        
    Returns:
        List of processing results
    """
    file_pairs = get_file_pairs(stock_index)
    
    if not file_pairs:
        logging.error(f"No files found for {stock_index}")
        return []
    
    logging.info(f"Processing {len(file_pairs)} files for {stock_index}")
    
    results = []
    
    # Use ProcessPoolExecutor for parallel processing
    # Note: Each process will load its own model, so be careful with memory
    max_workers = min(config.num_workers, len(file_pairs))
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_file = {
            executor.submit(process_single_file, input_path, output_path, config): (input_path, output_path)
            for input_path, output_path in file_pairs
        }
        
        # Process completed jobs
        with tqdm(total=len(file_pairs), desc=f"Processing {stock_index}", unit="file") as pbar:
            for future in as_completed(future_to_file):
                input_path, output_path = future_to_file[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['success']:
                        pbar.set_postfix({
                            'Climate %': f"{result['climate_ratio']:.1%}",
                            'Paragraphs': result['climate_paragraphs']
                        })
                    else:
                        logging.error(f"Failed to process {input_path}: {result.get('error', 'Unknown error')}")
                    
                except Exception as e:
                    logging.error(f"Error processing {input_path}: {e}")
                    results.append({
                        'input_file': os.path.basename(input_path),
                        'error': str(e),
                        'success': False
                    })
                
                pbar.update(1)
    
    return results


def process_stock_index_sequential(stock_index: str, config: ProcessingConfig) -> List[Dict[str, Any]]:
    """
    Process all files for a stock index sequentially (memory-efficient).
    
    Args:
        stock_index: Stock index to process
        config: Processing configuration
        
    Returns:
        List of processing results
    """
    file_pairs = get_file_pairs(stock_index)
    
    if not file_pairs:
        logging.error(f"No files found for {stock_index}")
        return []
    
    logging.info(f"Processing {len(file_pairs)} files for {stock_index} (sequential)")
    
    # Initialize classifier once
    classifier = OptimizedClimateBERTClassifier(config)
    results = []
    
    for input_path, output_path in tqdm(file_pairs, desc=f"Processing {stock_index}", unit="file"):
        try:
            # Load input data
            with open(input_path, "r", encoding="utf-8") as f:
                calls = json.load(f)
            
            climate_segments = []
            total_paragraphs = 0
            climate_paragraphs = 0
            
            for call in calls:
                # Extract paragraphs
                paragraphs_with_metadata = extract_paragraphs_from_call(call, config.min_words)
                total_paragraphs += len(paragraphs_with_metadata)
                
                # Classify paragraphs
                climate_texts = classifier.process_paragraphs(paragraphs_with_metadata)
                climate_paragraphs += len(climate_texts)
                
                # Create output structure
                climate_segment = {
                    'file': call.get('file', ''),
                    'company_name': call.get('company_name', ''),
                    'ticker': call.get('ticker', ''),
                    'quarter': call.get('quarter', ''),
                    'year': call.get('year', ''),
                    'date': call.get('date', ''),
                    'texts': climate_texts
                }
                
                climate_segments.append(climate_segment)
            
            # Save results
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(climate_segments, f, indent=2, ensure_ascii=False)
            
            result = {
                'input_file': os.path.basename(input_path),
                'output_file': os.path.basename(output_path),
                'total_calls': len(calls),
                'total_paragraphs': total_paragraphs,
                'climate_paragraphs': climate_paragraphs,
                'climate_ratio': climate_paragraphs / total_paragraphs if total_paragraphs > 0 else 0,
                'success': True
            }
            results.append(result)
            
        except Exception as e:
            logging.error(f"Error processing {input_path}: {e}")
            results.append({
                'input_file': os.path.basename(input_path),
                'error': str(e),
                'success': False
            })
    
    return results


def save_processing_summary(results: List[Dict[str, Any]], stock_index: str, output_dir: Path):
    """Save processing summary to JSON file."""
    summary = {
        'stock_index': stock_index,
        'total_files': len(results),
        'successful_files': sum(1 for r in results if r['success']),
        'failed_files': sum(1 for r in results if not r['success']),
        'total_calls': sum(r.get('total_calls', 0) for r in results if r['success']),
        'total_paragraphs': sum(r.get('total_paragraphs', 0) for r in results if r['success']),
        'climate_paragraphs': sum(r.get('climate_paragraphs', 0) for r in results if r['success']),
        'files': results
    }
    
    # Calculate overall climate ratio
    if summary['total_paragraphs'] > 0:
        summary['overall_climate_ratio'] = summary['climate_paragraphs'] / summary['total_paragraphs']
    else:
        summary['overall_climate_ratio'] = 0
    
    summary_path = output_dir / f"{stock_index.lower()}_processing_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Processing summary saved: {summary_path}")


def main():
    """Main function to run climate classification."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract climate-related paragraphs using ClimateBERT"
    )
    parser.add_argument(
        "stock_index",
        nargs='?',
        choices=SUPPORTED_INDICES + ['all'],
        help="Stock index to process (SP500, STOXX600, or all)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for classification (default: 32)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Use sequential processing instead of parallel"
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=10,
        help="Minimum words per paragraph (default: 10)"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if not args.stock_index:
        parser.error("Must specify a stock index or 'all'")
    
    # Setup logging
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Check available memory
    available_memory = psutil.virtual_memory().available / (1024**3)
    logging.info(f"Available memory: {available_memory:.1f}GB")
    
    # Adjust configuration based on memory
    if available_memory < 8 and not args.sequential:
        logging.warning("Low memory detected - using sequential processing")
        args.sequential = True
    
    # Create processing configuration
    config = ProcessingConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        min_words=args.min_words,
        device=args.device
    )
    
    # Log configuration
    logging.info(f"Configuration: {config}")
    
    # Determine stock indices to process
    if args.stock_index == "all":
        stock_indices = SUPPORTED_INDICES
    else:
        stock_indices = [args.stock_index]
    
    # Process each stock index
    all_results = {}
    
    for stock_index in stock_indices:
        print(f"\n{'='*60}")
        print(f"PROCESSING {stock_index}")
        print(f"{'='*60}")
        
        try:
            if args.sequential:
                results = process_stock_index_sequential(stock_index, config)
            else:
                results = process_stock_index_parallel(stock_index, config)
            
            all_results[stock_index] = results
            
            # Save summary
            output_dir = BASE_DIR / "data" / "climate_paragraphs" / stock_index
            save_processing_summary(results, stock_index, output_dir)
            
            # Print summary
            successful = sum(1 for r in results if r['success'])
            total_climate = sum(r.get('climate_paragraphs', 0) for r in results if r['success'])
            total_paragraphs = sum(r.get('total_paragraphs', 0) for r in results if r['success'])
            
            print(f"✅ {stock_index} completed:")
            print(f"  Files processed: {successful}/{len(results)}")
            print(f"  Climate paragraphs: {total_climate:,}")
            print(f"  Total paragraphs: {total_paragraphs:,}")
            if total_paragraphs > 0:
                print(f"  Climate ratio: {total_climate/total_paragraphs:.1%}")
            
        except Exception as e:
            logging.error(f"Error processing {stock_index}: {e}")
            continue
    
    print(f"\n🎉 Climate classification completed!")


if __name__ == "__main__":
    main()