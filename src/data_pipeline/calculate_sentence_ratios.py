#!/usr/bin/env python3
"""
Modified script to handle climate snippets and structured transcripts across different file numbers.
Uses content-based matching instead of file-number matching.
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import gc
import nltk
from nltk.tokenize import sent_tokenize

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

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
            logging.FileHandler(LOGS_DIR / 'cross_file_matching.log', mode='w')
        ]
    )


def count_sentences(text: str) -> int:
    """Count sentences using NLTK sentence tokenizer."""
    if not text or not text.strip():
        return 0
    
    sentences = sent_tokenize(text.strip())
    # Filter very short sentences (likely artifacts)
    valid_sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    return len(valid_sentences)


def load_all_structured_data(structured_path: Path) -> Dict[str, Dict]:
    """Load all structured transcript data into a single lookup dictionary."""
    logger = logging.getLogger(__name__)
    
    structured_files = sorted(structured_path.glob("structured_calls_*.json"))
    structured_lookup = {}
    total_transcripts = 0
    
    logger.info(f"Loading {len(structured_files)} structured files...")
    
    for structured_file in tqdm(structured_files, desc="Loading structured data"):
        try:
            with open(structured_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for transcript in data:
                filename = transcript.get('file') or transcript.get('filename', '')
                if filename:
                    structured_lookup[filename] = transcript
                    total_transcripts += 1
            
            # Clear memory after processing each file
            del data
            
        except Exception as e:
            logger.error(f"Error loading {structured_file}: {e}")
            continue
    
    logger.info(f"Loaded {total_transcripts} structured transcripts from {len(structured_files)} files")
    return structured_lookup


def process_climate_files_with_lookup(climate_path: Path, structured_lookup: Dict[str, Dict], 
                                    output_path: Path, stock_index: str):
    """Process climate files using the global structured lookup."""
    logger = logging.getLogger(__name__)
    
    climate_files = sorted(climate_path.glob("climate_segments_*.json"))
    
    if not climate_files:
        raise FileNotFoundError(f"No climate segment files found in {climate_path}")
    
    logger.info(f"Processing {len(climate_files)} climate files...")
    
    all_stats = {
        'total_transcripts': 0,
        'matched_transcripts': 0,
        'all_ratios': []
    }
    
    for i, climate_file in enumerate(climate_files, 1):
        print(f"\n📁 Processing climate file {i}/{len(climate_files)}: {climate_file.name}")
        
        try:
            # Load climate data
            with open(climate_file, 'r', encoding='utf-8') as f:
                climate_data = json.load(f)
            
            enhanced_data = []
            file_matches = 0
            
            # Process each climate transcript
            for climate_transcript in tqdm(climate_data, desc=f"Processing {climate_file.name}", leave=False):
                climate_filename = climate_transcript.get('file', '')
                
                if not climate_filename:
                    enhanced_data.append(create_unmatched_transcript(climate_transcript))
                    continue
                
                # Try to find matching structured transcript in global lookup
                if climate_filename in structured_lookup:
                    file_matches += 1
                    all_stats['matched_transcripts'] += 1
                    structured_transcript = structured_lookup[climate_filename]
                    enhanced_transcript = create_matched_transcript(climate_transcript, structured_transcript, climate_filename)
                    enhanced_data.append(enhanced_transcript)
                    
                    # Collect ratio for statistics
                    ratio = enhanced_transcript.get('climate_sentence_ratio')
                    if ratio is not None:
                        all_stats['all_ratios'].append(ratio)
                else:
                    enhanced_data.append(create_unmatched_transcript(climate_transcript))
                
                all_stats['total_transcripts'] += 1
            
            # Save enhanced data for this file
            file_number = re.search(r'(\d+)', climate_file.name)
            file_number = file_number.group(1) if file_number else str(i)
            
            save_enhanced_data(enhanced_data, output_path, stock_index, file_number)
            
            # Report progress
            match_rate = file_matches / len(climate_data) if climate_data else 0
            print(f"   Matches: {file_matches}/{len(climate_data)} ({match_rate:.1%})")
            
            # Memory cleanup
            del enhanced_data
            del climate_data
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error processing {climate_file}: {e}")
            print(f"❌ Error processing {climate_file.name}: {e}")
            continue
    
    return all_stats


def create_matched_transcript(climate_transcript: Dict, structured_transcript: Dict, filename: str) -> Dict:
    """Create enhanced transcript from matched climate and structured data."""
    
    # Count sentences from structured transcript
    mgmt_sentences = 0
    qa_sentences = 0
    
    # Count management sentences
    for segment in structured_transcript.get('speaker_segments_management', []):
        for paragraph in segment.get('paragraphs', []):
            mgmt_sentences += count_sentences(paragraph)
    
    # Count Q&A sentences
    for segment in structured_transcript.get('speaker_segments_qa', []):
        for paragraph in segment.get('paragraphs', []):
            qa_sentences += count_sentences(paragraph)
    
    total_sentences = mgmt_sentences + qa_sentences
    
    # Count climate sentences and enhance text snippets
    climate_sentence_count = 0
    enhanced_texts = []
    
    for text_snippet in climate_transcript.get('texts', []):
        snippet_text = text_snippet.get('text', '')
        snippet_sentences = count_sentences(snippet_text)
        climate_sentence_count += snippet_sentences
        
        enhanced_text = text_snippet.copy()
        enhanced_text['sentence_count'] = snippet_sentences
        enhanced_texts.append(enhanced_text)
    
    # Calculate ratio
    ratio = climate_sentence_count / total_sentences if total_sentences > 0 else 0.0
    
    # Create enhanced transcript
    enhanced = climate_transcript.copy()
    enhanced.update({
        'texts': enhanced_texts,
        'climate_sentence_count': climate_sentence_count,
        'total_sentences_in_call': total_sentences,
        'climate_sentence_ratio': ratio,
        'management_sentences': mgmt_sentences,
        'qa_sentences': qa_sentences,
        'matched_transcript_file': filename
    })
    
    return enhanced


def create_unmatched_transcript(climate_transcript: Dict) -> Dict:
    """Create enhanced transcript for unmatched climate data."""
    
    # Still count climate sentences
    climate_sentence_count = 0
    enhanced_texts = []
    
    for text_snippet in climate_transcript.get('texts', []):
        snippet_text = text_snippet.get('text', '')
        snippet_sentences = count_sentences(snippet_text)
        climate_sentence_count += snippet_sentences
        
        enhanced_text = text_snippet.copy()
        enhanced_text['sentence_count'] = snippet_sentences
        enhanced_texts.append(enhanced_text)
    
    # Create transcript with null ratios
    enhanced = climate_transcript.copy()
    enhanced.update({
        'texts': enhanced_texts,
        'climate_sentence_count': climate_sentence_count,
        'total_sentences_in_call': None,
        'climate_sentence_ratio': None,
        'management_sentences': None,
        'qa_sentences': None,
        'matched_transcript_file': None
    })
    
    return enhanced


def save_enhanced_data(enhanced_data: List[Dict], output_path: Path, stock_index: str, file_number: str):
    """Save enhanced data to file."""
    output_dir = output_path / stock_index
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"enhanced_climate_segments_{file_number}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(enhanced_data, f, indent=2, ensure_ascii=False)


def calculate_final_summary(all_stats: Dict, output_path: Path, stock_index: str):
    """Calculate and save final summary statistics."""
    import numpy as np
    
    summary = {
        'total_transcripts': all_stats['total_transcripts'],
        'matched_transcripts': all_stats['matched_transcripts'],
        'match_rate': all_stats['matched_transcripts'] / all_stats['total_transcripts'] if all_stats['total_transcripts'] > 0 else 0,
        'sentence_ratio_stats': {}
    }
    
    if all_stats['all_ratios']:
        ratios = all_stats['all_ratios']
        summary['sentence_ratio_stats'] = {
            'mean': float(np.mean(ratios)),
            'median': float(np.median(ratios)),
            'std': float(np.std(ratios)),
            'min': float(np.min(ratios)),
            'max': float(np.max(ratios)),
            'p25': float(np.percentile(ratios, 25)),
            'p75': float(np.percentile(ratios, 75))
        }
    
    # Save summary
    output_dir = output_path / stock_index
    summary_file = output_dir / 'cross_file_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Cross-file climate-structured transcript matcher'
    )
    
    parser.add_argument(
        'stock_index',
        nargs='?',
        choices=SUPPORTED_INDICES,
        help='Stock index to process (SP500 or STOXX600)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all available stock indices'
    )
    
    parser.add_argument(
        '--structured-path',
        type=Path,
        default=Path("data/processed/structured_jsons"),
        help='Path to structured transcript files'
    )
    
    parser.add_argument(
        '--climate-path',
        type=Path,
        default=Path("data/climate_paragraphs"),
        help='Path to climate snippet files'
    )
    
    parser.add_argument(
        '--output-path',
        type=Path,
        default=Path("outputs/enhanced_climate_snippets"),
        help='Output path for enhanced files'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.all and not args.stock_index:
        parser.error("Must specify either a stock index or --all")
    
    if args.all and args.stock_index:
        parser.error("Cannot specify both --all and a specific stock index")
    
    # Setup
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    stock_indices = SUPPORTED_INDICES if args.all else [args.stock_index]
    
    print("🔄 Cross-File Climate Matcher")
    print("=" * 50)
    print(f"Stock indices: {', '.join(stock_indices)}")
    print(f"Structured transcripts: {args.structured_path}")
    print(f"Climate snippets: {args.climate_path}")
    print(f"Output: {args.output_path}")
    
    try:
        for stock_index in stock_indices:
            print(f"\n{'='*50}")
            print(f"PROCESSING {stock_index}")
            print(f"{'='*50}")
            
            structured_path = args.structured_path / stock_index
            climate_path = args.climate_path / stock_index
            
            # Validate paths
            if not structured_path.exists():
                print(f"❌ Structured path not found: {structured_path}")
                continue
            if not climate_path.exists():
                print(f"❌ Climate path not found: {climate_path}")
                continue
            
            # Load all structured data into memory
            print("🔧 Loading all structured transcript data...")
            try:
                structured_lookup = load_all_structured_data(structured_path)
                if not structured_lookup:
                    print(f"⚠️ No structured transcripts found for {stock_index}")
                    continue
                print(f"✅ Loaded {len(structured_lookup)} structured transcripts")
            except Exception as e:
                print(f"❌ Error loading structured data: {e}")
                continue
            
            # Process climate files with global lookup
            print("🔍 Processing climate files with cross-file matching...")
            try:
                all_stats = process_climate_files_with_lookup(
                    climate_path, structured_lookup, args.output_path, stock_index
                )
                
                # Calculate final summary
                summary = calculate_final_summary(all_stats, args.output_path, stock_index)
                
                print(f"✅ {stock_index} completed!")
                print(f"📈 Overall match rate: {summary['match_rate']:.1%}")
                print(f"📊 Total transcripts: {summary['total_transcripts']}")
                print(f"🎯 Matched transcripts: {summary['matched_transcripts']}")
                
                if summary['sentence_ratio_stats']:
                    print(f"🎯 Average climate ratio: {summary['sentence_ratio_stats']['mean']:.3%}")
                
                # Clear memory
                del structured_lookup
                gc.collect()
                
            except Exception as e:
                print(f"❌ Error processing {stock_index}: {e}")
                logger.error(f"Error processing {stock_index}: {e}")
                continue
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Main error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 Cross-file matching complete!")
    print(f"📁 Results saved to: {args.output_path}")


if __name__ == "__main__":
    main()