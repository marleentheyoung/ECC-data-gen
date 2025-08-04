#!/usr/bin/env python3
"""
Clean script to calculate sentence ratios between climate snippets and structured earnings calls.

This script matches climate snippet files with structured transcript files and calculates
what percentage of each earnings call was devoted to climate topics.

Author: Marleen de Jonge
Date: 2025
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import gc

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
            logging.FileHandler(LOGS_DIR / 'sentence_ratios.log', mode='w')  # Fresh log file
        ]
    )


def count_sentences(text: str) -> int:
    """Count sentences in text using robust heuristics."""
    if not text or not text.strip():
        return 0
    
    text = text.strip()
    # Split on sentence endings followed by whitespace and capital letter
    sentences = re.split(r'[.!?]+(?=\s+[A-Z]|\s*$)', text)
    # Filter very short sentences (likely artifacts)
    valid_sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    return len(valid_sentences)


def find_file_pairs(structured_path: Path, climate_path: Path, stock_index: str) -> List[Tuple[Path, Path]]:
    """
    Find matching file pairs between climate and structured transcript files.
    
    Returns:
        List of (climate_file, structured_file) tuples
    """
    logger = logging.getLogger(__name__)
    
    # Get all files
    climate_files = sorted(climate_path.glob("climate_segments_*.json"))
    structured_files = sorted(structured_path.glob("structured_calls_*.json"))
    
    logger.info(f"Found {len(climate_files)} climate files")
    logger.info(f"Found {len(structured_files)} structured files")
    
    if not climate_files:
        raise FileNotFoundError(f"No climate segment files found in {climate_path}")
    if not structured_files:
        raise FileNotFoundError(f"No structured call files found in {structured_path}")
    
    # Try to pair files by number
    pairs = []
    
    for climate_file in climate_files:
        # Extract number from climate_segments_N.json
        climate_match = re.search(r'climate_segments_(\d+)\.json', climate_file.name)
        if not climate_match:
            logger.warning(f"Could not extract number from: {climate_file.name}")
            continue
        
        climate_num = int(climate_match.group(1))
        
        # Look for matching structured file
        matched = False
        for structured_file in structured_files:
            structured_match = re.search(r'structured_calls_(\d+)\.json', structured_file.name)
            if structured_match:
                structured_num = int(structured_match.group(1))
                if structured_num == climate_num:
                    pairs.append((climate_file, structured_file))
                    logger.info(f"✅ Exact match: {climate_file.name} ↔ {structured_file.name}")
                    matched = True
                    break
        
        if not matched:
            logger.warning(f"❌ No exact match for: {climate_file.name}")
    
    # If we have unmatched files, try sequential pairing
    if len(pairs) < min(len(climate_files), len(structured_files)):
        logger.info("Attempting sequential pairing for unmatched files...")
        
        # Get unmatched files
        paired_climate = {pair[0] for pair in pairs}
        paired_structured = {pair[1] for pair in pairs}
        
        unmatched_climate = [f for f in climate_files if f not in paired_climate]
        unmatched_structured = [f for f in structured_files if f not in paired_structured]
        
        # Pair sequentially
        for i in range(min(len(unmatched_climate), len(unmatched_structured))):
            pairs.append((unmatched_climate[i], unmatched_structured[i]))
            logger.info(f"📋 Sequential pair: {unmatched_climate[i].name} ↔ {unmatched_structured[i].name}")
    
    logger.info(f"📊 Total pairs created: {len(pairs)}")
    return pairs


def process_file_pair(climate_file: Path, structured_file: Path) -> List[Dict]:
    """
    Process one climate/structured file pair.
    
    Returns:
        List of enhanced climate transcript data
    """
    logger = logging.getLogger(__name__)
    
    # Load files
    with open(climate_file, 'r', encoding='utf-8') as f:
        climate_data = json.load(f)
    
    with open(structured_file, 'r', encoding='utf-8') as f:
        structured_data = json.load(f)
    
    # Create lookup dictionary for structured data
    structured_lookup = {}
    for transcript in structured_data:
        filename = transcript.get('file') or transcript.get('filename', '')
        if filename:
            structured_lookup[filename] = transcript
    
    logger.info(f"Processing: {len(climate_data)} climate transcripts vs {len(structured_lookup)} structured transcripts")
    
    enhanced_data = []
    matches = 0
    
    for climate_transcript in tqdm(climate_data, desc=f"Processing {climate_file.name}", leave=False):
        climate_filename = climate_transcript.get('file', '')
        
        if not climate_filename:
            logger.warning("Climate transcript missing filename field")
            enhanced_data.append(create_unmatched_transcript(climate_transcript))
            continue
        
        # Try to find matching structured transcript
        if climate_filename in structured_lookup:
            matches += 1
            structured_transcript = structured_lookup[climate_filename]
            enhanced_transcript = create_matched_transcript(climate_transcript, structured_transcript, climate_filename)
            enhanced_data.append(enhanced_transcript)
        else:
            logger.debug(f"No match for: {climate_filename}")
            enhanced_data.append(create_unmatched_transcript(climate_transcript))
    
    match_rate = matches / len(climate_data) if climate_data else 0
    logger.info(f"📈 Match rate: {matches}/{len(climate_data)} ({match_rate:.1%})")
    
    return enhanced_data


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
    logger = logging.getLogger(__name__)
    
    output_dir = output_path / stock_index
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"enhanced_climate_segments_{file_number}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(enhanced_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 Saved: {output_file}")
    print(f"💾 Saved: {output_file.name}")


def calculate_summary_stats(output_path: Path, stock_index: str) -> Optional[Dict]:
    """Calculate summary statistics from all enhanced files."""
    logger = logging.getLogger(__name__)
    
    output_dir = output_path / stock_index
    enhanced_files = list(output_dir.glob("enhanced_climate_segments_*.json"))
    
    if not enhanced_files:
        logger.warning("No enhanced files found for summary")
        return None
    
    all_ratios = []
    total_climate_sentences = 0
    total_call_sentences = 0
    matched_transcripts = 0
    total_transcripts = 0
    
    for enhanced_file in enhanced_files:
        with open(enhanced_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for transcript in data:
            total_transcripts += 1
            
            climate_sentences = transcript.get('climate_sentence_count', 0)
            total_sentences = transcript.get('total_sentences_in_call')
            ratio = transcript.get('climate_sentence_ratio')
            
            total_climate_sentences += climate_sentences
            
            if total_sentences is not None and ratio is not None:
                total_call_sentences += total_sentences
                all_ratios.append(ratio)
                matched_transcripts += 1
    
    # Calculate statistics
    import numpy as np
    
    summary = {
        'total_transcripts': total_transcripts,
        'matched_transcripts': matched_transcripts,
        'match_rate': matched_transcripts / total_transcripts if total_transcripts > 0 else 0,
        'total_climate_sentences': total_climate_sentences,
        'total_call_sentences': total_call_sentences,
        'sentence_ratio_stats': {
            'mean': float(np.mean(all_ratios)) if all_ratios else 0,
            'median': float(np.median(all_ratios)) if all_ratios else 0,
            'std': float(np.std(all_ratios)) if all_ratios else 0,
            'min': float(np.min(all_ratios)) if all_ratios else 0,
            'max': float(np.max(all_ratios)) if all_ratios else 0,
            'p25': float(np.percentile(all_ratios, 25)) if all_ratios else 0,
            'p75': float(np.percentile(all_ratios, 75)) if all_ratios else 0
        }
    }
    
    # Save summary
    summary_file = output_dir / 'sentence_ratio_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"📊 Summary saved: {summary_file}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Calculate sentence ratios between climate snippets and structured earnings calls'
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
    
    print("🧮 Climate Sentence Ratio Calculator")
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
            
            # Find file pairs
            print("🔍 Finding file pairs...")
            try:
                pairs = find_file_pairs(structured_path, climate_path, stock_index)
                if not pairs:
                    print(f"⚠️ No file pairs found for {stock_index}")
                    continue
                print(f"✅ Found {len(pairs)} file pairs")
            except Exception as e:
                print(f"❌ Error finding pairs: {e}")
                continue
            
            # Process each pair
            for i, (climate_file, structured_file) in enumerate(pairs, 1):
                print(f"\n📁 Processing pair {i}/{len(pairs)}")
                print(f"   Climate: {climate_file.name}")
                print(f"   Structured: {structured_file.name}")
                
                try:
                    enhanced_data = process_file_pair(climate_file, structured_file)
                    
                    # Extract file number for output naming
                    climate_match = re.search(r'(\d+)', climate_file.name)
                    file_number = climate_match.group(1) if climate_match else str(i)
                    
                    save_enhanced_data(enhanced_data, args.output_path, stock_index, file_number)
                    
                    # Memory cleanup
                    del enhanced_data
                    gc.collect()
                    
                    print(f"✅ Completed pair {i}/{len(pairs)}")
                    
                except Exception as e:
                    print(f"❌ Error processing pair {i}: {e}")
                    logger.error(f"Error processing pair {i}: {e}")
                    continue
            
            # Calculate summary
            print("\n📊 Calculating summary statistics...")
            summary = calculate_summary_stats(args.output_path, stock_index)
            
            if summary:
                print(f"✅ {stock_index} completed!")
                print(f"📈 Match rate: {summary['match_rate']:.1%}")
                print(f"📊 Total transcripts: {summary['total_transcripts']}")
                print(f"🎯 Average climate ratio: {summary['sentence_ratio_stats']['mean']:.3%}")
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Main error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 Processing complete!")
    print(f"📁 Results saved to: {args.output_path}")


if __name__ == "__main__":
    main()