#!/usr/bin/env python3
"""
Script to calculate sentence ratios between climate snippets and full earnings calls.

This script loads climate snippets and full transcript data, calculates the ratio
of climate-related sentences to total sentences in each earnings call, and saves
enhanced climate snippet files with sentence ratio information.

Usage:
    # Calculate ratios for both SP500 and STOXX600
    python scripts/2.5_calculate_sentence_ratios.py --all
    
    # Calculate for SP500 only
    python scripts/2.5_calculate_sentence_ratios.py SP500
    
    # Use custom paths
    python scripts/2.5_calculate_sentence_ratios.py --all --raw-transcripts-path /custom/path --climate-snippets-path /custom/path

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
from collections import defaultdict
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
            logging.FileHandler(LOGS_DIR / 'sentence_ratios.log', mode='a')
        ]
    )


def count_sentences(text: str) -> int:
    """
    Count sentences in text using robust heuristics.
    
    Args:
        text: Input text
        
    Returns:
        Number of sentences
    """
    if not text or not text.strip():
        return 0
    
    # Clean text first
    text = text.strip()
    
    # Split on sentence endings, but be careful with abbreviations and numbers
    # This regex looks for sentence endings followed by whitespace and capital letter or end of string
    sentences = re.split(r'[.!?]+(?=\s+[A-Z]|\s*$)', text)
    
    # Filter empty strings and very short "sentences" (likely artifacts)
    valid_sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    return len(valid_sentences)


def load_full_transcripts(raw_transcripts_path: Path, stock_index: str) -> Dict[str, Dict[str, any]]:
    """
    Load full transcript data from raw JSON files.
    
    Args:
        raw_transcripts_path: Path to raw transcripts folder
        stock_index: Stock index (SP500 or STOXX600)
        
    Returns:
        Dictionary mapping filenames to transcript data with sentence counts
    """
    logger = logging.getLogger(__name__)
    
    index_path = raw_transcripts_path / stock_index
    if not index_path.exists():
        raise FileNotFoundError(f"Raw transcripts path not found: {index_path}")
    
    # Find all transcript part files
    part_files = list(index_path.glob("transcripts_data_part*.json"))
    if not part_files:
        raise FileNotFoundError(f"No transcript part files found in: {index_path}")
    
    logger.info(f"Found {len(part_files)} transcript part files for {stock_index}")
    
    all_transcripts = {}
    
    for part_file in tqdm(part_files, desc=f"Loading {stock_index} transcripts"):
        try:
            with open(part_file, 'r', encoding='utf-8') as f:
                part_data = json.load(f)
            
            # Process each transcript in this part
            for filename, transcript_data in part_data.items():
                # Extract text sections
                mgmt_text = transcript_data.get('Management Discussion', '')
                qa_text = transcript_data.get('Q&A Section', '')
                
                # Count sentences in each section
                mgmt_sentences = count_sentences(mgmt_text)
                qa_sentences = count_sentences(qa_text)
                total_sentences = mgmt_sentences + qa_sentences
                
                # Store transcript info
                all_transcripts[filename] = {
                    'file': transcript_data.get('File', filename),
                    'management_text': mgmt_text,
                    'qa_text': qa_text,
                    'management_sentences': mgmt_sentences,
                    'qa_sentences': qa_sentences,
                    'total_sentences': total_sentences,
                    'original_data': transcript_data  # Keep original for reference
                }
            
            # Clear from memory
            del part_data
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error loading {part_file}: {e}")
            continue
    
    logger.info(f"✅ Loaded {len(all_transcripts)} full transcripts for {stock_index}")
    return all_transcripts


def load_climate_snippets(climate_snippets_path: Path, stock_index: str) -> List[Dict]:
    """
    Load climate snippet data.
    
    Args:
        climate_snippets_path: Path to climate snippets folder
        stock_index: Stock index (SP500 or STOXX600)
        
    Returns:
        List of climate snippet files data
    """
    logger = logging.getLogger(__name__)
    
    index_path = climate_snippets_path / stock_index
    if not index_path.exists():
        raise FileNotFoundError(f"Climate snippets path not found: {index_path}")
    
    # Find all climate segment files
    snippet_files = list(index_path.glob("climate_segments_*.json"))
    if not snippet_files:
        raise FileNotFoundError(f"No climate segment files found in: {index_path}")
    
    logger.info(f"Found {len(snippet_files)} climate snippet files for {stock_index}")
    
    all_snippet_data = []
    
    for snippet_file in tqdm(snippet_files, desc=f"Loading {stock_index} climate snippets"):
        try:
            with open(snippet_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            all_snippet_data.append({
                'file': snippet_file,
                'data': data
            })
            
        except Exception as e:
            logger.error(f"Error loading {snippet_file}: {e}")
            continue
    
    logger.info(f"✅ Loaded {len(all_snippet_data)} climate snippet files for {stock_index}")
    return all_snippet_data


def match_filenames(climate_filename: str, transcript_filenames: List[str]) -> Optional[str]:
    """
    Match climate snippet filename to transcript filename.
    
    Args:
        climate_filename: Filename from climate snippets
        transcript_filenames: List of available transcript filenames
        
    Returns:
        Matching transcript filename or None
    """
    # Direct match first
    if climate_filename in transcript_filenames:
        return climate_filename
    
    # Try with "CORRECTED TRANSCRIPT " prefix
    corrected_filename = f"CORRECTED TRANSCRIPT {climate_filename}"
    if corrected_filename in transcript_filenames:
        return corrected_filename
    
    # Try removing "CORRECTED TRANSCRIPT " prefix
    if climate_filename.startswith("CORRECTED TRANSCRIPT "):
        clean_filename = climate_filename.replace("CORRECTED TRANSCRIPT ", "")
        if clean_filename in transcript_filenames:
            return clean_filename
    
    # Fuzzy matching - look for key components (company, quarter, year)
    for transcript_filename in transcript_filenames:
        # Extract key identifiers and see if they match
        if (extract_key_identifiers(climate_filename) == 
            extract_key_identifiers(transcript_filename)):
            return transcript_filename
    
    return None


def extract_key_identifiers(filename: str) -> Tuple[str, str, str]:
    """
    Extract key identifiers (company ticker, quarter, year) from filename.
    
    Args:
        filename: Filename to parse
        
    Returns:
        Tuple of (ticker, quarter, year)
    """
    # Remove common prefixes and suffixes
    clean_name = filename.replace("CORRECTED TRANSCRIPT ", "").replace(".pdf", "")
    
    # Try to extract ticker, quarter, year using regex
    import re
    
    # Look for patterns like "Q1 2010" or "Q1 2010"
    quarter_year_match = re.search(r'Q([1-4])\s+(\d{4})', clean_name)
    quarter = quarter_year_match.group(1) if quarter_year_match else ""
    year = quarter_year_match.group(2) if quarter_year_match else ""
    
    # Look for ticker (usually 2-5 capital letters)
    ticker_match = re.search(r'\b([A-Z]{2,5})\b', clean_name)
    ticker = ticker_match.group(1) if ticker_match else ""
    
    return (ticker, f"Q{quarter}", year)

def get_structured_transcript_data(filename: str, stock_index: str) -> Optional[Dict]:
    """
    Load structured transcript data to get paragraph counts from speaker segments.
    
    Args:
        filename: Filename to match
        stock_index: Stock index (SP500 or STOXX600)
        
    Returns:
        Structured transcript data or None if not found
    """
    # Path to structured JSONs - adjust this path as needed
    structured_path = Path("/Users/marleendejonge/Desktop/ECC-data-generation/data/processed/structured_jsons") / stock_index
    
    if not structured_path.exists():
        return None
    
    # Find all structured JSON files
    structured_files = list(structured_path.glob("structured_calls_*.json"))
    
    for structured_file in structured_files:
        try:
            with open(structured_file, 'r', encoding='utf-8') as f:
                structured_data = json.load(f)
            
            # Look for matching transcript
            for transcript in structured_data:
                transcript_filename = transcript.get('filename') or transcript.get('file', '')
                
                # Try various matching approaches
                if (transcript_filename == filename or 
                    transcript_filename == filename.replace("CORRECTED TRANSCRIPT ", "") or
                    f"CORRECTED TRANSCRIPT {transcript_filename}" == filename):
                    return transcript
                    
        except Exception as e:
            continue
    
    return None

def calculate_sentence_ratios(climate_snippet_data: List[Dict], 
                            full_transcripts: Dict[str, Dict],
                            stock_index: str) -> List[Dict]:
    """
    Calculate sentence ratios for climate snippets.
    
    Args:
        climate_snippet_data: List of climate snippet file data
        full_transcripts: Dictionary of full transcript data
        
    Returns:
        List of enhanced climate snippet data with sentence ratios
    """
    logger = logging.getLogger(__name__)
    
    enhanced_snippet_data = []
    transcript_filenames = list(full_transcripts.keys())
    
    total_matches = 0
    total_transcripts = 0
    
    for snippet_file_data in climate_snippet_data:
        snippet_file = snippet_file_data['file']
        snippet_data = snippet_file_data['data']
        
        enhanced_data = []
        
        for transcript in tqdm(snippet_data, desc=f"Processing {snippet_file.name}", leave=False):
            total_transcripts += 1
            
            # Get the filename from the transcript
            climate_filename = transcript.get('file', '')
            if not climate_filename:
                logger.warning(f"No filename found in climate transcript")
                continue
            
            # Try to match with full transcript
            matched_filename = match_filenames(climate_filename, transcript_filenames)
            
            if matched_filename and matched_filename in full_transcripts:
                total_matches += 1
                full_transcript = full_transcripts[matched_filename]
                
                # Count sentences in climate snippets
                climate_sentence_count = 0
                enhanced_texts = []

                for text_snippet in transcript.get('texts', []):
                    snippet_text = text_snippet.get('text', '')
                    snippet_sentences = count_sentences(snippet_text)
                    climate_sentence_count += snippet_sentences
                    
                    # Add sentence count to snippet
                    enhanced_text_snippet = text_snippet.copy()
                    enhanced_text_snippet['sentence_count'] = snippet_sentences
                    enhanced_texts.append(enhanced_text_snippet)

                # Calculate paragraph counts from speaker segments (from the structured transcript data)
                # We need to load the structured transcript to get the correct paragraph counts
                mgmt_paragraph_count = 0
                qa_paragraph_count = 0

                # Try to get structured transcript data for paragraph counting
                # This would require loading the structured JSON files - we'll add a helper function
                structured_transcript = get_structured_transcript_data(climate_filename, stock_index)
                if structured_transcript:
                    # Count paragraphs from speaker segments
                    for segment in structured_transcript.get('speaker_segments_management', []):
                        mgmt_paragraph_count += len(segment.get('paragraphs', []))
                    
                    for segment in structured_transcript.get('speaker_segments_qa', []):
                        qa_paragraph_count += len(segment.get('paragraphs', []))

                total_paragraph_count = mgmt_paragraph_count + qa_paragraph_count

                # Calculate ratio
                total_sentences = full_transcript['total_sentences']
                sentence_ratio = climate_sentence_count / total_sentences if total_sentences > 0 else 0.0

                # Create enhanced transcript data
                enhanced_transcript = transcript.copy()
                enhanced_transcript.update({
                    'texts': enhanced_texts,
                    'climate_sentence_count': climate_sentence_count,
                    'total_sentences_in_call': total_sentences,
                    'climate_sentence_ratio': sentence_ratio,
                    'management_sentences': full_transcript['management_sentences'],
                    'qa_sentences': full_transcript['qa_sentences'],
                    'management_paragraph_count': mgmt_paragraph_count,
                    'qa_paragraph_count': qa_paragraph_count,
                    'total_paragraph_count': total_paragraph_count,
                    'matched_transcript_file': matched_filename
                })
                enhanced_data.append(enhanced_transcript)
                
            else:
                # No match found - add with zero ratios
                logger.warning(f"No matching full transcript found for: {climate_filename}")
                
                # Still count sentences in climate snippets
                climate_sentence_count = 0
                enhanced_texts = []
                
                for text_snippet in transcript.get('texts', []):
                    snippet_text = text_snippet.get('text', '')
                    snippet_sentences = count_sentences(snippet_text)
                    climate_sentence_count += snippet_sentences
                    
                    enhanced_text_snippet = text_snippet.copy()
                    enhanced_text_snippet['sentence_count'] = snippet_sentences
                    enhanced_texts.append(enhanced_text_snippet)
                
                # Create transcript with unknown total
                enhanced_transcript = transcript.copy()
                enhanced_transcript.update({
                    'texts': enhanced_texts,
                    'climate_sentence_count': climate_sentence_count,
                    'total_sentences_in_call': None,
                    'climate_sentence_ratio': None,
                    'management_sentences': None,
                    'qa_sentences': None,
                    'matched_transcript_file': None
                })
                
                enhanced_data.append(enhanced_transcript)
        
        enhanced_snippet_data.append({
            'file': snippet_file,
            'data': enhanced_data
        })
    
    match_rate = total_matches / total_transcripts if total_transcripts > 0 else 0
    logger.info(f"📊 Match statistics: {total_matches}/{total_transcripts} ({match_rate:.1%}) transcripts matched")
    
    return enhanced_snippet_data


def save_enhanced_climate_snippets(enhanced_data: List[Dict], 
                                 output_path: Path, 
                                 stock_index: str) -> None:
    """
    Save enhanced climate snippet data with sentence ratios.
    
    Args:
        enhanced_data: List of enhanced snippet file data
        output_path: Output path for enhanced files
        stock_index: Stock index name
    """
    logger = logging.getLogger(__name__)
    
    output_dir = output_path / stock_index
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for file_data in enhanced_data:
        original_file = file_data['file']
        data = file_data['data']
        
        # Create output filename
        output_filename = f"enhanced_{original_file.name}"
        output_file = output_dir / output_filename
        
        # Save enhanced data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved enhanced file: {output_file}")
    
    # Create summary statistics
    summary_stats = calculate_summary_statistics(enhanced_data)
    summary_file = output_dir / 'sentence_ratio_summary.json'
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_stats, f, indent=2, default=str)
    
    logger.info(f"📊 Summary statistics saved: {summary_file}")


def calculate_summary_statistics(enhanced_data: List[Dict]) -> Dict:
    """Calculate summary statistics for sentence ratios."""
    
    all_ratios = []
    total_climate_sentences = 0
    total_call_sentences = 0
    transcripts_with_ratios = 0
    transcripts_total = 0
    
    for file_data in enhanced_data:
        for transcript in file_data['data']:
            transcripts_total += 1
            
            climate_sentences = transcript.get('climate_sentence_count', 0)
            total_sentences = transcript.get('total_sentences_in_call')
            ratio = transcript.get('climate_sentence_ratio')
            
            total_climate_sentences += climate_sentences
            
            if total_sentences is not None and ratio is not None:
                total_call_sentences += total_sentences
                all_ratios.append(ratio)
                transcripts_with_ratios += 1
    
    import numpy as np
    
    summary = {
        'total_transcripts': transcripts_total,
        'transcripts_with_sentence_ratios': transcripts_with_ratios,
        'match_rate': transcripts_with_ratios / transcripts_total if transcripts_total > 0 else 0,
        'total_climate_sentences': total_climate_sentences,
        'total_call_sentences': total_call_sentences,
        'sentence_ratio_statistics': {
            'mean': float(np.mean(all_ratios)) if all_ratios else 0,
            'median': float(np.median(all_ratios)) if all_ratios else 0,
            'std': float(np.std(all_ratios)) if all_ratios else 0,
            'min': float(np.min(all_ratios)) if all_ratios else 0,
            'max': float(np.max(all_ratios)) if all_ratios else 0,
            'p25': float(np.percentile(all_ratios, 25)) if all_ratios else 0,
            'p75': float(np.percentile(all_ratios, 75)) if all_ratios else 0,
            'p90': float(np.percentile(all_ratios, 90)) if all_ratios else 0,
            'p99': float(np.percentile(all_ratios, 99)) if all_ratios else 0
        }
    }
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Calculate sentence ratios between climate snippets and full earnings calls',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Calculate ratios for both indices
    python scripts/3.5_calculate_sentence_ratios.py --all
    
    # Calculate for SP500 only
    python scripts/3.5_calculate_sentence_ratios.py SP500
    
    # Use custom paths
    python scripts/3.5_calculate_sentence_ratios.py STOXX600 --raw-transcripts-path /custom/raw --climate-snippets-path /custom/climate
        """
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
        '--raw-transcripts-path',
        type=Path,
        default=Path("/Users/marleendejonge/Desktop/ECC-data-generation/data/processed/raw_jsons"),
        help='Path to raw transcript JSON files'
    )
    
    parser.add_argument(
        '--climate-snippets-path',
        type=Path,
        default=Path("/Users/marleendejonge/Desktop/ECC-data-generation/data/climate_paragraphs"),
        help='Path to climate snippet files'
    )
    
    parser.add_argument(
        '--output-path',
        type=Path,
        default=Path("outputs/enhanced_climate_snippets"),
        help='Output path for enhanced climate snippets'
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
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Determine stock indices to process
    stock_indices = SUPPORTED_INDICES if args.all else [args.stock_index]
    
    print("📊 Climate Sentence Ratio Calculation")
    print("=" * 50)
    print(f"Stock indices: {', '.join(stock_indices)}")
    print(f"Raw transcripts path: {args.raw_transcripts_path}")
    print(f"Climate snippets path: {args.climate_snippets_path}")
    print(f"Output path: {args.output_path}")
    
    try:
        for stock_index in stock_indices:
            print(f"\n{'='*60}")
            print(f"PROCESSING {stock_index}")
            print(f"{'='*60}")
            
            # Load full transcripts
            print(f"📥 Loading full transcripts for {stock_index}...")
            try:
                full_transcripts = load_full_transcripts(args.raw_transcripts_path, stock_index)
                print(f"✅ Loaded {len(full_transcripts)} full transcripts")
            except Exception as e:
                print(f"❌ Error loading full transcripts for {stock_index}: {e}")
                continue
            
            # Load climate snippets
            print(f"🌍 Loading climate snippets for {stock_index}...")
            try:
                climate_snippets = load_climate_snippets(args.climate_snippets_path, stock_index)
                total_climate_transcripts = sum(len(data['data']) for data in climate_snippets)
                print(f"✅ Loaded {total_climate_transcripts} climate transcripts from {len(climate_snippets)} files")
            except Exception as e:
                print(f"❌ Error loading climate snippets for {stock_index}: {e}")
                continue
            
            # Calculate sentence ratios
            print(f"🔢 Calculating sentence ratios...")
            enhanced_data = calculate_sentence_ratios(climate_snippets, full_transcripts, stock_index)
            
            # Save enhanced data
            print(f"💾 Saving enhanced climate snippets...")
            save_enhanced_climate_snippets(enhanced_data, args.output_path, stock_index)
            
            print(f"✅ {stock_index} completed successfully!")
            
            # Clean up memory
            del full_transcripts, climate_snippets, enhanced_data
            gc.collect()
    
    except KeyboardInterrupt:
        print("\n🛑 Process interrupted by user")
        return
    except Exception as e:
        logger.error(f"❌ Error during processing: {e}")
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return
    
    print(f"\n✅ Sentence ratio calculation completed!")
    print(f"📁 Enhanced climate snippets saved to: {args.output_path}")
    print(f"\nOutput structure:")
    print(f"  outputs/enhanced_climate_snippets/")
    print(f"  ├── SP500/")
    print(f"  │   ├── enhanced_climate_segments_1.json")
    print(f"  │   ├── enhanced_climate_segments_2.json")
    print(f"  │   └── sentence_ratio_summary.json")
    print(f"  └── STOXX600/")
    print(f"      ├── enhanced_climate_segments_1.json")
    print(f"      └── sentence_ratio_summary.json")


if __name__ == "__main__":
    main()