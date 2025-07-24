"""
Paragraph extraction utilities for creating embeddings from structured transcripts.

This module extracts individual paragraphs from structured transcripts with metadata
for semantic search indexing.

Author: Marleen de Jonge
Date: 2025
"""

import json
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Iterator, Tuple
from tqdm import tqdm

logger = logging.getLogger(__name__)

import re

def clean_paragraph_text(text: str) -> str:
    """Clean paragraph text by removing metadata and formatting artifacts."""
    if not text:
        return text
    
    # Remove the entire header block (corrected transcript + company info + callstreet)
    text = re.sub(r'corrected transcript.*?(?=<[QA]|Operator:|[A-Z][a-z]+ [A-Z][a-z]+:)', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Alternative: Remove each piece separately if the above is too aggressive
    text = re.sub(r'corrected transcript\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'www\.CallStreet\.com.*?CallStreet\s*', '', text, flags=re.DOTALL)
    text = re.sub(r'.*?Company▲.*?Date▲.*?(?=\d|\n|<)', '', text, flags=re.DOTALL)
    
    # Remove page numbers and other artifacts
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)  # Remove standalone numbers (page numbers)
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Clean up excessive line breaks
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def extract_paragraphs_from_transcript(transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract all paragraphs from speaker segments instead of raw paragraph arrays."""
    
    base_metadata = {
        'company_name': transcript.get('company_name', ''),
        'ticker': transcript.get('ticker', ''),
        'quarter': transcript.get('quarter', ''),
        'year': transcript.get('year', ''),
        'date': transcript.get('date', ''),
        'filename': transcript.get('filename', '')
    }
    
    all_paragraphs = []
    
    # Process management section from speaker segments
    mgmt_segments = transcript.get('speaker_segments_management', [])
    for seg_idx, segment in enumerate(mgmt_segments):
        speaker = segment.get('speaker', 'Unknown')
        profession = segment.get('profession', 'Unknown')
        
        for para_idx, paragraph in enumerate(segment.get('paragraphs', [])):
            paragraph = paragraph.strip()
            if paragraph:  # Only add non-empty paragraphs
                para_data = {
                    'text': paragraph,
                    'section': 'management',
                    'paragraph_id': f"mgmt_{seg_idx}_{para_idx}",
                    'speaker': speaker,
                    'profession': profession,
                    **base_metadata
                }
                all_paragraphs.append(para_data)
    
    # Process Q&A section from speaker segments  
    qa_segments = transcript.get('speaker_segments_qa', [])
    for seg_idx, segment in enumerate(qa_segments):
        speaker = segment.get('speaker', 'Unknown')
        profession = segment.get('profession', 'Unknown')
        qa_type = segment.get('qa_type', None)
        
        for para_idx, paragraph in enumerate(segment.get('paragraphs', [])):
            paragraph = paragraph.strip()
            if paragraph:
                para_data = {
                    'text': paragraph,
                    'section': 'qa',
                    'paragraph_id': f"qa_{seg_idx}_{para_idx}",
                    'speaker': speaker,
                    'profession': profession,
                    'qa_type': qa_type,
                    **base_metadata
                }
                all_paragraphs.append(para_data)
    
    return all_paragraphs

def extract_paragraphs_from_json_file(json_file: Path) -> Iterator[Dict[str, Any]]:
    """
    Generator that yields paragraphs from a structured JSON file.
    
    Args:
        json_file: Path to structured JSON file
        
    Yields:
        Individual paragraph dictionaries with metadata
    """
    logger.info(f"Processing {json_file.name}")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        transcripts = json.load(f)
    
    paragraph_count = 0
    for transcript in transcripts:
        paragraphs = extract_paragraphs_from_transcript(transcript)
        for paragraph in paragraphs:
            paragraph['source_file'] = json_file.name
            paragraph_count += 1
            yield paragraph
    
    logger.info(f"Extracted {paragraph_count} paragraphs from {json_file.name}")


def extract_paragraphs_from_folder(structured_json_folder: Path, 
                                  max_files: int = None,
                                  stock_index: str = 'UNKNOWN') -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Extract all paragraphs from structured JSON files in a folder.
    
    Args:
        structured_json_folder: Path to folder containing structured JSON files
        max_files: Maximum number of JSON files to process (None for all)
        stock_index: Stock index name for metadata
        
    Returns:
        Tuple of (texts, metadata) lists
    """
    json_files = sorted(structured_json_folder.glob("structured_calls_*.json"))
    
    if not json_files:
        raise ValueError(f"No structured JSON files found in {structured_json_folder}")
    
    # Limit files if specified
    if max_files is not None:
        json_files = json_files[:max_files]
        logger.info(f"Processing {len(json_files)} files (limited to {max_files})")
    else:
        logger.info(f"Processing all {len(json_files)} files")
    
    texts = []
    metadata = []
    
    # First pass: count paragraphs for progress bar
    total_paragraphs = 0
    logger.info("Counting paragraphs...")
    for json_file in json_files[:3]:  # Sample first 3 files
        count = sum(1 for _ in extract_paragraphs_from_json_file(json_file))
        total_paragraphs += count
    
    # Estimate total based on sample
    if len(json_files) > 3:
        avg_per_file = total_paragraphs / 3
        estimated_total = int(avg_per_file * len(json_files))
    else:
        estimated_total = total_paragraphs
    
    logger.info(f"Estimated {estimated_total} total paragraphs")
    
    # Second pass: extract all paragraphs
    with tqdm(total=estimated_total, desc="Extracting paragraphs", unit="paragraph") as pbar:
        for json_file in json_files:
            file_paragraph_count = 0
            for paragraph in extract_paragraphs_from_json_file(json_file):
                text = paragraph.get('text', '').strip()
                if text:  # Only add non-empty texts
                    texts.append(text)
                    
                    # Add stock index to metadata
                    paragraph['stock_index'] = stock_index
                    metadata.append(paragraph)
                    file_paragraph_count += 1
                    pbar.update(1)
            
            logger.debug(f"Extracted {file_paragraph_count} paragraphs from {json_file.name}")
    
    logger.info(f"✅ Extracted {len(texts)} paragraphs with metadata")
    return texts, metadata


def estimate_paragraph_counts(structured_json_folder: Path, 
                             max_files: int = None) -> Dict[str, int]:
    """
    Estimate the number of paragraphs in the folder.
    
    Args:
        structured_json_folder: Path to folder containing structured JSON files
        max_files: Maximum number of files to process
        
    Returns:
        Dictionary with paragraph count estimates
    """
    json_files = sorted(structured_json_folder.glob("structured_calls_*.json"))
    
    if max_files is not None:
        json_files = json_files[:max_files]
    
    if not json_files:
        return {'error': 'No JSON files found'}
    
    # Sample first few files
    sample_size = min(3, len(json_files))
    sample_files = json_files[:sample_size]
    
    total_paragraphs = 0
    file_counts = {}
    
    logger.info(f"Sampling {sample_size} files for estimation...")
    
    for json_file in sample_files:
        count = sum(1 for _ in extract_paragraphs_from_json_file(json_file))
        file_counts[json_file.name] = count
        total_paragraphs += count
    
    avg_per_file = total_paragraphs / sample_size if sample_size > 0 else 0
    estimated_total = int(avg_per_file * len(json_files))
    
    return {
        'sampled_files': sample_size,
        'total_files': len(json_files),
        'files_to_process': len(json_files),
        'avg_paragraphs_per_file': int(avg_per_file),
        'estimated_total_paragraphs': estimated_total,
        'sample_counts': file_counts
    }


def preview_paragraphs(structured_json_folder: Path, num_examples: int = 5) -> List[Dict[str, Any]]:
    """
    Preview some paragraphs to understand the data structure.
    
    Args:
        structured_json_folder: Path to folder containing structured JSON files
        num_examples: Number of example paragraphs to return
        
    Returns:
        List of example paragraph dictionaries
    """
    json_files = sorted(structured_json_folder.glob("structured_calls_*.json"))
    
    if not json_files:
        return []
    
    examples = []
    for json_file in json_files[:2]:  # Check first 2 files
        for paragraph in extract_paragraphs_from_json_file(json_file):
            examples.append(paragraph)
            if len(examples) >= num_examples:
                return examples
    
    return examples


if __name__ == "__main__":
    # Test paragraph extraction
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    structured_folder = Path("data/processed/SP500")  # Adjust path
    if structured_folder.exists():
        # Preview some paragraphs
        examples = preview_paragraphs(structured_folder, 3)
        print("Example paragraphs:")
        for i, ex in enumerate(examples, 1):
            print(f"\n{i}. Company: {ex['company_name']}, Section: {ex['section']}")
            print(f"   Speaker: {ex['speaker']} ({ex['profession']})")
            print(f"   Text: {ex['text'][:100]}...")
        
        # Get estimates
        estimates = estimate_paragraph_counts(structured_folder, max_files=5)
        print(f"\nEstimates: {estimates}")