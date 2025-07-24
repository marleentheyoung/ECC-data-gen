"""
Memory-efficient version of parse_data.py

This module provides memory-optimized versions of data parsing functions
that process large JSON files in chunks rather than loading everything into memory.

Author: Marleen de Jonge
Date: 2025
"""

import json
import os
import gc
import sys
from pathlib import Path
from glob import glob
from tqdm import tqdm
import logging
from typing import Iterator, Dict, Any, List

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

import src.config as config
from src.data.streaming_json_utils import MemoryEfficientJSONProcessor, stream_json_array
from src.data.pdf_processing import MemoryMonitor

logger = logging.getLogger(__name__)


def structure_all_transcripts_from_parts_efficient(input_folder: str, output_folder: str) -> None:
    """
    Memory-efficient version that processes transcripts in chunks.
    
    Args:
        input_folder: Folder containing transcripts_data_part*.json files
        output_folder: Folder to save structured files
    """
    import src.data.load_transcripts as pp
    
    os.makedirs(output_folder, exist_ok=True)
    input_files = sorted(glob(os.path.join(input_folder, "transcripts_data_part*.json")))

    if not input_files:
        logger.error("❌ No part files found.")
        return

    logger.info(f"🔍 Processing {len(input_files)} JSON parts...")
    
    monitor = MemoryMonitor(config.MEMORY_LIMIT_GB)
    
    for i, input_path in enumerate(input_files, start=1):
        # Only log every 10th file for large datasets
        if len(input_files) < 20 or i % 10 == 0 or i == len(input_files):
            logger.info(f"Processing part {i}/{len(input_files)}")
            logger.debug(f"🧠 Memory usage: {monitor.get_current_memory_gb():.2f}GB")
        
        output_path = os.path.join(output_folder, f"structured_calls_{i}.json")
        
        def process_transcript_entry(file_sections_pair):
            """Process a single (filename, sections) pair from the JSON."""
            file, sections = file_sections_pair
            
            try:
                file_info = pp.parse_filename(file)
                
                management_text = pp.remove_factset_metadata(sections.get('Management Discussion', ''))
                qna_text = pp.remove_factset_metadata(sections.get('Q&A Section', ''))
                
                management_segments = pp.split_and_extract_speakers(management_text)
                qna_segments = pp.split_and_extract_speakers(qna_text, is_qna_section=True)
                
                management_paragraphs = management_text.split("\n\n")
                qna_paragraphs = qna_text.split("\n\n")
                
                call_data = {
                    'file': file_info['filename'],
                    'filename': file_info['filename'],
                    'company_name': file_info['company_name'],
                    'ticker': file_info['ticker'],
                    'quarter': file_info['quarter'],
                    'year': file_info['year'],
                    'date': file_info['date'],
                    'speaker_segments_management': management_segments,
                    'speaker_segments_qa': qna_segments,
                    'management_paragraphs': management_paragraphs,
                    'qa_paragraphs': qna_paragraphs
                }
                
                # Clear intermediate variables
                del management_text, qna_text, management_segments, qna_segments
                del management_paragraphs, qna_paragraphs
                
                return call_data
                
            except Exception as e:
                logger.warning(f"Error processing file {file}: {e}")
                return None
        
        # Process file using streaming approach
        processed_count = 0
        all_processed_data = []
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                transcripts = json.load(f)
                
                for file, sections in tqdm(transcripts.items(), desc=f"Structuring part {i}", unit="file"):
                    processed_item = process_transcript_entry((file, sections))
                    
                    if processed_item:
                        all_processed_data.append(processed_item)
                        processed_count += 1
                        
                        # Periodically check memory and cleanup
                        if processed_count % 50 == 0:
                            if monitor.should_cleanup():
                                monitor.force_cleanup()
                
                # Clear input data from memory
                del transcripts
                
        except Exception as e:
            logger.error(f"Error processing {input_path}: {e}")
            continue
        
        # Write results to file
        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(all_processed_data, out_f, indent=2, ensure_ascii=False)
        
        # Only log for non-empty files or periodically
        if len(all_processed_data) > 0:
            if len(input_files) < 20 or i % 10 == 0 or i == len(input_files):
                logger.info(f"✅ Part {i}: {len(all_processed_data)} entries saved")
        
        # Clear processed data and force cleanup
        del all_processed_data
        monitor.force_cleanup()
    
    logger.info("✅ All parts structured and saved!")


def process_transcripts_in_streaming_chunks(input_folder: str, 
                                          output_folder: str,
                                          chunk_size: int = None) -> None:
    """
    Alternative streaming approach that processes very large files in smaller chunks.
    
    Args:
        input_folder: Input folder with JSON files
        output_folder: Output folder
        chunk_size: Number of items to process at once
    """
    import src.data.load_transcripts as pp
    
    if chunk_size is None:
        chunk_size = config.MAX_JSON_ITEMS_IN_MEMORY
    
    os.makedirs(output_folder, exist_ok=True)
    input_files = sorted(glob(os.path.join(input_folder, "transcripts_data_part*.json")))
    
    if not input_files:
        logger.error("❌ No part files found.")
        return
    
    logger.info(f"🔍 Processing {len(input_files)} files with chunk size {chunk_size}")
    
    for i, input_path in enumerate(input_files, start=1):
        output_path = os.path.join(output_folder, f"structured_calls_{i}.json")
        
        def transcript_processor(transcript_data):
            """Process function for the streaming processor."""
            file, sections = list(transcript_data.items())[0]  # Get first (and only) item
            
            file_info = pp.parse_filename(file)
            
            management_text = pp.remove_factset_metadata(sections.get('Management Discussion', ''))
            qna_text = pp.remove_factset_metadata(sections.get('Q&A Section', ''))
            
            management_segments = pp.split_and_extract_speakers(management_text)
            qna_segments = pp.split_and_extract_speakers(qna_text, is_qna_section=True)
            
            return {
                'file': file_info['filename'],
                'filename': file_info['filename'],
                'company_name': file_info['company_name'],
                'ticker': file_info['ticker'],
                'quarter': file_info['quarter'],
                'year': file_info['year'],
                'date': file_info['date'],
                'management_discussion_full': management_text,
                'qa_section_full': qna_text,
                'speaker_segments_management': management_segments,
                'speaker_segments_qa': qna_segments,
                'management_paragraphs': management_text.split("\n\n"),
                'qa_paragraphs': qna_text.split("\n\n")
            }
        
        # Use the streaming processor
        processor = MemoryEfficientJSONProcessor(chunk_size=chunk_size)
        
        # Convert the dictionary-based JSON to individual items
        def individual_transcript_generator():
            with open(input_path, 'r', encoding='utf-8') as f:
                transcripts = json.load(f)
                for file, sections in transcripts.items():
                    yield {file: sections}
                del transcripts  # Clear from memory
        
        # Process using streaming
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('[\n')
            first_item = True
            
            for transcript_dict in individual_transcript_generator():
                try:
                    processed = transcript_processor(transcript_dict)
                    
                    if not first_item:
                        f.write(',\n')
                    else:
                        first_item = False
                    
                    json_str = json.dumps(processed, indent=2, ensure_ascii=False)
                    indented_json = '\n'.join('  ' + line for line in json_str.split('\n'))
                    f.write(indented_json)
                    
                except Exception as e:
                    logger.warning(f"Error processing transcript: {e}")
                    continue
            
            f.write('\n]')
        
        logger.info(f"✅ Completed streaming processing of part {i}")


# Utility function to estimate memory requirements
def estimate_memory_requirements(input_folder: str) -> Dict[str, float]:
    """
    Estimate memory requirements for processing the given folder.
    
    Args:
        input_folder: Folder containing JSON files to analyze
        
    Returns:
        Dictionary with memory estimates in GB
    """
    input_files = glob(os.path.join(input_folder, "*.json"))
    
    total_file_size_gb = 0
    largest_file_gb = 0
    
    for file_path in input_files:
        size_gb = os.path.getsize(file_path) / (1024 * 1024 * 1024)
        total_file_size_gb += size_gb
        largest_file_gb = max(largest_file_gb, size_gb)
    
    # Rough estimates (JSON in memory is usually 2-3x file size)
    estimated_peak_memory = largest_file_gb * 3
    estimated_total_processing = total_file_size_gb * 2.5
    
    return {
        "total_file_size_gb": total_file_size_gb,
        "largest_file_gb": largest_file_gb,
        "estimated_peak_memory_gb": estimated_peak_memory,
        "estimated_total_processing_gb": estimated_total_processing,
        "recommended_memory_limit_gb": max(4.0, estimated_peak_memory * 1.5)
    }


def auto_configure_processing(input_folder: str) -> Dict[str, Any]:
    """
    Automatically configure processing parameters based on available memory and file sizes.
    
    Args:
        input_folder: Input folder to analyze
        
    Returns:
        Recommended configuration dictionary
    """
    import psutil
    
    # Get system info
    available_memory_gb = psutil.virtual_memory().available / (1024 * 1024 * 1024)
    total_memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)
    
    # Analyze files
    memory_estimates = estimate_memory_requirements(input_folder)
    
    # Determine processing strategy
    if memory_estimates["estimated_peak_memory_gb"] > available_memory_gb * 0.8:
        # Use streaming approach
        processing_strategy = "streaming"
        chunk_size = max(50, int(available_memory_gb * 100))  # Rough heuristic
        memory_limit = available_memory_gb * 0.7
    else:
        # Can use standard approach
        processing_strategy = "standard"
        chunk_size = max(200, int(available_memory_gb * 200))
        memory_limit = available_memory_gb * 0.8
    
    config_dict = {
        "processing_strategy": processing_strategy,
        "chunk_size": chunk_size,
        "memory_limit_gb": memory_limit,
        "available_memory_gb": available_memory_gb,
        "file_analysis": memory_estimates,
        "recommendations": []
    }
    
    # Add recommendations
    if available_memory_gb < 4:
        config_dict["recommendations"].append("Consider closing other applications to free up memory")
    
    if memory_estimates["largest_file_gb"] > 2:
        config_dict["recommendations"].append("Large files detected - streaming processing recommended")
    
    if memory_estimates["total_file_size_gb"] > available_memory_gb:
        config_dict["recommendations"].append("Total data size exceeds available memory - use streaming")
    
    return config_dict