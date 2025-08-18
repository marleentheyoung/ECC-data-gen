"""
Memory-efficient PDF processing utilities.

This module provides memory-optimized versions of PDF processing functions
to handle large datasets without running out of memory.

Author: Marleen de Jonge
Date: 2025
"""

import os
import json
import gc
import sys
from pathlib import Path
from typing import Generator, Dict, Any, Optional
from tqdm import tqdm
import psutil
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Monitor memory usage and trigger cleanup when needed."""
    
    def __init__(self, memory_threshold_gb: float = 4.0):
        self.memory_threshold_bytes = memory_threshold_gb * 1024 * 1024 * 1024
        self.initial_memory = psutil.Process().memory_info().rss
        
    def get_current_memory_gb(self) -> float:
        """Get current memory usage in GB."""
        return psutil.Process().memory_info().rss / (1024 * 1024 * 1024)
    
    def get_memory_increase_gb(self) -> float:
        """Get memory increase since initialization in GB."""
        current = psutil.Process().memory_info().rss
        return (current - self.initial_memory) / (1024 * 1024 * 1024)
    
    def should_cleanup(self) -> bool:
        """Check if memory cleanup is needed."""
        current_memory = psutil.Process().memory_info().rss
        return current_memory > self.memory_threshold_bytes
    
    def force_cleanup(self):
        """Force garbage collection and log memory status."""
        before = self.get_current_memory_gb()
        gc.collect()
        after = self.get_current_memory_gb()
        freed = before - after
        
        if freed > 0.1:  # Only log if significant memory was freed
            logger.info(f"Memory cleanup: {before:.2f}GB → {after:.2f}GB (freed {freed:.2f}GB)")


def extract_transcripts_memory_efficient(pdf_root_folder: str, 
                                        output_basename: str, 
                                        num_parts: int,
                                        memory_limit_gb: float = 4.0) -> None:
    """
    Memory-efficient version of extract_transcripts.
    
    Args:
        pdf_root_folder: Folder containing PDF files
        output_basename: Base name for output files
        num_parts: Number of parts to split processing into
        memory_limit_gb: Memory limit in GB before forcing cleanup
    """
    from src.utils import PDFreader
    from math import ceil
    
    monitor = MemoryMonitor(memory_limit_gb)
    
    # Get all PDF files
    all_files = sorted([f for f in os.listdir(pdf_root_folder) if f.lower().endswith('.pdf')])
    total_files = len(all_files)

    if total_files == 0:
        logger.error("❌ No PDF files found in the input folder.")
        return

    logger.info(f"📄 Found {total_files} PDF transcripts. Splitting into {num_parts} parts.")
    if memory_limit_gb < 8:  # Only show memory limit for constrained systems
        logger.info(f"🧠 Memory limit set to {memory_limit_gb:.1f}GB")

    # Calculate batch size
    chunk_size = ceil(total_files / num_parts)

    for i in range(num_parts):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total_files)
        batch_files = all_files[start:end]
        
        if not batch_files:  # Skip empty parts
            continue
            
        # Only log every 10th part for large datasets, or all parts for small ones
        if total_files < 100 or i % 10 == 0 or i == num_parts - 1:
            logger.info(f"🧩 Processing part {i + 1}/{num_parts} | Files {start + 1} to {end}")
        
        # Only show memory usage periodically for large datasets
        if total_files < 100 or i % 10 == 0:
            logger.debug(f"🧠 Current memory usage: {monitor.get_current_memory_gb():.2f}GB")
        
        # Process files in smaller sub-batches to control memory
        sub_batch_size = max(1, min(10, len(batch_files)))  # Process max 10 files at a time
        all_transcripts = {}
        
        for j in range(0, len(batch_files), sub_batch_size):
            sub_batch = batch_files[j:j + sub_batch_size]
            
            for filename in tqdm(sub_batch, desc=f"Part {i + 1}, batch {j//sub_batch_size + 1}", unit="file"):
                pdf_path = os.path.join(pdf_root_folder, filename)
                
                try:
                    # Process single PDF
                    text = PDFreader.extract_text_from_pdf(pdf_path)
                    sections = PDFreader.split_text_sections(filename, text)
                    
                    if sections:
                        all_transcripts[filename] = sections
                    
                    # Clear variables immediately after use
                    del text
                    if 'sections' in locals():
                        del sections
                    
                    # Check memory and cleanup if needed
                    if monitor.should_cleanup():
                        monitor.force_cleanup()
                        
                except Exception as e:
                    logger.warning(f"⚠️ Error processing {filename}: {e}")
                    continue
            
            # Force cleanup after each sub-batch
            monitor.force_cleanup()

        # Save part to disk and clear from memory
        output_path = f"{output_basename}_part{i + 1}.json"
        
        # Use streaming write for large files
        with open(output_path, "w", encoding="utf-8") as json_file:
            json.dump(all_transcripts, json_file, indent=2, ensure_ascii=False)
        
        # Only log completion for meaningful parts or periodically for large datasets
        if len(all_transcripts) > 0:
            if total_files < 100 or i % 10 == 0 or i == num_parts - 1:
                logger.info(f"✅ Part {i + 1}: {len(all_transcripts)} transcripts saved")
        
        # Only show memory for small datasets or periodically
        if total_files < 100 or i % 10 == 0:
            logger.debug(f"🧠 Memory after part {i + 1}: {monitor.get_current_memory_gb():.2f}GB")
        
        # Clear the transcripts dict and force cleanup
        del all_transcripts
        monitor.force_cleanup()


def process_large_json_files_efficiently(input_folder: str, 
                                        output_folder: str,
                                        max_items_in_memory: int = 500) -> None:
    """
    Process large JSON files without loading them entirely into memory.
    
    Args:
        input_folder: Folder containing JSON files to process
        output_folder: Output folder for processed files
        max_items_in_memory: Maximum number of items to keep in memory
    """
    from glob import glob
    
    monitor = MemoryMonitor()
    os.makedirs(output_folder, exist_ok=True)
    
    input_files = sorted(glob(os.path.join(input_folder, "*.json")))
    
    for input_file in input_files:
        output_file = os.path.join(output_folder, f"processed_{os.path.basename(input_file)}")
        
        logger.info(f"Processing {input_file}")
        logger.info(f"🧠 Memory before processing: {monitor.get_current_memory_gb():.2f}GB")
        
        processed_items = []
        item_count = 0
        
        # Stream process the JSON file
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if not isinstance(data, list):
                    logger.warning(f"Skipping {input_file}: not a JSON array")
                    continue
                
                for item in tqdm(data, desc=f"Processing {os.path.basename(input_file)}"):
                    # Process individual item (your processing logic here)
                    processed_item = process_single_transcript_item(item)
                    processed_items.append(processed_item)
                    item_count += 1
                    
                    # Write to disk when we reach the memory limit
                    if len(processed_items) >= max_items_in_memory:
                        write_partial_results(processed_items, output_file, item_count == len(processed_items))
                        processed_items = []  # Clear from memory
                        monitor.force_cleanup()
                
                # Write remaining items
                if processed_items:
                    write_partial_results(processed_items, output_file, True)
                
                # Clear input data from memory
                del data
                
        except Exception as e:
            logger.error(f"Error processing {input_file}: {e}")
            continue
        
        logger.info(f"✅ Processed {item_count} items from {os.path.basename(input_file)}")
        logger.info(f"🧠 Memory after processing: {monitor.get_current_memory_gb():.2f}GB")
        monitor.force_cleanup()


def process_single_transcript_item(item: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Process a single transcript item efficiently.
    
    Args:
        item: Single transcript dictionary
        
    Returns:
        Processed transcript dictionary
    """
    # Your existing processing logic here
    # This function should process one transcript at a time
    
    # Example: just return the item (replace with your logic)
    return item


def write_partial_results(items: list, output_file: str, is_final: bool = False) -> None:
    """
    Write partial results to file, handling the JSON array format properly.
    
    Args:
        items: List of items to write
        output_file: Output file path
        is_final: Whether this is the final write
    """
    mode = 'w' if not os.path.exists(output_file) else 'a'
    
    with open(output_file, mode, encoding='utf-8') as f:
        if mode == 'w':
            f.write('[\n')
        
        for i, item in enumerate(items):
            if mode == 'a' or i > 0:
                f.write(',\n')
            
            json_str = json.dumps(item, indent=2, ensure_ascii=False)
            # Add proper indentation
            indented_json = '\n'.join('  ' + line for line in json_str.split('\n'))
            f.write(indented_json)
        
        if is_final:
            f.write('\n]')
        
        f.flush()  # Ensure data is written to disk


# Configuration function to automatically adjust settings based on available memory
def get_optimal_batch_sizes() -> Dict[str, int]:
    """
    Automatically determine optimal batch sizes based on available memory.
    
    Returns:
        Dictionary with recommended batch sizes
    """
    # Get available memory in GB
    available_memory_gb = psutil.virtual_memory().available / (1024 * 1024 * 1024)
    
    if available_memory_gb < 2:
        return {
            "pdf_sub_batch_size": 1,
            "json_max_items": 100,
            "num_parts_multiplier": 2
        }
    elif available_memory_gb < 4:
        return {
            "pdf_sub_batch_size": 3,
            "json_max_items": 250,
            "num_parts_multiplier": 1.5
        }
    elif available_memory_gb < 8:
        return {
            "pdf_sub_batch_size": 5,
            "json_max_items": 500,
            "num_parts_multiplier": 1
        }
    else:
        return {
            "pdf_sub_batch_size": 10,
            "json_max_items": 1000,
            "num_parts_multiplier": 0.8
        }