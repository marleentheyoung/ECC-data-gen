"""
Streaming JSON utilities for memory-efficient processing.

This module provides utilities to process large JSON files without loading
them entirely into memory.

Author: Marleen de Jonge
Date: 2025
"""

import json
import ijson
from typing import Iterator, Dict, Any, List
from pathlib import Path
import gc


def stream_json_array(file_path: str) -> Iterator[Dict[Any, Any]]:
    """
    Stream elements from a JSON array file one at a time.
    
    Args:
        file_path: Path to JSON file containing an array
        
    Yields:
        Individual elements from the JSON array
    """
    with open(file_path, 'rb') as file:
        # Use ijson to parse array elements one by one
        parser = ijson.items(file, 'item')
        for item in parser:
            yield item


def process_json_in_chunks(file_path: str, chunk_size: int = 100) -> Iterator[List[Dict[Any, Any]]]:
    """
    Process JSON array in chunks to control memory usage.
    
    Args:
        file_path: Path to JSON file
        chunk_size: Number of items to process at once
        
    Yields:
        Chunks of items from the JSON array
    """
    chunk = []
    for item in stream_json_array(file_path):
        chunk.append(item)
        
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
            # Force garbage collection after each chunk
            gc.collect()
    
    # Yield remaining items
    if chunk:
        yield chunk


def write_json_streaming(data_iterator: Iterator, output_path: str, indent: int = 2) -> None:
    """
    Write JSON data using streaming to avoid memory buildup.
    
    Args:
        data_iterator: Iterator yielding data items
        output_path: Output file path
        indent: JSON indentation
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('[\n')
        
        first_item = True
        for item in data_iterator:
            if not first_item:
                f.write(',\n')
            else:
                first_item = False
                
            # Write item with proper indentation
            json_str = json.dumps(item, indent=indent, ensure_ascii=False)
            # Add proper indentation for the entire item
            indented_json = '\n'.join('  ' + line for line in json_str.split('\n'))
            f.write(indented_json)
            
            # Force buffer flush periodically
            if not first_item and (hash(str(item)) % 50 == 0):
                f.flush()
        
        f.write('\n]')


class MemoryEfficientJSONProcessor:
    """
    A class for memory-efficient JSON processing with built-in monitoring.
    """
    
    def __init__(self, chunk_size: int = 100):
        self.chunk_size = chunk_size
        self.processed_count = 0
        
    def process_file(self, input_path: str, output_path: str, 
                    processor_func: callable) -> None:
        """
        Process a JSON file in memory-efficient chunks.
        
        Args:
            input_path: Input JSON file path
            output_path: Output JSON file path
            processor_func: Function to process each item
        """
        def processed_items():
            for chunk in process_json_in_chunks(input_path, self.chunk_size):
                for item in chunk:
                    processed_item = processor_func(item)
                    self.processed_count += 1
                    
                    if self.processed_count % 100 == 0:
                        print(f"Processed {self.processed_count} items...")
                        
                    yield processed_item
                
                # Clear chunk from memory
                del chunk
                gc.collect()
        
        write_json_streaming(processed_items(), output_path)
        print(f"✅ Completed processing {self.processed_count} items")


# Example usage functions
def memory_efficient_transcript_processing(input_folder: str, output_folder: str):
    """
    Example of how to use streaming processing for transcript data.
    """
    from glob import glob
    import os
    
    input_files = glob(os.path.join(input_folder, "*.json"))
    
    for input_file in input_files:
        output_file = os.path.join(output_folder, f"processed_{os.path.basename(input_file)}")
        
        def process_transcript(transcript_data):
            # Your processing logic here
            # This function processes one transcript at a time
            return transcript_data
        
        processor = MemoryEfficientJSONProcessor(chunk_size=50)
        processor.process_file(input_file, output_file, process_transcript)