#!/usr/bin/env python3
"""
Main pipeline for processing raw transcript PDFs to structured JSONs.

This script orchestrates the complete workflow using existing functions:
1. Find and remove duplicate files
2. Extract transcripts from PDFs
3. Structure transcripts into standardized JSON format
4. Process transcripts for analysis

Author: Marleen de Jonge
Date: 2025
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from load_transcripts import find_and_delete_duplicate_filenames, extract_transcripts
from parse_data import structure_all_transcripts_from_parts
from parser import process_all_pdfs_in_directory


def setup_logging(level: str = "INFO") -> None:
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("transcript_processing.log")
        ]
    )


def main(stock_index: str) -> None:
    """
    Main pipeline for processing transcript PDFs to structured JSONs.
    
    Args:
        stock_index: Stock index to process (e.g., 'STOXX600', 'SP500')
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting transcript processing pipeline for {stock_index}")
    
    try:
        # Define base paths
        base_dir = Path("/Users/marleendejonge/Desktop/ECC-data-generation")
        
        # Input paths
        pdf_folder = base_dir / "data" / "raw" / stock_index
        
        # Output paths
        raw_json_folder = base_dir / "data" / "processed" / "raw_jsons" / stock_index
        structured_json_folder = base_dir / "data" / "processed" / "structured_jsons" / stock_index
        final_output_folder = base_dir / "outputs" / "processed_transcripts" / stock_index
        
        # Create output directories
        for folder in [raw_json_folder, structured_json_folder, final_output_folder]:
            folder.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing PDFs from: {pdf_folder}")
        logger.info(f"Output will be saved to: {final_output_folder}")
        
        # Step 1: Find and remove duplicates
        logger.info("Step 1: Checking for duplicate files...")
        duplicates = find_and_delete_duplicate_filenames(str(pdf_folder), delete=True)
        
        if duplicates:
            logger.warning(f"Found and removed {len(duplicates)} duplicate filenames")
            for duplicate in duplicates[:5]:  # Show first 5
                logger.debug(f"Duplicate: {duplicate}")
            if len(duplicates) > 5:
                logger.debug(f"... and {len(duplicates) - 5} more")
        else:
            logger.info("✅ No duplicate filenames found")
        
        # Step 2: Extract transcripts from PDFs
        logger.info("Step 2: Extracting transcripts from PDFs...")
        output_file_base = raw_json_folder / "transcripts_data"
        
        extract_transcripts(
            pdf_root_folder=str(pdf_folder),
            output_basename=str(output_file_base),
            num_parts=15
        )
        logger.info("✅ Transcript extraction completed")
        
        # Step 3: Structure transcripts into standardized format
        logger.info("Step 3: Structuring transcripts into standardized JSON format...")
        structure_all_transcripts_from_parts(
            input_folder=str(raw_json_folder),
            output_folder=str(structured_json_folder)
        )
        logger.info("✅ Transcript structuring completed")
        
        # Step 4: Process transcripts for analysis
        logger.info("Step 4: Processing transcripts for metadata extraction...")
        process_all_pdfs_in_directory(
            folder_path=str(structured_json_folder),
            pdf_folder=str(pdf_folder), 
            index=stock_index
        )
        logger.info("✅ Transcript processing completed")
        
        # Move final results to outputs folder
        logger.info("Step 5: Moving final results to outputs folder...")
        import shutil
        
        # Copy structured JSONs to final output location
        if structured_json_folder.exists():
            if final_output_folder.exists():
                shutil.rmtree(final_output_folder)
            shutil.copytree(structured_json_folder, final_output_folder)
        
        logger.info(f"🎉 Pipeline completed successfully for {stock_index}")
        logger.info(f"📁 Final structured JSONs available at: {final_output_folder}")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process raw transcript PDFs to structured JSONs"
    )
    parser.add_argument(
        "stock_index",
        help="Stock index to process (e.g., STOXX600, SP500)"
    )
    
    args = parser.parse_args()
    
    main(stock_index=args.stock_index)