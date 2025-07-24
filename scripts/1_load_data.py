#!/usr/bin/env python3
"""
Memory-efficient main pipeline for processing raw transcript PDFs to structured JSONs.

This version includes memory monitoring, automatic cleanup, and streaming processing
to handle large datasets without running out of memory.

Author: Marleen de Jonge
Date: 2025
"""

import logging
import sys
import psutil
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src import config
from src.data.load_transcripts import find_and_delete_duplicate_filenames
from src.data.pdf_processing import extract_transcripts_memory_efficient, get_optimal_batch_sizes
from src.data.parse_data import (
    structure_all_transcripts_from_parts_efficient, 
    auto_configure_processing,
    estimate_memory_requirements
)
from src.data.parser import process_all_pdfs_in_directory


def setup_logging(level: str = None) -> None:
    """Setup logging with memory usage tracking."""
    level = level or config.DEFAULT_LOG_LEVEL
    log_file = config.get_log_file_path()
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=config.LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )


def log_system_info():
    """Log system information for debugging."""
    logger = logging.getLogger(__name__)
    
    # Memory info
    memory = psutil.virtual_memory()
    logger.info(f"🧠 System Memory: {memory.total / (1024**3):.1f}GB total, "
               f"{memory.available / (1024**3):.1f}GB available "
               f"({memory.percent:.1f}% used)")
    
    # CPU info
    cpu_count = psutil.cpu_count()
    logger.info(f"🖥️  CPU: {cpu_count} cores")
    
    # Disk space
    disk = psutil.disk_usage(str(config.BASE_DIR))
    logger.info(f"💾 Disk Space: {disk.free / (1024**3):.1f}GB free of "
               f"{disk.total / (1024**3):.1f}GB total")


def check_memory_requirements(stock_index: str) -> bool:
    """
    Check if system has enough memory for processing.
    
    Args:
        stock_index: Stock index to process
        
    Returns:
        True if system can handle the processing
    """
    logger = logging.getLogger(__name__)
    
    # Check available memory
    available_gb = psutil.virtual_memory().available / (1024**3)
    
    if available_gb < 2.0:
        logger.error(f"❌ Insufficient memory: {available_gb:.1f}GB available, minimum 2GB required")
        logger.info("💡 Try closing other applications or use a machine with more memory")
        return False
    
    if available_gb < 4.0:
        logger.warning(f"⚠️  Low memory: {available_gb:.1f}GB available. Processing will use streaming mode.")
    
    # Estimate requirements for input data
    pdf_folder = config.get_pdf_folder(stock_index)
    if pdf_folder.exists():
        try:
            # Quick estimate based on PDF folder size
            total_size_gb = sum(f.stat().st_size for f in pdf_folder.rglob("*.pdf")) / (1024**3)
            estimated_memory_needed = total_size_gb * 0.5  # Rough estimate
            
            logger.info(f"📊 Input PDFs: {total_size_gb:.1f}GB, estimated memory needed: {estimated_memory_needed:.1f}GB")
            
            if estimated_memory_needed > available_gb * 0.8:
                logger.warning("⚠️  Large dataset detected - will use maximum memory efficiency settings")
        except Exception as e:
            logger.debug(f"Could not estimate PDF folder size: {e}")
    
    return True


def main(stock_index: str, force_streaming: bool = False) -> None:
    """
    Memory-efficient main pipeline for processing transcript PDFs to structured JSONs.
    
    Args:
        stock_index: Stock index to process (e.g., 'STOXX600', 'SP500')
        force_streaming: Force use of streaming processing regardless of memory
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Log system information
    log_system_info()
    
    # Validate environment
    if not config.validate_environment():
        logger.error("Environment validation failed")
        return
    
    # Validate stock index
    if not config.validate_stock_index(stock_index):
        logger.error(f"Unsupported stock index: {stock_index}")
        logger.info(f"Supported indices: {', '.join(config.SUPPORTED_INDICES)}")
        return
    
    # Check memory requirements
    if not check_memory_requirements(stock_index):
        return
    
    logger.info(f"Starting memory-efficient transcript processing pipeline for {stock_index}")
    
    try:
        # Get paths from config
        pdf_folder = config.get_pdf_folder(stock_index)
        raw_json_folder = config.get_raw_json_folder(stock_index)
        structured_json_folder = config.get_structured_json_folder(stock_index)
        final_output_folder = config.get_final_output_folder(stock_index)
        
        # Create output directories
        config.create_output_directories(stock_index)
        
        logger.info(f"Processing PDFs from: {pdf_folder}")
        logger.info(f"Output will be saved to: {final_output_folder}")
        
        # Get optimal batch sizes based on available memory
        if config.AUTO_ADJUST_MEMORY:
            batch_config = get_optimal_batch_sizes()
            logger.info(f"🧠 Auto-configured batch sizes: {batch_config}")
        else:
            batch_config = {
                "pdf_sub_batch_size": config.PDF_SUB_BATCH_SIZE,
                "json_max_items": config.MAX_JSON_ITEMS_IN_MEMORY,
                "num_parts_multiplier": 1
            }
        
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
        
        # Step 2: Extract transcripts from PDFs (memory-efficient)
        logger.info("Step 2: Extracting transcripts from PDFs (memory-efficient)...")
        output_file_base = raw_json_folder / "transcripts_data"
        
        # Adjust number of parts based on memory
        base_num_parts = config.DEFAULT_NUM_PARTS
        if config.TEST_MODE:
            num_parts = min(2, base_num_parts)
        else:
            num_parts = max(5, int(base_num_parts * batch_config.get("num_parts_multiplier", 1)))
        
        logger.info(f"🧩 Using {num_parts} parts for PDF processing")
        
        extract_transcripts_memory_efficient(
            pdf_root_folder=str(pdf_folder),
            output_basename=str(output_file_base),
            num_parts=num_parts,
            memory_limit_gb=config.MEMORY_LIMIT_GB
        )
        logger.info("✅ Memory-efficient transcript extraction completed")
        
        # Step 3: Structure transcripts into standardized format (memory-efficient)
        logger.info("Step 3: Structuring transcripts (memory-efficient)...")
        
        # Auto-configure processing based on the raw JSON files
        if raw_json_folder.exists():
            processing_config = auto_configure_processing(str(raw_json_folder))
            logger.info(f"📊 Processing analysis: {processing_config['file_analysis']}")
            
            if processing_config['recommendations']:
                logger.info("💡 Recommendations:")
                for rec in processing_config['recommendations']:
                    logger.info(f"   • {rec}")
            
            # Use streaming if recommended or forced
            use_streaming = (processing_config["processing_strategy"] == "streaming" or 
                           force_streaming or 
                           psutil.virtual_memory().available < 3 * 1024**3)  # Less than 3GB available
            
            if use_streaming:
                logger.info("🌊 Using streaming processing for large datasets")
                from src.data.memory_efficient_parse_data import process_transcripts_in_streaming_chunks
                process_transcripts_in_streaming_chunks(
                    input_folder=str(raw_json_folder),
                    output_folder=str(structured_json_folder),
                    chunk_size=processing_config.get('chunk_size', config.MAX_JSON_ITEMS_IN_MEMORY)
                )
            else:
                logger.info("⚡ Using efficient batch processing")
                structure_all_transcripts_from_parts_efficient(
                    input_folder=str(raw_json_folder),
                    output_folder=str(structured_json_folder)
                )
        else:
            logger.warning("Raw JSON folder not found, using standard processing")
            structure_all_transcripts_from_parts_efficient(
                input_folder=str(raw_json_folder),
                output_folder=str(structured_json_folder)
            )
        
        logger.info("✅ Memory-efficient transcript structuring completed")
        
        # Step 4: Process transcripts for analysis
        logger.info("Step 4: Processing transcripts for metadata extraction...")
        process_all_pdfs_in_directory(
            folder_path=str(structured_json_folder),
            pdf_folder=str(pdf_folder), 
            index=stock_index
        )
        logger.info("✅ Transcript processing completed")
        
        # Step 5: Move final results to outputs folder
        logger.info("Step 5: Moving final results to outputs folder...")
        import shutil
        
        # Copy structured JSONs to final output location
        if structured_json_folder.exists():
            if final_output_folder.exists():
                shutil.rmtree(final_output_folder)
            shutil.copytree(structured_json_folder, final_output_folder)
        
        # Final memory check
        final_memory = psutil.virtual_memory()
        logger.info(f"🧠 Final memory usage: {final_memory.percent:.1f}% "
                   f"({final_memory.available / (1024**3):.1f}GB available)")
        
        logger.info(f"🎉 Memory-efficient pipeline completed successfully for {stock_index}")
        logger.info(f"📁 Final structured JSONs available at: {final_output_folder}")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        
        # Log memory state during error
        try:
            memory = psutil.virtual_memory()
            logger.error(f"💾 Memory at time of error: {memory.percent:.1f}% used, "
                        f"{memory.available / (1024**3):.1f}GB available")
        except:
            pass
        
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Memory-efficient processing of raw transcript PDFs to structured JSONs"
    )
    parser.add_argument(
        "stock_index",
        help=f"Stock index to process ({', '.join(config.SUPPORTED_INDICES)})"
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",  # Changed default to WARNING for less verbose output
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    parser.add_argument(
        "--force-streaming",
        action="store_true",
        help="Force use of streaming processing regardless of available memory"
    )
    parser.add_argument(
        "--memory-limit",
        type=float,
        default=config.MEMORY_LIMIT_GB,
        help="Memory limit in GB (default: from config)"
    )
    
    args = parser.parse_args()
    
    # Override config with command line arguments
    if args.memory_limit != config.MEMORY_LIMIT_GB:
        config.MEMORY_LIMIT_GB = args.memory_limit
        print(f"🧠 Memory limit set to {args.memory_limit}GB")
    
    main(stock_index=args.stock_index, force_streaming=args.force_streaming)