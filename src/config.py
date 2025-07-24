"""
Configuration settings for ECC Data Generation Pipeline.

This module centralizes all configuration settings, paths, and constants
used throughout the pipeline.

Author: Marleen de Jonge
Date: 2025
"""

import os
from pathlib import Path

# ================================
# BASE DIRECTORIES
# ================================

# Main project directory - can be overridden with environment variable
BASE_DIR = Path(os.getenv('ECC_BASE_DIR', '/Users/marleendejonge/Desktop/ECC-data-generation'))

# Input directories
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# Output directories
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"

# ================================
# PROCESSING SETTINGS
# ================================

# Number of parts to split PDF processing into (for memory management)
DEFAULT_NUM_PARTS = int(os.getenv('ECC_NUM_PARTS', '15'))

# Supported stock indices
SUPPORTED_INDICES = ['STOXX600', 'SP500']

# ================================
# FILE PATTERNS
# ================================

# PDF file patterns
PDF_EXTENSION = '.pdf'
PDF_PREFIX_TO_REMOVE = 'CORRECTED TRANSCRIPT'

# JSON file patterns
RAW_JSON_PATTERN = "transcripts_data_part*.json"
STRUCTURED_JSON_PATTERN = "structured_calls_*.json"

# ================================
# LOGGING SETTINGS
# ================================

# Default logging level - use WARNING for less verbose output
DEFAULT_LOG_LEVEL = os.getenv('ECC_LOG_LEVEL', 'WARNING')

# Log file name
LOG_FILE_NAME = "transcript_processing.log"

# Log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ================================
# PDF PROCESSING SETTINGS
# ================================

# Section headers to look for in transcripts
MANAGEMENT_HEADER = "MANAGEMENT DISCUSSION SECTION"
QA_HEADER = "QUESTION AND ANSWER SECTION"

# FactSet metadata patterns
FACTSET_METADATA_PATTERN = "FactSet CallStreet, LLC"
FACTSET_METADATA_LINES_BEFORE = 10

# ================================
# SPACY SETTINGS
# ================================

# spaCy model to use
SPACY_MODEL = "en_core_web_sm"

# ================================
# UTILITY FUNCTIONS
# ================================

def get_pdf_folder(stock_index: str) -> Path:
    """Get the PDF folder path for a given stock index."""
    return RAW_DATA_DIR / stock_index

def get_raw_json_folder(stock_index: str) -> Path:
    """Get the raw JSON folder path for a given stock index."""
    return PROCESSED_DATA_DIR / "raw_jsons" / stock_index

def get_structured_json_folder(stock_index: str) -> Path:
    """Get the structured JSON folder path for a given stock index."""
    return PROCESSED_DATA_DIR / "structured_jsons" / stock_index

def get_final_output_folder(stock_index: str) -> Path:
    """Get the final output folder path for a given stock index."""
    return OUTPUTS_DIR / "processed_transcripts" / stock_index

def get_log_file_path() -> Path:
    """Get the log file path."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return LOGS_DIR / LOG_FILE_NAME

def validate_stock_index(stock_index: str) -> bool:
    """Validate if the stock index is supported."""
    return stock_index in SUPPORTED_INDICES

def create_output_directories(stock_index: str) -> None:
    """Create all necessary output directories for a stock index."""
    directories = [
        get_raw_json_folder(stock_index),
        get_structured_json_folder(stock_index),
        get_final_output_folder(stock_index),
        LOGS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# ================================
# ENVIRONMENT VALIDATION
# ================================

def validate_environment() -> bool:
    """Validate that the environment is properly set up."""
    # Check if base directory exists
    if not BASE_DIR.exists():
        print(f"❌ Base directory does not exist: {BASE_DIR}")
        return False
    
    # Check if raw data directory exists
    if not RAW_DATA_DIR.exists():
        print(f"❌ Raw data directory does not exist: {RAW_DATA_DIR}")
        return False
    
    return True

# ================================
# MEMORY MANAGEMENT SETTINGS
# ================================

# Memory limit in GB before forcing cleanup
MEMORY_LIMIT_GB = float(os.getenv('ECC_MEMORY_LIMIT', '4.0'))

# Maximum items to keep in memory when processing JSON
MAX_JSON_ITEMS_IN_MEMORY = int(os.getenv('ECC_MAX_JSON_ITEMS', '500'))

# PDF processing sub-batch size (files processed before memory cleanup)
PDF_SUB_BATCH_SIZE = int(os.getenv('ECC_PDF_SUB_BATCH', '5'))

# Auto-adjust batch sizes based on available memory
AUTO_ADJUST_MEMORY = os.getenv('ECC_AUTO_ADJUST_MEMORY', 'True').lower() == 'true'

# ================================
# DEVELOPMENT SETTINGS
# ================================

# Enable debug mode with environment variable
DEBUG_MODE = os.getenv('ECC_DEBUG', 'False').lower() == 'true'

# Test mode - processes fewer files for testing
TEST_MODE = os.getenv('ECC_TEST_MODE', 'False').lower() == 'true'
TEST_MODE_MAX_FILES = int(os.getenv('ECC_TEST_MAX_FILES', '10'))

if DEBUG_MODE:
    print("🔧 Debug mode enabled")
    
if TEST_MODE:
    print(f"🧪 Test mode enabled - processing max {TEST_MODE_MAX_FILES} files")

if AUTO_ADJUST_MEMORY:
    import psutil
    available_gb = psutil.virtual_memory().available / (1024 * 1024 * 1024)
    print(f"🧠 Available memory: {available_gb:.1f}GB - auto-adjusting batch sizes")

# ================================
# SEMANTIC SEARCH SETTINGS  
# ================================

# Semantic indexes directory
SEMANTIC_INDEXES_DIR = BASE_DIR / "data" / "generated" / "semantic_indexes"

# Default sentence transformer model
DEFAULT_SENTENCE_TRANSFORMER = "sentence-transformers/all-MiniLM-L6-v2"

# Embedding settings
DEFAULT_BATCH_SIZE = 64
DEFAULT_TOP_K = 10

def get_semantic_index_folder(stock_index: str) -> Path:
    """Get the semantic index folder path for a given stock index."""
    return SEMANTIC_INDEXES_DIR / stock_index