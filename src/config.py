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

# Climate analysis directories
CLIMATE_DATA_DIR = BASE_DIR / "data" / "climate_paragraphs"
ENHANCED_CLIMATE_DATA_DIR = BASE_DIR / "data" / "enhanced_climate_snippets"

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
CLIMATE_SEGMENTS_PATTERN = "climate_segments_*.json"
ENHANCED_CLIMATE_SEGMENTS_PATTERN = "enhanced_climate_segments_*.json"

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
# CLIMATE ANALYSIS SETTINGS
# ================================

# ClimateBERT model settings
CLIMATE_BERT_MODEL = "climatebert/distilroberta-base-climate-detector"
CLIMATE_CLASSIFICATION_BATCH_SIZE = 32
CLIMATE_MIN_WORDS_PER_PARAGRAPH = 10

# Climate snippet processing
CLIMATE_RELEVANCE_THRESHOLD = 0.40
CLIMATE_SENTIMENT_CATEGORIES = ['opportunity', 'risk', 'neutral']

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

def get_climate_data_folder(stock_index: str) -> Path:
    """Get the climate paragraphs folder path for a given stock index."""
    return CLIMATE_DATA_DIR / stock_index

def get_enhanced_climate_data_folder(stock_index: str) -> Path:
    """Get the enhanced climate snippets folder path for a given stock index."""
    return ENHANCED_CLIMATE_DATA_DIR / stock_index

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
        get_climate_data_folder(stock_index),
        get_enhanced_climate_data_folder(stock_index),
        LOGS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def create_climate_directories() -> None:
    """Create climate analysis directories for all supported indices."""
    for stock_index in SUPPORTED_INDICES:
        get_climate_data_folder(stock_index).mkdir(parents=True, exist_ok=True)
        get_enhanced_climate_data_folder(stock_index).mkdir(parents=True, exist_ok=True)

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

def validate_climate_environment() -> bool:
    """Validate that the climate analysis environment is set up."""
    # Check if climate data directories exist
    if not CLIMATE_DATA_DIR.exists():
        print(f"❌ Climate data directory does not exist: {CLIMATE_DATA_DIR}")
        return False
    
    # Check if we have climate data for at least one index
    has_climate_data = False
    for stock_index in SUPPORTED_INDICES:
        climate_folder = get_climate_data_folder(stock_index)
        if climate_folder.exists() and list(climate_folder.glob("climate_segments_*.json")):
            has_climate_data = True
            break
    
    if not has_climate_data:
        print(f"❌ No climate segment data found in: {CLIMATE_DATA_DIR}")
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
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024 * 1024 * 1024)
        print(f"🧠 Available memory: {available_gb:.1f}GB - auto-adjusting batch sizes")
    except ImportError:
        pass

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

# ================================
# CLIMATE VARIABLES SETTINGS
# ================================

# Default paths for climate variables
CLIMATE_VARIABLES_DIR = OUTPUTS_DIR / "climate_variables"
EVENT_STUDIES_DIR = OUTPUTS_DIR / "event_studies"
DATA_SUMMARY_DIR = OUTPUTS_DIR / "data_summary"

def get_climate_variables_folder() -> Path:
    """Get the climate variables output folder."""
    return CLIMATE_VARIABLES_DIR

def get_event_studies_folder() -> Path:
    """Get the event studies output folder."""
    return EVENT_STUDIES_DIR

def get_data_summary_folder() -> Path:
    """Get the data summary output folder."""
    return DATA_SUMMARY_DIR

# ================================
# DATA QUALITY SETTINGS
# ================================

# Minimum requirements for transcript quality
MIN_WORDS_PER_TRANSCRIPT = 100
MIN_SENTENCES_PER_TRANSCRIPT = 10
MIN_PARAGRAPHS_PER_TRANSCRIPT = 5

# Speaker validation
COMMON_SPEAKER_ROLES = [
    'CEO', 'CFO', 'COO', 'CTO', 'President', 'Chairman', 'Director',
    'Analyst', 'Operator', 'Vice President', 'Manager', 'Executive'
]

# Date validation
MIN_YEAR = 2005
MAX_YEAR = 2030

def validate_transcript_quality(transcript_stats: dict) -> bool:
    """Validate if a transcript meets minimum quality requirements."""
    return (
        transcript_stats.get('total_word_count', 0) >= MIN_WORDS_PER_TRANSCRIPT and
        transcript_stats.get('total_sentence_count', 0) >= MIN_SENTENCES_PER_TRANSCRIPT and
        transcript_stats.get('total_paragraph_count', 0) >= MIN_PARAGRAPHS_PER_TRANSCRIPT and
        MIN_YEAR <= transcript_stats.get('year', 0) <= MAX_YEAR
    )