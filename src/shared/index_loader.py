#!/usr/bin/env python3
"""
Index Loader Utility for ECC Data Generation Pipeline

Handles the detailed loading of semantic indexes including FAISS indexes,
sentence transformer models, and climate snippets.

Author: Marleen de Jonge
Date: 2025
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, TYPE_CHECKING
import sys

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import torch
except ImportError as e:
    raise ImportError(f"Required packages not installed: {e}")

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

# Import climate snippet class
try:
    from data_pipeline.build_semantic_index import ClimateSnippet
except ImportError:
    # Use the fallback from base_searcher
    if TYPE_CHECKING:
        from analysis_pipeline.base_searcher import ClimateSnippet


class IndexLoader:
    """
    Utility class for loading semantic indexes and associated data.
    Handles all the detailed logic for loading FAISS indexes, models, and snippets.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _get_device(self) -> str:
        """Get optimal device for processing."""
        if torch.backends.mps.is_available():
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    
    def load(self, searcher, index_path: Path) -> bool:
        """
        Load semantic index and associated data into the searcher.
        
        Args:
            searcher: BaseSemanticSearcher instance to load data into
            index_path: Path to the semantic index directory
            
        Returns:
            bool: True if successfully loaded, False otherwise
            
        Raises:
            FileNotFoundError: If required index files are missing
            ValueError: If index format is invalid
            RuntimeError: If model loading fails
        """
        try:
            index_path = Path(index_path)
            self.logger.info(f"Loading semantic index from: {index_path}")
            
            # Validate index directory exists
            if not index_path.exists():
                raise FileNotFoundError(f"Index directory not found: {index_path}")
            
            if not index_path.is_dir():
                raise ValueError(f"Index path is not a directory: {index_path}")
            
            # Load and validate metadata
            metadata = self._load_metadata(index_path)
            
            # Load sentence transformer model
            model = self._load_model(metadata, searcher.model_name)
            
            # Load FAISS index
            faiss_index = self._load_faiss_index(metadata)
            
            # Load snippets
            snippets = self._load_snippets(metadata)
            
            # Validate consistency
            self._validate_consistency(metadata, faiss_index, snippets)
            
            # Update searcher with loaded components
            searcher.index_path = index_path
            searcher.metadata = metadata
            searcher.model = model
            searcher.model_name = model.get_model_name() if hasattr(model, 'get_model_name') else metadata['model_name']
            searcher.faiss_index = faiss_index
            searcher.snippets = snippets
            searcher.device = self._get_device()
            searcher._index_loaded = True
            
            # Log successful loading
            self.logger.info("✅ Index loaded successfully")
            self.logger.info(f"📊 Index statistics:")
            self.logger.info(f"   - Snippets: {len(snippets):,}")
            self.logger.info(f"   - Model: {searcher.model_name}")
            self.logger.info(f"   - Embedding dimension: {metadata['embedding_dimension']}")
            self.logger.info(f"   - Date range: {metadata.get('date_range', 'Unknown')}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load index: {e}")
            searcher._index_loaded = False
            raise
    
    def _load_metadata(self, index_path: Path) -> Dict[str, Any]:
        """Load and validate metadata file."""
        # Find metadata file
        metadata_files = list(index_path.glob("*_metadata.json"))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata file found in {index_path}")
        
        metadata_file = metadata_files[0]
        self.logger.debug(f"Loading metadata from: {metadata_file}")
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Validate metadata structure
        required_fields = ['files', 'model_name', 'num_snippets', 'embedding_dimension']
        for field in required_fields:
            if field not in metadata:
                raise ValueError(f"Missing required metadata field: {field}")
        
        # Validate files references
        files = metadata['files']
        required_files = ['index', 'snippets']
        for file_type in required_files:
            if file_type not in files:
                raise ValueError(f"Missing file reference in metadata: {file_type}")
        
        return metadata
    
    def _load_model(self, metadata: Dict[str, Any], override_model_name: str = None) -> SentenceTransformer:
        """Load sentence transformer model."""
        model_name = override_model_name or metadata['model_name']
        self.logger.info(f"Loading sentence transformer model: {model_name}")
        
        try:
            model = SentenceTransformer(model_name)
            device = self._get_device()
            model = model.to(device)
            self.logger.info(f"Model loaded successfully on device: {device}")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load sentence transformer model: {e}")
    
    def _load_faiss_index(self, metadata: Dict[str, Any]) -> faiss.Index:
        """Load FAISS index."""
        index_file = Path(metadata['files']['index'])
        
        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index file not found: {index_file}")
        
        self.logger.info(f"Loading FAISS index from: {index_file}")
        try:
            faiss_index = faiss.read_index(str(index_file))
            self.logger.info(f"FAISS index loaded: {faiss_index.ntotal} vectors")
            return faiss_index
        except Exception as e:
            raise RuntimeError(f"Failed to load FAISS index: {e}")
    
    def _load_snippets(self, metadata: Dict[str, Any]) -> List[ClimateSnippet]:
        """Load climate snippets."""
        snippets_file = Path(metadata['files']['snippets'])
        
        if not snippets_file.exists():
            raise FileNotFoundError(f"Snippets file not found: {snippets_file}")
        
        self.logger.info(f"Loading snippets from: {snippets_file}")
        try:
            with open(snippets_file, 'r', encoding='utf-8') as f:
                snippet_data = json.load(f)
            
            snippets = [ClimateSnippet.from_dict(data) for data in snippet_data]
            self.logger.info(f"Loaded {len(snippets)} snippets")
            return snippets
        except Exception as e:
            raise RuntimeError(f"Failed to load snippets: {e}")
    
    def _validate_consistency(self, metadata: Dict[str, Any], faiss_index, snippets: List[ClimateSnippet]):
        """Validate consistency between loaded components."""
        # Check snippet count consistency
        if len(snippets) != faiss_index.ntotal:
            self.logger.warning(
                f"Snippet count ({len(snippets)}) doesn't match "
                f"FAISS index vectors ({faiss_index.ntotal})"
            )
        
        if len(snippets) != metadata['num_snippets']:
            self.logger.warning(
                f"Snippet count ({len(snippets)}) doesn't match "
                f"metadata count ({metadata['num_snippets']})"
            )
        
        # Check embedding dimensions
        expected_dim = metadata['embedding_dimension']
        if faiss_index.d != expected_dim:
            raise ValueError(
                f"FAISS dimension ({faiss_index.d}) doesn't match "
                f"metadata dimension ({expected_dim})"
            )