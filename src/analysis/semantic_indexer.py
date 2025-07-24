"""
Semantic indexing utilities for building searchable indexes from transcript paragraphs.

This module builds FAISS indexes from individual transcript paragraphs for efficient 
semantic search, with options for batch processing and separate indices per stock index.

Author: Marleen de Jonge
Date: 2025
"""

import json
import os
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError as e:
    raise ImportError(f"Required packages not installed: {e}. Run: pip install sentence-transformers faiss-cpu")

from .paragraph_extractor import extract_paragraphs_from_folder, estimate_paragraph_counts

logger = logging.getLogger(__name__)


class SemanticIndexBuilder:
    """Build and manage semantic indexes for transcript paragraphs."""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', 
                 batch_size: int = 64):
        """
        Initialize the semantic index builder.
        
        Args:
            model_name: Name of the sentence transformer model to use
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None
        
        logger.info(f"Initializing SemanticIndexBuilder with model: {model_name}")
    
    def _load_model(self):
        """Lazy load the sentence transformer model."""
        if self.model is None:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("✅ Model loaded successfully")
    
    def build_index_from_folder(self, structured_json_folder: Path, 
                               output_folder: Path,
                               stock_index: str,
                               max_files: int = None,
                               index_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Build semantic index from structured JSON folder.
        
        Args:
            structured_json_folder: Input folder with structured JSONs
            output_folder: Output folder for index files
            stock_index: Stock index name (SP500, STOXX600, etc.)
            max_files: Maximum number of JSON files to process (None for all)
            index_name: Custom index name (defaults to stock_index)
            
        Returns:
            Dictionary with build information
        """
        if index_name is None:
            index_name = f"{stock_index.lower()}_paragraphs"
        
        logger.info(f"Building semantic index for {stock_index}")
        logger.info(f"Input folder: {structured_json_folder}")
        logger.info(f"Output folder: {output_folder}")
        logger.info(f"Max files: {max_files if max_files else 'All files'}")
        
        # Get estimates first
        estimates = estimate_paragraph_counts(structured_json_folder, max_files)
        logger.info(f"📊 Paragraph estimates: {estimates}")
        
        # Extract paragraphs
        texts, metadata = extract_paragraphs_from_folder(
            structured_json_folder, max_files, stock_index
        )
        
        if not texts:
            raise ValueError("No paragraphs extracted from the input folder")
        
        # Build index
        index_info = self.build_index(texts, output_folder, index_name)
        
        # Save metadata
        metadata_path = output_folder / f"{index_name}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Save build configuration
        build_info = {
            **index_info,
            'stock_index': stock_index,
            'input_folder': str(structured_json_folder),
            'metadata_path': str(metadata_path),
            'max_files_processed': max_files,
            'files_processed': estimates.get('files_to_process', 0),
            'build_timestamp': pd.Timestamp.now().isoformat(),
            'paragraph_estimates': estimates
        }
        
        config_path = output_folder / f"{index_name}_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(build_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Index built for {stock_index}")
        logger.info(f"📊 Total paragraphs indexed: {len(texts)}")
        logger.info(f"📁 Files saved to: {output_folder}")
        
        return build_info
    
    def build_index(self, texts: List[str], output_folder: Path, 
                   index_name: str) -> Dict[str, Any]:
        """
        Build FAISS index from texts.
        
        Args:
            texts: List of text paragraphs to index
            output_folder: Folder to save index files
            index_name: Base name for index files
            
        Returns:
            Dictionary with index information
        """
        self._load_model()
        
        logger.info(f"Encoding {len(texts)} paragraphs...")
        embeddings = self.model.encode(
            texts, 
            batch_size=self.batch_size, 
            show_progress_bar=True, 
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        logger.info(f"Building FAISS index with {embeddings.shape[1]} dimensions...")
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        index.add(embeddings.astype(np.float32))
        
        # Create output directory
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Save files
        embeddings_path = output_folder / f"{index_name}_embeddings.npy"
        index_path = output_folder / f"{index_name}_index.faiss"
        
        np.save(str(embeddings_path), embeddings)
        faiss.write_index(index, str(index_path))
        
        index_info = {
            'index_name': index_name,
            'num_paragraphs': len(texts),
            'embedding_dimension': dimension,
            'model_name': self.model_name,
            'embeddings_path': str(embeddings_path),
            'index_path': str(index_path)
        }
        
        logger.info(f"✅ Index built and saved: {len(texts)} paragraphs indexed")
        return index_info


class SemanticSearcher:
    """Search the built semantic indexes."""
    
    def __init__(self, index_folder: Path, index_name: str):
        """
        Initialize the semantic searcher.
        
        Args:
            index_folder: Folder containing index files
            index_name: Name of the index to load
        """
        self.index_folder = index_folder
        self.index_name = index_name
        self.model = None
        self.index = None
        self.metadata = None
        self.config = None
        
        self._load_index()
    
    def _load_index(self):
        """Load the FAISS index and metadata."""
        index_path = self.index_folder / f"{self.index_name}_index.faiss"
        metadata_path = self.index_folder / f"{self.index_name}_metadata.json"
        config_path = self.index_folder / f"{self.index_name}_config.json"
        
        if not all(p.exists() for p in [index_path, metadata_path, config_path]):
            missing_files = [str(p) for p in [index_path, metadata_path, config_path] if not p.exists()]
            raise FileNotFoundError(f"Index files not found: {missing_files}")
        
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # Load model
        model_name = self.config['model_name']
        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Load index
        logger.info(f"Loading FAISS index: {index_path}")
        self.index = faiss.read_index(str(index_path))
        
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        logger.info(f"✅ Index loaded: {len(self.metadata)} paragraphs available")
        logger.info(f"📊 Index info: {self.config['stock_index']}, {self.config['files_processed']} files")
    
    def search(self, query: str, top_k: int = 10, 
               filter_section: str = None, 
               filter_company: str = None) -> List[Dict[str, Any]]:
        """
        Search for similar paragraphs.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_section: Filter by section ('management' or 'qa')
            filter_company: Filter by company name
            
        Returns:
            List of result dictionaries with scores and metadata
        """
        # Encode query
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        # Search (get more results for filtering)
        search_k = min(top_k * 3, len(self.metadata))  # Get extra for filtering
        scores, indices = self.index.search(query_embedding.astype(np.float32), search_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):
                metadata = self.metadata[idx]
                
                # Apply filters
                if filter_section and metadata.get('section') != filter_section:
                    continue
                if filter_company and filter_company.lower() not in metadata.get('company_name', '').lower():
                    continue
                
                result = {
                    'score': float(score),
                    'rank': len(results) + 1,
                    **metadata
                }
                results.append(result)
                
                if len(results) >= top_k:
                    break
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the index."""
        if not self.metadata:
            return {}
        
        # Count by section
        sections = {}
        companies = {}
        years = {}
        
        for item in self.metadata:
            section = item.get('section', 'unknown')
            sections[section] = sections.get(section, 0) + 1
            
            company = item.get('company_name', 'unknown')
            companies[company] = companies.get(company, 0) + 1
            
            year = item.get('year', 'unknown')
            years[year] = years.get(year, 0) + 1
        
        return {
            'total_paragraphs': len(self.metadata),
            'sections': sections,
            'unique_companies': len(companies),
            'top_companies': dict(sorted(companies.items(), key=lambda x: x[1], reverse=True)[:10]),
            'years': dict(sorted(years.items())),
            'stock_index': self.config.get('stock_index', 'unknown'),
            'model_name': self.config.get('model_name', 'unknown')
        }


def build_separate_indexes(base_input_folder: Path, 
                          base_output_folder: Path,
                          stock_indices: List[str] = ['SP500', 'STOXX600'],
                          max_files_per_index: int = None) -> Dict[str, Any]:
    """
    Build separate indexes for different stock indices.
    
    Args:
        base_input_folder: Base folder containing stock index subfolders
        base_output_folder: Base output folder for indexes
        stock_indices: List of stock indices to process
        max_files_per_index: Maximum files to process per index
        
    Returns:
        Dictionary with build results for each index
    """
    results = {}
    builder = SemanticIndexBuilder()
    
    for stock_index in stock_indices:
        input_folder = base_input_folder / stock_index
        output_folder = base_output_folder / stock_index
        
        if not input_folder.exists():
            logger.warning(f"Input folder not found: {input_folder}")
            results[stock_index] = {'error': f'Input folder not found: {input_folder}'}
            continue
        
        logger.info(f"\n🚀 Building index for {stock_index}")
        
        try:
            build_info = builder.build_index_from_folder(
                structured_json_folder=input_folder,
                output_folder=output_folder,
                stock_index=stock_index,
                max_files=max_files_per_index
            )
            results[stock_index] = build_info
            
        except Exception as e:
            logger.error(f"❌ Failed to build index for {stock_index}: {e}")
            results[stock_index] = {'error': str(e)}
    
    # Save combined results
    combined_results_path = base_output_folder / "build_results.json"
    base_output_folder.mkdir(parents=True, exist_ok=True)
    with open(combined_results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n✅ All indexes built. Results saved to: {combined_results_path}")
    return results


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Adjust these paths to match your setup
    base_input_folder = Path("outputs/processed_transcripts")
    base_output_folder = Path("data/semantic_indexes")
    
    if base_input_folder.exists():
        # Build separate indexes for SP500 and STOXX600, processing max 5 files each
        results = build_separate_indexes(
            base_input_folder, 
            base_output_folder, 
            stock_indices=['SP500', 'STOXX600'],
            max_files_per_index=5
        )
        
        print("Build results:")
        for stock_index, result in results.items():
            if 'error' in result:
                print(f"  {stock_index}: ERROR - {result['error']}")
            else:
                print(f"  {stock_index}: {result['num_paragraphs']} paragraphs indexed")
    else:
        print(f"Base input folder not found: {base_input_folder}")