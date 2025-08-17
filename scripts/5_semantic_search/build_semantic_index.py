#!/usr/bin/env python3
"""
Enhanced Semantic Climate Index Builder for ECC Data Generation Pipeline

This script builds comprehensive semantic indexes from enhanced climate snippets,
supporting both aggregate time-series analyses and firm-level studies.

Author: Marleen de Jonge
Date: 2025
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import torch
except ImportError as e:
    raise ImportError(f"Required packages not installed: {e}")

from src.config import SUPPORTED_INDICES, LOGS_DIR


@dataclass
class ClimateSnippet:
    """Enhanced climate snippet with metadata for indexing."""
    text: str
    company_name: str
    ticker: str
    year: int
    quarter: str
    date: str
    speaker: str
    profession: str
    section: str  # 'management' or 'qa'
    
    # Enhanced metadata from sentence ratio calculations
    sentence_count: Optional[int] = None
    climate_sentence_count: Optional[int] = None
    total_sentences_in_call: Optional[int] = None
    climate_sentence_ratio: Optional[float] = None
    
    # Stock index information
    stock_index: str = 'UNKNOWN'
    source_file: str = ''
    
    # Semantic analysis results (populated during search)
    similarity_score: float = 0.0
    matched_query: str = ''
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert snippet to dictionary for JSON serialization."""
        return {
            'text': self.text,
            'company_name': self.company_name,
            'ticker': self.ticker,
            'year': self.year,
            'quarter': self.quarter,
            'date': self.date,
            'speaker': self.speaker,
            'profession': self.profession,
            'section': self.section,
            'sentence_count': self.sentence_count,
            'climate_sentence_count': self.climate_sentence_count,
            'total_sentences_in_call': self.total_sentences_in_call,
            'climate_sentence_ratio': self.climate_sentence_ratio,
            'stock_index': self.stock_index,
            'source_file': self.source_file,
            'similarity_score': self.similarity_score,
            'matched_query': self.matched_query
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClimateSnippet':
        """Create snippet from dictionary."""
        return cls(
            text=data.get('text', ''),
            company_name=data.get('company_name', ''),
            ticker=data.get('ticker', ''),
            year=data.get('year', 0),
            quarter=data.get('quarter', ''),
            date=data.get('date', ''),
            speaker=data.get('speaker', ''),
            profession=data.get('profession', ''),
            section=data.get('section', ''),
            sentence_count=data.get('sentence_count'),
            climate_sentence_count=data.get('climate_sentence_count'),
            total_sentences_in_call=data.get('total_sentences_in_call'),
            climate_sentence_ratio=data.get('climate_sentence_ratio'),
            stock_index=data.get('stock_index', 'UNKNOWN'),
            source_file=data.get('source_file', ''),
            similarity_score=data.get('similarity_score', 0.0),
            matched_query=data.get('matched_query', '')
        )


class EnhancedSemanticIndexBuilder:
    """
    Build and manage semantic indexes for climate snippets with support for
    both aggregate and firm-level analyses.
    """
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None
        self.device = self._get_device()
        
        # Climate topic queries for semantic search
        self.climate_topics = {
            'opportunities': [
                'renewable energy investments clean technology opportunities',
                'green innovation sustainable business models growth',
                'energy transition investment opportunities market',
                'clean energy competitive advantage sustainability',
                'ESG opportunities sustainable development goals'
            ],
            'physical_risks': [
                'extreme weather climate physical risk operational',
                'supply chain disruption weather events disasters',
                'flooding drought wildfire hurricane climate impacts',
                'climate adaptation resilience infrastructure vulnerability',
                'weather operational disruption facility damage'
            ],
            'transition_risks': [
                'stranded assets carbon intensive business risk',
                'technology disruption energy transition competitive',
                'carbon tax transition costs regulatory burden',
                'fossil fuel asset impairment devaluation risk',
                'business model transition technological change'
            ],
            'regulation': [
                'climate policy regulatory compliance requirements',
                'carbon pricing emissions trading regulations',
                'environmental standards regulatory changes',
                'climate disclosure requirements mandatory reporting',
                'Paris Agreement regulatory framework compliance'
            ],
            'disclosure': [
                'climate risk disclosure ESG reporting standards',
                'TCFD climate scenario analysis reporting',
                'sustainability reporting framework disclosure',
                'carbon footprint disclosure transparency',
                'climate governance reporting requirements'
            ],
            'green_finance': [
                'green bonds sustainability-linked financing',
                'sustainable finance ESG investment criteria',
                'climate finance transition financing mechanisms',
                'green investment sustainable debt instruments',
                'ESG rating climate performance assessment'
            ]
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _get_device(self) -> str:
        """Get optimal device for processing."""
        if torch.backends.mps.is_available():
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    
    def load_model(self):
        """Load sentence transformer model."""
        if self.model is None:
            self.logger.info(f"Loading model: {self.model_name} on {self.device}")
            self.model = SentenceTransformer(self.model_name)
            self.model = self.model.to(self.device)
            self.logger.info("✅ Model loaded successfully")
    
    def load_enhanced_climate_snippets(self, data_path: Path, stock_index: str) -> List[ClimateSnippet]:
        """Load enhanced climate snippets from JSON files."""
        self.logger.info(f"Loading enhanced climate snippets from: {data_path}")
        
        snippets = []
        json_files = list(data_path.glob("enhanced_climate_segments_*.json"))
        
        if not json_files:
            self.logger.warning(f"No enhanced climate segment files found in {data_path}")
            return snippets
        
        for json_file in tqdm(json_files, desc=f"Loading {stock_index}"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for transcript in data:
                    company_name = transcript.get('company_name', '')
                    ticker = transcript.get('ticker', '')
                    year = transcript.get('year', 0)
                    quarter = transcript.get('quarter', '')
                    date = transcript.get('date', '')
                    
                    # Enhanced metadata
                    climate_sentence_count = transcript.get('climate_sentence_count')
                    total_sentences = transcript.get('total_sentences_in_call')
                    climate_ratio = transcript.get('climate_sentence_ratio')
                    
                    for text_data in transcript.get('texts', []):
                        snippet = ClimateSnippet(
                            text=text_data.get('text', ''),
                            company_name=company_name,
                            ticker=ticker,
                            year=int(year) if year else 0,
                            quarter=quarter,
                            date=date,
                            speaker=text_data.get('speaker', ''),
                            profession=text_data.get('profession', ''),
                            section='climate',  # All from climate segments
                            sentence_count=text_data.get('sentence_count'),
                            climate_sentence_count=climate_sentence_count,
                            total_sentences_in_call=total_sentences,
                            climate_sentence_ratio=climate_ratio,
                            stock_index=stock_index,
                            source_file=json_file.name
                        )
                        snippets.append(snippet)
                        
            except Exception as e:
                self.logger.error(f"Error loading {json_file}: {e}")
                continue
        
        self.logger.info(f"✅ Loaded {len(snippets)} climate snippets from {stock_index}")
        return snippets
    
    def build_semantic_index(self, snippets: List[ClimateSnippet], 
                           output_dir: Path, index_name: str,
                           batch_size: int = 64) -> Dict[str, Any]:
        """Build FAISS semantic index from climate snippets."""
        if not snippets:
            raise ValueError("No snippets provided for indexing")
        
        self.load_model()
        
        # Extract texts for embedding
        texts = [snippet.text for snippet in snippets]
        
        self.logger.info(f"Creating embeddings for {len(texts)} snippets...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
            device=self.device
        )
        
        # Build FAISS index
        self.logger.info("Building FAISS index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        index.add(embeddings.astype(np.float32))
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save index components
        index_path = output_dir / f"{index_name}_index.faiss"
        embeddings_path = output_dir / f"{index_name}_embeddings.npy"
        snippets_path = output_dir / f"{index_name}_snippets.json"
        metadata_path = output_dir / f"{index_name}_metadata.json"
        
        # Save files
        faiss.write_index(index, str(index_path))
        np.save(str(embeddings_path), embeddings)
        
        # Save snippets
        with open(snippets_path, 'w', encoding='utf-8') as f:
            json.dump([snippet.to_dict() for snippet in snippets], f, indent=2, ensure_ascii=False)
        
        # Save metadata
        metadata = {
            'index_name': index_name,
            'model_name': self.model_name,
            'num_snippets': len(snippets),
            'embedding_dimension': dimension,
            'creation_date': datetime.now().isoformat(),
            'stock_indices': list(set(s.stock_index for s in snippets)),
            'date_range': {
                'start': min(s.date for s in snippets if s.date),
                'end': max(s.date for s in snippets if s.date)
            },
            'company_count': len(set(s.ticker for s in snippets)),
            'files': {
                'index': str(index_path),
                'embeddings': str(embeddings_path),
                'snippets': str(snippets_path),
                'metadata': str(metadata_path)
            }
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"✅ Semantic index built and saved to {output_dir}")
        self.logger.info(f"📊 Index stats: {len(snippets)} snippets, {dimension}D embeddings")
        
        return metadata
    
    def build_indexes_by_market(self, base_input_path: Path, base_output_path: Path) -> Dict[str, Any]:
        """Build separate indexes for each stock market."""
        results = {}
        
        for stock_index in SUPPORTED_INDICES:
            input_path = base_input_path / stock_index
            if not input_path.exists():
                self.logger.warning(f"Input path not found: {input_path}")
                continue
            
            self.logger.info(f"\n🚀 Building index for {stock_index}")
            
            try:
                # Load snippets
                snippets = self.load_enhanced_climate_snippets(input_path, stock_index)
                
                if not snippets:
                    self.logger.warning(f"No snippets loaded for {stock_index}")
                    continue
                
                # Build index
                output_dir = base_output_path / stock_index
                index_name = f"climate_semantic_{stock_index.lower()}"
                
                metadata = self.build_semantic_index(
                    snippets, output_dir, index_name
                )
                
                results[stock_index] = metadata
                
            except Exception as e:
                self.logger.error(f"Failed to build index for {stock_index}: {e}")
                results[stock_index] = {'error': str(e)}
        
        return results
    
    def build_combined_index(self, base_input_path: Path, base_output_path: Path) -> Dict[str, Any]:
        """Build combined index from all stock markets."""
        self.logger.info("🌍 Building combined semantic index...")
        
        all_snippets = []
        
        # Load from all markets
        for stock_index in SUPPORTED_INDICES:
            input_path = base_input_path / stock_index
            if input_path.exists():
                snippets = self.load_enhanced_climate_snippets(input_path, stock_index)
                all_snippets.extend(snippets)
        
        if not all_snippets:
            raise ValueError("No snippets loaded from any market")
        
        # Build combined index
        output_dir = base_output_path / "combined"
        index_name = "climate_semantic_combined"
        
        metadata = self.build_semantic_index(
            all_snippets, output_dir, index_name
        )
        
        self.logger.info(f"✅ Combined index built with {len(all_snippets)} total snippets")
        return metadata


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Build semantic indexes for climate snippets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Build indexes for all markets
    python build_semantic_index.py --all
    
    # Build index for specific market
    python build_semantic_index.py --market SP500
    
    # Build combined index only
    python build_semantic_index.py --combined-only
    
    # Use custom model
    python build_semantic_index.py --all --model sentence-transformers/all-mpnet-base-v2
        """
    )
    
    parser.add_argument(
        '--market',
        choices=SUPPORTED_INDICES,
        help='Build index for specific market'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Build indexes for all markets'
    )
    
    parser.add_argument(
        '--combined-only',
        action='store_true',
        help='Build only the combined index'
    )
    
    parser.add_argument(
        '--model',
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='Sentence transformer model to use'
    )
    
    parser.add_argument(
        '--input-path',
        type=Path,
        default=Path("data/enhanced_climate_snippets"),
        help='Input path to enhanced climate snippets'
    )
    
    parser.add_argument(
        '--output-path',
        type=Path,
        default=Path("data/semantic_indexes"),
        help='Output path for semantic indexes'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for embedding creation'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.market, args.all, args.combined_only]):
        parser.error("Must specify --market, --all, or --combined-only")
    
    print("🔍 Enhanced Semantic Climate Index Builder")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Input: {args.input_path}")
    print(f"Output: {args.output_path}")
    print(f"Batch size: {args.batch_size}")
    print()
    
    # Initialize builder
    builder = EnhancedSemanticIndexBuilder(model_name=args.model)
    results = {}
    
    try:
        if args.combined_only:
            # Build only combined index
            results['combined'] = builder.build_combined_index(
                args.input_path, args.output_path
            )
            
        elif args.all:
            # Build all market indexes
            results.update(builder.build_indexes_by_market(
                args.input_path, args.output_path
            ))
            
            # Also build combined index
            results['combined'] = builder.build_combined_index(
                args.input_path, args.output_path
            )
            
        elif args.market:
            # Build specific market index
            input_path = args.input_path / args.market
            if not input_path.exists():
                raise FileNotFoundError(f"Input path not found: {input_path}")
            
            snippets = builder.load_enhanced_climate_snippets(input_path, args.market)
            output_dir = args.output_path / args.market
            index_name = f"climate_semantic_{args.market.lower()}"
            
            results[args.market] = builder.build_semantic_index(
                snippets, output_dir, index_name, batch_size=args.batch_size
            )
        
        # Save build summary
        summary_path = args.output_path / "build_summary.json"
        args.output_path.mkdir(parents=True, exist_ok=True)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print("\n✅ Semantic index building completed!")
        print(f"📁 Results saved to: {args.output_path}")
        print(f"📊 Build summary: {summary_path}")
        
        # Print summary
        print("\n📈 Summary:")
        for name, result in results.items():
            if 'error' in result:
                print(f"  ❌ {name}: {result['error']}")
            else:
                print(f"  ✅ {name}: {result['num_snippets']:,} snippets indexed")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()