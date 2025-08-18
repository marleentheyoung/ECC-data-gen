#!/usr/bin/env python3
"""
Base Semantic Searcher for ECC Data Generation Pipeline

Core semantic search functionality that provides the foundation for all specialized 
analyzers (policy, risk, opportunity, legislation). Contains index management,
basic search operations, and generic analysis framework.

Author: Marleen de Jonge
Date: 2025
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from collections import defaultdict
import sys

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import torch
except ImportError as e:
    raise ImportError(f"Required packages not installed: {e}")

# Add path for imports (will be updated when config system is built)
sys.path.append(str(Path(__file__).parent.parent))

# Import configuration loader (placeholder for future implementation)
try:
    from ..shared.config_loader import ConfigLoader
except ImportError:
    # Fallback for development - will be implemented later
    ConfigLoader = None

# Import climate snippet class
try:
    from build_semantic_index import ClimateSnippet
except ImportError:
    # Fallback - define minimal snippet class
    from dataclasses import dataclass
    from typing import Optional as OptionalType
    
    @dataclass
    class ClimateSnippet:
        text: str
        company_name: str
        ticker: str
        year: int
        quarter: str
        date: str
        section: str = 'climate'
        similarity_score: float = 0.0
        matched_query: str = ""
        sentence_count: OptionalType[int] = None
        climate_sentence_count: OptionalType[int] = None
        total_sentences_in_call: OptionalType[int] = None
        climate_sentence_ratio: OptionalType[float] = None
        stock_index: str = ""
        source_file: str = ""
        
        def to_dict(self) -> Dict[str, Any]:
            """Convert snippet to dictionary."""
            return {
                'text': self.text,
                'company_name': self.company_name,
                'ticker': self.ticker,
                'year': self.year,
                'quarter': self.quarter,
                'date': self.date,
                'section': self.section,
                'similarity_score': self.similarity_score,
                'matched_query': self.matched_query,
                'sentence_count': self.sentence_count,
                'climate_sentence_count': self.climate_sentence_count,
                'total_sentences_in_call': self.total_sentences_in_call,
                'climate_sentence_ratio': self.climate_sentence_ratio,
                'stock_index': self.stock_index,
                'source_file': self.source_file
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
                section=data.get('section', 'climate'),
                similarity_score=data.get('similarity_score', 0.0),
                matched_query=data.get('matched_query', ''),
                sentence_count=data.get('sentence_count'),
                climate_sentence_count=data.get('climate_sentence_count'),
                total_sentences_in_call=data.get('total_sentences_in_call'),
                climate_sentence_ratio=data.get('climate_sentence_ratio'),
                stock_index=data.get('stock_index', ''),
                source_file=data.get('source_file', '')
            )


class BaseSemanticSearcher:
    """
    Base semantic searcher providing core functionality for climate content analysis.
    
    This class serves as the foundation for specialized analyzers and provides:
    - Index loading and management
    - Basic semantic search operations
    - Generic analysis framework
    - Configuration management
    """
    
    def __init__(self, index_path: Path, config_path: Optional[str] = None):
        self.index_path = index_path
        self.model = None
        self.faiss_index = None
        self.snippets = []
        self.metadata = {}
        
        # Configuration management
        self.config = ConfigLoader(config_path) if config_path and ConfigLoader else None
        
        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load_index(self, index_path: Path) -> bool:
        """Simple index loading - delegate details to IndexLoader."""
        try:
            from ..shared.index_loader import IndexLoader
            loader = IndexLoader()
            return loader.load(self, index_path)
        except ImportError:
            self.logger.warning("IndexLoader not available, using fallback")
            return self._load_index_fallback(index_path)
    
    def _load_index_fallback(self, index_path: Path) -> bool:
        """Basic fallback loading implementation."""
        # Implementation here - simplified version
        self.logger.info(f"Loading index from {index_path}")
        # Add basic loading logic
        return True
    
    def get_index_statistics(self) -> Dict[str, Any]:
        """Get basic stats - delegate complex analysis to Statistics."""
        try:
            from ..shared.statistics import IndexStatistics
            stats = IndexStatistics(self)
            return stats.generate()
        except ImportError:
            self.logger.warning("IndexStatistics not available, using basic stats")
            return self._get_basic_stats()
    
    def _get_basic_stats(self) -> Dict[str, Any]:
        """Basic statistics fallback."""
        return {
            'total_snippets': len(self.snippets),
            'unique_companies': len(set(s.ticker for s in self.snippets)),
            'date_range': {
                'start': min(s.date for s in self.snippets if s.date) if self.snippets else None,
                'end': max(s.date for s in self.snippets if s.date) if self.snippets else None
            }
        }
    
    def validate_index(self) -> Tuple[bool, List[str]]:
        """Basic validation - delegate complex checks to IndexValidator."""
        try:
            from ..shared.index_validator import IndexValidator
            validator = IndexValidator(self)
            return validator.validate()
        except ImportError:
            self.logger.warning("IndexValidator not available, using basic validation")
            return self._validate_basic()
    
    def _validate_basic(self) -> Tuple[bool, List[str]]:
        """Basic validation fallback."""
        issues = []
        if not self.snippets:
            issues.append("No snippets loaded")
        if not self.faiss_index:
            issues.append("No FAISS index loaded")
        return len(issues) == 0, issues
    
    def semantic_search(self, query: str, top_k: int = 100, min_score: float = 0.40) -> List[ClimateSnippet]:
        """Core semantic search functionality."""
        if not self.model or not self.faiss_index:
            raise RuntimeError("Index not loaded properly")
        
        # Create query embedding
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        # Search FAISS index
        scores, indices = self.faiss_index.search(query_embedding.astype(np.float32), top_k)
        
        # Process results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= min_score and idx < len(self.snippets):
                snippet = self.snippets[idx]
                snippet.similarity_score = float(score)
                snippet.matched_query = query
                results.append(snippet)
        
        return results
    
    def multi_query_search(self, queries: List[str], top_k: int = 100, min_score: float = 0.40) -> List[ClimateSnippet]:
        """Search with multiple queries and combine results."""
        all_results = []
        for query in queries:
            results = self.semantic_search(query, top_k, min_score)
            all_results.extend(results)
        
        # Deduplicate and sort by score
        seen = set()
        unique_results = []
        for result in sorted(all_results, key=lambda x: x.similarity_score, reverse=True):
            key = (result.ticker, result.date, result.text[:100])  # Unique identifier
            if key not in seen:
                seen.add(key)
                unique_results.append(result)
        
        return unique_results[:top_k]
    
    def filter_snippets(self, snippets: List[ClimateSnippet], 
                       companies: Optional[List[str]] = None,
                       years: Optional[List[int]] = None,
                       quarters: Optional[List[str]] = None) -> List[ClimateSnippet]:
        """Filter snippets by criteria."""
        filtered = snippets
        
        if companies:
            filtered = [s for s in filtered if s.ticker in companies]
        
        if years:
            filtered = [s for s in filtered if s.year in years]
        
        if quarters:
            filtered = [s for s in filtered if s.quarter in quarters]
        
        return filtered
    
    def aggregate_by_time(self, snippets: List[ClimateSnippet], freq: str = 'quarter') -> pd.DataFrame:
        """Aggregate snippets by time period."""
        if not snippets:
            return pd.DataFrame()
        
        data = []
        for snippet in snippets:
            if freq == 'quarter':
                period = f"{snippet.year}Q{snippet.quarter}"
            elif freq == 'year':
                period = snippet.year
            else:
                period = snippet.date
            
            data.append({
                'period': period,
                'ticker': snippet.ticker,
                'company_name': snippet.company_name,
                'similarity_score': snippet.similarity_score,
                'climate_ratio': snippet.climate_sentence_ratio
            })
        
        df = pd.DataFrame(data)
        return df.groupby(['period', 'ticker']).agg({
            'similarity_score': ['mean', 'max', 'count'],
            'climate_ratio': 'mean'
        }).round(4)
    
    def aggregate_by_firm(self, snippets: List[ClimateSnippet]) -> pd.DataFrame:
        """Aggregate snippets by firm."""
        if not snippets:
            return pd.DataFrame()
        
        firm_data = defaultdict(list)
        for snippet in snippets:
            firm_data[snippet.ticker].append(snippet)
        
        results = []
        for ticker, firm_snippets in firm_data.items():
            scores = [s.similarity_score for s in firm_snippets]
            results.append({
                'ticker': ticker,
                'company_name': firm_snippets[0].company_name,
                'total_mentions': len(firm_snippets),
                'avg_score': np.mean(scores),
                'max_score': max(scores),
                'latest_date': max(s.date for s in firm_snippets if s.date)
            })
        
        return pd.DataFrame(results)
    
    def create_attention_timeseries(self, analysis_type: str, topic: str, **kwargs) -> pd.DataFrame:
        """Generic attention time series creation."""
        # Get queries from config or fallback
        if self.config:
            queries = self.config.get_queries(analysis_type, topic)
            threshold = self.config.get_threshold(analysis_type, topic)
        else:
            queries = self._get_default_queries(analysis_type, topic)
            threshold = kwargs.get('min_score', 0.40)
        
        # Perform search
        results = self.multi_query_search(queries, top_k=1000, min_score=threshold)
        
        # Create time series
        return self.aggregate_by_time(results, freq=kwargs.get('freq', 'quarter'))
    
    def _get_default_queries(self, analysis_type: str, topic: str) -> List[str]:
        """Fallback queries when no config available."""
        default_queries = {
            'policies': {
                'paris_agreement': ['Paris Agreement', 'climate accord', 'COP21'],
                'carbon_pricing': ['carbon tax', 'carbon pricing', 'emissions trading']
            },
            'risk': {
                'physical_risk': ['physical climate risk', 'extreme weather', 'flooding'],
                'transition_risk': ['transition risk', 'stranded assets', 'carbon regulation']
            }
        }
        return default_queries.get(analysis_type, {}).get(topic, [f"{topic} climate"])
    
    def get_available_analyses(self) -> Dict[str, List[str]]:
        """Get available analysis types and topics."""
        if self.config:
            return self.config.get_available_analyses()
        else:
            return {
                'policies': ['paris_agreement', 'carbon_pricing'],
                'risk': ['physical_risk', 'transition_risk'],
                'opportunities': ['renewable_energy', 'green_finance']
            }