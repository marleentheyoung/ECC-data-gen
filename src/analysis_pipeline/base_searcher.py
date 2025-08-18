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
sys.path.append(str(Path(__file__).parent))

# Import configuration loader (placeholder for future implementation)
try:
    from shared.config_loader import ConfigLoader
except ImportError:
    # Fallback for development - will be implemented later
    ConfigLoader = None

# Import climate snippet class
try:
    from data_pipeline.build_semantic_index import ClimateSnippet
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
    
    def __init__(self, index_path: Optional[Path] = None, config_path: Optional[str] = None):
        self.index_path = index_path
        self.model = None
        self.faiss_index = None
        self.snippets = []
        self.metadata = {}
        self.model_name = 'sentence-transformers/all-mpnet-base-v2'
        
        # Configuration management
        self.config = ConfigLoader(config_path) if config_path and ConfigLoader else None
        
        # Device management
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load index if path provided
        if index_path:
            self.load_index(index_path)
    
    def load_index(self, index_path: Path) -> bool:
        """
        Load semantic index from disk.
        
        Args:
            index_path: Path to index directory containing:
                - semantic_index.faiss
                - embeddings.npy  
                - snippets.json
                - index_metadata.json
                
        Returns:
            True if successful
        """
        index_path = Path(index_path)
        if not index_path.exists():
            raise FileNotFoundError(f"Index path does not exist: {index_path}")
        
        self.logger.info(f"Loading semantic index from: {index_path}")
        
        try:
            # Load FAISS index
            faiss_file = index_path / "semantic_index.faiss"
            if faiss_file.exists():
                self.faiss_index = faiss.read_index(str(faiss_file))
                self.logger.info(f"✅ FAISS index loaded: {self.faiss_index.ntotal} vectors")
            else:
                raise FileNotFoundError(f"FAISS index not found: {faiss_file}")
            
            # Load snippets metadata
            snippets_file = index_path / "snippets.json"
            if snippets_file.exists():
                with open(snippets_file, 'r', encoding='utf-8') as f:
                    snippet_data = json.load(f)
                self.snippets = [ClimateSnippet.from_dict(data) for data in snippet_data]
                self.logger.info(f"✅ Snippets loaded: {len(self.snippets)} items")
            else:
                raise FileNotFoundError(f"Snippets file not found: {snippets_file}")
            
            # Load metadata if available
            metadata_file = index_path / "index_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                    if 'model_name' in self.metadata:
                        self.model_name = self.metadata['model_name']
            
            # Validate index
            is_valid, issues = self.validate_index()
            if not is_valid:
                self.logger.warning(f"Index validation issues: {issues}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load index: {e}")
            raise
    
    def load_model(self) -> None:
        """Load sentence transformer model."""
        if self.model is None:
            self.logger.info(f"Loading model: {self.model_name} on {self.device}")
            self.model = SentenceTransformer(self.model_name)
            self.model = self.model.to(self.device)
            self.logger.info("✅ Model loaded successfully")
    
    def get_index_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the loaded index.
        
        Returns:
            Dictionary containing index statistics
        """
        if not self.snippets:
            return {"error": "No index loaded"}
        
        # Basic counts
        total_snippets = len(self.snippets)
        unique_companies = len(set(s.ticker for s in self.snippets if s.ticker))
        unique_years = sorted(set(s.year for s in self.snippets if s.year > 0))
        
        # Stock index distribution
        stock_indices = {}
        for snippet in self.snippets:
            idx = snippet.stock_index or "Unknown"
            stock_indices[idx] = stock_indices.get(idx, 0) + 1
        
        # Temporal distribution
        year_counts = {}
        for snippet in self.snippets:
            if snippet.year > 0:
                year_counts[snippet.year] = year_counts.get(snippet.year, 0) + 1
        
        # Text statistics
        text_lengths = [len(snippet.text.split()) for snippet in self.snippets if snippet.text]
        
        statistics = {
            "total_snippets": total_snippets,
            "unique_companies": unique_companies,
            "unique_years": len(unique_years),
            "year_range": (min(unique_years), max(unique_years)) if unique_years else (None, None),
            "stock_indices": stock_indices,
            "year_distribution": year_counts,
            "text_statistics": {
                "average_words_per_snippet": round(np.mean(text_lengths), 1) if text_lengths else 0,
                "total_words": sum(text_lengths) if text_lengths else 0
            }
        }
        
        # FAISS index info
        if self.faiss_index:
            statistics["faiss_index"] = {
                "total_vectors": self.faiss_index.ntotal,
                "dimension": self.faiss_index.d,
                "is_trained": self.faiss_index.is_trained
            }
        
        return statistics
    
    def validate_index(self) -> Tuple[bool, List[str]]:
        """
        Validate the consistency of the loaded index.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if not self.faiss_index:
            issues.append("FAISS index not loaded")
        
        if not self.snippets:
            issues.append("No snippets loaded")
        
        # Check consistency
        if self.faiss_index and self.snippets:
            faiss_count = self.faiss_index.ntotal
            snippets_count = len(self.snippets)
            
            if faiss_count != snippets_count:
                issues.append(f"Count mismatch: FAISS has {faiss_count} vectors but {snippets_count} snippets")
        
        # Check data quality
        if self.snippets:
            empty_texts = sum(1 for s in self.snippets if not s.text.strip())
            if empty_texts > 0:
                issues.append(f"Found {empty_texts} snippets with empty text")
            
            missing_tickers = sum(1 for s in self.snippets if not s.ticker)
            if missing_tickers > 0:
                issues.append(f"Found {missing_tickers} snippets with missing tickers")
        
        is_valid = len(issues) == 0
        if is_valid:
            self.logger.info("✅ Index validation passed")
        else:
            self.logger.warning(f"⚠️ Index validation found {len(issues)} issues")
        
        return is_valid, issues
    
    def semantic_search(self, query: str, top_k: int = 100, min_score: float = 0.40) -> List[ClimateSnippet]:
        """
        Core semantic search functionality.
        
        Args:
            query: Search query string
            top_k: Maximum number of results
            min_score: Minimum similarity score threshold
            
        Returns:
            List of ClimateSnippet objects with similarity scores
        """
        if not self.model:
            self.load_model()
        
        if not self.faiss_index:
            raise RuntimeError("FAISS index not loaded")
        
        # Create query embedding
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        # Search FAISS index
        search_k = min(top_k * 2, len(self.snippets))  # Get extra for filtering
        scores, indices = self.faiss_index.search(query_embedding.astype(np.float32), search_k)
        
        # Process results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= min_score and idx < len(self.snippets):
                snippet = self.snippets[idx]
                # Create a copy to avoid modifying the original
                import copy
                result_snippet = copy.deepcopy(snippet)
                result_snippet.similarity_score = float(score)
                result_snippet.matched_query = query
                results.append(result_snippet)
                results.append(result_snippet)
                
                if len(results) >= top_k:
                    break
        
        return results
    
    def multi_query_search(self, queries: List[str], top_k: int = 100, 
                      min_score: float = 0.40, aggregation: str = 'max') -> List[ClimateSnippet]:
        """
        Search with multiple queries and combine results.
        
        Args:
            queries: List of query strings
            top_k: UNUSED - kept for API compatibility
            min_score: Minimum similarity score threshold
            aggregation: How to combine scores ('max', 'mean', 'sum')
            
        Returns:
            List of ALL unique ClimateSnippet objects above min_score threshold
        """
        if not queries:
            return []
        
        # Collect all results by snippet identifier
        snippet_scores = defaultdict(list)
        snippet_objects = {}
        
        for query in queries:
            # Get ALL results above threshold (no top_k limit)
            results = self.semantic_search(query, top_k=len(self.snippets), min_score=min_score)
            for snippet in results:
                # Use snippet object ID as identifier for deduplication
                snippet_id = id(snippet)
                snippet_scores[snippet_id].append(snippet.similarity_score)
                if snippet_id not in snippet_objects:
                    snippet_objects[snippet_id] = snippet
        
        # Aggregate scores for deduplicated results
        final_results = []
        for snippet_id, scores in snippet_scores.items():
            snippet = snippet_objects[snippet_id]
            
            if aggregation == 'max':
                final_score = max(scores)
            elif aggregation == 'mean':
                final_score = np.mean(scores)
            elif aggregation == 'sum':
                final_score = sum(scores)
            else:
                final_score = max(scores)  # Default to max
            
            # Update snippet with aggregated score
            snippet.similarity_score = final_score
            snippet.matched_query = f"Multi-query ({len(queries)} queries)"
            final_results.append(snippet)
        
        # Sort by score and return ALL results (no limiting)
        final_results.sort(key=lambda x: x.similarity_score, reverse=True)
        return final_results
    
    def filter_snippets(self, snippets: List[ClimateSnippet], 
                       companies: Optional[List[str]] = None,
                       years: Optional[List[int]] = None,
                       quarters: Optional[List[str]] = None,
                       stock_indices: Optional[List[str]] = None,
                       min_climate_ratio: Optional[float] = None) -> List[ClimateSnippet]:
        """
        Filter snippets by various criteria.
        
        Args:
            snippets: List of snippets to filter
            companies: List of company tickers to include
            years: List of years to include
            quarters: List of quarters to include  
            stock_indices: List of stock indices to include
            min_climate_ratio: Minimum climate sentence ratio
            
        Returns:
            Filtered list of snippets
        """
        filtered = snippets
        
        if companies:
            filtered = [s for s in filtered if s.ticker in companies]
        
        if years:
            filtered = [s for s in filtered if s.year in years]
        
        if quarters:
            filtered = [s for s in filtered if s.quarter in quarters]
        
        if stock_indices:
            filtered = [s for s in filtered if s.stock_index in stock_indices]
        
        if min_climate_ratio is not None:
            filtered = [s for s in filtered if s.climate_sentence_ratio and s.climate_sentence_ratio >= min_climate_ratio]
        
        return filtered
    
    def aggregate_by_time(self, snippets: List[ClimateSnippet], freq: str = 'quarter') -> pd.DataFrame:
        """
        Aggregate snippets by time period.
        
        Args:
            snippets: List of snippets to aggregate
            freq: Frequency ('quarter', 'year', 'month')
            
        Returns:
            DataFrame with time-aggregated results
        """
        if not snippets:
            return pd.DataFrame()
        
        data = []
        for snippet in snippets:            
            data.append({
                'date': snippet.date,
                'ticker': snippet.ticker,
                'company_name': snippet.company_name,
                'similarity_score': snippet.similarity_score,
                'climate_ratio': snippet.climate_sentence_ratio,
                'stock_index': snippet.stock_index
            })
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Aggregate by period and ticker
        agg_df = df.groupby(['date', 'ticker']).agg({
            'company_name': 'first',
            'similarity_score': ['mean', 'max', 'count'],
            'climate_ratio': 'mean',
            'stock_index': 'first'
        }).round(4)
        
        # Flatten column names
        agg_df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in agg_df.columns.values]
        agg_df = agg_df.reset_index()
        
        return agg_df
    
    def aggregate_by_firm(self, snippets: List[ClimateSnippet]) -> pd.DataFrame:
        """
        Aggregate snippets by firm.
        
        Args:
            snippets: List of snippets to aggregate
            
        Returns:
            DataFrame with firm-level aggregations
        """
        if not snippets:
            return pd.DataFrame()
        
        firm_data = defaultdict(list)
        for snippet in snippets:
            firm_data[snippet.ticker].append(snippet)
        
        results = []
        for ticker, firm_snippets in firm_data.items():
            scores = [s.similarity_score for s in firm_snippets]
            climate_ratios = [s.climate_sentence_ratio for s in firm_snippets if s.climate_sentence_ratio is not None]
            
            results.append({
                'ticker': ticker,
                'company_name': firm_snippets[0].company_name,
                'stock_index': firm_snippets[0].stock_index,
                'total_mentions': len(firm_snippets),
                'avg_score': np.mean(scores),
                'max_score': max(scores),
                'avg_climate_ratio': np.mean(climate_ratios) if climate_ratios else None,
                'year_range': f"{min(s.year for s in firm_snippets)}-{max(s.year for s in firm_snippets)}",
                'latest_date': max(s.date for s in firm_snippets if s.date)
            })
        
        return pd.DataFrame(results)
    
    def create_attention_timeseries(self, analysis_type: str, topic: str, **kwargs) -> pd.DataFrame:
        """
        Generic attention time series creation.
        
        Args:
            analysis_type: Type of analysis ('policies', 'risk_analysis', 'opportunities')
            topic: Specific topic within analysis type
            **kwargs: Additional parameters (freq, companies, etc.)
            
        Returns:
            DataFrame with time series data
        """
        # Get queries and threshold from config or fallback
        if self.config:
            try:
                queries = self.config.get_query_list(analysis_type, topic)
                threshold = self.config.get_threshold(analysis_type, topic)
            except:
                queries = self._get_default_queries(analysis_type, topic)
                threshold = kwargs.get('min_score', 0.40)
        else:
            queries = self._get_default_queries(analysis_type, topic)
            threshold = kwargs.get('min_score', 0.40)
        
        # Perform search
        results = self.multi_query_search(queries, top_k=kwargs.get('top_k', 1000), min_score=threshold)
        
        # Apply additional filters
        if 'companies' in kwargs:
            results = self.filter_snippets(results, companies=kwargs['companies'])
        if 'years' in kwargs:
            results = self.filter_snippets(results, years=kwargs['years'])
        
        # Create time series
        return self.aggregate_by_time(results, freq=kwargs.get('freq', 'quarter'))
    
    def _get_default_queries(self, analysis_type: str, topic: str) -> List[str]:
        """
        Fallback queries when no config available.
        
        Args:
            analysis_type: Type of analysis
            topic: Specific topic
            
        Returns:
            List of default query strings
        """
        default_queries = {
            'policies': {
                'paris_agreement': ['Paris Agreement', 'climate accord', 'COP21'],
                'carbon_pricing': ['carbon tax', 'carbon pricing', 'emissions trading'],
                'green_deal': ['European Green Deal', 'EU Green Deal', 'Green Deal']
            },
            'risk_analysis': {
                'physical_risk': ['physical climate risk', 'extreme weather', 'flooding', 'climate disasters'],
                'transition_risk': ['transition risk', 'stranded assets', 'carbon regulation', 'policy risk']
            },
            'opportunities': {
                'renewable_energy': ['renewable energy', 'clean energy', 'solar wind power'],
                'green_finance': ['green bonds', 'sustainable finance', 'ESG investments']
            }
        }
        
        return default_queries.get(analysis_type, {}).get(topic, [f"{topic} climate"])
    
    def get_available_analyses(self) -> Dict[str, List[str]]:
        """
        Get available analysis types and topics.
        
        Returns:
            Dictionary mapping analysis types to available topics
        """
        if self.config:
            return self.config.list_all_analyses()
        else:
            return {
                'policies': ['paris_agreement', 'carbon_pricing', 'green_deal'],
                'risk_analysis': ['physical_risk', 'transition_risk'],
                'opportunities': ['renewable_energy', 'green_finance']
            }
    
    def export_results(self, data: Union[List[ClimateSnippet], pd.DataFrame], 
                      output_path: Path, format: str = 'csv') -> None:
        """
        Export results to file.
        
        Args:
            data: Data to export (snippets or DataFrame)
            output_path: Output file path
            format: Export format ('csv', 'json', 'parquet')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(data, list):
            # Convert snippets to DataFrame
            df_data = [snippet.to_dict() for snippet in data]
            df = pd.DataFrame(df_data)
        else:
            df = data
        
        if format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'json':
            df.to_json(output_path, orient='records', indent=2)
        elif format == 'parquet':
            df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"✅ Results exported to {output_path}")