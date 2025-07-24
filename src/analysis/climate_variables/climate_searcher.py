"""
Climate semantic search system for earnings call analysis.

This module provides semantic search capabilities for climate-related content
in earnings calls, replacing the misnamed "RAG" system with proper semantic search.

Author: Marleen de Jonge
Date: 2025
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import logging
from sentence_transformers import SentenceTransformer
import faiss

logger = logging.getLogger(__name__)


class ClimateSnippet:
    """Data class for climate-related text snippets from earnings calls."""
    
    def __init__(self, company_name: str, ticker: str, year: int, quarter: str, 
                 date: str, speaker: str, profession: str, text: str, 
                 climate_sentiment: str = 'neutral', relevance_score: float = 1.0,
                 sentence_count: int = None, climate_sentence_count: int = None,
                 total_sentences_in_call: int = None, climate_sentence_ratio: float = None):
        self.company_name = company_name
        self.ticker = ticker
        self.year = int(year) if year else None
        self.quarter = quarter
        self.date = date
        self.speaker = speaker
        self.profession = profession
        self.text = text
        self.climate_sentiment = climate_sentiment
        self.relevance_score = relevance_score
        
        # Enhanced fields for sentence ratio analysis
        self.sentence_count = sentence_count
        self.climate_sentence_count = climate_sentence_count
        self.total_sentences_in_call = total_sentences_in_call
        self.climate_sentence_ratio = climate_sentence_ratio
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert snippet to dictionary."""
        base_dict = {
            'company_name': self.company_name,
            'ticker': self.ticker,
            'year': self.year,
            'quarter': self.quarter,
            'date': self.date,
            'speaker': self.speaker,
            'profession': self.profession,
            'text': self.text,
            'climate_sentiment': self.climate_sentiment,
            'relevance_score': self.relevance_score
        }
        
        # Add enhanced fields if available
        if self.sentence_count is not None:
            base_dict['sentence_count'] = self.sentence_count
        if self.climate_sentence_count is not None:
            base_dict['climate_sentence_count'] = self.climate_sentence_count
        if self.total_sentences_in_call is not None:
            base_dict['total_sentences_in_call'] = self.total_sentences_in_call
        if self.climate_sentence_ratio is not None:
            base_dict['climate_sentence_ratio'] = self.climate_sentence_ratio
            
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClimateSnippet':
        """Create snippet from dictionary."""
        return cls(
            company_name=data.get('company_name', ''),
            ticker=data.get('ticker', ''),
            year=data.get('year'),
            quarter=data.get('quarter', ''),
            date=data.get('date', ''),
            speaker=data.get('speaker', ''),
            profession=data.get('profession', ''),
            text=data.get('text', ''),
            climate_sentiment=data.get('climate_sentiment', 'neutral'),
            relevance_score=data.get('relevance_score', 1.0),
            sentence_count=data.get('sentence_count'),
            climate_sentence_count=data.get('climate_sentence_count'),
            total_sentences_in_call=data.get('total_sentences_in_call'),
            climate_sentence_ratio=data.get('climate_sentence_ratio')
        )


class ClimateSemanticSearcher:
    """
    Semantic search system for climate-related content in earnings calls.
    
    This system loads pre-extracted climate snippets and provides semantic search,
    filtering, and aggregation capabilities for climate exposure analysis.
    """
    
    def __init__(self, base_data_path: str = "/Users/marleendejonge/Desktop/ECC-data-generation/data/climate_paragraphs",
                 use_enhanced_files: bool = True):
        self.base_data_path = Path(base_data_path)
        self.use_enhanced_files = use_enhanced_files
        self.snippets = []
        self.model = None
        self.embeddings = None
        self.faiss_index = None
        
        # Investment categories for filtering
        self.investment_categories = {
            "Renewable Energy": ["solar", "wind", "hydro", "renewable", "clean energy"],
            "Electric Vehicles": ["EV", "electric vehicle", "battery", "charging"],
            "Energy Efficiency": ["efficiency", "savings", "optimization", "energy saving"],
            "Climate Strategy": ["climate", "sustainability", "net zero", "ESG", "carbon neutral"],
            "Green Finance": [
                "green bond", "sustainability-linked", "sustainable finance",
                "green finance", "ESG financing", "climate finance", "transition finance",
                "green investment", "sustainable investment", "responsible investment",
                "climate risk disclosure", "climate reporting", "ESG reporting",
                "carbon pricing", "internal carbon price", "carbon credits", "emissions trading",
                "taxonomy aligned", "EU taxonomy", "sustainable debt", "green capital"
            ]
        }
    
    def load_climate_data(self, stock_indices: List[str] = ['SP500', 'STOXX600']) -> None:
        """
        Load climate snippets from JSON files.
        
        Args:
            stock_indices: List of stock indices to load data for
        """
        logger.info(f"Loading climate data for indices: {stock_indices}")
        
        all_snippets = []
        
        for stock_index in stock_indices:
            index_path = self.base_data_path / stock_index
            if not index_path.exists():
                logger.warning(f"Data path not found: {index_path}")
                continue
            
            # Find all climate segment JSON files
            if self.use_enhanced_files:
                json_files = list(index_path.glob("enhanced_climate_segments_*.json"))
            else:
                json_files = list(index_path.glob("climate_segments_*.json"))
                
            logger.info(f"Found {len(json_files)} {'enhanced' if self.use_enhanced_files else 'original'} files for {stock_index}")
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Process each transcript in the file
                    for transcript in data:
                        company_name = transcript.get('company_name', '')
                        ticker = transcript.get('ticker', '')
                        year = transcript.get('year')
                        quarter = transcript.get('quarter', '')
                        date = transcript.get('date', '')
                        
                        # Process each text snippet in the transcript
                        for text_data in transcript.get('texts', []):
                            snippet = ClimateSnippet(
                                company_name=company_name,
                                ticker=ticker,
                                year=year,
                                quarter=quarter,
                                date=date,
                                speaker=text_data.get('speaker', ''),
                                profession=text_data.get('profession', ''),
                                text=text_data.get('text', ''),
                                climate_sentiment='neutral',  # Default, can be updated later
                                sentence_count=text_data.get('sentence_count'),
                                climate_sentence_count=transcript.get('climate_sentence_count'),
                                total_sentences_in_call=transcript.get('total_sentences_in_call'),
                                climate_sentence_ratio=transcript.get('climate_sentence_ratio')
                            )
                            all_snippets.append(snippet)
                
                except Exception as e:
                    logger.error(f"Error loading file {json_file}: {e}")
                    continue
        
        self.snippets = all_snippets
        logger.info(f"✅ Loaded {len(self.snippets)} climate snippets total")
    
    def build_semantic_index(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2') -> None:
        """
        Build semantic search index from loaded snippets.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        if not self.snippets:
            raise ValueError("No snippets loaded. Call load_climate_data() first.")
        
        logger.info(f"Building semantic index with model: {model_name}")
        
        # Load model
        self.model = SentenceTransformer(model_name)
        
        # Extract texts
        texts = [snippet.text for snippet in self.snippets]
        
        # Create embeddings
        logger.info("Creating embeddings...")
        self.embeddings = self.model.encode(
            texts, 
            batch_size=64, 
            show_progress_bar=True, 
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        # Build FAISS index
        logger.info("Building FAISS index...")
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.faiss_index.add(self.embeddings.astype(np.float32))
        
        logger.info(f"✅ Semantic index built with {len(self.snippets)} snippets")
    
    def semantic_search(self, query: str, top_k: int = 10, 
                       min_score: float = 0.40) -> List[Dict[str, Any]]:
        """
        Perform semantic search for climate-related content.
        
        Args:
            query: Search query
            top_k: Maximum number of results to return
            min_score: Minimum similarity score threshold
            
        Returns:
            List of search results with metadata and scores
        """
        if self.model is None or self.faiss_index is None:
            raise ValueError("Semantic index not built. Call build_semantic_index() first.")
        
        # Encode query
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        # Search
        scores, indices = self.faiss_index.search(
            query_embedding.astype(np.float32), 
            min(top_k * 2, len(self.snippets))  # Get extra for filtering
        )
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= min_score and idx < len(self.snippets):
                snippet = self.snippets[idx]
                result = {
                    'score': float(score),
                    'rank': len(results) + 1,
                    **snippet.to_dict()
                }
                results.append(result)
                
                if len(results) >= top_k:
                    break
        
        return results
    
    def search_by_category(self, category: str, **filters) -> List[Dict[str, Any]]:
        """
        Search by predefined investment category.
        
        Args:
            category: Investment category name
            **filters: Additional filters (company, year, etc.)
            
        Returns:
            List of matching snippets
        """
        if category not in self.investment_categories:
            raise ValueError(f"Unknown category: {category}")
        
        keywords = self.investment_categories[category]
        results = []
        
        for snippet in self.snippets:
            # Check if any keywords match
            text_lower = snippet.text.lower()
            if any(keyword.lower() in text_lower for keyword in keywords):
                # Apply additional filters
                if self._passes_filters(snippet, filters):
                    results.append(snippet.to_dict())
        
        return results
    
    def filter_snippets(self, snippets: List[Dict[str, Any]], 
                       companies: Optional[List[str]] = None,
                       sentiment: Optional[str] = None,
                       year_range: Optional[Tuple[int, int]] = None,
                       quarters: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Apply filters to a list of snippets.
        
        Args:
            snippets: List of snippet dictionaries
            companies: List of company tickers to include
            sentiment: Climate sentiment filter ('opportunity', 'risk', 'neutral')
            year_range: Tuple of (start_year, end_year)
            quarters: List of quarters to include (e.g., ['Q1', 'Q2'])
            
        Returns:
            Filtered list of snippets
        """
        filtered = snippets.copy()
        
        if companies:
            filtered = [s for s in filtered if s.get('ticker') in companies]
        
        if sentiment:
            filtered = [s for s in filtered if s.get('climate_sentiment') == sentiment]
        
        if year_range:
            start_year, end_year = year_range
            filtered = [
                s for s in filtered 
                if s.get('year') and start_year <= s['year'] <= end_year
            ]
        
        if quarters:
            filtered = [s for s in filtered if s.get('quarter') in quarters]
        
        return filtered
    
    def get_exposure_summary(self, category: str, **filters) -> Dict[str, Any]:
        """
        Get climate exposure summary for a category.
        
        Args:
            category: Investment category
            **filters: Additional filters
            
        Returns:
            Summary statistics
        """
        snippets = self.search_by_category(category, **filters)
        
        if not snippets:
            return {
                'total_mentions': 0,
                'unique_companies': 0,
                'company_breakdown': {}
            }
        
        # Count mentions by company
        company_counts = defaultdict(int)
        for snippet in snippets:
            ticker = snippet.get('ticker', 'Unknown')
            company_counts[ticker] += 1
        
        return {
            'total_mentions': len(snippets),
            'unique_companies': len(company_counts),
            'company_breakdown': dict(company_counts)
        }
    
    def _passes_filters(self, snippet: ClimateSnippet, filters: Dict[str, Any]) -> bool:
        """Check if snippet passes the given filters."""
        if 'company' in filters and snippet.ticker not in filters['company']:
            return False
        
        if 'year' in filters and snippet.year != filters['year']:
            return False
        
        if 'sentiment' in filters and snippet.climate_sentiment != filters['sentiment']:
            return False
        
        return True
    
    def save_search_index(self, output_path: str) -> None:
        """Save the search index and data for later use."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        if self.faiss_index:
            faiss.write_index(self.faiss_index, str(output_path / "climate_index.faiss"))
        
        # Save embeddings
        if self.embeddings is not None:
            np.save(str(output_path / "climate_embeddings.npy"), self.embeddings)
        
        # Save snippet metadata
        snippet_data = [snippet.to_dict() for snippet in self.snippets]
        with open(output_path / "climate_snippets.json", 'w', encoding='utf-8') as f:
            json.dump(snippet_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Search index saved to {output_path}")
    
    def load_search_index(self, index_path: str, 
                         model_name: str = 'sentence-transformers/all-MiniLM-L6-v2') -> None:
        """Load a previously saved search index."""
        index_path = Path(index_path)
        
        # Load model
        self.model = SentenceTransformer(model_name)
        
        # Load FAISS index
        faiss_file = index_path / "climate_index.faiss"
        if faiss_file.exists():
            self.faiss_index = faiss.read_index(str(faiss_file))
        
        # Load embeddings
        embeddings_file = index_path / "climate_embeddings.npy"
        if embeddings_file.exists():
            self.embeddings = np.load(str(embeddings_file))
        
        # Load snippets
        snippets_file = index_path / "climate_snippets.json"
        if snippets_file.exists():
            with open(snippets_file, 'r', encoding='utf-8') as f:
                snippet_data = json.load(f)
            self.snippets = [ClimateSnippet.from_dict(data) for data in snippet_data]
        
        logger.info(f"✅ Search index loaded from {index_path}")
        logger.info(f"📊 {len(self.snippets)} snippets available for search")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the loaded data."""
        if not self.snippets:
            return {}
        
        companies = {}
        years = {}
        sentiments = {}
        
        for snippet in self.snippets:
            # Count by company
            ticker = snippet.ticker or 'Unknown'
            companies[ticker] = companies.get(ticker, 0) + 1
            
            # Count by year
            year = snippet.year or 'Unknown'
            years[year] = years.get(year, 0) + 1
            
            # Count by sentiment
            sentiment = snippet.climate_sentiment or 'neutral'
            sentiments[sentiment] = sentiments.get(sentiment, 0) + 1
        
        return {
            'total_snippets': len(self.snippets),
            'unique_companies': len(companies),
            'year_range': (min(years.keys()), max(years.keys())) if years else None,
            'top_companies': dict(sorted(companies.items(), key=lambda x: x[1], reverse=True)[:10]),
            'year_distribution': dict(sorted(years.items())),
            'sentiment_distribution': sentiments
        }