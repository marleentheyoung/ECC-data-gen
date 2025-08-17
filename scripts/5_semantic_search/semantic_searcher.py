#!/usr/bin/env python3
"""
Semantic Climate Searcher for ECC Data Generation Pipeline

This script provides semantic search capabilities over the built climate indexes,
supporting both aggregate time-series analyses and firm-level studies.

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

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import torch
except ImportError as e:
    raise ImportError(f"Required packages not installed: {e}")

from build_semantic_index import ClimateSnippet


class SemanticClimateSearcher:
    """
    Semantic search system for climate-related content with support for
    aggregate time-series and firm-level analyses.
    """
    
    def __init__(self, index_path: Path, model_name: str = None):
        self.index_path = Path(index_path)
        self.model = None
        self.model_name = model_name
        self.faiss_index = None
        self.snippets = []
        self.metadata = {}
        
        self.device = self._get_device()
        self.logger = logging.getLogger(__name__)
        
        # Load index
        self.load_index()
    
    def _get_device(self) -> str:
        """Get optimal device for processing."""
        if torch.backends.mps.is_available():
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    
    def load_index(self):
        """Load the semantic index and associated data."""
        self.logger.info(f"Loading semantic index from: {self.index_path}")
        
        # Find metadata file
        metadata_files = list(self.index_path.glob("*_metadata.json"))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata file found in {self.index_path}")
        
        metadata_file = metadata_files[0]
        with open(metadata_file, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # Extract paths from metadata
        index_file = Path(self.metadata['files']['index'])
        snippets_file = Path(self.metadata['files']['snippets'])
        
        # Load model
        self.model_name = self.model_name or self.metadata['model_name']
        self.model = SentenceTransformer(self.model_name)
        self.model = self.model.to(self.device)
        
        # Load FAISS index
        self.faiss_index = faiss.read_index(str(index_file))
        
        # Load snippets
        with open(snippets_file, 'r', encoding='utf-8') as f:
            snippet_data = json.load(f)
        
        self.snippets = [ClimateSnippet.from_dict(data) for data in snippet_data]
        
        self.logger.info(f"✅ Loaded index: {len(self.snippets)} snippets")
        self.logger.info(f"📊 Model: {self.model_name}")
        self.logger.info(f"🗓️ Date range: {self.metadata['date_range']['start']} to {self.metadata['date_range']['end']}")
    
    def semantic_search(self, query: str, top_k: int = 100, 
                       min_score: float = 0.40) -> List[ClimateSnippet]:
        """
        Perform semantic search for climate-related content.
        
        Args:
            query: Search query
            top_k: Maximum number of results to return
            min_score: Minimum similarity score threshold
            
        Returns:
            List of matching climate snippets with similarity scores
        """
        # Encode query
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        # Search FAISS index
        scores, indices = self.faiss_index.search(
            query_embedding.astype(np.float32), 
            min(top_k * 2, len(self.snippets))  # Get extra for filtering
        )
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= min_score and idx < len(self.snippets):
                snippet = self.snippets[idx]
                # Update snippet with search results
                snippet.similarity_score = float(score)
                snippet.matched_query = query
                results.append(snippet)
                
                if len(results) >= top_k:
                    break
        
        return results
    
    def multi_query_search(self, queries: List[str], top_k: int = 50,
                          min_score: float = 0.40) -> List[ClimateSnippet]:
        """
        Search using multiple queries and combine results.
        
        Args:
            queries: List of search queries
            top_k: Results per query
            min_score: Minimum similarity score
            
        Returns:
            Combined and deduplicated results
        """
        all_results = []
        seen_indices = set()
        
        for query in queries:
            results = self.semantic_search(query, top_k, min_score)
            
            for snippet in results:
                # Create unique identifier for deduplication
                snippet_id = f"{snippet.ticker}_{snippet.date}_{hash(snippet.text[:100])}"
                
                if snippet_id not in seen_indices:
                    seen_indices.add(snippet_id)
                    all_results.append(snippet)
        
        # Sort by similarity score
        all_results.sort(key=lambda x: x.similarity_score, reverse=True)
        return all_results
    
    def filter_snippets(self, snippets: List[ClimateSnippet],
                       companies: Optional[List[str]] = None,
                       years: Optional[List[int]] = None,
                       quarters: Optional[List[str]] = None,
                       stock_indices: Optional[List[str]] = None,
                       min_sentence_ratio: Optional[float] = None) -> List[ClimateSnippet]:
        """
        Apply filters to snippet results.
        
        Args:
            snippets: List of snippets to filter
            companies: Company tickers to include
            years: Years to include
            quarters: Quarters to include
            stock_indices: Stock indices to include
            min_sentence_ratio: Minimum climate sentence ratio
            
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
        
        if min_sentence_ratio:
            filtered = [s for s in filtered 
                       if s.climate_sentence_ratio and s.climate_sentence_ratio >= min_sentence_ratio]
        
        return filtered
    
    def aggregate_by_time(self, snippets: List[ClimateSnippet],
                         freq: str = 'Q') -> pd.DataFrame:
        """
        Aggregate snippets by time period.
        
        Args:
            snippets: List of snippets to aggregate
            freq: Frequency ('Q' for quarterly, 'Y' for yearly, 'M' for monthly)
            
        Returns:
            DataFrame with time-series aggregation
        """
        # Convert to DataFrame
        data = []
        for snippet in snippets:
            try:
                date = pd.to_datetime(snippet.date)
                data.append({
                    'date': date,
                    'year': snippet.year,
                    'quarter': snippet.quarter,
                    'ticker': snippet.ticker,
                    'stock_index': snippet.stock_index,
                    'similarity_score': snippet.similarity_score,
                    'climate_sentence_ratio': snippet.climate_sentence_ratio,
                    'sentence_count': snippet.sentence_count or 1,
                    'count': 1
                })
            except:
                continue
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Create time period
        if freq == 'Q':
            df['period'] = df['date'].dt.to_period('Q')
        elif freq == 'Y':
            df['period'] = df['date'].dt.to_period('Y')
        elif freq == 'M':
            df['period'] = df['date'].dt.to_period('M')
        else:
            raise ValueError("freq must be 'Q', 'Y', or 'M'")
        
        # Aggregate
        agg_df = df.groupby('period').agg({
            'count': 'sum',
            'similarity_score': ['mean', 'std'],
            'climate_sentence_ratio': 'mean',
            'sentence_count': 'sum',
            'ticker': 'nunique'
        }).round(4)
        
        # Flatten column names
        agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
        agg_df = agg_df.reset_index()
        
        # Add date for plotting
        agg_df['date'] = agg_df['period'].dt.start_time
        
        return agg_df
    
    def aggregate_by_firm(self, snippets: List[ClimateSnippet]) -> pd.DataFrame:
        """
        Aggregate snippets by firm.
        
        Args:
            snippets: List of snippets to aggregate
            
        Returns:
            DataFrame with firm-level aggregation
        """
        # Convert to DataFrame
        data = []
        for snippet in snippets:
            data.append({
                'ticker': snippet.ticker,
                'company_name': snippet.company_name,
                'stock_index': snippet.stock_index,
                'similarity_score': snippet.similarity_score,
                'climate_sentence_ratio': snippet.climate_sentence_ratio,
                'sentence_count': snippet.sentence_count or 1,
                'count': 1
            })
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Aggregate by firm
        firm_df = df.groupby(['ticker', 'company_name', 'stock_index']).agg({
            'count': 'sum',
            'similarity_score': ['mean', 'std', 'max'],
            'climate_sentence_ratio': ['mean', 'std'],
            'sentence_count': 'sum'
        }).round(4)
        
        # Flatten column names
        firm_df.columns = ['_'.join(col).strip() for col in firm_df.columns.values]
        firm_df = firm_df.reset_index()
        
        return firm_df
    
    def topic_analysis(self, topic_queries: Dict[str, List[str]], 
                      top_k: int = 100, min_score: float = 0.40) -> Dict[str, List[ClimateSnippet]]:
        """
        Analyze multiple climate topics using predefined queries.
        
        Args:
            topic_queries: Dictionary mapping topic names to lists of queries
            top_k: Results per topic
            min_score: Minimum similarity score
            
        Returns:
            Dictionary mapping topic names to snippet results
        """
        results = {}
        
        for topic, queries in topic_queries.items():
            self.logger.info(f"Analyzing topic: {topic}")
            topic_results = self.multi_query_search(queries, top_k, min_score)
            results[topic] = topic_results
            self.logger.info(f"Found {len(topic_results)} snippets for {topic}")
        
        return results
    
    def create_time_series(self, query: str, start_date: str = None, 
                          end_date: str = None, freq: str = 'Q',
                          companies: Optional[List[str]] = None,
                          min_score: float = 0.35) -> pd.DataFrame:
        """
        Create time series of climate attention for a specific query.
        
        Args:
            query: Search query
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            freq: Time frequency ('Q', 'Y', 'M')
            companies: Optional list of companies to include
            min_score: Minimum similarity score threshold
            
        Returns:
            Time series DataFrame
        """
        # Search for relevant snippets
        snippets = self.semantic_search(query, top_k=2000, min_score=min_score)
        
        # Apply company filter
        if companies:
            snippets = self.filter_snippets(snippets, companies=companies)
        
        # Apply date filters
        if start_date or end_date:
            filtered_snippets = []
            for snippet in snippets:
                try:
                    snippet_date = pd.to_datetime(snippet.date)
                    if start_date and snippet_date < pd.to_datetime(start_date):
                        continue
                    if end_date and snippet_date > pd.to_datetime(end_date):
                        continue
                    filtered_snippets.append(snippet)
                except:
                    continue
            snippets = filtered_snippets
        
        # Aggregate by time
        return self.aggregate_by_time(snippets, freq)
    
    def create_policy_attention_timeseries(self, policy_type: str = 'paris_agreement',
                                         start_date: str = None, end_date: str = None,
                                         freq: str = 'Q', min_score: float = 0.45,
                                         companies: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create time series for specific climate policy attention (unnormalized).
        
        Args:
            policy_type: Type of policy ('paris_agreement', 'eu_green_deal', 'us_ira', 'custom')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD) 
            freq: Time frequency ('Q', 'Y', 'M')
            min_score: Minimum similarity score threshold (default 0.45)
            companies: Optional list of companies to include
            
        Returns:
            Time series DataFrame with unnormalized attention counts
        """
        # Define policy-specific queries
        policy_queries = {
            'paris_agreement': [
                'Paris Agreement COP21 international climate accord',
                'Paris climate agreement global climate deal',
                'COP21 Paris climate conference agreement',
                'international climate agreement Paris accord',
                'global climate deal Paris framework',
                'policy package to limit global warming to 1.5 degrees'
                'Paris Agreement commitments NDCs nationally determined',
                'climate accord Paris international cooperation',
                'Paris Agreement implementation compliance',
            ],
            'eu_green_deal': [
                'European Green Deal EU climate policy',
                'EU Green Deal climate neutrality 2050',
                'European climate law green deal',
                'EU taxonomy sustainable finance green deal',
                'Green Deal investment plan Europe',
                'European green transition climate policy',
                'EU climate strategy green deal framework'
            ],
            'us_ira': [
                'Inflation Reduction Act IRA climate provisions',
                'US IRA clean energy tax credits',
                'Inflation Reduction Act climate investment',
                'IRA renewable energy incentives',
                'US climate investment IRA subsidies',
                'Inflation Reduction Act clean technology'
            ],
            'cop_meetings': [
                'COP climate conference international negotiations',
                'Conference of Parties climate summit',
                'COP26 Glasgow climate conference',
                'COP27 Egypt climate summit',
                'COP28 Dubai climate conference',
                'international climate negotiations COP'
            ]
        }
        
        if policy_type not in policy_queries:
            raise ValueError(f"Policy type must be one of: {list(policy_queries.keys())}")
        
        queries = policy_queries[policy_type]
        
        # Use multi-query search for better coverage
        snippets = self.multi_query_search(queries, top_k=100, min_score=min_score)
        
        self.logger.info(f"Found {len(snippets)} snippets for {policy_type} (min_score={min_score})")
        
        # Apply filters
        if companies:
            snippets = self.filter_snippets(snippets, companies=companies)
        
        if start_date or end_date:
            filtered_snippets = []
            for snippet in snippets:
                try:
                    snippet_date = pd.to_datetime(snippet.date)
                    if start_date and snippet_date < pd.to_datetime(start_date):
                        continue
                    if end_date and snippet_date > pd.to_datetime(end_date):
                        continue
                    filtered_snippets.append(snippet)
                except:
                    continue
            snippets = filtered_snippets
        
        # Create time series
        ts_df = self.aggregate_by_time(snippets, freq)
        
        # Add policy metadata
        ts_df['policy_type'] = policy_type
        ts_df['min_score_threshold'] = min_score
        ts_df['query_count'] = len(queries)
        
        return ts_df
    
    def create_custom_policy_timeseries(self, custom_queries: List[str],
                                      policy_name: str = 'custom_policy',
                                      start_date: str = None, end_date: str = None,
                                      freq: str = 'Q', min_score: float = 0.45,
                                      companies: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create time series for custom policy queries.
        
        Args:
            custom_queries: List of custom search queries
            policy_name: Name for the policy (for metadata)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            freq: Time frequency ('Q', 'Y', 'M')
            min_score: Minimum similarity score threshold
            companies: Optional list of companies to include
            
        Returns:
            Time series DataFrame
        """
        snippets = self.multi_query_search(custom_queries, top_k=100, min_score=min_score)
        
        self.logger.info(f"Found {len(snippets)} snippets for {policy_name} (min_score={min_score})")
        
        # Apply filters
        if companies:
            snippets = self.filter_snippets(snippets, companies=companies)
        
        if start_date or end_date:
            filtered_snippets = []
            for snippet in snippets:
                try:
                    snippet_date = pd.to_datetime(snippet.date)
                    if start_date and snippet_date < pd.to_datetime(start_date):
                        continue
                    if end_date and snippet_date > pd.to_datetime(end_date):
                        continue
                    filtered_snippets.append(snippet)
                except:
                    continue
            snippets = filtered_snippets
        
        # Create time series
        ts_df = self.aggregate_by_time(snippets, freq)
        
        # Add metadata
        ts_df['policy_type'] = policy_name
        ts_df['min_score_threshold'] = min_score
        ts_df['query_count'] = len(custom_queries)
        
        return ts_df
    
    def firm_exposure_analysis(self, queries: Union[str, List[str]], 
                             min_mentions: int = 1) -> pd.DataFrame:
        """
        Analyze firm-level exposure to specific climate topics.
        
        Args:
            queries: Single query or list of queries
            min_mentions: Minimum number of mentions to include firm
            
        Returns:
            DataFrame with firm exposure metrics
        """
        if isinstance(queries, str):
            snippets = self.semantic_search(queries, top_k=2000, min_score=0.35)
        else:
            snippets = self.multi_query_search(queries, top_k=500, min_score=0.35)
        
        # Aggregate by firm
        firm_df = self.aggregate_by_firm(snippets)
        
        # Filter by minimum mentions
        firm_df = firm_df[firm_df['count_sum'] >= min_mentions]
        
        # Add exposure rankings
        firm_df['exposure_rank'] = firm_df['count_sum'].rank(method='dense', ascending=False)
        firm_df['similarity_rank'] = firm_df['similarity_score_mean'].rank(method='dense', ascending=False)
        
        # Sort by exposure
        firm_df = firm_df.sort_values('count_sum', ascending=False)
        
        return firm_df
    
    def comparative_analysis(self, query1: str, query2: str, 
                           label1: str = "Topic 1", label2: str = "Topic 2") -> Dict[str, Any]:
        """
        Compare two climate topics across time and firms.
        
        Args:
            query1: First query
            query2: Second query
            label1: Label for first topic
            label2: Label for second topic
            
        Returns:
            Dictionary with comparative analysis results
        """
        # Get snippets for both queries
        snippets1 = self.semantic_search(query1, top_k=1000, min_score=0.35)
        snippets2 = self.semantic_search(query2, top_k=1000, min_score=0.35)
        
        # Time series analysis
        ts1 = self.aggregate_by_time(snippets1)
        ts2 = self.aggregate_by_time(snippets2)
        
        # Firm analysis
        firm1 = self.aggregate_by_firm(snippets1)
        firm2 = self.aggregate_by_firm(snippets2)
        
        # Calculate overlap
        tickers1 = set(s.ticker for s in snippets1)
        tickers2 = set(s.ticker for s in snippets2)
        overlap = tickers1.intersection(tickers2)
        
        return {
            'time_series': {
                label1: ts1,
                label2: ts2
            },
            'firm_analysis': {
                label1: firm1,
                label2: firm2
            },
            'overlap_analysis': {
                'total_firms_topic1': len(tickers1),
                'total_firms_topic2': len(tickers2),
                'overlap_count': len(overlap),
                'overlap_ratio': len(overlap) / len(tickers1.union(tickers2)) if tickers1.union(tickers2) else 0,
                'overlap_firms': list(overlap)
            }
        }
    
    def export_results(self, snippets: List[ClimateSnippet], 
                      output_path: str, format: str = 'csv') -> None:
        """
        Export search results to file.
        
        Args:
            snippets: List of snippets to export
            output_path: Output file path
            format: Export format ('csv', 'json', 'excel')
        """
        # Convert to DataFrame
        data = [snippet.to_dict() for snippet in snippets]
        df = pd.DataFrame(data)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'csv':
            df.to_csv(output_path, index=False, encoding='utf-8')
        elif format.lower() == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif format.lower() == 'excel':
            df.to_excel(output_path, index=False)
        else:
            raise ValueError("Format must be 'csv', 'json', or 'excel'")
        
        self.logger.info(f"✅ Exported {len(snippets)} results to {output_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the loaded index."""
        if not self.snippets:
            return {}
        
        # Basic stats
        companies = {}
        years = {}
        stock_indices = {}
        
        for snippet in self.snippets:
            companies[snippet.ticker] = companies.get(snippet.ticker, 0) + 1
            years[snippet.year] = years.get(snippet.year, 0) + 1
            stock_indices[snippet.stock_index] = stock_indices.get(snippet.stock_index, 0) + 1
        
        # Sentence ratio stats
        ratios = [s.climate_sentence_ratio for s in self.snippets 
                 if s.climate_sentence_ratio is not None]
        
        return {
            'total_snippets': len(self.snippets),
            'unique_companies': len(companies),
            'year_range': (min(years.keys()), max(years.keys())) if years else None,
            'stock_indices': dict(stock_indices),
            'top_companies': dict(sorted(companies.items(), key=lambda x: x[1], reverse=True)[:10]),
            'climate_sentence_ratio_stats': {
                'mean': np.mean(ratios) if ratios else None,
                'median': np.median(ratios) if ratios else None,
                'std': np.std(ratios) if ratios else None,
                'min': min(ratios) if ratios else None,
                'max': max(ratios) if ratios else None
            }
        }


def main():
    """Example usage and testing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Semantic Climate Searcher - Test and demonstration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test search on combined index
    python semantic_searcher.py --index data/semantic_indexes/combined --test
    
    # Search specific topic
    python semantic_searcher.py --index data/semantic_indexes/SP500 --query "renewable energy investments"
    
    # Create time series
    python semantic_searcher.py --index data/semantic_indexes/combined --timeseries "climate risk" --freq Q
        """
    )
    
    parser.add_argument(
        '--index',
        type=Path,
        required=True,
        help='Path to semantic index directory'
    )
    
    parser.add_argument(
        '--query',
        type=str,
        help='Search query to test'
    )
    
    parser.add_argument(
        '--timeseries',
        type=str,
        help='Create time series for query'
    )
    
    parser.add_argument(
        '--freq',
        choices=['Q', 'Y', 'M'],
        default='Q',
        help='Time series frequency'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run test suite'
    )
    
    parser.add_argument(
        '--export',
        type=str,
        help='Export results to file'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("🔍 Semantic Climate Searcher")
    print("=" * 40)
    
    # Initialize searcher
    searcher = SemanticClimateSearcher(args.index)
    
    # Print statistics
    stats = searcher.get_statistics()
    print(f"📊 Index Statistics:")
    print(f"  Total snippets: {stats['total_snippets']:,}")
    print(f"  Unique companies: {stats['unique_companies']}")
    print(f"  Year range: {stats['year_range']}")
    print(f"  Stock indices: {stats['stock_indices']}")
    print()
    
    if args.test:
        # Test various functionalities
        print("🧪 Running test suite...")
        
        # Test 1: Basic search
        print("\n1. Testing basic semantic search...")
        results = searcher.semantic_search("renewable energy investments", top_k=10)
        print(f"   Found {len(results)} results")
        if results:
            print(f"   Top result: {results[0].company_name} - Score: {results[0].similarity_score:.3f}")
        
        # Test 2: Multi-query search
        print("\n2. Testing multi-query search...")
        queries = ["climate risk", "environmental sustainability", "carbon emissions"]
        results = searcher.multi_query_search(queries, top_k=20)
        print(f"   Found {len(results)} combined results")
        
        # Test 3: Time series
        print("\n3. Testing time series creation...")
        ts = searcher.create_time_series("climate change", freq='Y')
        print(f"   Created time series with {len(ts)} periods")
        if len(ts) > 0:
            print(f"   Date range: {ts['date'].min()} to {ts['date'].max()}")
    
    if args.query:
        # Search for specific query
        print(f"\n🔍 Searching for: '{args.query}'")
        results = searcher.semantic_search(args.query, top_k=20)
        
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results[:5], 1):
            print(f"{i:2d}. {result.company_name} ({result.ticker}) - {result.date}")
            print(f"    Score: {result.similarity_score:.3f}")
            print(f"    Text: {result.text[:100]}...")
            print()
        
        # Export if requested
        if args.export:
            searcher.export_results(results, args.export, 'csv')
    
    if args.timeseries:
        # Create time series
        print(f"\n📈 Creating time series for: '{args.timeseries}'")
        ts = searcher.create_time_series(args.timeseries, freq=args.freq)
        
        print(f"Time series with {len(ts)} periods:")
        print(ts.head(10))
        
        # Export if requested
        if args.export:
            ts_path = Path(args.export).with_suffix('.csv')
            ts.to_csv(ts_path, index=False)
            print(f"Time series exported to: {ts_path}")
    
    # Special demo for Paris Agreement
    if not args.query and not args.timeseries and not args.test:
        print("\n🌍 Demo: Paris Agreement Attention Analysis")
        print("=" * 50)
        
        # Create Paris Agreement time series
        pa_ts = searcher.create_policy_attention_timeseries(
            policy_type='paris_agreement',
            start_date='2010-01-01',
            end_date='2023-12-31',
            freq='Q',
            min_score=0.45
        )
        
        if len(pa_ts) > 0:
            print(f"Paris Agreement mentions over time ({len(pa_ts)} quarters):")
            print(pa_ts[['date', 'count_sum', 'similarity_score_mean']].head(10))
            
            # Create simple plot
            try:
                import matplotlib.pyplot as plt
                
                plt.figure(figsize=(12, 6))
                plt.plot(pa_ts['date'], pa_ts['count_sum'], marker='o', linewidth=2)
                plt.title('Paris Agreement Attention Over Time (Unnormalized)')
                plt.xlabel('Date')
                plt.ylabel('Number of Relevant Snippets')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save plot
                plot_path = 'paris_agreement_timeseries.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.show()
                print(f"📊 Plot saved as: {plot_path}")
                
            except ImportError:
                print("📊 Install matplotlib to create plots")
            
            # Export time series
            pa_ts.to_csv('paris_agreement_timeseries.csv', index=False)
            print("💾 Data exported as: paris_agreement_timeseries.csv")
            
            print("\n💡 To normalize this data later:")
            print("   1. Create total sentences per period file")
            print("   2. Merge: normalized_attention = PA_snippets / total_sentences")
            
        else:
            print("No Paris Agreement mentions found. Try lowering min_score threshold.")
        
        print("\n🔍 Other policy analyses available:")
        print("   searcher.create_policy_attention_timeseries('eu_green_deal')")
        print("   searcher.create_policy_attention_timeseries('us_ira')")
        print("   searcher.create_policy_attention_timeseries('cop_meetings')")


if __name__ == "__main__":
    main()