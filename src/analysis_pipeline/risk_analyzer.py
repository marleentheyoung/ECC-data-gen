#!/usr/bin/env python3
"""
Climate Risk Attention Analyzer

Simplified analyzer for climate risk attention using semantic search.
Creates time series CSV output for broad climate risk discussions.

Author: Marleen de Jonge
Date: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import sys

# Import base searcher
try:
    from .base_searcher import BaseSemanticSearcher, ClimateSnippet
except ImportError:
    # Fallback for development
    sys.path.append(str(Path(__file__).parent))
    from base_searcher import BaseSemanticSearcher, ClimateSnippet

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskAnalyzer(BaseSemanticSearcher):
    """
    Simplified analyzer for climate risk attention analysis.
    Inherits core semantic search functionality and creates CSV output.
    """
    
    def __init__(self, index_path: Path, config_path: Optional[str] = None):
        """
        Initialize risk analyzer.
        
        Args:
            index_path: Path to semantic index
            config_path: Path to configuration file
        """
        super().__init__(index_path, config_path)
        
        # Risk queries - broad climate risks (physical + transition)
        self.risk_queries = [
            'extreme weather climate physical risk operational disruption',
            'supply chain disruption weather events natural disasters',
            'flooding drought wildfire hurricane climate impacts business',
            'climate adaptation resilience infrastructure vulnerability',
            'weather operational disruption facility damage costs',
            'stranded assets carbon intensive business model risk',
            'technology disruption energy transition competitive threat',
            'carbon tax transition costs regulatory burden climate',
            'fossil fuel asset impairment devaluation stranded risk',
            'business model transition technological change disruption',
            'climate policy regulatory compliance risk burden',
            'carbon pricing emissions trading cost impact',
            'environmental standards regulatory changes compliance costs',
            'climate disclosure requirements reporting risk liability',
            'physical climate risk extreme weather operational impact',
            'transition risk carbon intensive assets regulatory',
            'climate litigation legal risk disclosure liability',
            'reputation risk climate performance ESG rating impact'
        ]
        
        logger.info("✅ Risk Analyzer initialized")
    
    def create_risk_timeseries(self,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None,
                              frequency: str = 'quarter',
                              companies: Optional[List[str]] = None,
                              min_score: Optional[float] = None) -> pd.DataFrame:
        """
        Create time series of climate risk attention.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            frequency: Time frequency ('quarter', 'year', 'month')
            companies: Optional list of company tickers to include
            min_score: Minimum similarity threshold
            
        Returns:
            DataFrame with climate risk attention time series
        """
        # Get threshold
        threshold = min_score or 0.40
        
        logger.info(f"Using {len(self.risk_queries)} queries with threshold {threshold}")
        
        # Perform semantic search
        results = self.multi_query_search(
            queries=self.risk_queries,
            top_k=2000,
            min_score=threshold,
            aggregation='max'
        )
        
        logger.info(f"Found {len(results)} relevant snippets")
        
        # Apply filters
        if companies:
            results = self.filter_snippets(results, companies=companies)
            logger.info(f"After company filter: {len(results)} snippets")
        
        if start_date or end_date:
            results = self._filter_by_date_range(results, start_date, end_date)
            logger.info(f"After date filter: {len(results)} snippets")
        
        # Create time series
        if not results:
            logger.warning("No results found for risk analysis")
            return pd.DataFrame()
        
        timeseries = self.aggregate_by_time(results, freq=frequency)
        
        # Add risk-specific columns to match policy format
        timeseries['risk_type'] = 'climate_risks'
        timeseries['analysis_date'] = datetime.now().strftime('%Y-%m-%d')
        
        # Rename columns to match policy output format
        timeseries = timeseries.rename(columns={
            'date': 'date',
            'ticker': 'ticker', 
            'company_name_first': 'company_name_first',
            'similarity_score_mean': 'similarity_score_mean',
            'similarity_score_max': 'similarity_score_max', 
            'similarity_score_count': 'similarity_score_count',
            'climate_ratio_mean': 'climate_ratio_mean',
            'stock_index_first': 'stock_index_first'
        })
        
        return timeseries
    
    def _filter_by_date_range(self, snippets: List[ClimateSnippet], 
                             start_date: Optional[str], 
                             end_date: Optional[str]) -> List[ClimateSnippet]:
        """Filter snippets by date range."""
        if not start_date and not end_date:
            return snippets
        
        filtered = []
        for snippet in snippets:
            try:
                snippet_date = pd.to_datetime(snippet.date)
                if start_date and snippet_date < pd.to_datetime(start_date):
                    continue
                if end_date and snippet_date > pd.to_datetime(end_date):
                    continue
                filtered.append(snippet)
            except:
                continue
        
        return filtered


# Convenience function for direct usage
def analyze_risks(index_path: Path, 
                 output_file: Optional[Path] = None,
                 **kwargs) -> pd.DataFrame:
    """
    Convenience function for climate risk analysis.
    
    Args:
        index_path: Path to semantic index
        output_file: Optional output CSV file path
        **kwargs: Additional arguments for analysis
        
    Returns:
        DataFrame with climate risk time series
    """
    analyzer = RiskAnalyzer(index_path)
    results = analyzer.create_risk_timeseries(**kwargs)
    
    if output_file and not results.empty:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_file, index=False)
        logger.info(f"✅ Risk analysis exported to {output_file}")
    
    return results