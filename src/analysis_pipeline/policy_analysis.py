#!/usr/bin/env python3
"""
Climate Policy Attention Analyzer

Specialized analyzer for climate policy attention using semantic search.
Creates time series of policy attention that can be used for research analysis.

This module focuses purely on policy analysis and data generation - 
all visualization is handled by the visualization module.

Author: Marleen de Jonge
Date: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
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


class PolicyAnalyzer(BaseSemanticSearcher):
    """
    Specialized analyzer for climate policy attention analysis.
    
    Inherits core semantic search functionality and adds policy-specific
    business logic for detecting policy events, analyzing attention patterns,
    and creating policy-focused datasets.
    """
    
    def __init__(self, index_path: Path, config_path: Optional[str] = None):
        """
        Initialize policy analyzer.
        
        Args:
            index_path: Path to semantic index
            config_path: Path to configuration file
        """
        super().__init__(index_path, config_path)
        
        # Policy-specific configuration
        self.policy_events = self._load_policy_events()
        self.policy_queries = self._load_policy_queries()
        
        logger.info("✅ Policy Analyzer initialized")
    
    def _load_policy_events(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        Load major climate policy events and their dates.
        
        Returns:
            Dictionary mapping policy types to list of (date, event_name) tuples
        """
        return {
            'paris_agreement': [
                ('2015-12-12', 'Paris Agreement Adoption'),
                ('2016-11-04', 'Paris Agreement Entry into Force'),
                ('2017-06-01', 'US Paris Withdrawal Announcement'),
                ('2021-02-19', 'US Paris Re-entry')
            ],
            'eu_green_deal': [
                ('2019-12-11', 'EU Green Deal Announcement'),
                ('2021-07-14', 'Fit for 55 Package'),
                ('2022-05-18', 'REPowerEU Plan')
            ],
            'us_ira': [
                ('2022-08-16', 'Inflation Reduction Act Signing'),
                ('2023-01-01', 'IRA Tax Credits Effective')
            ],
            'cop_meetings': [
                ('2015-12-12', 'COP21 Paris'),
                ('2021-11-13', 'COP26 Glasgow'),
                ('2022-11-20', 'COP27 Sharm el-Sheikh'),
                ('2023-12-13', 'COP28 Dubai')
            ]
        }
    
    def _load_policy_queries(self) -> Dict[str, List[str]]:
        """
        Load policy-specific queries if not available in config.
        
        Returns:
            Dictionary mapping policy types to query lists
        """
        return {
            'paris_agreement': [
                'Paris Agreement COP21 international climate accord',
                'Paris Agreement COP21 international climate accord global warming 1.5 degrees',
                'Paris climate agreement commitments nationally determined contributions NDCs'
            ],
            'eu_green_deal': [
                'European Green Deal climate policy',
                'European Green Deal EU climate law fit for 55 package',
                'Green Deal European climate policy EU taxonomy regulation',
            ],
            'us_ira': [
                'Inflation reduction act Biden 2022 climate legislation',
                "Inflation Reduction Act 2022 climate provisions Biden",
                "Biden Inflation Reduction Act climate legislation 2022"
            ],
            'cop_meetings': [
                'COP climate conference international negotiations UNFCCC',
                'Conference of Parties climate summit global negotiations',
                # 'COP19 Warsaw climate conference',
                # 'COP20 Lima climate conference',
                # 'COP21 Paris climate conference',
                # 'COP22 Marrakesh climate conference',
                # 'COP23 Bonn climate conference',
                # 'COP24 Katowice climate conference',
                # 'COP25 Madrid climate conference',
                # 'COP26 Glasgow climate conference',
                # 'COP27 Egypt climate conference',
                # 'COP28 Dubai climate conference',
                'international climate negotiations COP global climate action'
            ]
        }
    
    def analyze_policy_attention(self, 
                                policy_name: str,
                                start_date: Optional[str] = None,
                                end_date: Optional[str] = None,
                                frequency: str = 'quarter',
                                companies: Optional[List[str]] = None,
                                min_score: Optional[float] = None) -> Dict[str, Any]:
        """
        Comprehensive analysis of policy attention for a specific policy.
        
        Args:
            policy_name: Name of policy to analyze
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            frequency: Time frequency ('quarter', 'year', 'month')
            companies: Optional list of company tickers
            min_score: Minimum similarity threshold
            
        Returns:
            Dictionary containing time series, statistics, and events
        """
        logger.info(f"Analyzing policy attention: {policy_name}")
        
        # Get time series data
        timeseries = self.create_policy_timeseries(
            policy_name=policy_name,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            companies=companies,
            min_score=min_score
        )
        
        # Detect policy events in the time series
        events_analysis = self.detect_policy_events(policy_name, timeseries)
        
        # Calculate summary statistics
        summary_stats = self.calculate_policy_statistics(timeseries)
        
        # Firm-level analysis
        firm_analysis = self.analyze_firm_policy_exposure(
            policy_name=policy_name,
            companies=companies,
            min_score=min_score
        )
        
        return {
            'policy_name': policy_name,
            'timeseries': timeseries,
            'events_analysis': events_analysis,
            'summary_statistics': summary_stats,
            'firm_analysis': firm_analysis,
            'analysis_metadata': {
                'frequency': frequency,
                'date_range': (start_date, end_date),
                'companies_analyzed': len(companies) if companies else 'all',
                'threshold_used': min_score or (self.config.get_threshold('policies', policy_name) if self.config else 0.45)
            }
        }
    
    def create_policy_timeseries(self,
                                policy_name: str,
                                start_date: Optional[str] = None,
                                end_date: Optional[str] = None,
                                frequency: str = 'quarter',
                                companies: Optional[List[str]] = None,
                                min_score: Optional[float] = None) -> pd.DataFrame:
        """
        Create time series of policy attention.
        
        Args:
            policy_name: Name of policy ('paris_agreement', 'eu_green_deal', etc.)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            frequency: Time frequency ('quarter', 'year', 'month')
            companies: Optional list of company tickers to include
            min_score: Minimum similarity threshold
            
        Returns:
            DataFrame with policy attention time series
        """
        # Get queries and threshold
        if self.config:
            try:
                queries = self.config.get_query_list('policies', policy_name)
                threshold = min_score or self.config.get_threshold('policies', policy_name)
            except:
                queries = self.policy_queries.get(policy_name, [f"{policy_name} climate policy"])
                threshold = min_score or 0.45
        else:
            queries = self.policy_queries.get(policy_name, [f"{policy_name} climate policy"])
            threshold = min_score or 0.45
    
        logger.info(f"Using {len(queries)} queries with threshold {threshold}")
        
        # Perform semantic search
        results = self.multi_query_search(
            queries=queries,
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
            logger.warning("No results found for policy analysis")
            return pd.DataFrame()
        
        timeseries = self.aggregate_by_time(results, freq=frequency)
        
        # Add policy-specific columns
        timeseries['policy'] = policy_name
        timeseries['analysis_date'] = datetime.now().strftime('%Y-%m-%d')
        
        return timeseries
    
    def detect_policy_events(self, policy_name: str, timeseries: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect and analyze policy events in the time series.
        
        Args:
            policy_name: Name of policy
            timeseries: Time series DataFrame
            
        Returns:
            Dictionary with event analysis results
        """
        events = self.policy_events.get(policy_name, [])
        
        if not events or timeseries.empty:
            return {'events': [], 'event_analysis': 'No events or data available'}
        
        event_analysis = []
        
        for event_date, event_name in events:
            event_dt = pd.to_datetime(event_date)
            
            # Find closest periods before and after event
            before_period, after_period = self._find_event_periods(timeseries, event_dt)
            
            if before_period is not None and after_period is not None:
                # Calculate change in attention
                before_attention = before_period.get('similarity_score_mean', 0)
                after_attention = after_period.get('similarity_score_mean', 0)
                
                change = after_attention - before_attention
                percent_change = (change / before_attention * 100) if before_attention > 0 else 0
                
                event_analysis.append({
                    'event_date': event_date,
                    'event_name': event_name,
                    'before_attention': before_attention,
                    'after_attention': after_attention,
                    'absolute_change': change,
                    'percent_change': percent_change,
                    'significant_change': abs(percent_change) > 20  # 20% threshold
                })
        
        return {
            'events': event_analysis,
            'total_events_analyzed': len(event_analysis),
            'significant_events': sum(1 for e in event_analysis if e['significant_change'])
        }
    
    def calculate_policy_statistics(self, timeseries: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate summary statistics for policy attention.
        
        Args:
            timeseries: Time series DataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        if timeseries.empty:
            return {'error': 'No data available'}
        
        # Extract attention metrics
        attention_scores = timeseries['similarity_score_mean']
        mention_counts = timeseries['similarity_score_count']
        
        return {
            'total_periods': len(timeseries),
            'total_mentions': mention_counts.sum(),
            'attention_statistics': {
                'mean': attention_scores.mean(),
                'median': attention_scores.median(),
                'std': attention_scores.std(),
                'min': attention_scores.min(),
                'max': attention_scores.max(),
                'skewness': attention_scores.skew(),
                'kurtosis': attention_scores.kurtosis()
            },
            'mention_statistics': {
                'mean_per_period': mention_counts.mean(),
                'median_per_period': mention_counts.median(),
                'max_mentions_period': mention_counts.max(),
                'periods_with_mentions': (mention_counts > 0).sum(),
                'mention_frequency': (mention_counts > 0).mean()
            },
            'trend_analysis': {
                'correlation_with_time': self._calculate_time_trend(timeseries),
                'volatility': attention_scores.std() / attention_scores.mean() if attention_scores.mean() > 0 else 0
            }
        }
    
    def analyze_firm_policy_exposure(self,
                                   policy_name: str,
                                   companies: Optional[List[str]] = None,
                                   min_score: Optional[float] = None) -> pd.DataFrame:
        """
        Analyze firm-level policy exposure.
        
        Args:
            policy_name: Name of policy
            companies: Optional list of companies
            min_score: Minimum similarity threshold
            
        Returns:
            DataFrame with firm-level policy exposure metrics
        """
        # Get policy-specific results
        if self.config:
            try:
                queries = self.config.get_query_list('policies', policy_name)
                threshold = min_score or self.config.get_threshold('policies', policy_name)
            except:
                queries = self.policy_queries.get(policy_name, [f"{policy_name} climate policy"])
                threshold = min_score or 0.45
        else:
            queries = self.policy_queries.get(policy_name, [f"{policy_name} climate policy"])
            threshold = min_score or 0.45
        
        # Search for relevant snippets
        results = self.multi_query_search(queries, top_k=5000, min_score=threshold)
        
        # Apply company filter if specified
        if companies:
            results = self.filter_snippets(results, companies=companies)
        
        # Aggregate by firm
        firm_data = self.aggregate_by_firm(results)
        
        if firm_data.empty:
            return pd.DataFrame()
        
        # Add policy-specific metrics
        firm_data['policy'] = policy_name
        firm_data['policy_exposure_score'] = firm_data['avg_score']
        firm_data['policy_mention_intensity'] = firm_data['total_mentions']
        
        # Calculate relative rankings
        firm_data['exposure_rank'] = firm_data['policy_exposure_score'].rank(ascending=False)
        firm_data['mention_rank'] = firm_data['policy_mention_intensity'].rank(ascending=False)
        
        # Add percentiles
        firm_data['exposure_percentile'] = firm_data['policy_exposure_score'].rank(pct=True)
        firm_data['mention_percentile'] = firm_data['policy_mention_intensity'].rank(pct=True)
        
        return firm_data.sort_values('policy_exposure_score', ascending=False)
    
    def compare_policies(self,
                        policy_names: List[str],
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None,
                        frequency: str = 'quarter') -> Dict[str, Any]:
        """
        Compare attention across multiple policies.
        
        Args:
            policy_names: List of policy names to compare
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            frequency: Time frequency
            
        Returns:
            Dictionary with comparative analysis
        """
        policy_data = {}
        
        for policy in policy_names:
            try:
                analysis = self.analyze_policy_attention(
                    policy_name=policy,
                    start_date=start_date,
                    end_date=end_date,
                    frequency=frequency
                )
                policy_data[policy] = analysis
                logger.info(f"✅ {policy}: Analysis complete")
            except Exception as e:
                logger.error(f"❌ {policy}: {e}")
                continue
        
        # Create comparison metrics
        comparison_stats = self._create_policy_comparison_stats(policy_data)
        
        return {
            'policies_analyzed': list(policy_data.keys()),
            'individual_analyses': policy_data,
            'comparative_statistics': comparison_stats,
            'analysis_metadata': {
                'comparison_date': datetime.now().strftime('%Y-%m-%d'),
                'frequency': frequency,
                'date_range': (start_date, end_date)
            }
        }
    
    def export_policy_analysis(self, 
                             analysis_results: Dict[str, Any],
                             output_dir: Path,
                             formats: List[str] = ['csv', 'json']) -> Dict[str, Path]:
        """
        Export policy analysis results to files.
        
        Args:
            analysis_results: Results from analyze_policy_attention()
            output_dir: Output directory
            formats: List of export formats ('csv', 'json', 'parquet')
            
        Returns:
            Dictionary mapping format to output file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        policy_name = analysis_results['policy_name']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        output_files = {}
        
        for format_type in formats:
            if format_type == 'csv':
                # Export time series
                ts_file = output_dir / f"{policy_name}_timeseries_{timestamp}.csv"
                analysis_results['timeseries'].to_csv(ts_file, index=False)
                output_files['timeseries_csv'] = ts_file
                
                # Export firm analysis
                if not analysis_results['firm_analysis'].empty:
                    firm_file = output_dir / f"{policy_name}_firms_{timestamp}.csv"
                    analysis_results['firm_analysis'].to_csv(firm_file, index=False)
                    output_files['firms_csv'] = firm_file
            
            elif format_type == 'json':
                # Export complete analysis
                json_file = output_dir / f"{policy_name}_analysis_{timestamp}.json"
                
                # Convert DataFrames to records for JSON serialization
                json_data = analysis_results.copy()
                json_data['timeseries'] = analysis_results['timeseries'].to_dict('records')
                json_data['firm_analysis'] = analysis_results['firm_analysis'].to_dict('records')
                
                import json
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, default=str)
                output_files['analysis_json'] = json_file
        
        logger.info(f"✅ Policy analysis exported: {len(output_files)} files")
        return output_files
    
    # Helper methods
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
    
    def _find_event_periods(self, timeseries: pd.DataFrame, event_date: pd.Timestamp) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
        """Find periods before and after an event."""
        if timeseries.empty:
            return None, None
        
        # Convert period to datetime for comparison
        timeseries_copy = timeseries.copy()
        try:
            timeseries_copy['period_dt'] = pd.to_datetime(timeseries_copy['period'])
        except:
            return None, None
        
        before_periods = timeseries_copy[timeseries_copy['period_dt'] <= event_date]
        after_periods = timeseries_copy[timeseries_copy['period_dt'] > event_date]
        
        before_period = before_periods.iloc[-1] if not before_periods.empty else None
        after_period = after_periods.iloc[0] if not after_periods.empty else None
        
        return before_period, after_period
    
    def _calculate_time_trend(self, timeseries: pd.DataFrame) -> float:
        """Calculate correlation between attention and time."""
        if len(timeseries) < 2:
            return 0.0
        
        try:
            time_numeric = range(len(timeseries))
            attention = timeseries['similarity_score_mean']
            correlation = np.corrcoef(time_numeric, attention)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _create_policy_comparison_stats(self, policy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create comparative statistics across policies."""
        if not policy_data:
            return {}
        
        comparison_stats = {}
        
        # Compare total mentions
        total_mentions = {policy: data['summary_statistics']['total_mentions'] 
                         for policy, data in policy_data.items()}
        comparison_stats['mention_comparison'] = total_mentions
        
        # Compare average attention
        avg_attention = {policy: data['summary_statistics']['attention_statistics']['mean'] 
                        for policy, data in policy_data.items()}
        comparison_stats['attention_comparison'] = avg_attention
        
        # Find most/least discussed policies
        comparison_stats['most_discussed'] = max(total_mentions.items(), key=lambda x: x[1])
        comparison_stats['least_discussed'] = min(total_mentions.items(), key=lambda x: x[1])
        
        return comparison_stats


# Convenience functions for direct usage
def analyze_policy(index_path: Path, 
                  policy_name: str,
                  output_dir: Optional[Path] = None,
                  **kwargs) -> Dict[str, Any]:
    """
    Convenience function for policy analysis.
    
    Args:
        index_path: Path to semantic index
        policy_name: Policy to analyze
        output_dir: Optional output directory for exports
        **kwargs: Additional arguments for analysis
        
    Returns:
        Analysis results dictionary
    """
    analyzer = PolicyAnalyzer(index_path)
    results = analyzer.analyze_policy_attention(policy_name, **kwargs)
    
    if output_dir:
        analyzer.export_policy_analysis(results, output_dir)
    
    return results


def compare_policies(index_path: Path,
                    policy_names: List[str],
                    output_dir: Optional[Path] = None,
                    **kwargs) -> Dict[str, Any]:
    """
    Convenience function for policy comparison.
    
    Args:
        index_path: Path to semantic index
        policy_names: List of policies to compare
        output_dir: Optional output directory for exports
        **kwargs: Additional arguments for analysis
        
    Returns:
        Comparison results dictionary
    """
    analyzer = PolicyAnalyzer(index_path)
    results = analyzer.compare_policies(policy_names, **kwargs)
    
    if output_dir:
        # Export individual analyses
        for policy, analysis in results['individual_analyses'].items():
            analyzer.export_policy_analysis(analysis, output_dir / policy)
    
    return results