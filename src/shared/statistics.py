#!/usr/bin/env python3
"""
Index Statistics Utility for ECC Data Generation Pipeline

Generates comprehensive statistics and analytics for semantic indexes.
Provides insights into data coverage, quality, and performance metrics.

Author: Marleen de Jonge
Date: 2025
"""

import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..analysis_pipeline.base_searcher import BaseSemanticSearcher


class IndexStatistics:
    """
    Utility class for generating comprehensive index statistics.
    Analyzes loaded semantic indexes for coverage, quality, and performance insights.
    """
    
    def __init__(self, searcher: 'BaseSemanticSearcher'):
        self.searcher = searcher
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def generate(self) -> Dict[str, Any]:
        """
        Generate comprehensive statistics about the loaded index.
        
        Returns:
            Dict containing detailed index statistics including:
            - Basic info (counts, model, dimensions)
            - Coverage analysis (companies, markets, temporal)
            - Data quality metrics
            - Performance statistics
            - Validation status
            
        Raises:
            RuntimeError: If index is not loaded
        """
        if not getattr(self.searcher, '_index_loaded', False):
            raise RuntimeError("Index not loaded. Call load_index() first.")
        
        try:
            stats = {}
            
            # Generate all statistics components
            stats['basic_info'] = self._generate_basic_info()
            stats['coverage'] = self._generate_coverage_stats()
            stats['temporal_coverage'] = self._generate_temporal_stats()
            stats['data_quality'] = self._generate_quality_stats()
            stats['performance'] = self._generate_performance_stats()
            stats['validation'] = self._generate_validation_stats()
            stats['configuration'] = self._generate_config_stats()
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error generating index statistics: {e}")
            return {
                'error': str(e),
                'basic_info': {
                    'total_snippets': len(self.searcher.snippets) if self.searcher.snippets else 0,
                    'index_loaded': getattr(self.searcher, '_index_loaded', False)
                }
            }
    
    def _generate_basic_info(self) -> Dict[str, Any]:
        """Generate basic index information."""
        return {
            'total_snippets': len(self.searcher.snippets),
            'model_name': getattr(self.searcher, 'model_name', 'Unknown'),
            'embedding_dimension': self.searcher.metadata.get('embedding_dimension'),
            'index_creation_date': self.searcher.metadata.get('creation_date'),
            'index_path': str(self.searcher.index_path) if self.searcher.index_path else None,
            'faiss_vectors': self.searcher.faiss_index.ntotal if self.searcher.faiss_index else 0
        }
    
    def _generate_coverage_stats(self) -> Dict[str, Any]:
        """Generate coverage statistics for companies and markets."""
        companies = {}
        stock_indices = defaultdict(int)
        years = defaultdict(int)
        quarters = defaultdict(int)
        
        for snippet in self.searcher.snippets:
            # Company counts
            companies[snippet.ticker] = companies.get(snippet.ticker, 0) + 1
            
            # Stock index distribution
            if snippet.stock_index:
                stock_indices[snippet.stock_index] += 1
            
            # Temporal distribution
            if snippet.year:
                years[snippet.year] += 1
            if snippet.quarter:
                quarters[snippet.quarter] += 1
        
        return {
            'unique_companies': len(companies),
            'companies_by_index': dict(stock_indices),
            'top_10_companies': dict(sorted(companies.items(), key=lambda x: x[1], reverse=True)[:10]),
            'year_distribution': dict(sorted(years.items())),
            'quarter_distribution': dict(quarters),
            'total_company_observations': sum(companies.values()),
            'average_snippets_per_company': np.mean(list(companies.values())) if companies else 0
        }
    
    def _generate_temporal_stats(self) -> Dict[str, Any]:
        """Generate temporal coverage statistics."""
        valid_dates = []
        invalid_dates = 0
        
        for snippet in self.searcher.snippets:
            try:
                if snippet.date:
                    valid_dates.append(pd.to_datetime(snippet.date))
                else:
                    invalid_dates += 1
            except:
                invalid_dates += 1
        
        if valid_dates:
            date_range_days = (max(valid_dates) - min(valid_dates)).days
            return {
                'start_date': min(valid_dates).isoformat(),
                'end_date': max(valid_dates).isoformat(),
                'date_range_years': round(date_range_days / 365.25, 2),
                'date_range_days': date_range_days,
                'valid_dates_count': len(valid_dates),
                'invalid_dates_count': invalid_dates,
                'coverage_percentage': round(len(valid_dates) / len(self.searcher.snippets) * 100, 2)
            }
        else:
            return {
                'start_date': None,
                'end_date': None,
                'date_range_years': 0,
                'date_range_days': 0,
                'valid_dates_count': 0,
                'invalid_dates_count': len(self.searcher.snippets),
                'coverage_percentage': 0
            }
    
    def _generate_quality_stats(self) -> Dict[str, Any]:
        """Generate data quality metrics."""
        sentence_ratios = []
        sentence_counts = []
        text_lengths = []
        empty_texts = 0
        missing_metadata = 0
        
        for snippet in self.searcher.snippets:
            # Text quality
            if not snippet.text or not snippet.text.strip():
                empty_texts += 1
            else:
                text_lengths.append(len(snippet.text))
            
            # Metadata completeness
            if not snippet.company_name or not snippet.ticker or not snippet.date:
                missing_metadata += 1
            
            # Sentence metrics
            if snippet.climate_sentence_ratio is not None:
                sentence_ratios.append(snippet.climate_sentence_ratio)
            if snippet.sentence_count is not None:
                sentence_counts.append(snippet.sentence_count)
        
        quality_stats = {
            'snippets_with_ratios': len(sentence_ratios),
            'snippets_with_sentence_counts': len(sentence_counts),
            'empty_texts_count': empty_texts,
            'missing_metadata_count': missing_metadata,
            'data_completeness_percentage': round((len(self.searcher.snippets) - missing_metadata) / len(self.searcher.snippets) * 100, 2)
        }
        
        # Text length statistics
        if text_lengths:
            quality_stats['text_length_stats'] = {
                'mean': round(np.mean(text_lengths), 1),
                'median': round(np.median(text_lengths), 1),
                'std': round(np.std(text_lengths), 1),
                'min': min(text_lengths),
                'max': max(text_lengths)
            }
        
        # Climate ratio statistics
        if sentence_ratios:
            quality_stats['climate_ratio_stats'] = {
                'mean': round(np.mean(sentence_ratios), 4),
                'median': round(np.median(sentence_ratios), 4),
                'std': round(np.std(sentence_ratios), 4),
                'min': round(min(sentence_ratios), 4),
                'max': round(max(sentence_ratios), 4)
            }
        
        return quality_stats
    
    def _generate_performance_stats(self) -> Dict[str, Any]:
        """Generate performance statistics."""
        # Get search statistics if available
        search_stats = getattr(self.searcher, '_search_stats', {})
        
        performance_stats = {
            'total_searches': search_stats.get('total_searches', 0),
            'total_search_time': round(search_stats.get('total_search_time', 0.0), 3),
            'average_search_time': round(search_stats.get('average_search_time', 0.0), 3)
        }
        
        # Add device information
        device = getattr(self.searcher, 'device', 'unknown')
        performance_stats['device'] = device
        
        # Calculate index size metrics
        if self.searcher.faiss_index:
            performance_stats['faiss_index_size'] = {
                'total_vectors': self.searcher.faiss_index.ntotal,
                'dimension': self.searcher.faiss_index.d,
                'is_trained': self.searcher.faiss_index.is_trained
            }
        
        return performance_stats
    
    def _generate_validation_stats(self) -> Dict[str, Any]:
        """Generate validation status information."""
        return {
            'index_loaded': getattr(self.searcher, '_index_loaded', False),
            'index_validated': getattr(self.searcher, '_index_validated', False),
            'validation_date': getattr(self.searcher, '_validation_date', None),
            'last_validation_status': 'Passed' if getattr(self.searcher, '_index_validated', False) else 'Not validated'
        }
    
    def _generate_config_stats(self) -> Dict[str, Any]:
        """Generate configuration information."""
        config_stats = {
            'config_loaded': hasattr(self.searcher, 'config') and self.searcher.config is not None
        }
        
        if config_stats['config_loaded']:
            try:
                if hasattr(self.searcher.config, 'get_available_analyses'):
                    available_analyses = self.searcher.config.get_available_analyses()
                    config_stats['available_analyses'] = available_analyses
                else:
                    config_stats['available_analyses'] = 'Unknown - config object missing expected methods'
            except Exception as e:
                config_stats['config_error'] = str(e)
        else:
            config_stats['available_analyses'] = None
        
        return config_stats
    
    def generate_summary_report(self) -> str:
        """Generate a human-readable summary report."""
        stats = self.generate()
        
        report_lines = [
            "📊 SEMANTIC INDEX STATISTICS REPORT",
            "=" * 50,
            "",
            f"📈 Basic Information:",
            f"   Total Snippets: {stats['basic_info']['total_snippets']:,}",
            f"   Model: {stats['basic_info']['model_name']}",
            f"   Embedding Dimension: {stats['basic_info']['embedding_dimension']}",
            f"   FAISS Vectors: {stats['basic_info']['faiss_vectors']:,}",
            "",
            f"🏢 Coverage:",
            f"   Unique Companies: {stats['coverage']['unique_companies']}",
            f"   Stock Indices: {', '.join(stats['coverage']['companies_by_index'].keys())}",
            f"   Year Range: {min(stats['coverage']['year_distribution'].keys()) if stats['coverage']['year_distribution'] else 'N/A'} - {max(stats['coverage']['year_distribution'].keys()) if stats['coverage']['year_distribution'] else 'N/A'}",
            "",
            f"📅 Temporal Coverage:",
            f"   Date Range: {stats['temporal_coverage']['start_date']} to {stats['temporal_coverage']['end_date']}",
            f"   Coverage: {stats['temporal_coverage']['coverage_percentage']:.1f}% valid dates",
            f"   Time Span: {stats['temporal_coverage']['date_range_years']} years",
            "",
            f"✅ Data Quality:",
            f"   Data Completeness: {stats['data_quality']['data_completeness_percentage']:.1f}%",
            f"   Empty Texts: {stats['data_quality']['empty_texts_count']}",
            f"   Missing Metadata: {stats['data_quality']['missing_metadata_count']}",
            "",
            f"⚡ Performance:",
            f"   Device: {stats['performance']['device']}",
            f"   Total Searches: