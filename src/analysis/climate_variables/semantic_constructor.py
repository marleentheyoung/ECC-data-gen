"""
Semantic climate exposure variable construction.

This module constructs firm-level climate change exposure variables using semantic search
over pre-extracted climate-relevant snippets from earnings calls.

Author: Marleen de Jonge
Date: 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import logging

from .climate_searcher import ClimateSemanticSearcher, ClimateSnippet

logger = logging.getLogger(__name__)


class SemanticClimateVariableConstructor:
    """
    Construct firm-level climate change exposure variables using semantic search.
    
    This class takes pre-extracted climate snippets and constructs standardized
    firm-quarter panel variables for econometric analysis.
    """
    
    def __init__(self, searcher: ClimateSemanticSearcher, 
                 output_dir: str = "outputs/climate_variables"):
        self.searcher = searcher
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Semantic climate topics with sophisticated queries
        self.climate_topics = {
            'opportunities': {
                'queries': [
                    'renewable energy investments clean technology opportunities',
                    'green innovation sustainable business models',
                    'energy transition investment opportunities',
                    'clean energy growth potential',
                    'sustainability competitive advantage'
                ],
                'threshold': 0.40,
                'description': 'Climate-related business opportunities and investments'
            },
            'regulation': {
                'queries': [
                    'climate policy environmental regulation compliance',
                    'carbon pricing emissions trading requirements',
                    'Paris Agreement regulatory compliance',
                    'environmental standards regulatory changes',
                    'climate disclosure requirements'
                ],
                'threshold': 0.40,
                'description': 'Climate regulation and policy impacts'
            },
            'physical_risk': {
                'queries': [
                    'extreme weather climate physical risk',
                    'supply chain disruption weather events',
                    'flooding drought natural disasters',
                    'climate adaptation resilience',
                    'weather operational disruption'
                ],
                'threshold': 0.40,
                'description': 'Physical climate risks and impacts'
            },
            'transition_risk': {
                'queries': [
                    'stranded assets carbon intensive business',
                    'technology disruption energy transition',
                    'carbon tax transition costs',
                    'fossil fuel asset impairment',
                    'business model transition risk'
                ],
                'threshold': 0.40,
                'description': 'Climate transition risks and business model changes'
            },
            'disclosure': {
                'queries': [
                    'climate risk disclosure ESG reporting',
                    'TCFD climate scenario analysis',
                    'sustainability reporting framework',
                    'carbon footprint disclosure',
                    'climate governance reporting'
                ],
                'threshold': 0.40,
                'description': 'Climate disclosure and reporting activities'
            }
        }
        
        # Policy-specific topics
        self.policy_topics = {
            'paris_agreement': {
                'queries': [
                    'Paris Agreement COP21 international climate',
                    'NDCs national climate commitments',
                    'global climate accord Paris'
                ],
                'threshold': 0.45,
                'description': 'Paris Agreement and international climate commitments'
            },
            'carbon_pricing': {
                'queries': [
                    'carbon tax carbon pricing mechanism',
                    'internal carbon price shadow price',
                    'carbon credits emissions trading'
                ],
                'threshold': 0.45,
                'description': 'Carbon pricing mechanisms and impacts'
            },
            'eu_green_deal': {
                'queries': [
                    'European Green Deal EU climate policy',
                    'EU taxonomy sustainable finance',
                    'Green Deal climate neutrality'
                ],
                'threshold': 0.45,
                'description': 'EU Green Deal and related policies'
            },
            'inflation_reduction_act': {
                'queries': [
                    'Inflation Reduction Act IRA climate',
                    'US climate investment tax credits',
                    'clean energy incentives IRA'
                ],
                'threshold': 0.45,
                'description': 'US Inflation Reduction Act climate provisions'
            }
        }
    
    def construct_climate_variables(self, start_year: int = 2009, 
                                  end_year: int = 2024) -> pd.DataFrame:
        """
        Main pipeline to construct semantic climate exposure variables.
        
        Args:
            start_year: Start year for panel
            end_year: End year for panel
            
        Returns:
            DataFrame with firm-quarter climate exposure variables
        """
        logger.info("🧠 Starting semantic climate variable construction...")
        
        # Create firm-quarter panel structure
        panel_data = self._create_panel_structure(start_year, end_year)
        logger.info(f"📊 Created panel with {len(panel_data)} firm-quarter observations")
        
        # Calculate semantic exposure measures
        panel_data = self._calculate_semantic_exposure(panel_data)
        logger.info("🎯 Calculated semantic exposure measures")
        
        # Calculate policy-specific measures
        panel_data = self._calculate_policy_measures(panel_data)
        logger.info("📋 Calculated policy-specific measures")
        
        # Calculate sentiment measures
        panel_data = self._calculate_sentiment_measures(panel_data)
        logger.info("💭 Calculated sentiment measures")
        
        # Calculate temporal evolution measures
        panel_data = self._calculate_evolution_measures(panel_data)
        logger.info("📈 Calculated evolution measures")
        
        # Convert to DataFrame
        df = pd.DataFrame(panel_data)
        
        # Save datasets
        self._save_datasets(df)
        logger.info("💾 Saved datasets")
        
        logger.info("🎉 Semantic variable construction completed!")
        return df
    
    def _create_panel_structure(self, start_year: int, end_year: int) -> List[Dict]:
        """Create firm-quarter panel structure."""
        logger.info("Creating firm-quarter panel structure...")
        
        # Get unique firms from snippets
        firms = list(set(snippet.ticker for snippet in self.searcher.snippets if snippet.ticker))
        
        # Create date range
        quarters = pd.date_range(start=f'{start_year}-01-01', end=f'{end_year}-12-31', freq='QE')
        
        panel_data = []
        for firm in tqdm(firms, desc="Creating panel structure"):
            for quarter in quarters:
                year, q = quarter.year, quarter.quarter
                firm_snippets = self._get_firm_quarter_snippets(firm, year, q)
                
                panel_data.append({
                    'ticker': firm,
                    'year': year,
                    'quarter': q,
                    'date': quarter,
                    'snippets': firm_snippets,
                    'has_climate_content': len(firm_snippets) > 0,
                    'total_climate_snippets': len(firm_snippets),
                    # Add sentence ratio information if available
                    **self._extract_sentence_ratio_info(firm_snippets)
                })
        
        return panel_data
    
    def _extract_sentence_ratio_info(self, firm_snippets: List[ClimateSnippet]) -> Dict[str, Any]:
        """Extract sentence ratio information from firm snippets."""
        if not firm_snippets:
            return {
                'climate_sentence_count': 0,
                'total_sentences_in_call': None,
                'climate_sentence_ratio': None,
                'normalized_climate_attention': None
            }
        
        # Get sentence ratio info from first snippet (should be same for all snippets from same transcript)
        first_snippet = firm_snippets[0]
        
        # Sum up individual snippet sentence counts
        total_climate_sentences = sum(
            snippet.sentence_count for snippet in firm_snippets 
            if snippet.sentence_count is not None
        )
        
        # Use the total call sentences from the enhanced data
        total_call_sentences = first_snippet.total_sentences_in_call
        
        # Calculate normalized attention (same as sentence ratio for this firm-quarter)
        if total_call_sentences and total_call_sentences > 0:
            normalized_attention = total_climate_sentences / total_call_sentences
        else:
            normalized_attention = first_snippet.climate_sentence_ratio
        
        return {
            'climate_sentence_count': total_climate_sentences if total_climate_sentences > 0 else first_snippet.climate_sentence_count,
            'total_sentences_in_call': total_call_sentences,
            'climate_sentence_ratio': first_snippet.climate_sentence_ratio,
            'normalized_climate_attention': normalized_attention
        }
    
    def _get_firm_quarter_snippets(self, ticker: str, year: int, quarter: int) -> List[ClimateSnippet]:
        """Get all snippets for a firm in a specific quarter."""
        matching_snippets = []
        
        for snippet in self.searcher.snippets:
            if (snippet.ticker == ticker and 
                snippet.year == year and 
                self._parse_quarter(snippet.quarter) == quarter):
                matching_snippets.append(snippet)
        
        return matching_snippets
    
    def _parse_quarter(self, quarter_str: str) -> Optional[int]:
        """Parse quarter string to integer."""
        if not quarter_str:
            return None
        
        quarter_str = str(quarter_str).upper().strip()
        if quarter_str.startswith('Q'):
            quarter_str = quarter_str[1:]
        
        try:
            return int(quarter_str)
        except ValueError:
            return None
    
    def _calculate_semantic_exposure(self, panel_data: List[Dict]) -> List[Dict]:
        """Calculate semantic exposure measures for each topic."""
        logger.info("🔍 Calculating semantic exposure measures...")
        
        # Process only rows with earnings calls
        rows_with_calls = [row for row in panel_data if row['has_climate_content']]
        
        for row in tqdm(rows_with_calls, desc="Processing semantic exposure"):
            firm_snippets = row['snippets']
            total_snippets = len(firm_snippets)
            
            for topic_name, topic_config in self.climate_topics.items():
                # Count snippets that match any of the topic's queries
                matching_snippets = self._find_matching_snippets(
                    firm_snippets, topic_config['queries'], topic_config['threshold']
                )
                
                # Calculate measures
                exposure_ratio = len(matching_snippets) / total_snippets if total_snippets > 0 else 0.0
                avg_score = np.mean([s.relevance_score for s in matching_snippets]) if matching_snippets else 0.0
                
                # Store measures
                row[f'semantic_{topic_name}_exposure'] = exposure_ratio
                row[f'semantic_{topic_name}_count'] = len(matching_snippets)
                row[f'semantic_{topic_name}_avg_score'] = avg_score
        
        # Set zero values for rows without earnings calls
        for row in panel_data:
            if not row['has_climate_content']:
                for topic_name in self.climate_topics:
                    row[f'semantic_{topic_name}_exposure'] = 0.0
                    row[f'semantic_{topic_name}_count'] = 0
                    row[f'semantic_{topic_name}_avg_score'] = 0.0
        
        return panel_data
    
    def _calculate_policy_measures(self, panel_data: List[Dict]) -> List[Dict]:
        """Calculate policy-specific attention measures."""
        logger.info("📋 Calculating policy-specific measures...")
        
        rows_with_calls = [row for row in panel_data if row['has_climate_content']]
        
        for row in tqdm(rows_with_calls, desc="Processing policy measures"):
            firm_snippets = row['snippets']
            total_snippets = len(firm_snippets)
            
            for policy_name, policy_config in self.policy_topics.items():
                matching_snippets = self._find_matching_snippets(
                    firm_snippets, policy_config['queries'], policy_config['threshold']
                )
                
                attention_ratio = len(matching_snippets) / total_snippets if total_snippets > 0 else 0.0
                avg_score = np.mean([s.relevance_score for s in matching_snippets]) if matching_snippets else 0.0
                
                row[f'policy_{policy_name}_attention'] = attention_ratio
                row[f'policy_{policy_name}_mentions'] = len(matching_snippets)
                row[f'policy_{policy_name}_avg_score'] = avg_score
        
        # Set zero values for rows without earnings calls
        for row in panel_data:
            if not row['has_climate_content']:
                for policy_name in self.policy_topics:
                    row[f'policy_{policy_name}_attention'] = 0.0
                    row[f'policy_{policy_name}_mentions'] = 0
                    row[f'policy_{policy_name}_avg_score'] = 0.0
        
        return panel_data
    
    def _find_matching_snippets(self, snippets: List[ClimateSnippet], 
                               queries: List[str], threshold: float) -> List[ClimateSnippet]:
        """Find snippets that match any of the given queries using semantic search."""
        if not self.searcher.model or not self.searcher.faiss_index:
            # Fallback to keyword matching if semantic search not available
            return self._keyword_matching_fallback(snippets, queries)
        
        matching_snippets = []
        snippet_texts = [s.text for s in snippets]
        
        if not snippet_texts:
            return matching_snippets
        
        # Create embeddings for snippet texts
        snippet_embeddings = self.searcher.model.encode(
            snippet_texts, normalize_embeddings=True, convert_to_numpy=True
        )
        
        # Check each query
        for query in queries:
            query_embedding = self.searcher.model.encode([query], normalize_embeddings=True)
            
            # Calculate similarities
            similarities = np.dot(query_embedding, snippet_embeddings.T)[0]
            
            # Find matching snippets above threshold
            for i, similarity in enumerate(similarities):
                if similarity >= threshold:
                    snippet = snippets[i]
                    snippet.relevance_score = float(similarity)
                    if snippet not in matching_snippets:
                        matching_snippets.append(snippet)
        
        return matching_snippets
    
    def _keyword_matching_fallback(self, snippets: List[ClimateSnippet], 
                                  queries: List[str]) -> List[ClimateSnippet]:
        """Fallback keyword matching when semantic search is not available."""
        matching_snippets = []
        
        # Extract keywords from queries
        all_keywords = []
        for query in queries:
            keywords = query.lower().split()
            all_keywords.extend(keywords)
        
        # Remove duplicates and common words
        unique_keywords = list(set(all_keywords))
        stopwords = {'and', 'or', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
        keywords = [kw for kw in unique_keywords if kw not in stopwords and len(kw) > 2]
        
        # Find matching snippets
        for snippet in snippets:
            text_lower = snippet.text.lower()
            if any(keyword in text_lower for keyword in keywords):
                snippet.relevance_score = 0.5  # Default score for keyword matching
                matching_snippets.append(snippet)
        
        return matching_snippets
    
    def _calculate_sentiment_measures(self, panel_data: List[Dict]) -> List[Dict]:
        """Calculate sentiment measures using existing climate_sentiment labels."""
        logger.info("💭 Calculating sentiment measures...")
        
        rows_with_calls = [row for row in panel_data if row['has_climate_content']]
        
        for row in tqdm(rows_with_calls, desc="Processing sentiment measures"):
            firm_snippets = row['snippets']
            
            if not firm_snippets:
                continue
            
            # Count sentiments in climate-relevant snippets
            sentiment_counts = {'opportunity': 0, 'risk': 0, 'neutral': 0}
            
            for snippet in firm_snippets:
                sentiment = getattr(snippet, 'climate_sentiment', 'neutral')
                if sentiment in sentiment_counts:
                    sentiment_counts[sentiment] += 1
            
            total_climate_snippets = len(firm_snippets)
            
            if total_climate_snippets > 0:
                row.update({
                    'semantic_climate_sentiment_opportunity': sentiment_counts['opportunity'] / total_climate_snippets,
                    'semantic_climate_sentiment_risk': sentiment_counts['risk'] / total_climate_snippets,
                    'semantic_climate_sentiment_neutral': sentiment_counts['neutral'] / total_climate_snippets,
                    'semantic_sentiment_net': (sentiment_counts['opportunity'] - sentiment_counts['risk']) / total_climate_snippets
                })
            else:
                row.update({
                    'semantic_climate_sentiment_opportunity': 0.0,
                    'semantic_climate_sentiment_risk': 0.0,
                    'semantic_climate_sentiment_neutral': 0.0,
                    'semantic_sentiment_net': 0.0
                })
        
        # Set zero values for rows without earnings calls
        for row in panel_data:
            if not row['has_climate_content']:
                row.update({
                    'semantic_climate_sentiment_opportunity': 0.0,
                    'semantic_climate_sentiment_risk': 0.0,
                    'semantic_climate_sentiment_neutral': 0.0,
                    'semantic_sentiment_net': 0.0
                })
        
        return panel_data
    
    def _calculate_evolution_measures(self, panel_data: List[Dict]) -> List[Dict]:
        """Calculate how semantic climate attention evolves over time."""
        logger.info("📈 Calculating evolution measures...")
        
        df = pd.DataFrame(panel_data)
        df = df.sort_values(['ticker', 'year', 'quarter'])
        
        # Calculate changes in semantic exposure measures
        semantic_vars = [col for col in df.columns if col.startswith('semantic_') and col.endswith('_exposure')]
        
        for var in semantic_vars:
            # Quarter-over-quarter changes
            df[f'{var}_qoq'] = df.groupby('ticker')[var].pct_change()
            
            # Year-over-year changes  
            df[f'{var}_yoy'] = df.groupby('ticker')[var].pct_change(periods=4)
            
            # Rolling averages
            df[f'{var}_ma4q'] = df.groupby('ticker')[var].rolling(4).mean().reset_index(0, drop=True)
            
            # Semantic attention trend (slope over past 4 quarters)
            def calculate_trend(series):
                if len(series) < 4 or series.isna().all():
                    return np.nan
                x = np.arange(len(series))
                valid_mask = ~series.isna()
                if valid_mask.sum() < 3:
                    return np.nan
                return np.polyfit(x[valid_mask], series[valid_mask], 1)[0]
            
            df[f'{var}_trend_4q'] = df.groupby('ticker')[var].rolling(4).apply(calculate_trend).reset_index(0, drop=True)
        
        return df.to_dict('records')
    
    def _save_datasets(self, df: pd.DataFrame) -> None:
        """Save datasets with methodology documentation."""
        logger.info("💾 Saving datasets...")
        
        # Drop snippets column for saving (too large)
        analysis_df = df.drop('snippets', axis=1, errors='ignore')
        
        # 1. Main semantic panel dataset
        analysis_df.to_csv(self.output_dir / 'semantic_climate_panel.csv', index=False)
        analysis_df.to_parquet(self.output_dir / 'semantic_climate_panel.parquet')
        
        # 2. Methodology documentation
        methodology = {
            'approach': 'semantic_search_on_climate_snippets',
            'data_source': 'pre_extracted_climate_snippets',
            'climate_topics': self.climate_topics,
            'policy_topics': self.policy_topics,
            'advantages': [
                'Uses pre-filtered climate-relevant content',
                'Captures semantic meaning rather than just keywords',
                'Context-aware topic identification',
                'Adjustable relevance thresholds per topic',
                'Handles synonyms and related concepts automatically'
            ],
            'data_structure': 'firm_quarter_panel',
            'quality_metrics': [
                'Average relevance scores',
                'Topic coverage',
                'Cross-temporal consistency',
                'Sentiment distribution'
            ]
        }
        
        with open(self.output_dir / 'methodology.json', 'w') as f:
            json.dump(methodology, f, indent=2, default=str)
        
        # 3. Summary statistics
        self._create_summary_statistics(analysis_df)
        
        # 4. Quality assurance report
        self._create_quality_report(analysis_df)
        
        logger.info(f"📁 Datasets saved to {self.output_dir}")
    
    def _create_summary_statistics(self, df: pd.DataFrame) -> None:
        """Create summary statistics for the semantic variables."""
        
        # Get semantic exposure variables
        exposure_vars = [col for col in df.columns if col.startswith('semantic_') and col.endswith('_exposure')]
        policy_vars = [col for col in df.columns if col.startswith('policy_') and col.endswith('_attention')]
        
        summary_stats = {}
        
        # Summary for exposure variables
        for var in exposure_vars:
            topic = var.replace('semantic_', '').replace('_exposure', '')
            summary_stats[f'{topic}_exposure'] = {
                'mean': float(df[var].mean()),
                'std': float(df[var].std()),
                'coverage': float((df[var] > 0).mean()),  # % of firm-quarters with any mentions
                'p50': float(df[var].median()),
                'p90': float(df[var].quantile(0.9)),
                'p99': float(df[var].quantile(0.99))
            }
        
        # Summary for policy variables
        for var in policy_vars:
            policy = var.replace('policy_', '').replace('_attention', '')
            summary_stats[f'{policy}_policy'] = {
                'mean': float(df[var].mean()),
                'std': float(df[var].std()),
                'coverage': float((df[var] > 0).mean()),
                'p50': float(df[var].median()),
                'p90': float(df[var].quantile(0.9)),
                'p99': float(df[var].quantile(0.99))
            }
        
        # Overall statistics
        summary_stats['panel_overview'] = {
            'total_firm_quarters': len(df),
            'firm_quarters_with_calls': int((df['has_earnings_call'] == True).sum()),
            'unique_firms': int(df['ticker'].nunique()),
            'year_range': [int(df['year'].min()), int(df['year'].max())],
            'total_climate_snippets': int(df['total_climate_snippets'].sum())
        }
        
        # Save summary statistics
        with open(self.output_dir / 'summary_statistics.json', 'w') as f:
            json.dump(summary_stats, f, indent=2)
    
    def _create_quality_report(self, df: pd.DataFrame) -> None:
        """Create comprehensive quality report."""
        
        quality_metrics = {
            'data_coverage': {
                'total_firm_quarters': len(df),
                'firm_quarters_with_calls': int((df['has_climate_content'] == True).sum()),
                'coverage_rate': float((df['has_climate_content'] == True).mean()),
                'firms_with_any_climate_content': int((df['total_climate_snippets'] > 0).sum())
            },
            'temporal_distribution': {
                'quarters_covered': df.groupby(['year', 'quarter']).size().describe().to_dict(),
                'firms_covered': df.groupby('ticker').size().describe().to_dict(),
                'climate_content_by_year': df.groupby('year')['total_climate_snippets'].sum().to_dict()
            },
            'content_quality': {
                'avg_climate_snippets_per_call': float(df[df['has_earnings_call']]['total_climate_snippets'].mean()),
                'firms_with_high_climate_content': int((df['total_climate_snippets'] >= 10).sum()),
                'zero_climate_content_rate': float((df['total_climate_snippets'] == 0).mean())
            }
        }
        
        # Add topic-specific quality metrics
        for topic_name in self.climate_topics:
            exposure_var = f'semantic_{topic_name}_exposure'
            count_var = f'semantic_{topic_name}_count'
            
            if exposure_var in df.columns:
                quality_metrics[f'{topic_name}_topic_quality'] = {
                    'firms_with_content': int((df[exposure_var] > 0).sum()),
                    'avg_exposure': float(df[exposure_var].mean()),
                    'max_exposure': float(df[exposure_var].max()),
                    'total_mentions': int(df[count_var].sum()) if count_var in df.columns else 0
                }
        
        with open(self.output_dir / 'quality_report.json', 'w') as f:
            json.dump(quality_metrics, f, indent=2, default=str)
        
        # Add sentence ratio quality metrics if available
        if 'normalized_climate_attention' in df.columns:
            sentence_ratio_metrics = self._calculate_sentence_ratio_quality(df)
            
            with open(self.output_dir / 'sentence_ratio_quality.json', 'w') as f:
                json.dump(sentence_ratio_metrics, f, indent=2, default=str)
    
    def _calculate_sentence_ratio_quality(self, df: pd.DataFrame) -> Dict:
        """Calculate quality metrics for sentence ratios and normalized attention."""
        
        # Filter to observations with sentence ratio data
        with_ratios = df[df['normalized_climate_attention'].notna()]
        
        if len(with_ratios) == 0:
            return {'error': 'No sentence ratio data available'}
        
        import numpy as np
        
        quality_metrics = {
            'sentence_ratio_coverage': {
                'total_observations': len(df),
                'observations_with_ratios': len(with_ratios),
                'coverage_rate': len(with_ratios) / len(df),
                'firms_with_ratios': with_ratios['ticker'].nunique(),
                'years_with_ratios': sorted(with_ratios['year'].unique().tolist())
            },
            'normalized_attention_statistics': {
                'mean': float(with_ratios['normalized_climate_attention'].mean()),
                'median': float(with_ratios['normalized_climate_attention'].median()),
                'std': float(with_ratios['normalized_climate_attention'].std()),
                'min': float(with_ratios['normalized_climate_attention'].min()),
                'max': float(with_ratios['normalized_climate_attention'].max()),
                'p25': float(with_ratios['normalized_climate_attention'].quantile(0.25)),
                'p75': float(with_ratios['normalized_climate_attention'].quantile(0.75)),
                'p90': float(with_ratios['normalized_climate_attention'].quantile(0.90)),
                'p95': float(with_ratios['normalized_climate_attention'].quantile(0.95)),
                'p99': float(with_ratios['normalized_climate_attention'].quantile(0.99))
            },
            'sentence_ratio_distribution': {
                'zero_attention': int((with_ratios['normalized_climate_attention'] == 0).sum()),
                'low_attention_001': int((with_ratios['normalized_climate_attention'] <= 0.01).sum()),
                'medium_attention_01': int((with_ratios['normalized_climate_attention'] <= 0.1).sum()),
                'high_attention_over_01': int((with_ratios['normalized_climate_attention'] > 0.1).sum()),
                'very_high_attention_over_02': int((with_ratios['normalized_climate_attention'] > 0.2).sum())
            },
            'climate_vs_total_sentences': {
                'avg_climate_sentences': float(with_ratios['climate_sentence_count'].mean()),
                'avg_total_sentences': float(with_ratios['total_sentences_in_call'].mean()),
                'climate_sentences_p90': float(with_ratios['climate_sentence_count'].quantile(0.90)),
                'total_sentences_p90': float(with_ratios['total_sentences_in_call'].quantile(0.90))
            }
        }
        
        return quality_metrics
    
    def create_firm_level_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create firm-level summary statistics across all quarters."""
        
        # Aggregate by firm
        firm_summary = df.groupby('ticker').agg({
            'has_climate_content': 'sum',
            'total_climate_snippets': 'sum',
            'normalized_climate_attention': ['mean', 'std', 'max'] if 'normalized_climate_attention' in df.columns else 'count',
            **{f'semantic_{topic}_exposure': ['mean', 'std', 'max'] 
               for topic in self.climate_topics.keys()},
            **{f'policy_{policy}_attention': ['mean', 'std', 'max'] 
               for policy in self.policy_topics.keys()},
            'semantic_sentiment_net': ['mean', 'std']
        }).round(4)
        
        # Flatten column names
        firm_summary.columns = ['_'.join(col).strip() if col[1] else col[0] 
                               for col in firm_summary.columns.values]
        
        # Reset index to make ticker a column
        firm_summary = firm_summary.reset_index()
        
        # Save firm-level summary
        firm_summary.to_csv(self.output_dir / 'firm_level_summary.csv', index=False)
        
        return firm_summary
    
    def export_for_analysis(self, df: pd.DataFrame, format: str = 'stata') -> None:
        """Export data in format suitable for econometric analysis."""
        
        # Create analysis-ready dataset
        analysis_df = df.copy()
        
        # Convert date to string for better compatibility
        analysis_df['date'] = analysis_df['date'].dt.strftime('%Y-%m-%d')
        
        # Create quarter-year identifier
        analysis_df['quarter_year'] = analysis_df['year'].astype(str) + 'Q' + analysis_df['quarter'].astype(str)
        
        # Add firm fixed effects identifier
        analysis_df['firm_id'] = pd.Categorical(analysis_df['ticker']).codes
        
        # Add time fixed effects identifier  
        analysis_df['time_id'] = pd.Categorical(analysis_df['quarter_year']).codes
        
        if format.lower() == 'stata':
            # Export to Stata format
            try:
                analysis_df.to_stata(
                    self.output_dir / 'semantic_climate_panel.dta',
                    write_index=False,
                    version=117  # Stata 15 format
                )
                logger.info("✅ Exported to Stata format")
            except Exception as e:
                logger.warning(f"Could not export to Stata: {e}")
                # Fallback to CSV
                analysis_df.to_csv(self.output_dir / 'semantic_climate_panel_for_stata.csv', index=False)
        
        elif format.lower() == 'r':
            # Export to R-friendly format
            analysis_df.to_csv(self.output_dir / 'semantic_climate_panel_for_r.csv', index=False)
            
            # Create R script template
            r_script = f"""
# Load semantic climate variables
library(readr)
library(dplyr)
library(fixest)

# Load data
climate_data <- read_csv("{self.output_dir}/semantic_climate_panel_for_r.csv")

# Convert factors
climate_data <- climate_data %>%
  mutate(
    ticker = as.factor(ticker),
    quarter_year = as.factor(quarter_year)
  )

# Example regression with firm and time fixed effects
model1 <- feols(semantic_opportunities_exposure ~ 
                semantic_regulation_exposure + 
                has_earnings_call |
                ticker + quarter_year, 
                data = climate_data)

summary(model1)
"""
            with open(self.output_dir / 'analysis_template.R', 'w') as f:
                f.write(r_script)
            
            logger.info("✅ Exported to R format with analysis template")