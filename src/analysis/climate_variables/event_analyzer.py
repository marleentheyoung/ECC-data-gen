"""
Event study analyzer for climate-related policy events and firm attention.

This module analyzes how firm-level climate attention changes around major
climate policy events, natural disasters, and other external shocks.

Author: Marleen de Jonge
Date: 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import json
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

logger = logging.getLogger(__name__)


class ClimateEventStudyAnalyzer:
    """
    Analyze firm climate attention around major events using event study methodology.
    """
    
    def __init__(self, climate_panel_path: str):
        """
        Initialize the event study analyzer.
        
        Args:
            climate_panel_path: Path to the semantic_climate_panel.csv file
        """
        self.climate_panel_path = Path(climate_panel_path)
        self.df = None
        self.load_climate_panel()
        
        # Predefined major climate events
        self.predefined_events = {
            # International agreements
            "Paris Agreement Adoption": "2015-12-12",
            "US Paris Withdrawal Announcement": "2017-06-01", 
            "US Paris Re-entry": "2021-01-20",
            
            # Major policy announcements
            "EU Green Deal Announcement": "2019-12-11",
            "US IRA Signing": "2022-08-16",
            "EU Taxonomy Regulation": "2020-06-18",
            
            # COP meetings
            "COP21 Paris": "2015-11-30",
            "COP26 Glasgow": "2021-10-31",
            "COP27 Egypt": "2022-11-06",
            "COP28 Dubai": "2023-11-30",
            
            # Major climate disasters/reports
            "IPCC AR6 Report": "2021-08-09",
            "Hurricane Sandy": "2012-10-29",
            "Australia Bushfires": "2020-01-01",
            "European Heatwave": "2023-07-01",
            
            # Corporate/financial events
            "BlackRock Climate Letter": "2020-01-14",
            "Climate Action 100+ Launch": "2017-12-12",
            "TCFD Recommendations": "2017-06-29"
        }
    
    def load_climate_panel(self):
        """Load the climate panel dataset."""
        if not self.climate_panel_path.exists():
            raise FileNotFoundError(f"Climate panel file not found: {self.climate_panel_path}")
        
        logger.info(f"Loading climate panel from: {self.climate_panel_path}")
        self.df = pd.read_csv(self.climate_panel_path)
        
        # Convert date column to datetime
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Create year-quarter identifier
        self.df['year_quarter'] = self.df['year'].astype(str) + 'Q' + self.df['quarter'].astype(str)
        
        # Sort by firm and date
        self.df = self.df.sort_values(['ticker', 'date'])
        
        logger.info(f"✅ Loaded panel: {len(self.df):,} observations, {self.df['ticker'].nunique()} firms")
        logger.info(f"📅 Date range: {self.df['date'].min()} to {self.df['date'].max()}")
    
    def run_event_study(self, event_date: str, event_name: str = None,
                       outcome_vars: List[str] = None,
                       event_window: Tuple[int, int] = (-4, 4),
                       estimation_window: Tuple[int, int] = (-20, -5),
                       min_observations: int = 5,
                       compare_regions: bool = True) -> Dict:
        """
        Run event study analysis for a specific event.
        
        Args:
            event_date: Event date in YYYY-MM-DD format
            event_name: Name of the event (for reporting)
            outcome_vars: List of outcome variables to analyze
            event_window: Tuple of (quarters_before, quarters_after) event
            estimation_window: Tuple for estimation period (quarters before event)
            min_observations: Minimum observations required per firm
            compare_regions: Whether to compare EU vs US responses
            
        Returns:
            Dictionary with event study results
        """
        if event_name is None:
            event_name = f"Event_{event_date}"
        
        if outcome_vars is None:
            outcome_vars = [
                'normalized_climate_attention',
                'semantic_opportunities_exposure', 
                'semantic_regulation_exposure',
                'semantic_physical_risk_exposure',
                'semantic_disclosure_exposure'
            ]
        
        logger.info(f"🎯 Running event study: {event_name} ({event_date})")
        logger.info(f"📊 Outcome variables: {outcome_vars}")
        logger.info(f"🗓️ Event window: {event_window} quarters")
        
        event_date_dt = pd.to_datetime(event_date)
        
        # Filter data around event
        event_data = self._prepare_event_data(event_date_dt, event_window, estimation_window)
        
        if len(event_data) == 0:
            logger.warning("❌ No data found around event date")
            return {'error': 'No data available for this event'}
        
        results = {
            'event_name': event_name,
            'event_date': event_date,
            'event_window': event_window,
            'estimation_window': estimation_window,
            'outcome_variables': outcome_vars,
            'total_observations': len(event_data),
            'unique_firms': event_data['ticker'].nunique()
        }
        
        # Run analysis for each outcome variable
        variable_results = {}
        for var in outcome_vars:
            if var not in event_data.columns:
                logger.warning(f"⚠️ Variable {var} not found in data")
                continue
            
            var_result = self._analyze_variable_event_impact(
                event_data, var, event_date_dt, event_window, estimation_window, min_observations
            )
            variable_results[var] = var_result
        
        results['variable_results'] = variable_results
        
        # Regional comparison
        if compare_regions:
            regional_results = self._compare_regional_responses(
                event_data, outcome_vars, event_date_dt, event_window
            )
            results['regional_comparison'] = regional_results
        
        # Create summary statistics
        results['summary'] = self._create_event_summary(variable_results)
        
        logger.info(f"✅ Event study completed for {event_name}")
        return results
    
    def _prepare_event_data(self, event_date: pd.Timestamp, 
                           event_window: Tuple[int, int],
                           estimation_window: Tuple[int, int]) -> pd.DataFrame:
        """Prepare data for event study analysis."""
        
        # Calculate date ranges
        estimation_start = event_date + pd.DateOffset(months=3*estimation_window[0])
        estimation_end = event_date + pd.DateOffset(months=3*estimation_window[1])
        event_start = event_date + pd.DateOffset(months=3*event_window[0])
        event_end = event_date + pd.DateOffset(months=3*event_window[1])
        
        # Filter data to analysis period
        analysis_start = min(estimation_start, event_start)
        analysis_end = max(estimation_end, event_end)
        
        event_data = self.df[
            (self.df['date'] >= analysis_start) & 
            (self.df['date'] <= analysis_end)
        ].copy()
        
        # Add event-relative time variables
        event_data['quarters_from_event'] = (
            (event_data['date'] - event_date) / pd.Timedelta(days=91.25)
        ).round().astype(int)
        
        # Classify periods
        event_data['period'] = 'other'
        event_data.loc[
            (event_data['quarters_from_event'] >= estimation_window[0]) &
            (event_data['quarters_from_event'] <= estimation_window[1]), 
            'period'
        ] = 'estimation'
        event_data.loc[
            (event_data['quarters_from_event'] >= event_window[0]) &
            (event_data['quarters_from_event'] <= event_window[1]), 
            'period'
        ] = 'event'
        
        # Add market classification
        event_data['market'] = event_data['ticker'].apply(self._classify_market)
        
        return event_data
    
    def _classify_market(self, ticker: str) -> str:
        """Classify ticker as EU or US market (simplified heuristic)."""
        # This is a simplified classification - you might want to use a proper mapping
        us_patterns = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
        if any(pattern in ticker for pattern in us_patterns):
            return 'US'
        
        # European patterns (simplified)
        eu_patterns = ['ASML', 'SAP', 'LVMH', 'NESN', 'ROCHE']
        if any(pattern in ticker for pattern in eu_patterns):
            return 'EU'
        
        # Default classification based on your data structure
        # You should replace this with proper market classification
        return 'US'  # Default to US for SP500 data
    
    def _analyze_variable_event_impact(self, event_data: pd.DataFrame, 
                                     variable: str, event_date: pd.Timestamp,
                                     event_window: Tuple[int, int],
                                     estimation_window: Tuple[int, int],
                                     min_observations: int) -> Dict:
        """Analyze event impact for a specific variable."""
        
        # Calculate normal levels (estimation period average)
        estimation_data = event_data[event_data['period'] == 'estimation']
        event_period_data = event_data[event_data['period'] == 'event']
        
        # Firm-level normal levels
        normal_levels = estimation_data.groupby('ticker')[variable].mean()
        
        # Calculate abnormal levels in event period
        abnormal_results = []
        
        for ticker in normal_levels.index:
            firm_event_data = event_period_data[event_period_data['ticker'] == ticker]
            
            if len(firm_event_data) < min_observations:
                continue
            
            normal_level = normal_levels[ticker]
            
            for _, row in firm_event_data.iterrows():
                abnormal_level = row[variable] - normal_level
                abnormal_results.append({
                    'ticker': ticker,
                    'date': row['date'],
                    'quarters_from_event': row['quarters_from_event'],
                    'actual_level': row[variable],
                    'normal_level': normal_level,
                    'abnormal_level': abnormal_level,
                    'market': row['market']
                })
        
        if not abnormal_results:
            return {'error': f'No valid observations for {variable}'}
        
        abnormal_df = pd.DataFrame(abnormal_results)
        
        # Statistical tests
        mean_abnormal = abnormal_df['abnormal_level'].mean()
        std_abnormal = abnormal_df['abnormal_level'].std()
        n_obs = len(abnormal_df)
        
        # T-test for significance
        t_stat = mean_abnormal / (std_abnormal / np.sqrt(n_obs)) if std_abnormal > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_obs - 1)) if n_obs > 1 else 1.0
        
        # Time series of abnormal levels
        timeline = abnormal_df.groupby('quarters_from_event').agg({
            'abnormal_level': ['mean', 'std', 'count'],
            'actual_level': 'mean',
            'normal_level': 'mean'
        }).round(4)
        
        # Flatten column names
        timeline.columns = ['_'.join(col).strip() for col in timeline.columns.values]
        timeline = timeline.reset_index()
        
        return {
            'variable': variable,
            'mean_abnormal_level': mean_abnormal,
            'std_abnormal_level': std_abnormal,
            'n_observations': n_obs,
            'n_firms': abnormal_df['ticker'].nunique(),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'timeline': timeline.to_dict('records'),
            'firm_level_results': abnormal_df.groupby('ticker')['abnormal_level'].mean().to_dict()
        }
    
    def _compare_regional_responses(self, event_data: pd.DataFrame,
                                  outcome_vars: List[str],
                                  event_date: pd.Timestamp,
                                  event_window: Tuple[int, int]) -> Dict:
        """Compare regional responses to the event."""
        
        event_period_data = event_data[event_data['period'] == 'event']
        estimation_data = event_data[event_data['period'] == 'estimation']
        
        regional_results = {}
        
        for var in outcome_vars:
            if var not in event_data.columns:
                continue
            
            # Calculate normal levels by market
            normal_by_market = estimation_data.groupby(['ticker', 'market'])[var].mean().reset_index()
            normal_by_market = normal_by_market.groupby('market')[var].mean()
            
            # Calculate event period levels by market
            event_by_market = event_period_data.groupby('market')[var].mean()
            
            # Calculate abnormal levels
            abnormal_by_market = {}
            for market in ['US', 'EU']:
                if market in event_by_market.index and market in normal_by_market.index:
                    abnormal_by_market[market] = event_by_market[market] - normal_by_market[market]
                else:
                    abnormal_by_market[market] = None
            
            # Test for regional differences
            us_event_data = event_period_data[event_period_data['market'] == 'US'][var].dropna()
            eu_event_data = event_period_data[event_period_data['market'] == 'EU'][var].dropna()
            
            if len(us_event_data) > 5 and len(eu_event_data) > 5:
                t_stat, p_val = stats.ttest_ind(us_event_data, eu_event_data)
                regional_test = {
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'significant': p_val < 0.05
                }
            else:
                regional_test = {'error': 'Insufficient data for regional comparison'}
            
            regional_results[var] = {
                'us_abnormal_level': abnormal_by_market.get('US'),
                'eu_abnormal_level': abnormal_by_market.get('EU'),
                'regional_difference': (abnormal_by_market.get('US', 0) - 
                                      abnormal_by_market.get('EU', 0)),
                'regional_test': regional_test
            }
        
        return regional_results
    
    def _create_event_summary(self, variable_results: Dict) -> Dict:
        """Create summary statistics for the event study."""
        
        significant_vars = []
        effect_sizes = []
        
        for var, results in variable_results.items():
            if 'error' not in results:
                if results.get('significant', False):
                    significant_vars.append(var)
                effect_sizes.append({
                    'variable': var,
                    'effect_size': results.get('mean_abnormal_level', 0),
                    'p_value': results.get('p_value', 1.0)
                })
        
        return {
            'total_variables_tested': len(variable_results),
            'significant_variables': significant_vars,
            'num_significant': len(significant_vars),
            'effect_sizes': effect_sizes,
            'strongest_effect': max(effect_sizes, key=lambda x: abs(x['effect_size'])) if effect_sizes else None
        }
    
    def run_multiple_events(self, events: Dict[str, str] = None,
                          outcome_vars: List[str] = None,
                          **kwargs) -> Dict:
        """
        Run event studies for multiple events.
        
        Args:
            events: Dictionary of {event_name: event_date}
            outcome_vars: List of outcome variables
            **kwargs: Additional arguments for run_event_study
            
        Returns:
            Dictionary with results for all events
        """
        if events is None:
            events = self.predefined_events
        
        logger.info(f"🎯 Running event studies for {len(events)} events")
        
        all_results = {}
        
        for event_name, event_date in events.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"ANALYZING: {event_name}")
            logger.info(f"{'='*60}")
            
            try:
                event_result = self.run_event_study(
                    event_date=event_date,
                    event_name=event_name,
                    outcome_vars=outcome_vars,
                    **kwargs
                )
                all_results[event_name] = event_result
                
            except Exception as e:
                logger.error(f"❌ Error analyzing {event_name}: {e}")
                all_results[event_name] = {'error': str(e)}
        
        # Create cross-event summary
        all_results['cross_event_summary'] = self._create_cross_event_summary(all_results)
        
        return all_results
    
    def _create_cross_event_summary(self, all_results: Dict) -> Dict:
        """Create summary across all events."""
        
        successful_events = {k: v for k, v in all_results.items() 
                           if k != 'cross_event_summary' and 'error' not in v}
        
        if not successful_events:
            return {'error': 'No successful event studies'}
        
        # Count significant effects by variable
        variable_significance = defaultdict(int)
        variable_effects = defaultdict(list)
        
        for event_name, results in successful_events.items():
            for var, var_results in results.get('variable_results', {}).items():
                if 'error' not in var_results:
                    if var_results.get('significant', False):
                        variable_significance[var] += 1
                    variable_effects[var].append(var_results.get('mean_abnormal_level', 0))
        
        # Calculate average effects
        avg_effects = {}
        for var, effects in variable_effects.items():
            avg_effects[var] = {
                'mean_effect': np.mean(effects),
                'std_effect': np.std(effects),
                'times_significant': variable_significance[var],
                'total_events': len(effects)
            }
        
        return {
            'total_events_analyzed': len(successful_events),
            'variables_by_significance': dict(sorted(variable_significance.items(), 
                                                   key=lambda x: x[1], reverse=True)),
            'average_effects': avg_effects,
            'most_responsive_variable': max(variable_significance.items(), 
                                          key=lambda x: x[1])[0] if variable_significance else None
        }
    
    def save_results(self, results: Dict, output_path: str) -> None:
        """Save event study results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            return obj
        
        # Deep convert all numpy types
        def deep_convert(obj):
            if isinstance(obj, dict):
                return {key: deep_convert(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [deep_convert(item) for item in obj]
            else:
                return convert_numpy(obj)
        
        converted_results = deep_convert(results)
        
        with open(output_path, 'w') as f:
            json.dump(converted_results, f, indent=2, default=str)
        
        logger.info(f"💾 Event study results saved to: {output_path}")
    
    def plot_event_timeline(self, results: Dict, variable: str, 
                          save_path: str = None) -> None:
        """
        Plot timeline of abnormal levels around event.
        
        Args:
            results: Results from run_event_study
            variable: Variable to plot
            save_path: Path to save plot (optional)
        """
        if variable not in results.get('variable_results', {}):
            logger.error(f"Variable {variable} not found in results")
            return
        
        var_results = results['variable_results'][variable]
        if 'timeline' not in var_results:
            logger.error(f"No timeline data for {variable}")
            return
        
        timeline_df = pd.DataFrame(var_results['timeline'])
        
        plt.figure(figsize=(12, 6))
        
        # Plot abnormal levels
        plt.subplot(1, 2, 1)
        plt.plot(timeline_df['quarters_from_event'], timeline_df['abnormal_level_mean'], 
                marker='o', linewidth=2, markersize=6)
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Event Date')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.fill_between(timeline_df['quarters_from_event'],
                        timeline_df['abnormal_level_mean'] - timeline_df['abnormal_level_std'],
                        timeline_df['abnormal_level_mean'] + timeline_df['abnormal_level_std'],
                        alpha=0.3)
        plt.xlabel('Quarters from Event')
        plt.ylabel('Abnormal Level')
        plt.title(f'Abnormal {variable.replace("_", " ").title()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot actual vs normal levels
        plt.subplot(1, 2, 2)
        plt.plot(timeline_df['quarters_from_event'], timeline_df['actual_level_mean'], 
                marker='o', label='Actual', linewidth=2)
        plt.plot(timeline_df['quarters_from_event'], timeline_df['normal_level_mean'], 
                marker='s', label='Normal', linewidth=2)
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Event Date')
        plt.xlabel('Quarters from Event')
        plt.ylabel('Level')
        plt.title(f'Actual vs Normal {variable.replace("_", " ").title()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'Event Study: {results.get("event_name", "Unknown Event")}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Plot saved to: {save_path}")
        
        plt.show()