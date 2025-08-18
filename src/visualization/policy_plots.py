#!/usr/bin/env python3
"""
Policy Visualization Module - Journal of Finance Style

Creates publication-quality visualizations for climate policy attention analysis.
Supports data input from CSV files, DataFrames, or PolicyAnalyzer results.

Author: Marleen de Jonge
Date: 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path
from typing import Union, Dict, List, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolicyVisualizer:
    """
    Professional visualization class for climate policy attention analysis.
    
    Creates Journal of Finance style plots with proper typography, colors,
    and event annotations for climate policy research.
    """
    
    def __init__(self, jof_style: bool = True, output_dir: Optional[Path] = None):
        """
        Initialize the policy visualizer.
        
        Args:
            jof_style: Whether to apply Journal of Finance styling
            output_dir: Default output directory for plots
        """
        self.jof_style = jof_style
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Apply JoF styling
        if self.jof_style:
            self._setup_jof_style()
        
        # Policy configuration
        self.policy_configs = self._load_policy_configs()
        
        logger.info("✅ PolicyVisualizer initialized")
    
    def _setup_jof_style(self) -> None:
        """Set up Journal of Finance publication style."""
        plt.style.use('default')
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
            'font.size': 13,
            'axes.linewidth': 1,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': False,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.labelsize': 13,
            'axes.titlesize': 17,
            'xtick.labelsize': 13,
            'ytick.labelsize': 13,
            'legend.fontsize': 13,
            'legend.frameon': False
        })
    
    def _load_policy_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load policy-specific configurations for colors and events."""
        return {
            'paris_agreement': {
                'name': 'Paris Agreement',
                'color': '#2E86AB',
                'events': {
                    '2015-12-12': 'Paris Agreement\nAdoption',
                    '2017-06-01': 'US Withdrawal\nAnnouncement',
                    '2021-01-20': 'US Re-entry'
                }
            },
            'eu_green_deal': {
                'name': 'EU Green Deal',
                'color': '#70ad47',
                'events': {
                    '2019-12-11': 'EU Green Deal\nAnnouncement',
                    '2021-07-14': 'Fit for 55\nPackage'
                }
            },
            'us_ira': {
                'name': 'US IRA',
                'color': '#c5504b',
                'events': {
                    '2022-08-16': 'IRA Signed\ninto Law'
                }
            },
            'cop_meetings': {
                'name': 'COP Meetings',
                'color': '#7030a0',
                'events': {
                    '2015-12-12': 'COP21\nParis',
                    '2021-11-01': 'COP26\nGlasgow',
                    '2022-11-06': 'COP27\nEgypt',
                    '2023-11-30': 'COP28\nDubai'
                }
            }
        }
    
    def _load_data(self, 
                data_input: Union[str, Path, pd.DataFrame, Dict[str, Any]], 
                freq: str = 'day') -> pd.DataFrame:
        """
        Load data from various input formats and aggregate properly for smooth visualization.
        
        Args:
            data_input: CSV path, DataFrame, or PolicyAnalyzer results dict
            freq: Frequency for aggregation ('day', 'quarter', 'year', 'month')
            
        Returns:
            Standardized DataFrame with aggregated data for smooth plotting
        """
        if isinstance(data_input, (str, Path)):
            # Load from CSV file
            df = pd.read_csv(data_input)
            logger.info(f"✅ Loaded data from CSV: {len(df)} rows")
            
        elif isinstance(data_input, pd.DataFrame):
            # Use DataFrame directly
            df = data_input.copy()
            logger.info(f"✅ Using DataFrame: {len(df)} rows")
            
        elif isinstance(data_input, dict) and 'timeseries' in data_input:
            # Extract from PolicyAnalyzer results
            df = data_input['timeseries'].copy()
            logger.info(f"✅ Extracted from analysis results: {len(df)} rows")
            
        else:
            raise ValueError("Data input must be CSV path, DataFrame, or PolicyAnalyzer results dict")
        
        # Standardize column names and ensure date column
        if 'date' not in df.columns and 'period' in df.columns:
            df['date'] = df['period']
        
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])

        # Standardize count column before aggregation
        count_cols = ['count_sum', 'similarity_score_count', 'mentions', 'total_mentions']
        count_col = None
        for col in count_cols:
            if col in df.columns:
                count_col = col
                break
        
        if count_col and count_col != 'count_sum':
            df['count_sum'] = df[count_col]
        elif 'count_sum' not in df.columns:
            raise ValueError("No count column found in data")

        # Aggregate data by frequency for smooth plotting (like the old code)
        if freq == 'day':
            # For daily, group by month for smoother visualization
            df['period'] = df['date'].dt.to_period('M')
        elif freq == 'quarter':
            df['period'] = df['date'].dt.to_period('Q')
        elif freq == 'year':
            df['period'] = df['date'].dt.to_period('Y')
        elif freq == 'month':
            df['period'] = df['date'].dt.to_period('M')
        else:
            # Default to monthly for unknown frequencies
            df['period'] = df['date'].dt.to_period('M')
        
        # Aggregate data by period (this creates the smoothness like your old code)
        agg_columns = {
            'count_sum': 'sum',  # Sum mentions per period
        }
        
        # Add other numeric columns for aggregation if they exist
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['count_sum'] and col in df.columns:
                agg_columns[col] = 'mean'
        
        # Keep categorical columns with first value
        categorical_cols = ['ticker', 'company_name', 'stock_index', 'policy']
        for col in categorical_cols:
            if col in df.columns:
                agg_columns[col] = 'first'
        
        # Group and aggregate
        df_agg = df.groupby('period').agg(agg_columns).reset_index()
        
        # Convert period back to datetime for plotting
        df_agg['date'] = df_agg['period'].dt.to_timestamp()
        
        # Add 3-month rolling average for even smoother plots (like your old code)
        df_agg = df_agg.sort_values('date')
        df_agg['count_sum_smooth'] = df_agg['count_sum'].rolling(window=3, center=True, min_periods=1).mean()
        
        # Use the smoothed version for plotting
        df_agg['count_sum'] = df_agg['count_sum_smooth']
        
        logger.info(f"✅ Aggregated to {len(df_agg)} time periods for smooth visualization")
        
        return df_agg
    
    def _get_policy_info(self, policy_name: str) -> Dict[str, Any]:
        """Get policy configuration or create default."""
        if policy_name in self.policy_configs:
            return self.policy_configs[policy_name]
        else:
            # Create default config for unknown policies
            return {
                'name': policy_name.replace('_', ' ').title(),
                'color': '#2E86AB',
                'events': {}
            }
    
    def _add_event_markers(self, ax: plt.Axes, policy_name: str, df: pd.DataFrame) -> None:
        """Add vertical lines and labels for policy events."""
        policy_info = self._get_policy_info(policy_name)
        events = policy_info['events']
        
        if not events:
            return
        
        y_max = df['count_sum'].max()
        data_start = df['date'].min()
        data_end = df['date'].max()
        
        for i, (date_str, label) in enumerate(events.items()):
            try:
                event_date = pd.to_datetime(date_str)
                
                # Only add events within data range
                if data_start <= event_date <= data_end:
                    ax.axvline(event_date, color='red', linestyle='--', alpha=0.7, linewidth=2)
                    
                    # Position label to avoid overlap
                    label_height = y_max * (0.8 + 0.1 * (i % 2))
                    ax.text(event_date, label_height, label,
                           rotation=90, ha='right', va='top', fontsize=10, 
                           color='red', alpha=0.9)
            except Exception as e:
                logger.warning(f"Could not add event {date_str}: {e}")
    
    def _apply_jof_styling(self, ax: plt.Axes, fig: plt.Figure) -> None:
        """Apply Journal of Finance styling to plot."""
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        
        # Add subtle grid
        ax.grid(True, linestyle='-', linewidth=0.3, alpha=0.3, color='gray')
        ax.set_axisbelow(True)
        
        # Set white background
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        # Style ticks
        ax.tick_params(colors='black', which='both')
    
    def plot_policy_timeseries(self, 
                              data_input: Union[str, Path, pd.DataFrame, Dict[str, Any]],
                              policy_name: Optional[str] = None,
                              title: Optional[str] = None,
                              freq: str = 'day',
                              save_path: Optional[Path] = None,
                              show_events: bool = True,
                              figsize: tuple = (14, 8)) -> Path:
        """
        Create main time series plot for policy attention.
        
        Args:
            data_input: Data source (CSV, DataFrame, or analysis results)
            policy_name: Name of policy for styling and events
            title: Custom plot title
            freq: Frequency for aggregation ('day', 'quarter', 'year', 'month')
            save_path: Custom save path
            show_events: Whether to show policy events
            figsize: Figure size
            
        Returns:
            Path to saved plot
        """
        # Load and prepare data with proper aggregation
        df = self._load_data(data_input, freq=freq)
        
        # Detect policy name if not provided
        if policy_name is None:
            if 'policy' in df.columns:
                policy_name = df['policy'].iloc[0]
            else:
                policy_name = 'unknown_policy'
        
        policy_info = self._get_policy_info(policy_name)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Main time series plot with smooth line (no markers for cleaner look)
        ax.plot(df['date'], df['count_sum'], 
                linewidth=2.5, color=policy_info['color'])
        
        # Add event markers
        if show_events:
            self._add_event_markers(ax, policy_name, df)
        
        # Labels and title
        if title is None:
            title = f"{policy_info['name']} Attention in Earnings Calls"
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Number of Relevant Climate Snippets', fontsize=14)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_minor_locator(mdates.YearLocator())
        
        # Apply JoF styling
        if self.jof_style:
            self._apply_jof_styling(ax, fig)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / f'{policy_name}_timeseries_jof.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        
        plt.show()
        
        logger.info(f"📊 Policy timeseries plot saved: {save_path}")
        return save_path
    
    def plot_policy_comparison(self,
                              data_inputs: List[Union[str, Path, pd.DataFrame, Dict[str, Any]]],
                              policy_names: Optional[List[str]] = None,
                              title: str = "Climate Policy Attention Comparison",
                              freq: str = 'day',
                              save_path: Optional[Path] = None,
                              show_events: bool = True,
                              figsize: tuple = (14, 8)) -> Path:
        """
        Create comparison plot for multiple policies.
        
        Args:
            data_inputs: List of data sources for each policy
            policy_names: List of policy names (optional)
            title: Plot title
            freq: Frequency for aggregation
            save_path: Custom save path
            show_events: Whether to show major events
            figsize: Figure size
            
        Returns:
            Path to saved plot
        """
        # Load all datasets with proper aggregation
        policy_data = {}
        
        for i, data_input in enumerate(data_inputs):
            df = self._load_data(data_input, freq=freq)
            
            # Determine policy name
            if policy_names and i < len(policy_names):
                policy_name = policy_names[i]
            elif 'policy' in df.columns:
                policy_name = df['policy'].iloc[0]
            else:
                policy_name = f'policy_{i+1}'
            
            policy_data[policy_name] = df
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each policy with smooth lines
        for policy_name, df in policy_data.items():
            policy_info = self._get_policy_info(policy_name)
            
            ax.plot(df['date'], df['count_sum'],
                   linewidth=2.5, color=policy_info['color'], 
                   label=policy_info['name'])
        
        # Add major climate events
        if show_events:
            major_events = {
                '2015-12-12': 'Paris Agreement',
                '2019-12-11': 'EU Green Deal',
                '2022-08-16': 'US IRA'
            }
            
            # Find y-axis range for event labels
            all_values = []
            for df in policy_data.values():
                all_values.extend(df['count_sum'].values)
            
            if all_values:
                y_max = max(all_values)
                for i, (date_str, label) in enumerate(major_events.items()):
                    try:
                        event_date = pd.to_datetime(date_str)
                        
                        # Check if any policy has data in this range
                        has_data = any(df['date'].min() <= event_date <= df['date'].max() 
                                      for df in policy_data.values())
                        
                        if has_data:
                            ax.axvline(event_date, color='gray', alpha=0.6,
                                      linestyle=':', linewidth=1.5)
                            
                            label_height = y_max * (0.85 + 0.1 * (i % 2))
                            ax.text(event_date, label_height, label,
                                   rotation=90, ha='right', va='top', fontsize=9,
                                   color='gray', alpha=0.8)
                    except Exception as e:
                        logger.warning(f"Could not add event {date_str}: {e}")
        
        # Labels and formatting
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Year', fontsize=14)
        ax.set_ylabel('Policy Mentions in Earnings Calls', fontsize=14)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        
        # Legend
        ax.legend(loc='upper left', frameon=False, fontsize=12)
        
        # Apply JoF styling
        if self.jof_style:
            self._apply_jof_styling(ax, fig)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / 'climate_policies_comparison_jof.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        
        plt.show()
        
        logger.info(f"📊 Policy comparison plot saved: {save_path}")
        return save_path
    
    def plot_annual_summary(self,
                           data_input: Union[str, Path, pd.DataFrame, Dict[str, Any]],
                           policy_name: Optional[str] = None,
                           title: Optional[str] = None,
                           save_path: Optional[Path] = None,
                           figsize: tuple = (12, 6)) -> Path:
        """
        Create annual summary bar chart.
        
        Args:
            data_input: Data source
            policy_name: Name of policy
            title: Custom plot title
            save_path: Custom save path
            figsize: Figure size
            
        Returns:
            Path to saved plot
        """
        # Load and prepare data with yearly aggregation
        df = self._load_data(data_input, freq='year')
        
        # Detect policy name if not provided
        if policy_name is None:
            if 'policy' in df.columns:
                policy_name = df['policy'].iloc[0]
            else:
                policy_name = 'unknown_policy'
        
        policy_info = self._get_policy_info(policy_name)
        
        # Extract year from aggregated data
        df['year'] = df['date'].dt.year
        yearly_data = df.groupby('year').agg({
            'count_sum': 'sum'
        }).reset_index()
        
        # Only create plot if we have sufficient data
        if len(yearly_data) < 2:
            logger.warning("Insufficient data for annual summary plot")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Bar plot
        bars = ax.bar(yearly_data['year'], yearly_data['count_sum'],
                     color=policy_info['color'], alpha=0.8, width=0.7)
        
        # Add value labels on bars for significant years
        mean_value = yearly_data['count_sum'].mean()
        for bar in bars:
            height = bar.get_height()
            if height > mean_value:  # Only label above-average years
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=11, color='black')
        
        # Labels and title
        if title is None:
            title = f"Annual {policy_info['name']} Mentions"
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Year', fontsize=14)
        ax.set_ylabel(f'Annual {policy_info["name"]} Mentions', fontsize=14)
        
        # Format x-axis
        ax.set_xticks(yearly_data['year'])
        ax.set_xticklabels(yearly_data['year'], rotation=45)
        
        # Apply JoF styling
        if self.jof_style:
            self._apply_jof_styling(ax, fig)
            # Add y-axis grid only for bar charts
            ax.grid(True, linestyle='-', linewidth=0.3, alpha=0.3, color='gray', axis='y')
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / f'{policy_name}_annual_jof.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        
        plt.show()
        
        logger.info(f"📊 Annual summary plot saved: {save_path}")
        return save_path
    
    def plot_summary_comparison(self,
                               data_inputs: List[Union[str, Path, pd.DataFrame, Dict[str, Any]]],
                               policy_names: Optional[List[str]] = None,
                               title: str = "Climate Policy Mentions Summary",
                               save_path: Optional[Path] = None,
                               figsize: tuple = (10, 6)) -> Path:
        """
        Create summary bar chart comparing total mentions across policies.
        
        Args:
            data_inputs: List of data sources for each policy
            policy_names: List of policy names (optional)
            title: Plot title
            save_path: Custom save path
            figsize: Figure size
            
        Returns:
            Path to saved plot
        """
        # Load all datasets and calculate totals
        policy_totals = {}
        policy_colors = {}
        
        for i, data_input in enumerate(data_inputs):
            df = self._load_data(data_input)
            
            # Determine policy name
            if policy_names and i < len(policy_names):
                policy_name = policy_names[i]
            elif 'policy' in df.columns:
                policy_name = df['policy'].iloc[0]
            else:
                policy_name = f'policy_{i+1}'
            
            policy_info = self._get_policy_info(policy_name)
            policy_totals[policy_info['name']] = df['count_sum'].sum()
            policy_colors[policy_info['name']] = policy_info['color']
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Prepare data for plotting
        names = list(policy_totals.keys())
        totals = list(policy_totals.values())
        colors = [policy_colors[name] for name in names]
        
        # Bar plot
        bars = ax.bar(names, totals, color=colors, alpha=0.8, width=0.6)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=13, color='black')
        
        # Labels and formatting
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('Total Mentions', fontsize=14)
        
        # Apply JoF styling
        if self.jof_style:
            self._apply_jof_styling(ax, fig)
            ax.grid(True, linestyle='-', linewidth=0.3, alpha=0.3, color='gray', axis='y')
        
        # Rotate x-axis labels if needed
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / 'climate_policies_summary_jof.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        
        plt.show()
        
        logger.info(f"📊 Summary comparison plot saved: {save_path}")
        return save_path


# Convenience functions for direct usage
def quick_plot_policy(csv_path: Union[str, Path], 
                     policy_name: Optional[str] = None,
                     freq: str = 'day',
                     output_dir: Optional[Path] = None) -> Path:
    """
    Quick function to plot policy timeseries from CSV.
    
    Args:
        csv_path: Path to CSV file
        policy_name: Name of policy (auto-detected if None)
        freq: Frequency for aggregation
        output_dir: Output directory
        
    Returns:
        Path to saved plot
    """
    viz = PolicyVisualizer(output_dir=output_dir)
    return viz.plot_policy_timeseries(csv_path, freq=freq, policy_name=policy_name)


def quick_compare_policies(csv_paths: List[Union[str, Path]],
                          policy_names: Optional[List[str]] = None,
                          freq: str = 'day',
                          output_dir: Optional[Path] = None) -> Path:
    """
    Quick function to compare multiple policies from CSV files.
    
    Args:
        csv_paths: List of CSV file paths
        policy_names: List of policy names (auto-detected if None)
        freq: Frequency for aggregation
        output_dir: Output directory
        
    Returns:
        Path to saved plot
    """
    viz = PolicyVisualizer(output_dir=output_dir)
    return viz.plot_policy_comparison(csv_paths, policy_names=policy_names, freq=freq)