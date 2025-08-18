#!/usr/bin/env python3
"""
Risk & Opportunity Visualization Module - Journal of Finance Style

Creates publication-quality visualizations for climate risk and opportunity attention analysis.
Supports data input from CSV files, DataFrames, or analyzer results.

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


class RiskOpportunityVisualizer:
    """
    Professional visualization class for climate risk and opportunity attention analysis.
    
    Creates Journal of Finance style plots with proper typography, colors,
    and event annotations for climate research.
    """
    
    def __init__(self, jof_style: bool = True, output_dir: Optional[Path] = None):
        """
        Initialize the risk & opportunity visualizer.
        
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
        
        # Analysis type configurations
        self.analysis_configs = self._load_analysis_configs()
        
        logger.info("✅ RiskOpportunityVisualizer initialized")
    
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
    
    def _load_analysis_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load analysis-specific configurations for colors and events."""
        return {
            'climate_risks': {
                'name': 'Climate Risks',
                'color': '#c5504b',
                'events': {
                    '2012-10-29': 'Hurricane Sandy',
                    '2018-08-01': 'California Wildfires',
                    '2021-02-01': 'Texas Winter Storm',
                    '2022-09-01': 'Pakistan Floods',
                    '2023-07-01': 'European Heatwave'
                }
            },
            'climate_opportunities': {
                'name': 'Climate Opportunities',
                'color': '#70ad47',
                'events': {
                    '2020-12-01': 'Solar Grid Parity',
                    '2021-06-01': 'EV Sales Surge',
                    '2022-03-01': 'Green Bond Growth',
                    '2023-01-01': 'ESG Investment Peak'
                }
            },
            'physical_risk': {
                'name': 'Physical Risk',
                'color': '#d9534f',
                'events': {
                    '2012-10-29': 'Hurricane Sandy',
                    '2018-08-01': 'California Wildfires',
                    '2021-02-01': 'Texas Winter Storm'
                }
            },
            'transition_risk': {
                'name': 'Transition Risk',
                'color': '#f0ad4e',
                'events': {
                    '2015-12-12': 'Paris Agreement',
                    '2019-12-11': 'EU Green Deal',
                    '2022-08-16': 'US IRA'
                }
            },
            'renewable_energy': {
                'name': 'Renewable Energy',
                'color': '#5cb85c',
                'events': {
                    '2020-12-01': 'Solar Grid Parity',
                    '2021-01-01': 'Offshore Wind Growth'
                }
            },
            'green_finance': {
                'name': 'Green Finance',
                'color': '#5bc0de',
                'events': {
                    '2021-07-01': 'EU Taxonomy',
                    '2022-03-01': 'Green Bond Surge'
                }
            }
        }
    
    def _load_data(self, 
                data_input: Union[str, Path, pd.DataFrame, Dict[str, Any]], 
                freq: str = 'day') -> pd.DataFrame:
        """
        Load data from various input formats and aggregate properly for smooth visualization.
        
        Args:
            data_input: CSV path, DataFrame, or analyzer results dict
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
            # Extract from analyzer results
            df = data_input['timeseries'].copy()
            logger.info(f"✅ Extracted from analysis results: {len(df)} rows")
            
        else:
            raise ValueError("Data input must be CSV path, DataFrame, or analyzer results dict")
        
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

        # Aggregate data by frequency for smooth plotting
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
        
        # Aggregate data by period
        agg_columns = {
            'count_sum': 'sum',  # Sum mentions per period
        }
        
        # Add other numeric columns for aggregation if they exist
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['count_sum'] and col in df.columns:
                agg_columns[col] = 'mean'
        
        # Keep categorical columns with first value
        categorical_cols = ['ticker', 'company_name', 'stock_index', 'risk_type', 'opportunity_type']
        for col in categorical_cols:
            if col in df.columns:
                agg_columns[col] = 'first'
        
        # Group and aggregate
        df_agg = df.groupby('period').agg(agg_columns).reset_index()
        
        # Convert period back to datetime for plotting
        df_agg['date'] = df_agg['period'].dt.to_timestamp()
        
        # Add 3-month rolling average for smoother plots
        df_agg = df_agg.sort_values('date')
        df_agg['count_sum_smooth'] = df_agg['count_sum'].rolling(window=3, center=True, min_periods=1).mean()
        
        # Use the smoothed version for plotting
        df_agg['count_sum'] = df_agg['count_sum_smooth']
        
        logger.info(f"✅ Aggregated to {len(df_agg)} time periods for smooth visualization")
        
        return df_agg
    
    def _get_analysis_info(self, analysis_type: str) -> Dict[str, Any]:
        """Get analysis configuration or create default."""
        if analysis_type in self.analysis_configs:
            return self.analysis_configs[analysis_type]
        else:
            # Create default config for unknown analysis types
            return {
                'name': analysis_type.replace('_', ' ').title(),
                'color': '#2E86AB',
                'events': {}
            }
    
    def _add_event_markers(self, ax: plt.Axes, analysis_type: str, df: pd.DataFrame) -> None:
        """Add vertical lines and labels for relevant events."""
        analysis_info = self._get_analysis_info(analysis_type)
        events = analysis_info['events']
        
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
    
    def plot_risk_timeseries(self, 
                            data_input: Union[str, Path, pd.DataFrame, Dict[str, Any]],
                            title: Optional[str] = None,
                            freq: str = 'day',
                            save_path: Optional[Path] = None,
                            show_events: bool = True,
                            figsize: tuple = (14, 8)) -> Path:
        """
        Create time series plot for climate risk attention.
        
        Args:
            data_input: Data source (CSV, DataFrame, or analysis results)
            title: Custom plot title
            freq: Frequency for aggregation ('day', 'quarter', 'year', 'month')
            save_path: Custom save path
            show_events: Whether to show climate events
            figsize: Figure size
            
        Returns:
            Path to saved plot
        """
        # Load and prepare data with proper aggregation
        df = self._load_data(data_input, freq=freq)
        
        analysis_info = self._get_analysis_info('climate_risks')
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Main time series plot with smooth line
        ax.plot(df['date'], df['count_sum'], 
                linewidth=2.5, color=analysis_info['color'])
        
        # Add event markers
        if show_events:
            self._add_event_markers(ax, 'climate_risks', df)
        
        # Labels and title
        if title is None:
            title = "Climate Risk Attention in Earnings Calls"
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Number of Risk-Related Climate Snippets', fontsize=14)
        
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
            save_path = self.output_dir / 'climate_risks_timeseries_jof.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        
        plt.show()
        
        logger.info(f"📊 Risk timeseries plot saved: {save_path}")
        return save_path
    
    def plot_opportunity_timeseries(self, 
                                   data_input: Union[str, Path, pd.DataFrame, Dict[str, Any]],
                                   title: Optional[str] = None,
                                   freq: str = 'day',
                                   save_path: Optional[Path] = None,
                                   show_events: bool = True,
                                   figsize: tuple = (14, 8)) -> Path:
        """
        Create time series plot for climate opportunity attention.
        
        Args:
            data_input: Data source (CSV, DataFrame, or analysis results)
            title: Custom plot title
            freq: Frequency for aggregation ('day', 'quarter', 'year', 'month')
            save_path: Custom save path
            show_events: Whether to show market events
            figsize: Figure size
            
        Returns:
            Path to saved plot
        """
        # Load and prepare data with proper aggregation
        df = self._load_data(data_input, freq=freq)
        
        analysis_info = self._get_analysis_info('climate_opportunities')
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Main time series plot with smooth line
        ax.plot(df['date'], df['count_sum'], 
                linewidth=2.5, color=analysis_info['color'])
        
        # Add event markers
        if show_events:
            self._add_event_markers(ax, 'climate_opportunities', df)
        
        # Labels and title
        if title is None:
            title = "Climate Opportunity Attention in Earnings Calls"
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Number of Opportunity-Related Climate Snippets', fontsize=14)
        
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
            save_path = self.output_dir / 'climate_opportunities_timeseries_jof.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        
        plt.show()
        
        logger.info(f"📊 Opportunity timeseries plot saved: {save_path}")
        return save_path
    
    def plot_risk_opportunity_comparison(self,
                                       risk_data: Union[str, Path, pd.DataFrame, Dict[str, Any]],
                                       opportunity_data: Union[str, Path, pd.DataFrame, Dict[str, Any]],
                                       title: str = "Climate Risk vs. Opportunity Attention",
                                       freq: str = 'day',
                                       save_path: Optional[Path] = None,
                                       show_events: bool = True,
                                       figsize: tuple = (14, 8)) -> Path:
        """
        Create comparison plot for risk vs. opportunity attention.
        
        Args:
            risk_data: Risk data source
            opportunity_data: Opportunity data source
            title: Plot title
            freq: Frequency for aggregation
            save_path: Custom save path
            show_events: Whether to show major events
            figsize: Figure size
            
        Returns:
            Path to saved plot
        """
        # Load both datasets with proper aggregation
        risk_df = self._load_data(risk_data, freq=freq)
        opp_df = self._load_data(opportunity_data, freq=freq)
        
        # Get configurations
        risk_info = self._get_analysis_info('climate_risks')
        opp_info = self._get_analysis_info('climate_opportunities')
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot both time series
        ax.plot(risk_df['date'], risk_df['count_sum'],
               linewidth=2.5, color=risk_info['color'], 
               label=risk_info['name'])
        
        ax.plot(opp_df['date'], opp_df['count_sum'],
               linewidth=2.5, color=opp_info['color'], 
               label=opp_info['name'])
        
        # Add major climate events
        if show_events:
            major_events = {
                '2015-12-12': 'Paris Agreement',
                '2019-12-11': 'EU Green Deal',
                '2021-02-01': 'Texas Winter Storm',
                '2022-08-16': 'US IRA',
                '2023-07-01': 'Global Heatwaves'
            }
            
            # Find y-axis range for event labels
            all_values = list(risk_df['count_sum'].values) + list(opp_df['count_sum'].values)
            
            if all_values:
                y_max = max(all_values)
                for i, (date_str, label) in enumerate(major_events.items()):
                    try:
                        event_date = pd.to_datetime(date_str)
                        
                        # Check if event is in data range
                        risk_range = risk_df['date'].min() <= event_date <= risk_df['date'].max()
                        opp_range = opp_df['date'].min() <= event_date <= opp_df['date'].max()
                        
                        if risk_range or opp_range:
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
        ax.set_ylabel('Climate Mentions in Earnings Calls', fontsize=14)
        
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
            save_path = self.output_dir / 'risk_opportunity_comparison_jof.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        
        plt.show()
        
        logger.info(f"📊 Risk vs. Opportunity comparison plot saved: {save_path}")
        return save_path
    
    def plot_stacked_risk_opportunity(self,
                                    risk_data: Union[str, Path, pd.DataFrame, Dict[str, Any]],
                                    opportunity_data: Union[str, Path, pd.DataFrame, Dict[str, Any]],
                                    title: str = "Climate Risk and Opportunity Attention",
                                    freq: str = 'day',
                                    save_path: Optional[Path] = None,
                                    figsize: tuple = (14, 8)) -> Path:
        """
        Create stacked area plot for risk and opportunity attention.
        
        Args:
            risk_data: Risk data source
            opportunity_data: Opportunity data source
            title: Plot title
            freq: Frequency for aggregation
            save_path: Custom save path
            figsize: Figure size
            
        Returns:
            Path to saved plot
        """
        # Load both datasets with proper aggregation
        risk_df = self._load_data(risk_data, freq=freq)
        opp_df = self._load_data(opportunity_data, freq=freq)
        
        # Merge datasets on date for stacked plotting
        merged_df = pd.merge(risk_df[['date', 'count_sum']], 
                            opp_df[['date', 'count_sum']], 
                            on='date', how='outer', suffixes=('_risk', '_opp'))
        merged_df = merged_df.fillna(0).sort_values('date')
        
        # Get configurations
        risk_info = self._get_analysis_info('climate_risks')
        opp_info = self._get_analysis_info('climate_opportunities')
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create stacked area plot
        ax.fill_between(merged_df['date'], 0, merged_df['count_sum_risk'],
                       color=risk_info['color'], alpha=0.7, label=risk_info['name'])
        
        ax.fill_between(merged_df['date'], merged_df['count_sum_risk'], 
                       merged_df['count_sum_risk'] + merged_df['count_sum_opp'],
                       color=opp_info['color'], alpha=0.7, label=opp_info['name'])
        
        # Labels and formatting
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Year', fontsize=14)
        ax.set_ylabel('Climate Mentions in Earnings Calls', fontsize=14)
        
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
            save_path = self.output_dir / 'risk_opportunity_stacked_jof.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        
        plt.show()
        
        logger.info(f"📊 Stacked risk/opportunity plot saved: {save_path}")
        return save_path
    
    def plot_annual_summary(self,
                           risk_data: Union[str, Path, pd.DataFrame, Dict[str, Any]],
                           opportunity_data: Union[str, Path, pd.DataFrame, Dict[str, Any]],
                           title: str = "Annual Climate Risk vs. Opportunity Mentions",
                           save_path: Optional[Path] = None,
                           figsize: tuple = (12, 6)) -> Path:
        """
        Create annual summary bar chart comparing risks and opportunities.
        
        Args:
            risk_data: Risk data source
            opportunity_data: Opportunity data source
            title: Custom plot title
            save_path: Custom save path
            figsize: Figure size
            
        Returns:
            Path to saved plot
        """
        # Load and prepare data with yearly aggregation
        risk_df = self._load_data(risk_data, freq='year')
        opp_df = self._load_data(opportunity_data, freq='year')
        
        # Extract year and aggregate
        risk_df['year'] = risk_df['date'].dt.year
        opp_df['year'] = opp_df['date'].dt.year
        
        risk_yearly = risk_df.groupby('year')['count_sum'].sum().reset_index()
        opp_yearly = opp_df.groupby('year')['count_sum'].sum().reset_index()
        
        # Merge data
        yearly_data = pd.merge(risk_yearly, opp_yearly, on='year', how='outer', 
                              suffixes=('_risk', '_opp')).fillna(0)
        
        # Only create plot if we have sufficient data
        if len(yearly_data) < 2:
            logger.warning("Insufficient data for annual summary plot")
            return None
        
        # Get configurations
        risk_info = self._get_analysis_info('climate_risks')
        opp_info = self._get_analysis_info('climate_opportunities')
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Set up bar positions
        x = yearly_data['year']
        width = 0.35
        x1 = [i - width/2 for i in x]
        x2 = [i + width/2 for i in x]
        
        # Create grouped bar plot
        bars1 = ax.bar(x1, yearly_data['count_sum_risk'], width, 
                      color=risk_info['color'], alpha=0.8, label=risk_info['name'])
        bars2 = ax.bar(x2, yearly_data['count_sum_opp'], width, 
                      color=opp_info['color'], alpha=0.8, label=opp_info['name'])
        
        # Add value labels on significant bars
        risk_mean = yearly_data['count_sum_risk'].mean()
        opp_mean = yearly_data['count_sum_opp'].mean()
        
        for bar in bars1:
            height = bar.get_height()
            if height > risk_mean:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=10)
        
        for bar in bars2:
            height = bar.get_height()
            if height > opp_mean:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=10)
        
        # Labels and formatting
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Year', fontsize=14)
        ax.set_ylabel('Annual Climate Mentions', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(yearly_data['year'], rotation=45)
        
        # Legend
        ax.legend(frameon=False, fontsize=12)
        
        # Apply JoF styling
        if self.jof_style:
            self._apply_jof_styling(ax, fig)
            ax.grid(True, linestyle='-', linewidth=0.3, alpha=0.3, color='gray', axis='y')
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / 'risk_opportunity_annual_jof.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        
        plt.show()
        
        logger.info(f"📊 Annual summary plot saved: {save_path}")
        return save_path


# Convenience functions for direct usage
def quick_plot_risks(csv_path: Union[str, Path], 
                    freq: str = 'day',
                    output_dir: Optional[Path] = None) -> Path:
    """
    Quick function to plot risk timeseries from CSV.
    
    Args:
        csv_path: Path to CSV file
        freq: Frequency for aggregation
        output_dir: Output directory
        
    Returns:
        Path to saved plot
    """
    viz = RiskOpportunityVisualizer(output_dir=output_dir)
    return viz.plot_risk_timeseries(csv_path, freq=freq)


def quick_plot_opportunities(csv_path: Union[str, Path], 
                           freq: str = 'day',
                           output_dir: Optional[Path] = None) -> Path:
    """
    Quick function to plot opportunity timeseries from CSV.
    
    Args:
        csv_path: Path to CSV file
        freq: Frequency for aggregation
        output_dir: Output directory
        
    Returns:
        Path to saved plot
    """
    viz = RiskOpportunityVisualizer(output_dir=output_dir)
    return viz.plot_opportunity_timeseries(csv_path, freq=freq)


def quick_compare_risk_opportunity(risk_csv: Union[str, Path],
                                  opportunity_csv: Union[str, Path],
                                  freq: str = 'day',
                                  output_dir: Optional[Path] = None) -> Path:
    """
    Quick function to compare risk vs opportunity from CSV files.
    
    Args:
        risk_csv: Path to risk CSV file
        opportunity_csv: Path to opportunity CSV file
        freq: Frequency for aggregation
        output_dir: Output directory
        
    Returns:
        Path to saved plot
    """
    viz = RiskOpportunityVisualizer(output_dir=output_dir)
    return viz.plot_risk_opportunity_comparison(risk_csv, opportunity_csv, freq=freq)


def quick_stacked_plot(risk_csv: Union[str, Path],
                      opportunity_csv: Union[str, Path],
                      freq: str = 'day',
                      output_dir: Optional[Path] = None) -> Path:
    """
    Quick function to create stacked risk/opportunity plot from CSV files.
    
    Args:
        risk_csv: Path to risk CSV file
        opportunity_csv: Path to opportunity CSV file
        freq: Frequency for aggregation
        output_dir: Output directory
        
    Returns:
        Path to saved plot
    """
    viz = RiskOpportunityVisualizer(output_dir=output_dir)
    return viz.plot_stacked_risk_opportunity(risk_csv, opportunity_csv, freq=freq)