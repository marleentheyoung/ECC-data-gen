#!/usr/bin/env python3
"""
Climate attention visualization script.

This script creates comprehensive visualizations of monthly climate attention trends
from the aggregated data, including time series plots, regional comparisons, and
event overlays for major climate policy announcements.

Usage:
    python scripts/run_visualization.py
    
    # With custom input/output paths
    python scripts/3_agg_variables/run_visualization.py --data-path outputs/monthly_aggregates --output-dir plots

Author: Marleen de Jonge
Date: 2025
"""

import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))


def setup_plotting_style():
    """Set up publication-quality plotting style."""
    # Use a clean style
    plt.style.use('default')
    
    # Set up seaborn
    sns.set_palette("husl")
    
    # Configure matplotlib for better plots
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'lines.linewidth': 2.5,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False
    })


def load_monthly_data(data_path: Path) -> pd.DataFrame:
    """
    Load monthly climate attention data.
    
    Args:
        data_path: Path to monthly aggregates folder
        
    Returns:
        DataFrame with monthly climate attention data
    """
    csv_file = data_path / 'monthly_climate_attention.csv'
    
    if not csv_file.exists():
        raise FileNotFoundError(f"Monthly climate attention data not found: {csv_file}")
    
    df = pd.read_csv(csv_file)
    
    # Convert month to datetime
    df['date'] = pd.to_datetime(df['month'] + '-01')
    
    # Sort by date and region
    df = df.sort_values(['date', 'region']).reset_index(drop=True)
    
    print(f"✅ Loaded {len(df)} monthly observations")
    print(f"📅 Date range: {df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}")
    print(f"🌍 Regions: {', '.join(df['region'].unique())}")
    
    return df


def get_climate_events() -> Dict[str, str]:
    """Get major climate policy events for annotation."""
    return {
        "2015-12": "Paris Agreement",
        "2017-06": "US Paris Withdrawal",
        "2019-12": "EU Green Deal",
        "2021-01": "US Paris Re-entry", 
        "2022-08": "US IRA Signed",
        "2021-11": "COP26 Glasgow",
        "2023-12": "COP28 Dubai"
    }


def plot_overall_trend(df: pd.DataFrame, output_dir: Path, show_events: bool = True) -> None:
    """
    Plot overall climate attention trend over time.
    
    Args:
        df: Monthly climate attention data
        output_dir: Output directory for plots
        show_events: Whether to show major climate events
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get overall data
    overall_data = df[df['region'] == 'Overall'].copy()
    
    if len(overall_data) == 0:
        print("⚠️ No overall data found for plotting")
        return
    
    # Plot main trend
    ax.plot(overall_data['date'], overall_data['climate_attention_ratio'], 
           color='#2E86AB', linewidth=3, label='Climate Attention Ratio', marker='o', markersize=4)
    
    # Add trend line
    x_numeric = mdates.date2num(overall_data['date'])
    z = np.polyfit(x_numeric, overall_data['climate_attention_ratio'], 1)
    p = np.poly1d(z)
    ax.plot(overall_data['date'], p(x_numeric), "--", alpha=0.7, color='red', 
           linewidth=2, label=f'Trend (slope: {z[0]*365:.4f}/year)')
    
    # Add climate events
    if show_events:
        events = get_climate_events()
        y_max = overall_data['climate_attention_ratio'].max()
        
        for event_date, event_name in events.items():
            try:
                event_datetime = pd.to_datetime(event_date + '-01')
                if overall_data['date'].min() <= event_datetime <= overall_data['date'].max():
                    ax.axvline(event_datetime, color='red', alpha=0.6, linestyle=':', linewidth=1.5)
                    ax.text(event_datetime, y_max * 0.9, event_name, 
                           rotation=90, ha='right', va='top', fontsize=9, alpha=0.8)
            except:
                continue
    
    # Formatting
    ax.set_xlabel('Date')
    ax.set_ylabel('Climate Attention Ratio')
    ax.set_title('Overall Climate Attention in Earnings Calls\n(Climate Sentences / Total Sentences)', 
                fontweight='bold', pad=20)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    
    # Add statistics text box
    mean_attention = overall_data['climate_attention_ratio'].mean()
    max_attention = overall_data['climate_attention_ratio'].max()
    total_calls = overall_data['earnings_calls_count'].sum()
    
    stats_text = f'Mean: {mean_attention:.4f}\nMax: {max_attention:.4f}\nTotal Calls: {total_calls:,}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10, 
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_dir / 'overall_climate_attention_trend.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'overall_climate_attention_trend.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved overall trend plot")


def plot_regional_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot US vs EU climate attention comparison.
    
    Args:
        df: Monthly climate attention data
        output_dir: Output directory for plots
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Get regional data
    us_data = df[df['region'] == 'US'].copy()
    eu_data = df[df['region'] == 'EU'].copy()
    
    # Plot 1: Time series comparison
    if len(us_data) > 0:
        ax1.plot(us_data['date'], us_data['climate_attention_ratio'], 
                color='#A23B72', linewidth=2.5, label='US (S&P 500)', marker='s', markersize=3)
    
    if len(eu_data) > 0:
        ax1.plot(eu_data['date'], eu_data['climate_attention_ratio'], 
                color='#F18F01', linewidth=2.5, label='EU (STOXX 600)', marker='o', markersize=3)
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Climate Attention Ratio')
    ax1.set_title('Climate Attention by Region', fontweight='bold', pad=15)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 2: Rolling averages (6-month)
    if len(us_data) > 0:
        us_data['rolling_6m'] = us_data['climate_attention_ratio'].rolling(window=6, center=True).mean()
        ax2.plot(us_data['date'], us_data['rolling_6m'], 
                color='#A23B72', linewidth=3, label='US (6-month avg)', alpha=0.8)
    
    if len(eu_data) > 0:
        eu_data['rolling_6m'] = eu_data['climate_attention_ratio'].rolling(window=6, center=True).mean()
        ax2.plot(eu_data['date'], eu_data['rolling_6m'], 
                color='#F18F01', linewidth=3, label='EU (6-month avg)', alpha=0.8)
    
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Climate Attention Ratio (6-month avg)')
    ax2.set_title('Smoothed Regional Trends', fontweight='bold', pad=15)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_dir / 'regional_climate_attention_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'regional_climate_attention_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved regional comparison plot")


def plot_volume_and_attention(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot climate attention alongside earnings call volume.
    
    Args:
        df: Monthly climate attention data
        output_dir: Output directory for plots
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Get overall data
    overall_data = df[df['region'] == 'Overall'].copy()
    
    if len(overall_data) == 0:
        print("⚠️ No overall data found for volume plot")
        return
    
    # Plot 1: Climate attention ratio
    ax1.plot(overall_data['date'], overall_data['climate_attention_ratio'], 
            color='#2E86AB', linewidth=2.5, marker='o', markersize=4)
    ax1.set_ylabel('Climate Attention Ratio', color='#2E86AB')
    ax1.set_title('Climate Attention and Earnings Call Volume Over Time', fontweight='bold', pad=20)
    ax1.tick_params(axis='y', labelcolor='#2E86AB')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Number of earnings calls
    ax2.bar(overall_data['date'], overall_data['earnings_calls_count'], 
           color='#A23B72', alpha=0.7, width=20)
    ax2.set_ylabel('Number of Earnings Calls', color='#A23B72')
    ax2.set_xlabel('Date')
    ax2.tick_params(axis='y', labelcolor='#A23B72')
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_minor_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_dir / 'climate_attention_and_volume.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'climate_attention_and_volume.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved volume and attention plot")


def plot_coverage_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot climate coverage rate (% of calls with climate content).
    
    Args:
        df: Monthly climate attention data
        output_dir: Output directory for plots
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot coverage rates by region
    for region in ['US', 'EU', 'Overall']:
        region_data = df[df['region'] == region].copy()
        
        if len(region_data) > 0:
            color_map = {'US': '#A23B72', 'EU': '#F18F01', 'Overall': '#2E86AB'}
            ax.plot(region_data['date'], region_data['climate_coverage_rate'] * 100, 
                   color=color_map[region], linewidth=2.5, label=region, 
                   marker='o' if region == 'Overall' else 's', markersize=3)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Climate Coverage Rate (%)')
    ax.set_title('Percentage of Earnings Calls with Climate Content', fontweight='bold', pad=20)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    
    # Set y-axis to percentage
    ax.set_ylim(0, None)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_dir / 'climate_coverage_rates.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'climate_coverage_rates.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved coverage analysis plot")


def create_summary_dashboard(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create a comprehensive 4-panel dashboard.
    
    Args:
        df: Monthly climate attention data
        output_dir: Output directory for plots
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    
    # Panel 1: Overall trend with events
    overall_data = df[df['region'] == 'Overall'].copy()
    if len(overall_data) > 0:
        ax1.plot(overall_data['date'], overall_data['climate_attention_ratio'], 
                color='#2E86AB', linewidth=2.5, marker='o', markersize=3)
        
        # Add major events
        events = {"2015-12": "Paris", "2019-12": "Green Deal", "2022-08": "IRA"}
        y_max = overall_data['climate_attention_ratio'].max()
        
        for event_date, event_name in events.items():
            try:
                event_datetime = pd.to_datetime(event_date + '-01')
                if overall_data['date'].min() <= event_datetime <= overall_data['date'].max():
                    ax1.axvline(event_datetime, color='red', alpha=0.6, linestyle=':', linewidth=1.5)
                    ax1.text(event_datetime, y_max * 0.9, event_name, 
                           rotation=90, ha='right', va='top', fontsize=8)
            except:
                continue
    
    ax1.set_title('Overall Climate Attention', fontweight='bold')
    ax1.set_ylabel('Climate Attention Ratio')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Panel 2: Regional comparison
    us_data = df[df['region'] == 'US'].copy()
    eu_data = df[df['region'] == 'EU'].copy()
    
    if len(us_data) > 0:
        ax2.plot(us_data['date'], us_data['climate_attention_ratio'], 
                color='#A23B72', linewidth=2.5, label='US', marker='s', markersize=3)
    if len(eu_data) > 0:
        ax2.plot(eu_data['date'], eu_data['climate_attention_ratio'], 
                color='#F18F01', linewidth=2.5, label='EU', marker='o', markersize=3)
    
    ax2.set_title('Regional Comparison', fontweight='bold')
    ax2.set_ylabel('Climate Attention Ratio')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Panel 3: Coverage rates
    for region in ['US', 'EU']:
        region_data = df[df['region'] == region].copy()
        if len(region_data) > 0:
            color = '#A23B72' if region == 'US' else '#F18F01'
            ax3.plot(region_data['date'], region_data['climate_coverage_rate'] * 100, 
                    color=color, linewidth=2.5, label=region)
    
    ax3.set_title('Climate Coverage Rate', fontweight='bold')
    ax3.set_ylabel('% Calls with Climate Content')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Panel 4: Volume
    if len(overall_data) > 0:
        ax4.bar(overall_data['date'], overall_data['earnings_calls_count'], 
               color='#2E86AB', alpha=0.7, width=20)
    
    ax4.set_title('Earnings Call Volume', fontweight='bold')
    ax4.set_ylabel('Number of Calls')
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Format all x-axes
    for ax in [ax1, ax2, ax3, ax4]:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.suptitle('Climate Attention in Earnings Calls - Dashboard', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save dashboard
    plt.savefig(output_dir / 'climate_attention_dashboard.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'climate_attention_dashboard.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved comprehensive dashboard")


def create_summary_statistics(df: pd.DataFrame, output_dir: Path) -> None:
    """Create and save summary statistics."""
    
    summary_stats = {}
    
    for region in df['region'].unique():
        region_data = df[df['region'] == region].copy()
        
        if len(region_data) > 0:
            summary_stats[region] = {
                'observations': len(region_data),
                'date_range': [region_data['date'].min().strftime('%Y-%m'), 
                              region_data['date'].max().strftime('%Y-%m')],
                'climate_attention': {
                    'mean': float(region_data['climate_attention_ratio'].mean()),
                    'median': float(region_data['climate_attention_ratio'].median()),
                    'std': float(region_data['climate_attention_ratio'].std()),
                    'min': float(region_data['climate_attention_ratio'].min()),
                    'max': float(region_data['climate_attention_ratio'].max())
                },
                'coverage': {
                    'mean_coverage_rate': float(region_data['climate_coverage_rate'].mean()),
                    'max_coverage_rate': float(region_data['climate_coverage_rate'].max())
                },
                'volume': {
                    'total_calls': int(region_data['earnings_calls_count'].sum()),
                    'avg_calls_per_month': float(region_data['earnings_calls_count'].mean())
                }
            }
    
    # Save statistics
    import json
    with open(output_dir / 'visualization_summary_stats.json', 'w') as f:
        json.dump(summary_stats, f, indent=2, default=str)
    
    print(f"📊 Saved summary statistics")


#!/usr/bin/env python3
"""
Climate attention visualization script.

This script creates comprehensive visualizations of monthly climate attention trends
from the aggregated data, including time series plots, regional comparisons, and
event overlays for major climate policy announcements.

Usage:
    python scripts/3_agg_variables/run_visualization.py
    
    # With custom input/output paths
    python scripts/3_agg_variables/run_visualization.py --data-path outputs/monthly_aggregates --output-dir plots

Author: Marleen de Jonge
Date: 2025
"""

import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))


def setup_plotting_style():
    """Set up publication-quality plotting style."""
    # Use a clean style
    plt.style.use('default')
    
    # Set up seaborn
    sns.set_palette("husl")
    
    # Configure matplotlib for better plots
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'lines.linewidth': 2.5,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False
    })


def load_monthly_data(data_path: Path) -> pd.DataFrame:
    """
    Load monthly climate attention data.
    
    Args:
        data_path: Path to monthly aggregates folder
        
    Returns:
        DataFrame with monthly climate attention data
    """
    csv_file = data_path / 'monthly_climate_attention.csv'
    
    if not csv_file.exists():
        raise FileNotFoundError(f"Monthly climate attention data not found: {csv_file}")
    
    df = pd.read_csv(csv_file)
    
    # Convert month to datetime
    df['date'] = pd.to_datetime(df['month'] + '-01')
    
    # Sort by date and region
    df = df.sort_values(['date', 'region']).reset_index(drop=True)
    
    print(f"✅ Loaded {len(df)} monthly observations")
    print(f"📅 Date range: {df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}")
    print(f"🌍 Regions: {', '.join(df['region'].unique())}")
    
    return df


def get_climate_events() -> Dict[str, str]:
    """Get major climate policy events for annotation."""
    return {
        "2015-12": "Paris Agreement",
        "2017-06": "US Paris Withdrawal",
        "2019-12": "EU Green Deal",
        "2021-01": "US Paris Re-entry", 
        "2022-08": "US IRA Signed",
        "2021-11": "COP26 Glasgow",
        "2023-12": "COP28 Dubai"
    }


def plot_overall_trend(df: pd.DataFrame, output_dir: Path, show_events: bool = True) -> None:
    """
    Plot overall climate attention trend over time.
    
    Args:
        df: Monthly climate attention data
        output_dir: Output directory for plots
        show_events: Whether to show major climate events
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get overall data
    overall_data = df[df['region'] == 'Overall'].copy()
    
    if len(overall_data) == 0:
        print("⚠️ No overall data found for plotting")
        return
    
    # Plot main trend
    ax.plot(overall_data['date'], overall_data['climate_attention_ratio'], 
           color='#2E86AB', linewidth=3, label='Climate Attention Ratio', marker='o', markersize=4)
    
    # Add trend line
    x_numeric = mdates.date2num(overall_data['date'])
    z = np.polyfit(x_numeric, overall_data['climate_attention_ratio'], 1)
    p = np.poly1d(z)
    ax.plot(overall_data['date'], p(x_numeric), "--", alpha=0.7, color='red', 
           linewidth=2, label=f'Trend (slope: {z[0]*365:.4f}/year)')
    
    # Add climate events
    if show_events:
        events = get_climate_events()
        y_max = overall_data['climate_attention_ratio'].max()
        
        for event_date, event_name in events.items():
            try:
                event_datetime = pd.to_datetime(event_date + '-01')
                if overall_data['date'].min() <= event_datetime <= overall_data['date'].max():
                    ax.axvline(event_datetime, color='red', alpha=0.6, linestyle=':', linewidth=1.5)
                    ax.text(event_datetime, y_max * 0.9, event_name, 
                           rotation=90, ha='right', va='top', fontsize=9, alpha=0.8)
            except:
                continue
    
    # Formatting
    ax.set_xlabel('Date')
    ax.set_ylabel('Climate Attention Ratio')
    ax.set_title('Overall Climate Attention in Earnings Calls\n(Climate Sentences / Total Sentences)', 
                fontweight='bold', pad=20)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    
    # Add statistics text box
    mean_attention = overall_data['climate_attention_ratio'].mean()
    max_attention = overall_data['climate_attention_ratio'].max()
    total_calls = overall_data['earnings_calls_count'].sum()
    
    stats_text = f'Mean: {mean_attention:.4f}\nMax: {max_attention:.4f}\nTotal Calls: {total_calls:,}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10, 
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_dir / 'overall_climate_attention_trend.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'overall_climate_attention_trend.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved overall trend plot")


def plot_regional_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot US vs EU climate attention comparison.
    
    Args:
        df: Monthly climate attention data
        output_dir: Output directory for plots
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Get regional data
    us_data = df[df['region'] == 'US'].copy()
    eu_data = df[df['region'] == 'EU'].copy()
    
    # Plot 1: Time series comparison
    if len(us_data) > 0:
        ax1.plot(us_data['date'], us_data['climate_attention_ratio'], 
                color='#A23B72', linewidth=2.5, label='US (S&P 500)', marker='s', markersize=3)
    
    if len(eu_data) > 0:
        ax1.plot(eu_data['date'], eu_data['climate_attention_ratio'], 
                color='#F18F01', linewidth=2.5, label='EU (STOXX 600)', marker='o', markersize=3)
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Climate Attention Ratio')
    ax1.set_title('Climate Attention by Region', fontweight='bold', pad=15)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 2: Rolling averages (3-month/quarterly)
    if len(us_data) > 0:
        us_data['rolling_3m'] = us_data['climate_attention_ratio'].rolling(window=3, center=True).mean()
        ax2.plot(us_data['date'], us_data['rolling_3m'], 
                color='#A23B72', linewidth=3, label='US (3-month avg)', alpha=0.8)
    
    if len(eu_data) > 0:
        eu_data['rolling_3m'] = eu_data['climate_attention_ratio'].rolling(window=3, center=True).mean()
        ax2.plot(eu_data['date'], eu_data['rolling_3m'], 
                color='#F18F01', linewidth=3, label='EU (3-month avg)', alpha=0.8)
    
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Climate Attention Ratio (3-month avg)')
    ax2.set_title('Smoothed Regional Trends', fontweight='bold', pad=15)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_dir / 'regional_climate_attention_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'regional_climate_attention_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved regional comparison plot")


def plot_volume_and_attention(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot climate attention alongside earnings call volume.
    
    Args:
        df: Monthly climate attention data
        output_dir: Output directory for plots
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Get overall data
    overall_data = df[df['region'] == 'Overall'].copy()
    
    if len(overall_data) == 0:
        print("⚠️ No overall data found for volume plot")
        return
    
    # Plot 1: Climate attention ratio
    ax1.plot(overall_data['date'], overall_data['climate_attention_ratio'], 
            color='#2E86AB', linewidth=2.5, marker='o', markersize=4)
    ax1.set_ylabel('Climate Attention Ratio', color='#2E86AB')
    ax1.set_title('Climate Attention and Earnings Call Volume Over Time', fontweight='bold', pad=20)
    ax1.tick_params(axis='y', labelcolor='#2E86AB')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Number of earnings calls
    ax2.bar(overall_data['date'], overall_data['earnings_calls_count'], 
           color='#A23B72', alpha=0.7, width=20)
    ax2.set_ylabel('Number of Earnings Calls', color='#A23B72')
    ax2.set_xlabel('Date')
    ax2.tick_params(axis='y', labelcolor='#A23B72')
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_minor_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_dir / 'climate_attention_and_volume.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'climate_attention_and_volume.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved volume and attention plot")


def plot_jof_style_regional_trends(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create journal-style (JOF-style) regional trends plot with clean formatting.
    
    Args:
        df: Monthly climate attention data
        output_dir: Output directory for plots
    """
    # Set journal style
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.linewidth': 1,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'grid.alpha': 0,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white'
    })
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get regional data
    us_data = df[df['region'] == 'US'].copy()
    eu_data = df[df['region'] == 'EU'].copy()
    
    # Calculate 3-month rolling averages
    if len(us_data) > 0:
        us_data = us_data.sort_values('date')
        us_data['rolling_3m'] = us_data['climate_attention_ratio'].rolling(window=3, center=True).mean()
        ax.plot(us_data['date'], us_data['rolling_3m'], 
                color='#2E4057', linewidth=2.5, label='US (S&P 500)', linestyle='-')
    
    if len(eu_data) > 0:
        eu_data = eu_data.sort_values('date')
        eu_data['rolling_3m'] = eu_data['climate_attention_ratio'].rolling(window=3, center=True).mean()
        ax.plot(eu_data['date'], eu_data['rolling_3m'], 
                color='#048A81', linewidth=2.5, label='EU (STOXX 600)', linestyle='--')
    
    # Add major climate events as red vertical lines
    events = {
        "2009-12": "Copenhagen Summit",
        "2015-12": "Paris Agreement", 
        "2017-06": "US Paris Withdrawal",
        "2019-12": "EU Green Deal",
        "2021-01": "US Paris Re-entry",
        "2022-08": "US IRA"
    }
    
    # Determine y-axis range for event labels
    y_min, y_max = ax.get_ylim() if ax.has_data() else (0, 0.1)
    if not ax.has_data():
        # Set reasonable defaults if no data yet
        if len(us_data) > 0:
            y_max = max(us_data['rolling_3m'].max(), y_max) if us_data['rolling_3m'].notna().any() else y_max
        if len(eu_data) > 0:
            y_max = max(eu_data['rolling_3m'].max(), y_max) if eu_data['rolling_3m'].notna().any() else y_max
        y_max *= 1.1  # Add some padding
    
    for event_date, event_name in events.items():
        try:
            event_datetime = pd.to_datetime(event_date + '-01')
            # Check if event is within data range
            all_dates = []
            if len(us_data) > 0:
                all_dates.extend(us_data['date'].tolist())
            if len(eu_data) > 0:
                all_dates.extend(eu_data['date'].tolist())
            
            if all_dates and min(all_dates) <= event_datetime <= max(all_dates):
                ax.axvline(event_datetime, color='red', alpha=0.7, linestyle='-', linewidth=1.2)
                
                # Add event labels at different heights to avoid overlap
                label_height = y_max * (0.85 + 0.1 * (hash(event_name) % 3))
                ax.text(event_datetime, label_height, event_name, 
                       rotation=90, ha='right', va='top', fontsize=10, 
                       color='red', alpha=0.8, weight='normal')
        except:
            continue
    
    # Format axes
    ax.set_xlabel('Year', fontsize=12, weight='normal')
    ax.set_ylabel('Climate Attention Ratio', fontsize=12, weight='normal')
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))  # Every 2 years
    ax.xaxis.set_minor_locator(mdates.YearLocator())
    
    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
    
    # Legend positioned to the right side
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=11)
    
    # Remove all grid lines
    ax.grid(False)
    
    # Set background to white
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot - PNG only
    plt.savefig(output_dir / 'jof_style_regional_trends.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"📊 Saved JOF-style regional trends plot")


def plot_jof_style_overall_trend(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create journal-style overall trend plot similar to Sautner et al. (2023).
    
    Args:
        df: Monthly climate attention data
        output_dir: Output directory for plots
    """
    # Set journal style
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.linewidth': 1,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'grid.alpha': 0,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white'
    })
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get overall data
    overall_data = df[df['region'] == 'Overall'].copy()
    
    if len(overall_data) == 0:
        print("⚠️ No overall data found for JOF-style plot")
        return
    
    overall_data = overall_data.sort_values('date')
    
    # Calculate 3-month rolling average
    overall_data['rolling_3m'] = overall_data['climate_attention_ratio'].rolling(window=3, center=True).mean()
    
    # Plot main trend line
    ax.plot(overall_data['date'], overall_data['rolling_3m'], 
           color='#1f4e79', linewidth=3, label='Climate Attention')
    
    # Add major climate events as red vertical lines
    events = {
        "2009-12": "Copenhagen",
        "2015-12": "Paris Agreement", 
        "2017-06": "US Withdrawal",
        "2019-12": "EU Green Deal",
        "2021-01": "US Re-entry",
        "2022-08": "US IRA"
    }
    
    y_max = overall_data['rolling_3m'].max() * 1.15
    
    for i, (event_date, event_name) in enumerate(events.items()):
        try:
            event_datetime = pd.to_datetime(event_date + '-01')
            if overall_data['date'].min() <= event_datetime <= overall_data['date'].max():
                ax.axvline(event_datetime, color='red', alpha=0.7, linestyle='-', linewidth=1.2)
                
                # Alternate label heights to avoid overlap
                label_height = y_max * (0.8 + 0.15 * (i % 2))
                ax.text(event_datetime, label_height, event_name, 
                       rotation=90, ha='right', va='top', fontsize=10, 
                       color='red', alpha=0.8)
        except:
            continue
    
    # Format axes
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Climate Attention Ratio', fontsize=12)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_minor_locator(mdates.YearLocator())
    
    # Format y-axis  
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
    
    # Remove grid
    ax.grid(False)
    
    # Set white background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    
    # Save plot - PNG only
    plt.savefig(output_dir / 'jof_style_overall_trend.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"📊 Saved JOF-style overall trend plot")
    """
    Plot climate coverage rate (% of calls with climate content).
    
    Args:
        df: Monthly climate attention data
        output_dir: Output directory for plots
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot coverage rates by region
    for region in ['US', 'EU', 'Overall']:
        region_data = df[df['region'] == region].copy()
        
        if len(region_data) > 0:
            color_map = {'US': '#A23B72', 'EU': '#F18F01', 'Overall': '#2E86AB'}
            ax.plot(region_data['date'], region_data['climate_coverage_rate'] * 100, 
                   color=color_map[region], linewidth=2.5, label=region, 
                   marker='o' if region == 'Overall' else 's', markersize=3)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Climate Coverage Rate (%)')
    ax.set_title('Percentage of Earnings Calls with Climate Content', fontweight='bold', pad=20)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    
    # Set y-axis to percentage
    ax.set_ylim(0, None)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_dir / 'climate_coverage_rates.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'climate_coverage_rates.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved coverage analysis plot")


def create_summary_dashboard(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create a comprehensive 4-panel dashboard.
    
    Args:
        df: Monthly climate attention data
        output_dir: Output directory for plots
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    
    # Panel 1: Overall trend with events
    overall_data = df[df['region'] == 'Overall'].copy()
    if len(overall_data) > 0:
        ax1.plot(overall_data['date'], overall_data['climate_attention_ratio'], 
                color='#2E86AB', linewidth=2.5, marker='o', markersize=3)
        
        # Add major events
        events = {"2015-12": "Paris", "2019-12": "Green Deal", "2022-08": "IRA"}
        y_max = overall_data['climate_attention_ratio'].max()
        
        for event_date, event_name in events.items():
            try:
                event_datetime = pd.to_datetime(event_date + '-01')
                if overall_data['date'].min() <= event_datetime <= overall_data['date'].max():
                    ax1.axvline(event_datetime, color='red', alpha=0.6, linestyle=':', linewidth=1.5)
                    ax1.text(event_datetime, y_max * 0.9, event_name, 
                           rotation=90, ha='right', va='top', fontsize=8)
            except:
                continue
    
    ax1.set_title('Overall Climate Attention', fontweight='bold')
    ax1.set_ylabel('Climate Attention Ratio')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Panel 2: Regional comparison
    us_data = df[df['region'] == 'US'].copy()
    eu_data = df[df['region'] == 'EU'].copy()
    
    if len(us_data) > 0:
        ax2.plot(us_data['date'], us_data['climate_attention_ratio'], 
                color='#A23B72', linewidth=2.5, label='US', marker='s', markersize=3)
    if len(eu_data) > 0:
        ax2.plot(eu_data['date'], eu_data['climate_attention_ratio'], 
                color='#F18F01', linewidth=2.5, label='EU', marker='o', markersize=3)
    
    ax2.set_title('Regional Comparison', fontweight='bold')
    ax2.set_ylabel('Climate Attention Ratio')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Panel 3: Coverage rates
    for region in ['US', 'EU']:
        region_data = df[df['region'] == region].copy()
        if len(region_data) > 0:
            color = '#A23B72' if region == 'US' else '#F18F01'
            ax3.plot(region_data['date'], region_data['climate_coverage_rate'] * 100, 
                    color=color, linewidth=2.5, label=region)
    
    ax3.set_title('Climate Coverage Rate', fontweight='bold')
    ax3.set_ylabel('% Calls with Climate Content')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Panel 4: Volume
    if len(overall_data) > 0:
        ax4.bar(overall_data['date'], overall_data['earnings_calls_count'], 
               color='#2E86AB', alpha=0.7, width=20)
    
    ax4.set_title('Earnings Call Volume', fontweight='bold')
    ax4.set_ylabel('Number of Calls')
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Format all x-axes
    for ax in [ax1, ax2, ax3, ax4]:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.suptitle(        'Smoothed Regional Trends',
        fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save dashboard
    plt.savefig(output_dir / 'climate_attention_dashboard.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'climate_attention_dashboard.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved comprehensive dashboard")


def create_summary_statistics(df: pd.DataFrame, output_dir: Path) -> None:
    """Create and save summary statistics."""
    
    summary_stats = {}
    
    for region in df['region'].unique():
        region_data = df[df['region'] == region].copy()
        
        if len(region_data) > 0:
            summary_stats[region] = {
                'observations': len(region_data),
                'date_range': [region_data['date'].min().strftime('%Y-%m'), 
                              region_data['date'].max().strftime('%Y-%m')],
                'climate_attention': {
                    'mean': float(region_data['climate_attention_ratio'].mean()),
                    'median': float(region_data['climate_attention_ratio'].median()),
                    'std': float(region_data['climate_attention_ratio'].std()),
                    'min': float(region_data['climate_attention_ratio'].min()),
                    'max': float(region_data['climate_attention_ratio'].max())
                },
                'coverage': {
                    'mean_coverage_rate': float(region_data['climate_coverage_rate'].mean()),
                    'max_coverage_rate': float(region_data['climate_coverage_rate'].max())
                },
                'volume': {
                    'total_calls': int(region_data['earnings_calls_count'].sum()),
                    'avg_calls_per_month': float(region_data['earnings_calls_count'].mean())
                }
            }
    
    # Save statistics
    import json
    with open(output_dir / 'visualization_summary_stats.json', 'w') as f:
        json.dump(summary_stats, f, indent=2, default=str)
    
    print(f"📊 Saved summary statistics")


def main():
    parser = argparse.ArgumentParser(
        description='Create visualizations of monthly climate attention trends'
    )
    
    parser.add_argument(
        '--data-path',
        type=Path,
        default=Path("outputs/monthly_aggregates"),
        help='Path to monthly aggregates data'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path("outputs/plots"),
        help='Output directory for plots'
    )
    
    parser.add_argument(
        '--show-events',
        action='store_true',
        help='Show major climate events on plots'
    )
    
    parser.add_argument(
        '--format',
        choices=['png', 'pdf', 'both'],
        default='both',
        help='Output format for plots'
    )
    
    args = parser.parse_args()
    
    print("📊 Climate Attention Visualization")
    print("=" * 50)
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Set up plotting style
        setup_plotting_style()
        
        # Load data
        print(f"\n📥 Loading monthly climate attention data...")
        df = load_monthly_data(args.data_path)
        
        # Create visualizations
        print(f"\n🎨 Creating visualizations...")
        
        plot_overall_trend(df, args.output_dir, args.show_events)
        plot_regional_comparison(df, args.output_dir)
        plot_volume_and_attention(df, args.output_dir)
        plot_coverage_analysis(df, args.output_dir)
        create_summary_dashboard(df, args.output_dir)
        
        # Create JOF-style plots
        print(f"📰 Creating journal-style plots...")
        plot_jof_style_overall_trend(df, args.output_dir)
        plot_jof_style_regional_trends(df, args.output_dir)
        
        # Create summary statistics
        create_summary_statistics(df, args.output_dir)
        
        print(f"\n✅ Visualization completed!")
        print(f"📁 Plots saved to: {args.output_dir}")
        print(f"\nGenerated visualizations:")
        print(f"  • overall_climate_attention_trend.png/pdf - Main trend with events")
        print(f"  • regional_climate_attention_comparison.png/pdf - US vs EU comparison")
        print(f"  • climate_attention_and_volume.png/pdf - Attention vs call volume")
        print(f"  • climate_coverage_rates.png/pdf - Coverage rate analysis")
        print(f"  • climate_attention_dashboard.png/pdf - Comprehensive 4-panel dashboard")
        print(f"  • jof_style_overall_trend.png - Journal-style overall trend")
        print(f"  • jof_style_regional_trends.png - Journal-style regional comparison")
        print(f"  • visualization_summary_stats.json - Summary statistics")
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()