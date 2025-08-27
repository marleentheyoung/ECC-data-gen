#!/usr/bin/env python3
"""
Climate Attention Time Series Visualization

This script creates a stacked line chart showing the evolution of five types of 
climate attention from earnings calls data over time, aggregated across all firms.

Usage:
    python plot_climate_attention_timeseries.py

Author: Marleen de Jonge
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_attention_data(data_dir: Path) -> dict:
    """
    Load all five types of climate attention data.
    
    Args:
        data_dir: Path to the firm_level_panel directory
        
    Returns:
        Dictionary with DataFrames for each attention type
    """
    attention_types = {
        'General Climate': 'general_climate/firm_level_climate_attention.csv',
        'Opportunities': 'opportunity/firm_level_opportunity_attention.csv', 
        'Physical Risk': 'physical_risk/firm_level_physical_risk_attention.csv',
        'Transition Risk': 'transition_risk/firm_level_transition_risk_attention.csv',
        'Transparency & Disclosure': 'transparency_disclosure/firm_level_transparency_disclosure_attention.csv'
    }
    
    data = {}
    
    for attention_type, file_path in attention_types.items():
        full_path = data_dir / file_path
        
        if full_path.exists():
            print(f"✅ Loading {attention_type} data from {full_path}")
            df = pd.read_csv(full_path)
            data[attention_type] = df
        else:
            print(f"⚠️ File not found: {full_path}")
            print(f"   Expected structure: {data_dir}/{file_path}")
    
    return data

def prepare_timeseries_data(data: dict, aggregation: str = 'year', region: str = 'all') -> pd.DataFrame:
    """
    Prepare aggregated time series data for plotting.
    
    Args:
        data: Dictionary of DataFrames for each attention type
        aggregation: 'year' or 'quarter'
        region: 'all', 'US', or 'EU'
        
    Returns:
        DataFrame with time series data ready for plotting
    """
    all_timeseries = []
    
    for attention_type, df in data.items():
        if df.empty:
            continue
            
        # Filter by region if specified
        if region != 'all':
            df = df[df['region'] == region].copy()
        
        if df.empty:
            print(f"⚠️ No data for {attention_type} in region {region}")
            continue
        
        # Determine the main ratio column for each attention type
        ratio_columns = {
            'General Climate': 'climate_attention_ratio',
            'Opportunities': 'climate_opportunity_ratio', 
            'Physical Risk': 'climate_physical_risk_ratio',
            'Transition Risk': 'climate_transition_risk_ratio',
            'Transparency & Disclosure': 'climate_transparency_disclosure_ratio'
        }
        
        ratio_col = ratio_columns.get(attention_type)
        if ratio_col not in df.columns:
            print(f"⚠️ Column {ratio_col} not found in {attention_type} data")
            continue
        
        # Create time period
        if aggregation == 'year':
            df['time_period'] = df['year']
            group_cols = ['year']
        else:  # quarter
            df['time_period'] = df['year'].astype(str) + '-' + df['quarter'].astype(str)
            group_cols = ['year', 'quarter']
        
        # Aggregate by time period
        agg_data = df.groupby(group_cols).agg({
            ratio_col: 'mean',  # Average across firms
            'ticker': 'nunique'  # Count of firms
        }).reset_index()
        
        # Add metadata
        agg_data['attention_type'] = attention_type
        agg_data['attention_ratio'] = agg_data[ratio_col]
        agg_data['firm_count'] = agg_data['ticker']
        
        if aggregation == 'year':
            agg_data['time_period'] = agg_data['year']
        else:
            agg_data['time_period'] = agg_data['year'].astype(str) + '-' + agg_data['quarter'].astype(str)
        
        all_timeseries.append(agg_data[['time_period', 'attention_type', 'attention_ratio', 'firm_count']])
    
    if not all_timeseries:
        raise ValueError("No valid data found for any attention type")
    
    # Combine all data
    combined_df = pd.concat(all_timeseries, ignore_index=True)
    
    # Pivot to get attention types as columns
    pivot_df = combined_df.pivot(index='time_period', columns='attention_type', values='attention_ratio')
    pivot_df = pivot_df.fillna(0)  # Fill missing values with 0
    
    # Ensure all expected columns exist
    expected_columns = ['General Climate', 'Opportunities', 'Physical Risk', 'Transition Risk', 'Transparency & Disclosure']
    for col in expected_columns:
        if col not in pivot_df.columns:
            pivot_df[col] = 0
    
    # Reorder columns
    pivot_df = pivot_df[expected_columns]
    
    # Sort by time period
    if aggregation == 'year':
        pivot_df.index = pd.to_numeric(pivot_df.index)
        pivot_df = pivot_df.sort_index()
    else:
        # For quarters, create proper sorting
        pivot_df = pivot_df.reset_index()
        pivot_df[['year', 'quarter']] = pivot_df['time_period'].str.split('-', expand=True)
        pivot_df['year'] = pd.to_numeric(pivot_df['year'])
        pivot_df['quarter'] = pivot_df['quarter'].str.replace('Q', '').astype(int)
        pivot_df = pivot_df.sort_values(['year', 'quarter'])
        pivot_df = pivot_df.set_index('time_period')
        pivot_df = pivot_df.drop(['year', 'quarter'], axis=1)
    
    return pivot_df

def create_stacked_line_chart(df: pd.DataFrame, 
                            title: str,
                            aggregation: str = 'year',
                            region: str = 'all',
                            save_path: Path = None) -> None:
    """
    Create a stacked area chart showing climate attention evolution.
    
    Args:
        df: DataFrame with time series data
        title: Chart title
        aggregation: 'year' or 'quarter'  
        region: 'all', 'US', or 'EU'
        save_path: Path to save the figure
    """
    # Set up the figure with high DPI for quality
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[3, 1])
    
    # Define colors for each attention type
    colors = {
        'General Climate': '#2E86AB',
        'Opportunities': '#A23B72', 
        'Physical Risk': '#F18F01',
        'Transition Risk': '#C73E1D',
        'Transparency & Disclosure': '#6A994E'
    }
    
    # Main stacked area chart
    x = range(len(df))
    x_labels = df.index
    
    # Create stacked areas
    bottom = np.zeros(len(df))
    
    for attention_type in df.columns:
        values = df[attention_type].values
        ax1.fill_between(x, bottom, bottom + values, 
                        label=attention_type, 
                        color=colors[attention_type],
                        alpha=0.8,
                        edgecolor='white',
                        linewidth=0.5)
        
        # Add line on top for clarity
        ax1.plot(x, bottom + values, 
                color=colors[attention_type], 
                linewidth=2,
                alpha=0.9)
        
        bottom += values
    
    # Customize main chart
    ax1.set_xlabel('Time Period', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Climate Attention Ratio (%)', fontsize=14, fontweight='bold')
    ax1.set_title(title, fontsize=18, fontweight='bold', pad=20)
    
    # Format y-axis as percentage
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.1f}%'))
    
    # Set x-axis labels
    if aggregation == 'year':
        # Show every 2nd year to avoid crowding
        step = max(1, len(x_labels) // 8)
        ax1.set_xticks(x[::step])
        ax1.set_xticklabels(x_labels[::step], rotation=45)
    else:
        # For quarters, show every 4th quarter (annual)
        step = max(1, len(x_labels) // 16)
        ax1.set_xticks(x[::step])
        ax1.set_xticklabels([str(label) for label in x_labels[::step]], rotation=45)
    
    # Add grid
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    
    # Legend
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    
    # Individual line charts for each attention type
    for i, attention_type in enumerate(df.columns):
        values = df[attention_type].values
        ax2.plot(x, values, 
                label=attention_type, 
                color=colors[attention_type],
                linewidth=2,
                marker='o',
                markersize=3,
                alpha=0.8)
    
    # Customize individual lines chart
    ax2.set_xlabel('Time Period', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Attention Ratio (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Individual Attention Types', fontsize=14, fontweight='bold')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.2f}%'))
    
    # Set x-axis labels for bottom chart
    if aggregation == 'year':
        step = max(1, len(x_labels) // 8)
        ax2.set_xticks(x[::step])
        ax2.set_xticklabels(x_labels[::step], rotation=45)
    else:
        step = max(1, len(x_labels) // 16)
        ax2.set_xticks(x[::step])
        ax2.set_xticklabels([str(label) for label in x_labels[::step]], rotation=45)
    
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10, ncol=2)
    
    # Add summary statistics
    total_latest = df.iloc[-1].sum()
    total_earliest = df.iloc[0].sum()
    growth = ((total_latest - total_earliest) / total_earliest * 100) if total_earliest > 0 else 0
    
    stats_text = f"""Summary Statistics:
Latest Total Attention: {total_latest*100:.2f}%
Growth from {df.index[0]} to {df.index[-1]}: {growth:.1f}%
Dominant Type (Latest): {df.iloc[-1].idxmax()}
Time Periods: {len(df)}
Region: {region.upper() if region != 'all' else 'All Regions'}"""
    
    ax1.text(0.02, 0.98, stats_text, 
            transform=ax1.transAxes, 
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=10)
    
    plt.tight_layout()
    
    # Save the figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Chart saved to: {save_path}")
    
    plt.show()

def create_summary_table(df: pd.DataFrame, region: str = 'all') -> pd.DataFrame:
    """Create a summary table with key statistics."""
    
    summary_stats = []
    
    for attention_type in df.columns:
        values = df[attention_type]
        
        stats = {
            'Attention Type': attention_type,
            'Latest Value (%)': f"{values.iloc[-1]*100:.3f}%",
            'Average (%)': f"{values.mean()*100:.3f}%",
            'Peak Value (%)': f"{values.max()*100:.3f}%", 
            'Peak Period': df.index[values.idxmax()],
            'Growth (pp)': f"{(values.iloc[-1] - values.iloc[0])*100:.3f}pp",
            'Volatility (std)': f"{values.std()*100:.3f}%"
        }
        summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Add total row
    total_latest = df.sum(axis=1).iloc[-1] 
    total_avg = df.sum(axis=1).mean()
    total_peak = df.sum(axis=1).max()
    total_peak_period = df.index[df.sum(axis=1).idxmax()]
    total_growth = (df.sum(axis=1).iloc[-1] - df.sum(axis=1).iloc[0])
    total_vol = df.sum(axis=1).std()
    
    total_row = {
        'Attention Type': 'TOTAL',
        'Latest Value (%)': f"{total_latest*100:.3f}%",
        'Average (%)': f"{total_avg*100:.3f}%", 
        'Peak Value (%)': f"{total_peak*100:.3f}%",
        'Peak Period': total_peak_period,
        'Growth (pp)': f"{total_growth*100:.3f}pp",
        'Volatility (std)': f"{total_vol*100:.3f}%"
    }
    
    summary_df = pd.concat([summary_df, pd.DataFrame([total_row])], ignore_index=True)
    
    return summary_df

def main():
    parser = argparse.ArgumentParser(
        description='Plot climate attention time series from earnings calls data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage - annual aggregation, all regions
    python plot_climate_attention_timeseries.py
    
    # Quarterly data for US only
    python plot_climate_attention_timeseries.py --aggregation quarter --region US
    
    # EU data with custom output
    python plot_climate_attention_timeseries.py --region EU --output results/eu_climate_attention.png
    
    # Specify custom data directory
    python plot_climate_attention_timeseries.py --data-dir /path/to/outputs/firm_level_panel
        """
    )
    
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path("outputs/firm_level_panel"),
        help='Path to firm-level panel data directory'
    )
    
    parser.add_argument(
        '--aggregation',
        choices=['year', 'quarter'],
        default='year',
        help='Time aggregation level'
    )
    
    parser.add_argument(
        '--region', 
        choices=['all', 'US', 'EU'],
        default='all',
        help='Region to analyze'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output path for saving the chart (optional)'
    )
    
    parser.add_argument(
        '--save-summary',
        action='store_true',
        help='Save summary statistics table to CSV'
    )
    
    args = parser.parse_args()
    
    print("📊 Climate Attention Time Series Visualization")
    print("=" * 50)
    print(f"Data directory: {args.data_dir}")
    print(f"Aggregation: {args.aggregation}")
    print(f"Region: {args.region}")
    print()
    
    try:
        # Load data
        print("🔄 Loading attention data...")
        data = load_attention_data(args.data_dir)
        
        if not data:
            raise ValueError("No data files found! Check your data directory path.")
        
        print(f"✅ Loaded {len(data)} attention types")
        
        # Prepare time series data
        print("🔄 Preparing time series data...")
        df = prepare_timeseries_data(data, args.aggregation, args.region)
        
        print(f"✅ Prepared data: {len(df)} time periods, {len(df.columns)} attention types")
        print(f"   Time range: {df.index[0]} to {df.index[-1]}")
        print(f"   Total attention (latest): {df.iloc[-1].sum()*100:.3f}%")
        
        # Create visualization
        print("🔄 Creating visualization...")
        
        region_label = f"({args.region.upper()})" if args.region != 'all' else "(All Regions)"
        aggregation_label = args.aggregation.capitalize()
        
        title = f"Climate Attention Evolution in Earnings Calls {region_label}\n{aggregation_label} Data • Based on Sautner et al. (2023) + Semantic Search Enhancement"
        
        # Generate output path if not provided
        if args.output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"climate_attention_timeseries_{args.region}_{args.aggregation}_{timestamp}.png"
            args.output = Path("outputs/visualizations") / filename
            args.output.parent.mkdir(parents=True, exist_ok=True)
        
        create_stacked_line_chart(df, title, args.aggregation, args.region, args.output)
        
        # Create and display summary table
        print("\n📋 Summary Statistics:")
        summary_df = create_summary_table(df, args.region)
        print(summary_df.to_string(index=False))
        
        # Save summary table if requested
        if args.save_summary:
            summary_path = args.output.parent / f"climate_attention_summary_{args.region}_{args.aggregation}.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"💾 Summary table saved to: {summary_path}")
        
        print(f"\n✅ Visualization completed successfully!")
        print(f"📊 Chart saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.data_dir and not args.data_dir.exists():
            print(f"💡 Make sure the data directory exists: {args.data_dir}")
            print("   Expected structure:")
            print("   └── firm_level_panel/")
            print("       ├── general_climate/")
            print("       ├── opportunity/") 
            print("       ├── physical_risk/")
            print("       ├── transition_risk/")
            print("       └── transparency_disclosure/")

if __name__ == "__main__":
    main()