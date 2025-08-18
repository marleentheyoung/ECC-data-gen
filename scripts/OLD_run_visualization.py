#!/usr/bin/env python3
"""
Simple JOF-style climate visualization script.

Creates journal-style plots for climate exposure and risk data,
similar to the attached image format.

Usage:
    python run_visualization.py

Author: Marleen de Jonge
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime


def setup_jof_style():
    """Set up JOF publication style."""
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.linewidth': 1,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white'
    })


def get_climate_events():
    """Get major climate policy events for vertical lines."""
    return {
        "2009-12": "Copenhagen Summit",
        "2015-12": "Paris Agreement", 
        "2017-06": "US Paris Withdrawal",
        "2019-12": "EU Green Deal",
        "2021-01": "US Paris Re-entry",
        "2022-08": "US IRA"
    }


def load_exposure_data(start_year=None, end_year=None):
    """Load climate exposure data."""
    file_path = Path("/Users/marleendejonge/Desktop/ECC-data-generation/outputs/variables/total_exposure/climate_exposure_ratios.csv")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Exposure data not found: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"📊 Initial data shape: {df.shape}")
    
    # Convert date and handle invalid dates
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Check for and remove rows with invalid dates
    initial_count = len(df)
    df = df.dropna(subset=['date'])
    final_count = len(df)
    
    if initial_count != final_count:
        print(f"⚠️  Removed {initial_count - final_count} rows with invalid dates")
    
    if len(df) == 0:
        raise ValueError("No valid dates found in the data")
    
    # Filter by year range if specified
    if start_year is not None:
        df = df[df['date'].dt.year >= start_year]
        print(f"📅 Filtered data from {start_year} onwards")
    
    if end_year is not None:
        df = df[df['date'].dt.year <= end_year]
        print(f"📅 Filtered data up to {end_year}")
    
    if len(df) == 0:
        raise ValueError(f"No data found in the specified year range: {start_year}-{end_year}")
    
    # Create monthly periods
    df['year_month'] = df['date'].dt.to_period('M')
    
    # Determine region based on source_index
    df['region'] = df['source_index'].map({'SP500': 'US', 'STOXX600': 'EU'})
    
    # Remove rows with unknown regions
    df = df.dropna(subset=['region'])
    
    # Calculate monthly averages by region
    monthly_data = df.groupby(['year_month', 'region']).agg({
        'climate_exposure': 'mean'
    }).reset_index()
    
    monthly_data['date'] = monthly_data['year_month'].dt.to_timestamp()
    
    print(f"✅ Loaded exposure data: {len(monthly_data)} monthly observations")
    print(f"📅 Date range: {monthly_data['date'].min()} to {monthly_data['date'].max()}")
    return monthly_data


def load_risk_data(start_year=None, end_year=None):
    """Load climate risk data."""
    file_path = Path("/Users/marleendejonge/Desktop/ECC-data-generation/outputs/variables/cc_risk/full_df.csv")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Risk data not found: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"📊 Initial data shape: {df.shape}")
    
    # Convert date and handle invalid dates
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Check for and remove rows with invalid dates
    initial_count = len(df)
    df = df.dropna(subset=['date'])
    final_count = len(df)
    
    if initial_count != final_count:
        print(f"⚠️  Removed {initial_count - final_count} rows with invalid dates")
    
    if len(df) == 0:
        raise ValueError("No valid dates found in the data")
    
    # Filter by year range if specified
    if start_year is not None:
        df = df[df['date'].dt.year >= start_year]
        print(f"📅 Filtered data from {start_year} onwards")
    
    if end_year is not None:
        df = df[df['date'].dt.year <= end_year]
        print(f"📅 Filtered data up to {end_year}")
    
    if len(df) == 0:
        raise ValueError(f"No data found in the specified year range: {start_year}-{end_year}")
    
    # Create monthly periods
    df['year_month'] = df['date'].dt.to_period('M')
    
    # Determine region based on source_index
    df['region'] = df['source_index'].map({'SP500': 'US', 'STOXX600': 'EU'})
    
    # Remove rows with unknown regions
    df = df.dropna(subset=['region'])
    
    # Calculate monthly averages by region
    monthly_data = df.groupby(['year_month', 'region']).agg({
        'climate_risk_exposure': 'mean'
    }).reset_index()
    
    monthly_data['date'] = monthly_data['year_month'].dt.to_timestamp()
    
    print(f"✅ Loaded risk data: {len(monthly_data)} monthly observations")
    print(f"📅 Date range: {monthly_data['date'].min()} to {monthly_data['date'].max()}")
    return monthly_data


def plot_jof_style_single_region(data, variable_name, title, region, output_path, year_range=[2010,2025]):
    """
    Create JOF-style plot for a single region.
    
    Args:
        data: DataFrame with regional data
        variable_name: Column name for the variable to plot
        title: Plot title
        region: 'US' or 'EU'
        output_path: Output file path
    """
    setup_jof_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get regional data
    region_data = data[data['region'] == region].copy().sort_values('date')
    
    if len(region_data) == 0:
        print(f"⚠️  No data found for {region}")
        return
    
    date_con = (region_data.date.dt.year >= year_range[0]) & (region_data.date.dt.year <= year_range[1])
    
    region_data = region_data.loc[date_con]

    # Calculate 3-month rolling average
    region_data['rolling_3m'] = region_data[variable_name].rolling(window=3, center=True).mean()
    
    # Plot main line with region-specific styling
    if region == 'US':
        color = '#1f4e79'  # Dark blue
        label = 'S&P 500'
        linestyle = '-'
    else:  # EU
        color = '#70ad47'  # Green
        label = 'STOXX 600'
        linestyle = '-'
    
    ax.plot(region_data['date'], region_data['rolling_3m'], 
            color=color, linewidth=2.5, label=label, linestyle=linestyle)
    
    # Add climate events as vertical lines
    events = get_climate_events()
    
    # Determine y-axis range for event labels
    valid_values = region_data['rolling_3m'].dropna()
    if len(valid_values) > 0:
        y_min, y_max = valid_values.min(), valid_values.max()
        y_range = y_max - y_min
        plot_y_max = y_max + (y_range * 0.2)
        
        # Set y-axis limits
        ax.set_ylim(max(0, y_min - (y_range * 0.05)), plot_y_max)
        
        # Get date range
        min_date, max_date = region_data['date'].min(), region_data['date'].max()
        
        for i, (event_date, event_name) in enumerate(events.items()):
            try:
                event_datetime = pd.to_datetime(event_date + '-01')
                if min_date <= event_datetime <= max_date:
                    ax.axvline(event_datetime, color='#c5504b', alpha=0.7, 
                             linestyle='--', linewidth=1.5)
                    
                    # Alternate label heights to avoid overlap
                    label_height = plot_y_max * (0.85 + 0.1 * (i % 2))
                    ax.text(event_datetime, label_height, event_name, 
                           rotation=90, ha='right', va='top', fontsize=9, 
                           color='#c5504b', alpha=0.9, weight='normal')
            except:
                continue
    
    # Format axes in JOF style
    ax.set_xlabel('Year', fontsize=12, color='black')
    ax.set_ylabel('Climate Attention Ratio', fontsize=12, color='black')
    ax.set_title(f'{title} - {region}', fontsize=13, fontweight='bold', pad=15, color='black')
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_minor_locator(mdates.YearLocator())
    
    # Format y-axis with more precision
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.4f}'))
    
    # JOF styling: remove top and right spines, keep left and bottom
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    
    # Add subtle grid lines (JOF style)
    ax.grid(True, linestyle='-', linewidth=0.3, alpha=0.3, color='gray')
    ax.set_axisbelow(True)
    
    # Legend with JOF styling
    ax.legend(loc='upper left', frameon=False, fontsize=11)
    
    # Set white background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Tick styling
    ax.tick_params(colors='black', which='both')
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"📊 Saved {output_path}")


def plot_jof_style_regional(data, variable_name, title, output_path, year_range=[2009,2025]):
    """
    Create JOF-style regional comparison plot.
    
    Args:
        data: DataFrame with regional data
        variable_name: Column name for the variable to plot
        title: Plot title
        output_path: Output file path
    """
    setup_jof_style()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Get regional data
    us_data = data[data['region'] == 'US'].copy().sort_values('date')
    eu_data = data[data['region'] == 'EU'].copy().sort_values('date')

    date_con_us = (us_data.date.dt.year >= year_range[0]) & (us_data.date.dt.year <= year_range[1])
    date_con_eu = (eu_data.date.dt.year >= year_range[0]) & (eu_data.date.dt.year <= year_range[1])
    
    us_data = us_data.loc[date_con_us]
    eu_data = eu_data.loc[date_con_eu]

    # Calculate 3-month rolling averages and plot
    if len(us_data) > 0:
        us_data['rolling_3m'] = us_data[variable_name].rolling(window=3, center=True).mean()
        ax.plot(us_data['date'], us_data['rolling_3m'], 
                color='#1f4e79', linewidth=2.5, label='S&P 500', linestyle='-')
    
    if len(eu_data) > 0:
        eu_data['rolling_3m'] = eu_data[variable_name].rolling(window=3, center=True).mean()
        ax.plot(eu_data['date'], eu_data['rolling_3m'], 
                color='#70ad47', linewidth=2.5, label='STOXX 600', linestyle='--')
    
    # Add climate events as vertical lines
    events = get_climate_events()
    
    # Determine y-axis range for event labels
    all_values = []
    if len(us_data) > 0 and us_data['rolling_3m'].notna().any():
        all_values.extend(us_data['rolling_3m'].dropna().values)
    if len(eu_data) > 0 and eu_data['rolling_3m'].notna().any():
        all_values.extend(eu_data['rolling_3m'].dropna().values)
    
    if all_values:
        y_min, y_max = min(all_values), max(all_values)
        y_range = y_max - y_min
        plot_y_max = y_max + (y_range * 0.25)
        
        # Set y-axis limits
        ax.set_ylim(max(0, y_min - (y_range * 0.05)), plot_y_max)
        
        # Get date range
        all_dates = []
        if len(us_data) > 0:
            all_dates.extend(us_data['date'].tolist())
        if len(eu_data) > 0:
            all_dates.extend(eu_data['date'].tolist())
        
        if all_dates:
            min_date, max_date = min(all_dates), max(all_dates)
            
            for i, (event_date, event_name) in enumerate(events.items()):
                try:
                    event_datetime = pd.to_datetime(event_date + '-01')
                    if min_date <= event_datetime <= max_date:
                        ax.axvline(event_datetime, color='#c5504b', alpha=0.7, 
                                 linestyle='--', linewidth=1.5)
                        
                        # Alternate label heights
                        label_height = plot_y_max * (0.8 + 0.15 * (i % 2))
                        ax.text(event_datetime, label_height, event_name, 
                               rotation=90, ha='right', va='top', fontsize=9, 
                               color='#c5504b', alpha=0.9, weight='normal')
                except:
                    continue
    
    # Format axes in JOF style
    ax.set_xlabel('Year', fontsize=12, color='black')
    ax.set_ylabel('Climate Attention Ratio', fontsize=12, color='black')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20, color='black')
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_minor_locator(mdates.YearLocator())
    
    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.4f}'))
    
    # JOF styling: remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    
    # Add subtle grid lines
    ax.grid(True, linestyle='-', linewidth=0.3, alpha=0.3, color='gray')
    ax.set_axisbelow(True)
    
    # Legend with JOF styling
    ax.legend(loc='upper left', frameon=False, fontsize=11)
    
    # Set white background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Tick styling
    ax.tick_params(colors='black', which='both')
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"📊 Saved {output_path}")

def main():
    """Main execution function."""
    print("📊 Creating JOF-Style Climate Plots")
    print("=" * 40)
    
    # Create output directory
    output_dir = Path("/Users/marleendejonge/Desktop/ECC-data-generation/outputs/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Climate Exposure Plot
        print("\n📥 Loading climate exposure data...")
        exposure_data = load_exposure_data()
        
        plot_jof_style_regional(
            data=exposure_data,
            variable_name='climate_exposure',
            title='Climate Exposure in Earnings Calls',
            output_path=output_dir / 'jof_climate_exposure.png',
            year_range=[2011,2025]
        )
        
        plot_jof_style_single_region(
            data=exposure_data,
            variable_name='climate_exposure',
            title='Climate Exposure in STOXX600 Earnings Calls',
            region='EU',
            output_path=output_dir / 'jof_climate_exposure_EU.png',
            year_range=[2011,2025]
        )

        plot_jof_style_single_region(
            data=exposure_data,
            variable_name='climate_exposure',
            title='Climate Exposure in SP500 Earnings Calls',
            region='US',
            output_path=output_dir / 'jof_climate_exposure_US.png',
            year_range=[2010,2025]
        )

        # 2. Climate Risk Plot
        print("\n📥 Loading climate risk data...")
        risk_data = load_risk_data()
        
        plot_jof_style_regional(
            data=risk_data,
            variable_name='climate_risk_exposure',
            title='Climate Risk Exposure in Earnings Calls',
            output_path=output_dir / 'jof_climate_risk.png'
        )
        
        print(f"\n✅ Plots completed!")
        print(f"📁 Saved to: {output_dir}")
        print(f"  • jof_climate_exposure.png - Overall climate exposure")
        print(f"  • jof_climate_risk.png - Climate risk exposure")
        
    except FileNotFoundError as e:
        print(f"❌ Data file not found: {e}")
        print("💡 Please run the exposure and risk analysis scripts first:")
        print("   python semantic_cc_exposure.py")
        print("   python semantic_cc_risk.py")
        
    except Exception as e:
        print(f"❌ Error creating plots: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()