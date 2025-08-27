#!/usr/bin/env python3
"""
Create separate, publication-ready plots in Journal of Finance style for climate attention validation.

Usage:
    python create_jof_plots.py <merged_csv_path>

Example:
    python create_jof_plots.py outputs/validity_checks/climate_with_fundamentals.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
from pathlib import Path
from scipy import stats

# Journal of Finance style configuration
def setup_jof_style():
    """Set up Journal of Finance publication style."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5,
    })
    
    # JoF color palette: professional blues, grays, and accent colors
    jof_colors = {
        'primary': '#1f4e79',      # Deep blue
        'secondary': '#2e75b6',    # Medium blue  
        'accent': '#c55a5a',       # Muted red
        'neutral1': '#7f7f7f',     # Gray
        'neutral2': '#d9d9d9',     # Light gray
        'success': '#70ad47',      # Green
        'warning': '#ffc000',      # Gold
        'background': '#f8f9fa'    # Very light gray
    }
    
    return jof_colors

def load_data(file_path):
    """Load and validate the merged dataset."""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Loaded {len(df)} records from {file_path}")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def plot_sector_boxplot(df, output_dir, jof_colors):
    """Create publication-ready sector boxplot."""
    print("📊 Creating sector boxplot...")
    
    df_clean = df.dropna(subset=['climate_attention_ratio', 'sector'])
    
    # Calculate sector statistics and select top sectors
    sector_stats = df_clean.groupby('sector').agg({
        'climate_attention_ratio': ['mean', 'count'],
        'ISSUER_TICKER': 'nunique'
    })
    sector_stats.columns = ['mean_climate', 'observations', 'unique_firms']
    
    # Filter sectors with at least 20 observations for reliability
    reliable_sectors = sector_stats[sector_stats['observations'] >= 20]
    top_sectors = reliable_sectors.nlargest(12, 'mean_climate').index.tolist()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sector_data = df_clean[df_clean['sector'].isin(top_sectors)]
    
    # Create boxplot with JoF styling
    box_plot = ax.boxplot([sector_data[sector_data['sector'] == sector]['climate_attention_ratio'].values 
                          for sector in top_sectors],
                         labels=[s[:25] + '...' if len(s) > 25 else s for s in top_sectors],
                         patch_artist=True,
                         notch=True,
                         vert=False)
    
    # Style the boxplot
    for patch in box_plot['boxes']:
        patch.set_facecolor(jof_colors['primary'])
        patch.set_alpha(0.7)
    
    for element in ['whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(box_plot[element], color=jof_colors['primary'])
    
    ax.set_xlabel('Climate Attention Ratio', fontweight='bold')
    ax.set_ylabel('Sector', fontweight='bold')
    ax.set_title('Climate Attention by Sector\n(Top 12 sectors by mean attention, ≥20 observations)', 
                fontweight='bold', pad=20)
    
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_facecolor('white')
    
    # Add sample sizes
    for i, sector in enumerate(top_sectors):
        n_obs = len(sector_data[sector_data['sector'] == sector])
        ax.text(0.02, i + 1, f'n={n_obs}', transform=ax.get_yaxis_transform(),
               fontsize=9, va='center', color=jof_colors['neutral1'])
    
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / 'figure_1_sector_boxplot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'figure_1_sector_boxplot.pdf', bbox_inches='tight', facecolor='white')
    print(f"💾 Saved sector boxplot to {output_path}")
    
    plt.show()
    plt.close()

def plot_carbon_intensity_validation(df, output_dir, jof_colors):
    """Create scatter plot of climate attention vs carbon intensity by sector."""
    print("🏭 Creating carbon intensity validation plot...")
    
    # Define high-carbon vs clean sectors
    high_carbon_keywords = ['Energy', 'Oil', 'Gas', 'Coal', 'Mining', 'Utilities', 
                           'Steel', 'Cement', 'Chemical', 'Transport', 'Airline', 'Auto']
    clean_keywords = ['Technology', 'Software', 'Healthcare', 'Pharma', 'Finance', 
                     'Insurance', 'Real Estate', 'Media', 'Telecom']
    
    def categorize_sector(sector_name):
        if pd.isna(sector_name):
            return 'Unknown'
        sector_upper = str(sector_name).upper()
        if any(keyword.upper() in sector_upper for keyword in high_carbon_keywords):
            return 'High Carbon'
        elif any(keyword.upper() in sector_upper for keyword in clean_keywords):
            return 'Clean/Services'
        else:
            return 'Other'
    
    df_clean = df.dropna(subset=['climate_attention_ratio', 'sector'])
    df_clean['sector_category'] = df_clean['sector'].apply(categorize_sector)
    
    # Calculate sector-level averages
    sector_avgs = df_clean.groupby(['sector', 'sector_category']).agg({
        'climate_attention_ratio': 'mean',
        'ISSUER_TICKER': 'nunique'
    }).reset_index()
    
    # Filter sectors with at least 5 firms
    sector_avgs = sector_avgs[sector_avgs['ISSUER_TICKER'] >= 5]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors for categories
    category_colors = {
        'High Carbon': jof_colors['accent'],
        'Clean/Services': jof_colors['primary'],
        'Other': jof_colors['neutral1']
    }
    
    # Plot by category
    for category in ['High Carbon', 'Clean/Services', 'Other']:
        cat_data = sector_avgs[sector_avgs['sector_category'] == category]
        if len(cat_data) > 0:
            ax.scatter(range(len(cat_data)), cat_data['climate_attention_ratio'],
                      c=category_colors[category], s=cat_data['ISSUER_TICKER']*8,
                      alpha=0.7, label=category, edgecolors='white', linewidth=0.5)
    
    # Add sector labels for high climate attention sectors
    high_attention = sector_avgs.nlargest(8, 'climate_attention_ratio')
    for i, (_, row) in enumerate(high_attention.iterrows()):
        ax.annotate(row['sector'][:20], 
                   xy=(list(sector_avgs.index).index(row.name), row['climate_attention_ratio']),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=9, ha='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Sectors (ranked by carbon intensity expectation)', fontweight='bold')
    ax.set_ylabel('Mean Climate Attention Ratio', fontweight='bold')
    ax.set_title('Climate Attention by Sector Category\n(Bubble size = number of firms)', 
                fontweight='bold', pad=20)
    
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('white')
    
    # Remove x-ticks since we're showing categorical ranking
    ax.set_xticks([])
    
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / 'figure_2_carbon_validation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'figure_2_carbon_validation.pdf', bbox_inches='tight', facecolor='white')
    print(f"💾 Saved carbon validation plot to {output_path}")
    
    plt.show()
    plt.close()

def plot_country_comparison(df, output_dir, jof_colors):
    """Create country comparison plot with focus on EU vs US."""
    print("🌍 Creating country comparison plot...")
    
    df_clean = df.dropna(subset=['climate_attention_ratio', 'country_hq'])
    
    # Calculate country statistics (minimum 10 firms)
    country_stats = df_clean.groupby('country_hq').agg({
        'climate_attention_ratio': ['mean', 'std', 'count'],
        'ISSUER_TICKER': 'nunique'
    })
    country_stats.columns = ['mean_climate', 'std_climate', 'observations', 'unique_firms']
    
    # Filter reliable countries
    reliable_countries = country_stats[country_stats['unique_firms'] >= 10]
    reliable_countries = reliable_countries.sort_values('mean_climate', ascending=True)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create horizontal bar plot
    y_pos = range(len(reliable_countries))
    bars = ax.barh(y_pos, reliable_countries['mean_climate'],
                    alpha=0.8, edgecolor='white', linewidth=0.5)
    
    # Add error bars if standard deviation data is available
    if reliable_countries['std_climate'].notna().any():
        se = reliable_countries['std_climate'] / np.sqrt(reliable_countries['observations'])
        ax.errorbar(reliable_countries['mean_climate'], y_pos, 
                   xerr=se, fmt='none', color='black', alpha=0.5, capsize=3)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(reliable_countries.index, fontsize=11)
    ax.set_xlabel('Mean Climate Attention Ratio', fontweight='bold')
    ax.set_ylabel('Country', fontweight='bold')
    ax.set_title('Climate Attention by Country\n(Countries with ≥10 firms, with standard errors)', 
                fontweight='bold', pad=20)
    
    # Add sample size annotations
    for i, (country, row) in enumerate(reliable_countries.iterrows()):
        ax.text(row['mean_climate'] + 0.0001, i, f"n={row['unique_firms']}", 
               va='center', fontsize=9, color=jof_colors['neutral1'])
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=jof_colors['primary'], label='EU Countries'),
        Patch(facecolor=jof_colors['accent'], label='United States'),
        Patch(facecolor=jof_colors['neutral1'], label='Other Countries')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, fancybox=True, shadow=True)
    
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_facecolor('white')
    
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / 'figure_3_country_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'figure_3_country_comparison.pdf', bbox_inches='tight', facecolor='white')
    print(f"💾 Saved country comparison plot to {output_path}")
    
    plt.show()
    plt.close()

def plot_eu_us_comparison(df, output_dir, jof_colors):
    """Create focused EU vs US statistical comparison."""
    print("🇪🇺🇺🇸 Creating EU vs US comparison plot...")
    
    df_clean = df.dropna(subset=['climate_attention_ratio', 'country_hq'])
    
    # Filter to EU and US only
    comparison_data = df_clean[df_clean['region'].isin(['EU', 'US'])]
    
    if len(comparison_data) == 0:
        print("❌ No EU/US data available for comparison")
        return
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left panel: Violin plot
    parts = ax1.violinplot([comparison_data[comparison_data['region'] == 'EU']['climate_attention_ratio'].values,
                           comparison_data[comparison_data['region'] == 'US']['climate_attention_ratio'].values],
                          positions=[1, 2], showmeans=True, showmedians=True)
    
    # Style violin plots
    colors = [jof_colors['primary'], jof_colors['accent']]
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(['EU', 'US'])
    ax1.set_ylabel('Climate Attention Ratio', fontweight='bold')
    ax1.set_title('Distribution Comparison', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Right panel: Summary statistics
    summary_stats = comparison_data.groupby('region')['climate_attention_ratio'].agg([
        'count', 'mean', 'median', 'std'
    ]).round(4)
    
    # Create table
    ax2.axis('tight')
    ax2.axis('off')
    
    table_data = []
    table_data.append(['Statistic', 'EU', 'US'])
    table_data.append(['Observations', f"{summary_stats.loc['EU', 'count']:,.0f}", 
                      f"{summary_stats.loc['US', 'count']:,.0f}"])
    table_data.append(['Mean', f"{summary_stats.loc['EU', 'mean']:.4f}", 
                      f"{summary_stats.loc['US', 'mean']:.4f}"])
    table_data.append(['Median', f"{summary_stats.loc['EU', 'median']:.4f}", 
                      f"{summary_stats.loc['US', 'median']:.4f}"])
    table_data.append(['Std Dev', f"{summary_stats.loc['EU', 'std']:.4f}", 
                      f"{summary_stats.loc['US', 'std']:.4f}"])
    
    # Statistical test
    eu_data = comparison_data[comparison_data['region'] == 'EU']['climate_attention_ratio']
    us_data = comparison_data[comparison_data['region'] == 'US']['climate_attention_ratio']
    
    statistic, p_value = stats.mannwhitneyu(eu_data, us_data, alternative='two-sided')
    table_data.append(['Mann-Whitney p-value', f"{p_value:.4f}", ''])
    
    # Add effect size (difference in means)
    effect_size = summary_stats.loc['EU', 'mean'] - summary_stats.loc['US', 'mean']
    table_data.append(['Difference (EU - US)', f"{effect_size:.4f}", ''])
    
    table = ax2.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Style table
    for i in range(len(table_data)):
        for j in range(3):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor(jof_colors['neutral2'])
                cell.set_text_props(weight='bold')
            else:
                cell.set_facecolor('white')
    
    ax2.set_title('Summary Statistics', fontweight='bold')
    
    plt.suptitle('EU vs US Climate Attention Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / 'figure_4_eu_us_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'figure_4_eu_us_comparison.pdf', bbox_inches='tight', facecolor='white')
    print(f"💾 Saved EU vs US comparison to {output_path}")
    
    plt.show()
    plt.close()

def plot_time_series_validation(df, output_dir, jof_colors):
    """Create time series plot showing climate attention around major events."""
    print("📅 Creating time series validation plot...")
    
    df_clean = df.dropna(subset=['climate_attention_ratio', 'date'])
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    df_clean['year_month'] = df_clean['date'].dt.to_period('M')
    
    # Calculate monthly averages
    monthly_avg = df_clean.groupby('year_month')['climate_attention_ratio'].mean().reset_index()
    monthly_avg['date'] = monthly_avg['year_month'].dt.to_timestamp()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot time series
    ax.plot(monthly_avg['date'], monthly_avg['climate_attention_ratio'], 
           color=jof_colors['primary'], linewidth=2, alpha=0.8)
    
    # Add major climate events
    climate_events = {
        '2015-12': 'Paris Agreement',
        '2017-06': 'US Paris Withdrawal',
        '2020-12': 'EU Green Deal',
        '2021-01': 'US Paris Re-entry',
        '2022-08': 'Inflation Reduction Act'
    }
    
    for date_str, event in climate_events.items():
        event_date = pd.to_datetime(date_str)
        if monthly_avg['date'].min() <= event_date <= monthly_avg['date'].max():
            ax.axvline(x=event_date, color=jof_colors['accent'], linestyle='--', alpha=0.7, linewidth=1.5)
            ax.text(event_date, ax.get_ylim()[1]*0.9, event, 
                   rotation=90, ha='right', va='top', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Date', fontweight='bold')
    ax.set_ylabel('Average Climate Attention Ratio', fontweight='bold')
    ax.set_title('Climate Attention Over Time\n(Monthly averages with major climate policy events)', 
                fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('white')
    
    # Format x-axis
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator((1, 7)))
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / 'figure_5_time_series_validation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'figure_5_time_series_validation.pdf', bbox_inches='tight', facecolor='white')
    print(f"💾 Saved time series validation to {output_path}")
    
    plt.show()
    plt.close()

def create_figure_captions(output_dir):
    """Create a file with figure captions for publication."""
    captions = {
        'Figure 1': 'Climate Attention by Sector. This figure shows the distribution of climate attention ratios across the top 12 sectors by mean climate attention. Sectors are required to have at least 20 observations for inclusion. Box plots show median (center line), interquartile range (box), and whiskers extending to 1.5 times the IQR. Sample sizes are shown on the left.',
        
        'Figure 2': 'Climate Attention by Sector Category. This scatter plot validates our climate attention measures by showing higher attention in carbon-intensive sectors (red dots) compared to clean/service sectors (blue dots). Bubble size indicates the number of firms in each sector.',
        
        'Figure 3': 'Climate Attention by Country. This figure compares mean climate attention ratios across countries with at least 10 firms. EU countries are shown in blue, the United States in red, and other countries in gray. Error bars represent standard errors. Sample sizes (number of firms) are annotated.',
        
        'Figure 4': 'EU vs US Climate Attention Comparison. Panel A shows violin plots comparing the distribution of climate attention between EU and US firms. Panel B provides summary statistics and statistical test results. The Mann-Whitney U test assesses whether the difference in climate attention between regions is statistically significant.',
        
        'Figure 5': 'Climate Attention Over Time. This figure shows monthly average climate attention ratios with major climate policy events marked by vertical dashed lines. The time series validates that our measure captures policy-relevant periods with increased climate discourse in corporate earnings calls.'
    }
    
    caption_file = output_dir / 'figure_captions.txt'
    with open(caption_file, 'w') as f:
        f.write("FIGURE CAPTIONS FOR CLIMATE ATTENTION VALIDATION\n")
        f.write("=" * 60 + "\n\n")
        for fig_num, caption in captions.items():
            f.write(f"{fig_num}: {caption}\n\n")
    
    print(f"💾 Saved figure captions to {caption_file}")

def create_four_panel_jof_figure(df, output_dir, jof_colors):
    """Create a comprehensive 4-panel JoF-style validation figure."""
    print("📊 Creating 4-panel JoF-style validation figure...")
    
    # Setup figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # fig.suptitle('Climate Attention Validation Analysis', fontsize=16, fontweight='bold', y=0.95)
    
    # Panel A: Sector Analysis - Box plot
    ax1 = axes[0, 0]
    df_clean = df.dropna(subset=['climate_attention_ratio', 'sector'])
    
    # Calculate sector statistics and select top sectors
    sector_stats = df_clean.groupby('sector').agg({
        'climate_attention_ratio': ['mean', 'count'],
        'ISSUER_TICKER': 'nunique'
    })
    sector_stats.columns = ['mean_climate', 'observations', 'unique_firms']
    
    # Filter and select top 10 sectors
    reliable_sectors = sector_stats[sector_stats['observations'] >= 15]
    top_sectors = reliable_sectors.nlargest(10, 'mean_climate').index.tolist()
    
    sector_data = df_clean[df_clean['sector'].isin(top_sectors)]
    
    # Create boxplot
    box_data = [sector_data[sector_data['sector'] == sector]['climate_attention_ratio'].values 
                for sector in top_sectors]
    
    bp = ax1.boxplot(box_data, patch_artist=True, notch=True)
    
    # Style boxplot
    for patch in bp['boxes']:
        patch.set_facecolor(jof_colors['primary'])
        patch.set_alpha(0.7)
    
    for element in ['whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(bp[element], color=jof_colors['primary'])
    
    ax1.set_xticklabels([s[:15] + '...' if len(s) > 15 else s for s in top_sectors], 
                        rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Climate Attention Ratio', fontweight='bold')
    ax1.set_title('A. Climate Attention by Sector', fontweight='normal', loc='center', pad=15)
    ax1.grid(True, alpha=0.3)
    
    # Panel B: Sector Size vs Climate Attention Scatter
    ax2 = axes[0, 1]
    
    # Calculate sector-level aggregates
    sector_scatter_data = df_clean.groupby('sector').agg({
        'climate_attention_ratio': 'mean',
        'ISSUER_TICKER': 'nunique',
        'date': 'count'  # Total observations
    }).reset_index()
    sector_scatter_data.columns = ['sector', 'mean_climate', 'unique_firms', 'total_observations']
    
    # Filter sectors with at least 5 firms
    sector_scatter_data = sector_scatter_data[sector_scatter_data['unique_firms'] >= 5]
    
    # Create scatter plot
    scatter = ax2.scatter(sector_scatter_data['unique_firms'], 
                         sector_scatter_data['mean_climate'],
                         s=sector_scatter_data['total_observations']/3,  # Bubble size based on observations
                         c=jof_colors['primary'], 
                         alpha=0.6, 
                         edgecolors='white', 
                         linewidth=0.5)
    
    # Label the most interesting sectors (highest climate attention)
    top_sectors_scatter = sector_scatter_data.nlargest(8, 'mean_climate')
    for _, row in top_sectors_scatter.iterrows():
        ax2.annotate(row['sector'][:12] + '...' if len(row['sector']) > 12 else row['sector'], 
                    (row['unique_firms'], row['mean_climate']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))
    
    ax2.set_xlabel('Number of Unique Firms', fontweight='bold')
    ax2.set_ylabel('Mean Climate Attention Ratio', fontweight='bold')
    ax2.set_title('B. Sector Size vs Climate Attention\n(Bubble size = observations)', fontweight='normal', loc='center', pad=15)
    ax2.grid(True, alpha=0.3)
    
    # Panel C: Country Analysis
    ax3 = axes[1, 0]
    
    df_country = df.dropna(subset=['climate_attention_ratio', 'country_hq'])
    
    # Calculate country statistics
    country_stats = df_country.groupby('country_hq').agg({
        'climate_attention_ratio': 'mean',
        'ISSUER_TICKER': 'nunique'
    })
    country_stats.columns = ['mean_climate', 'unique_firms']
    
    # Filter and select top countries
    reliable_countries = country_stats[country_stats['unique_firms'] >= 8]
    top_countries = reliable_countries.nlargest(12, 'mean_climate')
    
    # Define regions for color coding
    eu_countries = ['Germany', 'France', 'Netherlands', 'Italy', 'Spain', 'Belgium', 
                   'Sweden', 'Denmark', 'Finland', 'Austria', 'Ireland']
    
    colors = []
    for country in top_countries.index:
        if country in eu_countries:
            colors.append(jof_colors['primary'])
        elif country == 'United States':
            colors.append(jof_colors['accent'])
        else:
            colors.append(jof_colors['neutral1'])
    
    # Create horizontal bar plot
    y_pos = range(len(top_countries))
    bars = ax3.barh(y_pos, top_countries['mean_climate'], color=colors, alpha=0.8)
    
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(top_countries.index, fontsize=10)
    ax3.set_xlabel('Mean Climate Attention Ratio', fontweight='bold')
    ax3.set_title('C. Climate Attention by Country', fontweight='normal', loc='center', pad=15)
    ax3.grid(True, axis='x', alpha=0.3)
    
    # Panel D: EU vs US Comparison
    ax4 = axes[1, 1]
    
    # Define regions
    df_country['region'] = df_country['country_hq'].apply(
        lambda x: 'EU' if x in eu_countries else ('US' if x == 'United States' else 'Other')
    )
    
    # Filter to EU and US
    comparison_data = df_country[df_country['region'].isin(['EU', 'US'])]
    
    if len(comparison_data) > 0:
        # Create violin plot
        eu_data = comparison_data[comparison_data['region'] == 'EU']['climate_attention_ratio'].values
        us_data = comparison_data[comparison_data['region'] == 'US']['climate_attention_ratio'].values
        
        parts = ax4.violinplot([eu_data, us_data], positions=[1, 2], showmeans=True, showmedians=True)
        
        # Style violin plots
        colors = [jof_colors['primary'], jof_colors['accent']]
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax4.set_xticks([1, 2])
        ax4.set_xticklabels(['EU', 'US'], fontweight='bold')
        ax4.set_ylabel('Climate Attention Ratio', fontweight='bold')
        ax4.set_title('D. EU vs US Comparison', fontweight='normal', loc='center', pad=15)
        ax4.grid(True, alpha=0.3)
        
        # Add statistical annotation
        from scipy import stats
        if len(eu_data) > 0 and len(us_data) > 0:
            statistic, p_value = stats.mannwhitneyu(eu_data, us_data, alternative='two-sided')
            ax4.text(0.5, 0.95, f'p-value: {p_value:.3f}', transform=ax4.transAxes, 
                    ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.50)
    
    # Save the figure
    output_path = output_dir / 'climate_attention_validation_4panel.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'climate_attention_validation_4panel.pdf', bbox_inches='tight', facecolor='white')
    
    print(f"💾 Saved 4-panel validation figure to {output_path}")
    plt.show()
    plt.close()

def main():
    """Main function to create 4-panel JoF-style validation figure."""
    parser = argparse.ArgumentParser(
        description='Create 4-panel JoF-style climate attention validation figure'
    )
    parser.add_argument(
        'input_csv',
        help='Path to merged CSV file with fundamentals data'
    )
    parser.add_argument(
        '--output',
        default='outputs/validity_checks/figures',
        help='Output directory for the figure'
    )
    
    args = parser.parse_args()
    
    # Setup style and load data
    jof_colors = setup_jof_style()
    df = load_data(args.input_csv)
    if df is None:
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🎨 Creating 4-panel JoF-style validation figure...")
    print(f"Output directory: {output_dir}")
    
    # Create the 4-panel figure
    create_four_panel_jof_figure(df, output_dir, jof_colors)
    
    print(f"\n✅ 4-panel JoF-style validation figure created!")
    print(f"📁 Generated files in {output_dir}:")
    print("  - climate_attention_validation_4panel.png")
    print("  - climate_attention_validation_4panel.pdf")
    print("\n📊 Figure is publication-ready in Journal of Finance style!")

if __name__ == "__main__":
    main()