import numpy as np
import pandas as pd
import sys

from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))

from ap_pipeline.hedge_method import run_cross_sectional_factor_regression

from ap_pipeline.Z_clim import clim_matrix

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_engle_dataset(df_engle):
    """
    Comprehensive analysis of the Engle dataset for hedge portfolio construction.
    """
    print("📊 ENGLE DATASET COMPREHENSIVE ANALYSIS")
    print("=" * 60)
    
    # Convert MONTH to datetime if needed
    df_engle['MONTH'] = pd.to_datetime(df_engle['MONTH'])
    
    # ==========================================
    # BASIC DATASET INFO
    # ==========================================
    print("\n📋 BASIC DATASET INFO")
    print("-" * 30)
    print(f"Total observations: {len(df_engle):,}")
    print(f"Total columns: {len(df_engle.columns)}")
    print(f"Memory usage: {df_engle.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print(f"Columns: {list(df_engle.columns)}")
    
    # ==========================================
    # TEMPORAL COVERAGE
    # ==========================================
    print("\n📅 TEMPORAL COVERAGE")
    print("-" * 30)
    
    print(f"Date range: {df_engle['MONTH'].min().strftime('%Y-%m')} to {df_engle['MONTH'].max().strftime('%Y-%m')}")
    print(f"Total months: {df_engle['MONTH'].nunique()}")
    
    # Year-by-year coverage
    year_stats = df_engle.groupby(df_engle['MONTH'].dt.year).agg({
        'ISSUER_TICKER': ['count', 'nunique'],
        'MONTH': 'nunique'
    }).round(1)
    year_stats.columns = ['Total_Obs', 'Unique_Firms', 'Months_in_Year']
    year_stats['Avg_Obs_per_Month'] = year_stats['Total_Obs'] / year_stats['Months_in_Year']
    
    print(f"\nYear-by-year breakdown:")
    print(year_stats)
    
    # ==========================================
    # FIRM COVERAGE
    # ==========================================
    print("\n🏢 FIRM COVERAGE")
    print("-" * 30)
    
    print(f"Unique firms: {df_engle['ISSUER_TICKER'].nunique()}")
    
    # Firms per month statistics
    firms_per_month = df_engle.groupby('MONTH')['ISSUER_TICKER'].nunique()
    print(f"\nFirms per month statistics:")
    print(f"  Mean: {firms_per_month.mean():.1f}")
    print(f"  Median: {firms_per_month.median():.1f}")
    print(f"  Min: {firms_per_month.min()}")
    print(f"  Max: {firms_per_month.max()}")
    print(f"  Std: {firms_per_month.std():.1f}")
    
    # Observations per firm
    obs_per_firm = df_engle['ISSUER_TICKER'].value_counts()
    print(f"\nObservations per firm statistics:")
    print(f"  Mean: {obs_per_firm.mean():.1f}")
    print(f"  Median: {obs_per_firm.median():.1f}")
    print(f"  Min: {obs_per_firm.min()}")
    print(f"  Max: {obs_per_firm.max()}")
    
    # Panel balance
    expected_obs = df_engle['MONTH'].nunique() * df_engle['ISSUER_TICKER'].nunique()
    balance_ratio = len(df_engle) / expected_obs
    print(f"\nPanel balance: {balance_ratio:.1%} (1.0 = perfectly balanced)")
    
    # ==========================================
    # KEY VARIABLES ANALYSIS
    # ==========================================
    print("\n📈 KEY VARIABLES ANALYSIS")
    print("-" * 30)
    
    # Check for key Engle methodology columns
    key_cols = ['Stock_Return', 'WSJ.AR1', 'UMC.ALL', 'Z_MKT', 'Z_SIZE', 'Z_HML']
    available_cols = [col for col in key_cols if col in df_engle.columns]
    missing_cols = [col for col in key_cols if col not in df_engle.columns]
    
    print(f"Available key columns: {available_cols}")
    if missing_cols:
        print(f"Missing key columns: {missing_cols}")
    
    # Data completeness for each key variable
    print(f"\nData completeness:")
    for col in available_cols:
        non_null = df_engle[col].notna().sum()
        print(f"  {col}: {non_null:,} / {len(df_engle):,} ({non_null/len(df_engle)*100:.1f}%)")
    
    # Summary statistics for numeric columns
    if available_cols:
        numeric_cols = df_engle[available_cols].select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"\nSummary statistics:")
            print(df_engle[numeric_cols].describe().round(4))
    
    # ==========================================
    # CLIMATE CONCERN VARIABLES
    # ==========================================
    print("\n🌍 CLIMATE CONCERN VARIABLES")
    print("-" * 30)
    
    climate_cols = [col for col in df_engle.columns if any(term in col.upper() for term in ['WSJ', 'UMC', 'CLIM', 'GREEN'])]
    
    if climate_cols:
        print(f"Climate-related columns found: {climate_cols}")
        
        for col in climate_cols:
            if df_engle[col].dtype in ['float64', 'int64']:
                print(f"\n{col} statistics:")
                print(f"  Non-null: {df_engle[col].notna().sum():,} ({df_engle[col].notna().sum()/len(df_engle)*100:.1f}%)")
                print(f"  Mean: {df_engle[col].mean():.4f}")
                print(f"  Std: {df_engle[col].std():.4f}")
                print(f"  Min: {df_engle[col].min():.4f}")
                print(f"  Max: {df_engle[col].max():.4f}")
    else:
        print("No climate-related columns detected")
    
    # ==========================================
    # FACTOR VARIABLES
    # ==========================================
    print("\n📊 FACTOR VARIABLES")
    print("-" * 30)
    
    factor_cols = [col for col in df_engle.columns if col.startswith('Z_')]
    
    if factor_cols:
        print(f"Factor columns found: {factor_cols}")
        
        # Check factor properties (should be standardized)
        for col in factor_cols:
            if df_engle[col].dtype in ['float64', 'int64']:
                mean_val = df_engle[col].mean()
                std_val = df_engle[col].std()
                print(f"  {col}: Mean={mean_val:.4f}, Std={std_val:.4f}")
    else:
        print("No factor columns (Z_*) found")
    
    # ==========================================
    # POTENTIAL DATA ISSUES
    # ==========================================
    print("\n⚠️  POTENTIAL DATA ISSUES")
    print("-" * 30)
    
    # Check for duplicates
    duplicates = df_engle.duplicated(subset=['MONTH', 'ISSUER_TICKER']).sum()
    print(f"Duplicate (MONTH, ISSUER_TICKER) pairs: {duplicates}")
    
    # Check for missing tickers or dates
    missing_tickers = df_engle['ISSUER_TICKER'].isna().sum()
    missing_months = df_engle['MONTH'].isna().sum()
    print(f"Missing tickers: {missing_tickers}")
    print(f"Missing months: {missing_months}")
    
    # Check date continuity
    month_gaps = pd.Series(df_engle['MONTH'].unique()).sort_values().diff().dt.days
    irregular_gaps = month_gaps[(month_gaps > 35) | (month_gaps < 25)]
    if len(irregular_gaps) > 0:
        print(f"Irregular month gaps detected: {len(irregular_gaps)} instances")
    else:
        print("Month spacing appears regular")
    
    return {
        'total_obs': len(df_engle),
        'unique_firms': df_engle['ISSUER_TICKER'].nunique(),
        'unique_months': df_engle['MONTH'].nunique(),
        'date_range': (df_engle['MONTH'].min(), df_engle['MONTH'].max()),
        'firms_per_month': firms_per_month.describe(),
        'available_key_cols': available_cols,
        'climate_cols': climate_cols,
        'factor_cols': factor_cols,
        'panel_balance': balance_ratio
    }

def plot_engle_time_series(df_engle):
    """
    Create time series plots for the Engle dataset.
    """
    df_engle['MONTH'] = pd.to_datetime(df_engle['MONTH'])
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Engle Dataset Time Series Analysis', fontsize=16)
    
    # Plot 1: Number of firms over time
    firms_over_time = df_engle.groupby('MONTH')['ISSUER_TICKER'].nunique()
    axes[0, 0].plot(firms_over_time.index, firms_over_time.values)
    axes[0, 0].set_title('Number of Firms Over Time')
    axes[0, 0].set_ylabel('Number of Firms')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Climate concern over time (if available)
    if 'UMC.ALL' in df_engle.columns:
        climate_over_time = df_engle.groupby('MONTH')['UMC.ALL'].first()  # Same value for all firms in month
        axes[0, 1].plot(climate_over_time.index, climate_over_time.values)
        axes[0, 1].set_title('Climate Concern (UMC.ALL) Over Time')
        axes[0, 1].set_ylabel('Climate Concern')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Average returns over time
    if 'Stock_Return' in df_engle.columns:
        returns_over_time = df_engle.groupby('MONTH')['Stock_Return'].mean()
        axes[1, 0].plot(returns_over_time.index, returns_over_time.values)
        axes[1, 0].set_title('Average Stock Returns Over Time')
        axes[1, 0].set_ylabel('Average Return')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Data completeness over time
    completeness_over_time = df_engle.groupby('MONTH').apply(
        lambda x: x.notna().mean().mean()
    )
    axes[1, 1].plot(completeness_over_time.index, completeness_over_time.values)
    axes[1, 1].set_title('Data Completeness Over Time')
    axes[1, 1].set_ylabel('Completeness Ratio')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

clim_ratio_path = "/Users/marleendejonge/Desktop/ECC-data-generation/data/asset_pricing/df_fin_sus_sem2608.csv"
df_clim_ratio = pd.read_csv(clim_ratio_path)

weights = clim_matrix(df_clim_ratio, name_col='climate_transition_risk_ratio', save=True, output_path="outputs/engle/Z_clim.csv")

engle_path = "data/asset_pricing/hedge_df_engle_full_1905_extended.csv"
factor_path = "outputs/engle/Z_clim.csv"

df_engle = pd.read_csv(engle_path)
df_clim = weights

stats = analyze_engle_dataset(df_engle)
plot_engle_time_series(df_engle)

start_year = 2010
end_year = 2019

run_cross_sectional_factor_regression(
    df_engle=df_engle,
    df_clim=df_clim,
    clim_col='climate_transition_risk_ratio_R',
    concern_col='UMC.ALL',
    start_year=2010,
    end_year=2019
)

weights = clim_matrix(df_clim_ratio, name_col='climate_physical_risk_ratio', save=True, output_path="outputs/engle/Z_clim.csv")
df_clim = weights

run_cross_sectional_factor_regression(
    df_engle=df_engle,
    df_clim=df_clim,
    clim_col='climate_physical_risk_ratio_R',
    concern_col='UMC.ALL',
    start_year=2010,
    end_year=2019
)

weights = clim_matrix(df_clim_ratio, name_col='climate_drought_ratio', save=True, output_path="outputs/engle/Z_clim.csv")
df_clim = weights

run_cross_sectional_factor_regression(
    df_engle=df_engle,
    df_clim=df_clim,
    clim_col='climate_drought_ratio_R',
    concern_col='UMC.Water.Drought',
    start_year=2010,
    end_year=2019
)