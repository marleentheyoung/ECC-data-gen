import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import ast
from datetime import datetime

from sklearn.preprocessing import StandardScaler

def rank_with_zero_floor(series):
    series = series.copy()

    mask = series > 0.9999
    ranked = series[~mask].rank(method='average')
    standardized = (ranked - (len(ranked)+1)/2) / len(ranked)

    result = pd.Series(index=series.index, dtype=float)
    result[mask] = 0.499999
    result[~mask] = standardized

    return result

def forward_fill_quarterly_to_monthly(ratio_df: pd.DataFrame, 
                                     forward_months: int = 2) -> pd.DataFrame:
    """
    Forward-fill quarterly earnings call data to monthly observations.
    
    Takes quarterly earnings call climate attention data and creates monthly 
    observations by carrying forward each quarterly observation for the 
    specified number of months.
    
    Args:
        ratio_df: DataFrame with quarterly earnings call data
        forward_months: Number of months to carry each observation forward
        
    Returns:
        DataFrame with monthly observations, keeping most recent call per month
    """
    print(f"📅 Forward-filling quarterly data for {forward_months} months...")
    
    # Ensure date column is datetime
    ratio_df = ratio_df.copy()
    ratio_df['date'] = pd.to_datetime(ratio_df['date'])
    
    # Generate shifted versions (original + forward months)
    shifted_dfs = [ratio_df.copy()]  # Original observations
    
    for i in range(1, forward_months + 1):
        shifted = ratio_df.copy()
        shifted['date'] = shifted['date'] + pd.DateOffset(months=i)
        shifted_dfs.append(shifted)
    
    # Concatenate all shifted versions
    monthly_df = pd.concat(shifted_dfs, ignore_index=True)
    
    # Align to month-end for consistent grouping
    monthly_df['MONTH'] = monthly_df['date'] + pd.offsets.MonthEnd(0)
    
    # Keep only the latest observation per (ticker, month)
    # This ensures we use the most recent earnings call available for each month
    monthly_df = monthly_df.sort_values(['ISSUER_TICKER', 'MONTH', 'date'])
    monthly_df = monthly_df.drop_duplicates(
        subset=['ISSUER_TICKER', 'MONTH'], 
        keep='last'
    )
    
    # Clean up and sort
    monthly_df = monthly_df.sort_values(['ISSUER_TICKER', 'date']).reset_index(drop=True)
    monthly_df = monthly_df.drop_duplicates()
    
    print(f"✅ Created {len(monthly_df)} monthly observations from {len(ratio_df)} quarterly observations")
    
    return monthly_df

def clim_matrix(ratio_df, name_col, output_path: str = "outputs/engle/Z_clim.csv", save=False):
    """
    Create climate factor matrix with Engle-style normalizations.
    
    Args:
        ratio_df: DataFrame with quarterly climate attention ratios
        output_path: Path to save output CSV
        save: Whether to save to file
        
    Returns:
        DataFrame with normalized climate factors
    """
    print("📥 Processing climate attention data...")

    # Convert quarterly to monthly observations
    ratio_df_monthly = forward_fill_quarterly_to_monthly(ratio_df, forward_months=2)

    # Select relevant columns
    clim_combined = ratio_df_monthly[[
        'ISSUER_TICKER', 'MONTH', 'company_name', 'year', 
        'quarter', 'date', name_col
    ]].drop_duplicates()

    # 📊 RANK-BASED normalization (Engle's ranked scores)
    print("📊 Rank-normalizing factors within each month...")
    
    rank_cols = [name_col]

    for col in rank_cols:
        for t in clim_combined['MONTH'].unique():
            subset = clim_combined[clim_combined['MONTH'] == t]
            values = subset[col]

            # Reverse ranking: highest value gets lowest rank
            ranked = values.rank(method='average', ascending=False)
            standardized = (ranked - (len(values) + 1) / 2) / len(values)

            clim_combined.loc[subset.index, f"{col}_R"] = standardized.astype(np.float32)
    
    # 📊 ABSOLUTE cross-sectional normalization (Engle's demeaned scores, reversed)
    print("📊 Demeaning (absolute) factors within each month (reversed)...")
    for col in rank_cols:
        col_abs = f"{col}_A"
        clim_combined[col_abs] = np.nan

        for t in clim_combined['MONTH'].unique():
            subset = clim_combined[clim_combined['MONTH'] == t]
            demeaned = subset[col] - subset[col].mean()
            # Flip the sign to reverse interpretation
            clim_combined.loc[subset.index, col_abs] = (-demeaned).astype(np.float32)
    
    # 📊 STANDARDIZED absolute normalization
    for col in rank_cols:
        col_abs = f"{col}_AS"
        clim_combined[col_abs] = np.nan

        for t in clim_combined['MONTH'].unique():
            subset = clim_combined[clim_combined['MONTH'] == t]
            mean = subset[col].mean()
            std = subset[col].std()

            # Standardized absolute score (mean 0, std 1), and reverse if needed
            standardized = -(subset[col] - mean) / std
            clim_combined.loc[subset.index, col_abs] = standardized.astype(np.float32)
    
    # Fill NaN values with 0
    factor_cols = [f'{name_col}_R', f'{name_col}_A', f'{name_col}_AS']
    clim_combined[factor_cols] = clim_combined[factor_cols].fillna(0)

    if save:
        clim_combined.to_csv(output_path, index=False)
        print(f"✅ Saved climate factors to {output_path}")
    
    return clim_combined
