import pandas as pd
import os

import pandas as pd
import numpy as np


# Define file paths
base_path = "outputs/firm_level_panel"
file_configs = {
    'general_climate': {
        'path': f"{base_path}/general_climate/firm_level_climate_attention.csv",
        'main_cols': ['climate_attention_ratio'],
        'lag_cols': ['climate_attention_ratio_lag1', 'climate_attention_ratio_lag2']
    },
    'opportunity': {
        'path': f"{base_path}/opportunity/firm_level_opportunity_attention.csv", 
        'main_cols': ['climate_opportunity_ratio'],
        'lag_cols': ['climate_opportunity_ratio_lag1', 'climate_opportunity_ratio_lag2']
    },
    'flood': {
        'path': f"{base_path}/flood/firm_level_flood_attention.csv", 
        'main_cols': ['climate_flood_ratio'],
        'lag_cols': ['climate_flood_ratio_lag1', 'climate_flood_ratio_lag2']
    },
    'drought': {
        'path': f"{base_path}/drought/firm_level_drought_attention.csv", 
        'main_cols': ['climate_drought_ratio'],
        'lag_cols': ['climate_drought_ratio_lag1', 'climate_drought_ratio_lag2']
    },
    'physical_risk': {
        'path': f"{base_path}/physical_risk/firm_level_physical_risk_attention.csv",
        'main_cols': ['climate_physical_risk_ratio', 'acute_physical_risk_ratio', 'chronic_physical_risk_ratio'],
        'lag_cols': ['climate_physical_risk_ratio_lag1', 'climate_physical_risk_ratio_lag2',
                    'acute_physical_risk_ratio_lag1', 'acute_physical_risk_ratio_lag2',
                    'chronic_physical_risk_ratio_lag1', 'chronic_physical_risk_ratio_lag2']
    },
    'transition_risk': {
        'path': f"{base_path}/transition_risk/firm_level_transition_risk_attention.csv",
        'main_cols': ['climate_transition_risk_ratio', 'policy_transition_risk_ratio', 
                     'technology_transition_risk_ratio', 'market_transition_risk_ratio'],
        'lag_cols': ['climate_transition_risk_ratio_lag1', 'climate_transition_risk_ratio_lag2',
                    'policy_transition_risk_ratio_lag1', 'policy_transition_risk_ratio_lag2',
                    'technology_transition_risk_ratio_lag1', 'technology_transition_risk_ratio_lag2',
                    'market_transition_risk_ratio_lag1', 'market_transition_risk_ratio_lag2']
    },
    'transparency_disclosure': {
        'path': f"{base_path}/transparency_disclosure/firm_level_transparency_disclosure_attention.csv",
        'main_cols': ['climate_transparency_disclosure_ratio', 'esg_reporting_ratio', 
                     'emissions_disclosure_ratio', 'governance_strategy_ratio', 'regulatory_compliance_ratio'],
        'lag_cols': ['climate_transparency_disclosure_ratio_lag1', 'climate_transparency_disclosure_ratio_lag2',
                    'esg_reporting_ratio_lag1', 'esg_reporting_ratio_lag2',
                    'emissions_disclosure_ratio_lag1', 'emissions_disclosure_ratio_lag2',
                    'governance_strategy_ratio_lag1', 'governance_strategy_ratio_lag2',
                    'regulatory_compliance_ratio_lag1', 'regulatory_compliance_ratio_lag2']
    }
}

def load_and_select_columns(file_path, main_cols, lag_cols, merge_keys):
    """Load dataframe and select only required columns"""
    try:
        df = pd.read_csv(file_path)
        
        # Convert date column to ensure proper merging
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Remove duplicates FIRST - keep first occurrence, drop subsequent ones
        initial_rows = len(df)
        df = df.drop_duplicates(subset=['ISSUER_TICKER', 'date'], keep='first')
        final_rows = len(df)
        duplicates_removed = initial_rows - final_rows
        
        if duplicates_removed > 0:
            print(f"  Removed {duplicates_removed} duplicate firm-date combinations (kept first occurrence)")
        
        # Select columns to keep: merge keys + main columns + lag columns
        cols_to_keep = merge_keys + main_cols + lag_cols
        
        # Only keep columns that actually exist in the dataframe
        available_cols = [col for col in cols_to_keep if col in df.columns]
        
        print(f"Loading {file_path}")
        print(f"  Requested columns: {len(cols_to_keep)}")
        print(f"  Available columns: {len(available_cols)}")
        print(f"  Missing columns: {set(cols_to_keep) - set(available_cols)}")
        print(f"  Final rows after deduplication: {final_rows}")
        
        return df[available_cols]
    
    except FileNotFoundError:
        print(f"Warning: File not found - {file_path}")
        return None
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None

# Define merge keys (common columns for joining)
merge_keys = ['ISSUER_TICKER', 'date']  # Using 'date' instead of 'month' as it's more standard

# Load the first dataframe (general_climate) as the base
base_config = file_configs['general_climate']
merged_df = load_and_select_columns(
    base_config['path'], 
    base_config['main_cols'], 
    base_config['lag_cols'],
    merge_keys + ['stock_index', 'region', 'ticker', 'company_name', 'year', 'quarter', 'month']  # Keep fundamentals from first df
)

if merged_df is None:
    print("Error: Could not load base dataframe (general_climate)")
    exit()

print(f"Base dataframe shape: {merged_df.shape}")

# Merge with other dataframes
for category, config in file_configs.items():
    if category == 'general_climate':  # Skip base dataframe
        continue
    
    print(f"\nMerging {category}...")
    
    # Load the dataframe (duplicates already removed in load_and_select_columns)
    df_to_merge = load_and_select_columns(
        config['path'],
        config['main_cols'],
        config['lag_cols'], 
        merge_keys
    )
    
    if df_to_merge is None:
        print(f"Skipping {category} due to loading error")
        continue
    
    # Perform the merge
    before_shape = merged_df.shape
    merged_df = pd.merge(
        merged_df, 
        df_to_merge, 
        on=merge_keys, 
        how='outer',  # Use outer join to keep all observations
        suffixes=('', f'_{category}')
    )
    after_shape = merged_df.shape
    
    print(f"  Before merge: {before_shape}")
    print(f"  After merge: {after_shape}")
    print(f"  Added columns: {after_shape[1] - before_shape[1]}")
    
    # Check if merge introduced any new duplicates (shouldn't happen with clean data)
    post_merge_duplicates = merged_df.duplicated(subset=['ISSUER_TICKER', 'date']).sum()
    if post_merge_duplicates > 0:
        print(f"  WARNING: {post_merge_duplicates} duplicates found after merge - this shouldn't happen!")

# Display final dataframe info
print(f"\n=== FINAL MERGED DATAFRAME ===")
print(f"Shape: {merged_df.shape}")
print(f"Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
print(f"Number of unique tickers: {merged_df['ISSUER_TICKER'].nunique()}")

# Final check for duplicates in merged dataframe
final_duplicates = merged_df.duplicated(subset=['ISSUER_TICKER', 'date']).sum()
print(f"Final duplicate firm-date combinations: {final_duplicates}")

# Show column names for verification
print(f"\nColumn names ({len(merged_df.columns)} total):")
for i, col in enumerate(merged_df.columns):
    print(f"  {i+1:2d}. {col}")

# Check for missing values in key columns
print(f"\nMissing values in key attention measures:")
attention_cols = [col for col in merged_df.columns if any(x in col for x in ['_ratio', '_attention'])]
for col in sorted(attention_cols):
    missing_pct = (merged_df[col].isna().sum() / len(merged_df)) * 100
    print(f"  {col}: {missing_pct:.1f}% missing")

# Add MONTH column - convert date to end of month
merged_df['MONTH'] = pd.to_datetime(merged_df['date']).dt.to_period('M').dt.to_timestamp('M') + pd.offsets.MonthEnd(0)

# Reorder columns to put MONTH right after ISSUER_TICKER
cols = merged_df.columns.tolist()
# Remove MONTH from its current position
cols.remove('MONTH')
# Find position of ISSUER_TICKER and insert MONTH after it
ticker_idx = cols.index('ISSUER_TICKER')
cols.insert(ticker_idx + 1, 'MONTH')
# Reorder the dataframe
merged_df = merged_df[cols]

# Sort by ticker and date for better organization
merged_df = merged_df.sort_values(['ISSUER_TICKER', 'date']).reset_index(drop=True)

tickers_to_skip = ['-E-US',
 '000DZ8-E-US',
 '000HWL -E',
 '0017ZH-E',
 '0017ZH-E-US',
 '001SGG-E-US',
 '0028NS -E',
 '05DZG8 -E',
 '05VHQ2 -E',
 '0801VG-E-US',
 '08XM5M -E',
 '0CV4BY-E',
 '0GPW6G -E',
 '0JDN79 -E',
 '0M5QM0 -E']

merged_df = merged_df.loc[~merged_df.ISSUER_TICKER.isin(tickers_to_skip)]

sem_vars = ['climate_attention_ratio',
       'climate_attention_ratio_lag1', 'climate_attention_ratio_lag2',
       'climate_opportunity_ratio', 'climate_opportunity_ratio_lag1',
       'climate_opportunity_ratio_lag2', 'climate_flood_ratio', 'climate_flood_ratio_lag1',
       'climate_flood_ratio_lag2','climate_physical_risk_ratio',
       'acute_physical_risk_ratio', 'chronic_physical_risk_ratio',
       'climate_physical_risk_ratio_lag1', 'climate_physical_risk_ratio_lag2',
       'acute_physical_risk_ratio_lag1', 'acute_physical_risk_ratio_lag2',
       'chronic_physical_risk_ratio_lag1', 'chronic_physical_risk_ratio_lag2',
       'climate_transition_risk_ratio', 'policy_transition_risk_ratio',
       'technology_transition_risk_ratio', 'market_transition_risk_ratio',
       'climate_transition_risk_ratio_lag1',
       'climate_transition_risk_ratio_lag2',
       'policy_transition_risk_ratio_lag1',
       'policy_transition_risk_ratio_lag2',
       'technology_transition_risk_ratio_lag1',
       'technology_transition_risk_ratio_lag2',
       'market_transition_risk_ratio_lag1',
       'market_transition_risk_ratio_lag2',
       'climate_transparency_disclosure_ratio', 'esg_reporting_ratio',
       'emissions_disclosure_ratio', 'governance_strategy_ratio',
       'regulatory_compliance_ratio',
       'climate_transparency_disclosure_ratio_lag1',
       'climate_transparency_disclosure_ratio_lag2',
       'esg_reporting_ratio_lag1', 'esg_reporting_ratio_lag2',
       'emissions_disclosure_ratio_lag1', 'emissions_disclosure_ratio_lag2',
       'governance_strategy_ratio_lag1', 'governance_strategy_ratio_lag2',
       'regulatory_compliance_ratio_lag1', 'regulatory_compliance_ratio_lag2']

def forward_fill_quarterly_data(df, max_fill_months=5):
    """
    Forward fill quarterly earnings call metrics to upcoming months.
    Each observation can be duplicated for up to max_fill_months (default=5),
    so there are 6 months total with the same value.
    More recent observations take precedence - forward-filled values are only
    kept if no more recent actual transcript exists for that month.
    
    Parameters:
    df: DataFrame with ISSUER_TICKER, MONTH, and semantic attention variables
    max_fill_months: Maximum number of months to forward fill (default=5)
    
    Returns:
    DataFrame with forward-filled monthly observations
    """
    
    # Ensure MONTH is datetime
    df = df.copy()
    df['MONTH'] = pd.to_datetime(df['MONTH'])
    
    # Identify semantic attention columns (excluding identifier and date columns)
    exclude_cols = ['ISSUER_TICKER', 'MONTH', 'date', 'stock_index', 'region', 
                   'ticker', 'company_name', 'year', 'quarter', 'month']
    
    semantic_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"Forward filling {len(semantic_cols)} semantic attention variables")
    print(f"Original data shape: {df.shape}")
    print(f"Forward filling for {max_fill_months} months (so {max_fill_months + 1} months total per transcript)")
    
    # Create a list to store all filled observations
    filled_dfs = []
    
    # Process each firm separately
    for ticker in df['ISSUER_TICKER'].unique():
        firm_data = df[df['ISSUER_TICKER'] == ticker].copy()
        firm_data = firm_data.sort_values('MONTH')
        
        # Get all actual transcript months for this firm to check against
        actual_months = set(firm_data['MONTH'])
        
        # Create forward-filled observations for this firm
        firm_filled = []
        
        for idx, row in firm_data.iterrows():
            base_month = row['MONTH']
            
            # Add the original observation
            firm_filled.append(row.copy())
            
            # Add forward-filled observations for next max_fill_months months
            for month_offset in range(1, max_fill_months + 1):
                fill_month = base_month + pd.DateOffset(months=month_offset)
                fill_month = fill_month + pd.offsets.MonthEnd(0)
                
                # Only create forward-filled observation if no actual transcript exists for this month
                if fill_month not in actual_months:
                    # Create forward-filled observation
                    filled_row = row.copy()
                    filled_row['MONTH'] = fill_month
                    
                    # Update month-related fields
                    filled_row['month'] = fill_month.strftime('%Y-%m')
                    filled_row['year'] = fill_month.year
                    
                    # Mark as forward-filled (optional - for tracking)
                    filled_row['forward_filled'] = True
                    filled_row['original_month'] = base_month
                    
                    firm_filled.append(filled_row)
                # If actual transcript exists for this month, we don't forward fill
                # (the actual transcript will take precedence)
        
        # Convert to DataFrame and add to list
        if firm_filled:
            firm_df = pd.DataFrame(firm_filled)
            filled_dfs.append(firm_df)
    
    # Combine all firms
    if filled_dfs:
        result_df = pd.concat(filled_dfs, ignore_index=True)
    else:
        result_df = df.copy()
    
    # Remove duplicates (keep the most recent observation for each ticker-month)
    # This ensures actual observations take precedence over forward-filled ones
    result_df['is_forward_filled'] = result_df.get('forward_filled', False)
    
    # Sort by ticker, month, and whether it's forward-filled (actual data first)
    result_df = result_df.sort_values(['ISSUER_TICKER', 'MONTH', 'is_forward_filled'])
    
    # Keep only the first observation for each ticker-month combination
    result_df = result_df.drop_duplicates(subset=['ISSUER_TICKER', 'MONTH'], keep='first')
    
    # Clean up and sort
    result_df = result_df.sort_values(['ISSUER_TICKER', 'MONTH']).reset_index(drop=True)
    
    print(f"After forward filling: {result_df.shape}")
    print(f"Added {result_df.shape[0] - df.shape[0]} forward-filled observations")
    
    # Show summary by ticker
    fill_summary = result_df.groupby('ISSUER_TICKER').agg({
        'MONTH': 'count',
        'is_forward_filled': 'sum'
    }).rename(columns={'MONTH': 'total_obs', 'is_forward_filled': 'forward_filled_obs'})
    
    print(f"\nSample of forward-fill summary:")
    print(fill_summary.head(10))
    
    return result_df

# Apply the forward fill to your merged dataframe
# Assuming your merged dataframe is called 'merged_df'
merged_df = forward_fill_quarterly_data(merged_df, max_fill_months=5)

# Optional: Remove the tracking columns if you don't need them
if 'forward_filled' in merged_df.columns:
    merged_df = merged_df.drop(['forward_filled', 'original_month', 'is_forward_filled'], axis=1)

# Show example of the forward filling effect
print(f"\nExample of forward filling (first firm):")
example_ticker = merged_df['ISSUER_TICKER'].iloc[0]
example_data = merged_df[merged_df['ISSUER_TICKER'] == example_ticker][
    ['ISSUER_TICKER', 'MONTH', 'climate_attention_ratio', 'quarter']
].head(10)

# Save the merged dataframe
output_path = f"{base_path}/merged_semantic_attention_panel.csv"
merged_df.to_csv(output_path, index=False)
print(f"\nMerged dataframe saved to: {output_path}")

# Create a summary of the merge
print(f"\n=== MERGE SUMMARY ===")
print(f"Successfully merged {len(file_configs)} semantic attention datasets")
print(f"Final dataset contains:")
print(f"  - {merged_df.shape[0]:,} observations")
print(f"  - {merged_df.shape[1]} variables") 
print(f"  - {merged_df['ISSUER_TICKER'].nunique()} unique firms")
print(f"  - Time period: {merged_df['year'].min()}-{merged_df['year'].max()}")

# Display sample of merged data
print(f"\nSample of merged data (first 5 rows, key columns):")
sample_cols = ['ISSUER_TICKER', 'date', 'climate_attention_ratio', 'climate_opportunity_ratio', 
               'climate_physical_risk_ratio', 'climate_transition_risk_ratio', 'climate_transparency_disclosure_ratio']
available_sample_cols = [col for col in sample_cols if col in merged_df.columns]
print(merged_df[available_sample_cols].head())