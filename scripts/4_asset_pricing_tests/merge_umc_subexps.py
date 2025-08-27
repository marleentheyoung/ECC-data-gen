import pandas as pd
from datetime import datetime, timedelta

# Read the dataframes
hedge_df = pd.read_csv('/Users/marleendejonge/Desktop/ECC-data-generation/data/asset_pricing/hedge_df_engle_full_1905.csv')
umc_df = pd.read_csv('/Users/marleendejonge/Desktop/ECC-data-generation/data/asset_pricing/UMC_monthly.csv')

# Convert date columns to datetime
hedge_df['MONTH'] = pd.to_datetime(hedge_df['MONTH'])
umc_df['UMC.Date'] = pd.to_datetime(umc_df['UMC.Date'])

# Create a mapping date for UMC data (shift by one month forward to match with previous year's December)
umc_df['merge_date'] = umc_df['UMC.Date'] - pd.DateOffset(days=1)

# Merge the dataframes
merged_df = hedge_df.merge(
    umc_df[['merge_date', "UMC.Water.Drought"]], 
    left_on='MONTH', 
    right_on='merge_date', 
    how='left'
)

# Drop the helper column
merged_df = merged_df.drop('merge_date', axis=1)

# Save the result
merged_df.to_csv('/Users/marleendejonge/Desktop/ECC-data-generation/data/asset_pricing/hedge_df_engle_full_1905_extended.csv', index=False)

print("Merge completed successfully!")
print(f"Original hedge_df shape: {hedge_df.shape}")
print(f"Merged dataframe shape: {merged_df.shape}")
print(f"Number of non-null UMC.Water.Drought values: {merged_df['UMC.Water.Drought'].notna().sum()}")