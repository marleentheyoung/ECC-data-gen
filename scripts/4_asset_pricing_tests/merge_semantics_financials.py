import pandas as pd
import numpy as np
import os

def setup_working_directory():
    """Set up the working directory for the project."""
    working_dir = "/Users/marleendejonge/Library/CloudStorage/OneDrive-UvA/PhD/PhD planning/Papers/PaperI/Equity pricing/"
    os.chdir(working_dir)
    return working_dir

def load_data(working_dir):
    """Load all required datasets."""
    # Load panel data
    df_EU = pd.read_csv("Cleaned panels/complete_panel_EU.csv")
    df_US = pd.read_csv("Cleaned panels/complete_panel_US.csv")
    
    # Load sustainability data
    eu_sustainability_path = f"{working_dir}Data_Rianne/Sustainability/EU/SXXP november 2024 - sustainability data2.xlsx"
    us_sustainability_path = f"{working_dir}Data_Rianne/Sustainability/US/SPX november 2024 - sustainability data2.xlsx"
    
    EU_data = pd.read_excel(eu_sustainability_path)
    US_data = pd.read_excel(us_sustainability_path)
    
    # Load semantic data
    sem_df = pd.read_csv("/Users/marleendejonge/Desktop/ECC-data-generation/outputs/firm_level_panel/merged_semantic_attention_panel.csv")
    
    return df_EU, df_US, EU_data, US_data, sem_df

def create_ticker_mapping():
    """Define the ticker mapping dictionary."""
    return {
        'GMAB': 'GMAB-DK', 'GSK': 'GSK-GB', 'HLN': 'HLN-GB', 'IGH': 'IGH-GB',
        'QGEN': 'QGEN-NL', 'RACE': 'RACE-NL', 'REP': 'REP-ES', 'RIO': 'RIO-GB',
        'SAN': 'SAN-ES', 'SAP': 'SAP-DE', 'SHEL': 'SHEL-GB', 'SUBC': 'SUBC-LU',
        'TEF': 'TEF-ES', 'TTE': 'TTE-FR', 'VOD': 'VOD-GB', 'WPP': 'WPP-JE',
        'ZEAL': 'ZEAL-DK', 'CCH': 'CCH-CH', 'FRO': 'FRO-CY', 'EQNR': 'EQNR-NO',
        'BBVA': 'BBVA-ES', 'AZN': 'AZN-GB', 'ARGX': 'ARGX-NL', 'ASML': 'ASML-NL',
        'ALC': 'ALC-CH', 'SCTBY-SE': 'SECU-B-SE', 'IAG-ES': 'IAG-GB',
        'ELUX-B-SE': 'ELUXY-SE', 'VOLV-B-SE': 'SVNLY-SE', 'EDPR-ES': 'EDPR-PT',
        'HSX-BM': 'HSX-GB', 'GFTU-IE': 'GFTU-GB', 'BT.A-GB': 'BT-A-GB',
        'UU.-GB': 'UU-GB', 'AV.-GB': 'AV-GB', 'TUI1-DE': 'TUI-GB',
        'EKTA-B-SE': 'EKTAY-SE', 'MTRS-SE': 'LOIMY-SE', 'VOLCAR-B-SE': 'SVCBY-SE'
    }

def process_panel_data_robust(df_EU, df_US, EU_data, US_data):
    """Process panel data with robust ticker matching."""
    # Rename ISIN columns
    df_EU = df_EU.rename(columns={"ISIN": "ISSUER_ISIN"})
    df_US = df_US.rename(columns={"ISIN": "ISSUER_ISIN"})
    
    # For EU data - create base ticker mapping
    eu_ticker_map = EU_data[['ISSUER_ISIN', 'ISSUER_TICKER']].drop_duplicates(subset=['ISSUER_ISIN'])
    
    # Clean the base ticker BEFORE using it
    eu_ticker_map['BASE_TICKER'] = eu_ticker_map['ISSUER_TICKER'].str.replace(' ', '-').str.replace('.', '-')
    
    # Add base ticker to df_EU
    df_EU = df_EU.merge(eu_ticker_map[['ISSUER_ISIN', 'BASE_TICKER']], on='ISSUER_ISIN', how='left')
    
    # Add country code from ISIN to create full ticker
    df_EU['ISSUER_TICKER'] = df_EU['BASE_TICKER'] + '-' + df_EU['ISSUER_ISIN'].str[:2]
    
    # Store base ticker for matching (clean and remove trailing dashes)
    df_EU['BASE_TICKER_FOR_MATCHING'] = df_EU['BASE_TICKER'].str.rstrip('-')
    
    # Process US data normally
    df_US = df_US.merge(
        US_data[['ISSUER_ISIN', 'ISSUER_TICKER']].drop_duplicates(subset=['ISSUER_ISIN']), 
        on='ISSUER_ISIN', 
        how='left'
    )
    
    # Clean US ticker and create base ticker
    df_US['ISSUER_TICKER'] = (df_US['ISSUER_TICKER'].str.replace(' ', '-').str.replace('.', '-') + '-US')
    df_US['BASE_TICKER_FOR_MATCHING'] = df_US['ISSUER_TICKER'].str.rsplit('-', n=1).str[0].str.rstrip('-')
    
    return df_EU, df_US

def prepare_semantic_data(sem_df, ticker_mapping):
    """Prepare semantic data with ticker mappings."""
    # Apply ticker mapping
    sem_df['ISSUER_TICKER'] = sem_df['ISSUER_TICKER'].map(ticker_mapping).fillna(sem_df['ISSUER_TICKER'])
    
    # Convert date column
    sem_df['MONTH'] = pd.to_datetime(sem_df['MONTH'])
    
    return sem_df

def merge_datasets_robust(df_EU, df_US, sem_df):
    """Merge datasets using base ticker matching with regional constraints."""
    # Convert date columns first
    df_EU['MONTH'] = pd.to_datetime(df_EU['MONTH'])
    df_US['MONTH'] = pd.to_datetime(df_US['MONTH'])
    sem_df['MONTH'] = pd.to_datetime(sem_df['MONTH'])
    
    # Clean ticker columns FIRST (before creating base tickers)
    df_EU['ISSUER_TICKER'] = df_EU['ISSUER_TICKER'].str.replace(' ', '-').str.replace('.', '-')
    df_US['ISSUER_TICKER'] = df_US['ISSUER_TICKER'].str.replace(' ', '-').str.replace('.', '-')
    
    # Clean semantic ticker columns and create base tickers
    sem_df['ISSUER_TICKER'] = sem_df['ISSUER_TICKER'].str.replace(' ', '-').str.replace('.', '-')
    
    # Remove only the last dash part (country code)
    sem_df['BASE_TICKER_FOR_MATCHING'] = sem_df['ISSUER_TICKER'].str.rsplit('-', n=1).str[0]
    
    # Remove trailing dashes if they exist (e.g., "BT-" becomes "BT")
    sem_df['BASE_TICKER_FOR_MATCHING'] = sem_df['BASE_TICKER_FOR_MATCHING'].str.rstrip('-')
       
    # Split semantic data by region to prevent cross-regional matching
    sem_df_us = sem_df[sem_df['region'] == 'US'].copy()
    sem_df_eu = sem_df[sem_df['region'] == 'EU'].copy()
    
    # Merge US panel data with US semantic data only
    df_us_with_semantics = df_US.merge(
        sem_df_us,
        on=['BASE_TICKER_FOR_MATCHING', 'MONTH'],
        how='left',
        suffixes=('', '_sem')
    )
    
    # Merge EU panel data with EU semantic data only  
    df_eu_with_semantics = df_EU.merge(
        sem_df_eu,
        on=['BASE_TICKER_FOR_MATCHING', 'MONTH'],
        how='left',
        suffixes=('', '_sem')
    )
    
    # Combine the regionally-matched datasets
    df_full = pd.concat([df_EU, df_US], ignore_index=True)
    df_full_with_semantics = pd.concat([df_us_with_semantics, df_eu_with_semantics], ignore_index=True)
    
    return df_full, df_full_with_semantics

def main():
    """Main execution function."""
    print("Loading and merging datasets...")
    
    # Setup
    working_dir = setup_working_directory()
    
    # Load data
    df_EU, df_US, EU_data, US_data, sem_df = load_data(working_dir)
    
    # Create ticker mapping
    ticker_mapping = create_ticker_mapping()
    
    # Process panel data with robust matching
    df_EU, df_US = process_panel_data_robust(df_EU, df_US, EU_data, US_data)
    
    # Prepare semantic data
    sem_df = prepare_semantic_data(sem_df, ticker_mapping)
    
    # Merge datasets with robust matching
    df_full, df_full_with_semantics = merge_datasets_robust(df_EU, df_US, sem_df)
    
    # Calculate coverage for both datasets
    print(f"Merge completed successfully!")
    
    # Panel data info
    print(f"\nPanel data (df_full): {df_full.shape[0]:,} observations, {df_full.shape[1]} variables")
    print(f"Panel unique firms: {df_full['ISSUER_TICKER'].nunique():,}")
    
    # Merged data info and coverage
    semantic_cols = [col for col in df_full_with_semantics.columns if 'climate_attention_ratio' == col]
    if semantic_cols:
        sample_col = semantic_cols[0]
        missing_count = df_full_with_semantics[sample_col].isna().sum()
        coverage_rate = (1 - missing_count / len(df_full_with_semantics)) * 100
        
        print(f"\nMerged data (df_full_with_semantics): {df_full_with_semantics.shape[0]:,} observations, {df_full_with_semantics.shape[1]} variables")
        print(f"Semantic data coverage: {coverage_rate:.1f}%")
        print(f"Missing semantic observations: {missing_count:,}")
        print(f"Merged unique firms: {df_full_with_semantics['ISSUER_TICKER'].nunique():,}")
    
    # Save the datasets
    df_full_with_semantics.to_csv("/Users/marleendejonge/Desktop/ECC-data-generation/data/asset_pricing/df_fin_sus_sem2608.csv", index=False)
    
    return df_full, df_full_with_semantics

# Execute the pipeline
if __name__ == "__main__":
    df_full, df_final = main()