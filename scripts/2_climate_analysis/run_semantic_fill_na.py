#!/usr/bin/env python3
"""
Fill NaN values in semantic columns with firm-specific means.

This script:
1. Loads the financial sustainability dataframe
2. Identifies primary semantic columns (excludes lag columns)
3. For each firm, calculates the mean of non-NaN values for each semantic column
4. Fills NaN values with firm-specific means
5. If all values for a firm are NaN (no mean available), fills with 0
6. Saves the updated dataframe

Author: Your Name
Date: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def identify_semantic_columns(df):
    """
    Identify primary semantic columns (excludes lag columns).
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        List of primary semantic column names
    """
    # Define the specific primary semantic columns (no lag columns)
    primary_semantic_cols = [
        'climate_attention_ratio',
        'climate_opportunity_ratio', 
        'climate_physical_risk_ratio',
        'acute_physical_risk_ratio',
        'chronic_physical_risk_ratio',
        'climate_transition_risk_ratio',
        'policy_transition_risk_ratio',
        'technology_transition_risk_ratio',
        'market_transition_risk_ratio',
        'climate_transparency_disclosure_ratio',
        'esg_reporting_ratio',
        'emissions_disclosure_ratio',
        'governance_strategy_ratio',
        'regulatory_compliance_ratio'
    ]
    
    # Filter to only include columns that exist in the dataframe
    existing_cols = [col for col in primary_semantic_cols if col in df.columns]
    
    return existing_cols

def fill_semantic_nans(csv_path):
    """
    Fill NaN values in semantic columns with firm-specific means.
    
    Args:
        csv_path: Path to the CSV file
    """
    logger.info(f"Loading dataframe from: {csv_path}")
    
    # Load the dataframe
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded dataframe with shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        return
    
    # Display basic info about the dataframe
    logger.info(f"Columns: {list(df.columns)}")
    
    # Identify semantic columns
    semantic_cols = identify_semantic_columns(df)
    
    if not semantic_cols:
        logger.warning("No semantic columns found! Looking for columns containing 'semantic'")
        # Show all columns for debugging
        logger.info("All columns:")
        for i, col in enumerate(df.columns):
            logger.info(f"  {i}: {col}")
        return
    
    logger.info(f"Found {len(semantic_cols)} primary semantic columns:")
    for col in semantic_cols:
        nan_count = df[col].isna().sum()
        total_count = len(df)
        logger.info(f"  {col}: {nan_count}/{total_count} ({nan_count/total_count*100:.1f}%) NaN values")
    
    # Identify firm identifier column (common names)
    firm_id_candidates = ['ticker', 'firm_id', 'company_id', 'ISSUER_TICKER', 'company_name', 'firm']
    firm_id_col = None
    
    for candidate in firm_id_candidates:
        if candidate in df.columns:
            firm_id_col = candidate
            break
    
    if not firm_id_col:
        logger.error("Could not identify firm identifier column!")
        logger.info("Available columns that might be firm identifiers:")
        possible_firm_cols = [col for col in df.columns if any(x in col.lower() for x in ['ticker', 'firm', 'company', 'id'])]
        for col in possible_firm_cols:
            logger.info(f"  {col}")
        return
    
    logger.info(f"Using '{firm_id_col}' as firm identifier")
    logger.info(f"Number of unique firms: {df[firm_id_col].nunique()}")
    
    # Process each semantic column
    total_filled = 0
    
    for col in semantic_cols:
        logger.info(f"Processing column: {col}")
        
        # Count NaNs before processing
        nans_before = df[col].isna().sum()
        
        if nans_before == 0:
            logger.info(f"  No NaN values in {col}, skipping")
            continue
        
        # Calculate firm-specific means
        firm_means = df.groupby(firm_id_col)[col].mean()
        
        # Create a mapping function
        def fill_nan_with_firm_mean(row):
            if pd.isna(row[col]):
                firm_mean = firm_means.get(row[firm_id_col], 0)  # Default to 0 if no mean available
                return firm_mean if not pd.isna(firm_mean) else 0
            else:
                return row[col]
        
        # Apply the filling
        df[col] = df.apply(fill_nan_with_firm_mean, axis=1)
        
        # Count NaNs after processing
        nans_after = df[col].isna().sum()
        filled_count = nans_before - nans_after
        total_filled += filled_count
        
        logger.info(f"  Filled {filled_count} NaN values in {col}")
        logger.info(f"  NaN values: {nans_before} -> {nans_after}")
        
        # Show some statistics
        firms_with_all_nan = (df.groupby(firm_id_col)[col].apply(lambda x: x.isna().all())).sum()
        if firms_with_all_nan > 0:
            logger.info(f"  {firms_with_all_nan} firms had all NaN values (filled with 0)")
    
    logger.info(f"Total NaN values filled across all semantic columns: {total_filled}")
    
    # Save the updated dataframe
    output_path = Path(csv_path).parent / f"{Path(csv_path).stem}_filled.csv"
    
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"Saved updated dataframe to: {output_path}")
    except Exception as e:
        logger.error(f"Error saving updated dataframe: {e}")
    
    # Show final statistics
    logger.info("Final statistics for semantic columns:")
    for col in semantic_cols:
        nan_count = df[col].isna().sum()
        mean_val = df[col].mean()
        logger.info(f"  {col}: {nan_count} NaN values remaining, mean = {mean_val:.4f}")

def main():
    """Main function."""
    
    # File path
    csv_path = "/Users/marleendejonge/Desktop/ECC-data-generation/data/asset_pricing/df_fin_sus_sem2608.csv"
    
    logger.info("Starting semantic NaN filling process...")
    
    # Check if file exists
    if not Path(csv_path).exists():
        logger.error(f"File not found: {csv_path}")
        return
    
    # Fill NaN values
    fill_semantic_nans(csv_path)
    
    logger.info("Process completed!")

if __name__ == "__main__":
    main()