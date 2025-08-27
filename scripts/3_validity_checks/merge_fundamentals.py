#!/usr/bin/env python3
"""
Merge firm fundamentals data with climate attention measures for validity checks.

Usage:
    python merge_fundamentals.py <input_csv_path>

Example:
    python merge_fundamentals.py outputs/firm_level_panel/firm_level_climate_attention.csv
"""

import pandas as pd
import sys
import argparse
from pathlib import Path

def load_fundamentals_data(fundamentals_path: str = "data/firm_fundamentals/HEDGE_UNIVERSE.xlsx"):
    """
    Load and clean the firm fundamentals data from Excel file.
    
    Args:
        fundamentals_path: Path to the HEDGE_UNIVERSE.xlsx file
        
    Returns:
        DataFrame with firm fundamentals
    """
    print(f"📥 Loading fundamentals data from {fundamentals_path}")
    
    try:
        # Load Excel file
        fundamentals_df = pd.read_excel(fundamentals_path, skiprows=4)
        
        print(f"✅ Loaded {len(fundamentals_df)} rows from fundamentals file")
        print(f"Columns: {list(fundamentals_df.columns)}")
        
        # Select and rename required columns
        required_cols = {
            'FE Ticker': 'ISSUER_TICKER',
            'RBICS Economy': 'sector',
            'Entity Country HQ': 'country_hq', 
            'Entity Country Incorp': 'country_incorp',
            'Entity Country Risk': 'country_risk',
            'Exchange Country Name': 'exchange_country'
        }
        
        # Check if required columns exist
        missing_cols = []
        for col in required_cols.keys():
            if col not in fundamentals_df.columns:
                missing_cols.append(col)
        
        if missing_cols:
            print(f"⚠️  Missing columns in fundamentals file: {missing_cols}")
            print(f"Available columns: {list(fundamentals_df.columns)}")
            return None
        
        # Select and rename columns
        fundamentals_clean = fundamentals_df[list(required_cols.keys())].copy()
        fundamentals_clean = fundamentals_clean.rename(columns=required_cols)
        
        # Clean ticker format (remove any extra spaces, standardize)
        fundamentals_clean['ISSUER_TICKER'] = fundamentals_clean['ISSUER_TICKER'].astype(str).str.strip()
        
        # Remove duplicates based on ticker
        initial_count = len(fundamentals_clean)
        fundamentals_clean = fundamentals_clean.drop_duplicates(subset=['ISSUER_TICKER'])
        final_count = len(fundamentals_clean)
        
        if initial_count != final_count:
            print(f"⚠️  Removed {initial_count - final_count} duplicate tickers")
        
        print(f"📊 Fundamentals data summary:")
        print(f"  Unique firms: {fundamentals_clean['ISSUER_TICKER'].nunique()}")
        print(f"  Unique sectors: {fundamentals_clean['sector'].nunique()}")
        print(f"  Unique countries (HQ): {fundamentals_clean['country_hq'].nunique()}")
        
        return fundamentals_clean
        
    except FileNotFoundError:
        print(f"❌ File not found: {fundamentals_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading fundamentals data: {e}")
        return None

def merge_with_fundamentals(climate_df, fundamentals_df):
    """
    Merge climate attention data with firm fundamentals.
    
    Args:
        climate_df: DataFrame with climate attention measures
        fundamentals_df: DataFrame with firm fundamentals
        
    Returns:
        Merged DataFrame
    """
    print(f"🔗 Merging datasets...")
    
    # Check ticker formats
    print(f"Climate data tickers sample: {climate_df['ISSUER_TICKER'].head().tolist()}")
    print(f"Fundamentals tickers sample: {fundamentals_df['ISSUER_TICKER'].head().tolist()}")
    
    # Merge datasets
    merged_df = climate_df.merge(
        fundamentals_df, 
        on='ISSUER_TICKER', 
        how='left'
    )
    
    # Check merge success
    total_records = len(climate_df)
    matched_records = merged_df['sector'].notna().sum()
    match_rate = matched_records / total_records
    
    print(f"📈 Merge results:")
    print(f"  Total climate records: {total_records:,}")
    print(f"  Records with sector data: {matched_records:,}")
    print(f"  Match rate: {match_rate:.1%}")
    
    if match_rate < 0.8:
        print(f"⚠️  Low match rate! Checking ticker mismatches...")
        
        # Find unmatched tickers
        climate_tickers = set(climate_df['ISSUER_TICKER'].unique())
        fundamentals_tickers = set(fundamentals_df['ISSUER_TICKER'].unique())
        
        unmatched = climate_tickers - fundamentals_tickers
        print(f"Unmatched tickers (first 10): {list(unmatched)[:10]}")
    
    # Summary by sector
    if 'sector' in merged_df.columns:
        sector_summary = merged_df.groupby('sector').agg({
            'ISSUER_TICKER': 'nunique',
            'climate_attention_ratio': ['count', 'mean']
        }).round(4)
        
        print(f"\n📊 Summary by sector:")
        print(sector_summary.head(10))
    
    return merged_df

def main():
    """
    Main function to handle command line arguments and execute merge.
    """
    parser = argparse.ArgumentParser(
        description='Merge climate attention data with firm fundamentals for validity checks'
    )
    parser.add_argument(
        'input_csv', 
        help='Path to CSV file with climate attention data'
    )
    parser.add_argument(
        '--fundamentals', 
        default='data/firm_fundamentals/HEDGE_UNIVERSE.xlsx',
        help='Path to fundamentals Excel file (default: data/firm_fundamentals/HEDGE_UNIVERSE.xlsx)'
    )
    parser.add_argument(
        '--output', 
        help='Output path for merged CSV (optional)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input_csv).exists():
        print(f"❌ Input file not found: {args.input_csv}")
        sys.exit(1)
    
    # Load climate data
    print(f"📥 Loading climate data from {args.input_csv}")
    try:
        climate_df = pd.read_csv(args.input_csv)
        print(f"✅ Loaded {len(climate_df)} records")
        
        # Check required columns
        if 'ISSUER_TICKER' not in climate_df.columns:
            print("❌ ISSUER_TICKER column not found in climate data")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error loading climate data: {e}")
        sys.exit(1)
    
    # Load fundamentals data
    fundamentals_df = load_fundamentals_data(args.fundamentals)
    if fundamentals_df is None:
        sys.exit(1)
    
    # Merge datasets
    merged_df = merge_with_fundamentals(climate_df, fundamentals_df)
    
    # Save output if specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(output_path, index=False)
        print(f"💾 Saved merged data to {output_path}")
    else:
        # Auto-generate output path
        input_path = Path(args.input_csv)
        output_path = input_path.parent / f"{input_path.stem}_with_fundamentals.csv"
        merged_df.to_csv(output_path, index=False)
        print(f"💾 Saved merged data to {output_path}")
    
    print(f"\n✅ Merge completed successfully!")
    print(f"Final dataset: {len(merged_df)} records with sector/country information")
    
    return merged_df

if __name__ == "__main__":
    merged_data = main()