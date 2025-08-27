import json
import os
import glob
import re
import pandas as pd

def load_ticker_mapping(mapping_path):
    """Load the ticker mapping dictionary"""
    with open(mapping_path, 'r') as f:
        ticker_mapping = json.load(f)
    
    # Create reverse mapping: alternative ticker -> main ticker
    reverse_mapping = {}
    for main_ticker, alternatives in ticker_mapping.items():
        for alt_ticker in alternatives:
            reverse_mapping[alt_ticker] = main_ticker
    
    return reverse_mapping

def clean_ticker(ticker):
    """Clean ticker by removing country extension and replacing dashes with spaces"""
    if not ticker:
        return ""
    
    import re
    # Convert to string and strip whitespace
    ticker = str(ticker).strip()
    
    # Remove country extension (e.g., "ASML-NL" -> "ASML")
    # Look for pattern: dash followed by 2-3 uppercase letters at the end
    ticker = re.sub(r'-[A-Z]{2,3}$', '', ticker)  
    
    # Replace remaining dashes with spaces
    ticker = ticker.replace('-', ' ')
    
    return ticker

def update_file_tickers(file_path, reverse_mapping):
    """Update tickers in a single JSON file"""
    try:
        # Load the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Track changes
        changes_made = 0
        
        # Update tickers in the list of dictionaries
        for entry in data:
            if 'ticker' in entry:
                original_ticker = entry['ticker']
                cleaned_ticker = clean_ticker(original_ticker)
                
                # Check if cleaned ticker is in the reverse mapping
                if cleaned_ticker in reverse_mapping:
                    new_ticker = reverse_mapping[cleaned_ticker]
                    entry['ticker'] = new_ticker
                    changes_made += 1
                    print(f"  Changed: {original_ticker} (cleaned: {cleaned_ticker}) -> {new_ticker}")
        
        # Save the updated file if changes were made
        if changes_made > 0:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"  Saved {changes_made} changes to {file_path}")
        
        return changes_made
    
    except Exception as e:
        print(f"  Error processing {file_path}: {e}")
        return 0

def main():
    # Load ticker mapping
    mapping_path = "/Users/marleendejonge/Desktop/ECC-data-generation/outputs/data_summary/ticker_mapping.json"
    
    try:
        reverse_mapping = load_ticker_mapping(mapping_path)
        print(f"Loaded ticker mapping with {len(reverse_mapping)} alternative tickers")
        print("Alternative -> Main ticker mappings:")
        for alt, main in list(reverse_mapping.items())[:10]:  # Show first 10
            print(f"  {alt} -> {main}")
        if len(reverse_mapping) > 10:
            print(f"  ... and {len(reverse_mapping) - 10} more")
        print()
        
    except FileNotFoundError:
        print(f"Ticker mapping file not found: {mapping_path}")
        return
    except Exception as e:
        print(f"Error loading ticker mapping: {e}")
        return
    
    # Find all structured JSON files
    base_path = "/Users/marleendejonge/Desktop/ECC-data-generation/data/climate_paragraphs/STOXX600/"
    pattern = os.path.join(base_path, "climate_segments_*.json")
    json_files = glob.glob(pattern)
    
    if not json_files:
        print(f"No files found matching pattern: {pattern}")
        return
    
    print(f"Found {len(json_files)} JSON files to process")
    print()
    
    # Process each file
    total_changes = 0
    files_changed = 0
    
    for i, file_path in enumerate(sorted(json_files), 1):
        filename = os.path.basename(file_path)
        print(f"Processing file {i}/{len(json_files)}: {filename}")
        
        changes = update_file_tickers(file_path, reverse_mapping)
        if changes > 0:
            files_changed += 1
            total_changes += changes
        else:
            print(f"  No changes needed for {filename}")
        print()
    
    # Summary
    print("="*50)
    print("SUMMARY:")
    print(f"- Files processed: {len(json_files)}")
    print(f"- Files changed: {files_changed}")
    print(f"- Total ticker changes: {total_changes}")
    print("="*50)

if __name__ == "__main__":
    main()
