#!/usr/bin/env python3
"""
Convenience runner for building semantic indexes.

Author: Marleen de Jonge
Date: 2025
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run semantic index building with common configurations."""
    
    print("🔍 ECC Semantic Index Builder")
    print("=" * 40)
    print("This script will build semantic indexes for your climate snippets.")
    print()
    
    # Check if enhanced climate snippets exist
    base_path = Path("data/enhanced_climate_snippets")
    if not base_path.exists():
        print("❌ Enhanced climate snippets not found!")
        print(f"Expected path: {base_path}")
        print()
        print("Please run the sentence ratio calculation first:")
        print("  python scripts/3_agg_variables/calculate_sentence_ratios.py --all")
        return False
    
    # Check which markets have data
    available_markets = []
    for market in ["SP500", "STOXX600"]:
        market_path = base_path / market
        if market_path.exists() and list(market_path.glob("enhanced_climate_segments_*.json")):
            available_markets.append(market)
    
    if not available_markets:
        print("❌ No enhanced climate snippet files found!")
        print("Please run the sentence ratio calculation first.")
        return False
    
    print(f"✅ Found data for markets: {', '.join(available_markets)}")
    print()
    
    # Ask user for configuration
    print("Configuration options:")
    print("1. Build all indexes (SP500 + STOXX600 + Combined) [Recommended]")
    print("2. Build SP500 only")
    print("3. Build STOXX600 only") 
    print("4. Build combined index only")
    print("5. Custom configuration")
    print()
    
    while True:
        try:
            choice = input("Select option (1-5): ").strip()
            if choice in ['1', '2', '3', '4', '5']:
                break
            print("Please enter 1, 2, 3, 4, or 5")
        except KeyboardInterrupt:
            print("\n❌ Cancelled by user")
            return False
    
    # Model selection
    print()
    print("Model options:")
    print("1. Fast model (all-MiniLM-L6-v2) [Recommended for testing]")
    print("2. High quality model (all-mpnet-base-v2) [Better results, slower]")
    print()
    
    while True:
        try:
            model_choice = input("Select model (1-2): ").strip()
            if model_choice in ['1', '2']:
                break
            print("Please enter 1 or 2")
        except KeyboardInterrupt:
            print("\n❌ Cancelled by user")
            return False
    
    # Set model
    if model_choice == '1':
        model = 'sentence-transformers/all-MiniLM-L6-v2'
        print("Using fast model (good for testing)")
    else:
        model = 'sentence-transformers/all-mpnet-base-v2'
        print("Using high quality model (better for final analysis)")
    
    print()
    print("Building semantic indexes...")
    print()
    
    # Build command
    script_path = Path(__file__).parent / "scripts" / "5_semantic_search" / "build_semantic_index.py"
    
    # Handle the case where script is in different location
    if not script_path.exists():
        script_path = Path("scripts/5_semantic_search/build_semantic_index.py")
    if not script_path.exists():
        # Try finding it relative to current file
        current_dir = Path(__file__).parent
        script_path = current_dir / "build_semantic_index.py"
    
    base_cmd = ["python", str(script_path), "--model", model]
    
    if choice == '1':
        cmd = base_cmd + ["--all"]
        print("🚀 Building all indexes (SP500 + STOXX600 + Combined)")
    elif choice == '2':
        cmd = base_cmd + ["--market", "SP500"]
        print("🚀 Building SP500 index")
    elif choice == '3':
        cmd = base_cmd + ["--market", "STOXX600"]
        print("🚀 Building STOXX600 index")
    elif choice == '4':
        cmd = base_cmd + ["--combined-only"]
        print("🚀 Building combined index only")
    else:  # choice == '5'
        print("Custom configuration - please run the script manually:")
        print(f"python {script_path} --help")
        return True
    
    try:
        # Run the build command
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        
        print("✅ Index building completed successfully!")
        print()
        print("Output summary:")
        # Print last few lines of output
        output_lines = result.stdout.strip().split('\n')
        for line in output_lines[-10:]:
            if line.strip():
                print(f"  {line}")
    except:
        print("error!!!")

if __name__ == "__main__":
    main()