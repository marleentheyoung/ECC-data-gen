#!/usr/bin/env python3
"""
Batch runner for STOXX600 semantic climate risk analysis.
Runs the semantic risk analysis for all enhanced_climate_segments files (1-12).

Author: Marleen de Jonge
Date: 2025
"""

import subprocess
import sys
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('stoxx600_batch_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

def run_semantic_analysis_batch():
    """
    Run semantic climate risk analysis for all STOXX600 files (1-12).
    """
    base_command = "python scripts/4_firm_variables/cc_semantic_risk_opt.py"
    index = "STOXX600"
    
    # Track results
    successful_runs = []
    failed_runs = []
    total_start_time = time.time()
    
    logger.info("="*60)
    logger.info("STARTING STOXX600 SEMANTIC CLIMATE RISK BATCH ANALYSIS")
    logger.info("="*60)
    logger.info(f"Processing files: enhanced_climate_segments_1.json to enhanced_climate_segments_12.json")
    logger.info(f"Index: {index}")
    logger.info(f"Command template: {base_command} --file enhanced_climate_segments_{{i}}.json --index {index}")
    logger.info("")
    
    for i in range(1, 13):  # Files 1 through 12
        file_name = f"enhanced_climate_segments_{i}.json"
        
        # Construct the full command
        command = [
            "python", 
            "scripts/4_firm_variables/cc_semantic_risk_opt.py",
            "--file", file_name,
            "--index", index
        ]
        
        logger.info(f"[{i}/12] Starting analysis for {file_name}")
        logger.info(f"Command: {' '.join(command)}")
        
        start_time = time.time()
        
        try:
            # Run the command
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout per file
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                logger.info(f"✅ SUCCESS: {file_name} completed in {duration:.1f} seconds")
                successful_runs.append((i, file_name, duration))
                
                # Log some output for verification
                if result.stdout:
                    # Just log the last few lines to avoid clutter
                    output_lines = result.stdout.strip().split('\n')
                    if len(output_lines) > 3:
                        logger.info("Last few lines of output:")
                        for line in output_lines[-3:]:
                            logger.info(f"  {line}")
                    else:
                        logger.info(f"Output: {result.stdout.strip()}")
            else:
                logger.error(f"❌ FAILED: {file_name} (exit code: {result.returncode})")
                failed_runs.append((i, file_name, result.returncode))
                
                # Log error details
                if result.stderr:
                    logger.error(f"Error output: {result.stderr}")
                if result.stdout:
                    logger.error(f"Standard output: {result.stdout}")
        
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ TIMEOUT: {file_name} exceeded 1 hour limit")
            failed_runs.append((i, file_name, "TIMEOUT"))
            
        except Exception as e:
            logger.error(f"💥 EXCEPTION: {file_name} - {str(e)}")
            failed_runs.append((i, file_name, f"EXCEPTION: {str(e)}"))
        
        logger.info("")  # Add spacing between files
    
    # Final summary
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    logger.info("="*60)
    logger.info("BATCH ANALYSIS COMPLETE")
    logger.info("="*60)
    logger.info(f"Total time: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    logger.info(f"Successful: {len(successful_runs)}/12")
    logger.info(f"Failed: {len(failed_runs)}/12")
    
    if successful_runs:
        logger.info("\n✅ SUCCESSFUL RUNS:")
        for i, filename, duration in successful_runs:
            logger.info(f"  {i:2d}. {filename:<30} ({duration:.1f}s)")
    
    if failed_runs:
        logger.info("\n❌ FAILED RUNS:")
        for i, filename, error in failed_runs:
            logger.info(f"  {i:2d}. {filename:<30} (Error: {error})")
    
    # Check output files
    logger.info("\n📁 CHECKING OUTPUT FILES:")
    output_dir = Path("outputs/variables/cc_risk")
    if output_dir.exists():
        csv_files = list(output_dir.glob("cc_risk_STOXX600_enhanced_climate_segments_*.csv"))
        logger.info(f"Found {len(csv_files)} output CSV files:")
        for csv_file in sorted(csv_files):
            try:
                import pandas as pd
                df = pd.read_csv(csv_file)
                logger.info(f"  {csv_file.name:<40} ({len(df):,} rows)")
            except:
                logger.info(f"  {csv_file.name:<40} (could not read)")
    else:
        logger.warning(f"Output directory not found: {output_dir}")
    
    logger.info("="*60)
    
    return len(successful_runs) == 12  # Return True if all succeeded

def check_prerequisites():
    """Check if required files and directories exist."""
    logger.info("Checking prerequisites...")
    
    # Check if script exists
    script_path = Path("scripts/4_firm_variables/cc_semantic_risk_opt.py")
    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return False
    
    # Check if input directory exists
    input_dir = Path("data/enhanced_climate_snippets/STOXX600")
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return False
    
    # Check if at least some input files exist
    existing_files = []
    for i in range(1, 13):
        file_path = input_dir / f"enhanced_climate_segments_{i}.json"
        if file_path.exists():
            existing_files.append(i)
    
    if not existing_files:
        logger.error("No enhanced_climate_segments_*.json files found in input directory")
        return False
    
    logger.info(f"Found {len(existing_files)} input files: {existing_files}")
    
    # Create output directory if it doesn't exist
    output_dir = Path("outputs/variables/cc_risk")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory ready: {output_dir}")
    
    return True

if __name__ == "__main__":
    logger.info("STOXX600 Semantic Climate Risk Batch Analysis")
    logger.info("=" * 50)
    
    # Check prerequisites
    if not check_prerequisites():
        logger.error("Prerequisites check failed. Exiting.")
        sys.exit(1)
    
    # Run the batch analysis
    success = run_semantic_analysis_batch()
    
    if success:
        logger.info("🎉 All files processed successfully!")
        sys.exit(0)
    else:
        logger.error("⚠️  Some files failed to process. Check the log for details.")
        sys.exit(1)