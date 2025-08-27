#!/usr/bin/env python3
"""
Remove climate snippet entries that reference PDFs deleted from STOXX600 folder.
Creates cleaned versions in STOXX600_2608 folder with removal log.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set
from tqdm import tqdm
import shutil

def setup_logging(log_file: Path):
    """Set up logging to track removals."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )

def get_existing_pdfs(stoxx600_path: Path) -> Set[str]:
    """Get set of PDF filenames that still exist in STOXX600 folder."""
    if not stoxx600_path.exists():
        raise FileNotFoundError(f"STOXX600 path not found: {stoxx600_path}")
    
    pdf_files = set()
    for pdf_file in stoxx600_path.glob("*.pdf"):
        pdf_files.add(pdf_file.name)
    
    print(f"Found {len(pdf_files)} PDF files in STOXX600 folder")
    return pdf_files

def clean_climate_file(input_file: Path, output_file: Path, existing_pdfs: Set[str]) -> Dict:
    """
    Clean a single climate snippets file, removing entries for deleted PDFs.
    
    Returns:
        Dictionary with statistics about removals
    """
    logger = logging.getLogger(__name__)
    
    # Load climate data
    with open(input_file, 'r', encoding='utf-8') as f:
        climate_data = json.load(f)
    
    original_count = len(climate_data)
    cleaned_data = []
    removed_entries = []
    
    for entry in climate_data:
        filename = entry.get('file', '')
        
        if not filename:
            # Keep entries without filename (shouldn't happen but just in case)
            cleaned_data.append(entry)
            continue
        
        # Extract just the filename from potential path
        pdf_name = Path(filename).name
        
        if pdf_name in existing_pdfs:
            # PDF still exists, keep this entry
            cleaned_data.append(entry)
        else:
            # PDF was deleted, remove this entry
            removed_entries.append({
                'file': filename,
                'company': entry.get('company_name', 'Unknown'),
                'ticker': entry.get('ticker', 'Unknown'),
                'year': entry.get('year', 'Unknown'),
                'quarter': entry.get('quarter', 'Unknown')
            })
            logger.info(f"Removing entry for deleted PDF: {pdf_name}")
    
    # Save cleaned data
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
    
    # Statistics
    removed_count = len(removed_entries)
    kept_count = len(cleaned_data)
    
    stats = {
        'file': input_file.name,
        'original_count': original_count,
        'kept_count': kept_count,
        'removed_count': removed_count,
        'removal_rate': removed_count / original_count if original_count > 0 else 0,
        'removed_entries': removed_entries
    }
    
    logger.info(f"File {input_file.name}: kept {kept_count}/{original_count} entries ({removed_count} removed)")
    
    return stats

def save_removal_report(all_stats: List[Dict], output_dir: Path):
    """Save detailed removal report."""
    
    # Summary statistics
    total_original = sum(s['original_count'] for s in all_stats)
    total_kept = sum(s['kept_count'] for s in all_stats)
    total_removed = sum(s['removed_count'] for s in all_stats)
    
    summary = {
        'summary': {
            'total_files_processed': len(all_stats),
            'total_original_entries': total_original,
            'total_kept_entries': total_kept,
            'total_removed_entries': total_removed,
            'overall_removal_rate': total_removed / total_original if total_original > 0 else 0
        },
        'file_details': all_stats
    }
    
    # Save summary report
    report_file = output_dir / 'removal_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Save CSV of all removed entries for easy inspection
    all_removed = []
    for file_stats in all_stats:
        for entry in file_stats['removed_entries']:
            entry['source_file'] = file_stats['file']
            all_removed.append(entry)
    
    if all_removed:
        import csv
        csv_file = output_dir / 'removed_entries.csv'
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['source_file', 'file', 'company', 'ticker', 'year', 'quarter'])
            writer.writeheader()
            writer.writerows(all_removed)
        
        print(f"Saved {len(all_removed)} removed entries to: {csv_file}")
    
    print(f"Removal report saved to: {report_file}")
    return summary

def main():
    # Paths
    stoxx600_pdf_path = Path("/Users/marleendejonge/Desktop/ECC-data-generation/data/raw/STOXX600")
    climate_input_path = Path("/Users/marleendejonge/Desktop/ECC-data-generation/data/climate_paragraphs/STOXX600")
    output_base = Path("/Users/marleendejonge/Desktop/ECC-data-generation/data/climate_paragraphs")
    output_path = output_base / "STOXX600_2608"
    log_file = output_path / "cleanup.log"
    
    print("Climate Snippet Orphan Cleaner")
    print("=" * 50)
    print(f"STOXX600 PDFs: {stoxx600_pdf_path}")
    print(f"Climate snippets: {climate_input_path}")
    print(f"Output: {output_path}")
    
    # Setup logging
    output_path.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file)
    logger = logging.getLogger(__name__)
    
    try:
        # Get existing PDFs
        existing_pdfs = get_existing_pdfs(stoxx600_pdf_path)
        
        # Find climate snippet files
        climate_files = list(climate_input_path.glob("climate_segments_*.json"))
        
        if not climate_files:
            print(f"No climate segment files found in {climate_input_path}")
            return
        
        print(f"Found {len(climate_files)} climate snippet files to process")
        
        # Process each file
        all_stats = []
        
        for climate_file in tqdm(climate_files, desc="Processing files"):
            output_file = output_path / climate_file.name
            
            try:
                file_stats = clean_climate_file(climate_file, output_file, existing_pdfs)
                all_stats.append(file_stats)
            except Exception as e:
                logger.error(f"Error processing {climate_file}: {e}")
                continue
        
        # Generate removal report
        summary = save_removal_report(all_stats, output_path)
        
        # Print summary
        print("\nCleaning Summary:")
        print(f"Files processed: {summary['summary']['total_files_processed']}")
        print(f"Original entries: {summary['summary']['total_original_entries']:,}")
        print(f"Kept entries: {summary['summary']['total_kept_entries']:,}")
        print(f"Removed entries: {summary['summary']['total_removed_entries']:,}")
        print(f"Overall removal rate: {summary['summary']['overall_removal_rate']:.1%}")
        
        print(f"\nCleaned files saved to: {output_path}")
        print(f"Detailed log: {log_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Main error: {e}")
        raise

if __name__ == "__main__":
    main()