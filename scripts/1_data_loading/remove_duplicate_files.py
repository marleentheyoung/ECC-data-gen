#!/usr/bin/env python3
"""
Check PDF file overlap between SP500 and STOXX600 folders and move duplicates to trash.
"""

import os
from pathlib import Path

try:
    import send2trash
    TRASH_AVAILABLE = True
except ImportError:
    TRASH_AVAILABLE = False

def get_pdf_files(folder_path):
    """Get all PDF filenames from a folder."""
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Folder does not exist: {folder_path}")
        return set()
    
    pdf_files = set()
    for file in folder.glob("*.pdf"):
        pdf_files.add(file.name)
    
    return pdf_files

def main():
    sp500_path = "/Users/marleendejonge/Desktop/ECC-data-generation/data/raw/SP500"
    stoxx600_path = "/Users/marleendejonge/Desktop/ECC-data-generation/data/raw/STOXX600"
    
    # Get PDF files from both folders
    sp500_pdfs = get_pdf_files(sp500_path)
    stoxx600_pdfs = get_pdf_files(stoxx600_path)
    
    # Print folder contents
    print(f"SP500 folder: {len(sp500_pdfs)} PDF files")
    print(f"STOXX600 folder: {len(stoxx600_pdfs)} PDF files")
    print()
    
    # Find overlaps
    sp500_in_stoxx = sp500_pdfs.intersection(stoxx600_pdfs)
    
    # Files unique to each folder
    sp500_only = sp500_pdfs - stoxx600_pdfs
    stoxx600_only = stoxx600_pdfs - sp500_pdfs
    
    # Print results
    print(f"PDFs in both folders: {len(sp500_in_stoxx)}")
    print(f"PDFs only in SP500: {len(sp500_only)}")
    print(f"PDFs only in STOXX600: {len(stoxx600_only)}")
    print()
    
    # Delete duplicate PDFs from STOXX600 folder
    if sp500_in_stoxx:
        if not TRASH_AVAILABLE:
            print("WARNING: send2trash not installed. Files will be PERMANENTLY deleted!")
            print("Install with: pip install send2trash")
            response = input("Continue with permanent deletion? (yes/no): ").lower().strip()
            if response != 'yes':
                print("Operation cancelled.")
                return
        
        print(f"Moving {len(sp500_in_stoxx)} duplicate PDFs from STOXX600 folder to {'trash' if TRASH_AVAILABLE else 'permanent deletion'}...")
        deleted_count = 0
        
        for pdf_filename in sp500_in_stoxx:
            stoxx600_file_path = Path(stoxx600_path) / pdf_filename
            try:
                if stoxx600_file_path.exists():
                    if TRASH_AVAILABLE:
                        send2trash.send2trash(str(stoxx600_file_path))
                        print(f"Moved to trash: {pdf_filename}")
                    else:
                        stoxx600_file_path.unlink()
                        print(f"Permanently deleted: {pdf_filename}")
                    deleted_count += 1
            except Exception as e:
                print(f"Error deleting {pdf_filename}: {e}")
        
        print(f"\nProcessed {deleted_count} duplicate files from STOXX600 folder")
        print(f"STOXX600 folder now has {len(stoxx600_only)} unique PDF files")
    else:
        print("No duplicate files to delete")

if __name__ == "__main__":
    main()