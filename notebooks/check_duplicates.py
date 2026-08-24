#!/usr/bin/env python3
"""
Minimal script to check for duplicate rows within CSV/TXT files in a folder.
"""

import argparse
import pandas as pd
from pathlib import Path


def check_file_duplicates(file_path, show_rows=False):
    """Check for duplicate rows in a single file."""
    try:
        # Try different separators for CSV/TXT files
        separators = [',', '\t', ';', '|']
        df = None
        
        for sep in separators:
            try:
                df = pd.read_csv(file_path, sep=sep)
                if df.shape[1] > 1:  # If we got multiple columns, separator worked
                    break
            except:
                continue
        
        if df is None or df.empty:
            return None
            
        # Check for duplicates
        duplicate_mask = df.duplicated()
        num_duplicates = duplicate_mask.sum()
        
        if num_duplicates > 0:
            result = {
                'file': file_path,
                'total_rows': len(df),
                'duplicate_rows': num_duplicates
            }
            
            if show_rows:
                result['duplicates'] = df[df.duplicated(keep=False)]
                
            return result
            
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        
    return None


def main():
    parser = argparse.ArgumentParser(description='Check for duplicate rows in CSV/TXT files')
    parser.add_argument('folder', help='Folder to scan for CSV/TXT files')
    parser.add_argument('--show-rows', action='store_true', help='Show actual duplicate rows')
    parser.add_argument('--no-recursive', dest='recursive', action='store_false', 
                       help='Scan only the target folder (not subfolders)')
    parser.set_defaults(recursive=True)
    
    args = parser.parse_args()
    
    folder_path = Path(args.folder)
    if not folder_path.exists():
        print(f"Error: Folder '{folder_path}' does not exist")
        return
    
    # Find all CSV and TXT files
    if args.recursive:
        file_patterns = ['**/*.csv', '**/*.txt']
        scan_type = "recursively"
    else:
        file_patterns = ['*.csv', '*.txt']
        scan_type = "in current directory only"
        
    files_found = []
    for pattern in file_patterns:
        files_found.extend(folder_path.glob(pattern))
    
    if not files_found:
        print(f"No CSV or TXT files found in {folder_path} ({scan_type})")
        return
    
    print(f"Scanning {len(files_found)} files for duplicates ({scan_type})...")
    print("-" * 50)
    
    files_with_duplicates = 0
    
    for file_path in files_found:
        result = check_file_duplicates(file_path, args.show_rows)
        if result:
            files_with_duplicates += 1
            print(f"\n📄 {result['file']}")
            print(f"   Total rows: {result['total_rows']}")
            print(f"   Duplicate rows: {result['duplicate_rows']}")
            
            if args.show_rows and 'duplicates' in result:
                print("   Duplicate data:")
                print(result['duplicates'].to_string(index=False))
    
    print(f"\n" + "=" * 50)
    print(f"Summary: {files_with_duplicates} files contain duplicate rows")


if __name__ == "__main__":
    main()