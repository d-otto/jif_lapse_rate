#!/usr/bin/env python3
"""
Debug script to understand CSV structure
"""

import pandas as pd
from pathlib import Path

def debug_csv_structure():
    """Debug the CSV structure"""
    
    test_file = Path('/Users/drotto/src/jiflr/data/2025/raw_exported/A01 1m 2025-07-27 10_22_50 AKDT (Data AKDT).csv')
    
    print("Reading CSV...")
    df = pd.read_csv(test_file)
    
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"First few rows:")
    print(df.head(10))
    
    print(f"\nFirst column data types and first few values:")
    first_col = df.columns[0]
    print(f"Column name: '{first_col}'")
    print(f"Data type: {df[first_col].dtype}")
    print(f"First 10 values: {df[first_col].head(10).tolist()}")
    
    # Check for plot title
    first_cell = str(df.iloc[0, 0])
    print(f"\nFirst cell: '{first_cell}'")
    print(f"Starts with 'Plot Title:': {first_cell.startswith('Plot Title:')}")

if __name__ == "__main__":
    debug_csv_structure()