"""
Data Cleaning Script for SPX Options CSV

This script removes duplicate entries from the OpenBB SPX options CSV file.
Run this BEFORE loading data into the WRDSOptionsDataset.

Problem: The CSV may contain duplicate rows with the same:
- date (quote date)
- exdate (expiry date)
- strike_price
- cp_flag (call/put)

This causes "duplicate index" errors when the dataset tries to reshape.

Solution: Keep the first occurrence of each duplicate.

Usage:
    python clean_data.py
"""

import pandas as pd
from pathlib import Path

# Configuration
DATA_DIR = Path("volatility_smoothing/data/openbb/spx")
INPUT_CSV = DATA_DIR / "spx_options_2025-12-11.csv"  # Update this filename
OUTPUT_CSV = DATA_DIR / "spx_options_2025-12-11_cleaned.csv"

def clean_csv(input_path, output_path):
    """Remove duplicates from SPX options CSV"""

    print(f"Reading CSV: {input_path}")
    df = pd.read_csv(input_path)

    print(f"Total rows: {len(df)}")

    # Check for duplicates
    duplicate_cols = ['date', 'exdate', 'strike_price', 'cp_flag']
    duplicates = df.duplicated(subset=duplicate_cols, keep='first')
    num_duplicates = duplicates.sum()

    print(f"Duplicate rows found: {num_duplicates}")

    if num_duplicates > 0:
        # Remove duplicates (keep first occurrence)
        df_clean = df[~duplicates].copy()

        print(f"Rows after cleaning: {len(df_clean)}")
        print(f"Rows removed: {num_duplicates}")

        # Save cleaned CSV
        df_clean.to_csv(output_path, index=False)
        print(f"\nCleaned CSV saved to: {output_path}")
        print(f"\nNext steps:")
        print(f"1. Delete the old cached data: rm -rf ~/.cache/opds/WRDSOptionsDataset/")
        print(f"2. Either:")
        print(f"   a) Rename cleaned file to replace original")
        print(f"   b) Update OPDS_WRDS_DATA_DIR to point to cleaned file")
    else:
        print("No duplicates found. CSV is clean!")

    return df_clean if num_duplicates > 0 else df

if __name__ == "__main__":
    # Check if input file exists
    if not INPUT_CSV.exists():
        print(f"Error: File not found: {INPUT_CSV}")
        print(f"\nPlease update INPUT_CSV in this script to point to your CSV file.")
        exit(1)

    # Clean the data
    clean_csv(INPUT_CSV, OUTPUT_CSV)
