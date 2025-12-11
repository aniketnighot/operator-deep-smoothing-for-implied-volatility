#!/usr/bin/env python3
"""
Fetch SPX Options Data from OpenBB and save to WRDS-compatible CSV format.

This script:
1. Fetches SPX options data from OpenBB Platform (using yfinance provider)
2. Transforms to exact WRDS format (including strike × 1000)
3. Cleans and validates data
4. Saves to CSV in volatility_smoothing/data/openbb/spx/

No changes to existing code required - the output CSV works directly with WRDSOptionsDataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from openbb import obb

# Configure OpenBB
obb.user.preferences.output_type = "dataframe"

def fetch_spx_options():
    """Fetch SPX options data from OpenBB."""
    print("=" * 80)
    print("Fetching SPX options data from OpenBB...")
    print("Provider: CBOE (free tier)")
    print("=" * 80)

    # Fetch data
    df_openbb = obb.derivatives.options.chains(symbol="SPX", provider="cboe")

    print(f"\n✓ Fetched {len(df_openbb)} option quotes")
    print(f"\nAvailable columns: {list(df_openbb.columns)}")

    return df_openbb


def transform_to_wrds_format(df_openbb):
    """Transform OpenBB data to WRDS-compatible format."""
    print("\n" + "=" * 80)
    print("Transforming to WRDS format...")
    print("=" * 80)

    # Create WRDS-compatible DataFrame
    df_wrds = pd.DataFrame()

    # Map quote date - use today's date for all rows
    quote_date = datetime.now().strftime('%Y-%m-%d')
    df_wrds['date'] = [quote_date] * len(df_openbb)

    # Map expiration date
    df_wrds['exdate'] = pd.to_datetime(df_openbb['expiration'])

    # Map strike price - CRITICAL: Multiply by 1000!
    df_wrds['strike_price'] = df_openbb['strike'] * 1000

    # Map option type to 'C' or 'P'
    df_wrds['cp_flag'] = df_openbb['option_type'].str.upper().str[0]  # 'call' -> 'C', 'put' -> 'P'

    # Map bid and ask prices
    df_wrds['best_bid'] = df_openbb['bid']
    df_wrds['best_offer'] = df_openbb['ask']

    # Set am_settlement to 1 (required by WRDS format)
    df_wrds['am_settlement'] = 1

    print(f"\n✓ Transformed {len(df_wrds)} rows to WRDS format")
    print(f"\nWRDS columns: {list(df_wrds.columns)}")

    return df_wrds


def validate_data(df_wrds):
    """Perform data quality checks."""
    print("\n" + "=" * 80)
    print("Data Quality Checks")
    print("=" * 80)

    # Check for missing values
    print("\n1. Missing values:")
    missing = df_wrds.isnull().sum()
    if missing.sum() == 0:
        print("   ✓ No missing values")
    else:
        print(missing[missing > 0])

    # Check option types
    print("\n2. Option types:")
    print(df_wrds['cp_flag'].value_counts())

    # Check for zero or negative prices
    zero_bid = (df_wrds['best_bid'] <= 0).sum()
    zero_ask = (df_wrds['best_offer'] <= 0).sum()
    print(f"\n3. Zero or negative bid prices: {zero_bid}")
    print(f"4. Zero or negative ask prices: {zero_ask}")

    # Check strike price range (after multiplication by 1000)
    print("\n5. Strike price range:")
    print(f"   Min: {df_wrds['strike_price'].min():.0f} (actual strike: {df_wrds['strike_price'].min()/1000:.2f})")
    print(f"   Max: {df_wrds['strike_price'].max():.0f} (actual strike: {df_wrds['strike_price'].max()/1000:.2f})")

    # Check expiration dates
    print("\n6. Expiration dates:")
    print(f"   Unique expiries: {df_wrds['exdate'].nunique()}")
    print(f"   Date range: {df_wrds['exdate'].min()} to {df_wrds['exdate'].max()}")

    # Count options per expiry
    options_per_expiry = df_wrds.groupby('exdate').size()
    print("\n7. Options per expiry:")
    print(f"   Min: {options_per_expiry.min()}")
    print(f"   Max: {options_per_expiry.max()}")
    print(f"   Mean: {options_per_expiry.mean():.1f}")


def clean_data(df_wrds):
    """Remove invalid entries."""
    print("\n" + "=" * 80)
    print("Cleaning data...")
    print("=" * 80)

    print(f"\nInitial rows: {len(df_wrds)}")

    # Remove rows with missing values
    df_clean = df_wrds.dropna()
    print(f"After removing NaN: {len(df_clean)}")

    # Remove expired options (expiry date in the past)
    df_clean['exdate'] = pd.to_datetime(df_clean['exdate'])
    today = pd.Timestamp.now().normalize()
    df_clean = df_clean[df_clean['exdate'] > today]
    print(f"After removing expired options: {len(df_clean)}")

    # Remove zero or negative prices
    df_clean = df_clean[
        (df_clean['best_bid'] > 0) &
        (df_clean['best_offer'] > 0)
    ]
    print(f"After removing zero/negative prices: {len(df_clean)}")

    # Ensure valid option types
    df_clean = df_clean[df_clean['cp_flag'].isin(['C', 'P'])]
    print(f"After validating option types: {len(df_clean)}")

    # Remove duplicates - keep first occurrence for each (date, exdate, strike, option_type)
    # This handles cases where CBOE has multiple quotes for the same option
    before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['date', 'exdate', 'strike_price', 'cp_flag'], keep='first')
    print(f"After removing duplicates: {len(df_clean)} (removed {before_dedup - len(df_clean)} duplicates)")

    # Sort by date, expiry, strike
    df_clean = df_clean.sort_values(['date', 'exdate', 'strike_price'])

    print(f"\n✓ Final clean data: {len(df_clean)} rows")
    print(f"✓ Unique expiries: {df_clean['exdate'].nunique()}")

    return df_clean


def save_to_csv(df_clean):
    """Save to CSV in WRDS-compatible format."""
    print("\n" + "=" * 80)
    print("Saving to CSV...")
    print("=" * 80)

    # Create output directory
    output_dir = Path(__file__).parent / 'openbb' / 'spx'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create filename with date
    quote_date = df_clean['date'].iloc[0]
    if isinstance(quote_date, str):
        date_str = quote_date
    else:
        date_str = quote_date.strftime('%Y-%m-%d')

    output_file = output_dir / f'spx_options_{date_str}.csv'

    # Save to CSV
    df_clean.to_csv(output_file, index=False)

    print(f"\n✓ Saved WRDS-compatible CSV to:")
    print(f"  {output_file}")
    print(f"\n✓ File size: {output_file.stat().st_size / 1024:.1f} KB")

    return output_file


def verify_csv(output_file):
    """Verify CSV format by reading it back."""
    print("\n" + "=" * 80)
    print("Verifying CSV format...")
    print("=" * 80)

    # Read back the CSV
    df_verify = pd.read_csv(output_file)

    expected_columns = ['date', 'exdate', 'strike_price', 'cp_flag', 'best_bid', 'best_offer', 'am_settlement']

    print(f"\nColumns: {list(df_verify.columns)}")
    print(f"Expected: {expected_columns}")

    if list(df_verify.columns) == expected_columns:
        print("\n✓ Column names match expected format")
    else:
        print("\n✗ Column names do NOT match expected format")

    print(f"\nRows: {len(df_verify)}")
    print(f"\nData types:")
    print(df_verify.dtypes)

    print(f"\nSample data (first 5 rows):")
    print(df_verify.head())

    print("\n" + "=" * 80)
    print("✓ SUCCESS! CSV is ready to use with WRDSOptionsDataset")
    print("=" * 80)

    print("\nTo use with existing code, set in your notebook:")
    print(f"  import os")
    print(f"  os.environ['OPDS_WRDS_DATA_DIR'] = '{output_file.parent.absolute()}'")


def main():
    """Main execution function."""
    try:
        # Step 1: Fetch data
        df_openbb = fetch_spx_options()

        # Step 2: Transform to WRDS format
        df_wrds = transform_to_wrds_format(df_openbb)

        # Step 3: Validate data
        validate_data(df_wrds)

        # Step 4: Clean data
        df_clean = clean_data(df_wrds)

        # Step 5: Save to CSV
        output_file = save_to_csv(df_clean)

        # Step 6: Verify CSV
        verify_csv(output_file)

        print("\n" + "=" * 80)
        print("DONE!")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
