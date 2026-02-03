import pandas as pd
import re
from pathlib import Path
import argparse


def remove_html_tags(text):
    """Remove HTML tags from text."""
    if pd.isna(text):
        return text
    text = str(text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove HTML entities
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    text = text.replace('&#39;', "'")
    # Remove other common HTML entities
    text = re.sub(r'&#\d+;', '', text)
    text = re.sub(r'&\w+;', '', text)
    return text


def clean_text(text):
    """Remove HTML tags and clean weird characters."""
    if pd.isna(text):
        return text
    text = remove_html_tags(text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def preprocess_csv(input_file, output_file):
    """
    Preprocess CSV file:
    1. Extract test_id, headline, lede, impressions, clicks
    2. Remove duplicates based on (test_id, headline, lede)
    3. Remove test_ids with only 1 row
    4. Clean HTML tags and weird characters
    5. Calculate 'best' column (best CTR within test_id = 1, others = 0)
    6. Save to new CSV
    """
    print("=" * 80)
    print("DATA PREPROCESSING")
    print("=" * 80)
    
    # Load CSV
    print(f"\nLoading CSV: {input_file}")
    df = pd.read_csv(input_file, low_memory=False)
    print(f"Original rows: {len(df):,}")
    print(f"Original columns: {list(df.columns)}")
    
    # Extract required columns
    required_cols = ['clickability_test_id', 'headline', 'lede', 'impressions', 'clicks']
    
    # Check if columns exist (handle different naming)
    col_mapping = {}
    for col in required_cols:
        if col in df.columns:
            col_mapping[col] = col
        elif col.replace('_', '') in [c.replace('_', '') for c in df.columns]:
            # Try to find similar column name
            for df_col in df.columns:
                if df_col.replace('_', '').lower() == col.replace('_', '').lower():
                    col_mapping[col] = df_col
                    break
    
    # Check if all required columns are found
    missing_cols = [col for col in required_cols if col not in col_mapping]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Select and rename columns
    df_clean = df[list(col_mapping.values())].copy()
    df_clean.columns = required_cols
    
    print(f"\nStep 1: Extracted columns: {required_cols}")
    print(f"Rows after extraction: {len(df_clean):,}")
    
    # Remove duplicates based on (test_id, headline, lede)
    print(f"\nStep 2: Removing duplicates based on (test_id, headline, lede)...")
    before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['clickability_test_id', 'headline', 'lede'], keep='first')
    after_dedup = len(df_clean)
    print(f"  Removed {before_dedup - after_dedup:,} duplicate rows")
    print(f"  Rows after deduplication: {after_dedup:,}")
    
    # Remove test_ids with only 1 row
    print(f"\nStep 3: Removing test_ids with only 1 row...")
    test_counts = df_clean['clickability_test_id'].value_counts()
    tests_with_multiple = test_counts[test_counts > 1].index
    before_filter = len(df_clean)
    df_clean = df_clean[df_clean['clickability_test_id'].isin(tests_with_multiple)]
    after_filter = len(df_clean)
    print(f"  Removed {before_filter - after_filter:,} rows from tests with only 1 row")
    print(f"  Removed {len(test_counts) - len(tests_with_multiple):,} test_ids")
    print(f"  Rows after filtering: {after_filter:,}")
    print(f"  Remaining test_ids: {len(tests_with_multiple):,}")
    
    # Clean HTML tags and weird characters
    print(f"\nStep 4: Cleaning HTML tags and weird characters...")
    df_clean['headline'] = df_clean['headline'].apply(clean_text)
    df_clean['lede'] = df_clean['lede'].apply(clean_text)
    print(f"  Cleaned headline and lede columns")
    
    # Calculate CTR and 'best' column
    print(f"\nStep 5: Calculating CTR and 'best' column...")
    df_clean['ctr'] = df_clean['clicks'] / df_clean['impressions']
    # Replace inf/NaN from division by zero
    df_clean['ctr'] = df_clean['ctr'].replace([float('inf'), float('-inf')], 0)
    df_clean['ctr'] = df_clean['ctr'].fillna(0)
    
    # For each test_id, mark the row with highest CTR as best=1, others as 0
    df_clean['best'] = 0
    for test_id, group in df_clean.groupby('clickability_test_id'):
        max_ctr_idx = group['ctr'].idxmax()
        df_clean.loc[max_ctr_idx, 'best'] = 1
    
    best_count = df_clean['best'].sum()
    print(f"  Marked {best_count:,} rows as best (highest CTR within each test)")
    
    # Reorder columns for output
    output_cols = ['clickability_test_id', 'headline', 'lede', 'impressions', 'clicks', 'ctr', 'best']
    df_clean = df_clean[output_cols]
    
    # Save to new CSV
    print(f"\nStep 6: Saving to {output_file}...")
    df_clean.to_csv(output_file, index=False)
    print(f"  Saved {len(df_clean):,} rows to {output_file}")
    
    # Print summary statistics
    print(f"\n" + "=" * 80)
    print("PREPROCESSING SUMMARY")
    print("=" * 80)
    print(f"Final rows: {len(df_clean):,}")
    print(f"Final test_ids: {df_clean['clickability_test_id'].nunique():,}")
    print(f"Rows marked as best: {best_count:,}")
    print(f"Average CTR: {df_clean['ctr'].mean():.6f} ({df_clean['ctr'].mean()*100:.4f}%)")
    print(f"Median CTR: {df_clean['ctr'].median():.6f} ({df_clean['ctr'].median()*100:.4f}%)")
    print("=" * 80)
    print("Preprocessing complete!")


if __name__ == "__main__":
    # python3 data_preprocess.py data/osfstorage-archive/upworthy-archive-datasets/upworthy-archive-holdout-packages-03.12.2020.csv data/cleaned_archive/holdout-preprocessed.csv
    parser = argparse.ArgumentParser(description='Preprocess Upworthy CSV data')
    parser.add_argument('input_file', type=str, help='Input CSV file path')
    parser.add_argument('output_file', type=str, help='Output CSV file path')
    
    args = parser.parse_args()
    
    preprocess_csv(args.input_file, args.output_file)

