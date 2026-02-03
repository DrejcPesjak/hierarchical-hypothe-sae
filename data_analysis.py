import pandas as pd
import os
from pathlib import Path

# Set up paths
data_dir = Path("data/osfstorage-archive/upworthy-archive-datasets")

# CSV files to analyze
csv_files = {
    "exploratory": "upworthy-archive-exploratory-packages-03.12.2020.csv",
    "confirmatory": "upworthy-archive-confirmatory-packages-03.12.2020.csv",
    "holdout": "upworthy-archive-holdout-packages-03.12.2020.csv",
    "undeployed": "upworthy-archive-undeployed-packages.01.12.2021.csv"
}

print("=" * 80)
print("DATA ANALYSIS - Upworthy Archive Datasets")
print("=" * 80)

# 1. Count lines for each CSV
print("\n1. LINE COUNTS")
print("-" * 80)
for name, filename in csv_files.items():
    filepath = data_dir / filename
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f) - 1  # Subtract header
        print(f"{name:15s}: {line_count:>8,} rows ({filename})")
    else:
        print(f"{name:15s}: File not found ({filename})")

# 2. Analyze confirmatory dataset
print("\n" + "=" * 80)
print("2. CONFIRMATORY DATASET ANALYSIS")
print("=" * 80)

confirmatory_path = data_dir / csv_files["confirmatory"]
if not confirmatory_path.exists():
    print(f"Error: {confirmatory_path} not found")
    exit(1)

# Load confirmatory data
print("\nLoading confirmatory dataset...")
df = pd.read_csv(confirmatory_path, low_memory=False)

print(f"Total rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")

# Calculate CTR
print("\n2.1. CTR CALCULATION")
print("-" * 80)
df['ctr'] = df['clicks'] / df['impressions']
print(f"CTR statistics:")
print(f"  Mean CTR: {df['ctr'].mean():.6f} ({df['ctr'].mean()*100:.4f}%)")
print(f"  Median CTR: {df['ctr'].median():.6f} ({df['ctr'].median()*100:.4f}%)")
print(f"  Min CTR: {df['ctr'].min():.6f} ({df['ctr'].min()*100:.4f}%)")
print(f"  Max CTR: {df['ctr'].max():.6f} ({df['ctr'].max()*100:.4f}%)")
print(f"  Rows with zero impressions: {(df['impressions'] == 0).sum()}")
print(f"  Rows with zero clicks: {(df['clicks'] == 0).sum()}")

# Check if highest CTR = first_place within each test
print("\n2.2. CHECKING IF HIGHEST CTR = first_place WITHIN EACH TEST")
print("-" * 80)

# Group by clickability_test_id
test_groups = df.groupby('clickability_test_id')

matches = 0
mismatches = 0
tests_with_multiple_rows = 0
tests_with_single_row = 0
mismatch_details = []

for test_id, group in test_groups:
    if len(group) == 1:
        tests_with_single_row += 1
        continue
    
    tests_with_multiple_rows += 1
    
    # Find row with highest CTR
    max_ctr_idx = group['ctr'].idxmax()
    max_ctr_row = group.loc[max_ctr_idx]
    
    # Check if this row has first_place = True
    if max_ctr_row['first_place'] == True:
        matches += 1
    else:
        mismatches += 1
        # Find which row actually has first_place = True
        first_place_rows = group[group['first_place'] == True]
        if len(first_place_rows) > 0:
            first_place_row = first_place_rows.iloc[0]
            mismatch_details.append({
                'test_id': test_id,
                'max_ctr': max_ctr_row['ctr'],
                'max_ctr_first_place': max_ctr_row['first_place'],
                'first_place_ctr': first_place_row['ctr'],
                'first_place_headline': first_place_row['headline'][:60] if pd.notna(first_place_row['headline']) else 'N/A'
            })

print(f"Tests with multiple rows: {tests_with_multiple_rows:,}")
print(f"Tests with single row: {tests_with_single_row:,}")
print(f"Matches (highest CTR = first_place): {matches:,}")
print(f"Mismatches (highest CTR ≠ first_place): {mismatches:,}")
print(f"Match rate: {matches/(matches+mismatches)*100:.2f}%" if (matches+mismatches) > 0 else "N/A")

if mismatch_details:
    print(f"\nFirst {min(5, len(mismatch_details))} mismatch examples:")
    for i, detail in enumerate(mismatch_details[:5], 1):
        print(f"\n  Example {i}:")
        print(f"    Test ID: {detail['test_id']}")
        print(f"    Highest CTR: {detail['max_ctr']:.6f} (first_place={detail['max_ctr_first_place']})")
        print(f"    Actual first_place CTR: {detail['first_place_ctr']:.6f}")
        print(f"    first_place headline: {detail['first_place_headline']}")

# Check if highest clicks = first_place within each test
print("\n2.3. CHECKING IF HIGHEST CLICKS = first_place WITHIN EACH TEST")
print("-" * 80)

# Group by clickability_test_id (reuse the same grouping)
clicks_matches = 0
clicks_mismatches = 0
clicks_mismatch_details = []

for test_id, group in test_groups:
    if len(group) == 1:
        continue
    
    # Find row with highest clicks
    max_clicks_idx = group['clicks'].idxmax()
    max_clicks_row = group.loc[max_clicks_idx]
    
    # Check if this row has first_place = True
    if max_clicks_row['first_place'] == True:
        clicks_matches += 1
    else:
        clicks_mismatches += 1
        # Find which row actually has first_place = True
        first_place_rows = group[group['first_place'] == True]
        if len(first_place_rows) > 0:
            first_place_row = first_place_rows.iloc[0]
            clicks_mismatch_details.append({
                'test_id': test_id,
                'max_clicks': max_clicks_row['clicks'],
                'max_clicks_first_place': max_clicks_row['first_place'],
                'first_place_clicks': first_place_row['clicks'],
                'first_place_headline': first_place_row['headline'][:60] if pd.notna(first_place_row['headline']) else 'N/A'
            })

print(f"Tests with multiple rows: {tests_with_multiple_rows:,}")
print(f"Matches (highest clicks = first_place): {clicks_matches:,}")
print(f"Mismatches (highest clicks ≠ first_place): {clicks_mismatches:,}")
print(f"Match rate: {clicks_matches/(clicks_matches+clicks_mismatches)*100:.2f}%" if (clicks_matches+clicks_mismatches) > 0 else "N/A")

if clicks_mismatch_details:
    print(f"\nFirst {min(5, len(clicks_mismatch_details))} mismatch examples:")
    for i, detail in enumerate(clicks_mismatch_details[:5], 1):
        print(f"\n  Example {i}:")
        print(f"    Test ID: {detail['test_id']}")
        print(f"    Highest clicks: {detail['max_clicks']} (first_place={detail['max_clicks_first_place']})")
        print(f"    Actual first_place clicks: {detail['first_place_clicks']}")
        print(f"    first_place headline: {detail['first_place_headline']}")

# Check uniqueness of (headline, lede) within each test_id
print("\n2.4. UNIQUENESS OF (headline, lede) WITHIN EACH TEST")
print("-" * 80)

# Create a signature from only headline and lede
df['headline_lede_signature'] = df[['headline', 'lede']].apply(
    lambda row: '|'.join([str(val) if pd.notna(val) else '' for val in row]), 
    axis=1
)

# Group by test_id and count unique (headline, lede) combinations per test
test_uniqueness = df.groupby('clickability_test_id')['headline_lede_signature'].nunique()

total_unique_test_ids = len(test_uniqueness)
tests_with_multiple_unique_combos = (test_uniqueness > 1).sum()
tests_with_single_unique_combo = (test_uniqueness == 1).sum()

total_rows = len(df)
unique_headline_lede_combos = df['headline_lede_signature'].nunique()

print(f"Total rows: {total_rows:,}")
print(f"Total unique test_ids: {total_unique_test_ids:,}")
print(f"Unique (headline, lede) combinations across all data: {unique_headline_lede_combos:,}")
print(f"\nWithin each test_id:")
print(f"  Tests with >1 unique (headline, lede) combinations: {tests_with_multiple_unique_combos:,}")
print(f"  Tests with exactly 1 unique (headline, lede) combination: {tests_with_single_unique_combo:,}")
print(f"  Percentage of tests with >1 unique combinations: {tests_with_multiple_unique_combos/total_unique_test_ids*100:.2f}%")

# Calculate mean and median for tests with >1 unique combinations
multi_combo_counts = test_uniqueness[test_uniqueness > 1]
if len(multi_combo_counts) > 0:
    mean_unique_combos = multi_combo_counts.mean()
    median_unique_combos = multi_combo_counts.median()
    print(f"\nFor tests with >1 unique (headline, lede) combinations:")
    print(f"  Mean number of unique combinations: {mean_unique_combos:.2f}")
    print(f"  Median number of unique combinations: {median_unique_combos:.1f}")
    print(f"  Min number of unique combinations: {multi_combo_counts.min()}")
    print(f"  Max number of unique combinations: {multi_combo_counts.max()}")

# Show some examples of tests with multiple unique (headline, lede) combinations
if tests_with_multiple_unique_combos > 0:
    print(f"\nExamples of tests with multiple unique (headline, lede) combinations:")
    multi_combo_tests = test_uniqueness[test_uniqueness > 1].sort_values(ascending=False)
    for i, (test_id, num_combos) in enumerate(multi_combo_tests.head(5).items(), 1):
        test_rows = df[df['clickability_test_id'] == test_id][['headline', 'lede']].drop_duplicates()
        print(f"\n  Example {i}: Test ID {test_id} has {num_combos} unique (headline, lede) combinations")
        for idx, row in test_rows.head(3).iterrows():
            headline_preview = str(row['headline'])[:60] + '...' if pd.notna(row['headline']) and len(str(row['headline'])) > 60 else (str(row['headline']) if pd.notna(row['headline']) else 'N/A')
            lede_preview = str(row['lede'])[:60] + '...' if pd.notna(row['lede']) and len(str(row['lede'])) > 60 else (str(row['lede']) if pd.notna(row['lede']) else 'N/A')
            print(f"    - Headline: {headline_preview}")
            print(f"      Lede: {lede_preview}")

print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)

