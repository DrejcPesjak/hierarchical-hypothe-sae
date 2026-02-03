#!/usr/bin/env python3
"""
Quick analysis of token counts in CSV files using Gemma-3 tokenizer.
"""

import pandas as pd
from transformers import AutoTokenizer
from pathlib import Path
import numpy as np
from tqdm import tqdm

def analyze_token_counts(csv_path):
    """Analyze token counts in a CSV file."""
    print(f"Loading: {csv_path}")
    
    # Load tokenizer
    print("Loading Gemma-3 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Total rows: {len(df):,}")
    
    # Combine headline + lede into sentences
    df['sentence'] = df.apply(
        lambda row: f"{row['headline']} {row['lede']}" if pd.notna(row['headline']) and pd.notna(row['lede']) 
        else (str(row['headline']) if pd.notna(row['headline']) else ''),
        axis=1
    )
    
    # Filter out empty sentences
    df = df[df['sentence'].str.strip() != ''].copy()
    print(f"Rows with valid sentences: {len(df):,}")
    
    # Format prompts (same as in extraction script)
    def format_prompt(user_prompt: str) -> str:
        return f"""<start_of_turn>user
{user_prompt}<end_of_turn>
<start_of_turn>model
"""
    
    sentences = df['sentence'].tolist()
    prompts = [format_prompt(s) for s in sentences]
    
    # Tokenize all sentences (batch for speed)
    print("Tokenizing sentences...")
    token_counts = []
    
    batch_size = 1000
    for i in tqdm(range(0, len(prompts), batch_size), desc="Tokenizing"):
        batch = prompts[i:i+batch_size]
        tokenized = tokenizer(batch, add_special_tokens=True)
        # Get length of each sequence
        batch_counts = [len(ids) for ids in tokenized['input_ids']]
        token_counts.extend(batch_counts)
    
    token_counts = np.array(token_counts)
    
    # Calculate statistics
    print("\n" + "=" * 80)
    print("TOKEN COUNT STATISTICS")
    print("=" * 80)
    print(f"Min tokens:    {token_counts.min():>6.0f}")
    print(f"Max tokens:    {token_counts.max():>6.0f}")
    print(f"Mean tokens:   {token_counts.mean():>6.1f}")
    print(f"Median tokens: {np.median(token_counts):>6.1f}")
    print(f"Std tokens:    {token_counts.std():>6.1f}")
    
    # Percentiles
    print("\nPercentiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        val = np.percentile(token_counts, p)
        print(f"  {p:2d}th percentile: {val:>6.1f}")
    
    # Counts over thresholds
    print("\n" + "=" * 80)
    print("THRESHOLD COUNTS")
    print("=" * 80)
    thresholds = [128, 256, 512, 1024]
    for threshold in thresholds:
        count = (token_counts > threshold).sum()
        pct = (count / len(token_counts)) * 100
        print(f"Over {threshold:4d} tokens: {count:>6,} ({pct:>5.2f}%)")
    
    # Distribution bins
    print("\n" + "=" * 80)
    print("DISTRIBUTION")
    print("=" * 80)
    bins = [0, 64, 128, 256, 512, 1024, float('inf')]
    bin_labels = ['0-64', '64-128', '128-256', '256-512', '512-1024', '1024+']
    
    for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        if high == float('inf'):
            count = (token_counts >= low).sum()
        else:
            count = ((token_counts >= low) & (token_counts < high)).sum()
        pct = (count / len(token_counts)) * 100
        print(f"{bin_labels[i]:>12s}: {count:>6,} ({pct:>5.2f}%)")
    
    print("=" * 80)
    
    return token_counts

if __name__ == "__main__":
    csv_path = Path("data/cleaned_archive/confirmatory_preprocessed.csv")
    analyze_token_counts(csv_path)

