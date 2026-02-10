#!/usr/bin/env python3
"""
Download all feature explanations for Gemma 3 1B, layer 22, 16k SAE (res)
from Neuronpedia, filter out embedding data, and combine into a single file.
"""

import gzip
import json
import requests
from pathlib import Path
from tqdm import tqdm

# Configuration

BASE_URL = "https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/v1/gemma-3-4b-it/22-gemmascope-2-res-16k/explanations"
NUM_BATCHES = 61  # batches 0-60
OUTPUT_DIR = Path(__file__).parent
OUTPUT_FILE = OUTPUT_DIR / "gemma3_4b_it_layer22_16k_explanations.json"

# Fields to exclude (embedding/vector data we don't need)
EXCLUDE_FIELDS = {"embedding", "embeddings", "vector", "vectors", "embed"}


def download_batch(batch_num: int) -> list[dict]:
    """Download and parse a single batch file."""
    url = f"{BASE_URL}/batch-{batch_num}.jsonl.gz"
    
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    
    # Decompress and parse JSONL
    decompressed = gzip.decompress(response.content)
    lines = decompressed.decode('utf-8').strip().split('\n')
    
    features = []
    for line in lines:
        if line.strip():
            feature = json.loads(line)
            # Filter out embedding fields
            filtered = {k: v for k, v in feature.items() 
                       if k.lower() not in EXCLUDE_FIELDS 
                       and not (isinstance(v, list) and len(v) > 100)}  # Also filter large arrays
            features.append(filtered)
    
    return features


def main():
    print(f"Downloading {NUM_BATCHES} batches from Neuronpedia...")
    print(f"Source: {BASE_URL}")
    print()
    
    all_features = []
    failed_batches = []
    
    for batch_num in tqdm(range(NUM_BATCHES), desc="Downloading batches"):
        try:
            features = download_batch(batch_num)
            all_features.extend(features)
        except Exception as e:
            print(f"\nWarning: Failed to download batch {batch_num}: {e}")
            failed_batches.append(batch_num)
    
    print(f"\nDownloaded {len(all_features)} feature explanations")
    
    if failed_batches:
        print(f"Failed batches: {failed_batches}")
    
    # Sort by feature index numerically
    if all_features and "index" in all_features[0]:
        all_features.sort(key=lambda x: int(x.get("index", 0)))
    
    # Show sample of what we're keeping
    if all_features:
        print(f"\nSample feature keys: {list(all_features[0].keys())}")
        print(f"Sample feature: {json.dumps(all_features[0], indent=2)[:500]}...")
    
    # Save to single file
    print(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_features, f, indent=2)
    
    file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"Done! File size: {file_size_mb:.2f} MB")
    print(f"Total features: {len(all_features)}")


if __name__ == "__main__":
    main()

