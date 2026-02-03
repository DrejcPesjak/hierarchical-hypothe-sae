#!/usr/bin/env python3
"""
Utility to load saved sparse SAE vectors.
"""

import numpy as np
from scipy.sparse import csr_matrix
from pathlib import Path
import json


def load_sae_vectors(base_path):
    """
    Load sparse SAE vectors and best labels.
    
    Args:
        base_path: Base path (without extensions) to the saved files
        
    Returns:
        sparse_matrix: scipy.sparse.csr_matrix of SAE vectors
        best_labels: numpy array of best labels (0/1)
        metadata: dict with metadata
    """
    base_path = Path(base_path)
    
    # Load sparse matrix
    sparse_path = base_path.parent / f"{base_path.name}_sparse.npz"
    sparse_data = np.load(sparse_path)
    sparse_matrix = csr_matrix(
        (sparse_data['data'], sparse_data['indices'], sparse_data['indptr']),
        shape=sparse_data['shape']
    )
    
    # Load best labels
    best_path = base_path.parent / f"{base_path.name}_best.npy"
    best_labels = np.load(best_path)
    
    # Load metadata
    metadata_path = base_path.parent / f"{base_path.name}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    return sparse_matrix, best_labels, metadata


def combine_sparse_with_best(sparse_matrix, best_labels):
    """
    Combine sparse matrix with best column into a single sparse matrix.
    
    Args:
        sparse_matrix: scipy.sparse.csr_matrix of SAE vectors
        best_labels: numpy array of best labels (0/1)
        
    Returns:
        combined: scipy.sparse.csr_matrix with SAE vectors + best column
    """
    from scipy.sparse import hstack
    best_sparse = csr_matrix(best_labels)
    combined = hstack([sparse_matrix, best_sparse])
    return combined


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python load_sae_vectors.py <base_path>")
        print("Example: python load_sae_vectors.py data/sae_vectors/confirmatory_preprocessed_sae_vectors")
        sys.exit(1)
    
    base_path = sys.argv[1]
    sparse_matrix, best_labels, metadata = load_sae_vectors(base_path)
    
    print(f"Loaded sparse matrix: {sparse_matrix.shape}")
    print(f"Loaded best labels: {best_labels.shape}")
    print(f"Metadata: {metadata}")
    print(f"Sparsity: {1.0 - (sparse_matrix.nnz / sparse_matrix.size):.4f}")
    
    # Show first few rows
    print(f"\nFirst 5 rows (dense, showing first 10 features):")
    print(sparse_matrix[:5, :10].toarray())
    print(f"\nFirst 5 best labels:")
    print(best_labels[:5].flatten())

