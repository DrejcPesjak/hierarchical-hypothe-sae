#!/usr/bin/env python3
"""
Analyze SAE vectors: sparsity, statistics, scaling effects, and top activating examples.

Note: SAE vectors are DELTA vectors constructed as:
- For each test_id: extract best (max CTR) and worst (min CTR) headlines
- Row 2i: best - worst (label=1)
- Row 2i+1: worst - best (label=0)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.sparse import csr_matrix, vstack
from sklearn.preprocessing import StandardScaler, MaxAbsScaler

# ============================================================================
# Configuration
# ============================================================================

# Top 10 feature indices and importance scores:
# Feature 9859: 3.9675557613372803
# Feature 11911: 3.7424869537353516
# Feature 13417: 3.6737632751464844
# Feature 5439: 3.661475658416748
# Feature 11052: 3.4065022468566895
# Feature 433: 3.3649163246154785
# Feature 3759: 3.3260498046875
# Feature 4294: 3.177598237991333
# Feature 1847: 2.9842286109924316
# Feature 15267: 2.94632887840271

# Bottom 10 feature indices and importance scores:
# Feature 11550: -3.4492571353912354
# Feature 15081: -3.2961766719818115
# Feature 5035: -2.9246556758880615
# Feature 6050: -2.796236991882324
# Feature 11541: -2.7860469818115234
# Feature 11411: -2.7794342041015625
# Feature 13014: -2.7538905143737793
# Feature 2963: -2.70485258102417
# Feature 15180: -2.6903228759765625
# Feature 8327: -2.6799654960632324

# Features to analyze (from logistic regression)
TOP_FEATURES = [9859, 11911, 13417, 5439, 11052, 433, 3759, 4294, 1847, 15267]
BOTTOM_FEATURES = [11550, 15081, 5035, 6050, 11541, 11411, 13014, 2963, 15180, 8327]


# Path 1: f11052: How to write clickbait titles
# Path 2: f538: regulation, risk, and governance
# Path 3: f12614: regex questions and explanations
# Path 4: f613: clarifying question
# Path 5: f726: foreign language text
# Path 6: f1847: shocking or horrifying scenarios
# Path 7: f2727: internet social media slang
# Path 8: f605: past-tense narration
# Path 9: f1389: past events
# Path 10: f605: past-tense narration
# Path 11: f538: regulation, risk, and governance
# Path 12: f5439: offensive content
# Path 13: f538: regulation, risk, and governance
# Path 14: f1531: discovery and revelation
# Path 15: f613: clarifying question

FEATURE_LABELS = {
    9859: "objective factual rational",
    11911: "undesirable qualities",
    13417: "rigorous hiring and screening",
    5439: "offensive content",
    11052: "How to write clickbait titles",
    433: "",
    3759: "evil AI, monsters",
    4294: "biotic entities and ecosystems",
    1847: "shocking or horrifying scenarios",
    15267: "place names and locations",
    11550: "parts and components breakdown",
    15081: "i understand you",
    5035: "erotic story",
    6050: "guaranteed limits",
    11541: "historical and natural landmarks",
    11411: "handling, dealing, managing",
    13014: "critical thinking, role, pedagogy, path",
    2963: "constantly evolving",
    15180: "tax revenue and fees",
    8327: "referencing previous results or text",
    
    538: "regulation, risk, and governance",
    12614: "regex questions and explanations",
    613: "clarifying question",
    726: "foreign language text",
    1847: "shocking or horrifying scenarios",
    2727: "internet social media slang",
    605: "past-tense narration",
    1389: "past events",
    5439: "offensive content",
    1531: "discovery and revelation",
}

# ============================================================================
# Data Loading
# ============================================================================

def load_sae_vectors(dataset="confirmatory"):
    """Load SAE vectors from sparse minibatch files."""
    batch_dir = Path("./data/sae_vectors")
    pattern = f"{dataset}_preprocessed_sae_vectors_sparse_minibatch_*.npz"
    batch_files = sorted(batch_dir.glob(pattern))
    
    if not batch_files:
        raise FileNotFoundError(f"No files found matching {pattern}")
    
    print(f"Loading {len(batch_files)} batch files for {dataset}...")
    
    Xs = []
    for f in batch_files:
        z = np.load(f)
        Xs.append(csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"])))
    
    X = vstack(Xs, format="csr")
    
    # Load best labels
    best_pattern = f"{dataset}_preprocessed_sae_vectors_best_minibatch_*.npy"
    best_files = sorted(batch_dir.glob(best_pattern))
    y = np.concatenate([np.load(f).reshape(-1) for f in best_files], axis=0)
    
    return X, y


def load_csv_data(dataset="confirmatory"):
    """Load the original CSV with headlines."""
    csv_path = Path(f"./data/cleaned_archive/{dataset}_preprocessed.csv")
    return pd.read_csv(csv_path)


def get_best_worst_pairs(df):
    """
    Extract best and worst headline pairs per test_id based on CTR.
    This replicates the logic from extract_sae_vectors.py.
    
    Args:
        df: DataFrame with columns: clickability_test_id, headline, ctr
    
    Returns:
        List of tuples: (best_headline, worst_headline, test_id, best_ctr, worst_ctr)
    """
    pairs = []
    
    for test_id, group in df.groupby('clickability_test_id'):
        # Need at least 2 different headlines
        if len(group) < 2:
            continue
        
        # Get best (max CTR) and worst (min CTR) headlines
        best_idx = group['ctr'].idxmax()
        worst_idx = group['ctr'].idxmin()
        
        # Skip if same headline (tie in CTR)
        if best_idx == worst_idx:
            continue
        
        best_headline = group.loc[best_idx, 'headline']
        worst_headline = group.loc[worst_idx, 'headline']
        best_ctr = group.loc[best_idx, 'ctr']
        worst_ctr = group.loc[worst_idx, 'ctr']
        
        # Skip if either is empty/NaN
        if pd.isna(best_headline) or pd.isna(worst_headline):
            continue
        if str(best_headline).strip() == '' or str(worst_headline).strip() == '':
            continue
            
        pairs.append((str(best_headline), str(worst_headline), test_id, best_ctr, worst_ctr))
    
    return pairs


# ============================================================================
# Sparsity Analysis
# ============================================================================

def analyze_sparsity(X, name="Dataset"):
    """Analyze sparsity statistics of the sparse matrix."""
    print(f"\n{'='*60}")
    print(f"SPARSITY ANALYSIS: {name}")
    print(f"{'='*60}")
    
    n_rows, n_cols = X.shape
    total_elements = n_rows * n_cols
    nnz = X.nnz
    n_zeros = total_elements - nnz
    
    print(f"\nMatrix Shape: {n_rows:,} rows × {n_cols:,} columns")
    print(f"Total elements: {total_elements:,}")
    print(f"Non-zero elements: {nnz:,}")
    print(f"Zero elements: {n_zeros:,}")
    print(f"Overall sparsity: {100 * n_zeros / total_elements:.4f}%")
    print(f"Density: {100 * nnz / total_elements:.4f}%")
    
    # Per-row statistics
    nnz_per_row = np.diff(X.indptr)  # Number of non-zeros per row
    zeros_per_row = n_cols - nnz_per_row
    zeros_pct_per_row = 100 * zeros_per_row / n_cols
    
    print(f"\nPer-Row Statistics:")
    print(f"  Avg non-zeros per row: {np.mean(nnz_per_row):.2f} ({100*np.mean(nnz_per_row)/n_cols:.4f}%)")
    print(f"  Avg zeros per row: {np.mean(zeros_per_row):.2f} ({np.mean(zeros_pct_per_row):.4f}%)")
    print(f"  Min non-zeros: {np.min(nnz_per_row)}, Max non-zeros: {np.max(nnz_per_row)}")
    print(f"  Std non-zeros per row: {np.std(nnz_per_row):.2f}")
    
    # Value statistics (non-zero values only)
    data = X.data
    print(f"\nValue Statistics (non-zero values):")
    print(f"  Min: {np.min(data):.6f}")
    print(f"  Max: {np.max(data):.6f}")
    print(f"  Mean: {np.mean(data):.6f}")
    print(f"  Std: {np.std(data):.6f}")
    print(f"  Median: {np.median(data):.6f}")
    
    # Note: these are DELTA vectors, so values can be negative!
    positive_count = np.sum(data > 0)
    negative_count = np.sum(data < 0)
    print(f"\n  Positive values: {positive_count:,} ({100*positive_count/len(data):.2f}%)")
    print(f"  Negative values: {negative_count:,} ({100*negative_count/len(data):.2f}%)")
    
    # Percentiles
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    pct_values = np.percentile(data, percentiles)
    print(f"\n  Percentiles of non-zero values:")
    for p, v in zip(percentiles, pct_values):
        print(f"    {p}th: {v:.6f}")
    
    # Dead neurons (columns with all zeros)
    # For CSR matrix, we need to check which columns have no entries
    col_nnz = np.diff(X.tocsc().indptr)  # Convert to CSC for efficient column access
    dead_neurons = np.sum(col_nnz == 0)
    print(f"\nDead Neurons (columns with all zeros):")
    print(f"  Dead neurons: {dead_neurons:,} / {n_cols:,} ({100 * dead_neurons / n_cols:.2f}%)")
    print(f"  Active neurons: {n_cols - dead_neurons:,} ({100 * (n_cols - dead_neurons) / n_cols:.2f}%)")
    
    return {
        'shape': X.shape,
        'sparsity': n_zeros / total_elements,
        'avg_zeros_per_row_pct': np.mean(zeros_pct_per_row),
        'min_val': np.min(data),
        'max_val': np.max(data),
        'mean_val': np.mean(data),
        'dead_neurons': dead_neurons,
        'dead_neurons_pct': 100 * dead_neurons / n_cols,
    }


# ============================================================================
# Scaling Analysis
# ============================================================================

def analyze_scaling(X, name="Dataset"):
    """Analyze effects of StandardScaler and MaxAbsScaler."""
    print(f"\n{'='*60}")
    print(f"SCALING ANALYSIS: {name}")
    print(f"{'='*60}")
    
    # Original statistics
    print(f"\n--- Original Data ---")
    print(f"Non-zero value range: [{X.data.min():.6f}, {X.data.max():.6f}]")
    print(f"Non-zero mean: {X.data.mean():.6f}")
    print(f"Non-zero std: {X.data.std():.6f}")
    
    # StandardScaler (with_mean=False for sparse)
    print(f"\n--- After StandardScaler (with_mean=False) ---")
    scaler_std = StandardScaler(with_mean=False)
    X_std = scaler_std.fit_transform(X)
    print(f"Non-zero value range: [{X_std.data.min():.6f}, {X_std.data.max():.6f}]")
    print(f"Non-zero mean: {X_std.data.mean():.6f}")
    print(f"Non-zero std: {X_std.data.std():.6f}")
    print(f"Scale factors - min: {scaler_std.scale_.min():.6f}, max: {scaler_std.scale_.max():.6f}, mean: {scaler_std.scale_.mean():.6f}")
    
    # How many features have scale close to 0 (rarely activated)?
    small_scale = np.sum(scaler_std.scale_ < 0.01)
    print(f"Features with scale < 0.01 (rarely activated): {small_scale} ({100*small_scale/len(scaler_std.scale_):.2f}%)")
    
    # MaxAbsScaler
    print(f"\n--- After MaxAbsScaler ---")
    scaler_max = MaxAbsScaler()
    X_max = scaler_max.fit_transform(X)
    print(f"Non-zero value range: [{X_max.data.min():.6f}, {X_max.data.max():.6f}]")
    print(f"Non-zero mean: {X_max.data.mean():.6f}")
    print(f"Non-zero std: {X_max.data.std():.6f}")
    print(f"Max absolute values per feature - min: {scaler_max.max_abs_.min():.6f}, max: {scaler_max.max_abs_.max():.6f}")
    
    # Features that are never activated
    never_activated = np.sum(scaler_max.max_abs_ == 0)
    print(f"Features never activated: {never_activated} ({100*never_activated/len(scaler_max.max_abs_):.2f}%)")
    
    return X_std, X_max, scaler_std, scaler_max


# ============================================================================
# Feature Distribution Histograms
# ============================================================================

def plot_feature_histograms(X, X_std, X_max, feature_indices, output_path="feature_histograms.png"):
    """
    Plot 3x3 histogram grid: rows = features, columns = original/StandardScaler/MaxAbsScaler
    
    Args:
        X: Original sparse matrix
        X_std: StandardScaler transformed matrix
        X_max: MaxAbsScaler transformed matrix
        feature_indices: List of 3 feature indices to plot
        output_path: Path to save the figure
    """
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    
    titles = ["Original", "StandardScaler", "MaxAbsScaler"]
    matrices = [X, X_std, X_max]
    
    for row, feat_idx in enumerate(feature_indices):
        for col, (title, matrix) in enumerate(zip(titles, matrices)):
            ax = axes[row, col]
            
            # Get feature column values (non-zero only for histogram)
            col_data = matrix[:, feat_idx].toarray().flatten()
            nonzero_data = col_data[col_data != 0]
            
            if len(nonzero_data) > 0:
                # Use automatic binning
                ax.hist(nonzero_data, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
                ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
                
                # Stats annotation
                stats_text = f"n={len(nonzero_data)}\nμ={nonzero_data.mean():.3f}\nσ={nonzero_data.std():.3f}"
                ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', horizontalalignment='right',
                       fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                ax.text(0.5, 0.5, "No non-zero values", transform=ax.transAxes,
                       ha='center', va='center', fontsize=10)
            
            # Labels
            if row == 0:
                ax.set_title(title, fontsize=12, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f"Feature {feat_idx}", fontsize=10, fontweight='bold')
            if row == 2:
                ax.set_xlabel("Value", fontsize=10)
    
    plt.suptitle("Feature Value Distributions Across Scaling Methods", fontsize=14, fontweight='bold')
    plt.tight_layout()
    # plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nHistogram figure saved to: {output_path}")


# ============================================================================
# Top Activating Examples
# ============================================================================

def get_top_activating_examples(X, pairs, feature_idx, n_examples=3, direction="positive"):
    """
    Get the top n pairs with highest (or lowest) activation for a given feature.
    
    Since vectors are DELTAS (best-worst, worst-best):
    - High positive value in feature = "best headline activates this feature MORE than worst"
    - High negative value = "best headline activates this feature LESS than worst"
    
    Args:
        X: Sparse matrix of delta vectors
        pairs: List of (best_headline, worst_headline, test_id, best_ctr, worst_ctr) tuples
        feature_idx: Feature index to analyze
        n_examples: Number of examples to return
        direction: "positive" for highest values, "negative" for lowest values
    
    Returns:
        List of result dicts with pair info and activations
    """
    # Get the feature column as dense array
    feature_col = X[:, feature_idx].toarray().flatten()
    
    # Rows are: [best-worst_0, worst-best_0, best-worst_1, worst-best_1, ...]
    # We want the best-worst rows (even indices) for cleaner interpretation
    # For positive direction: sort best-worst rows by descending value
    # For negative direction: sort by ascending value
    
    # Extract just the best-worst rows (even indices)
    best_worst_indices = np.arange(0, len(feature_col), 2)
    best_worst_values = feature_col[best_worst_indices]
    
    if direction == "positive":
        # Highest positive deltas (best >> worst for this feature)
        sorted_indices = np.argsort(best_worst_values)[::-1]
    else:
        # Lowest (most negative) deltas (worst >> best for this feature)
        sorted_indices = np.argsort(best_worst_values)
    
    results = []
    for i in range(min(n_examples, len(sorted_indices))):
        pair_idx = sorted_indices[i]
        delta_value = best_worst_values[pair_idx]
        
        if pair_idx >= len(pairs):
            continue
            
        best_headline, worst_headline, test_id, best_ctr, worst_ctr = pairs[pair_idx]
        
        results.append({
            'pair_idx': pair_idx,
            'test_id': test_id,
            'delta_value': delta_value,  # best - worst for this feature
            'best_headline': best_headline,
            'worst_headline': worst_headline,
            'best_ctr': best_ctr,
            'worst_ctr': worst_ctr,
        })
    
    return results


def print_top_activating_examples(X, pairs, features, feature_labels, n_examples=3, direction="positive"):
    """Print top activating examples for a list of features."""
    dir_label = "HIGHEST" if direction == "positive" else "LOWEST"
    print(f"\n{'='*70}")
    print(f"TOP ACTIVATING EXAMPLES ({dir_label} delta values)")
    print(f"{'='*70}")
    print(f"\nNote: Delta = (best headline) - (worst headline) activation")
    print(f"      Positive delta = feature activates MORE in winning headline")
    print(f"      Negative delta = feature activates LESS in winning headline")
    
    for feat_idx in features:
        label = feature_labels.get(feat_idx, "")
        print(f"\n{'─'*70}")
        print(f"Feature {feat_idx}: {label}")
        print(f"{'─'*70}")
        
        results = get_top_activating_examples(X, pairs, feat_idx, n_examples, direction)
        
        if not results:
            print("  No data found for this feature.")
            continue
        
        for i, r in enumerate(results, 1):
            print(f"\n  Example {i} (Test ID: {r['test_id']})")
            print(f"  Delta (best-worst): {r['delta_value']:.4f}")
            print(f"  ★ BEST  (CTR {r['best_ctr']:.4f}): {r['best_headline'][:100]}{'...' if len(r['best_headline']) > 100 else ''}")
            print(f"    WORST (CTR {r['worst_ctr']:.4f}): {r['worst_headline'][:100]}{'...' if len(r['worst_headline']) > 100 else ''}")


# ============================================================================
# Feature Activation Distribution
# ============================================================================

def analyze_feature_activations(X, features, feature_labels, scaler_std=None, scaler_max=None):
    """Analyze activation distribution for specific features."""
    print(f"\n{'='*60}")
    print(f"FEATURE ACTIVATION DISTRIBUTION")
    print(f"{'='*60}")
    print(f"\nNote: Values are DELTAS. Non-zero means the feature activated")
    print(f"      differently between best and worst headline.")
    
    for feat_idx in features:
        label = feature_labels.get(feat_idx, "")
        col = X[:, feat_idx].toarray().flatten()
        
        nnz = np.sum(col != 0)
        nnz_pct = 100 * nnz / len(col)
        
        print(f"\nFeature {feat_idx}: {label}")
        print(f"  Non-zero deltas: {nnz:,} / {len(col):,} ({nnz_pct:.2f}%)")
        
        if nnz > 0:
            nonzero_vals = col[col != 0]
            positive = np.sum(nonzero_vals > 0)
            negative = np.sum(nonzero_vals < 0)
            print(f"  Positive deltas: {positive} ({100*positive/nnz:.1f}%), Negative: {negative} ({100*negative/nnz:.1f}%)")
            print(f"  Raw range: [{nonzero_vals.min():.4f}, {nonzero_vals.max():.4f}], Mean: {nonzero_vals.mean():.4f}")
            
            # Show scaled ranges
            if scaler_std is not None:
                scale = scaler_std.scale_[feat_idx]
                std_min = nonzero_vals.min() / scale
                std_max = nonzero_vals.max() / scale
                print(f"  StandardScaler range: [{std_min:.4f}, {std_max:.4f}] (scale={scale:.4f})")
            
            if scaler_max is not None:
                max_abs = scaler_max.max_abs_[feat_idx]
                if max_abs > 0:
                    maxabs_min = nonzero_vals.min() / max_abs
                    maxabs_max = nonzero_vals.max() / max_abs
                    print(f"  MaxAbsScaler range: [{maxabs_min:.4f}, {maxabs_max:.4f}] (max_abs={max_abs:.4f})")


# ============================================================================
# Main
# ============================================================================

def main():
    # Load data
    print("Loading SAE vectors and CSV data...")
    X, y = load_sae_vectors("confirmatory")
    df = load_csv_data("confirmatory")
    
    print(f"Loaded {X.shape[0]} delta vectors with {X.shape[1]} features")
    print(f"CSV has {len(df)} rows")
    
    # Get pairs using same logic as extraction script
    pairs = get_best_worst_pairs(df)
    print(f"Reconstructed {len(pairs)} headline pairs from CSV")
    
    # Verify alignment: should have 2 rows per pair
    expected_rows = len(pairs) * 2
    actual_rows = X.shape[0]
    print(f"Expected rows (2 per pair): {expected_rows}, Actual: {actual_rows}")
    
    if expected_rows != actual_rows:
        print("WARNING: Row count mismatch! There may be minibatch boundary issues.")
        print("         Proceeding with min(expected, actual) pairs...")
        # Truncate pairs if needed
        max_pairs = actual_rows // 2
        pairs = pairs[:max_pairs]
    
    # 1. Sparsity Analysis
    stats = analyze_sparsity(X, "Confirmatory SAE Delta Vectors")
    
    # 2. Scaling Analysis
    X_std, X_max, scaler_std, scaler_max = analyze_scaling(X, "Confirmatory SAE Delta Vectors")
    
    # # 3. Feature Distribution Histograms (3x3 grid)
    # histogram_features = [11540, 3464, 11052]
    # plot_feature_histograms(X, X_std, X_max, histogram_features, output_path="feature_histograms.png")
    
    # # 4. Feature Activation Distribution (for the interesting features)
    # all_features = TOP_FEATURES + BOTTOM_FEATURES
    # analyze_feature_activations(X, all_features, FEATURE_LABELS, scaler_std, scaler_max)
    
    # # 5. Top Activating Examples
    # # For TOP features (positive log reg coefficients): show pairs where this feature
    # # has highest positive delta (best >> worst)
    # print("\n" + "="*70)
    # print("TOP 10 FEATURES (positive log reg coefficients)")
    # print("="*70)
    # print_top_activating_examples(X, pairs, TOP_FEATURES, FEATURE_LABELS, 
    #                               n_examples=3, direction="positive")
    
    # # For BOTTOM features (negative log reg coefficients): show pairs where this feature
    # # has most negative delta (worst >> best, or best << worst)
    # print("\n" + "="*70)
    # print("BOTTOM 10 FEATURES (negative log reg coefficients)")  
    # print("="*70)
    # print_top_activating_examples(X, pairs, BOTTOM_FEATURES, FEATURE_LABELS,
    #                               n_examples=3, direction="negative")


    tree_features = [11052, 538, 12614, 613, 726, 1847, 2727, 605, 1389, 5439, 1531]
    additional_features = [1234, 5852, 136, 4127]
    combined_features = tree_features + additional_features

    print_top_activating_examples(X, pairs, combined_features, FEATURE_LABELS,
                                  n_examples=5, direction="positive")


if __name__ == "__main__":
    main()
