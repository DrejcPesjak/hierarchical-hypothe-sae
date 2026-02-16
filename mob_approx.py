# %% import libraries
import numpy as np
import json
from pathlib import Path
from scipy.sparse import csr_matrix, vstack
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, classification_report
import matplotlib.pyplot as plt

# %% configuration
d_sae = 16384
N = None                    # set to int to limit batches (for debugging)
OUTPUT_DIR = Path("tree_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# %% load data
print("Loading data...")
batch_dir = Path("./data/sae_vectors")
sparse_files = sorted(batch_dir.glob("confirmatory_preprocessed_sae_vectors_sparse_minibatch_*.npz"))
meta_files = sorted(batch_dir.glob("confirmatory_preprocessed_sae_vectors_1meta_minibatch_*.npz"))
sparse_files = sparse_files[:N] if N is not None else sparse_files
meta_files = meta_files[:N] if N is not None else meta_files

# Load sparse vectors [context | diff]  (each row is 2*d_sae wide)
Xs = []
for f in sparse_files:
    z = np.load(f)
    Xs.append(csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"])))

X_full = vstack(Xs, format="csr")

# Split into context and diff halves
X_context = X_full[:, :d_sae]
X_diff = X_full[:, d_sae:]
del X_full

# Load enriched metadata (1meta files with target_logit_diff + weight_sqrt_min_impr)
test_ids_list = []
target_list, weight_list = [], []

for f in meta_files:
    m = np.load(f, allow_pickle=True)
    test_ids_list.append(m["test_id"])
    target_list.append(m["target_logit_diff"])
    weight_list.append(m["weight_sqrt_min_impr"])

test_ids = np.concatenate(test_ids_list)
y_target = np.concatenate(target_list)
sample_weight = np.concatenate(weight_list)

print(f"Context:  {X_context.shape}")
print(f"Diff:     {X_diff.shape}")
print(f"Samples:  {len(y_target)}")
print(f"Unique test_ids: {len(np.unique(test_ids))}")

# %% data stats
print(f"\ny_target (signed logit diff) stats:")
print(f"  Range: [{y_target.min():.4f}, {y_target.max():.4f}]")
print(f"  Mean:  {y_target.mean():.4f}")
print(f"  Std:   {y_target.std():.4f}")
print(f"  Median: {np.median(y_target):.4f}")

print(f"\nsample_weight (sqrt min impressions) stats:")
print(f"  Range: [{sample_weight.min():.2f}, {sample_weight.max():.2f}]")
print(f"  Mean:  {sample_weight.mean():.2f}")
print(f"  Median: {np.median(sample_weight):.2f}")

# %% load feature names
print("\nLoading feature names...")
with open("data/gemmascope_explanations/ex/gemma3_4b_it_layer22_16k_explanations.json") as f:
    explanations_list = json.load(f)

explanation_map = {int(item["index"]): item["description"] for item in explanations_list}

def get_feature_name(sae_idx, prefix=""):
    desc = explanation_map.get(sae_idx, "")
    if desc:
        return f"{prefix}f{sae_idx}: {desc}"
    return f"{prefix}f{sae_idx}"

context_feature_names = [get_feature_name(i, prefix="ctx_") for i in range(d_sae)]
diff_feature_names = [get_feature_name(i, prefix="Δ_") for i in range(d_sae)]

# %% train/test split (by test_id)
print("\nSplitting data by test_id...")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X_diff, y_target, groups=test_ids))

X_diff_train, X_diff_test = X_diff[train_idx], X_diff[test_idx]
X_ctx_train, X_ctx_test = X_context[train_idx], X_context[test_idx]
y_train, y_test = y_target[train_idx], y_target[test_idx]
w_train, w_test = sample_weight[train_idx], sample_weight[test_idx]

print(f"Train: {len(train_idx)} samples, Test: {len(test_idx)} samples")
assert len(set(test_ids[train_idx]) & set(test_ids[test_idx])) == 0, "Data leakage!"

# %% standardize features
scaler_diff = StandardScaler(with_mean=False)
X_diff_train = scaler_diff.fit_transform(X_diff_train)
X_diff_test = scaler_diff.transform(X_diff_test)

scaler_ctx = StandardScaler(with_mean=False)
X_ctx_train = scaler_ctx.fit_transform(X_ctx_train)
X_ctx_test = scaler_ctx.transform(X_ctx_test)

# %% Step 1: Global effect (Lasso)
# ============================================================================
# STEP 1: Global Effect — Lasso on X_diff → y_target
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1: GLOBAL EFFECT — Lasso on X_diff → y_target")
print("=" * 70)

lasso = Lasso(alpha=0.01, max_iter=5000, random_state=42)
lasso.fit(X_diff_train, y_train, sample_weight=w_train)

y_pred_global_train = lasso.predict(X_diff_train)
y_pred_global_test = lasso.predict(X_diff_test)

# Evaluate
print(f"\nLasso results:")
print(f"  Non-zero coefficients: {np.sum(lasso.coef_ != 0)} / {len(lasso.coef_)}")
print(f"  Train R²: {r2_score(y_train, y_pred_global_train):.4f}")
print(f"  Test  R²: {r2_score(y_test, y_pred_global_test):.4f}")
print(f"  Train MAE: {mean_absolute_error(y_train, y_pred_global_train):.4f}")
print(f"  Test  MAE: {mean_absolute_error(y_test, y_pred_global_test):.4f}")
print(f"  Train RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_global_train)):.4f}")
print(f"  Test  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_global_test)):.4f}")

# Top Lasso features
nonzero_mask = lasso.coef_ != 0
nonzero_idx = np.where(nonzero_mask)[0]
nonzero_coefs = lasso.coef_[nonzero_idx]
sorted_order = np.argsort(np.abs(nonzero_coefs))[::-1]

print(f"\nTop 20 Lasso features (by |coef|):")
for rank, i in enumerate(sorted_order[:20], 1):
    idx = nonzero_idx[i]
    coef = nonzero_coefs[i]
    name = diff_feature_names[idx]
    print(f"  {rank:2d}. [{coef:+.6f}] {name}")

# %% Step 1: Residuals
resid_train = y_train - y_pred_global_train
resid_test = y_test - y_pred_global_test

print(f"\nResiduals (train):")
print(f"  Mean:  {resid_train.mean():.6f}")
print(f"  Std:   {resid_train.std():.4f}")
print(f"  Range: [{resid_train.min():.4f}, {resid_train.max():.4f}]")

# Plot: y_target vs predictions & residual distribution
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(y_pred_global_test, y_test, alpha=0.1, s=2, rasterized=True)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=1)
axes[0].set_xlabel("Predicted (Lasso)")
axes[0].set_ylabel("Actual y_target")
axes[0].set_title("Step 1: Global Effect — Predicted vs Actual (test)")

axes[1].hist(resid_train, bins=80, alpha=0.7, edgecolor='black', lw=0.3)
axes[1].axvline(0, color='red', ls='--')
axes[1].set_xlabel("Residual")
axes[1].set_ylabel("Count")
axes[1].set_title("Residual Distribution (train)")

# Lasso coefficient magnitudes (top 30)
top_n = min(30, len(nonzero_idx))
top_order = sorted_order[:top_n]
top_names = [diff_feature_names[nonzero_idx[i]][:30] for i in top_order]
top_coefs = [nonzero_coefs[i] for i in top_order]
colors = ['tab:blue' if c > 0 else 'tab:red' for c in top_coefs]
axes[2].barh(range(top_n), top_coefs, color=colors)
axes[2].set_yticks(range(top_n))
axes[2].set_yticklabels(top_names, fontsize=6)
axes[2].invert_yaxis()
axes[2].set_xlabel("Lasso Coefficient")
axes[2].set_title(f"Top {top_n} Lasso Coefficients")

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "causal_step1_global_effect.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR / 'causal_step1_global_effect.png'}")
plt.show()

# %% Step 2: Heterogeneity tree
# ============================================================================
# STEP 2: Heterogeneity — Decision Tree on X_context → residuals
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: HETEROGENEITY — DecisionTree on X_context → residuals")
print("=" * 70)

hetero_tree = DecisionTreeRegressor(
    max_depth=4,
    min_samples_leaf=200,
    min_impurity_decrease=0.001,
    random_state=42,
)
hetero_tree.fit(X_ctx_train, resid_train, sample_weight=w_train)

resid_pred_train = hetero_tree.predict(X_ctx_train)
resid_pred_test = hetero_tree.predict(X_ctx_test)

print(f"\nHeterogeneity tree:")
print(f"  Depth: {hetero_tree.get_depth()}, Leaves: {hetero_tree.get_n_leaves()}")
print(f"  Train R² (residuals): {r2_score(resid_train, resid_pred_train):.4f}")
print(f"  Test  R² (residuals): {r2_score(resid_test, resid_pred_test):.4f}")

# Print tree as text
tree_text = export_text(hetero_tree, feature_names=context_feature_names, max_depth=4)
print(f"\nHeterogeneity tree structure:")
print(tree_text)

# Plot heterogeneity tree
fig, ax = plt.subplots(figsize=(40, 16), dpi=150)
plot_tree(
    hetero_tree,
    feature_names=context_feature_names,
    filled=True,
    rounded=True,
    fontsize=7,
    ax=ax,
)
ax.set_title("Step 2: Heterogeneity Tree (X_context → residuals)", fontsize=18)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "causal_step2_heterogeneity_tree.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR / 'causal_step2_heterogeneity_tree.png'}")
plt.show()

# %% Step 3: Segmentation
# ============================================================================
# STEP 3: Segmentation — assign samples to leaf nodes
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: SEGMENTATION — assign samples to leaf nodes")
print("=" * 70)

# apply() returns the leaf node id for each sample
leaf_ids_train = hetero_tree.apply(X_ctx_train)
leaf_ids_test = hetero_tree.apply(X_ctx_test)

unique_leaves = np.unique(leaf_ids_train)
n_segments = len(unique_leaves)
# Remap to 0..n_segments-1 for nicer labels
leaf_to_segment = {leaf: seg for seg, leaf in enumerate(unique_leaves)}

segment_train = np.array([leaf_to_segment[l] for l in leaf_ids_train])
segment_test = np.array([leaf_to_segment.get(l, -1) for l in leaf_ids_test])

print(f"\nSegments found: {n_segments}")
print(f"\n{'Segment':<10} {'Train N':<10} {'Test N':<10} {'Mean resid':<12} {'Mean y_target':<14} {'Std y_target':<12}")
print("-" * 70)
for seg in range(n_segments):
    train_mask = segment_train == seg
    test_mask = segment_test == seg
    n_train = train_mask.sum()
    n_test = test_mask.sum()
    mean_r = resid_train[train_mask].mean() if n_train > 0 else float('nan')
    mean_y = y_train[train_mask].mean() if n_train > 0 else float('nan')
    std_y = y_train[train_mask].std() if n_train > 0 else float('nan')
    print(f"  {seg:<8} {n_train:<10} {n_test:<10} {mean_r:<12.4f} {mean_y:<14.4f} {std_y:<12.4f}")

# Distinctive context features per segment
print("\nDistinctive context features per segment (top 5):")
global_ctx_mean = np.asarray(X_ctx_train.mean(axis=0)).flatten()
for seg in range(n_segments):
    mask = segment_train == seg
    seg_mean = np.asarray(X_ctx_train[mask].mean(axis=0)).flatten()
    diff = seg_mean - global_ctx_mean
    top5 = np.argsort(diff)[::-1][:5]
    print(f"\n  Segment {seg} ({mask.sum()} samples):")
    for i in top5:
        print(f"    [{diff[i]:+.4f}] {context_feature_names[i][:60]}")

# Plot segment distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Segment sizes
seg_sizes = [np.sum(segment_train == s) for s in range(n_segments)]
axes[0].bar(range(n_segments), seg_sizes, color='steelblue', edgecolor='black', lw=0.5)
axes[0].set_xlabel("Segment")
axes[0].set_ylabel("Count")
axes[0].set_title("Segment Sizes (train)")
for i, v in enumerate(seg_sizes):
    axes[0].text(i, v + 10, str(v), ha='center', fontsize=8)

# Mean y_target per segment
seg_means = [y_train[segment_train == s].mean() for s in range(n_segments)]
seg_stds = [y_train[segment_train == s].std() for s in range(n_segments)]
colors = ['tab:green' if m > 0 else 'tab:red' for m in seg_means]
axes[1].bar(range(n_segments), seg_means, yerr=seg_stds, color=colors,
            edgecolor='black', lw=0.5, capsize=3, alpha=0.8)
axes[1].axhline(0, color='black', ls='--', lw=0.5)
axes[1].set_xlabel("Segment")
axes[1].set_ylabel("Mean y_target")
axes[1].set_title("Mean y_target per Segment (train)")

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "causal_step3_segments.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: {OUTPUT_DIR / 'causal_step3_segments.png'}")
plt.show()

# %% Step 4: Per-segment explanation trees
# ============================================================================
# STEP 4: Explanation — per-segment trees on X_diff → y_target
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: EXPLANATION — per-segment trees on X_diff → y_target")
print("=" * 70)

segment_trees = {}

# Figure layout: one column per segment, up to ~4 cols
n_cols = min(n_segments, 4)
n_rows = (n_segments + n_cols - 1) // n_cols

for seg in range(n_segments):
    train_mask = segment_train == seg
    test_mask = segment_test == seg
    
    n_train_seg = train_mask.sum()
    n_test_seg = test_mask.sum()
    
    print(f"\n{'─'*50}")
    print(f"Segment {seg}: {n_train_seg} train / {n_test_seg} test samples")
    print(f"{'─'*50}")
    
    if n_train_seg < 50:
        print("  Skipping (too few samples)")
        continue
    
    X_seg_train = X_diff_train[train_mask]
    y_seg_train = y_train[train_mask]
    w_seg_train = w_train[train_mask]
    X_seg_test = X_diff_test[test_mask]
    y_seg_test = y_test[test_mask]
    
    # Train a small explanation tree
    seg_tree = DecisionTreeRegressor(
        max_depth=3,
        min_samples_leaf=50,
        random_state=42,
    )
    seg_tree.fit(X_seg_train, y_seg_train, sample_weight=w_seg_train)
    
    y_seg_pred_train = seg_tree.predict(X_seg_train)
    
    print(f"  Tree depth: {seg_tree.get_depth()}, leaves: {seg_tree.get_n_leaves()}")
    print(f"  Train R²: {r2_score(y_seg_train, y_seg_pred_train):.4f}")
    if n_test_seg > 10:
        y_seg_pred_test = seg_tree.predict(X_seg_test)
        print(f"  Test  R²: {r2_score(y_seg_test, y_seg_pred_test):.4f}")
    
    # Print tree text
    tree_text = export_text(seg_tree, feature_names=diff_feature_names, max_depth=3)
    print(f"\n  Explanation tree:")
    for line in tree_text.split('\n'):
        print(f"    {line}")
    
    # Feature importance for this segment
    feat_imp = seg_tree.feature_importances_
    top_feats = np.argsort(feat_imp)[::-1][:10]
    print(f"\n  Top features by importance:")
    for rank, fi in enumerate(top_feats, 1):
        if feat_imp[fi] > 0:
            print(f"    {rank}. [{feat_imp[fi]:.4f}] {diff_feature_names[fi][:60]}")
    
    segment_trees[seg] = seg_tree

# %% Plot all segment explanation trees
print("\nPlotting per-segment explanation trees...")
for seg, seg_tree in segment_trees.items():
    n_in_seg = np.sum(segment_train == seg)
    fig, ax = plt.subplots(figsize=(24, 10), dpi=150)
    plot_tree(
        seg_tree,
        feature_names=diff_feature_names,
        filled=True,
        rounded=True,
        fontsize=7,
        ax=ax,
    )
    ax.set_title(f"Segment {seg} Explanation Tree ({n_in_seg} samples, X_diff → y_target)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"causal_step4_segment_{seg}_tree.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

print(f"Saved segment tree plots to {OUTPUT_DIR}/causal_step4_segment_*_tree.png")

# %% Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Total samples: {len(y_target)}")
print(f"  Lasso non-zero features: {np.sum(lasso.coef_ != 0)} / {d_sae}")
print(f"  Lasso test R²: {r2_score(y_test, y_pred_global_test):.4f}")
print(f"  Heterogeneity tree: depth={hetero_tree.get_depth()}, leaves={hetero_tree.get_n_leaves()}")
print(f"  Segments: {n_segments}")
for seg in range(n_segments):
    n = np.sum(segment_train == seg)
    mean_y = y_train[segment_train == seg].mean()
    if seg in segment_trees:
        st = segment_trees[seg]
        print(f"    Segment {seg}: {n:5d} samples, mean_y={mean_y:+.4f}, "
              f"explanation tree depth={st.get_depth()}, leaves={st.get_n_leaves()}")
    else:
        print(f"    Segment {seg}: {n:5d} samples, mean_y={mean_y:+.4f}, (skipped)")

