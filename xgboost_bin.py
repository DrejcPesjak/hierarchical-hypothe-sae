# %% import libraries
"""
XGBoost on binarised/trinarised SAE features.

Context features (first 16k):  raw → 0/1   (1 if above walk-right threshold)
Diff features    (second 16k): raw → -1/0/1 (walk-outward thresholds)

Thresholds are computed on RAW (unscaled) non-zero values using the
"walk" histogram method from feature_analysis.py.
They are cached to disk so subsequent runs skip the slow computation.
"""
import time
import numpy as np
import json
from pathlib import Path
from scipy.sparse import csr_matrix, vstack, csc_matrix
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score, roc_auc_score,
)

# %% configuration
d_sae = 16384
N = None  # set to int to limit batches (debugging)
THRESHOLD_FILE = Path("tree_outputs/walk_thresholds.npz")
CTX_BB = 4    # walk-right: need next bb bars all higher
DIFF_BB = 4   # walk-outward: need next bb bars all higher
CTX_BINS = 200
DIFF_BINS = 300

# %% threshold functions (same logic as feature_analysis.py) ----------------
def walk_right_threshold(values, n_bins=200, bb=4):
    """Context: walk from smallest non-zero value rightward.
    Returns a single positive threshold, or None."""
    if len(values) < 10:
        return None
    hist, edges = np.histogram(values, bins=n_bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    for i in range(len(hist) - bb):
        if all(hist[i + j] > hist[i] for j in range(1, bb + 1)):
            return centres[i]
    return None


def walk_outward_threshold(values, n_bins=300, bb=4):
    """Diff: walk right from 0, mirror to negative.
    Returns (neg_th, pos_th) or (None, None)."""
    if len(values) < 10:
        return None, None
    hist, edges = np.histogram(values, bins=n_bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    zero_idx = np.argmin(np.abs(centres))
    pos_th = None
    for i in range(zero_idx, len(hist) - bb):
        if all(hist[i + j] > hist[i] for j in range(1, bb + 1)):
            pos_th = centres[i]
            break
    if pos_th is None:
        return None, None
    return -pos_th, pos_th


# %% load data ---------------------------------------------------------------
print("Loading sparse vectors and metadata …")
batch_dir = Path("./data/sae_vectors")
sparse_files = sorted(batch_dir.glob(
    "confirmatory_preprocessed_sae_vectors_sparse_minibatch_*.npz"))
meta_files = sorted(batch_dir.glob(
    "confirmatory_preprocessed_sae_vectors_meta_minibatch_*.npz"))
sparse_files = sparse_files[:N] if N is not None else sparse_files
meta_files = meta_files[:N] if N is not None else meta_files

Xs = []
for f in sparse_files:
    z = np.load(f)
    mat = csr_matrix((z["data"], z["indices"], z["indptr"]),
                     shape=tuple(z["shape"]))
    Xs.append(mat)

X_full = vstack(Xs, format="csr")
n_samples, n_cols = X_full.shape
print(f"Full matrix: {n_samples} × {n_cols}")
assert n_cols == 2 * d_sae, f"Expected {2*d_sae} columns, got {n_cols}"

# Load metadata
y_list, w_list = [], []
for f in meta_files:
    m = np.load(f, allow_pickle=True)
    y_list.append(m["better_label"])
    w_list.append(m["weight_diff"])

y = np.concatenate(y_list)
w = np.concatenate(w_list)
print(f"Label distribution: 0={np.sum(y == 0)}, 1={np.sum(y == 1)}")

# %% load train/test split
print("Loading train/test split …")
ca = np.load("tree_outputs/cluster_assignments.npz")
train_idx = ca["train_idx"]
test_idx = ca["test_idx"]
print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

# %% compute or load thresholds ---------------------------------------------
if THRESHOLD_FILE.exists():
    print(f"Loading cached thresholds from {THRESHOLD_FILE} …")
    tz = np.load(THRESHOLD_FILE)
    ctx_thresholds = tz["ctx_thresholds"]       # shape (d_sae,), NaN = no threshold
    diff_neg_thresholds = tz["diff_neg_thresholds"]  # shape (d_sae,)
    diff_pos_thresholds = tz["diff_pos_thresholds"]  # shape (d_sae,)
    print("  Loaded.")
else:
    print(f"Computing walk thresholds for {d_sae} ctx + {d_sae} diff features …")
    # Work on TRAINING data only to avoid leaking test info into thresholds
    X_train_full = X_full[train_idx]

    # --- context thresholds (columns 0 .. d_sae-1) ---
    ctx_thresholds = np.full(d_sae, np.nan)
    print("  Context features …")
    X_ctx_csc = X_train_full[:, :d_sae].tocsc()
    for j in range(d_sae):
        col = X_ctx_csc[:, j].toarray().ravel()
        nz = col[col != 0]
        th = walk_right_threshold(nz, n_bins=CTX_BINS, bb=CTX_BB)
        if th is not None:
            ctx_thresholds[j] = th
        if (j + 1) % 2000 == 0:
            print(f"    {j+1}/{d_sae}")

    n_ctx_found = np.sum(~np.isnan(ctx_thresholds))
    print(f"  Context: {n_ctx_found}/{d_sae} thresholds found")

    # --- diff thresholds (columns d_sae .. 2*d_sae-1) ---
    diff_neg_thresholds = np.full(d_sae, np.nan)
    diff_pos_thresholds = np.full(d_sae, np.nan)
    print("  Diff features …")
    X_diff_csc = X_train_full[:, d_sae:].tocsc()
    for j in range(d_sae):
        col = X_diff_csc[:, j].toarray().ravel()
        nz = col[col != 0]
        neg_th, pos_th = walk_outward_threshold(nz, n_bins=DIFF_BINS, bb=DIFF_BB)
        if neg_th is not None:
            diff_neg_thresholds[j] = neg_th
            diff_pos_thresholds[j] = pos_th
        if (j + 1) % 2000 == 0:
            print(f"    {j+1}/{d_sae}")

    n_diff_found = np.sum(~np.isnan(diff_pos_thresholds))
    print(f"  Diff: {n_diff_found}/{d_sae} thresholds found")

    del X_train_full, X_ctx_csc, X_diff_csc

    # Save
    THRESHOLD_FILE.parent.mkdir(exist_ok=True)
    np.savez(THRESHOLD_FILE,
             ctx_thresholds=ctx_thresholds,
             diff_neg_thresholds=diff_neg_thresholds,
             diff_pos_thresholds=diff_pos_thresholds)
    print(f"  Saved thresholds to {THRESHOLD_FILE}")

# %% binarise / trinarise ---------------------------------------------------
print("Applying thresholds to full matrix …")
t0 = time.time()

# Context: 0 or 1
X_ctx = X_full[:, :d_sae]
X_ctx_bin = np.zeros((n_samples, d_sae), dtype=np.int8)
X_ctx_csc = X_ctx.tocsc()
for j in range(d_sae):
    th = ctx_thresholds[j]
    if np.isnan(th):
        continue  # leave as all-zeros (no threshold → no signal)
    col = X_ctx_csc[:, j].toarray().ravel()
    X_ctx_bin[:, j] = (col >= th).astype(np.int8)

# Diff: -1, 0, or 1
X_diff = X_full[:, d_sae:]
X_diff_tri = np.zeros((n_samples, d_sae), dtype=np.int8)
X_diff_csc = X_diff.tocsc()
for j in range(d_sae):
    neg_th = diff_neg_thresholds[j]
    pos_th = diff_pos_thresholds[j]
    if np.isnan(pos_th):
        continue
    col = X_diff_csc[:, j].toarray().ravel()
    X_diff_tri[:, j] = np.where(col >= pos_th, 1,
                        np.where(col <= neg_th, -1, 0)).astype(np.int8)

X_bin = np.hstack([X_ctx_bin, X_diff_tri])
print(f"Binarised matrix: {X_bin.shape}  ({time.time()-t0:.1f}s)")
print(f"  ctx 1s: {(X_ctx_bin == 1).sum():,}  /  diff ±1s: {(np.abs(X_diff_tri) == 1).sum():,}")

del X_ctx, X_diff, X_ctx_csc, X_diff_csc, X_ctx_bin, X_diff_tri

# %% train/test split
X_train = X_bin[train_idx]
X_test = X_bin[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
w_train, w_test = w[train_idx], w[test_idx]
print(f"X_train: {X_train.shape},  X_test: {X_test.shape}")

# %% build feature names
print("Loading feature names …")
with open("data/gemmascope_explanations/ex/gemma3_4b_it_layer22_16k_explanations.json") as f:
    explanations_list = json.load(f)

explanation_map = {int(item["index"]): item["description"] for item in explanations_list}

def sae_feature_name(prefix, sae_idx):
    desc = explanation_map.get(sae_idx, "")
    short = desc[:60] if desc else ""
    return f"{prefix}_{sae_idx}_{short}" if short else f"{prefix}_{sae_idx}"

feature_names = (
    [sae_feature_name("ctx", i) for i in range(d_sae)] +
    [sae_feature_name("diff", i) for i in range(d_sae)]
)

# %% train XGBoost -----------------------------------------------------------
print("\n" + "=" * 80)
print("Training XGBoost on binarised features …")
print("=" * 80)

start = time.time()
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    n_jobs=-1,
    random_state=42,
    eval_metric="logloss",
    colsample_bytree=0.7,
    verbosity=1,
)
# model = XGBClassifier(
#     n_estimators=1200,
#     max_depth=1,              # stumps = additive model
#     learning_rate=0.05,
#     subsample=0.8,
#     colsample_bytree=0.5,
#     min_child_weight=10,
#     reg_lambda=2.0,
#     n_jobs=-1,
#     random_state=42,
#     eval_metric="logloss",
#     verbosity=1,
# )
model.fit(X_train, y_train, sample_weight=w_train)
print(f"Training done in {time.time()-start:.1f}s")

# %% evaluate ----------------------------------------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)
w_acc = accuracy_score(y_test, y_pred, sample_weight=w_test)
f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)
try:
    auc = roc_auc_score(y_test, y_prob[:, 1], sample_weight=w_test)
except ValueError:
    auc = float("nan")

print(f"\nAccuracy:          {acc:.4f}")
print(f"Weighted Accuracy: {w_acc:.4f}")
print(f"F1:                {f1:.4f}")
print(f"AUC-ROC:           {auc:.4f}")
print()
print(classification_report(y_test, y_pred))

# %% feature importance -------------------------------------------------------
importances = model.feature_importances_
nonzero_idx = np.where(importances > 0)[0]
print(f"\nNon-zero feature importances: {len(nonzero_idx)} / {len(importances)}")

# Sort by importance descending
sorted_nz = nonzero_idx[np.argsort(importances[nonzero_idx])[::-1]]

print(f"\nTop 30 features by importance:")
print("-" * 100)
for rank, idx in enumerate(sorted_nz[:30]):
    imp = importances[idx]
    name = feature_names[idx]
    print(f"  {rank+1:3d}. [{idx:5d}] {imp:.6f}  {name}")

# %% save non-zero feature importances to file
out_path = Path("tree_outputs/feature_importances_xgb_bin444.txt")
with open(out_path, "w") as f:
    for idx in sorted_nz:
        f.write(f"{idx}: {importances[idx]}\n")
print(f"\nSaved {len(sorted_nz)} non-zero feature importances to {out_path}")

# %% save results JSON
results = {
    "accuracy": float(acc),
    "weighted_accuracy": float(w_acc),
    "f1": float(f1),
    "auc_roc": float(auc),
    "n_train": int(len(y_train)),
    "n_test": int(len(y_test)),
    "n_nonzero_features": int(len(nonzero_idx)),
    "n_ctx_thresholds": int(np.sum(~np.isnan(ctx_thresholds))),
    "n_diff_thresholds": int(np.sum(~np.isnan(diff_pos_thresholds))),
}
json_path = Path("tree_outputs/xgboost_bin_results444.json")
json_path.write_text(json.dumps(results, indent=2))
print(f"Saved metrics to {json_path}")

# %% dump tree text with stats
booster = model.get_booster()
booster.feature_names = feature_names
tree_dump = booster.get_dump(with_stats=True)
dump_path = Path("tree_outputs/xgboost_bin_tree_dump444.txt")
with open(dump_path, "w") as f:
    for i, tree_text in enumerate(tree_dump):
        f.write(f"booster[{i}]:\n{tree_text}\n")
print(f"Saved {len(tree_dump)} tree dumps to {dump_path}")
