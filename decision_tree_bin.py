# %% import libraries
"""
Decision Tree on binarised/trinarised SAE features.

Uses the same binarisation pipeline as xgboost_bin.py:
  Context features (first 16k):  raw → 0/1   (walk-right threshold)
  Diff features    (second 16k): raw → -1/0/1 (walk-outward thresholds)

Outputs:
  - tree_outputs/decision_tree_bin_results.json  (metrics)
  - tree_outputs/decision_tree_bin_dump.txt      (readable tree text)
"""
import time
import numpy as np
import json
from pathlib import Path
from scipy.sparse import csr_matrix, vstack
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score, roc_auc_score,
)

# %% configuration
d_sae = 16384
N = None  # set to int to limit batches (debugging)
THRESHOLD_FILE = Path("tree_outputs/walk_thresholds.npz")

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

# %% load canonical train/test split
print("Loading train/test split …")
ca = np.load("tree_outputs/cluster_assignments.npz")
train_idx = ca["train_idx"]
test_idx = ca["test_idx"]
print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

# %% load cached thresholds --------------------------------------------------
print(f"Loading cached thresholds from {THRESHOLD_FILE} …")
tz = np.load(THRESHOLD_FILE)
ctx_thresholds = tz["ctx_thresholds"]            # (d_sae,), NaN = no threshold
diff_neg_thresholds = tz["diff_neg_thresholds"]   # (d_sae,)
diff_pos_thresholds = tz["diff_pos_thresholds"]   # (d_sae,)

# %% binarise / trinarise ----------------------------------------------------
print("Applying thresholds to full matrix …")
t0 = time.time()

# Context: 0 or 1
X_ctx = X_full[:, :d_sae]
X_ctx_bin = np.zeros((n_samples, d_sae), dtype=np.int8)
X_ctx_csc = X_ctx.tocsc()
for j in range(d_sae):
    th = ctx_thresholds[j]
    if np.isnan(th):
        continue
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

del X_full, X_ctx, X_diff, X_ctx_csc, X_diff_csc, X_ctx_bin, X_diff_tri

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

# %% train Decision Tree -----------------------------------------------------
print("\n" + "=" * 80)
print("Training Decision Tree on binarised features …")
print("=" * 80)

start = time.time()
dt = DecisionTreeClassifier(
    max_depth=8,
    min_samples_leaf=20,
    random_state=42,
)
dt.fit(X_train, y_train, sample_weight=w_train)
print(f"Training done in {time.time()-start:.1f}s")
print(f"Tree depth: {dt.get_depth()}, leaves: {dt.get_n_leaves()}")

# %% evaluate ----------------------------------------------------------------
y_pred = dt.predict(X_test)
y_prob = dt.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)
w_acc = accuracy_score(y_test, y_pred, sample_weight=w_test)
f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
try:
    auc = roc_auc_score(y_test, y_prob[:, 1], sample_weight=w_test)
except ValueError:
    auc = float("nan")

print(f"\nAccuracy:          {acc:.4f}")
print(f"Weighted Accuracy: {w_acc:.4f}")
print(f"F1 (macro):        {f1:.4f}")
print(f"AUC-ROC:           {auc:.4f}")
print()
print(classification_report(y_test, y_pred))

# %% save results JSON -------------------------------------------------------
results = {
    "accuracy": float(acc),
    "weighted_accuracy": float(w_acc),
    "f1_macro": float(f1),
    "auc_roc": float(auc),
    "n_train": int(len(y_train)),
    "n_test": int(len(y_test)),
    "tree_depth": int(dt.get_depth()),
    "n_leaves": int(dt.get_n_leaves()),
    "settings": "DecisionTree(max_depth=8, min_samples_leaf=20)",
    "data": "binarised 32768 (ctx 0/1 + diff -1/0/1), w_diff weights",
    "split": "cluster_assignments.npz canonical split",
}

json_path = Path("tree_outputs/decision_tree_bin_results.json")
json_path.write_text(json.dumps(results, indent=2))
print(f"Saved metrics to {json_path}")

# %% dump readable tree text -------------------------------------------------
tree_text = export_text(dt, feature_names=feature_names, max_depth=20)
dump_path = Path("tree_outputs/decision_tree_bin_dump.txt")
dump_path.write_text(tree_text)
print(f"Saved tree dump ({len(tree_text)} chars) to {dump_path}")

