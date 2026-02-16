# %% import libraries
"""
Train a global XGBoost and per-cluster XGBoost models, then compare performance.

Clustering is done on context features (first d_sae columns).
Prediction is done on diff features (second d_sae columns) with better_label as target
and weight_diff as sample weight.

Requires: run clustering_context.py first to produce tree_outputs/cluster_assignments.npz
"""
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix, vstack
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
import json

# %% configuration
d_sae = 16384
N = None  # set to int to limit batches (for debugging)

# %% load data (diff vectors + metadata)
print("Loading diff vectors and metadata...")
batch_dir = Path("./data/sae_vectors")
sparse_files = sorted(batch_dir.glob("confirmatory_preprocessed_sae_vectors_sparse_minibatch_*.npz"))
meta_files = sorted(batch_dir.glob("confirmatory_preprocessed_sae_vectors_meta_minibatch_*.npz"))
sparse_files = sparse_files[:N] if N is not None else sparse_files
meta_files = meta_files[:N] if N is not None else meta_files

# Load sparse vectors — take diff half (columns d_sae .. 2*d_sae)
Xs = []
for f in sparse_files:
    z = np.load(f)
    mat = csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    Xs.append(mat[:, d_sae:])  # diff half only

X_diff = vstack(Xs, format="csr")
print(f"Diff vectors: {X_diff.shape[0]} examples × {X_diff.shape[1]} features")

# Load metadata
y_list, w_list = [], []
for f in meta_files:
    m = np.load(f, allow_pickle=True)
    y_list.append(m["better_label"])
    w_list.append(m["weight_diff"])

y = np.concatenate(y_list)
w = np.concatenate(w_list)
print(f"Label distribution: 0={np.sum(y == 0)}, 1={np.sum(y == 1)}")

# %% load cluster assignments (from clustering_context.py)
print("Loading cluster assignments...")
ca = np.load("tree_outputs/cluster_assignments.npz")
labels = ca["labels"]
is_train = ca["is_train"]
train_idx = ca["train_idx"]
test_idx = ca["test_idx"]

assert len(labels) == X_diff.shape[0], f"Mismatch: {len(labels)} labels vs {X_diff.shape[0]} rows"

cluster_ids = sorted(set(labels))
n_clusters = len(cluster_ids)
print(f"Clusters: {n_clusters}, Train: {len(train_idx)}, Test: {len(test_idx)}")

# %% prepare train / test splits
X_train, X_test = X_diff[train_idx], X_diff[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
w_train, w_test = w[train_idx], w[test_idx]
labels_train = labels[train_idx]
labels_test = labels[test_idx]

# Standardize features (fit on train only)
scaler = StandardScaler(with_mean=False)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %% helper: train & evaluate XGBoost
def train_and_eval(X_tr, y_tr, w_tr, X_te, y_te, w_te, name="model"):
    """Train XGBoost and return metrics dict."""
    if len(np.unique(y_tr)) < 2:
        print(f"  [{name}] Skipped — only one class in training set")
        return None

    model = XGBClassifier(
        n_estimators=40,
        max_depth=6,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X_tr, y_tr, sample_weight=w_tr)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)

    acc = accuracy_score(y_te, y_pred)
    w_acc = accuracy_score(y_te, y_pred, sample_weight=w_te)
    f1 = f1_score(y_te, y_pred, average="binary", zero_division=0)

    try:
        auc = roc_auc_score(y_te, y_prob[:, 1], sample_weight=w_te)
    except ValueError:
        auc = float("nan")

    return {
        "name": name,
        "n_train": len(y_tr),
        "n_test": len(y_te),
        "accuracy": acc,
        "weighted_accuracy": w_acc,
        "f1": f1,
        "auc_roc": auc,
        "model": model,
    }

# %% train GLOBAL XGBoost (all data)
print("\n" + "=" * 80)
print("Training GLOBAL XGBoost on all diff features...")
print("=" * 80)
global_result = train_and_eval(X_train, y_train, w_train, X_test, y_test, w_test, name="GLOBAL")
print(f"  Accuracy:          {global_result['accuracy']:.4f}")
print(f"  Weighted Accuracy: {global_result['weighted_accuracy']:.4f}")
print(f"  F1:                {global_result['f1']:.4f}")
print(f"  AUC-ROC:           {global_result['auc_roc']:.4f}")
print(classification_report(y_test, global_result["model"].predict(X_test)))

# %% train PER-CLUSTER XGBoost
print("\n" + "=" * 80)
print("Training PER-CLUSTER XGBoost models...")
print("=" * 80)

cluster_results = {}
# also collect per-cluster predictions for combined evaluation
y_pred_combined = np.empty_like(y_test)
y_prob_combined = np.empty((len(y_test), 2), dtype=np.float64)

for c in cluster_ids:
    mask_tr = labels_train == c
    mask_te = labels_test == c

    n_tr = mask_tr.sum()
    n_te = mask_te.sum()
    print(f"\nCluster {c}: {n_tr} train, {n_te} test")

    if n_te == 0:
        print(f"  No test samples — skipping evaluation")
        continue

    res = train_and_eval(
        X_train[mask_tr], y_train[mask_tr], w_train[mask_tr],
        X_test[mask_te], y_test[mask_te], w_test[mask_te],
        name=f"Cluster_{c}",
    )

    if res is None:
        # Fallback: predict majority class
        maj = int(np.round(y_train[mask_tr].mean())) if n_tr > 0 else 0
        y_pred_combined[mask_te] = maj
        y_prob_combined[mask_te] = [1 - maj, maj]
        cluster_results[c] = {"name": f"Cluster_{c}", "n_train": int(n_tr), "n_test": int(n_te),
                               "accuracy": float("nan"), "note": "only one class"}
        continue

    cluster_results[c] = {k: v for k, v in res.items() if k != "model"}

    # Store per-cluster predictions in combined arrays
    y_pred_combined[mask_te] = res["model"].predict(X_test[mask_te])
    y_prob_combined[mask_te] = res["model"].predict_proba(X_test[mask_te])

    print(f"  Accuracy:          {res['accuracy']:.4f}")
    print(f"  Weighted Accuracy: {res['weighted_accuracy']:.4f}")
    print(f"  F1:                {res['f1']:.4f}")
    print(f"  AUC-ROC:           {res['auc_roc']:.4f}")

# %% combined per-cluster metrics (predict each test sample with its cluster model)
print("\n" + "=" * 80)
print("COMBINED per-cluster predictions (each test sample scored by its cluster model)")
print("=" * 80)
combined_acc = accuracy_score(y_test, y_pred_combined)
combined_wacc = accuracy_score(y_test, y_pred_combined, sample_weight=w_test)
combined_f1 = f1_score(y_test, y_pred_combined, average="binary", zero_division=0)
try:
    combined_auc = roc_auc_score(y_test, y_prob_combined[:, 1], sample_weight=w_test)
except ValueError:
    combined_auc = float("nan")

print(f"  Accuracy:          {combined_acc:.4f}")
print(f"  Weighted Accuracy: {combined_wacc:.4f}")
print(f"  F1:                {combined_f1:.4f}")
print(f"  AUC-ROC:           {combined_auc:.4f}")

# %% comparison summary
print("\n" + "=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)

header = f"{'Model':<20} {'N_train':>8} {'N_test':>8} {'Acc':>8} {'W_Acc':>8} {'F1':>8} {'AUC':>8}"
print(header)
print("-" * len(header))

# Global
print(f"{'GLOBAL':<20} {global_result['n_train']:>8} {global_result['n_test']:>8} "
      f"{global_result['accuracy']:>8.4f} {global_result['weighted_accuracy']:>8.4f} "
      f"{global_result['f1']:>8.4f} {global_result['auc_roc']:>8.4f}")

# Per-cluster
for c in cluster_ids:
    if c not in cluster_results:
        continue
    r = cluster_results[c]
    acc_str = f"{r['accuracy']:>8.4f}" if not (isinstance(r.get('accuracy'), float) and np.isnan(r['accuracy'])) else "     N/A"
    wacc_str = f"{r.get('weighted_accuracy', float('nan')):>8.4f}" if 'weighted_accuracy' in r and not np.isnan(r.get('weighted_accuracy', float('nan'))) else "     N/A"
    f1_str = f"{r.get('f1', float('nan')):>8.4f}" if 'f1' in r and not np.isnan(r.get('f1', float('nan'))) else "     N/A"
    auc_str = f"{r.get('auc_roc', float('nan')):>8.4f}" if 'auc_roc' in r and not np.isnan(r.get('auc_roc', float('nan'))) else "     N/A"
    print(f"{r['name']:<20} {r['n_train']:>8} {r['n_test']:>8} {acc_str} {wacc_str} {f1_str} {auc_str}")

# Combined per-cluster
print(f"{'COMBINED-CLUSTER':<20} {len(y_train):>8} {len(y_test):>8} "
      f"{combined_acc:>8.4f} {combined_wacc:>8.4f} "
      f"{combined_f1:>8.4f} {combined_auc:>8.4f}")

print()
print(f"Δ Accuracy (combined_cluster - global): {combined_acc - global_result['accuracy']:+.4f}")
print(f"Δ W_Acc    (combined_cluster - global): {combined_wacc - global_result['weighted_accuracy']:+.4f}")
print(f"Δ F1       (combined_cluster - global): {combined_f1 - global_result['f1']:+.4f}")
print(f"Δ AUC      (combined_cluster - global): {combined_auc - global_result['auc_roc']:+.4f}")

# %% save results to JSON
results_out = {
    "global": {k: v for k, v in global_result.items() if k != "model"},
    "per_cluster": {str(c): {k: v for k, v in r.items() if k != "model"} for c, r in cluster_results.items()},
    "combined_cluster": {
        "accuracy": combined_acc,
        "weighted_accuracy": combined_wacc,
        "f1": combined_f1,
        "auc_roc": combined_auc,
    },
}

out_path = Path("tree_outputs/xgboost_cluster_comparison.json")
out_path.write_text(json.dumps(results_out, indent=2, default=str))
print(f"\nSaved results to {out_path}")

