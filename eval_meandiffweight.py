# %% import libraries
"""
Evaluate Logistic Regression, Decision Tree, and XGBoost
on raw continuous SAE features (meandiffweight pipeline).

Uses the canonical train/test split from cluster_assignments.npz
so results are directly comparable with the binarised-feature models.
"""
import time
import numpy as np
import json
from pathlib import Path
from scipy.sparse import csr_matrix, vstack
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score, roc_auc_score,
)
from xgboost import XGBClassifier

# %% configuration
d_sae = 16384
N = None  # set to int to limit batches (debugging)

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

# %% train/test arrays
X_train = X_full[train_idx]
X_test = X_full[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
w_train, w_test = w[train_idx], w[test_idx]
print(f"X_train: {X_train.shape},  X_test: {X_test.shape}")

# %% standardize features
print("Standardizing features …")
scaler = StandardScaler(with_mean=False)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# %% helper: evaluate a model ------------------------------------------------
def evaluate(name, y_true, y_pred, y_prob, w):
    acc = accuracy_score(y_true, y_pred)
    w_acc = accuracy_score(y_true, y_pred, sample_weight=w)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob, sample_weight=w)
    except ValueError:
        auc = float("nan")

    print(f"\n--- {name} ---")
    print(f"Accuracy:          {acc:.4f}")
    print(f"Weighted Accuracy: {w_acc:.4f}")
    print(f"F1 (macro):        {f1:.4f}")
    print(f"AUC-ROC:           {auc:.4f}")
    print(classification_report(y_true, y_pred))

    return {
        "accuracy": float(acc),
        "weighted_accuracy": float(w_acc),
        "f1_macro": float(f1),
        "auc_roc": float(auc),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }


# %% 1. Logistic Regression --------------------------------------------------
print("\n" + "=" * 80)
print("1. Logistic Regression (SGD, log_loss, L2)")
print("=" * 80)

t0 = time.time()
logreg = SGDClassifier(
    loss="log_loss",
    penalty="l2",
    alpha=0.0001,
    max_iter=2000,
    tol=1e-3,
    random_state=42,
    n_jobs=-1,
)
logreg.fit(X_train, y_train, sample_weight=w_train)
print(f"Training done in {time.time()-t0:.1f}s")

y_pred_lr = logreg.predict(X_test)
y_prob_lr = logreg.decision_function(X_test)   # log_loss → decision_function gives logits
res_logreg = evaluate("Logistic Regression", y_test, y_pred_lr, y_prob_lr, w_test)
res_logreg["settings"] = "SGDClassifier(log_loss, L2, alpha=1e-4, max_iter=2000)"


# %% 2. Decision Tree --------------------------------------------------------
print("\n" + "=" * 80)
print("2. Decision Tree")
print("=" * 80)

t0 = time.time()
dt = DecisionTreeClassifier(
    max_depth=8,
    min_samples_leaf=20,
    random_state=42,
)
dt.fit(X_train, y_train, sample_weight=w_train)
print(f"Training done in {time.time()-t0:.1f}s")
print(f"Tree depth: {dt.get_depth()}, leaves: {dt.get_n_leaves()}")

y_pred_dt = dt.predict(X_test)
y_prob_dt = dt.predict_proba(X_test)[:, 1]
res_dt = evaluate("Decision Tree", y_test, y_pred_dt, y_prob_dt, w_test)
res_dt["settings"] = "DecisionTree(max_depth=8, min_samples_leaf=20)"
res_dt["tree_depth"] = int(dt.get_depth())
res_dt["n_leaves"] = int(dt.get_n_leaves())


# %% 3. XGBoost --------------------------------------------------------------
print("\n" + "=" * 80)
print("3. XGBoost")
print("=" * 80)

t0 = time.time()
xgb = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    n_jobs=-1,
    random_state=42,
    eval_metric="logloss",
    verbosity=1,
)
xgb.fit(X_train, y_train, sample_weight=w_train)
print(f"Training done in {time.time()-t0:.1f}s")

y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
res_xgb = evaluate("XGBoost", y_test, y_pred_xgb, y_prob_xgb, w_test)
res_xgb["settings"] = "XGBClassifier(n_est=100, depth=6, lr=0.1)"


# %% save results ------------------------------------------------------------
results = {
    "logreg": res_logreg,
    "decision_tree": res_dt,
    "xgboost": res_xgb,
    "data": "continuous 32768 (ctx+diff), StandardScaler(with_mean=False), w_diff weights",
    "split": "cluster_assignments.npz canonical split",
}

out_path = Path("tree_outputs/eval_meandiffweight_results.json")
out_path.write_text(json.dumps(results, indent=2))
print(f"\nSaved all results to {out_path}")

