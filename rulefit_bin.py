# %% import libraries
"""
RuleFit on binarised/trinarised features (from xgboost_bin.py pipeline).

1. Load full sparse matrix
2. Select only features that XGBoost-bin found important
3. Apply walk thresholds (RAW scale) → ctx 0/1, diff -1/0/1
4. Free the full matrix
5. Train RuleFit on the small dense binarised matrix

pip install imodels
"""
import time
import numpy as np
import json
from pathlib import Path
from scipy.sparse import csr_matrix, vstack
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score, roc_auc_score,
)
from imodels import RuleFitClassifier

# %% configuration
d_sae = 16384
N = None
XGB_BIN_FEAT_FILE = "tree_outputs/feature_importances_xgb_bin.txt"
THRESHOLD_FILE = Path("tree_outputs/walk_thresholds.npz")

# %% load XGBoost-bin feature indices (sorted by importance)
print(f"Loading feature indices from {XGB_BIN_FEAT_FILE} …")
sel_cols = []
with open(XGB_BIN_FEAT_FILE) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        idx_str, _ = line.split(":", 1)
        sel_cols.append(int(idx_str.strip()))

sel_cols = np.array(sorted(sel_cols))
print(f"  {len(sel_cols)} features selected")

# %% load walk thresholds (RAW scale, not std-scaled)
print(f"Loading walk thresholds from {THRESHOLD_FILE} …")
tz = np.load(THRESHOLD_FILE)
ctx_thresholds = tz["ctx_thresholds"]            # shape (d_sae,), NaN = no threshold
diff_neg_thresholds = tz["diff_neg_thresholds"]   # shape (d_sae,)
diff_pos_thresholds = tz["diff_pos_thresholds"]   # shape (d_sae,)

# %% load sparse vectors (only selected columns)
print("Loading sparse vectors …")
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
n_samples = X_full.shape[0]
print(f"Full matrix: {n_samples} × {X_full.shape[1]}")

# Select only the features XGBoost found important (still RAW values)
X_sel_raw = X_full[:, sel_cols]
del X_full, Xs  # free ~big memory
print(f"Selected sub-matrix: {X_sel_raw.shape}")

# %% load metadata
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

# %% apply walk thresholds → binarise/trinarise
print("Applying walk thresholds (RAW scale) …")
X_bin = np.zeros((n_samples, len(sel_cols)), dtype=np.int8)

for j, col_idx in enumerate(sel_cols):
    col_raw = X_sel_raw[:, j].toarray().ravel()

    if col_idx < d_sae:
        # Context feature → 0/1
        sae_idx = col_idx
        th = ctx_thresholds[sae_idx]
        if not np.isnan(th):
            X_bin[:, j] = (col_raw >= th).astype(np.int8)
    else:
        # Diff feature → -1/0/1
        sae_idx = col_idx - d_sae
        neg_th = diff_neg_thresholds[sae_idx]
        pos_th = diff_pos_thresholds[sae_idx]
        if not np.isnan(pos_th):
            X_bin[:, j] = np.where(col_raw >= pos_th, 1,
                          np.where(col_raw <= neg_th, -1, 0)).astype(np.int8)

del X_sel_raw

n_ctx = np.sum(sel_cols < d_sae)
n_diff = np.sum(sel_cols >= d_sae)
print(f"Binarised matrix: {X_bin.shape}  ({n_ctx} ctx + {n_diff} diff)")

# %% build feature names
print("Loading feature names …")
with open("data/gemmascope_explanations/ex/gemma3_4b_it_layer22_16k_explanations.json") as f:
    explanations_list = json.load(f)

explanation_map = {int(item["index"]): item["description"] for item in explanations_list}

def sae_feature_name(prefix, sae_idx):
    desc = explanation_map.get(sae_idx, "")
    short = desc[:60] if desc else ""
    return f"{prefix}_{sae_idx}_{short}" if short else f"{prefix}_{sae_idx}"

sel_names = []
for col_idx in sel_cols:
    if col_idx < d_sae:
        sel_names.append(sae_feature_name("ctx", int(col_idx)))
    else:
        sel_names.append(sae_feature_name("diff", int(col_idx - d_sae)))

# %% train/test split
X_train = X_bin[train_idx]
X_test = X_bin[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
w_train, w_test = w[train_idx], w[test_idx]

print(f"X_train: {X_train.shape},  X_test: {X_test.shape}")
dense_mb = X_train.shape[0] * X_train.shape[1] * 8 / 1e6
print(f"Dense matrix ~{dense_mb:.0f} MB")

# %% fit RuleFit
print("\n" + "=" * 80)
print("Training RuleFit classifier …")
print("=" * 80)

start_time = time.time()
rulefit = RuleFitClassifier(
    n_estimators=100,
    tree_size=4,
    max_rules=2000,
    memory_par=0.01,
    random_state=42,
)

rulefit.fit(X_train, y_train, feature_names=sel_names)
print(f"RuleFit training complete in {time.time() - start_time:.2f} seconds.")

# %% evaluate
y_pred = rulefit.predict(X_test)
y_prob = rulefit.predict_proba(X_test)

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

# %% extract rules
print("\n" + "=" * 80)
print("Extracting rules …")
print("=" * 80)

rules = rulefit._get_rules(exclude_zero_coef=True)
rules = rules[rules["coef"] != 0].sort_values("importance", ascending=False)

n_rules = (rules["type"] == "rule").sum()
n_linear = (rules["type"] == "linear").sum()
print(f"Non-zero terms: {len(rules)}  ({n_rules} rules, {n_linear} linear terms)")

# %% print top rules
TOP_N = 50
print(f"\nTop {TOP_N} rules/terms by importance:")
print("-" * 120)

for i, (_, row) in enumerate(rules.head(TOP_N).iterrows()):
    kind = row["type"]
    coef = row["coef"]
    imp = row["importance"]
    support = row.get("support", float("nan"))
    rule_str = row["rule"]
    print(f"{i+1:3d}. [{kind:6s}] coef={coef:+.4f}  imp={imp:.4f}  support={support:.3f}  {rule_str}")

# %% save rules to file
output_lines = [
    f"RuleFit-Bin Results  ({len(sel_cols)} XGBoost-bin features: {n_ctx} ctx + {n_diff} diff)",
    f"Accuracy: {acc:.4f}  |  Weighted Acc: {w_acc:.4f}  |  F1: {f1:.4f}  |  AUC: {auc:.4f}",
    f"Non-zero terms: {len(rules)}  ({n_rules} rules, {n_linear} linear terms)",
    "",
    f"{'Rank':<5} {'Type':<8} {'Coef':>10} {'Importance':>12} {'Support':>9}  Rule",
    "-" * 120,
]

for i, (_, row) in enumerate(rules.iterrows()):
    kind = row["type"]
    coef = row["coef"]
    imp = row["importance"]
    support = row.get("support", float("nan"))
    rule_str = row["rule"]
    output_lines.append(
        f"{i+1:<5d} {kind:<8s} {coef:>+10.4f} {imp:>12.4f} {support:>9.3f}  {rule_str}"
    )

out_path = Path("tree_outputs1/rulefit_bin_rules.txt")
out_path.write_text("\n".join(output_lines) + "\n")
print(f"\nSaved all rules to {out_path}")

# %% save results JSON
results = {
    "accuracy": float(acc),
    "weighted_accuracy": float(w_acc),
    "f1": float(f1),
    "auc_roc": float(auc),
    "n_rules": int(n_rules),
    "n_linear": int(n_linear),
    "n_nonzero_terms": int(len(rules)),
    "n_train": int(len(y_train)),
    "n_test": int(len(y_test)),
    "n_features_selected": int(len(sel_cols)),
    "xgb_bin_feat_file": XGB_BIN_FEAT_FILE,
}

json_path = Path("tree_outputs1/rulefit_bin_results.json")
json_path.write_text(json.dumps(results, indent=2))
print(f"Saved metrics to {json_path}")
