# %% import libraries
"""
RuleFit model on context+diff features → predict better_label (with sample weights).
Uses the same train/test split produced by clustering_context.py.

To keep RAM manageable we select only the top-K most-active features
from context and diff separately (by nnz count in training data).
RuleFit internally builds a dense [samples × (features + rules)] matrix,
so 29k features × 45k samples would need >10 GB before rules even start.

pip install imodels
"""
import time
import numpy as np
import json
from pathlib import Path
from scipy.sparse import csr_matrix, vstack, hstack
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score, roc_auc_score,
)
from imodels import RuleFitClassifier

# %% configuration
d_sae = 16384
N = None                # set to int to limit batches (for debugging)
XGB_FEAT_FILE = "tree_outputs/feature_importances_xgb_new.txt"  # use XGBoost-selected features

# %% load data (full context+diff vectors + metadata)
print("Loading sparse vectors and metadata...")
batch_dir = Path("./data/sae_vectors")
sparse_files = sorted(batch_dir.glob("confirmatory_preprocessed_sae_vectors_sparse_minibatch_*.npz"))
meta_files = sorted(batch_dir.glob("confirmatory_preprocessed_sae_vectors_meta_minibatch_*.npz"))
sparse_files = sparse_files[:N] if N is not None else sparse_files
meta_files = meta_files[:N] if N is not None else meta_files

Xs = []
for f in sparse_files:
    z = np.load(f)
    mat = csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    Xs.append(mat)  # full [context | diff]

X_full = vstack(Xs, format="csr")
print(f"Full vectors: {X_full.shape[0]} examples × {X_full.shape[1]} features")

# Load metadata
y_list, w_list = [], []
for f in meta_files:
    m = np.load(f, allow_pickle=True)
    y_list.append(m["better_label"])
    w_list.append(m["weight_diff"])

y = np.concatenate(y_list)
w = np.concatenate(w_list)
print(f"Label distribution: 0={np.sum(y == 0)}, 1={np.sum(y == 1)}")

# %% load train/test split (from clustering_context.py)
print("Loading train/test split...")
ca = np.load("tree_outputs/cluster_assignments.npz")
train_idx = ca["train_idx"]
test_idx = ca["test_idx"]
print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

# %% build feature names
print("Loading feature names...")
with open("data/gemmascope_explanations/ex/gemma3_4b_it_layer22_16k_explanations.json") as f:
    explanations_list = json.load(f)

explanation_map = {int(item["index"]): item["description"] for item in explanations_list}

def sae_feature_name(prefix, sae_idx):
    desc = explanation_map.get(sae_idx, "")
    short = desc[:60] if desc else ""
    return f"{prefix}_{sae_idx}_{short}" if short else f"{prefix}_{sae_idx}"

# %% select features from XGBoost feature importances
print(f"Loading XGBoost feature indices from {XGB_FEAT_FILE}...")

sel_cols = []
with open(XGB_FEAT_FILE) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        idx_str, _ = line.split(":", 1)
        sel_cols.append(int(idx_str.strip()))

sel_cols = np.array(sorted(sel_cols))

# Build feature names: indices < d_sae are context, >= d_sae are diff
sel_names = []
n_ctx = 0
n_diff = 0
for col in sel_cols:
    if col < d_sae:
        sel_names.append(sae_feature_name("ctx", int(col)))
        n_ctx += 1
    else:
        sel_names.append(sae_feature_name("diff", int(col - d_sae)))
        n_diff += 1

X_sel = X_full[:, sel_cols]
n_feat = X_sel.shape[1]
print(f"Selected {n_feat} XGBoost features  ({n_ctx} context + {n_diff} diff)")

# %% train/test split
X_train = X_sel[train_idx]
X_test = X_sel[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
w_train, w_test = w[train_idx], w[test_idx]

# Standardize (no mean-centering for sparse data)
scaler = StandardScaler(with_mean=False)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"X_train: {X_train.shape},  X_test: {X_test.shape}")
dense_mb = X_train.shape[0] * X_train.shape[1] * 8 / 1e6
print(f"Dense matrix would be ~{dense_mb:.0f} MB — manageable for RuleFit")

# %% fit RuleFit
print("\n" + "=" * 80)
print("Training RuleFit classifier...")
print("=" * 80)

start_time = time.time()
rulefit = RuleFitClassifier(
    n_estimators=50,       # number of trees to generate rules from
    tree_size=4,            # max leaf nodes per tree (controls rule complexity)
    max_rules=500,         # max rules to keep after L1 selection
    memory_par=0.01,        # Friedman's memory parameter
    random_state=42,
)

# Note: imodels RuleFitClassifier does NOT support sample_weight in .fit()
# Weights are still used for evaluation metrics below.
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
print("Extracting rules...")
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
    f"RuleFit Results  ({n_feat} XGBoost-selected features: {n_ctx} ctx + {n_diff} diff)",
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

out_path = Path("tree_outputs/rulefit_rules.txt")
out_path.write_text("\n".join(output_lines) + "\n")
print(f"\nSaved all rules to {out_path}")

# %% save results JSON
results = {
    "accuracy": acc,
    "weighted_accuracy": w_acc,
    "f1": f1,
    "auc_roc": auc,
    "n_rules": int(n_rules),
    "n_linear": int(n_linear),
    "n_nonzero_terms": int(len(rules)),
    "n_train": int(len(y_train)),
    "n_test": int(len(y_test)),
    "n_features_selected": int(n_feat),
    "xgb_feat_file": XGB_FEAT_FILE,
}

json_path = Path("tree_outputs/rulefit_results.json")
json_path.write_text(json.dumps(results, indent=2))
print(f"Saved metrics to {json_path}")
