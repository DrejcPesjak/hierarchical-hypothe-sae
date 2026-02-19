#!/usr/bin/env python3
"""
Print example A/B headline pairs from the dataset.

1. Top 3 activating pairs for the top 15 RuleFit features (hardcoded SAE indices)
2. Top 3 activating pairs for 3 specific paths in XGBoost booster 0 (hardcoded paths)

Uses the same data loading and headline lookup as feature_analysis.py.
Feature names from explanations JSON. Thresholding by tree on binarized data.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix, vstack, csc_matrix
from sklearn.preprocessing import StandardScaler

# %% configuration
d_sae = 16384
CSV_PATH = Path("data/cleaned_archive/confirmatory_preprocessed.csv")
EXPL_PATH = Path("data/gemmascope_explanations/ex/gemma3_4b_it_layer22_16k_explanations.json")
THRESHOLD_FILE = Path("tree_outputs/walk_thresholds.npz")

# Top 15 RuleFit features (all diff) - SAE indices. Column = d_sae + idx.
TOP15_SAE_INDICES = [
    2220,   # 1: 'this'
    11052,  # 2: How to write clickbait titles
    1730,   # 3: remarkable and amazing qualities
    433,    # 4
    9410,   # 5: subtitles
    1531,   # 6: discovery and revelation
    2223,   # 7: past actions or events
    1847,   # 8: shocking or horrifying scenarios
    604,    # 9: harmful unethical behavior
    1170,   # 10: math and trick questions
    2203,   # 11: stereotypes about gender roles
    11979,  # 12: sexual and explicit themes
    5439,   # 13: offensive content
    70,     # 14
    2161,   # 15: female names and women
]

# Tree paths (booster 0): list of (sae_idx, threshold, take_yes_branch).
# take_yes=True: row satisfies value < threshold. take_yes=False: value >= threshold.
# PATH_A = [  # 0→1→3→7→14: clickbait → neg emotion → governance/risk NO
#     (1389, 1, True),   # diff_1389 < 1
#     (11052, 0, True),  # diff_11052 < 0
#     (1213, 1, True),   # diff_1213 < 1
#     (538, 0, False),  # diff_538 >= 0
# ]
# PATH_B = [  # 0→1→4→10→19→25→32: not clickbait → not offensive → video → they → neg emotion NO
#     (1389, 1, True),
#     (11052, 0, False),
#     (5439, 0, False),
#     (3669, 1, True),
#     (926, 1, True),
#     (1213, 0, False),
# ]
# PATH_LEAF_34 = [  # 0→1→4→10→19→26→34
#     (1389, 1, True),   # diff_1389 < 1
#     (11052, 0, False),  # diff_11052 >= 0
#     (5439, 0, False),   # diff_5439 >= 0
#     (3669, 1, True),    # diff_3669 < 1
#     (926, 1, False),    # diff_926 >= 1
#     (1213, 0, False),  # diff_1213 >= 0
# ]
PATH_LEAF_34 = [  # 0→1→4→10→19→26→34
    # (1389, 1, False),   # diff_1389 >= 1
    (11052, 1, False),  # diff_11052 >= 1
    # (5439, 1, False),   # diff_5439 >= 1
    # (3669, 1, False),    # diff_3669 >= 1
    # (926, 1, False),    # diff_926 >= 1
    # (1213, 1, False),  # diff_1213 >= 1
]
PATH_LEAF_30 = [  # 0→2→5→11→22→30
    (1389, 1, False),   # diff_1389 >= 1
    (538, 1, True),     # diff_538 < 1
    (12614, 1, True),   # diff_12614 < 1
    (3669, 0, False),   # diff_3669 >= 0
    (2633, 0, False),   # diff_2633 >= 0
]
PATH_C = [  # 0→2→5→11→21: past events NO → governance YES → questions YES → video YES
    (1389, 1, False),
    (538, 1, True),
    (12614, 1, True),
    (3669, 0, True),
]

# %% load sparse vectors
print("Loading sparse vectors …")
batch_dir = Path("data/sae_vectors")
sparse_files = sorted(batch_dir.glob("confirmatory_preprocessed_sae_vectors_sparse_minibatch_*.npz"))
meta_files = sorted(batch_dir.glob("confirmatory_preprocessed_sae_vectors_meta_minibatch_*.npz"))

Xs = []
for f in sparse_files:
    z = np.load(f)
    mat = csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    Xs.append(mat)
X_full = vstack(Xs, format="csr")
n_samples = X_full.shape[0]
print(f"Full matrix: {n_samples} × {X_full.shape[1]}")

# %% load metadata
print("Loading metadata …")
test_ids_list, id_first_list, id_second_list, w_list, y_list = [], [], [], [], []
for f in meta_files:
    m = np.load(f, allow_pickle=True)
    test_ids_list.append(m["test_id"])
    id_first_list.append(m["id_first"])
    id_second_list.append(m["id_second"])
    w_list.append(m["weight_diff"])
    y_list.append(m["better_label"])

all_test_ids = np.concatenate(test_ids_list)
all_id_first = np.concatenate(id_first_list)
all_id_second = np.concatenate(id_second_list)
w = np.concatenate(w_list)
y = np.concatenate(y_list)  # better_label: 1 = A wins, 0 = B wins

# %% headline lookup
print("Loading CSV for headline lookup …")
df = pd.read_csv(CSV_PATH)
headline_lookup = {}
for test_id, group in df.groupby("clickability_test_id"):
    group = group.reset_index(drop=True)
    for local_idx in range(len(group)):
        headline_lookup[(str(test_id), int(local_idx))] = str(group.loc[local_idx, "headline"])


def get_headline_first(i):
    return headline_lookup.get((str(all_test_ids[i]), int(all_id_first[i])), "<unknown>")


def get_headline_second(i):
    return headline_lookup.get((str(all_test_ids[i]), int(all_id_second[i])), "<unknown>")


# %% load thresholds and binarize
print("Loading thresholds and binarizing …")
tz = np.load(THRESHOLD_FILE)
ctx_thresholds = tz["ctx_thresholds"]
diff_neg_thresholds = tz["diff_neg_thresholds"]
diff_pos_thresholds = tz["diff_pos_thresholds"]

X_ctx = X_full[:, :d_sae]
X_ctx_bin = np.zeros((n_samples, d_sae), dtype=np.int8)
X_ctx_csc = X_ctx.tocsc()
for j in range(d_sae):
    th = ctx_thresholds[j]
    if np.isnan(th):
        continue
    col = X_ctx_csc[:, j].toarray().ravel()
    X_ctx_bin[:, j] = (col >= th).astype(np.int8)

X_diff = X_full[:, d_sae:]
scaler_diff = StandardScaler(with_mean=False)
scaler_diff.fit(X_diff)
X_diff_tri = np.zeros((n_samples, d_sae), dtype=np.int8)
X_diff_csc = X_diff.tocsc()
for j in range(d_sae):
    neg_th = diff_neg_thresholds[j]
    pos_th = diff_pos_thresholds[j]
    if np.isnan(pos_th):
        continue
    col = X_diff_csc[:, j].toarray().ravel()
    X_diff_tri[:, j] = np.where(col >= pos_th, 1, np.where(col <= neg_th, -1, 0)).astype(np.int8)

X_bin = np.hstack([X_ctx_bin, X_diff_tri])
del X_ctx, X_diff, X_ctx_csc, X_diff_csc, X_ctx_bin, X_diff_tri

# %% load SAE explanations
with open(EXPL_PATH) as f:
    explanations_list = json.load(f)
explanation_map = {int(item["index"]): item["description"] for item in explanations_list}


def feat_label(sae_idx):
    desc = explanation_map.get(sae_idx, "")
    short = desc[:60] if desc else ""
    return f"diff_{sae_idx}_{short}" if short else f"diff_{sae_idx}"


# %% print top 3 activating pairs for each top 15 feature
print("\n" + "=" * 80)
print("TOP 3 ACTIVATING PAIRS FOR TOP 15 RULEFIT FEATURES")
print("=" * 80)

for rank, sae_idx in enumerate(TOP15_SAE_INDICES, start=1):
    col_idx = d_sae + sae_idx
    col_raw = X_full[:, col_idx].toarray().ravel()
    order = np.argsort(col_raw)[::-1]

    print(f"\n--- Rank {rank}: {feat_label(sae_idx)}")
    shown = 0
    for idx in order:
        if shown >= 3:
            break
        v = col_raw[idx]
        if v <= 0:
            continue  # for diff, we want A > B (positive)
        h1 = get_headline_first(idx)
        h2 = get_headline_second(idx)
        lbl = y[idx]
        wt = w[idx]
        print(f"  [{v:.4f}] label={lbl} w={wt:.4f}  H1 (A): {h1}")
        print(f"         H2 (B): {h2}")
        shown += 1
    if shown == 0:
        order_abs = np.argsort(np.abs(col_raw))[::-1]
        for idx in order_abs[:3]:
            v = col_raw[idx]
            h1 = get_headline_first(idx)
            h2 = get_headline_second(idx)
            lbl = y[idx]
            wt = w[idx]
            print(f"  [{v:.4f}] label={lbl} w={wt:.4f}  H1 (A): {h1}")
            print(f"         H2 (B): {h2}")

# %% tree path filtering on binarized data
def mask_satisfies_path(X_bin, path):
    """Return boolean mask of rows satisfying all tree-split conditions. path: [(sae_idx, thresh, take_yes), ...]"""
    mask = np.ones(X_bin.shape[0], dtype=bool)
    for sae_idx, thresh, take_yes in path:
        col = d_sae + sae_idx
        val = X_bin[:, col]
        if take_yes:
            mask &= (val < thresh)
        else:
            mask &= (val >= thresh)
    return mask


def cumulative_path_score(X_full, indices, path, scaler_diff):
    """
    For each row in indices, compute cumulative score from standardized diff values.
    take_yes (e.g. < 0): contribution = -std_val (most negative = best)
    take_no (e.g. >= 0): contribution = +std_val (most positive = best)
    Returns array of scores, same length as indices.
    """
    scores = np.zeros(len(indices), dtype=np.float64)
    for sae_idx, thresh, take_yes in path:
        col = d_sae + sae_idx
        raw_vals = X_full[indices, col].toarray().ravel()
        scale = scaler_diff.scale_[sae_idx]
        scale = max(scale, 1e-10)  # avoid div by zero
        std_vals = raw_vals / scale
        if take_yes:
            scores += -std_vals  # most negative = highest contribution
        else:
            scores += std_vals   # most positive = highest contribution
    return scores


# %% find pairs satisfying each path, rank by cumulative feat score, print top 3
print("\n" + "=" * 80)
print("TOP 3 PAIRS FOR XGBOOST TREE PATHS (booster 0)")
print("=" * 80)

paths_info = [
    ("Path A / Leaf 34 (0→1→4→10→19→26→34): past events NO → clickbait YES → offensive NO → video NO → they YES → neg emotion YES", PATH_LEAF_34),
    ("Path B / Leaf 30 (0→2→5→11→22→30): past events YES → governance NO → questions NO → video YES → descriptions YES", PATH_LEAF_30),
    ("Path C (0→2→5→11→21): past events NO → governance YES → questions YES → video YES", PATH_C),
]

for path_name, path in paths_info:
    print(f"\n--- {path_name}")
    mask = mask_satisfies_path(X_bin, path)
    indices = np.where(mask)[0]
    print(f"Pairs satisfying path: {len(indices)}")
    if len(indices) > 0:
        scores = cumulative_path_score(X_full, indices, path, scaler_diff)
        order = np.argsort(scores)[::-1]
        sorted_idx = indices[order]
        for k, idx in enumerate(sorted_idx[:3]):
            h1 = get_headline_first(idx)
            h2 = get_headline_second(idx)
            cum = scores[order[k]]
            lbl = y[idx]
            wt = w[idx]
            print(f"  [{k+1}] cum_score={cum:.4f}  better_label={lbl}  weight={wt:.4f}")
            print(f"       H1 (A): {h1}")
            print(f"       H2 (B): {h2}")
    else:
        print("  (no pairs satisfy this path)")

print("\nDone.")

