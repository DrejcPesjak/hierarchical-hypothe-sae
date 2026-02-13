# %% import libraries
import numpy as np
import json
from pathlib import Path
from scipy.sparse import csr_matrix, vstack
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
import matplotlib.pyplot as plt

# %% load important feature indices from xgboost
important = {}
with open("tree_outputs/feature_importances_xgb_new.txt") as f:
    for line in f:
        idx_str, score_str = line.strip().split(": ", 1)
        important[int(idx_str)] = float(score_str)

important_indices = np.array(sorted(important.keys()))
print(f"Number of important features: {len(important_indices)}")


# %% load data
N = None
batch_dir = Path("./data/sae_vectors")
sparse_files = sorted(batch_dir.glob("confirmatory_preprocessed_sae_vectors_sparse_minibatch_*.npz"))
meta_files = sorted(batch_dir.glob("confirmatory_preprocessed_sae_vectors_meta_minibatch_*.npz"))
sparse_files = sparse_files[:N] if N is not None else sparse_files
meta_files = meta_files[:N] if N is not None else meta_files

# Load sparse vectors [context | diff] (each row is 2*d_sae wide)
Xs = []
for f in sparse_files:
    z = np.load(f)
    mat = csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    Xs.append(mat[:, important_indices])

X = vstack(Xs, format="csr")  # rows = examples

# Load metadata
test_ids_list, y_list = [], []
w_diff_list, w_logit_list, w_beta_list = [], [], []

for f in meta_files:
    m = np.load(f, allow_pickle=True)
    test_ids_list.append(m["test_id"])
    y_list.append(m["better_label"])
    w_diff_list.append(m["weight_diff"])
    w_logit_list.append(m["weight_logit"])
    w_beta_list.append(m["weight_beta"])

test_ids = np.concatenate(test_ids_list)
y = np.concatenate(y_list)
w_diff = np.concatenate(w_diff_list)
w_logit = np.concatenate(w_logit_list)
w_beta = np.concatenate(w_beta_list)

print(f"Loaded {X.shape[0]} examples, {X.shape[1]} features (context+diff)")
print(f"Unique test_ids: {len(np.unique(test_ids))}")
print(f"Label distribution: 0={np.sum(y == 0)}, 1={np.sum(y == 1)}")

# %% choose weight scheme
# Use simplest weight: absolute CTR difference
# sample_weight = w_diff
# sample_weight = w_logit    # logit difference (eps-clamped)
sample_weight = w_beta     # Jeffreys beta posterior + sqrt(min impressions)

# %% split data (by test_id to avoid data leakage)
# All pairs from the same test_id must go into the same split
print("Splitting data into train and test sets (by test_id)...")

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=test_ids))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
w_train, w_test = sample_weight[train_idx], sample_weight[test_idx]

n_train_ids = len(np.unique(test_ids[train_idx]))
n_test_ids = len(np.unique(test_ids[test_idx]))
print(f"Train: {n_train_ids} test_ids ({X_train.shape[0]} rows), Test: {n_test_ids} test_ids ({X_test.shape[0]} rows)")

# %% standardize features
scaler = StandardScaler(with_mean=False)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %% load feature names from neuronpedia explanations
print("Loading feature names from explanations JSON...")
with open("data/gemmascope_explanations/ex/gemma3_4b_it_layer22_16k_explanations.json") as f:
    explanations_list = json.load(f)

# Build index -> description map
explanation_map = {int(item["index"]): item["description"] for item in explanations_list}

feature_names = []
d_sae = 16384
for i in important_indices:
    e = 'c'
    idx = i
    if i >= d_sae:
        idx = i - d_sae
        e = 'Δ'

    if idx in explanation_map:
        feature_names.append(f"{e}f_{idx}: {explanation_map[idx]}")
    else:
        feature_names.append(f"{e}f_{idx}")

print(f"Feature names loaded: {len(feature_names)} ({sum(1 for n in feature_names if ':' not in n)} missing explanations)")

# %% train decision tree
print("Training decision tree...")
dt = DecisionTreeClassifier(
    max_depth=8,
    min_samples_leaf=20,
    random_state=42,
)
dt.fit(X_train, y_train, sample_weight=w_train)
print(f"Tree depth: {dt.get_depth()}, leaves: {dt.get_n_leaves()}")

# %% evaluate on test set
y_pred = dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred, sample_weight=w_test)
print(f"\nAccuracy: {accuracy}")
print(classification_report(y_test, y_pred))

# %% plot tree in UHD resolution
print("Plotting tree (UHD)...")
class_names = [str(c) for c in dt.classes_]

fig, ax = plt.subplots(figsize=(120, 60), dpi=160)  # UHD: ~19200x9600 px
plot_tree(
    dt,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=6,
    ax=ax,
)
ax.set_title("Decision Tree (XGB-selected features)", fontsize=24)
fig.tight_layout()
fig.savefig("tree_outputs/decision_tree_uhd5.png", dpi=160, bbox_inches="tight")
print("Saved to tree_outputs/decision_tree_uhd5.png")
plt.show()


# %% get first 4 levels feature names
def get_tree_paths(tree, feature_names, max_depth=4):
    paths = []

    def recurse(node_id=0, path=[]):
        if node_id == -1 or len(path) >= max_depth:
            return
        feature_idx = tree.feature[node_id]
        if feature_idx != -2:  # Not a leaf
            feature_name = feature_names[feature_idx]
            paths.append([feature_name])
            recurse(tree.children_left[node_id], path + [feature_name + " <= ..."])
            recurse(tree.children_right[node_id], path + [feature_name + " > ..."])

    recurse()
    return paths

tree = dt.tree_
paths = get_tree_paths(tree, feature_names, max_depth=4)
print("\nFirst 4 levels of the tree (feature names):")
for i, path in enumerate(paths):
    print(f"Path {i+1}: {' -> '.join(path)}")
