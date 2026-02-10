# %% import libraries
import numpy as np
import json
from pathlib import Path
from scipy.sparse import csr_matrix, vstack
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# %% load important feature indices from xgboost
important = {}
with open("feature_importances_xgb.txt") as f:
    for line in f:
        idx_str, score_str = line.strip().split(": ", 1)
        important[int(idx_str)] = float(score_str)

important_indices = np.array(sorted(important.keys()))
print(f"Number of important features: {len(important_indices)}")

# %% load data (only important columns)
N = None
batch_dir = Path("./data/sae_vectors")
batch_files = sorted(batch_dir.glob("confirmatory_preprocessed_sae_vectors_sparse_minibatch_*.npz"))
batch_files = batch_files[:N] if N is not None else batch_files

Xs = []
for f in batch_files:
    z = np.load(f)
    mat = csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    Xs.append(mat[:, important_indices])

X = vstack(Xs, format="csr")
print(f"X shape (filtered): {X.shape}")

best_files = sorted(batch_dir.glob("confirmatory_preprocessed_sae_vectors_best_minibatch_*.npy"))
best_files = best_files[:N] if N is not None else best_files
y = np.concatenate([np.load(f).reshape(-1) for f in best_files], axis=0)
print(f"y shape: {y.shape}")

# %% split data (by pairs to avoid data leakage)
print("Splitting data into train and test sets (by pair)...")
n_pairs = X.shape[0] // 2
pair_indices = np.arange(n_pairs)

train_pairs, test_pairs = train_test_split(pair_indices, test_size=0.2, random_state=42)

train_idx = np.sort(np.concatenate([train_pairs * 2, train_pairs * 2 + 1]))
test_idx = np.sort(np.concatenate([test_pairs * 2, test_pairs * 2 + 1]))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"Train: {len(train_pairs)} pairs ({X_train.shape[0]} rows), Test: {len(test_pairs)} pairs ({X_test.shape[0]} rows)")

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
for idx in important_indices:
    if idx in explanation_map:
        feature_names.append(f"f{idx}: {explanation_map[idx]}")
    else:
        feature_names.append(f"f_{idx}")

print(f"Feature names loaded: {len(feature_names)} ({sum(1 for n in feature_names if n.startswith('f_'))} missing explanations)")

# %% train decision tree
print("Training decision tree...")
dt = DecisionTreeClassifier(
    max_depth=8,
    min_samples_leaf=20,
    random_state=42,
)
dt.fit(X_train, y_train)
print(f"Tree depth: {dt.get_depth()}, leaves: {dt.get_n_leaves()}")

# %% evaluate on test set
y_pred = dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
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
fig.savefig("decision_tree_uhd.png", dpi=160, bbox_inches="tight")
print("Saved to decision_tree_uhd.png")
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
