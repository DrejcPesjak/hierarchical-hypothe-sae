# %% import libraries
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix, vstack
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# %% load data
N = None
batch_dir = Path("./data/sae_vectors")
batch_files = sorted(batch_dir.glob("confirmatory_preprocessed_sae_vectors_sparse_minibatch_*.npz"))
batch_files = batch_files[:N] if N is not None else batch_files

Xs = []
for f in batch_files:
    z = np.load(f)
    Xs.append(csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"])))

X = vstack(Xs, format="csr")  # rows = examples
# print(X.shape, X.nnz)

best_files = sorted(batch_dir.glob("confirmatory_preprocessed_sae_vectors_best_minibatch_*.npy"))
best_files = best_files[:N] if N is not None else best_files
y = np.concatenate([np.load(f).reshape(-1) for f in best_files], axis=0)
# print(y.shape)


# %% split data (by pairs to avoid data leakage)
# Each pair has 2 rows: row 2i = best-worst, row 2i+1 = worst-best
# We must keep pairs together since worst-best = -1 * (best-worst)
print("Splitting data into train and test sets (by pair)...")
n_pairs = X.shape[0] // 2
pair_indices = np.arange(n_pairs)

# Split at the pair level
train_pairs, test_pairs = train_test_split(pair_indices, test_size=0.2, random_state=42)

# Convert pair indices to row indices
train_idx = np.sort(np.concatenate([train_pairs * 2, train_pairs * 2 + 1]))
test_idx = np.sort(np.concatenate([test_pairs * 2, test_pairs * 2 + 1]))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"Train: {len(train_pairs)} pairs ({X_train.shape[0]} rows), Test: {len(test_pairs)} pairs ({X_test.shape[0]} rows)")

# %% standardize features
scaler = StandardScaler(with_mean=False)
# scaler = MaxAbsScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %% train logistic regression
print("Training logistic regression model...")

model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    n_jobs=-1,
    random_state=42,
    eval_metric='mlogloss',
    verbosity=1 
)

model.fit(X_train, y_train)


# %% evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))

# %% feature importance
if isinstance(model, SGDClassifier):
    feature_importances = model.coef_
elif isinstance(model, XGBClassifier):
    feature_importances = model.feature_importances_.reshape(1, -1)
else:
    raise ValueError("Unsupported model type for feature importance extraction.")

top10 = np.argsort(feature_importances.flatten())[-10:][::-1]
bottom10 = np.argsort(feature_importances.flatten())[:10]
print("\nTop 10 feature indices and importance scores:")
for idx in top10:
    print(f"Feature {idx}: {feature_importances[0][idx]}")
print("\nBottom 10 feature indices and importance scores:")
for idx in bottom10:
    print(f"Feature {idx}: {feature_importances[0][idx]}")

# %% get feature names from neuronpedia
from IPython.display import IFrame
html_template = "https://neuronpedia.org/{}/{}/{}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"
# example_url = "https://neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/1500?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"

def get_dashboard_html(sae_release = "gemma-3-4b-it", sae_id="22-gemmascope-2-res-16k", feature_idx=0):
    return html_template.format(sae_release, sae_id, feature_idx)

html = get_dashboard_html(feature_idx=top10[0])
IFrame(html, width=1200, height=600)

# %% get explanations
# class exact match: "text-left font-sans text-[11.5px] sm:text-[13px]"
# if it exists
import requests
from bs4 import BeautifulSoup
def get_explanation(url):
    response = requests.get(url)
    html_content = response.text
    soup = BeautifulSoup(html_content, 'html.parser')
    explanation = soup.find_all(class_="text-left font-sans text-[11.5px] sm:text-[13px]")
    if explanation:
        return explanation[0].text
    return None

print("\nTop 10 feature explanations:")
for feature_idx in top10:
    url = get_dashboard_html(feature_idx=feature_idx)
    explanation = get_explanation(url)
    print(f"Feature {feature_idx}: {explanation}")

print("\nBottom 10 feature explanations:")
for feature_idx in bottom10:
    url = get_dashboard_html(feature_idx=feature_idx)
    explanation = get_explanation(url)
    print(f"Feature {feature_idx}: {explanation}")

# %%
from xgboost import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(3000, 2000))  
plot_tree(model, num_trees=1) 
plt.show()

# %% graphviz .dot
import graphviz
import xgboost as xgb
import os
tree_dot = xgb.to_graphviz(model, num_trees=1)
# Save the dot file
dot_file_path = "xgboost_tree.dot"
tree_dot.save(dot_file_path)
# Convert dot file to png and display
with open(dot_file_path) as f:
    dot_graph = f.read()
# Use graphviz to display the tree
graph = graphviz.Source(dot_graph)
graph.render("xgboost_tree")
# Optionally, visualize the graph directly
graph