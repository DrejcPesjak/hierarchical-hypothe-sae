# %% import libraries
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix, vstack
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GroupShuffleSplit

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
    Xs.append(csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"])))

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
sample_weight = w_diff
# sample_weight = w_logit    # logit difference (eps-clamped)
# sample_weight = w_beta     # Jeffreys beta posterior + sqrt(min impressions)

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

# Verify no overlap
assert len(set(test_ids[train_idx]) & set(test_ids[test_idx])) == 0, "Data leakage: overlapping test_ids!"

# %% standardize features
scaler = StandardScaler(with_mean=False)
# scaler = MaxAbsScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %% train logistic regression (with sample weights)
print("Training weighted logistic regression model...")

# model = SGDClassifier(loss='log_loss', penalty='l2', alpha=0.0001, max_iter=2000, tol=1e-3, verbose=1, n_jobs=-1, random_state=0)

model = XGBClassifier(
    n_estimators=40,
    max_depth=6,
    learning_rate=0.1,
    n_jobs=-1,
    random_state=42,
    eval_metric='mlogloss',
    verbosity=1 
)

model.fit(X_train, y_train, sample_weight=w_train)

# %% evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))

# Weighted accuracy (using test weights)
weighted_acc = accuracy_score(y_test, y_pred, sample_weight=w_test)
print(f"Weighted accuracy: {weighted_acc}")

# %% feature importance
# Features are [context (0..d_sae-1) | diff (d_sae..2*d_sae-1)]
d_sae = X.shape[1] // 2
# feature_importances = model.coef_
feature_importances = model.feature_importances_.reshape(1, -1)

# Overall top/bottom features
top10 = np.argsort(feature_importances.flatten())[-10:][::-1]
bottom10 = np.argsort(feature_importances.flatten())[:10]

print("\nTop 10 feature indices and importance scores:")
for idx in top10:
    region = "context" if idx < d_sae else "diff"
    sae_idx = idx if idx < d_sae else idx - d_sae
    print(f"  Feature {idx} ({region}[{sae_idx}]): {feature_importances[0][idx]:.6f}")

print("\nBottom 10 feature indices and importance scores:")
for idx in bottom10:
    region = "context" if idx < d_sae else "diff"
    sae_idx = idx if idx < d_sae else idx - d_sae
    print(f"  Feature {idx} ({region}[{sae_idx}]): {feature_importances[0][idx]:.6f}")

# %% dump all non zero feature importances indices and scores into file
with open("tree_outputs/feature_importances_xgb_new.txt", "w") as f:
    for idx in np.where(feature_importances.flatten() != 0)[0]:
        f.write(f"{idx}: {feature_importances[0][idx]}\n")

# %% get feature names from neuronpedia
from IPython.display import IFrame
html_template = "https://neuronpedia.org/{}/{}/{}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"

def get_dashboard_html(sae_release="gemma-3-4b-it", sae_id="22-gemmascope-2-res-16k", feature_idx=0):
    return html_template.format(sae_release, sae_id, feature_idx)

# Show dashboard for top feature (mapped back to SAE index)
top_sae_idx = top10[0] if top10[0] < d_sae else top10[0] - d_sae
html = get_dashboard_html(feature_idx=top_sae_idx)
IFrame(html, width=1200, height=600)

# %% get explanations
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
for idx in top10:
    sae_idx = idx if idx < d_sae else idx - d_sae
    region = "context" if idx < d_sae else "diff"
    url = get_dashboard_html(feature_idx=sae_idx)
    explanation = get_explanation(url)
    print(f"  Feature {idx} ({region}[{sae_idx}]): {explanation}")

print("\nBottom 10 feature explanations:")
for idx in bottom10:
    sae_idx = idx if idx < d_sae else idx - d_sae
    region = "context" if idx < d_sae else "diff"
    url = get_dashboard_html(feature_idx=sae_idx)
    explanation = get_explanation(url)
    print(f"  Feature {idx} ({region}[{sae_idx}]): {explanation}")

