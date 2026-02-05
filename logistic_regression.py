# %% import libraries
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix, vstack
from sklearn.linear_model import SGDClassifier
# from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
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


# %% split data
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% standardize features
scaler = StandardScaler(with_mean=False)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %% train logistic regression
print("Training logistic regression model...")

# model = SGDClassifier(
#     loss="log_loss",      
#     penalty="l2",
#     alpha=1e-4,           
#     max_iter=500,         
#     tol=1e-3,
#     verbose=1,
#     n_jobs=-1,
#     random_state=42,
# )

model = SGDClassifier(loss='log_loss', penalty='l2', alpha=0.0001, max_iter=2000, tol=1e-3, verbose=1, n_jobs=-1, random_state=0)
# Accuracy: 0.5883507269645883
#               precision    recall  f1-score   support

#            0       0.74      0.71      0.72      8511
#            1       0.18      0.21      0.19      2700

#     accuracy                           0.59     11211
#    macro avg       0.46      0.46      0.46     11211
# weighted avg       0.60      0.59      0.60     11211

# model = XGBClassifier(
#     n_estimators=100,
#     max_depth=6,
#     learning_rate=0.1,
#     n_jobs=-1,
#     random_state=42,
#     eval_metric='mlogloss',
#     verbosity=1 
# )
# # Accuracy: 0.7565783605387566
# #               precision    recall  f1-score   support

# #            0       0.76      1.00      0.86      8511
# #            1       0.13      0.00      0.00      2700

# #     accuracy                           0.76     11211
# #    macro avg       0.44      0.50      0.43     11211
# # weighted avg       0.61      0.76      0.65     11211

model.fit(X_train, y_train)


# %% evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))

# %% feature importance
feature_importances = model.coef_
# feature_importances = model.feature_importances_.reshape(1, -1)
# print(feature_importances.shape)
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


# Top 10 feature explanations:
# Feature 1529: teacher and school
# Feature 604: harmful unethical behavior
# Feature 3782: function calls in code
# Feature 5543: support and resources
# Feature 14972: computer vision science
# Feature 1448: frankly, ultimately, consequently, therefore
# Feature 2053: word suffixes
# Feature 11021: exhibitions, trade shows, and stalls
# Feature 10303: dimensions or quantities
# Feature 11784: code and technical terms

# Bottom 10 feature explanations:
# Feature 13041: values and comparisons
# Feature 6380: health and loading
# Feature 3962: Psychology Today, Chief Technology
# Feature 6069: Chinese punctuation and structure
# Feature 10106: soccer game events and actions
# Feature 9490: violence, abuse, and trauma
# Feature 2003: 
# Feature 4421: design and designing
# Feature 183: events following a transition
# Feature 14: names of people, places, or things