# %% import libraries
# pip install hdbscan umap-learn
import numpy as np
import json
from pathlib import Path
from scipy.sparse import csr_matrix, vstack
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.manifold import TSNE
import hdbscan
import umap
import matplotlib.pyplot as plt

# %% configuration
d_sae = 16384
N = None                # set to int to limit batches (for debugging)
TOP_K_FEATURES = 20     # distinctive features per cluster
SVD_DIMS = 200          # dimensionality for TruncatedSVD before clustering

# %% load data (context vectors only)
print("Loading context vectors...")
batch_dir = Path("./data/sae_vectors")
sparse_files = sorted(batch_dir.glob("confirmatory_preprocessed_sae_vectors_sparse_minibatch_*.npz"))
sparse_files = sparse_files[:N] if N is not None else sparse_files

Xs = []
for f in sparse_files:
    z = np.load(f)
    mat = csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    Xs.append(mat[:, :d_sae])  # context half only

X_ctx = vstack(Xs, format="csr")
print(f"Context vectors: {X_ctx.shape[0]} examples × {X_ctx.shape[1]} features")

# %% remove dead neurons (columns that are all zero across all examples)
print("Removing dead neurons...")
col_nnz = np.diff(X_ctx.tocsc().indptr)   # nnz per column
alive_mask = col_nnz > 0
alive_indices = np.where(alive_mask)[0]    # original SAE feature indices

X_alive = X_ctx[:, alive_indices]
n_dead = d_sae - len(alive_indices)
print(f"Alive neurons: {len(alive_indices)} / {d_sae}  (removed {n_dead} dead, {n_dead/d_sae*100:.1f}%)")
print(f"Data shape after pruning: {X_alive.shape}")

# mapping: new column index -> original SAE feature index
new_to_old = dict(enumerate(alive_indices.tolist()))

# data stats
row_nnz = np.diff(X_alive.indptr)
row_l2 = np.sqrt(X_alive.multiply(X_alive).sum(axis=1)).A1
print("row nnz:  min/med/p95", np.min(row_nnz), np.median(row_nnz), np.percentile(row_nnz,95))
print("row l2 :  min/med/p95", np.min(row_l2), np.median(row_l2), np.percentile(row_l2,95))
print("zeros rows:", np.sum(row_nnz == 0))

# %% load feature names from neuronpedia explanations
print("Loading feature names...")
with open("data/gemmascope_explanations/ex/gemma3_4b_it_layer22_16k_explanations.json") as f:
    explanations_list = json.load(f)

explanation_map = {int(item["index"]): item["description"] for item in explanations_list}

def get_feature_name(sae_idx):
    """Human-readable name for an SAE feature index."""
    if sae_idx in explanation_map:
        return f"f_{sae_idx}: {explanation_map[sae_idx]}"
    return f"f_{sae_idx}"

alive_feature_names = [get_feature_name(idx) for idx in alive_indices]
n_missing = sum(1 for n in alive_feature_names if ":" not in n)
print(f"Feature names: {len(alive_feature_names)}  ({n_missing} missing explanations)")

# X_alive = X_alive.astype("float32", copy=False)

# %% Truncated SVD dimensionality reduction
print(f"Running TruncatedSVD to {SVD_DIMS} dimensions...")
#svd = TruncatedSVD(n_components=SVD_DIMS, random_state=42)
#X_svd = svd.fit_transform(X_alive)
#explained_var = svd.explained_variance_ratio_.sum()
#print(f"SVD shape: {X_svd.shape}")
#print(f"Explained variance: {explained_var:.4f} ({explained_var*100:.2f}%)")
#print(f"Top 10 singular values: {svd.singular_values_[:10].round(2)}")

svd = TruncatedSVD(n_components=SVD_DIMS, random_state=42)
norm = Normalizer(copy=False)  # L2 normalize rows
X_svd = make_pipeline(svd, norm).fit_transform(X_alive)
print("Explained variance:", svd.explained_variance_ratio_.sum())

# %% HDBSCAN clustering (on SVD-reduced space)
print(f"Running HDBSCAN on {SVD_DIMS}D SVD space...")
clusterer = hdbscan.HDBSCAN(
    algorithm="boruvka_kdtree",
    min_cluster_size=20,
    min_samples=5,
    metric="euclidean",
    cluster_selection_method="eom",
    gen_min_span_tree=False,
    prediction_data=False
)
labels = clusterer.fit_predict(X_svd)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = np.sum(labels == -1)
print(f"Clusters: {n_clusters},  Noise points: {n_noise} ({n_noise/len(labels)*100:.1f}%)")
for c in sorted(set(labels) - {-1}):
    print(f"  Cluster {c}: {np.sum(labels == c)} samples")

# %% 2D projections: SVD top-2, UMAP, t-SNE

# Method 1: Top 2 SVD dimensions (already computed)
print("Using top 2 SVD dimensions for first projection...")
X_svd2 = X_svd[:, :2]

# Method 2: UMAP on SVD space
print("Running UMAP (2D) on SVD space...")
reducer_umap = umap.UMAP(
    n_components=2,
    metric="cosine",
    n_neighbors=30,
    min_dist=0.1,
    random_state=42,
    verbose=True,
)
X_umap = reducer_umap.fit_transform(X_svd)
print(f"UMAP 2D shape: {X_umap.shape}")

# Method 3: t-SNE on SVD space
print("Running t-SNE (2D) on SVD space...")
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    random_state=42,
    verbose=1,
    n_jobs=-1,
)
X_tsne = tsne.fit_transform(X_svd)
print(f"t-SNE 2D shape: {X_tsne.shape}")

# %% plot clusters — 3 methods side by side
print("Plotting clusters (3 methods)...")

embeddings = [
    ("SVD (dim 1 & 2)", X_svd2, "SVD 1", "SVD 2"),
    ("UMAP", X_umap, "UMAP 1", "UMAP 2"),
    ("t-SNE", X_tsne, "t-SNE 1", "t-SNE 2"),
]

fig, axes = plt.subplots(1, 3, figsize=(24, 7), dpi=150)

noise_mask = labels == -1
cluster_mask = labels >= 0

for ax, (title, X_2d, xlabel, ylabel) in zip(axes, embeddings):
    # noise in light gray
    if noise_mask.any():
        ax.scatter(X_2d[noise_mask, 0], X_2d[noise_mask, 1],
                   c="lightgray", s=1, alpha=0.3, label="noise", rasterized=True)

    # clusters coloured
    scatter = ax.scatter(
        X_2d[cluster_mask, 0], X_2d[cluster_mask, 1],
        c=labels[cluster_mask], cmap="tab20", s=3, alpha=0.6, rasterized=True,
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", markerscale=5)

fig.colorbar(scatter, ax=axes, label="Cluster", shrink=0.8)
fig.suptitle(f"HDBSCAN Clusters on SVD-{SVD_DIMS}D Context Vectors ({n_clusters} clusters)", fontsize=16, fontweight="bold")
fig.tight_layout(rect=[0, 0, 0.95, 0.95])
fig.savefig("tree_outputs/context_clusters_3methods.png", dpi=150, bbox_inches="tight")
print("Saved plot to tree_outputs/context_clusters_3methods.png")
plt.show()

# %% per-cluster centroids & distinctive features
print("Computing cluster centroids and distinctive features...")

# Use sparse .mean() to stay memory-efficient
global_mean = np.asarray(X_alive.mean(axis=0)).flatten()

cluster_ids = sorted(set(labels) - {-1})
cluster_results = {}

for c in cluster_ids:
    mask = labels == c
    cluster_mean = np.asarray(X_alive[mask].mean(axis=0)).flatten()

    # distinctiveness: how much more this feature fires in this cluster vs. globally
    diff = cluster_mean - global_mean

    top_new_idx = np.argsort(diff)[::-1][:TOP_K_FEATURES]

    features = []
    for ni in top_new_idx:
        old_idx = new_to_old[int(ni)]
        name = get_feature_name(old_idx)
        features.append({
            "name": name,
            "sae_idx": old_idx,
            "diff": float(diff[ni]),
            "cluster_mean": float(cluster_mean[ni]),
            "global_mean": float(global_mean[ni]),
        })
    
    c1 = int(c)
    cluster_results[c1] = {
        "size": int(mask.sum()),
        "distinctive_features": features,
    }

# %% print distinctive features per cluster
separator = "=" * 90
output_lines = []

for c in cluster_ids:
    res = cluster_results[c]
    header = f"\n{separator}\nCluster {c}  ({res['size']} samples)\n{separator}"
    print(header)
    output_lines.append(header)

    for feat in res["distinctive_features"]:
        line = (
            f"  [{feat['diff']:+.4f}]  "
            f"(cluster={feat['cluster_mean']:.4f}, global={feat['global_mean']:.4f})  "
            f"{feat['name']}"
        )
        print(line)
        output_lines.append(line)

# %% save to file
output_path = Path("tree_outputs/cluster_distinctive_features.txt")
output_path.write_text("\n".join(output_lines) + "\n")
print(f"\nSaved distinctive features to {output_path}")

# also save full results as JSON for downstream use
json_path = Path("tree_outputs/cluster_results.json")
json_path.write_text(json.dumps(cluster_results, indent=2))
print(f"Saved cluster results JSON to {json_path}")

