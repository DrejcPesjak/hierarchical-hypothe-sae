# %% imports
import numpy as np
import json
from pathlib import Path
from scipy.sparse import csr_matrix, vstack
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

from mob_tree import MOBTree
from mob_tree import plot_mob_tree

# ============================================================================
# %% Configuration
# ============================================================================
d_sae = 16384
N = None                        # set to int to limit batches (debugging)
OUTPUT_DIR = Path("mob_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# %% Load data
# ============================================================================
print("Loading data...")
batch_dir = Path("./data/sae_vectors")
sparse_files = sorted(
    batch_dir.glob("confirmatory_preprocessed_sae_vectors_sparse_minibatch_*.npz")
)
meta_files = sorted(
    batch_dir.glob("confirmatory_preprocessed_sae_vectors_1meta_minibatch_*.npz")
)
sparse_files = sparse_files[:N] if N is not None else sparse_files
meta_files = meta_files[:N] if N is not None else meta_files

# sparse vectors  [context | diff]  (each row = 2 * d_sae wide)
Xs = []
for f in sparse_files:
    z = np.load(f)
    Xs.append(
        csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    )
X_full = vstack(Xs, format="csr")
X_context = X_full[:, :d_sae]
X_diff = X_full[:, d_sae:]
del X_full

# enriched metadata
test_ids_list, target_list, weight_list = [], [], []
for f in meta_files:
    m = np.load(f, allow_pickle=True)
    test_ids_list.append(m["test_id"])
    target_list.append(m["target_logit_diff"])
    weight_list.append(m["weight_sqrt_min_impr"])

test_ids = np.concatenate(test_ids_list)
y_target = np.concatenate(target_list)
sample_weight = np.concatenate(weight_list)

print(f"Context:  {X_context.shape}")
print(f"Diff:     {X_diff.shape}")
print(f"Samples:  {len(y_target)}")
print(f"Unique test_ids: {len(np.unique(test_ids))}")

# ============================================================================
# %% Data stats
# ============================================================================
print(f"\ny_target stats:  "
      f"range=[{y_target.min():.4f}, {y_target.max():.4f}]  "
      f"mean={y_target.mean():.4f}  std={y_target.std():.4f}")
print(f"sample_weight stats:  "
      f"range=[{sample_weight.min():.2f}, {sample_weight.max():.2f}]  "
      f"mean={sample_weight.mean():.2f}")

# ============================================================================
# %% Feature names
# ============================================================================
print("\nLoading feature names...")
with open(
    "data/gemmascope_explanations/ex/gemma3_4b_it_layer22_16k_explanations.json"
) as fh:
    explanations_list = json.load(fh)

explanation_map = {int(item["index"]): item["description"] for item in explanations_list}


def _fname(sae_idx, prefix=""):
    desc = explanation_map.get(sae_idx, "")
    return f"{prefix}f{sae_idx}: {desc}" if desc else f"{prefix}f{sae_idx}"


ctx_names = [_fname(i, "ctx_") for i in range(d_sae)]
diff_names = [_fname(i, "Δ_") for i in range(d_sae)]

# ============================================================================
# %% Train / test split (by test_id — no leakage)
# ============================================================================
print("\nSplitting data by test_id...")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X_diff, y_target, groups=test_ids))

X_diff_train, X_diff_test = X_diff[train_idx], X_diff[test_idx]
X_ctx_train, X_ctx_test = X_context[train_idx], X_context[test_idx]
y_train, y_test = y_target[train_idx], y_target[test_idx]
w_train, w_test = sample_weight[train_idx], sample_weight[test_idx]

print(f"Train: {len(train_idx)},  Test: {len(test_idx)}")
assert len(set(test_ids[train_idx]) & set(test_ids[test_idx])) == 0, "Leakage!"

# ============================================================================
# %% Scale features  (StandardScaler, with_mean=False for sparse)
# ============================================================================
print("Scaling features...")
scaler_diff = StandardScaler(with_mean=False)
X_diff_train = scaler_diff.fit_transform(X_diff_train)
X_diff_test = scaler_diff.transform(X_diff_test)

scaler_ctx = StandardScaler(with_mean=False)
X_ctx_train = scaler_ctx.fit_transform(X_ctx_train)
X_ctx_test = scaler_ctx.transform(X_ctx_test)

# ============================================================================
# %% Global Lasso baseline
# ============================================================================
print("\n" + "=" * 70)
print("GLOBAL BASELINE — Lasso on X_diff → y_target")
print("=" * 70)

lasso_global = Lasso(alpha=0.01, max_iter=3000, random_state=42)
lasso_global.fit(X_diff_train, y_train, sample_weight=w_train)

y_pred_gl_train = lasso_global.predict(X_diff_train)
y_pred_gl_test = lasso_global.predict(X_diff_test)

n_nz_global = int(np.sum(lasso_global.coef_ != 0))
r2_gl_train = r2_score(y_train, y_pred_gl_train)
r2_gl_test = r2_score(y_test, y_pred_gl_test)
mae_gl_test = mean_absolute_error(y_test, y_pred_gl_test)
rmse_gl_test = float(np.sqrt(mean_squared_error(y_test, y_pred_gl_test)))

print(f"  Non-zero coefs: {n_nz_global} / {len(lasso_global.coef_)}")
print(f"  Train R²: {r2_gl_train:.4f}")
print(f"  Test  R²: {r2_gl_test:.4f}   MAE: {mae_gl_test:.4f}   RMSE: {rmse_gl_test:.4f}")

# ============================================================================
# %% Fit MOB tree
# ============================================================================
print("\n" + "=" * 70)
print("FITTING MOB TREE")
print("=" * 70)

mob = MOBTree(
    max_depth=4,
    min_samples_leaf=200,
    min_samples_split=500,
    instability_alpha=0.05,
    trim=0.1,
    screen_k=200,
    max_score_dim=20,
    # post_lasso_ols=True,        # refit OLS on active set for valid scores
    # n_sim=5000,                 # Monte-Carlo draws for supLM null
    inner_model_class=Lasso,
    inner_model_params=dict(alpha=0.01, max_iter=3000, random_state=42),
    # inner_model_class=Ridge,
    # inner_model_params=dict(alpha=1.0, solver="sag", max_iter=3000, random_state=42),
    verbose=True,
)
mob.fit(X_diff_train, X_ctx_train, y_train, sample_weight=w_train)

# ============================================================================
# %% Evaluate MOB  vs  Global
# ============================================================================
print("\n" + "=" * 70)
print("EVALUATION — MOB vs Global Lasso")
print("=" * 70)

mob_pred_train, leaf_ids_train = mob.predict(X_diff_train, X_ctx_train)
mob_pred_test, leaf_ids_test = mob.predict(X_diff_test, X_ctx_test)

r2_mob_train = r2_score(y_train, mob_pred_train)
r2_mob_test = r2_score(y_test, mob_pred_test)
mae_mob_test = mean_absolute_error(y_test, mob_pred_test)
rmse_mob_test = float(np.sqrt(mean_squared_error(y_test, mob_pred_test)))

print(f"\n{'Model':<20} {'Train R²':>10} {'Test R²':>10} {'Test MAE':>10} {'Test RMSE':>10}")
print("-" * 62)
print(f"{'Global Lasso':<20} {r2_gl_train:>10.4f} {r2_gl_test:>10.4f} "
      f"{mae_gl_test:>10.4f} {rmse_gl_test:>10.4f}")
print(f"{'MOB Tree':<20} {r2_mob_train:>10.4f} {r2_mob_test:>10.4f} "
      f"{mae_mob_test:>10.4f} {rmse_mob_test:>10.4f}")
print(f"\nΔ Test R²  = {r2_mob_test - r2_gl_test:+.4f}")

# ============================================================================
# %% Print tree structure
# ============================================================================
print("\n" + "=" * 70)
print("MOB TREE STRUCTURE")
print("=" * 70)
mob.print_tree(ctx_names=ctx_names, diff_names=diff_names, top_k=5)

# ============================================================================
# %% Plot tree
# ============================================================================
fig = plot_mob_tree(mob, ctx_names=ctx_names, diff_names=diff_names, top_k=3)
fig.savefig(OUTPUT_DIR / "mob_tree.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: {OUTPUT_DIR / 'mob_tree.png'}")
plt.show()

# ============================================================================
# %% Per-leaf analysis
# ============================================================================
print("\n" + "=" * 70)
print("PER-LEAF ANALYSIS")
print("=" * 70)

leaves = mob.get_leaves()
paths = mob.leaf_paths(ctx_names=ctx_names)

print(f"\n{'Leaf':>6} {'N train':>8} {'N test':>8} {'R²(train)':>10} "
      f"{'R²(test)':>10} {'ȳ(train)':>10} {'NZ coefs':>9}")
print("-" * 70)

for lf in leaves:
    train_mask = leaf_ids_train == lf.node_id
    test_mask = leaf_ids_test == lf.node_id
    n_tr = train_mask.sum()
    n_te = test_mask.sum()

    # per-leaf test R²
    if n_te > 10:
        r2_lf = r2_score(y_test[test_mask], mob_pred_test[test_mask])
    else:
        r2_lf = float("nan")

    n_nz = int(np.sum(lf.model.coef_ != 0))
    print(
        f"  {lf.node_id:>4d} {n_tr:>8d} {n_te:>8d} {lf.train_r2:>10.4f} "
        f"{r2_lf:>10.4f} {lf.mean_y:>+10.4f} {n_nz:>9d}"
    )

# leaf paths + top features
for lf in leaves:
    print(f"\n{'─' * 60}")
    print(f"Leaf {lf.node_id}  (n={lf.n_samples})")
    path = paths[lf.node_id]
    print("  Path from root:")
    for fname, direction, thresh in path:
        print(f"    {fname[:55]} {direction} {thresh:.4f}")

    coef = lf.model.coef_
    nz = np.where(coef != 0)[0]
    if len(nz):
        order = nz[np.argsort(np.abs(coef[nz]))[::-1]]
        print(f"  Top diff features ({len(nz)} non-zero):")
        for j in order[:8]:
            print(f"    [{coef[j]:+.6f}] {diff_names[j][:60]}")

# ============================================================================
# %% Coefficient comparison heatmap (Global vs Leaves)
# ============================================================================
print("\n" + "=" * 70)
print("COEFFICIENT COMPARISON")
print("=" * 70)

# gather union of non-zero features across global + all leaves
all_feats = set(np.where(lasso_global.coef_ != 0)[0])
for lf in leaves:
    all_feats.update(np.where(lf.model.coef_ != 0)[0])

feat_list = sorted(all_feats)
n_feats_total = len(feat_list)
n_models = 1 + len(leaves)

# build coefficient matrix  (features × models)
coef_matrix = np.zeros((n_feats_total, n_models))
coef_matrix[:, 0] = [lasso_global.coef_[f] for f in feat_list]
for j, lf in enumerate(leaves):
    coef_matrix[:, j + 1] = [lf.model.coef_[f] for f in feat_list]

# select top features by cross-model variance (most heterogeneous)
coef_var = coef_matrix.var(axis=1)
n_show = min(30, n_feats_total)
top_var_idx = np.argsort(coef_var)[::-1][:n_show]

mat_show = coef_matrix[top_var_idx]
feat_labels = [diff_names[feat_list[i]][:45] for i in top_var_idx]
model_labels = ["Global"] + [f"Leaf {lf.node_id}\n(n={lf.n_samples})" for lf in leaves]

fig, ax = plt.subplots(
    figsize=(max(8, n_models * 1.2), max(6, n_show * 0.35))
)
vmax = np.abs(mat_show).max()
im = ax.imshow(mat_show, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
ax.set_yticks(range(n_show))
ax.set_yticklabels(feat_labels, fontsize=6)
ax.set_xticks(range(n_models))
ax.set_xticklabels(model_labels, fontsize=7, rotation=45, ha="right")
plt.colorbar(im, ax=ax, label="Lasso Coefficient", shrink=0.8)
ax.set_title(
    f"Top {n_show} most heterogeneous features (by cross-model variance)", fontsize=11
)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "mob_coef_heatmap.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR / 'mob_coef_heatmap.png'}")
plt.show()

# ============================================================================
# %% Scatter: Global vs MOB predictions (test set)
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Global predicted vs actual
axes[0].scatter(y_pred_gl_test, y_test, alpha=0.1, s=2, rasterized=True)
lo, hi = y_test.min(), y_test.max()
axes[0].plot([lo, hi], [lo, hi], "r--", lw=1)
axes[0].set_xlabel("Predicted (Global Lasso)")
axes[0].set_ylabel("Actual")
axes[0].set_title(f"Global Lasso  (R²={r2_gl_test:.4f})")

# MOB predicted vs actual
axes[1].scatter(mob_pred_test, y_test, alpha=0.1, s=2, rasterized=True)
axes[1].plot([lo, hi], [lo, hi], "r--", lw=1)
axes[1].set_xlabel("Predicted (MOB)")
axes[1].set_ylabel("Actual")
axes[1].set_title(f"MOB Tree  (R²={r2_mob_test:.4f})")

# Residual improvement histogram
resid_gl = y_test - y_pred_gl_test
resid_mob = y_test - mob_pred_test
axes[2].hist(np.abs(resid_gl), bins=60, alpha=0.5, label="Global", color="tab:blue")
axes[2].hist(np.abs(resid_mob), bins=60, alpha=0.5, label="MOB", color="tab:orange")
axes[2].set_xlabel("|Residual|")
axes[2].set_ylabel("Count")
axes[2].set_title("Absolute residual distribution (test)")
axes[2].legend()

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "mob_eval_plots.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR / 'mob_eval_plots.png'}")
plt.show()

# ============================================================================
# %% Per-leaf segment sizes + mean y bar chart
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

leaf_node_ids = [lf.node_id for lf in leaves]
seg_sizes = [np.sum(leaf_ids_train == lid) for lid in leaf_node_ids]
seg_means = [y_train[leaf_ids_train == lid].mean() for lid in leaf_node_ids]
seg_stds = [y_train[leaf_ids_train == lid].std() for lid in leaf_node_ids]
x_pos = range(len(leaves))

axes[0].bar(x_pos, seg_sizes, color="steelblue", edgecolor="black", lw=0.5)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels([f"Leaf {lid}" for lid in leaf_node_ids], fontsize=8, rotation=45)
axes[0].set_ylabel("Count")
axes[0].set_title("Leaf sizes (train)")
for i, v in enumerate(seg_sizes):
    axes[0].text(i, v + max(seg_sizes) * 0.01, str(v), ha="center", fontsize=7)

colors = ["tab:green" if m > 0 else "tab:red" for m in seg_means]
axes[1].bar(
    x_pos, seg_means, yerr=seg_stds, color=colors,
    edgecolor="black", lw=0.5, capsize=3, alpha=0.8,
)
axes[1].axhline(0, color="black", ls="--", lw=0.5)
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels([f"Leaf {lid}" for lid in leaf_node_ids], fontsize=8, rotation=45)
axes[1].set_ylabel("Mean y_target")
axes[1].set_title("Mean y_target per leaf (train)")

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "mob_leaf_stats.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR / 'mob_leaf_stats.png'}")
plt.show()

# ============================================================================
# %% Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Samples:                {len(y_target)}")
print(f"  Global Lasso NZ coefs:  {n_nz_global}")
print(f"  Global test R²:         {r2_gl_test:.4f}")
print(f"  MOB nodes / leaves:     {mob._cnt} / {len(leaves)}")
print(f"  MOB test R²:            {r2_mob_test:.4f}")
print(f"  Δ R² (MOB − Global):    {r2_mob_test - r2_gl_test:+.4f}")
print()
for lf in leaves:
    nz = int(np.sum(lf.model.coef_ != 0))
    print(
        f"    Leaf {lf.node_id:>3d}: n={lf.n_samples:>5d}  "
        f"R²={lf.train_r2:.4f}  ȳ={lf.mean_y:+.4f}  NZ={nz}"
    )

