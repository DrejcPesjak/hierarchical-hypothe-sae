# %% import libraries
"""
Analyse specific SAE features in context and diff vectors.
For each feature:
  - Distribution plots (raw + StandardScaler no-mean) for context
  - Top/bottom activating headlines
  - Diff distribution with automatic threshold methods
"""
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.sparse import csr_matrix, vstack
from sklearn.preprocessing import StandardScaler
from scipy.signal import argrelextrema
from sklearn.mixture import GaussianMixture

# %% configuration
d_sae = 16384
# FEATURES = [1497, 11052, 1730, 1531, 604, 11540]
FEATURES = [433, 2220, 9410, 1170, 70, 507]
# Manual thresholds (on standardised diff); None = no manual line
# MANUAL_TH_STD = [-0.561, -0.0125, -0.0522, -0.208, 0.3369, None]
MANUAL_TH_STD = [None, None, None, None, None, None]

CSV_PATH = Path("data/cleaned_archive/confirmatory_preprocessed.csv")
EXPL_PATH = Path("data/gemmascope_explanations/ex/gemma3_4b_it_layer22_16k_explanations.json")
OUT_DIR = Path("tree_outputs1")
OUT_DIR.mkdir(exist_ok=True)

# %% load sparse vectors
print("Loading sparse vectors …")
batch_dir = Path("./data/sae_vectors")
sparse_files = sorted(batch_dir.glob(
    "confirmatory_preprocessed_sae_vectors_sparse_minibatch_*.npz"))
meta_files = sorted(batch_dir.glob(
    "confirmatory_preprocessed_sae_vectors_meta_minibatch_*.npz"))

Xs = []
for f in sparse_files:
    z = np.load(f)
    mat = csr_matrix((z["data"], z["indices"], z["indptr"]),
                     shape=tuple(z["shape"]))
    Xs.append(mat)

X_full = vstack(Xs, format="csr")
print(f"Full matrix: {X_full.shape[0]} × {X_full.shape[1]}")

# %% load metadata (test_id, id_first, id_second for headline lookup)
print("Loading metadata …")
test_ids_list, id_first_list, id_second_list = [], [], []
for f in meta_files:
    m = np.load(f, allow_pickle=True)
    test_ids_list.append(m["test_id"])
    id_first_list.append(m["id_first"])
    id_second_list.append(m["id_second"])

all_test_ids = np.concatenate(test_ids_list)
all_id_first = np.concatenate(id_first_list)
all_id_second = np.concatenate(id_second_list)

# %% build headline lookup from CSV
print("Loading CSV for headline lookup …")
df = pd.read_csv(CSV_PATH)
headline_lookup = {}  # (test_id, local_idx) -> headline string
for test_id, group in df.groupby("clickability_test_id"):
    group = group.reset_index(drop=True)
    for local_idx in range(len(group)):
        headline_lookup[(str(test_id), int(local_idx))] = str(
            group.loc[local_idx, "headline"])

# %% load SAE feature explanations
print("Loading SAE explanations …")
with open(EXPL_PATH) as f:
    explanations_list = json.load(f)
explanation_map = {int(item["index"]): item["description"]
                   for item in explanations_list}


def feat_label(idx):
    desc = explanation_map.get(idx, "")
    short = desc[:80] if desc else ""
    return f"SAE {idx}: {short}" if short else f"SAE {idx}"


# %% helper – get headline for row i (first headline)
def get_headline_first(i):
    return headline_lookup.get(
        (str(all_test_ids[i]), int(all_id_first[i])), "<unknown>")


def get_headline_second(i):
    return headline_lookup.get(
        (str(all_test_ids[i]), int(all_id_second[i])), "<unknown>")


# ========================================================================
# CONTEXT VECTORS  (columns 0 … d_sae-1)
# ========================================================================
# %% extract context columns for our features
print("\n" + "=" * 80)
print("CONTEXT FEATURES")
print("=" * 80)

ctx_cols = np.array(FEATURES)
X_ctx = X_full[:, ctx_cols].toarray()  # dense (n, 6)

# StandardScaler (no mean)
scaler = StandardScaler(with_mean=False)
X_ctx_std = scaler.fit_transform(X_ctx)

# %% context threshold methods -------------------------------------------
def ctx_method1_kde_valley(values, n_bins=200):
    """KDE valley on non-zero context values (1 threshold)."""
    hist, edges = np.histogram(values, bins=n_bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    kernel = np.ones(9) / 9
    smooth = np.convolve(hist, kernel, mode="same")
    mins = argrelextrema(smooth, np.less, order=8)[0]
    if len(mins) == 0:
        return None
    # pick the deepest minimum
    best = mins[np.argmin(smooth[mins])]
    return centres[best]


def ctx_method2_gmm2(values):
    """2-component GMM: 'barely-on' exponential-ish vs 'real' Gaussian.
    Threshold = midpoint between the two means."""
    gmm = GaussianMixture(n_components=2, random_state=42, max_iter=300)
    gmm.fit(values.reshape(-1, 1))
    means = np.sort(gmm.means_.ravel())
    stds = np.sqrt(gmm.covariances_.ravel())
    # order stds to match sorted means
    order = np.argsort(gmm.means_.ravel())
    stds = stds[order]
    # threshold: upper edge of lower component (mean + 2σ),
    # or midpoint, whichever is smaller
    return min(means[0] + 2 * stds[0], 0.5 * (means[0] + means[1]))


def ctx_method3_iqr_elbow(values):
    """Simple: threshold at Q1 of non-zero values (the 'elbow' where the
    exponential onset meets the Gaussian body)."""
    return np.percentile(values, 25)


def ctx_method4_walk_right(values, n_bins=200, bb=4):
    """Walk from the left (smallest non-zero values) rightward.
    If the next `bb` bars are ALL higher than the current bar,
    that's where the exponential tail transitions into the Gaussian body.
    Returns the threshold (single positive value)."""
    hist, edges = np.histogram(values, bins=n_bins)
    centres = 0.5 * (edges[:-1] + edges[1:])

    for i in range(0, len(hist) - bb):
        if all(hist[i + j] > hist[i] for j in range(1, bb + 1)):
            return centres[i]
    return None


CTX_COLORS = {1: "red", 2: "green", 3: "purple", 4: "deepskyblue"}
CTX_LABELS = {1: "KDE valley", 2: "GMM-2 split", 3: "Q1 elbow", 4: "Walk right"}

# %% plot context distributions
fig, axes = plt.subplots(len(FEATURES), 2, figsize=(16, 4.5 * len(FEATURES)))

for row, (fidx, ax_pair) in enumerate(zip(FEATURES, axes)):
    raw = X_ctx[:, row]
    std = X_ctx_std[:, row]
    nonzero_raw = raw[raw != 0]
    nonzero_std = std[std != 0]
    print(f"{fidx}: {nonzero_raw.shape[0]} / {raw.shape[0]}")

    # compute context thresholds on raw non-zero values
    cth1 = ctx_method1_kde_valley(nonzero_raw)
    cth2 = ctx_method2_gmm2(nonzero_raw)
    cth3 = ctx_method3_iqr_elbow(nonzero_raw)
    cth4 = ctx_method4_walk_right(nonzero_raw)
    # convert to std scale
    ctx_scale = scaler.scale_[row]
    cth1_std = cth1 / ctx_scale if cth1 is not None else None
    cth2_std = cth2 / ctx_scale
    cth3_std = cth3 / ctx_scale
    cth4_std = cth4 / ctx_scale if cth4 is not None else None

    desc = explanation_map.get(fidx, "")

    # --- raw ---
    ax = ax_pair[0]
    ax.hist(nonzero_raw, bins=100, alpha=0.7, color="steelblue", edgecolor="none")
    if cth1 is not None:
        ax.axvline(cth1, color=CTX_COLORS[1], ls="-", lw=1.2, label=CTX_LABELS[1])
    ax.axvline(cth2, color=CTX_COLORS[2], ls="--", lw=1.2, label=CTX_LABELS[2])
    ax.axvline(cth3, color=CTX_COLORS[3], ls=":", lw=1.5, label=CTX_LABELS[3])
    if cth4 is not None:
        ax.axvline(cth4, color=CTX_COLORS[4], ls=(0, (3, 1, 1, 1)), lw=1.3, label=CTX_LABELS[4])
    ax.set_title(f"SAE {fidx} — raw (non-zero)", fontsize=10, pad=6)
    ax.set_xlabel("activation", fontsize=9)
    ax.set_ylabel("count", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    if desc:
        ax.text(0.02, 0.95, desc[:90], transform=ax.transAxes,
                fontsize=7, va="top", ha="left", color="0.3",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    # --- std ---
    ax = ax_pair[1]
    ax.hist(nonzero_std, bins=100, alpha=0.7, color="darkorange", edgecolor="none")
    if cth1_std is not None:
        ax.axvline(cth1_std, color=CTX_COLORS[1], ls="-", lw=1.2, label=CTX_LABELS[1])
    ax.axvline(cth2_std, color=CTX_COLORS[2], ls="--", lw=1.2, label=CTX_LABELS[2])
    ax.axvline(cth3_std, color=CTX_COLORS[3], ls=":", lw=1.5, label=CTX_LABELS[3])
    if cth4_std is not None:
        ax.axvline(cth4_std, color=CTX_COLORS[4], ls=(0, (3, 1, 1, 1)), lw=1.3, label=CTX_LABELS[4])
    ax.set_title(f"SAE {fidx} — scaled (non-zero)", fontsize=10, pad=6)
    ax.set_xlabel("activation (scaled)", fontsize=9)
    ax.set_ylabel("count", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    if desc:
        ax.text(0.02, 0.95, desc[:90], transform=ax.transAxes,
                fontsize=7, va="top", ha="left", color="0.3",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

fig.subplots_adjust(hspace=0.45, top=0.97)
fig.savefig(OUT_DIR / "feature_ctx_distributions.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved {OUT_DIR / 'feature_ctx_distributions.png'}")

# %% print headline examples per context feature (walk-right threshold)
print("\n" + "=" * 80)
print("CONTEXT HEADLINE EXAMPLES  (walk-right threshold)")
print("=" * 80)

for col_i, fidx in enumerate(FEATURES):
    vals = X_ctx[:, col_i]
    nonzero_raw = vals[vals != 0]
    th = ctx_method4_walk_right(nonzero_raw)
    print(f"\n--- {feat_label(fidx)}  (threshold={th:.4f} raw)" if th is not None
          else f"\n--- {feat_label(fidx)}  (no threshold found)")

    # Top 3 most activating
    order_desc = np.argsort(vals)[::-1]
    print("  TOP 3 most activating:")
    shown = 0
    for idx in order_desc:
        if shown >= 3:
            break
        v = vals[idx]
        h1 = get_headline_first(idx)
        h2 = get_headline_second(idx)
        print(f"    [{v:.4f}]  H1: {h1}")
        print(f"              H2: {h2}")
        shown += 1

    # 3 examples just below the threshold (largest values still under th)
    if th is not None:
        below_mask = (vals > 0) & (vals < th)
        if below_mask.sum() >= 3:
            below_idx = np.where(below_mask)[0]
            below_vals = vals[below_idx]
            # sort descending — closest to threshold first
            order = np.argsort(below_vals)[::-1]
            print(f"  3 just BELOW threshold ({th:.4f}):")
            for k in range(min(3, len(order))):
                idx = below_idx[order[k]]
                v = vals[idx]
                h1 = get_headline_first(idx)
                h2 = get_headline_second(idx)
                print(f"    [{v:.4f}]  H1: {h1}")
                print(f"              H2: {h2}")
        else:
            print(f"  (fewer than 3 examples below threshold)")


# ========================================================================
# DIFF VECTORS  (columns d_sae … 2*d_sae-1)
# ========================================================================
# %% extract diff columns
print("\n" + "=" * 80)
print("DIFF FEATURES")
print("=" * 80)

diff_cols = np.array([d_sae + f for f in FEATURES])
X_diff = X_full[:, diff_cols].toarray()

scaler_diff = StandardScaler(with_mean=False)
X_diff_std = scaler_diff.fit_transform(X_diff)

# %% diff threshold methods -----------------------------------------------
# The diff distribution (non-zero) is a "trident":
#   - Central Gaussian centred at 0  (pair had no difference on this feature)
#   - Positive exponential tail at +μ  (first headline activated more)
#   - Mirror exponential tail at −μ   (second headline activated more)
# There are clear gaps between the centre and the tails.

def diff_method1_kde_valleys(values, n_bins=300):
    """
    Method 1 – KDE valleys (improved).
    Histogram → smooth → find local minima.
    Exploits symmetry: if only one good valley is found, mirror it.
    """
    hist, edges = np.histogram(values, bins=n_bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    kernel = np.ones(9) / 9
    smooth = np.convolve(hist, kernel, mode="same")

    mins = argrelextrema(smooth, np.less, order=10)[0]
    if len(mins) == 0:
        return []

    # sort by depth (shallowest count = deepest valley)
    sorted_mins = mins[np.argsort(smooth[mins])]

    # try to pick one negative and one positive valley
    neg = [centres[m] for m in sorted_mins if centres[m] < 0]
    pos = [centres[m] for m in sorted_mins if centres[m] > 0]

    if neg and pos:
        return [neg[0], pos[0]]  # deepest on each side
    elif neg:
        return [neg[0], -neg[0]]  # mirror
    elif pos:
        return [-pos[0], pos[0]]
    return []


# def diff_method2_symmetric_gap(values, n_bins=300):
#     """
#     Method 2 – Symmetric gap finder.
#     Fold |values|, find the valley between the centre half-Gaussian
#     and the exponential tail, then mirror to ±.
#     """
#     absv = np.abs(values)
#     hist, edges = np.histogram(absv, bins=n_bins)
#     centres = 0.5 * (edges[:-1] + edges[1:])
#     kernel = np.ones(9) / 9
#     smooth = np.convolve(hist, kernel, mode="same")
#
#     mins = argrelextrema(smooth, np.less, order=8)[0]
#     if len(mins) == 0:
#         peak = smooth.max()
#         below = np.where(smooth < 0.05 * peak)[0]
#         if len(below) == 0:
#             return []
#         gap = centres[below[0]]
#         return [-gap, gap]
#
#     best = mins[np.argmin(smooth[mins])]
#     gap = centres[best]
#     return [-gap, gap]


def diff_method3_central_sigma(values):
    """
    Method 3 – Central Gaussian σ estimate.
    The centre peak is roughly Gaussian around 0.
    Estimate its σ from the IQR of values near the median,
    then set thresholds at ±3σ.
    """
    # use the middle 50% of values to estimate the central peak's spread
    q25, q75 = np.percentile(values, [25, 75])
    iqr = q75 - q25
    # for a Gaussian, IQR ≈ 1.349 σ
    sigma_est = iqr / 1.349
    return [-3 * sigma_est, 3 * sigma_est]


def diff_method4_walk_outward(values, n_bins=300, bb=4):
    """
    Method 4 – Walk outward from 0.
    Build a histogram, find the bin closest to 0, then walk right.
    If the next `bb` bars are ALL higher than the current bar,
    place the threshold here (= start of the rising tail).
    Mirror to negative side.
    Returns [neg_threshold, pos_threshold].
    """
    hist, edges = np.histogram(values, bins=n_bins)
    centres = 0.5 * (edges[:-1] + edges[1:])

    # find the bin closest to 0
    zero_idx = np.argmin(np.abs(centres))

    # --- walk RIGHT (positive direction) ---
    pos_th = None
    for i in range(zero_idx, len(hist) - bb):
        if all(hist[i + j] > hist[i] for j in range(1, bb + 1)):
            pos_th = centres[i]
            break

    if pos_th is None:
        return []

    return [-pos_th, pos_th]


# %% plot diff distributions with thresholds
DIFF_COLORS = {1: "red", 3: "purple", 4: "deepskyblue"}
DIFF_LABELS = {1: "KDE valleys", 3: "Central 3σ", 4: "Walk outward"}

fig, axes = plt.subplots(len(FEATURES), 2, figsize=(18, 5 * len(FEATURES)))

for row, (fidx, manual_th, ax_pair) in enumerate(
        zip(FEATURES, MANUAL_TH_STD, axes)):

    raw = X_diff[:, row]
    std = X_diff_std[:, row]

    # Remove absolute zeros
    nz_mask = raw != 0
    raw_nz = raw[nz_mask]
    std_nz = std[nz_mask]

    # --- compute thresholds on STANDARDISED non-zero values ---
    th1_std = diff_method1_kde_valleys(std_nz)
    th3_std = diff_method3_central_sigma(std_nz)
    th4_std = diff_method4_walk_outward(std_nz)

    # convert std thresholds back to raw scale
    scale = scaler_diff.scale_[row]
    th1_raw = [t * scale for t in th1_std]
    th3_raw = [t * scale for t in th3_std]
    th4_raw = [t * scale for t in th4_std]

    # manual threshold -> raw
    manual_th_raw = manual_th * scale if manual_th is not None else None

    desc = explanation_map.get(fidx, "")

    # ---- RAW plot (non-zero) ----
    all_raw_ths = [
        (th1_raw, DIFF_COLORS[1], DIFF_LABELS[1], "-"),
        (th3_raw, DIFF_COLORS[3], DIFF_LABELS[3], ":"),
        (th4_raw, DIFF_COLORS[4], DIFF_LABELS[4], (0, (3, 1, 1, 1))),
    ]
    ax = ax_pair[0]
    ax.hist(raw_nz, bins=200, alpha=0.65, color="steelblue", edgecolor="none")
    for ths, col, lab, ls in all_raw_ths:
        for k, t in enumerate(ths):
            ax.axvline(t, color=col, ls=ls, lw=1.3,
                       label=lab if k == 0 else "")
    if manual_th_raw is not None:
        ax.axvline(manual_th_raw, color="black", ls="-.", lw=2,
                   label=f"manual ({manual_th:.4f} std)")
    ax.set_title(f"SAE {fidx} — diff RAW (non-zero)", fontsize=10, pad=6)
    ax.set_xlabel("diff activation", fontsize=9)
    ax.set_ylabel("count", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    if desc:
        ax.text(0.02, 0.95, desc[:90], transform=ax.transAxes,
                fontsize=7, va="top", ha="left", color="0.3",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    # ---- STD plot (non-zero) ----
    all_std_ths = [
        (th1_std, DIFF_COLORS[1], DIFF_LABELS[1], "-"),
        (th3_std, DIFF_COLORS[3], DIFF_LABELS[3], ":"),
        (th4_std, DIFF_COLORS[4], DIFF_LABELS[4], (0, (3, 1, 1, 1))),
    ]
    ax = ax_pair[1]
    ax.hist(std_nz, bins=200, alpha=0.65, color="darkorange", edgecolor="none")
    for ths, col, lab, ls in all_std_ths:
        for k, t in enumerate(ths):
            ax.axvline(t, color=col, ls=ls, lw=1.3,
                       label=lab if k == 0 else "")
    if manual_th is not None:
        ax.axvline(manual_th, color="black", ls="-.", lw=2,
                   label=f"manual ({manual_th:.4f})")
    ax.set_title(f"SAE {fidx} — diff SCALED (non-zero)", fontsize=10, pad=6)
    ax.set_xlabel("diff activation (scaled)", fontsize=9)
    ax.set_ylabel("count", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    if desc:
        ax.text(0.02, 0.95, desc[:90], transform=ax.transAxes,
                fontsize=7, va="top", ha="left", color="0.3",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

fig.subplots_adjust(hspace=0.45, top=0.97)
fig.savefig(OUT_DIR / "feature_diff_distributions.png", dpi=150,
            bbox_inches="tight")
plt.show()
print(f"Saved {OUT_DIR / 'feature_diff_distributions.png'}")

# %% print headline examples per diff feature (positive side, walk-outward threshold)
print("\n" + "=" * 80)
print("DIFF HEADLINE EXAMPLES  (positive side, walk-outward threshold)")
print("=" * 80)

for col_i, fidx in enumerate(FEATURES):
    raw = X_diff[:, col_i]
    std = X_diff_std[:, col_i]
    nz_mask = raw != 0
    std_nz = std[nz_mask]

    th_pair = diff_method4_walk_outward(std_nz)
    pos_th = th_pair[1] if len(th_pair) == 2 else None
    # convert to raw scale
    scale = scaler_diff.scale_[col_i]
    pos_th_raw = pos_th * scale if pos_th is not None else None

    print(f"\n--- {feat_label(fidx)}  (pos threshold={pos_th_raw:.4f} raw / {pos_th:.4f} std)"
          if pos_th is not None
          else f"\n--- {feat_label(fidx)}  (no threshold found)")

    # Top 3 most activating on the positive side
    pos_mask = raw > 0
    if pos_mask.sum() == 0:
        print("  (no positive diff values)")
        continue
    pos_indices = np.where(pos_mask)[0]
    pos_vals = raw[pos_indices]
    order_desc = np.argsort(pos_vals)[::-1]
    print("  TOP 3 most activating (positive diff):")
    for k in range(min(3, len(order_desc))):
        idx = pos_indices[order_desc[k]]
        v = raw[idx]
        h1 = get_headline_first(idx)
        h2 = get_headline_second(idx)
        print(f"    [{v:.4f}]  H1: {h1}")
        print(f"              H2: {h2}")

    # 3 examples just below the positive threshold (largest positive values still under th)
    if pos_th_raw is not None:
        below_mask = (raw > 0) & (raw < pos_th_raw)
        if below_mask.sum() >= 3:
            below_idx = np.where(below_mask)[0]
            below_vals = raw[below_idx]
            order = np.argsort(below_vals)[::-1]
            print(f"  3 just BELOW pos threshold ({pos_th_raw:.4f} raw):")
            for k in range(min(3, len(order))):
                idx = below_idx[order[k]]
                v = raw[idx]
                h1 = get_headline_first(idx)
                h2 = get_headline_second(idx)
                print(f"    [{v:.4f}]  H1: {h1}")
                print(f"              H2: {h2}")
        else:
            print(f"  (fewer than 3 positive examples below threshold)")

# %% print threshold summary
print("\n" + "=" * 80)
print("DIFF THRESHOLD SUMMARY  (standardised scale)")
print("=" * 80)
for row, (fidx, manual_th) in enumerate(zip(FEATURES, MANUAL_TH_STD)):
    raw = X_diff[:, row]
    std_nz = X_diff_std[:, row][raw != 0]
    th1 = diff_method1_kde_valleys(std_nz)
    th3 = diff_method3_central_sigma(std_nz)
    th4 = diff_method4_walk_outward(std_nz)
    print(f"\n  Feature {fidx}  ({feat_label(fidx)})")
    print(f"    M1 KDE valleys:       {[f'{t:.4f}' for t in th1]}")
    print(f"    M3 Central 3σ:        {[f'{t:.4f}' for t in th3]}")
    print(f"    M4 Walk outward:      {[f'{t:.4f}' for t in th4]}")
    if manual_th is not None:
        print(f"    Manual:               {manual_th}")
    else:
        print(f"    Manual:               (none)")

print("\n" + "=" * 80)
print("CONTEXT THRESHOLD SUMMARY  (raw scale)")
print("=" * 80)
for row, fidx in enumerate(FEATURES):
    nz = X_ctx[:, row]
    nz = nz[nz != 0]
    cth1 = ctx_method1_kde_valley(nz)
    cth2 = ctx_method2_gmm2(nz)
    cth3 = ctx_method3_iqr_elbow(nz)
    cth4 = ctx_method4_walk_right(nz)
    print(f"\n  Feature {fidx}  ({feat_label(fidx)})")
    print(f"    M1 KDE valley:   {cth1:.4f}" if cth1 is not None else "    M1 KDE valley:   (none)")
    print(f"    M2 GMM-2 split:  {cth2:.4f}")
    print(f"    M3 Q1 elbow:     {cth3:.4f}")
    print(f"    M4 Walk right:   {cth4:.4f}" if cth4 is not None else "    M4 Walk right:   (none)")

print("\nDone.")

