# %% imports
import numpy as np
from time import time
from scipy.sparse import issparse
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

# ============================================================================
# %% MOBNode
# ============================================================================
class MOBNode:
    """Single node of a MOB tree."""

    def __init__(self, node_id, depth, indices):
        self.node_id = node_id
        self.depth = depth
        self.indices = indices              # row indices into training data
        self.n_samples = len(indices)
        self.is_leaf = True
        # split info (internal nodes only)
        self.split_feature = None
        self.split_threshold = None
        self.split_impr = None              # relative WSS improvement
        self.left = None
        self.right = None
        # model info
        self.model = None
        self.train_r2 = None
        self.mean_y = None
        self._resid = None                  # temp: freed after split search


# ============================================================================
# %% MOBTree
# ============================================================================
class MOBTree:
    r"""
    Custom efficient Model-Based recursive Partitioning (MOB).

    * Splits on  Z  (context features) to find subgroups where the
      inner model  f(X_diff) -> y  has heterogeneous parameters.

    Per-node algorithm
    ------------------
    1. Fit inner model on  X_diff[node] -> y[node]  ->  residuals.
    2. **Vectorised screening** — weighted |corr(z_j, residual)| for
       every context column in O(nnz) time via sparse-mat x vec products.
       Keep top `screen_k` features.
    3. **Detailed split eval** — for each screened z_j find the threshold
       that maximises weighted-SS reduction of residuals (cumsum trick).
    4. If  best_reduction / parent_wss > `min_improvement`  -> split.
    5. Recurse on children.

    Parameters
    ----------
    max_depth : int
        Maximum tree depth.
    min_samples_leaf : int
        Minimum samples in each child after a split.
    min_samples_split : int
        Minimum samples in a node to attempt splitting.
    min_improvement : float
        Minimum relative WSS reduction to accept a split.
    screen_k : int
        Number of context features to evaluate in detail (after screening).
    inner_model_class : sklearn estimator class
        Must support .fit(X, y, sample_weight=...) and .predict(X).
    inner_model_params : dict
        Kwargs forwarded to inner_model_class().
    verbose : bool
        Print build progress.
    """

    def __init__(
        self,
        *,
        max_depth=4,
        min_samples_leaf=200,
        min_samples_split=500,
        min_improvement=0.005,
        screen_k=200,
        inner_model_class=Lasso,
        inner_model_params=None,
        verbose=True,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_improvement = min_improvement
        self.screen_k = screen_k
        self.ModelCls = inner_model_class
        self.model_kw = inner_model_params or dict(
            alpha=0.01, max_iter=3000, random_state=42
        )
        self.verbose = verbose
        self._cnt = 0

    # ---- public API ---------------------------------------------------------

    def fit(self, X_diff, X_ctx, y, sample_weight=None):
        """Fit the MOB tree.

        Parameters
        ----------
        X_diff : array-like (n, p_diff)   — inner-model features
        X_ctx  : array-like (n, p_ctx)    — partitioning (context) features
        y      : array (n,)               — target
        sample_weight : array (n,) or None
        """
        self._Xd = X_diff
        self._Xc = X_ctx
        self._y = y
        self._w = sample_weight if sample_weight is not None else np.ones(len(y))
        self._cnt = 0
        t0 = time()
        self.root_ = self._grow(np.arange(len(y)), depth=0)
        if self.verbose:
            leaves = self.get_leaves()
            print(
                f"\nMOB built: {self._cnt} nodes, {len(leaves)} leaves  "
                f"({time() - t0:.1f} s)"
            )
        return self

    def predict(self, X_diff, X_ctx):
        """Return (predictions, leaf_ids)."""
        ids = self.apply(X_ctx)
        preds = np.empty(X_diff.shape[0])
        for lf in self.get_leaves():
            mask = ids == lf.node_id
            if mask.any():
                preds[mask] = lf.model.predict(X_diff[mask])
        return preds, ids

    def apply(self, X_ctx):
        """Return leaf node_id for every row."""
        out = np.empty(X_ctx.shape[0], dtype=np.int32)
        self._route(X_ctx, np.arange(out.size), self.root_, out)
        return out

    def get_leaves(self):
        """Return list of all leaf MOBNodes."""
        acc = []
        self._collect_leaves(self.root_, acc)
        return acc

    def get_all_nodes(self):
        """Return list of all MOBNodes in BFS order."""
        acc, queue = [], [self.root_]
        while queue:
            nd = queue.pop(0)
            acc.append(nd)
            if not nd.is_leaf:
                queue += [nd.left, nd.right]
        return acc

    def leaf_paths(self, ctx_names=None):
        """Return dict  {leaf_node_id: [(feature_name, direction, threshold), ...]}."""
        out = {}
        self._trace_paths(self.root_, [], out, ctx_names)
        return out

    def print_tree(self, ctx_names=None, diff_names=None, top_k=5):
        """Pretty-print the tree structure."""
        self._print_node(self.root_, "", ctx_names, diff_names, top_k)

    # ---- recursive build ----------------------------------------------------

    def _grow(self, idx, depth):
        nd = MOBNode(self._cnt, depth, idx)
        self._cnt += 1
        self._fit_model(nd)
        if self.verbose:
            print(
                f"  Node {nd.node_id:>3d}  d={depth}  n={nd.n_samples:>6d}  "
                f"R²={nd.train_r2:.4f}  ȳ={nd.mean_y:+.4f}"
            )
        # stopping criteria
        if depth >= self.max_depth or len(idx) < self.min_samples_split:
            nd._resid = None
            return nd
        split = self._find_best_split(nd)
        nd._resid = None  # free cached residuals
        if split is None:
            return nd
        feat, thresh, impr, left_idx, right_idx = split
        nd.is_leaf = False
        nd.split_feature = feat
        nd.split_threshold = thresh
        nd.split_impr = impr
        nd.left = self._grow(left_idx, depth + 1)
        nd.right = self._grow(right_idx, depth + 1)
        return nd

    def _fit_model(self, nd):
        """Fit inner model on the node's data, cache residuals."""
        X = self._Xd[nd.indices]
        y = self._y[nd.indices]
        w = self._w[nd.indices]
        m = self.ModelCls(**self.model_kw)
        m.fit(X, y, sample_weight=w)
        yhat = m.predict(X)
        resid = y - yhat
        nd.model = m
        nd._resid = resid
        nd.mean_y = float(np.average(y, weights=w))
        ss_res = float(w @ (resid ** 2))
        ss_tot = float(w @ ((y - nd.mean_y) ** 2))
        nd.train_r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    # ---- split search (vectorised screening + cumsum) -----------------------

    def _find_best_split(self, nd):
        """Two-stage split search: screening then detailed evaluation."""
        idx = nd.indices
        Zn = self._Xc[idx]              # context rows for this node
        w = self._w[idx]
        resid = nd._resid
        parent_wss = float(w @ (resid ** 2))
        if parent_wss < 1e-12:
            return None

        # --- Stage 1: vectorised screening via sparse matmul ---
        wr = w * resid
        total_w = w.sum()
        wmean_r = wr.sum() / total_w

        _sparse = issparse(Zn)
        if _sparse:
            Zc = Zn.tocsc()
            Zt_wr = np.asarray(Zc.T.dot(wr)).ravel()
            Zt_w = np.asarray(Zc.T.dot(w)).ravel()
            Zt_wz2 = np.asarray(Zc.power(2).T.dot(w)).ravel()
        else:
            Zc = Zn
            Zt_wr = Zn.T @ wr
            Zt_w = Zn.T @ w
            Zt_wz2 = (Zn ** 2).T @ w

        wmean_z = Zt_w / total_w
        cov = Zt_wr / total_w - wmean_r * wmean_z
        var_z = Zt_wz2 / total_w - wmean_z ** 2

        with np.errstate(divide="ignore", invalid="ignore"):
            scores = np.where(var_z > 1e-12, np.abs(cov) / np.sqrt(var_z), 0.0)

        n_active = int((scores > 0).sum())
        k = min(self.screen_k, n_active)
        if k == 0:
            return None
        candidates = np.argsort(scores)[::-1][:k]
        if self.verbose:
            print(f"    screen: {n_active} active ctx features → top {k}")

        # --- Stage 2: detailed split evaluation for each candidate ---
        best_reduction = -1.0
        best_result = None

        for col in candidates:
            if _sparse:
                z = np.asarray(Zc[:, col].toarray()).ravel()
            else:
                z = Zc[:, col]
            out = self._eval_threshold(z, resid, w, parent_wss)
            if out is not None and out[0] > best_reduction:
                best_reduction = out[0]
                best_result = (col, out[1])

        if best_result is None:
            return None

        col, thresh = best_result
        rel_impr = best_reduction / parent_wss
        if rel_impr < self.min_improvement:
            if self.verbose:
                print(
                    f"    best ctx[{col}] rel_impr={rel_impr:.6f} "
                    f"< {self.min_improvement} → stop"
                )
            return None

        # build child index arrays
        if _sparse:
            z = np.asarray(Zc[:, col].toarray()).ravel()
        else:
            z = Zc[:, col]
        left_mask = z <= thresh
        if self.verbose:
            print(
                f"    → ctx[{col}] ≤ {thresh:.4f}  "
                f"(rel_impr={rel_impr:.4f}, L={left_mask.sum()}, "
                f"R={len(idx) - left_mask.sum()})"
            )
        return (col, thresh, rel_impr, idx[left_mask], idx[~left_mask])

    def _eval_threshold(self, z, resid, w, parent_wss):
        """Find the best split threshold for one feature (cumsum trick).

        Returns (reduction, threshold) or None.
        """
        n = len(z)
        ml = self.min_samples_leaf
        order = np.argsort(z)
        zs = z[order]
        rs = resid[order]
        ws = w[order]

        # positions where z changes value
        change_pos = np.where(zs[:-1] != zs[1:])[0]
        valid = change_pos[(change_pos + 1 >= ml) & (n - change_pos - 1 >= ml)]
        if len(valid) == 0:
            return None

        # cumulative sums
        cum_w = np.cumsum(ws)
        cum_wr = np.cumsum(ws * rs)
        cum_wr2 = np.cumsum(ws * rs ** 2)
        tw, twr, twr2 = cum_w[-1], cum_wr[-1], cum_wr2[-1]

        # vectorised WSS for all valid splits
        lw = cum_w[valid]
        lwr = cum_wr[valid]
        lwr2 = cum_wr2[valid]
        rw = tw - lw
        rwr = twr - lwr
        rwr2 = twr2 - lwr2

        reduction = parent_wss - (lwr2 - lwr ** 2 / lw) - (rwr2 - rwr ** 2 / rw)
        best = int(np.argmax(reduction))
        if reduction[best] <= 0:
            return None
        pos = valid[best]
        threshold = float((zs[pos] + zs[pos + 1]) / 2.0)
        return (float(reduction[best]), threshold)

    # ---- traversal helpers --------------------------------------------------

    def _route(self, Xc, ix, nd, out):
        if nd.is_leaf:
            out[ix] = nd.node_id
            return
        if issparse(Xc):
            z = np.asarray(Xc[ix, nd.split_feature].toarray()).ravel()
        else:
            z = Xc[ix, nd.split_feature]
        left = z <= nd.split_threshold
        if left.any():
            self._route(Xc, ix[left], nd.left, out)
        if (~left).any():
            self._route(Xc, ix[~left], nd.right, out)

    def _collect_leaves(self, nd, acc):
        if nd.is_leaf:
            acc.append(nd)
        else:
            self._collect_leaves(nd.left, acc)
            self._collect_leaves(nd.right, acc)

    def _trace_paths(self, nd, path, out, ctx_names):
        if nd.is_leaf:
            out[nd.node_id] = list(path)
            return
        fname = (
            ctx_names[nd.split_feature]
            if ctx_names
            else f"ctx[{nd.split_feature}]"
        )
        self._trace_paths(
            nd.left, path + [(fname, "≤", nd.split_threshold)], out, ctx_names
        )
        self._trace_paths(
            nd.right, path + [(fname, ">", nd.split_threshold)], out, ctx_names
        )

    def _print_node(self, nd, indent, ctx_names, diff_names, top_k):
        if nd.is_leaf:
            print(
                f"{indent}LEAF {nd.node_id}: n={nd.n_samples}, "
                f"R²={nd.train_r2:.4f}, ȳ={nd.mean_y:+.4f}"
            )
            if diff_names and hasattr(nd.model, "coef_"):
                c = nd.model.coef_
                nz = np.where(c != 0)[0]
                if len(nz):
                    order = nz[np.argsort(np.abs(c[nz]))[::-1]]
                    for j in order[:top_k]:
                        print(f"{indent}  [{c[j]:+.4f}] {diff_names[j][:55]}")
        else:
            fname = (
                ctx_names[nd.split_feature][:55]
                if ctx_names
                else f"ctx[{nd.split_feature}]"
            )
            print(
                f"{indent}NODE {nd.node_id}: n={nd.n_samples}, "
                f"R²={nd.train_r2:.4f}  split: {fname} ≤ {nd.split_threshold:.4f}"
            )
            self._print_node(nd.left, indent + "  ├─ ", ctx_names, diff_names, top_k)
            self._print_node(
                nd.right, indent + "  └─ ", ctx_names, diff_names, top_k
            )


# ============================================================================
# %% plot_mob_tree
# ============================================================================
def plot_mob_tree(tree, ctx_names=None, diff_names=None, top_k=3, figsize=None):
    """Draw the MOB tree structure with matplotlib."""
    nodes = tree.get_all_nodes()
    max_depth = max(n.depth for n in nodes)
    n_leaves = len(tree.get_leaves())
    if figsize is None:
        figsize = (max(16, n_leaves * 4), max(10, (max_depth + 1) * 3.5))

    fig, ax = plt.subplots(figsize=figsize)
    pos = {}
    _layout(tree.root_, pos, 0.0, 1.0, 0)

    # edges
    for nd in nodes:
        if not nd.is_leaf:
            x0, y0 = pos[nd.node_id]
            for child, label in [(nd.left, "≤"), (nd.right, ">")]:
                x1, y1 = pos[child.node_id]
                ax.plot([x0, x1], [y0, y1], "k-", lw=0.8, zorder=1)
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                ax.text(
                    mx, my, label,
                    fontsize=7, ha="center", va="center", color="gray",
                )

    # nodes
    for nd in nodes:
        x, y = pos[nd.node_id]
        if nd.is_leaf:
            lines = [
                f"Leaf {nd.node_id}  (n={nd.n_samples})",
                f"R²={nd.train_r2:.3f}   ȳ={nd.mean_y:+.3f}",
            ]
            if diff_names and hasattr(nd.model, "coef_"):
                c = nd.model.coef_
                nz = np.where(c != 0)[0]
                if len(nz):
                    order = nz[np.argsort(np.abs(c[nz]))[::-1]]
                    for j in order[:top_k]:
                        lines.append(f"{c[j]:+.3f} {diff_names[j][:24]}")
            ax.text(
                x, y, "\n".join(lines),
                ha="center", va="center", fontsize=5.5,
                bbox=dict(
                    boxstyle="round,pad=0.4", fc="#c8e6c9", ec="green",
                    lw=0.8, alpha=0.95,
                ),
                zorder=2,
            )
        else:
            fname = (
                ctx_names[nd.split_feature][:30]
                if ctx_names
                else f"ctx[{nd.split_feature}]"
            )
            txt = f"Node {nd.node_id}  (n={nd.n_samples})\n{fname}\n≤ {nd.split_threshold:.3f}"
            ax.text(
                x, y, txt,
                ha="center", va="center", fontsize=6,
                bbox=dict(
                    boxstyle="round,pad=0.4", fc="#fff9c4", ec="orange",
                    lw=0.8, alpha=0.95,
                ),
                zorder=2,
            )

    ax.set_ylim(-max_depth - 0.8, 0.8)
    ax.axis("off")
    ax.set_title("MOB Tree  (context splits → per-leaf diff models)", fontsize=14)
    fig.tight_layout()
    return fig


def _layout(nd, pos, xmin, xmax, depth):
    """Recursive binary layout."""
    x = (xmin + xmax) / 2
    pos[nd.node_id] = (x, -depth)
    if not nd.is_leaf:
        mid = (xmin + xmax) / 2
        _layout(nd.left, pos, xmin, mid, depth + 1)
        _layout(nd.right, pos, mid, xmax, depth + 1)

