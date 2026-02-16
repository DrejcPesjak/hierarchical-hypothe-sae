# %% imports
import numpy as np
from time import time
from scipy.sparse import issparse
from sklearn.linear_model import Lasso, LinearRegression
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
        self.split_impr = None              # supLM statistic value
        self.split_pval = None              # Bonferroni-corrected p-value
        self.left = None
        self.right = None
        # model info
        self.model = None
        self.train_r2 = None
        self.mean_y = None
        # temporaries — freed after split search
        self._resid = None
        self._scores = None                 # score matrix Ψ  (n × q)
        self._Sigma_inv = None              # inverse score variance  (q × q)


# ============================================================================
# %% MOBTree
# ============================================================================
class MOBTree:
    r"""
    Custom efficient Model-Based recursive Partitioning (MOB).

    Splits on Z (context features) to find subgroups where the inner
    model f(X_diff) → y has **heterogeneous parameters**.

    Uses the **supLM (supremum Lagrange Multiplier)** test for parameter
    instability (Andrews 1993, Zeileis et al. 2008).

    Score calibration
    -----------------
    The supLM test requires scores ψ_i from a smooth estimating equation.
    *  For Ridge / OLS inner models:  ψ_i = w_i·e_i·[1, x_i]  is exact.
    *  For Lasso: the L1 penalty makes the score non-smooth.  When
       `post_lasso_ols=True` (default), we refit unpenalised OLS on the
       Lasso-selected active set and use *those* residuals for the score
       matrix.  This gives a valid smooth M-estimator on the selected
       sub-model (standard "post-Lasso inference" approach).

    Null distribution & multiple testing
    -------------------------------------
    *  P-values are computed by **Monte-Carlo simulation** of the supLM
       null distribution:  sup_{π≤s≤1-π}  ||BB_q(s)||² / (s(1-s)),
       where BB_q is a q-dimensional standard Brownian bridge.
       Simulated once per unique (q, trim) and cached.
    *  **Bonferroni correction** is applied over n_testable — the number
       of context features with non-degenerate variance in the node.

    Per-node algorithm
    ------------------
    1. Fit inner model on X_diff[node] → y[node].
    2. Build score matrix Ψ (post-Lasso OLS if applicable).
    3. Screen context features: multi-dim R²(z_j → Ψ) in O(nnz·q).
    4. For top-k screened z_j: compute supLM statistic.
    5. Compute raw p-value from simulated null, Bonferroni-correct.
    6. Split on best z_j if corrected p < α.
    7. Split point = supLM argmax position.

    Parameters
    ----------
    max_depth, min_samples_leaf, min_samples_split : int
        Tree structure constraints.
    instability_alpha : float
        Significance level for the Bonferroni-corrected supLM test.
    trim : float
        Trimming fraction — exclude first/last `trim` of the ordered
        sample from the supremum search.
    screen_k : int
        Number of context features evaluated in detail after screening.
    max_score_dim : int
        Cap on the number of active coefficients in the score vector.
    post_lasso_ols : bool
        If True and inner model is Lasso-like, refit OLS on the active
        set and use those residuals for the instability scores.
    n_sim : int
        Number of Monte-Carlo draws for the supLM null distribution.
    inner_model_class, inner_model_params :
        Sklearn estimator class and kwargs.
    verbose : bool
    """

    def __init__(
        self,
        *,
        max_depth=4,
        min_samples_leaf=200,
        min_samples_split=500,
        instability_alpha=0.05,
        trim=0.1,
        screen_k=200,
        max_score_dim=20,
        post_lasso_ols=True,
        n_sim=5000,
        inner_model_class=Lasso,
        inner_model_params=None,
        verbose=True,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.instability_alpha = instability_alpha
        self.trim = trim
        self.screen_k = screen_k
        self.max_score_dim = max_score_dim
        self.post_lasso_ols = post_lasso_ols
        self.n_sim = n_sim
        self.ModelCls = inner_model_class
        self.model_kw = inner_model_params or dict(
            alpha=0.01, max_iter=3000, random_state=42
        )
        self.verbose = verbose
        self._cnt = 0
        self._null_cache = {}               # (q, trim) → sorted null stats

    # ---- public API ---------------------------------------------------------

    def fit(self, X_diff, X_ctx, y, sample_weight=None):
        """Fit the MOB tree."""
        self._Xd = X_diff
        self._Xc = X_ctx
        self._y = y
        self._w = sample_weight if sample_weight is not None else np.ones(len(y))
        self._cnt = 0
        self._null_cache = {}               # reset on new fit
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
        acc = []
        self._collect_leaves(self.root_, acc)
        return acc

    def get_all_nodes(self):
        acc, queue = [], [self.root_]
        while queue:
            nd = queue.pop(0)
            acc.append(nd)
            if not nd.is_leaf:
                queue += [nd.left, nd.right]
        return acc

    def leaf_paths(self, ctx_names=None):
        out = {}
        self._trace_paths(self.root_, [], out, ctx_names)
        return out

    def print_tree(self, ctx_names=None, diff_names=None, top_k=5):
        self._print_node(self.root_, "", ctx_names, diff_names, top_k)

    # ---- recursive build ----------------------------------------------------

    def _grow(self, idx, depth):
        nd = MOBNode(self._cnt, depth, idx)
        self._cnt += 1
        self._fit_model(nd)
        if self.verbose:
            q = nd._scores.shape[1] if nd._scores is not None else 0
            print(
                f"  Node {nd.node_id:>3d}  d={depth}  n={nd.n_samples:>6d}  "
                f"R²={nd.train_r2:.4f}  ȳ={nd.mean_y:+.4f}  q={q}"
            )
        if depth >= self.max_depth or len(idx) < self.min_samples_split:
            self._free_caches(nd)
            return nd
        split = self._find_best_split(nd)
        self._free_caches(nd)
        if split is None:
            return nd
        feat, thresh, stat_val, bonf_pval, left_idx, right_idx = split
        nd.is_leaf = False
        nd.split_feature = feat
        nd.split_threshold = thresh
        nd.split_impr = stat_val
        nd.split_pval = bonf_pval
        nd.left = self._grow(left_idx, depth + 1)
        nd.right = self._grow(right_idx, depth + 1)
        return nd

    @staticmethod
    def _free_caches(nd):
        nd._resid = None
        nd._scores = None
        nd._Sigma_inv = None

    def _fit_model(self, nd):
        """Fit inner model; build score matrix for instability test."""
        idx = nd.indices
        X = self._Xd[idx]
        y = self._y[idx]
        w = self._w[idx]

        # Fit the actual prediction model
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

        # Build scores for instability test
        nd._scores, nd._Sigma_inv = self._build_scores(
            m.coef_, X, y, resid, w
        )

    # ---- score matrix -------------------------------------------------------

    def _build_scores(self, coef, X_node, y_node, resid_model, w):
        """Build score matrix Ψ and its inverse variance Σ⁻¹.

        When post_lasso_ols=True and the model has an L1-selected active
        set, we refit OLS on the active set and use *those* residuals.
        This gives a valid smooth M-estimator score:
            ψ_i = w_i · e_i^{OLS} · [1,  x_{i, active}]
        """
        n = len(resid_model)
        active = np.where(coef != 0)[0]

        # Cap score dimensions
        if len(active) > self.max_score_dim:
            top_k = np.argsort(np.abs(coef[active]))[::-1][: self.max_score_dim]
            active = active[top_k]

        # Extract active columns (needed for both OLS refit and scores)
        if len(active) > 0:
            if issparse(X_node):
                X_act = np.asarray(X_node[:, active].toarray())
            else:
                X_act = X_node[:, active]
        else:
            X_act = None

        # Get residuals for the score function
        if self.post_lasso_ols and X_act is not None and len(active) > 0:
            # Refit unpenalised OLS on the Lasso-selected active set
            # → smooth M-estimator with valid score function
            ols = LinearRegression(fit_intercept=True)
            ols.fit(X_act, y_node, sample_weight=w)
            resid = y_node - ols.predict(X_act)
        else:
            resid = resid_model

        # Score:  ψ_i = w_i · e_i · [1, x_{i, active}]
        we = w * resid
        if X_act is None:
            Psi = we.reshape(-1, 1)
        else:
            Psi = np.column_stack([we, X_act * we[:, None]])

        q = Psi.shape[1]

        # Empirical score variance  Σ̂ = (1/n) Σ ψ_i ψ_i'
        Sigma = (Psi.T @ Psi) / n + 1e-8 * np.eye(q)
        try:
            Sigma_inv = np.linalg.inv(Sigma)
        except np.linalg.LinAlgError:
            Sigma_inv = np.linalg.pinv(Sigma)

        return Psi, Sigma_inv

    # ---- supLM null distribution (Monte-Carlo) ------------------------------

    def _get_null_dist(self, q):
        """Return sorted array of simulated supLM values under H0.

        Cached per (q, trim).  Under H0 the supLM converges to
            sup_{π ≤ s ≤ 1-π}  ||BB_q(s)||² / (s·(1-s))
        where BB_q is a q-dimensional standard Brownian bridge.
        """
        key = (q, self.trim)
        if key not in self._null_cache:
            if self.verbose:
                print(f"    [simulating supLM null: q={q}, trim={self.trim}, "
                      f"n_sim={self.n_sim} ...]")
            self._null_cache[key] = self._simulate_supLM_null(q)
        return self._null_cache[key]

    def _simulate_supLM_null(self, q, n_grid=1000):
        """Monte-Carlo simulation of the supLM null distribution.

        Generates n_sim realisations of
            sup_{π ≤ s ≤ 1-π}  ||BB_q(s)||² / (s·(1-s))
        by discretising the Brownian bridge on a grid of n_grid points.
        """
        rng = np.random.RandomState(42)
        trim = self.trim
        n_sim = self.n_sim
        dt = 1.0 / n_grid
        grid = np.linspace(dt, 1.0 - dt, n_grid)
        mask = (grid >= trim) & (grid <= 1.0 - trim)
        grid_m = grid[mask]
        n_m = int(mask.sum())
        scale = 1.0 / (grid_m * (1.0 - grid_m))        # (n_m,)

        sup_stats = np.empty(n_sim)
        batch = 200
        for start in range(0, n_sim, batch):
            end = min(start + batch, n_sim)
            b = end - start
            # Brownian motion increments → cumsum → Brownian bridge
            inc = rng.randn(b, n_grid, q) * np.sqrt(dt)
            Bm = np.cumsum(inc, axis=1)                 # (b, n_grid, q)
            # BB(s) = B(s) - s·B(1)
            BB = Bm - np.einsum("k,bq->bkq", grid, Bm[:, -1, :])
            BB_m = BB[:, mask, :]                        # (b, n_m, q)
            # ||BB(s)||² / (s(1-s))
            quad = np.sum(BB_m ** 2, axis=2)             # (b, n_m)
            stat = quad * scale[None, :]                 # (b, n_m)
            sup_stats[start:end] = stat.max(axis=1)

        return np.sort(sup_stats)

    def _supLM_pvalue(self, stat, q):
        """CDF of the simulated supLM null: P(supLM_null < stat).

        Use  1 - _supLM_pvalue(...)  to get the upper-tail p-value.
        When stat exceeds all simulated values, returns  1 - 1/n_sim
        (floor to avoid exact zero p-values).
        """
        null = self._get_null_dist(q)
        n = len(null)
        idx = int(np.searchsorted(null, stat, side="left"))
        # floor: never return CDF=1.0 exactly (p-value never exactly 0)
        return min(float(idx) / n, 1.0 - 1.0 / n)

    # ---- split search (parameter instability) -------------------------------

    def _find_best_split(self, nd):
        """MOB parameter instability test with Bonferroni correction.

        1. Screen context features via multi-dim R²(z_j, Ψ).
        2. For top-k candidates: compute supLM statistic.
        3. Compute raw p-value from simulated null.
        4. Bonferroni-correct over n_testable features.
        5. Split if corrected p < α.
        """
        idx = nd.indices
        n = len(idx)
        Zn = self._Xc[idx]
        Psi = nd._scores
        Sigma_inv = nd._Sigma_inv
        q = Psi.shape[1]

        # --- Stage 1: vectorised multi-dimensional screening ---
        Psi_mean = Psi.mean(axis=0)
        Psi_var = np.maximum(
            ((Psi - Psi_mean) ** 2).mean(axis=0), 1e-12
        )

        _sp = issparse(Zn)
        ones_n = np.ones(n)
        if _sp:
            Zc = Zn.tocsc()
            ZtP = np.asarray(Zc.T.dot(Psi))
            Z_sum = np.asarray(Zc.T.dot(ones_n)).ravel()
            Z_sq_sum = np.asarray(Zc.power(2).T.dot(ones_n)).ravel()
        else:
            Zc = Zn
            ZtP = Zn.T @ Psi
            Z_sum = Zn.T @ ones_n
            Z_sq_sum = (Zn ** 2).T @ ones_n

        Z_mean = Z_sum / n
        Z_var = Z_sq_sum / n - Z_mean ** 2

        C = ZtP / n - np.outer(Z_mean, Psi_mean)

        with np.errstate(divide="ignore", invalid="ignore"):
            C_norm_sq = C ** 2 / Psi_var[None, :]
            screen = np.where(
                Z_var > 1e-12,
                C_norm_sq.sum(axis=1) / Z_var,
                0.0,
            )

        # n_testable = features with non-degenerate variance
        # (this is what we Bonferroni-correct over)
        n_testable = int((Z_var > 1e-12).sum())
        n_active = int((screen > 0).sum())
        k = min(self.screen_k, n_active)
        if k == 0:
            return None
        candidates = np.argsort(screen)[::-1][:k]
        if self.verbose:
            print(
                f"    screen: {n_testable} testable, {n_active} correlated "
                f"→ top {k}  (q={q} score dims)"
            )

        # --- Stage 2: supLM for each screened candidate ---
        best_stat = -1.0
        best_result = None

        for col in candidates:
            if _sp:
                z = np.asarray(Zc[:, col].toarray()).ravel()
            else:
                z = Zc[:, col]
            result = self._compute_supLM(z, Psi, Sigma_inv)
            if result is not None:
                stat_val, split_pos = result
                if stat_val > best_stat:
                    best_stat = stat_val
                    best_result = (col, split_pos, z)

        if best_result is None:
            return None

        col, split_pos, z = best_result

        # --- Stage 3: p-value + Bonferroni ---
        raw_pval = 1.0 - self._supLM_pvalue(best_stat, q)
        bonf_pval = min(1.0, raw_pval * n_testable)

        if bonf_pval >= self.instability_alpha:
            if self.verbose:
                print(
                    f"    best supLM={best_stat:.2f} on ctx[{col}]  "
                    f"raw_p={raw_pval:.4f}  bonf_p={bonf_pval:.4f} "
                    f"(m={n_testable})  ≥ α={self.instability_alpha} → stop"
                )
            return None

        # --- Build split ---
        order = np.argsort(z)
        z_sorted = z[order]
        threshold = float(
            (z_sorted[split_pos] + z_sorted[split_pos + 1]) / 2.0
        )
        left_mask = z <= threshold
        if self.verbose:
            print(
                f"    → ctx[{col}] ≤ {threshold:.4f}  "
                f"(supLM={best_stat:.2f}, raw_p={raw_pval:.4f}, "
                f"bonf_p={bonf_pval:.4f}, "
                f"L={left_mask.sum()}, R={n - left_mask.sum()})"
            )
        return (col, threshold, best_stat, bonf_pval,
                idx[left_mask], idx[~left_mask])

    def _compute_supLM(self, z, Psi, Sigma_inv):
        """supLM statistic for one partitioning variable z.

        Orders by z, computes the empirical fluctuation process
        B_t = S_t − (t/n)·S_n, then evaluates
            stat(t) = [n / (t·(n−t))] · B_t' Σ̂⁻¹ B_t
        at every valid split point.

        Returns (supLM_value, split_position_in_sorted_order) or None.
        """
        n = len(z)
        ml = self.min_samples_leaf
        i_trim = max(int(n * self.trim), ml)

        order = np.argsort(z)
        z_sorted = z[order]
        Psi_sorted = Psi[order]

        change = np.where(z_sorted[:-1] != z_sorted[1:])[0]
        valid = change[
            (change + 1 >= i_trim) & (change + 1 <= n - i_trim)
        ]
        if len(valid) == 0:
            return None

        S = np.cumsum(Psi_sorted, axis=0)
        S_n = S[-1]
        frac = (valid + 1).astype(np.float64) / n
        B = S[valid] - np.outer(frac, S_n)

        BM = B @ Sigma_inv
        quad = np.sum(BM * B, axis=1)

        i_arr = (valid + 1).astype(np.float64)
        scaling = n / (i_arr * (n - i_arr))
        stat = scaling * quad

        best = int(np.argmax(stat))
        if stat[best] <= 0:
            return None
        return (float(stat[best]), int(valid[best]))

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
                f"  (supLM={nd.split_impr:.1f}, p_bonf={nd.split_pval:.4f})"
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
            txt = (
                f"Node {nd.node_id}  (n={nd.n_samples})\n"
                f"{fname}\n≤ {nd.split_threshold:.3f}"
                f"\np={nd.split_pval:.3f}"
            )
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
    x = (xmin + xmax) / 2
    pos[nd.node_id] = (x, -depth)
    if not nd.is_leaf:
        mid = (xmin + xmax) / 2
        _layout(nd.left, pos, xmin, mid, depth + 1)
        _layout(nd.right, pos, mid, xmax, depth + 1)
