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
        self.split_impr = None              # supLM statistic value
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
    instability (Andrews 1993, Zeileis et al. 2008).  This tests whether
    model parameters are stable across the ordering induced by each
    candidate partitioning variable — it detects changes in slopes, not
    just in residual means.

    Per-node algorithm
    ------------------
    1. Fit inner model on X_diff[node] → y[node] → residuals.
    2. Build **score matrix** Ψ from the model's estimating equations:
       ψ_i = w_i * e_i * [1, x_{i,active}]   (q-dimensional).
    3. **Screen** context features: multi-dimensional R²(z_j → Ψ)
       computed in O(nnz·q) via sparse-mat × vec products.
    4. For top-k screened z_j: compute **supLM statistic** (structural
       break test on the ordered score process).
    5. Split on z with highest supLM if above critical value.
    6. Split point = position where instability peaks (supLM argmax).

    Parameters
    ----------
    max_depth : int
        Maximum tree depth.
    min_samples_leaf : int
        Minimum samples in each child after a split.
    min_samples_split : int
        Minimum samples in a node to attempt splitting.
    instability_alpha : float
        Significance level for the supLM instability test.
        Lower → harder to split (more conservative).
    trim : float
        Trimming fraction for the supLM test — exclude positions
        in the first/last `trim` fraction from the supremum.
    screen_k : int
        Number of context features to evaluate in detail (after screening).
    max_score_dim : int
        Maximum number of active model coefficients to include in
        the score vector (caps q for efficiency).
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
        instability_alpha=0.05,
        trim=0.1,
        screen_k=200,
        max_score_dim=20,
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
            q = nd._scores.shape[1] if nd._scores is not None else 0
            print(
                f"  Node {nd.node_id:>3d}  d={depth}  n={nd.n_samples:>6d}  "
                f"R²={nd.train_r2:.4f}  ȳ={nd.mean_y:+.4f}  q={q}"
            )
        # stopping criteria
        if depth >= self.max_depth or len(idx) < self.min_samples_split:
            self._free_caches(nd)
            return nd
        split = self._find_best_split(nd)
        self._free_caches(nd)
        if split is None:
            return nd
        feat, thresh, stat_val, left_idx, right_idx = split
        nd.is_leaf = False
        nd.split_feature = feat
        nd.split_threshold = thresh
        nd.split_impr = stat_val        # supLM value
        nd.left = self._grow(left_idx, depth + 1)
        nd.right = self._grow(right_idx, depth + 1)
        return nd

    @staticmethod
    def _free_caches(nd):
        nd._resid = None
        nd._scores = None
        nd._Sigma_inv = None

    def _fit_model(self, nd):
        """Fit inner model on the node's data; build score matrix."""
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
        # Build score matrix for instability test
        nd._scores, nd._Sigma_inv = self._build_scores(
            m.coef_, X, resid, w
        )

    # ---- score matrix & instability helpers ---------------------------------

    def _build_scores(self, coef, X_node, resid, w):
        """Build score matrix Ψ and its inverse variance Σ⁻¹.

        For a (weighted) linear model, the estimating-equation score is:
            ψ_i = w_i · e_i · [1,  x_{i, active features}]
        This is a q-dimensional vector (q = 1 + |active|, capped by
        max_score_dim).  The supLM test checks whether these scores
        are stable when ordered by each partitioning variable z_j.
        """
        n = len(resid)
        active = np.where(coef != 0)[0]

        # Limit score dimensions for computational efficiency
        if len(active) > self.max_score_dim:
            top_k = np.argsort(np.abs(coef[active]))[::-1][: self.max_score_dim]
            active = active[top_k]

        we = w * resid                      # w_i · e_i

        if len(active) == 0:
            # Only intercept score
            Psi = we.reshape(-1, 1)
        else:
            if issparse(X_node):
                X_act = np.asarray(X_node[:, active].toarray())
            else:
                X_act = X_node[:, active]
            # Columns: [intercept_score, coef_1_score, coef_2_score, ...]
            Psi = np.column_stack([we, X_act * we[:, None]])

        q = Psi.shape[1]

        # Empirical score variance  Σ̂ = (1/n) Σ ψ_i ψ_i'
        Sigma = (Psi.T @ Psi) / n + 1e-8 * np.eye(q)
        try:
            Sigma_inv = np.linalg.inv(Sigma)
        except np.linalg.LinAlgError:
            Sigma_inv = np.linalg.pinv(Sigma)

        return Psi, Sigma_inv

    @staticmethod
    def _supLM_critical(q, alpha=0.05):
        """Approximate critical value for the supLM statistic.

        Uses the approximation  crit ≈ (√q + c)²  where c depends on
        alpha, fitted to the Hansen (1997) / Andrews (1993) tables for
        trim ≈ 0.10.

        Examples (α=0.05):  q=1 → 10.2,  q=5 → 19.7,  q=10 → 28.7,
                            q=20 → 48.2
        """
        c_table = {0.01: 2.8, 0.05: 2.2, 0.10: 1.9}
        c = c_table.get(alpha)
        if c is None:
            if alpha <= 0.01:
                c = 2.8
            elif alpha <= 0.05:
                c = 2.8 - (alpha - 0.01) / 0.04 * 0.6
            elif alpha <= 0.10:
                c = 2.2 - (alpha - 0.05) / 0.05 * 0.3
            else:
                c = max(1.5, 1.9 - (alpha - 0.10) / 0.10 * 0.4)
        return (np.sqrt(q) + c) ** 2

    # ---- split search (parameter instability) -------------------------------

    def _find_best_split(self, nd):
        """MOB parameter instability test + split search.

        1. Screen context features via multi-dim R²(z_j, Ψ).
        2. For top-k candidates: compute supLM statistic.
        3. Split on the z_j with highest supLM (if above critical value).
        4. Split point = position where instability is most pronounced.
        """
        idx = nd.indices
        n = len(idx)
        Zn = self._Xc[idx]
        Psi = nd._scores              # (n, q)
        Sigma_inv = nd._Sigma_inv     # (q, q)
        q = Psi.shape[1]

        # --- Stage 1: vectorised multi-dimensional screening ---
        # For each z_j compute  R²(z_j → Ψ) = Σ_k corr(z_j, ψ_k)²
        # via  cov = Z'Ψ / n − μ_z · μ_ψ'   (one sparse matmul)

        Psi_mean = Psi.mean(axis=0)                             # (q,)
        Psi_var = np.maximum(
            ((Psi - Psi_mean) ** 2).mean(axis=0), 1e-12
        )                                                       # (q,)

        _sp = issparse(Zn)
        ones_n = np.ones(n)
        if _sp:
            Zc = Zn.tocsc()
            ZtP = np.asarray(Zc.T.dot(Psi))                    # (p, q)
            Z_sum = np.asarray(Zc.T.dot(ones_n)).ravel()        # (p,)
            Z_sq_sum = np.asarray(
                Zc.power(2).T.dot(ones_n)
            ).ravel()                                           # (p,)
        else:
            Zc = Zn
            ZtP = Zn.T @ Psi
            Z_sum = Zn.T @ ones_n
            Z_sq_sum = (Zn ** 2).T @ ones_n

        Z_mean = Z_sum / n
        Z_var = Z_sq_sum / n - Z_mean ** 2

        # cov(z_j, ψ_k) = (Z'Ψ)_{jk}/n  −  μ_{z_j}·μ_{ψ_k}
        C = ZtP / n - np.outer(Z_mean, Psi_mean)               # (p, q)

        # Screening score:  Σ_k  corr(z_j, ψ_k)²
        with np.errstate(divide="ignore", invalid="ignore"):
            C_norm_sq = C ** 2 / Psi_var[None, :]               # (p, q)
            screen = np.where(
                Z_var > 1e-12,
                C_norm_sq.sum(axis=1) / Z_var,
                0.0,
            )                                                   # (p,)

        n_active = int((screen > 0).sum())
        k = min(self.screen_k, n_active)
        if k == 0:
            return None
        candidates = np.argsort(screen)[::-1][:k]
        if self.verbose:
            print(
                f"    screen: {n_active} active ctx → top {k}  "
                f"(q={q} score dims)"
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

        # --- Stage 3: critical value check ---
        crit = self._supLM_critical(q, self.instability_alpha)
        if best_stat < crit:
            if self.verbose:
                print(
                    f"    best supLM={best_stat:.2f} on ctx[{col}] "
                    f"< crit={crit:.2f} (α={self.instability_alpha}, q={q}) "
                    f"→ stop"
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
                f"(supLM={best_stat:.2f}, crit={crit:.2f}, "
                f"L={left_mask.sum()}, R={n - left_mask.sum()})"
            )
        return (col, threshold, best_stat, idx[left_mask], idx[~left_mask])

    def _compute_supLM(self, z, Psi, Sigma_inv):
        """Compute the supLM statistic for one partitioning variable z.

        Orders observations by z, computes the cumulative score process
        B_t = S_t − (t/n)·S_n, and evaluates the LM-type quadratic form
        stat(t) = [n / (t·(n−t))] · B_t' Σ⁻¹ B_t
        at every valid split point.

        Returns (supLM_value, split_position_in_sorted_order) or None.
        """
        n = len(z)
        ml = self.min_samples_leaf
        i_trim = max(int(n * self.trim), ml)

        order = np.argsort(z)
        z_sorted = z[order]
        Psi_sorted = Psi[order]                         # (n, q)

        # Valid split positions: z must change AND respect trim/leaf size
        change = np.where(z_sorted[:-1] != z_sorted[1:])[0]
        valid = change[
            (change + 1 >= i_trim) & (change + 1 <= n - i_trim)
        ]
        if len(valid) == 0:
            return None

        # Cumulative score sums  S_i = Σ_{t=1}^{i} ψ_t
        S = np.cumsum(Psi_sorted, axis=0)               # (n, q)
        S_n = S[-1]                                     # (q,)

        # Fluctuation process at valid positions:
        #   B_i = S_i − (i/n) · S_n
        frac = (valid + 1).astype(np.float64) / n       # (|valid|,)
        B = S[valid] - np.outer(frac, S_n)              # (|valid|, q)

        # Quadratic form  B_i' Σ⁻¹ B_i  (vectorised over valid positions)
        BM = B @ Sigma_inv                              # (|valid|, q)
        quad = np.sum(BM * B, axis=1)                   # (|valid|,)

        # Scaling:  stat(i) = n / (i · (n − i)) · quad(i)
        i_arr = (valid + 1).astype(np.float64)
        scaling = n / (i_arr * (n - i_arr))
        stat = scaling * quad

        best = int(np.argmax(stat))
        if stat[best] <= 0:
            return None
        return (float(stat[best]), int(valid[best]))

    # ---- OLD split search (residual WSS reduction) --------------------------
    # Commented out: this approach only detects mean-shift in residuals,
    # NOT proper parameter instability.  Replaced by supLM above.
    #
    # def _find_best_split_wss(self, nd):
    #     """Residual WSS reduction (CART-like).  NOT proper MOB."""
    #     idx = nd.indices
    #     Zn = self._Xc[idx]; w = self._w[idx]; resid = nd._resid
    #     parent_wss = float(w @ (resid ** 2))
    #     if parent_wss < 1e-12: return None
    #     wr = w * resid; total_w = w.sum(); wmean_r = wr.sum() / total_w
    #     _sp = issparse(Zn)
    #     if _sp:
    #         Zc = Zn.tocsc()
    #         Zt_wr = np.asarray(Zc.T.dot(wr)).ravel()
    #         Zt_w = np.asarray(Zc.T.dot(w)).ravel()
    #         Zt_wz2 = np.asarray(Zc.power(2).T.dot(w)).ravel()
    #     else:
    #         Zc = Zn
    #         Zt_wr = Zn.T @ wr; Zt_w = Zn.T @ w
    #         Zt_wz2 = (Zn ** 2).T @ w
    #     wmean_z = Zt_w / total_w
    #     cov = Zt_wr / total_w - wmean_r * wmean_z
    #     var_z = Zt_wz2 / total_w - wmean_z ** 2
    #     with np.errstate(divide="ignore", invalid="ignore"):
    #         scores = np.where(var_z > 1e-12, np.abs(cov)/np.sqrt(var_z), 0.)
    #     k = min(self.screen_k, int((scores > 0).sum()))
    #     if k == 0: return None
    #     cands = np.argsort(scores)[::-1][:k]
    #     best_red, best_res = -1., None
    #     for col in cands:
    #         z = np.asarray(Zc[:,col].toarray()).ravel() if _sp else Zc[:,col]
    #         out = self._eval_threshold_wss(z, resid, w, parent_wss)
    #         if out and out[0] > best_red:
    #             best_red, best_res = out[0], (col, out[1])
    #     if best_res is None: return None
    #     col, thr = best_res
    #     ri = best_red / parent_wss
    #     if ri < 0.005: return None
    #     z = np.asarray(Zc[:,col].toarray()).ravel() if _sp else Zc[:,col]
    #     lm = z <= thr
    #     return (col, thr, ri, idx[lm], idx[~lm])
    #
    # def _eval_threshold_wss(self, z, resid, w, parent_wss):
    #     """Best threshold via cumsum WSS trick."""
    #     n = len(z); ml = self.min_samples_leaf
    #     o = np.argsort(z); zs, rs, ws = z[o], resid[o], w[o]
    #     ch = np.where(zs[:-1] != zs[1:])[0]
    #     v = ch[(ch+1 >= ml) & (n-ch-1 >= ml)]
    #     if len(v) == 0: return None
    #     cw  = np.cumsum(ws);  cwr = np.cumsum(ws*rs)
    #     cwr2 = np.cumsum(ws*rs**2)
    #     tw, twr, twr2 = cw[-1], cwr[-1], cwr2[-1]
    #     lw,lwr,lwr2 = cw[v],cwr[v],cwr2[v]
    #     rw,rwr,rwr2 = tw-lw,twr-lwr,twr2-lwr2
    #     red = parent_wss - (lwr2 - lwr**2/lw) - (rwr2 - rwr**2/rw)
    #     b = int(np.argmax(red))
    #     if red[b] <= 0: return None
    #     return (float(red[b]), float((zs[v[b]]+zs[v[b]+1])/2))

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
                f"  (supLM={nd.split_impr:.1f})"
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
            txt = (
                f"Node {nd.node_id}  (n={nd.n_samples})\n"
                f"{fname}\n≤ {nd.split_threshold:.3f}"
                f"\nsupLM={nd.split_impr:.1f}"
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
    """Recursive binary layout."""
    x = (xmin + xmax) / 2
    pos[nd.node_id] = (x, -depth)
    if not nd.is_leaf:
        mid = (xmin + xmax) / 2
        _layout(nd.left, pos, xmin, mid, depth + 1)
        _layout(nd.right, pos, mid, xmax, depth + 1)
