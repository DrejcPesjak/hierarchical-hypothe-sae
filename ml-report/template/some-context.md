
You are helping me write a 4-page double-column seminar mini-paper (IMRaD: Abstract, Introduction, Related Work, Methods, Results, Discussion, Conclusion) about my project on explaining Upworthy headline A/B test outcomes using sparse autoencoder (SAE) features from an LLM. The paper must be concise, technical, and defensible, and should emphasize a coherent hypothesis-driven experimental narrative rather than a “model zoo”.

## Project goal and framing

We want interpretable explanations for why one headline variant wins over another in real A/B tests. We treat interpretability as **hypothesis generation / model explainability**, not causal proof. Prediction performance is used as a sanity check that representations capture real signal, while the main goal is to surface human-readable patterns from SAE features.

A central hypothesis we tested was **hierarchical / regime structure**: topic/context might gate which “difference” features matter (different clickability rules in different content subgroups). We attempted multiple methods to discover such supergroups but did not find stable, generalizable topic-gated hierarchy. However, we do observe non-additive interaction structure (depth-6 boosting beats depth-1 stumps).

## Dataset

* Source: Upworthy Research Archive (OSF). Each test_id contains 2–5 headline variants for the same article with impressions and clicks.
* We primarily use the **confirmatory** split.
* After sampling within-test pairs (capped at 10 random pairs per test) we have ~57,289 pairs, balanced labels.
* Target: pairwise winner classification (headline A vs B): label=1 if CTR_A > CTR_B else 0.
* Splitting: **Group split by test_id** (GroupShuffleSplit) so all pairs from the same test_id stay in train or test (prevents leakage).

## Representation: SAE features + context/diff decomposition

We encode headlines with **Gemma-3-4B-IT** and extract residual stream activations at **layer 22**. We mask out instruction template tokens so only headline content tokens remain. We encode activations with a pretrained **JumpReLU SAE (GemmaScope-2)** of width **16,384** (medium L0). We max-pool over content token positions to get one sparse vector per headline.

For a pair (A,B) with SAE vectors vA, vB:

* **context** = (vA + vB)/2 (shared topic/content)
* **diff** = vA − vB (what differs between variants)
  We concatenate [context | diff] → 32,768 features.

Rationale: direct CTR prediction from single headline vectors mostly learns topic base rates; pairwise diff focuses on within-test wording differences. Context was included to allow possible regime switching (topic-dependent rules).

## Binarization / trinarization

SAE activations have characteristic distributions with noisy near-zero tails. We developed a histogram “walk” threshold heuristic:

* Context: binary 0/1 (inactive vs meaningfully active)
* Diff: ternary −1/0/+1 (B higher, similar, A higher)
  This improves interpretability and acts as denoising; it also yields a small but consistent performance gain and concentrates importance on a sparse subset of features.

## Models evaluated (all on the same group split)

We report Accuracy, F1, and AUC on the test set.

| # | Method                 | Features                            | Key Settings                             | Acc       | F1        | AUC       |
| - | ---------------------- | ----------------------------------- | ---------------------------------------- | --------- | --------- | --------- |
| 1 | Logistic Regression    | continuous 32k, StandardScaler      | SGD L2 α=1e-4                            | 0.606     | 0.606     | 0.750     |
| 2 | Decision Tree          | continuous 32k, StandardScaler      | depth=8, leaf≥20                         | 0.556     | 0.551     | 0.630     |
| 3 | XGBoost                | continuous 32k, StandardScaler      | 100×d6, lr=0.1                           | 0.616     | 0.616     | 0.759     |
| 4 | XGBoost clusters (agg) | continuous diff 16k, StandardScaler | 10 clusters, 40×d6 each                  | 0.571     | 0.564     | 0.673     |
| 5 | **XGBoost bin d=6**    | **binarised 32k**                   | **100×d6, lr=0.1, colsample_bytree≈0.7** | **0.618** | **0.619** | **0.759** |
| 6 | XGBoost bin stumps d=1 | binarised 32k                       | 1200×d1, lr=0.05, subsample=0.8          | 0.577     | 0.578     | 0.681     |
| 7 | Decision Tree bin      | binarised 32k                       | depth=8, leaf≥20                         | 0.553     | 0.549     | 0.625     |
| 8 | RuleFit (XGB-selected) | binarised 467                       | n_est=25, tree_sz=4, max_rules=500       | 0.588     | 0.556     | 0.761     |

Interpretation highlights:

* LR is strong → substantial additive/global signal.
* Single decision trees are weak → high variance / greedy instability.
* XGBoost slightly beats LR → nonlinear interactions exist.
* Stumps (depth=1) are much worse than depth=6 → interactions matter.
* Clustering context and training per-cluster models performed worse → no evidence that unsupervised topic clusters capture distinct clickability regimes.
* RuleFit preserves ranking signal (AUC comparable to best) but has lower Acc/F1 under default threshold; still useful for readable conjunctive rules.

## Interpretability artifacts available

1. **XGBoost feature importance (binarized depth=6):** only 467/32,768 features have non-zero importance; top features are overwhelmingly **diff** features, with interpretable labels such as:

* diff_11052 “How to write clickbait titles”
* diff_1531 “discovery and revelation”
* diff_2220 “‘this’”
* diff_1213 “intense negative emotions”
* diff_3669 “video content and analysis”
* diff_4053 “joke, humor, comedy”
* diff_2193 “tragic and horrific events”
  (plus other diff features like “introduced person or concept”, “past events”, etc.)

2. **Tree dump audit of XGBoost (depth=6):** ctx splits are rare and typically appear near leaves; approximate split counts: ctx ~160 vs diff ~1400. This suggests the ensemble does not implement “context-first gating” in a strong way.
3. **RuleFit output:** produces conjunction rules over binarized features (up to 4-way conjunctions) plus linear terms; we can quote a handful of highest-weight rules/terms as examples of feature interactions.

## Negative results / regime discovery attempts

* HDBSCAN on context vectors (after SVD) was unstable: either 0 clusters (all noise) or many tiny clusters.
* k-means (10 clusters) produced balanced clusters but per-cluster XGBoost aggregated predictions were ~4% worse than global.
* MOB / model-based recursive partitioning (context splits + per-leaf lasso) found parameter instability but overfit severely due to p≫n in leaves (train R² high, test R² low ~0.04), so we do not treat it as successful evidence of stable regimes.

## What the paper should emphasize

* A coherent pipeline: Upworthy A/B → Gemma activations → SAE features → context/diff → binarization → predictive modeling → interpretable features/rules.
* Main empirical result: modest but real predictability (AUC ≈ 0.76), and top SAE diff features correspond to recognizable clickbait/curiosity/emotion cues.
* Hierarchy hypothesis outcome: interactions exist, but we did not recover stable topic-gated hierarchical regimes with clustering/MOB; context features are underutilized relative to diff.
* Clear limitations: predictive associations only (not causal), noisy labels when CTR differences are small, SAE feature labels are automated approximations, and hard partitioning may miss soft/overlapping regimes.

Write the paper in a compact academic style suitable for a short seminar report, using the table above as the primary quantitative result and using one figure/list for top feature importances plus a short qualitative list of example RuleFit rules/terms to illustrate interactions.

---
