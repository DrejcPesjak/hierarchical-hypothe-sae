
---

## (2) Headline / clickbait linguistics + engagement prediction

### Upworthy + headline wording effects (closest to your dataset framing)

* **Matias et al. (2021)** *The Upworthy Research Archive…* (dataset paper; how the experiments are structured, validation, scope). ([The Upworthy Research Archive][1])
* **Robertson et al. (2023)** *Negativity drives online news consumption* (uses randomized Upworthy headline trials; shows measurable effects of negative vs positive wording on CTR). ([Nature][2])

### Linguistics / pragmatics of clickbait and “curiosity gap” mechanisms

* **Scott (2021)** *Clickbait, relevance, and the curiosity gap* (pragmatics + corpus analysis; forward reference, definite expressions, intensifiers—very aligned with your “this”, curiosity-gap SAE features). ([Kingston University London][3])
* **Nature HSS Communications (2025)** *The evolution of online news headlines* (surveys headline linguistic devices; explicitly mentions demonstratives/pronouns/forward reference and links to Upworthy/clickbait style). ([Nature][4])

### Clickbait detection (bridges linguistic cues ↔ ML models)

* **Chakraborty et al. (2016)** *Stop Clickbait: Detecting and preventing clickbaits in online news media* (classic ML framing + feature patterns; often cited as a “clickbait features” anchor). ([GitHub][5])
* **Anand et al. (2016)** *We used Neural Networks to Detect Clickbaits…* (neural detection baseline + dataset framing around curiosity gap). ([arXiv][6])

### Engagement/CTR prediction from headline text (method neighbors to your supervised step)

* **Yamamoto et al. (2020, JSAI)** *CTR prediction with confidence for news headlines using NGBoost* (explicit CTR prediction from A/B logs; shows tree-based probabilistic regression framing). ([CiNii Research][7])
* **(Optional if you want an “engagement as a double-edged sword” angle)** **Investigating clickbait effects on engagement** (2025; finds “information gap” + “emotional intensity” as key clickbait features in experiments/observational studies). ([ScienceDirect][8])

---

## (6) Pairwise modeling literature (directly matches “A vs B winner”)

### Classical paired-comparison models (clean theory for A/B outcomes)

* **Bradley & Terry (1952)** *Rank analysis… the method of paired comparisons* (the Bradley–Terry model: probability one item beats another). ([OUP Academic][9])
* **Thurstone (1927)** *A Law of Comparative Judgment* (psychometrics foundation for scaling preferences from pairwise judgments). ([Brock University][10])

These are perfect to cite when you explain that “headline winner labels are pairwise preferences,” and your model is learning a structured predictor of that preference.

### Learning-to-rank / pairwise preference learning (ML framing of pairwise comparisons)

* **Burges et al. (2005)** *Learning to Rank using Gradient Descent* (RankNet: pairwise loss over document pairs; canonical “pairwise winner” training objective). ([Microsoft][11])
* **Cao et al. (2007, MSR-TR)** *Learning to Rank: From Pairwise Approach to Listwise Approach* (positions pairwise methods vs listwise; helpful for justifying why pairwise is natural here). ([Microsoft][12])

### (Optional) Ranking-probability models (nice if you want “context gating / mixtures / regimes” analogies)

* **Zhao et al. (2016, ICML)** *Learning Mixtures of Plackett–Luce Models* (mixtures over preference/ranking models; conceptually adjacent to your “regimes / gating” hypothesis). ([Proceedings of Machine Learning Research][13])

---


[1]: https://upworthy.natematias.com/about-the-archive.html?utm_source=chatgpt.com "Data in the Upworthy Research Archive | The Upworthy Research Archive"
[2]: https://www.nature.com/articles/s41562-023-01538-4?utm_source=chatgpt.com "Negativity drives online news consumption | Nature Human Behaviour"
[3]: https://researchinnovation.kingston.ac.uk/en/publications/you-wont-believe-whats-in-this-paper-clickbait-relevance-and-the--3?utm_source=chatgpt.com "You won't believe what's in this paper! Clickbait, relevance, and the curiosity gap - Kingston University London"
[4]: https://www.nature.com/articles/s41599-025-04514-7?utm_source=chatgpt.com "The evolution of online news headlines | Humanities and Social Sciences Communications"
[5]: https://github.com/bhargaviparanjape/clickbait?utm_source=chatgpt.com "GitHub - bhargaviparanjape/clickbait"
[6]: https://arxiv.org/abs/1612.01340?utm_source=chatgpt.com "We used Neural Networks to Detect Clickbaits: You won't believe what happened Next!"
[7]: https://cir.nii.ac.jp/crid/1390285300166384768?utm_source=chatgpt.com "Click-Through Rate Prediction with Confidence for News Headlines Using Natural Gradient Boosting | CiNii Research"
[8]: https://www.sciencedirect.com/science/article/abs/pii/S037872062500134X?utm_source=chatgpt.com "Investigating the effects of clickbait on user engagement in health communication: A mixed-method study - ScienceDirect"
[9]: https://academic.oup.com/biomet/article-abstract/39/3-4/324/326091?utm_source=chatgpt.com "RANK ANALYSIS OF INCOMPLETE BLOCK DESIGNS | Biometrika | Oxford Academic"
[10]: https://brocku.ca/MeadProject/Thurstone/Thurstone_1927f.html?utm_source=chatgpt.com "L. L. Thurstone: A Law of Comparative Judgment"
[11]: https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/?utm_source=chatgpt.com "Learning to Rank using Gradient Descent - Microsoft Research"
[12]: https://www.microsoft.com/en-us/research/publication/learning-to-rank-from-pairwise-approach-to-listwise-approach/?utm_source=chatgpt.com "Learning to Rank: From Pairwise Approach to Listwise Approach - Microsoft Research"
[13]: https://proceedings.mlr.press/v48/zhaob16.html?utm_source=chatgpt.com "Learning Mixtures of Plackett-Luce Models"




1. **Gligorić, Lifchits, West, Anderson — “Linguistic effects on news headline success: Evidence from thousands of online field experiments” (PLOS ONE, Registered Report / final paper on PMC)**

* **Dataset:** Upworthy Research Archive (headline A/B tests).
* **Model:** **Logistic regression** to predict which headline in a pair wins.
* **Feature importance:** They **interpret the logistic regression coefficients** (with CIs / significance) as which linguistic features are associated with higher odds of winning. ([PMC][1])

2. **Banerjee & Urminsky — “A Systematic Large-scale Analysis of Headline Experiments” (paper PDF)**

* **Dataset:** Upworthy Research Archive (thousands of field A/B tests).
* **Model:** A **linear regression / linear probability-style model** with experiment fixed effects; they also use **LASSO** for variable selection.
* **Feature importance:** They report **regression coefficients** (including LASSO-selected ones) as the effects/importance of headline constructs on click-through rate. 


[1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10038272/ "
            Linguistic effects on news headline success: Evidence from thousands of online field experiments (Registered Report) - PMC
        "
https://home.uchicago.edu/ourminsky/Banerjee_Urminsky_Headlines.pdf


1. **Frick & Li (2016) — “Personalization in Social Retargeting – A Field Experiment”**

* **A/B setup:** users were **randomly assigned** to see **product-specific** vs **category-specific** retargeting ads on Facebook; also analyzes **social targeting** effects. ([RePub][1])
* **Model:** they run **logistic regressions for purchase probability** (see their “Logistic Regressions for Purchase Probabilities” table). ([RePub][1])
* **“Feature importances”:** the **regression coefficients** (and their significance) on indicators like personalization type / social targeting / controls are interpreted as which factors most move purchase probability. ([RePub][1])

2. **Huang et al. (forthcoming in ISR; PDF preprint) — “A Randomized Field Experiment on the Ex-ante … [registration request]”**

* **A/B setup:** a website design intervention (ex-ante registration request) evaluated via a **randomized field experiment**. ([faculty.marshall.usc.edu][2])
* **Model:** they estimate treatment effects with regression-style models (they report estimated effects/parameters over outcomes like registration and downstream purchasing). ([faculty.marshall.usc.edu][2])
* **“Feature importances”:** coefficients on the **treatment indicator** (and interactions/covariates if included) function as importances for what drives the outcome under the experimental setup. ([faculty.marshall.usc.edu][2])

If you want this to be *closer to your Upworthy-style “text features → coefficients”* (i.e., n-grams / topic features / writing-style features learned from many A/B headline variants), tell me the domain (email subject lines, ads, UX copy, etc.) and I’ll pull 1–2 that are explicitly **text-feature linear models on A/B message variants**.

[1]: https://repub.eur.nl/pub/100005/REPUB_100005.pdf "Personalization in Social Retargeting â•fi A Field Experiment"
[2]: https://faculty.marshall.usc.edu/jinchi-lv/publications/ISR-HMSLG21.pdf "Microsoft Word - SSRN_Login_ISR_Manuscript_final.docx"



## The paper

* **Movva, Peng, Garg, Kleinberg, Pierson (ICML 2025)** — *Sparse Autoencoders for Hypothesis Generation* (HypotheSAEs). ([Proceedings of Machine Learning Research][1])
  Core idea: train an SAE on text embeddings → fit **L1 linear/logistic regression** on SAE activations to pick predictive features → use an LLM to label/interpret those features.

(They also released a HF dataset repo for the experiments.) ([Hugging Face][2])

## Work backwards (what it’s built out of)

### 1) “SAEs give interpretable, sparse features”

These are basically the mechanistic-interpretability lineage that made SAEs “a thing”:

* **Cunningham et al. (2023)** — *Sparse Autoencoders Find Highly Interpretable Features in Language Models*. ([arXiv][3])
* **Bricken et al. / Anthropic (2023)** — *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning*. ([Transformer Circuits][4])
* **O’Neill et al. (2024)** — *Disentangling Dense Embeddings with Sparse Autoencoders* (important because it’s SAEs on **embeddings**, not internal residual streams). ([arXiv][5])

### 2) “k-sparse / dictionary-learning style autoencoders”

This is the older “algorithmic primitive” underlying the HypotheSAEs-style SAE:

* **Makhzani & Frey (2014)** — *k-Sparse Autoencoders*. ([CatalyzeX][6])

### 3) “Hypothesis generation from text labels” (LLM era)

This is the closest “same problem, different approach” line they’re explicitly positioning against:

* **Zhou et al. (2024)** — *Hypothesis Generation with Large Language Models* (iterative hypothesis updating from labeled examples). ([arXiv][7])
* (Nearby extension) **Liu et al. (2024)** — *Literature Meets Data* (mix literature + dataset-driven HG). ([arXiv][8])

### 4) “Classic: discover predictive themes/topics from text”

The pre-LLM backbone: topic models / embedding-topic hybrids:

* **Blei, Ng, Jordan (2003)** — *Latent Dirichlet Allocation*. 
* **Grootendorst (2022)** — *BERTopic*. ([arXiv][9])
* And a “reality check” paper they actually cite in their dataset card lineage: **Hoyle et al. (2022)** — *Are Neural Topic Models Broken?* ([Hugging Face][10])

---
[1]: https://proceedings.mlr.press/v267/movva25a.html?utm_source=chatgpt.com "Sparse Autoencoders for Hypothesis Generation"
[2]: https://huggingface.co/datasets/rmovva/HypotheSAEs/blob/main/README.md?utm_source=chatgpt.com "README.md · rmovva/HypotheSAEs at main"
[3]: https://arxiv.org/abs/2309.08600?utm_source=chatgpt.com "Sparse Autoencoders Find Highly Interpretable Features in Language Models"
[4]: https://transformer-circuits.pub/2023/monosemantic-features/index.html?utm_source=chatgpt.com "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning"
[5]: https://arxiv.org/abs/2408.00657?utm_source=chatgpt.com "Disentangling Dense Embeddings with Sparse Autoencoders"
[6]: https://www.catalyzex.com/paper/k-sparse-autoencoders?utm_source=chatgpt.com "k-Sparse Autoencoders"
[7]: https://arxiv.org/abs/2404.04326?utm_source=chatgpt.com "Hypothesis Generation with Large Language Models"
[8]: https://arxiv.org/abs/2410.17309?utm_source=chatgpt.com "Literature Meets Data: A Synergistic Approach to Hypothesis Generation"
[9]: https://arxiv.org/abs/2203.05794?utm_source=chatgpt.com "BERTopic: Neural topic modeling with a class-based TF-IDF procedure"
[10]: https://huggingface.co/papers/2210.16162?utm_source=chatgpt.com "Paper page - Are Neural Topic Models Broken?"
