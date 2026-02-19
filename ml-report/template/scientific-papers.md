# Scientific Paper Writing Rules

This file defines rules and preferences for writing scientific papers that Claude Code should follow.

## Paper Structure

### Standard Section Organization

```
Title
Authors and Affiliations
Abstract
Keywords/Index Terms
1. Introduction
2. Related Work
3. Methodology / Proposed Approach
4. Experimental Setup / Data
5. Results
6. Discussion (optional, can merge with Results)
7. Conclusion
Acknowledgments
References
```

### Section Guidelines

**Introduction**
- Start with broad context, narrow to specific problem
- State the research gap clearly
- End with contributions list (explicitly numbered)
- Keep to 1-2 pages

**Related Work**
- Group by themes, not chronologically
- Compare and contrast approaches
- End each paragraph showing how your work differs
- Use phrases: "In contrast to...", "Unlike previous approaches...", "Building upon..."

**Methodology**
- Present in logical order, not chronological
- Define all notation before use
- Include algorithm pseudocode for complex procedures
- Justify design decisions

**Results**
- Lead with main findings
- Reference all tables/figures explicitly in text
- Report statistical significance where applicable
- Compare against baselines quantitatively

**Conclusion**
- Summarize contributions (mirror introduction)
- State limitations honestly
- Suggest future work directions
- Keep concise (half page typical)

## Abstract Writing

### Structure (Single Paragraph, 150-250 words)

Follow this 5-part formula:

1. **Context** (1-2 sentences): Broad problem area
2. **Gap** (1 sentence): What's missing in current approaches
3. **Approach** (2-3 sentences): Your proposed solution
4. **Results** (2-3 sentences): Key quantitative findings
5. **Impact** (1 sentence): Significance or availability

### Example Pattern

```
[Context] Modern X systems achieve Y but lack Z.
[Gap] Existing approaches fail to address W.
[Approach] We propose METHOD that combines A and B to achieve C.
We evaluate on DATASET across N conditions.
[Results] Results show improvements of X% over baselines on metric M.
Our approach achieves state-of-the-art performance on BENCHMARK.
[Impact] Code is publicly available at URL.
```

### Abstract Rules

- No citations in abstract
- No undefined acronyms (define on first use or avoid)
- No mathematical notation unless essential
- Self-contained (readable without paper)
- Use active voice: "We propose..." not "A method is proposed..."

## Writing Style

### Voice and Tense

- **Methods**: Past tense ("We trained the model...")
- **Results**: Past tense ("The model achieved...")
- **General truths**: Present tense ("Neural networks are...")
- **Contributions**: Present tense ("We propose...", "This paper presents...")

Use "we" even for single-author papers (editorial we).

### Precision and Clarity

**Be specific:**
```
❌ "The model performs well"
✅ "The model achieves 94.2% accuracy on the test set"

❌ "significantly better"
✅ "3.2 percentage points higher (p < 0.01)"

❌ "large dataset"
✅ "dataset of 1.2M examples"
```

**Avoid hedging without evidence:**
```
❌ "This might suggest that..."
✅ "This indicates that..." (if supported by evidence)
✅ "We hypothesize that..." (if speculative)
```

### Sentence Structure

- Lead with the main point
- One idea per sentence
- Keep sentences under 25 words when possible
- Vary sentence length for rhythm

**Paragraph structure:**
- Topic sentence first
- Supporting evidence/details
- Connection to next paragraph or section

### Common Patterns

**Introducing contributions:**
```
"The main contributions of this paper are:
(1) We propose...
(2) We demonstrate...
(3) We release..."
```

**Comparing to baselines:**
```
"Compared to METHOD, our approach achieves X% higher accuracy
while requiring Y% less computation."
```

**Stating limitations:**
```
"Our approach assumes X, which may not hold in scenarios where Y.
Future work could address this by Z."
```

## Technical Writing

### Format Selection

**Prefer LaTeX** for scientific papers. Use format-native symbols and notation.

| Format | Use Case |
|--------|----------|
| LaTeX  | Journal/conference submissions, complex math |
| Markdown | Quick drafts, README files, documentation |
| Plain text | Abstracts, submission forms |

### Notation

- Define all symbols on first use
- Use consistent notation throughout
- Prefer standard notation from the field
- Use format-appropriate rendering (never ASCII approximations)

**LaTeX notation conventions:**
| Element | LaTeX | Rendered |
|---------|-------|----------|
| Vectors | `\mathbf{x}` or `\vec{x}` | **x** or x⃗ |
| Matrices | `\mathbf{M}` or `\mathbf{A}` | **M** |
| Sets | `\mathcal{S}` | 𝒮 |
| Real numbers | `\mathbb{R}` | ℝ |
| Expectations | `\mathbb{E}` | 𝔼 |
| Loss functions | `\mathcal{L}` | ℒ |
| Norms | `\| \mathbf{x} \|` | ‖**x**‖ |
| Partial derivatives | `\frac{\partial}{\partial x}` | ∂/∂x |

### Mathematical Symbols

**Always use proper symbols, never ASCII substitutes:**

```
❌ ASCII approximations     ✅ Proper symbols (LaTeX)
   ->                          \rightarrow (→)
   >=                          \geq (≥)
   <=                          \leq (≤)
   !=                          \neq (≠)
   ~=                          \approx (≈)
   alpha, beta                 \alpha, \beta (α, β)
   sum                         \sum (∑)
   prod                        \prod (∏)
   inf                         \infty (∞)
   in                          \in (∈)
   sqrt()                      \sqrt{} (√)
   x^2                         x^{2} (x²)
   x_i                         x_{i} (xᵢ)
```

### Equations

- Number important equations for reference
- Introduce equations with a colon or "as follows:"
- Explain variables immediately after equation
- Reference equations as "Equation (1)" or "Eq. (1)"

**LaTeX equation example:**
```latex
The cross-entropy loss is defined as:
\begin{equation}
    \mathcal{L} = -\sum_{i=1}^{N} y_i \log(\hat{y}_i)
    \label{eq:cross-entropy}
\end{equation}
where $y_i \in \{0, 1\}$ is the ground truth label and
$\hat{y}_i \in [0, 1]$ is the predicted probability.
```

**Inline vs display math:**
```latex
% Inline: for simple expressions within text
The probability $p(x) = \frac{1}{Z} \exp(-E(x))$ represents...

% Display: for important or complex equations
\begin{equation}
    p(x) = \frac{1}{Z} \exp\left(-\frac{E(x)}{kT}\right)
\end{equation}
```

### Common LaTeX Patterns

**Optimization problems:**
```latex
\begin{equation}
    \min_{\theta} \mathcal{L}(\theta) =
    \frac{1}{N} \sum_{i=1}^{N} \ell(f_\theta(x_i), y_i) +
    \lambda \|\theta\|_2^2
\end{equation}
```

**Probability distributions:**
```latex
\begin{equation}
    p(y \mid x) = \frac{p(x \mid y) p(y)}{p(x)} =
    \frac{p(x \mid y) p(y)}{\sum_{y'} p(x \mid y') p(y')}
\end{equation}
```

**Matrices and vectors:**
```latex
\begin{equation}
    \mathbf{h} = \sigma(\mathbf{W} \mathbf{x} + \mathbf{b})
\end{equation}
where $\mathbf{W} \in \mathbb{R}^{d \times n}$ is the weight matrix,
$\mathbf{x} \in \mathbb{R}^n$ is the input, and
$\mathbf{b} \in \mathbb{R}^d$ is the bias vector.
```

**Norms and distances:**
```latex
% L2 norm
\|\mathbf{x}\|_2 = \sqrt{\sum_{i} x_i^2}

% Frobenius norm
\|\mathbf{A}\|_F = \sqrt{\sum_{i,j} A_{ij}^2}

% KL divergence
D_{\mathrm{KL}}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}
```

### LaTeX Packages for ML/NLP Papers

```latex
\usepackage{amsmath}      % Core math environments
\usepackage{amssymb}      % Mathematical symbols (ℝ, ∈, etc.)
\usepackage{amsfonts}     % Additional fonts
\usepackage{mathtools}    % Extensions to amsmath
\usepackage{bm}           % Bold math symbols (\bm{x})
\usepackage{algorithm}    % Algorithm environment
\usepackage{algorithmic}  % Algorithm pseudocode
\usepackage{booktabs}     % Professional tables (\toprule, \midrule, \bottomrule)
\usepackage{subcaption}   % Subfigures
\usepackage{hyperref}     % Clickable cross-references
\usepackage{cleveref}     % Smart references (\cref, \Cref)
```

### Cross-References

**Use `\label` and `\ref` for all numbered elements:**

```latex
% Defining labels
\section{Introduction} \label{sec:intro}
\begin{equation} ... \label{eq:loss} \end{equation}
\begin{figure} ... \label{fig:arch} \end{figure}
\begin{table} ... \label{tab:results} \end{table}
\begin{algorithm} ... \label{alg:train} \end{algorithm}

% Referencing
Section~\ref{sec:intro}     % → Section 1
Equation~\ref{eq:loss}      % → Equation (1)
Figure~\ref{fig:arch}       % → Figure 1
Table~\ref{tab:results}     % → Table 1
Algorithm~\ref{alg:train}   % → Algorithm 1
```

**With cleveref (recommended):**
```latex
\cref{sec:intro}            % → section 1 (auto-detects type)
\Cref{sec:intro}            % → Section 1 (capitalized)
\cref{fig:arch,fig:model}   % → figures 1 and 2
\cref{eq:loss,eq:grad}      % → equations (1) and (2)
```

### Label Naming Conventions

Use consistent prefixes for labels:
- `sec:` for sections
- `eq:` for equations
- `fig:` for figures
- `tab:` for tables
- `alg:` for algorithms
- `thm:` for theorems
- `def:` for definitions

### Figures and Tables

**Figures:**
- Caption below figure
- Self-contained captions (readable without text)
- Reference as "Figure 1" or "Fig. 1"
- Use vector graphics when possible (PDF, SVG)

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{figures/architecture.pdf}
    \caption{Overview of the proposed architecture. The encoder
    transforms input $\mathbf{x}$ into latent representation
    $\mathbf{z}$, which is decoded to output $\hat{\mathbf{y}}$.}
    \label{fig:architecture}
\end{figure}
```

**Tables:**
- Caption above table
- Align numbers by decimal point
- Bold best results
- Include standard deviation/confidence intervals (±)
- Reference as "Table 1"

```latex
\begin{table}[t]
    \centering
    \caption{Results on benchmark datasets. Best results in \textbf{bold}.}
    \label{tab:results}
    \begin{tabular}{lcccc}
        \toprule
        Method & Accuracy & F1 & Precision & Recall \\
        \midrule
        Baseline & 82.3 {\scriptsize ±0.4} & 79.1 & 80.2 & 78.1 \\
        Previous SOTA & 85.7 {\scriptsize ±0.3} & 83.4 & 84.1 & 82.8 \\
        \textbf{Ours} & \textbf{88.2} {\scriptsize ±0.2} & \textbf{86.1} & \textbf{87.3} & \textbf{85.0} \\
        \bottomrule
    \end{tabular}
\end{table}
```

### Algorithms

Use `algorithm` and `algorithmic` packages for pseudocode:

```latex
\begin{algorithm}[t]
    \caption{Training procedure for proposed method}
    \label{alg:training}
    \begin{algorithmic}[1]
        \REQUIRE Dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$,
                 learning rate $\eta$, epochs $T$
        \ENSURE Trained parameters $\theta^*$
        \STATE Initialize $\theta$ randomly
        \FOR{$t = 1$ \TO $T$}
            \FOR{$(x, y) \in \mathcal{D}$}
                \STATE $\hat{y} \leftarrow f_\theta(x)$
                \STATE $\mathcal{L} \leftarrow \ell(\hat{y}, y)$
                \STATE $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$
            \ENDFOR
        \ENDFOR
        \RETURN $\theta$
    \end{algorithmic}
\end{algorithm}
```

## Citations

### LaTeX Citation Commands

```latex
% Numeric style (IEEE, ACM)
\cite{smith2024}                    % → [1]
\cite{smith2024,jones2023}          % → [1], [2]

% Author-year style (Elsevier, natbib)
\citep{smith2024}                   % → (Smith, 2024)
\citet{smith2024}                   % → Smith (2024)
\citep{smith2024,jones2023}         % → (Smith, 2024; Jones, 2023)
\citeauthor{smith2024}              % → Smith
\citeyear{smith2024}                % → 2024
```

### Inline Citation Style

**Numeric (IEEE style):**
```latex
Recent work~\cite{vaswani2017} shows that transformers
outperform RNNs~\cite{hochreiter1997,cho2014}.
As demonstrated by Smith~et~al.~\cite{smith2024}, this approach...
```

**Author-year (Elsevier style):**
```latex
Recent work \citep{smith2023} shows that transformers outperform
RNNs \citep{jones2022,lee2023}. As demonstrated by \citet{smith2024}...
```

### Citation Guidelines

- Cite primary sources, not reviews (when possible)
- Include recent work (last 2-3 years)
- Cite your own relevant work (but don't over-cite)
- Verify citations are accurate
- Use "et al." for 3+ authors
- Use non-breaking space before citations: `work~\cite{}`

### BibTeX Entry Formats

**Journal article:**
```bibtex
@article{smith2024method,
    author    = {Smith, Alice and Jones, Bob},
    title     = {Title of the Paper},
    journal   = {Journal Name},
    volume    = {12},
    number    = {3},
    pages     = {100--115},
    year      = {2024},
    doi       = {10.1000/example}
}
```

**Conference paper:**
```bibtex
@inproceedings{smith2024conf,
    author    = {Smith, Alice},
    title     = {Title of the Paper},
    booktitle = {Proceedings of Conference Name},
    pages     = {50--60},
    year      = {2024},
    address   = {City, Country}
}
```

**Preprint:**
```bibtex
@article{smith2024arxiv,
    author    = {Smith, Alice},
    title     = {Title of the Paper},
    journal   = {arXiv preprint arXiv:2024.12345},
    year      = {2024}
}
```

## Keywords

- 3-5 keywords/index terms
- Mix broad and specific terms
- Include method names if novel
- Consider search discoverability

```
Keywords: cross-lingual retrieval, language models,
Earth Mover's Distance, document ranking, interpretability
```

## Common Mistakes to Avoid

### Language

- ❌ "very unique" (unique is absolute)
- ❌ "in order to" → ✅ "to"
- ❌ "due to the fact that" → ✅ "because"
- ❌ "a number of" → ✅ "several" or specific number
- ❌ "utilize" → ✅ "use"
- ❌ "methodology" (when you mean "method")
- ❌ Starting sentences with "It is" or "There are"

### Structure

- ❌ Burying main contribution in middle of intro
- ❌ Results without comparison to baselines
- ❌ Figures/tables not referenced in text
- ❌ Related work that doesn't connect to your work
- ❌ Conclusion that introduces new information

### Claims

- ❌ Overclaiming ("revolutionary", "breakthrough")
- ❌ Unsupported comparisons ("better than all existing methods")
- ❌ Ignoring limitations
- ❌ Cherry-picking results

## Reproducibility

Include in paper or supplementary:

- [ ] Dataset details (size, splits, preprocessing)
- [ ] Hyperparameters (learning rate, batch size, epochs)
- [ ] Hardware used (GPU type, training time)
- [ ] Random seeds or variance across runs
- [ ] Code availability statement
- [ ] License for code/data

```
"Experiments were conducted on NVIDIA A100 GPUs.
Training took approximately 12 hours.
Code is available at https://github.com/user/repo
under the MIT license."
```

## Checklist Before Submission

### Content
- [ ] Abstract follows 5-part structure
- [ ] Contributions clearly stated in introduction
- [ ] All claims supported by evidence
- [ ] Limitations discussed
- [ ] All figures/tables referenced in text

### Formatting
- [ ] Consistent notation throughout
- [ ] All acronyms defined on first use
- [ ] References complete and correctly formatted
- [ ] Page limit respected
- [ ] Figures legible when printed in grayscale

### LaTeX-Specific
- [ ] No undefined references (`\ref` warnings)
- [ ] No missing citations (`\cite` warnings)
- [ ] All labels use consistent naming (`sec:`, `eq:`, `fig:`, etc.)
- [ ] Proper math symbols used (no ASCII substitutes)
- [ ] Non-breaking spaces before references (`~\ref`, `~\cite`)
- [ ] BibTeX entries complete (no missing fields)
- [ ] Compiles without errors or warnings

### Quality
- [ ] No grammar/spelling errors
- [ ] No orphaned headings (heading at bottom of page)
- [ ] Consistent capitalization in headings
- [ ] All authors and affiliations correct
- [ ] Acknowledgments include funding sources
