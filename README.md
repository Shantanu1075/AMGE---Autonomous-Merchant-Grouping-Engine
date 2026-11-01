# Autonomous Merchant Grouping Engine (AMGE)

**A transaction enrichment pipeline for merchant identification, text normalization, and unsupervised merchant grouping.**

## Overview

Financial transaction descriptions often identify the same merchant in different ways. For example, `AMAZON MUMBAI IN`, `AMAZON GURGAON IN`, and `AMAZON INDIA` may refer to the same merchant despite differences in location and formatting.

The **Autonomous Merchant Grouping Engine (AMGE)** processes transaction descriptions, extracts merchant-related patterns, and groups transactions using a combination of text features, semantic embeddings, and unsupervised clustering.

The pipeline also uses feedback-driven noise-token discovery to refine subsequent clustering passes and a Bayesian-inspired audit to estimate cluster consistency and prioritize manual review.

### Key Features

- **Text normalization:** Standardizes transaction descriptions by removing URLs, punctuation, and long numeric identifiers.
- **Compound token splitting:** Uses corpus-derived token patterns to identify and split selected concatenated words.
- **Hybrid text representation:** Combines character-level TF-IDF features with SentenceTransformer embeddings.
- **Adaptive clustering:** Evaluates multiple agglomerative clustering thresholds using silhouette scores.
- **Merchant root discovery:** Uses token frequency and association statistics to identify candidate merchant roots and distinguish meaningful sub-brand patterns.
- **Canonical cluster assignment:** Assigns unified cluster IDs based on extracted merchant roots.
- **Iterative feedback:** Discovers potentially generic tokens and feeds them back into feature extraction.
- **Cluster audit:** Calculates Bayesian-smoothed consistency estimates, credible intervals, and review categories.
- **Merchant distribution analysis:** Visualizes the most frequently assigned merchant labels.

## Pipeline Architecture

AMGE is organized into seven processing layers, followed by a cluster audit.

| Stage | Responsibility |
|---|---|
| Layer 1 — Structural Normalization | Cleans and standardizes transaction text. |
| Layer 2 — Compound Token Normalization | Splits selected concatenated tokens using corpus-derived patterns. |
| Layers 3–4 — Hybrid Features and Distance Matrix | Combines character n-gram TF-IDF and semantic embeddings, then calculates cosine distances. |
| Layer 5 — Adaptive Clustering | Searches distance thresholds and selects a candidate using silhouette scores. |
| Layer 6.1 — Merchant Root Discovery | Identifies merchant roots and associated token patterns. |
| Layer 6.2 — Cluster Consolidation | Assigns canonical cluster IDs using extracted merchant roots. |
| Layer 7 — Feedback Controller | Repeats feature extraction, clustering, and root discovery when new noise tokens are found. |
| Layer 7.2 — Bayesian Cluster Audit | Estimates within-cluster root consistency and assigns review categories. |

## Techniques Worth Exploring

### 1. Regular-expression text normalization

The preprocessing stage uses Python's `re` module to normalize transaction descriptions, remove URLs and punctuation, and filter long numeric identifiers.

Regular expressions are useful for handling inconsistent transaction formats before feature extraction.

Documentation: [Python `re` module](https://docs.python.org/3/library/re.html) · [MDN Regular Expressions Guide](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Regular_expressions)

### 2. Corpus-derived compound token splitting

The compound-token stage uses token frequencies from the dataset to identify candidate substrings and split selected concatenated tokens. This reduces dependence on a manually maintained merchant dictionary.

The approach is heuristic: its results depend on the vocabulary and frequency patterns present in the input data.

### 3. Hybrid lexical and semantic representations

AMGE combines two complementary representations:

- **Character-level TF-IDF:** Captures textual patterns, partial matches, and spelling variations using character n-grams.
- **SentenceTransformer embeddings:** Represents transaction descriptions as dense numerical vectors intended to capture semantic similarities.

The resulting sparse and dense feature blocks are combined and normalized before pairwise cosine distances are calculated.

This design combines surface-level text matching with semantic information rather than relying on either representation alone.

Documentation: [scikit-learn — TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) · [Sentence Transformers](https://www.sbert.net/) · [scikit-learn — cosine distances](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_distances.html)

### 4. Threshold selection using silhouette scores

Instead of fixing a single clustering threshold, the clustering stage evaluates multiple candidate thresholds and selects the candidate with the highest valid silhouette score.

Silhouette scores measure how well samples fit their assigned clusters compared with neighboring clusters. They provide an internal clustering diagnostic, not proof that merchant assignments are correct.

Documentation: [scikit-learn — AgglomerativeClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html) · [scikit-learn — silhouette_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)

### 5. Statistical merchant root discovery

The root-discovery stage analyzes token positions, frequencies, successor patterns, and token associations. A graph-inspired association heuristic helps decide whether a transaction should receive a one-token or two-token merchant root.

The extracted roots are subsequently used to assign canonical cluster IDs.

### 6. Feedback-driven refinement

The feedback controller identifies tokens appearing across multiple merchant roots and adds qualifying tokens to a noise mask. It then reruns feature extraction, clustering, and root discovery with the updated mask.

The process terminates when no new noise tokens are discovered or the configured score-change condition is met.

This makes the pipeline iterative rather than a single-pass clustering workflow.

### 7. Bayesian-smoothed cluster consistency

The audit uses a Beta prior and posterior mean to estimate dominant-root consistency for each cluster. It also calculates a 95% credible interval and assigns a review category.

These estimates should be interpreted as **internal consistency indicators, not independently validated probabilities of merchant-label correctness**. Because cluster IDs are assigned from extracted roots, the audit is not an independent measure of clustering accuracy.

## Technologies and Libraries

| Technology | Role |
|---|---|
| [Python](https://www.python.org/) | Core implementation and pipeline orchestration |
| [Pandas](https://pandas.pydata.org/) | Transaction data manipulation and CSV export |
| [NumPy](https://numpy.org/) | Numerical operations and threshold generation |
| [scikit-learn](https://scikit-learn.org/) | TF-IDF vectorization, normalization, agglomerative clustering, silhouette scoring, and cosine distances |
| [Sentence Transformers](https://www.sbert.net/) | Semantic embeddings for transaction descriptions |
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Pretrained sentence-embedding model used by the notebook |
| [SciPy](https://scipy.org/) | Sparse matrix operations and Beta-distribution calculations |
| [Matplotlib](https://matplotlib.org/) | Plot generation |
| [Seaborn](https://seaborn.pydata.org/) | Merchant distribution visualization |
| [Jupyter Notebook](https://jupyter.org/) | Interactive development and execution of the pipeline |

**Fonts:** The notebook does not specify any external fonts.

## Project Structure

```text
AMGE/
├── README.md
├── amge.ipynb
└── cleaned_merchant_clusters.csv
```

- **`amge.ipynb`** — Contains the transaction dataset, preprocessing functions, feature engineering, clustering, merchant root discovery, feedback controller, Bayesian audit, and visualization.
- **`cleaned_merchant_clusters.csv`** — Generated transaction-level output containing the final merchant assignments. This file is created when the notebook's CSV export cell runs.
- **`README.md`** — Project overview, architecture, techniques, and technology references.

The notebook currently embeds its demonstration transaction descriptions directly in the code. The CSV output is generated by the notebook rather than supplied as a separate input dataset.

## Evaluation and Limitations

AMGE uses unsupervised clustering metrics and root-based consistency estimates to inspect its results without requiring a manually labeled training dataset.

Several limitations should be considered:

- Silhouette scores measure geometric separation, not merchant identity correctness.
- Merchant roots and noise tokens are inferred using heuristics that may not generalize to every transaction format.
- Root-based cluster consolidation can propagate errors from incorrect root extraction.
- The Bayesian audit measures consistency with extracted roots; independent labeled data or manual validation is needed to establish actual assignment accuracy.
- Results depend on the input transaction descriptions, selected model, and clustering configuration.

These limitations make independent validation and representative transaction data important when evaluating the engine for production use.