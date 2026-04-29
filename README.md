# Tracking Semantic Change in Technology-Related Words Using Temporal Word Embeddings

This project studies temporal semantic change in English technology-related words using dynamic word embeddings.

The core idea is to split a timestamped corpus into time periods, train one Word2Vec model per period, align the embedding spaces with Orthogonal Procrustes, and compare how target-word vectors and nearest neighbors change over time.

## Research Questions

- How do selected technology-related words change meaning across time?
- Which target words show the strongest semantic drift?
- Do target words shift more than stable control words?

## Target Words

Changing / technology-related words:

```text
cloud, virus, web, tablet, stream, token, model, network, apple, platform
```

Stable control words:

```text
water, table, city, mother, winter, animal
```

## Expected Input Data

Place a CSV file at:

```text
data/raw/corpus.csv
```

Required columns:

| column | description |
| --- | --- |
| `year` | publication year, e.g. `2018` |
| `text` | document text, abstract, or article body |

Example:

```csv
year,text
1999,"Cloud patterns were studied using satellite observations."
2021,"Cloud platforms provide scalable storage and compute resources."
```

## Time Periods

The default periods are:

| Period | Years |
| --- | --- |
| 1990s | 1990-1999 |
| 2000s | 2000-2009 |
| 2010s | 2010-2019 |
| 2020s | 2020-2025 |

These can be changed in the notebooks or in calls to `src.preprocessing.assign_period`.

## Methodology

1. Load a timestamped text corpus.
2. Preprocess text with lowercase tokenization and light cleanup.
3. Split documents into time periods.
4. Train one Skip-gram Word2Vec model per period.
5. Align all later models to the earliest available base period, excluding target and control words from alignment anchors.
6. Measure semantic shift with cosine distance and nearest-neighbor Jaccard distance.
7. Visualize target-word drift and analyze changed neighbors qualitatively.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── data/
│   ├── README.md
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_train_embeddings.ipynb
│   ├── 03_alignment_and_analysis.ipynb
│   └── 04_visualizations.ipynb
├── src/
│   ├── __init__.py
│   ├── alignment.py
│   ├── metrics.py
│   ├── preprocessing.py
│   └── train_embeddings.py
├── models/
│   └── README.md
├── figures/
│   └── README.md
└── poster/
    └── poster_outline.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Workflow

Run the notebooks in order:

1. `notebooks/01_data_preparation.ipynb`
2. `notebooks/02_train_embeddings.ipynb`
3. `notebooks/03_alignment_and_analysis.ipynb`
4. `notebooks/04_visualizations.ipynb`

## Current Limitations

- Results depend strongly on the corpus domain and size.
- Separately trained Word2Vec models can be unstable on small periods.
- Rare words may be missing from some period vocabularies.
- Scientific abstracts capture domain language change, not necessarily general English.
