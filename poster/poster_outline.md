# Dynamic Embeddings for Temporal Language Change

## Motivation

Word meanings change over time. Static embeddings trained on a single corpus mix old and new meanings and cannot show when semantic change happened.

## Research Questions

- How do selected technology-related words change across periods?
- Which words show the strongest semantic drift?
- Do technology-related words shift more than stable control words?

## Dataset

Timestamped English text corpus with `year` and `text` columns. The intended corpus is scientific or technology-related abstracts, split into decade-level periods.

## Methodology

Preprocessing -> time slicing -> Word2Vec Skip-gram per period -> Orthogonal Procrustes alignment -> cosine and Jaccard shift metrics -> visualizations.

## Experimental Setup

- Periods: 1990s, 2000s, 2010s, 2020s
- Embedding model: Word2Vec Skip-gram
- Vector size: 100
- Window: 5
- Minimum count: 20
- Epochs: 10
- Alignment: Orthogonal Procrustes using shared frequent anchor words

## Results

Add:

- Line plot of cosine shift scores.
- Heatmap of target words by period.
- Target vs. control average shift comparison.
- Nearest-neighbor examples for cloud, token, virus, stream.

## Qualitative Analysis

Example interpretation:

The word `cloud` is expected to move from weather-related neighbors toward computing-related neighbors such as server, storage, platform, and service.

## Conclusions

Dynamic embeddings can capture interpretable temporal semantic shifts. Technology-related words are expected to show stronger drift than stable control words.

## Limitations

- Corpus domain bias.
- Uneven period sizes.
- Word2Vec instability on small corpora.
- Rare target words may be missing from some periods.
