# Current Results

This file summarizes the current working version of the project.

## Implementation Status

- Repository structure for the temporal semantic shift project is created.
- Four Jupyter notebooks are prepared:
  - `01_data_preparation.ipynb`
  - `02_train_embeddings.ipynb`
  - `03_alignment_and_analysis.ipynb`
  - `04_visualizations.ipynb`
- Core source modules are implemented:
  - preprocessing and period assignment
  - Word2Vec training
  - Orthogonal Procrustes alignment
  - cosine and nearest-neighbor Jaccard shift metrics
- Poster outline is drafted in `poster/poster_outline.md`.
- Helper scripts are present:
  - `1.py`: converts News Category Dataset JSONL into `data/raw/corpus.csv`
  - `2.py`: checks target/control word frequencies

## Experimental Results Snapshot

Existing CSV outputs are stored in `results/`:

- `results/cosine_shift.csv`
- `results/group_shift.csv`
- `results/nearest_neighbors.csv`
- `results/neighbor_shift.csv`

Current period coverage in the result files is `2010s` and `2020s`.

### Cosine Shift From Base Period

The strongest available 2020s cosine shifts among inspected words are:

| word | 2020s cosine distance |
| --- | ---: |
| model | 0.541 |
| network | 0.535 |
| platform | 0.530 |
| water | 0.544 |
| animal | 0.546 |
| winter | 0.437 |
| city | 0.392 |
| virus | 0.346 |

Some words are missing in the 2020s model vocabulary under the current settings, including `web` and `stream`.

### Target vs. Control

Average cosine shift in the current result snapshot:

| group | 2010s | 2020s |
| --- | ---: | ---: |
| target | 0.000 | 0.488 |
| control | 0.000 | 0.494 |

At this stage, target words do not yet show a clearly higher average shift than control words. This likely reflects corpus/domain effects, limited period coverage, and missing target vocabulary in the 2020s slice.

### Nearest-Neighbor Examples

`virus` changes from medical and disease-specific neighbors in the 2010s:

```text
mosquito-borne, zika, influenza, sars-like, vaccine, transmission, transmitted, outbreaks, drug-resistant, infections
```

to more news/COVID-era neighbors in the 2020s:

```text
disease, cases, outbreak, surge, low, infections, number, omicron, rise, china
```

`model` changes from fashion/modeling-related neighbors in the 2010s:

```text
models, lawley, supermodel, plus-size, img, daria, vodianova, narciso, mag, lingerie
```

to celebrity/news-style neighbors in the 2020s:

```text
stunning, heartbreaking, amazing, mogul, hadid, grammy, alongside, dennis, dubbed, career
```

## Interpretation

The current results show that the pipeline works end to end, but the current corpus/results behave more like a news-domain semantic analysis than a clean technology-focused semantic shift study.

Before final poster/report conclusions, the project needs either:

- a larger and more balanced timestamped corpus,
- adjusted target words for the News Category Dataset,
- lower `min_count` for sparse target words,
- or a more technology/science-focused dataset.
