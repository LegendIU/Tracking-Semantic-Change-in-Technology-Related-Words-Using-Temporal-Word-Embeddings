# Tracking Semantic Change in Technology-Related Words Using Temporal Word Embeddings

This project investigates how word meanings change over time using temporal word embeddings.

The main idea is to train separate Word2Vec models for different time periods, align the embedding spaces, and compare how selected words move between periods. The project focuses on semantic change in news language from the 2010s to the 2020s.

## Project Topic

**Dynamic Embeddings for Temporal Language Change**

This case study uses word embeddings for diachronic semantic analysis. The goal is to track how word meanings evolve across time by comparing nearest neighbors and cosine distances in aligned embedding spaces.

## Research Questions

1. Can temporal word embeddings capture semantic shifts in news language?
2. Which selected words show the strongest semantic change between the 2010s and 2020s?
3. Do technology-related target words shift more than stable control words?
4. What qualitative changes can be observed in the nearest neighbors of shifted words?

## Dataset

The project uses the **News Category Dataset** from Kaggle.

The dataset contains news headlines, short descriptions, categories, and publication dates. For this project, the headline and short description are combined into one text field, and the year is extracted from the publication date.

Final corpus format:

```csv
year,text
2020,"news headline. news short description"
2018,"news headline. news short description"

The prepared corpus is stored as:

```text
data/raw/corpus.csv
```

## Time Periods

The corpus is split into two time periods:

| Period | Years     |
| ------ | --------- |
| 2010s  | 2012–2019 |
| 2020s  | 2020–2022 |

The 2010s period contains significantly more documents than the 2020s period, so this is considered as a limitation of the experiment.

## Target and Control Words

Final target words:

```python
["virus", "platform", "network", "model", "web", "stream"]
```

Control words:

```python
["water", "city", "winter", "animal", "family", "school"]
```

Target words were selected because they are expected to appear in changing technological, media, social, or pandemic-related contexts.

Control words were selected as more stable comparison words.

## Methodology

The workflow consists of the following steps:

1. Load and preprocess the corpus.
2. Split texts into time periods.
3. Train a separate Word2Vec Skip-gram model for each period.
4. Align embedding spaces using Orthogonal Procrustes alignment.
5. Compute cosine distance from the base period.
6. Compare nearest neighbors across periods.
7. Visualize semantic shift scores and group-level differences.

## Models

Word embeddings are trained using Word2Vec Skip-gram.

Main parameters:

```python
vector_size = 100
window = 5
min_count = 5
sg = 1
epochs = 10
seed = 42
```

The trained models are saved in:

```text
models/
```

## Results

The strongest semantic shifts were observed for the following words:

| Word     | Cosine distance in 2020s |
| -------- | -----------------------: |
| model    |                    ~0.54 |
| network  |                    ~0.53 |
| platform |                    ~0.53 |
| virus    |                    ~0.35 |

The results suggest that several words changed their distributional contexts between the 2010s and 2020s.

### Qualitative Interpretation

The word **virus** shifted toward pandemic and disease-related contexts in the 2020s, likely reflecting COVID-19 news coverage.

The words **platform** and **network** shifted toward digital, media, and political contexts.

The word **model** also showed a strong shift, which may reflect changes in how news texts discuss models, predictions, systems, and public events.

However, the average difference between target and control words was modest. This means that the experiment does not fully confirm that target words shift more than control words on average. Instead, the main finding is that individual words show interpretable semantic shifts.

## Output Files

Main result files:

```text
results/cosine_shift.csv
results/group_shift.csv
results/nearest_neighbors.csv
results/neighbor_shift.csv
```

Main figures:

```text
figures/shift_scores.png
figures/shift_heatmap.png
figures/target_vs_control.png
```

## Repository Structure

```text
.
├── app.py
├── README.md
├── RESULTS.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── corpus.csv
│   └── processed/
│       └── corpus_tokenized.pkl
├── figures/
│   ├── shift_scores.png
│   ├── shift_heatmap.png
│   └── target_vs_control.png
├── models/
│   ├── word2vec_2010s.model
│   └── word2vec_2020s.model
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_train_embeddings.ipynb
│   ├── 03_alignment_and_analysis.ipynb
│   └── 04_visualizations.ipynb
├── poster/
│   └── poster_outline.md
├── results/
│   ├── cosine_shift.csv
│   ├── group_shift.csv
│   ├── nearest_neighbors.csv
│   └── neighbor_shift.csv
└── src/
    ├── alignment.py
    ├── metrics.py
    ├── preprocessing.py
    └── train_embeddings.py
```

## How to Run

### 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

If you want to run notebooks:

```bash
pip install ipykernel
python -m ipykernel install --user --name temporal-embeddings --display-name "temporal-embeddings"
```

### 3. Prepare the dataset

Place the Kaggle dataset file in:

```text
data/raw/News_Category_Dataset_v3.json
```

Then run the dataset preparation script or notebook.

Expected final file:

```text
data/raw/corpus.csv
```

### 4. Run notebooks

Run the notebooks in this order:

```text
notebooks/01_data_preparation.ipynb
notebooks/02_train_embeddings.ipynb
notebooks/03_alignment_and_analysis.ipynb
notebooks/04_visualizations.ipynb
```

## Interactive Demo

The project also includes a simple Streamlit application.

Run it with:

```bash
streamlit run app.py
```

The app allows the user to enter a word and view:

1. cosine shift score;
2. nearest neighbors by period;
3. a simple interpretation of the shift strength.

## Limitations

1. The dataset represents news-domain language, not general English.
2. The 2020s period is much smaller than the 2010s period.
3. Word2Vec models can be unstable on smaller corpora.
4. Some words may be missing from a period if they are too rare.
5. Semantic shift scores may reflect changes in corpus composition, not only real language change.
6. Only two broad periods are used, so fine-grained year-by-year semantic change is not analyzed.

## Conclusion

The project shows that temporal word embeddings can be used to detect and visualize semantic change. The strongest shifts were found for **model**, **network**, **platform**, and **virus**. The nearest-neighbor analysis suggests that these shifts are interpretable and connected to changes in news topics, especially digital media and pandemic-related coverage.

At the same time, the difference between target and control words was limited, which shows the importance of corpus balance, word frequency, and careful qualitative interpretation in diachronic embedding studies.
