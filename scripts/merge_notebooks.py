from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT / "notebooks"

input_files = [
    NOTEBOOKS_DIR / "01_data_preparation.ipynb",
    NOTEBOOKS_DIR / "02_train_embeddings.ipynb",
    NOTEBOOKS_DIR / "03_alignment_and_analysis.ipynb",
    NOTEBOOKS_DIR / "04_visualizations.ipynb",
]

output_file = NOTEBOOKS_DIR / "final_case_study.ipynb"

section_titles = [
    "# 1. Data Preparation",
    "# 2. Training Temporal Word2Vec Models",
    "# 3. Alignment and Semantic Shift Analysis",
    "# 4. Visualization and Results",
]

merged = nbf.v4.new_notebook()
merged.cells = []

intro = """
# Dynamic Embeddings for Temporal Language Change

This notebook contains the full pipeline for the case study:

1. Data preparation
2. Word2Vec training for temporal slices
3. Embedding alignment using Orthogonal Procrustes
4. Semantic shift analysis
5. Visualization of results

Dataset: News Category Dataset  
Periods: 2010s and 2020s  
Main target words: model, network, platform, virus
"""

merged.cells.append(nbf.v4.new_markdown_cell(intro))

for title, path in zip(section_titles, input_files):
    nb = nbf.read(path, as_version=4)

    merged.cells.append(nbf.v4.new_markdown_cell(title))

    for cell in nb.cells:
        # Skip duplicate notebook titles if needed
        if cell.cell_type == "markdown" and cell.source.strip().startswith("#"):
            continue
        merged.cells.append(cell)

nbf.write(merged, output_file)

print(f"Saved merged notebook to: {output_file}")