from pathlib import Path
import textwrap

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")

TARGET_WORDS = ["virus", "platform", "network", "model", "web", "stream"]
CORE_WORDS = ["virus", "platform", "network", "model"]
CONTROL_WORDS = ["water", "city", "winter", "animal", "family", "school"]

cosine_shift = pd.read_csv(RESULTS_DIR / "cosine_shift.csv")
group_shift = pd.read_csv(RESULTS_DIR / "group_shift.csv")
neighbors = pd.read_csv(RESULTS_DIR / "nearest_neighbors.csv")

# ---------- 1. Semantic shift line plot ----------
plot_df = cosine_shift[cosine_shift["word"].isin(TARGET_WORDS)].copy()

plt.figure(figsize=(12, 7))
sns.lineplot(
    data=plot_df,
    x="period",
    y="cosine_distance",
    hue="word",
    marker="o"
)
plt.title("Semantic Shift from Base Period")
plt.xlabel("Period")
plt.ylabel("Cosine distance")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "shift_scores.png", dpi=300)
plt.close()

# ---------- 2. Heatmap ----------
heatmap_df = (
    plot_df.pivot(index="word", columns="period", values="cosine_distance")
    .reindex(TARGET_WORDS)
)

plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="rocket_r", linewidths=0.5)
plt.title("Cosine Shift Heatmap")
plt.xlabel("Period")
plt.ylabel("Word")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "shift_heatmap.png", dpi=300)
plt.close()

# ---------- 3. Target vs Control ----------
plt.figure(figsize=(8, 6))
sns.lineplot(
    data=group_shift,
    x="period",
    y="cosine_distance",
    hue="group",
    marker="o"
)
plt.title("Target vs. Control Words")
plt.xlabel("Period")
plt.ylabel("Average cosine distance")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "target_vs_control.png", dpi=300)
plt.close()

# ---------- 4. Top shifted target words in 2020s ----------
top_df = cosine_shift[
    (cosine_shift["word"].isin(TARGET_WORDS)) &
    (cosine_shift["period"] == "2020s")
].dropna(subset=["cosine_distance"]).sort_values("cosine_distance", ascending=False)

plt.figure(figsize=(9, 6))
sns.barplot(data=top_df, x="cosine_distance", y="word")
plt.title("Top Shifted Target Words in the 2020s")
plt.xlabel("Cosine distance")
plt.ylabel("Word")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "top_shifted_targets.png", dpi=300)
plt.close()

# ---------- 5. Core words only ----------
core_df = cosine_shift[cosine_shift["word"].isin(CORE_WORDS)].copy()

plt.figure(figsize=(10, 6))
sns.barplot(
    data=core_df[core_df["period"] == "2020s"].sort_values("cosine_distance", ascending=False),
    x="word",
    y="cosine_distance"
)
plt.title("Core Semantic Shift Scores (2020s)")
plt.xlabel("Word")
plt.ylabel("Cosine distance")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "core_shift_scores.png", dpi=300)
plt.close()

# ---------- 6. Nearest neighbors table as image ----------
table_df = (
    neighbors[neighbors["word"].isin(CORE_WORDS)]
    .pivot(index="word", columns="period", values="neighbors")
    .reindex(CORE_WORDS)
    .fillna("-")
)

for col in table_df.columns:
    table_df[col] = table_df[col].apply(lambda x: "\n".join(textwrap.wrap(str(x), width=45)))

fig, ax = plt.subplots(figsize=(14, 7))
ax.axis("off")
tbl = ax.table(
    cellText=table_df.values,
    rowLabels=table_df.index,
    colLabels=table_df.columns,
    loc="center",
    cellLoc="left"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 2.3)
plt.title("Nearest Neighbors Across Periods", pad=20)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "neighbors_table.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved figures to:", FIGURES_DIR)
for p in sorted(FIGURES_DIR.glob("*.png")):
    print("-", p.name)