"""Metrics for temporal semantic-shift analysis."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from gensim.models import Word2Vec


def cosine_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine distance between two vectors."""
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return math.nan
    return float(1 - np.dot(vec_a, vec_b) / denom)


def jaccard_distance(items_a: list[str], items_b: list[str]) -> float:
    """Compute Jaccard distance between two neighbor lists."""
    set_a = set(items_a)
    set_b = set(items_b)
    union = set_a | set_b
    if not union:
        return math.nan
    return 1 - (len(set_a & set_b) / len(union))


def word_in_all_models(word: str, models: dict[str, Word2Vec]) -> bool:
    """Check whether a word exists in every model vocabulary."""
    return all(word in model.wv for model in models.values())


def nearest_neighbors(
    models: dict[str, Word2Vec],
    words: list[str],
    topn: int = 10,
) -> pd.DataFrame:
    """Build a table of nearest neighbors for target words in each period."""
    rows = []
    for word in words:
        for period, model in models.items():
            if word not in model.wv:
                rows.append({"word": word, "period": period, "neighbors": ""})
                continue
            neighbors = [neighbor for neighbor, _ in model.wv.most_similar(word, topn=topn)]
            rows.append({"word": word, "period": period, "neighbors": ", ".join(neighbors)})
    return pd.DataFrame(rows)


def cosine_shift_from_base(
    models: dict[str, Word2Vec],
    words: list[str],
    base_period: str | None = None,
) -> pd.DataFrame:
    """Compute cosine distance from the base period for each word and period."""
    if not models:
        raise ValueError("No models were provided.")
    base_period = base_period or sorted(models)[0]
    base_model = models[base_period]

    rows = []
    for word in words:
        for period, model in models.items():
            if word not in base_model.wv or word not in model.wv:
                distance = math.nan
            else:
                distance = cosine_distance(base_model.wv[word], model.wv[word])
            rows.append({"word": word, "period": period, "cosine_distance": distance})
    return pd.DataFrame(rows)


def neighbor_shift_from_base(
    models: dict[str, Word2Vec],
    words: list[str],
    base_period: str | None = None,
    topn: int = 10,
) -> pd.DataFrame:
    """Compute nearest-neighbor Jaccard distance from the base period."""
    if not models:
        raise ValueError("No models were provided.")
    base_period = base_period or sorted(models)[0]
    base_model = models[base_period]

    rows = []
    for word in words:
        if word not in base_model.wv:
            base_neighbors = []
        else:
            base_neighbors = [neighbor for neighbor, _ in base_model.wv.most_similar(word, topn=topn)]

        for period, model in models.items():
            if word not in model.wv:
                distance = math.nan
            else:
                period_neighbors = [neighbor for neighbor, _ in model.wv.most_similar(word, topn=topn)]
                distance = jaccard_distance(base_neighbors, period_neighbors)
            rows.append({"word": word, "period": period, "jaccard_distance": distance})
    return pd.DataFrame(rows)


def summarize_group_shift(
    shift_df: pd.DataFrame,
    target_words: list[str],
    control_words: list[str],
) -> pd.DataFrame:
    """Compare average shift for changing target words and stable control words."""
    labeled = shift_df.copy()
    labeled["group"] = np.where(
        labeled["word"].isin(target_words),
        "target",
        np.where(labeled["word"].isin(control_words), "control", "other"),
    )
    return (
        labeled[labeled["group"].isin(["target", "control"])]
        .groupby(["group", "period"], as_index=False)["cosine_distance"]
        .mean()
    )
