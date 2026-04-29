"""Embedding-space alignment with Orthogonal Procrustes."""

from __future__ import annotations

import copy

import numpy as np
from gensim.models import Word2Vec
from scipy.linalg import orthogonal_procrustes


def shared_vocabulary(
    base_model: Word2Vec,
    other_model: Word2Vec,
    max_words: int = 10000,
    min_count: int = 20,
    exclude_words: set[str] | None = None,
) -> list[str]:
    """Return frequent words shared by two models, ordered by base-model frequency."""
    exclude_words = exclude_words or set()
    candidates = []
    for word in base_model.wv.index_to_key:
        if word in exclude_words:
            continue
        if word not in other_model.wv:
            continue
        base_count = base_model.wv.get_vecattr(word, "count")
        other_count = other_model.wv.get_vecattr(word, "count")
        if base_count >= min_count and other_count >= min_count:
            candidates.append(word)
        if len(candidates) >= max_words:
            break
    return candidates


def procrustes_rotation(
    base_model: Word2Vec,
    other_model: Word2Vec,
    anchor_words: list[str],
) -> np.ndarray:
    """Compute an orthogonal rotation from other_model space into base_model space."""
    if len(anchor_words) < 2:
        raise ValueError("At least two anchor words are required for Procrustes alignment.")

    base_matrix = np.vstack([base_model.wv[word] for word in anchor_words])
    other_matrix = np.vstack([other_model.wv[word] for word in anchor_words])
    rotation, _ = orthogonal_procrustes(other_matrix, base_matrix)
    return rotation


def align_model_to_base(
    base_model: Word2Vec,
    other_model: Word2Vec,
    max_anchor_words: int = 10000,
    min_anchor_count: int = 20,
    exclude_anchor_words: set[str] | None = None,
) -> Word2Vec:
    """Return a copy of other_model rotated into base_model coordinates."""
    aligned = copy.deepcopy(other_model)
    anchors = shared_vocabulary(
        base_model,
        aligned,
        max_words=max_anchor_words,
        min_count=min_anchor_count,
        exclude_words=exclude_anchor_words,
    )
    rotation = procrustes_rotation(base_model, aligned, anchors)
    aligned.wv.vectors = aligned.wv.vectors @ rotation
    aligned.wv.fill_norms(force=True)
    return aligned


def align_models_to_base(
    models: dict[str, Word2Vec],
    base_period: str | None = None,
    **kwargs,
) -> dict[str, Word2Vec]:
    """Align all models to a base period. The base defaults to the first sorted period."""
    if not models:
        raise ValueError("No models were provided.")

    base_period = base_period or sorted(models)[0]
    if base_period not in models:
        raise KeyError(f"Base period {base_period!r} not found.")

    base_model = models[base_period]
    aligned = {base_period: base_model}
    for period, model in models.items():
        if period == base_period:
            continue
        aligned[period] = align_model_to_base(base_model, model, **kwargs)
    return dict(sorted(aligned.items()))
