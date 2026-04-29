"""Training and persistence helpers for period-specific Word2Vec models."""

from __future__ import annotations

from pathlib import Path

from gensim.models import Word2Vec


def train_word2vec(
    sentences: list[list[str]],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 20,
    sg: int = 1,
    workers: int = 4,
    epochs: int = 10,
    seed: int = 42,
) -> Word2Vec:
    """Train one Word2Vec model for a period."""
    if not sentences:
        raise ValueError("Cannot train Word2Vec on an empty sentence list.")

    return Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        workers=workers,
        epochs=epochs,
        seed=seed,
    )


def train_models_by_period(
    period_sentences: dict[str, list[list[str]]],
    **kwargs,
) -> dict[str, Word2Vec]:
    """Train a Word2Vec model for each period."""
    return {
        period: train_word2vec(sentences, **kwargs)
        for period, sentences in period_sentences.items()
    }


def save_models(models: dict[str, Word2Vec], output_dir: str | Path) -> None:
    """Save models as gensim `.model` files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for period, model in models.items():
        model.save(str(output_path / f"word2vec_{period}.model"))


def load_models(model_dir: str | Path) -> dict[str, Word2Vec]:
    """Load all period Word2Vec models from a directory."""
    model_path = Path(model_dir)
    models: dict[str, Word2Vec] = {}
    for path in sorted(model_path.glob("word2vec_*.model")):
        period = path.stem.replace("word2vec_", "", 1)
        models[period] = Word2Vec.load(str(path))
    if not models:
        raise FileNotFoundError(f"No Word2Vec models found in {model_path}")
    return models
