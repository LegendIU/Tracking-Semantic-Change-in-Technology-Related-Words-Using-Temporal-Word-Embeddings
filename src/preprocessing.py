"""Utilities for loading, cleaning, and time-slicing a timestamped corpus."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd


DEFAULT_PERIODS: dict[str, tuple[int, int]] = {
    "1990s": (1990, 1999),
    "2000s": (2000, 2009),
    "2010s": (2010, 2019),
    "2020s": (2020, 2025),
}

TOKEN_RE = re.compile(r"[a-z][a-z_+-]{1,}")


def load_corpus(path: str | Path, year_col: str = "year", text_col: str = "text") -> pd.DataFrame:
    """Load a CSV corpus with year and text columns."""
    df = pd.read_csv(path)
    missing = {year_col, text_col}.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)}")

    df = df[[year_col, text_col]].rename(columns={year_col: "year", text_col: "text"})
    df = df.dropna(subset=["year", "text"]).copy()
    df["year"] = df["year"].astype(int)
    df["text"] = df["text"].astype(str)
    return df


def assign_period(year: int, periods: dict[str, tuple[int, int]] | None = None) -> str | None:
    """Return the period label for a year, or None if the year is outside all periods."""
    periods = periods or DEFAULT_PERIODS
    for label, (start, end) in periods.items():
        if start <= year <= end:
            return label
    return None


def tokenize(text: str, stopwords: set[str] | None = None) -> list[str]:
    """Tokenize text with a light cleanup suitable for Word2Vec training."""
    tokens = TOKEN_RE.findall(text.lower())
    if stopwords:
        tokens = [token for token in tokens if token not in stopwords]
    return tokens


def preprocess_dataframe(
    df: pd.DataFrame,
    periods: dict[str, tuple[int, int]] | None = None,
    min_tokens: int = 5,
    stopwords: set[str] | None = None,
) -> pd.DataFrame:
    """Add period and token columns, dropping rows outside the selected periods."""
    processed = df.copy()
    processed["period"] = processed["year"].apply(lambda year: assign_period(int(year), periods))
    processed = processed.dropna(subset=["period"])
    processed["tokens"] = processed["text"].apply(lambda text: tokenize(text, stopwords=stopwords))
    processed = processed[processed["tokens"].map(len) >= min_tokens].reset_index(drop=True)
    return processed


def sentences_by_period(df: pd.DataFrame) -> dict[str, list[list[str]]]:
    """Group tokenized documents by period."""
    if "period" not in df.columns or "tokens" not in df.columns:
        raise ValueError("DataFrame must contain 'period' and 'tokens' columns.")
    grouped: dict[str, list[list[str]]] = {}
    for period, group in df.groupby("period", sort=True):
        grouped[str(period)] = group["tokens"].tolist()
    return grouped


def flatten_sentences(sentences: Iterable[Iterable[str]]) -> list[str]:
    """Flatten a list of tokenized documents into a token stream."""
    return [token for sentence in sentences for token in sentence]
