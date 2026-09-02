"""
NarrativeGuard — Pre-processing
================================
Text cleaning and label binarisation for the LIAR2 dataset.

Supports coursework section I.2 (Pre-processing).

Pipeline steps
--------------
1. Lower-casing
2. URL / noise removal
3. Whitespace normalisation
4. English stopword removal (NLTK)
5. Short-token filtering (tokens ≤ 2 characters)

Binarisation
------------
- **Fake** (``binary_label = 1``): labels {0, 1, 2} — pants-fire, false, barely-true
- **Real** (``binary_label = 0``): labels {3, 4, 5} — half-true, mostly-true, true
"""

from __future__ import annotations

import re
from typing import Optional

import nltk
import pandas as pd

from src.config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    FAKE_LABELS,
    LABEL_NAMES,
    REAL_LABELS,
)

# ── Ensure NLTK stopwords are available ──────────────────────────────
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords  # noqa: E402

_STOP_WORDS: set[str] = set(stopwords.words("english"))


def _clean_text(text: str) -> str:
    """Apply the full cleaning pipeline to a single statement."""
    # 1. Lower-case
    text = text.lower()
    # 2. Strip URLs
    text = re.sub(r"http\S+|www\S+", " ", text)
    # 3. Strip non-alphabetic characters
    text = re.sub(r"[^a-z\s]", " ", text)
    # 4. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # 5. Remove stopwords and tokens ≤ 2 characters
    tokens = [
        tok for tok in text.split()
        if tok not in _STOP_WORDS and len(tok) > 2
    ]
    return " ".join(tokens)


def preprocess_splits(
    splits: Optional[dict[str, pd.DataFrame]] = None,
) -> dict[str, pd.DataFrame]:
    """Clean and binarise every split; save to ``data/processed/``.

    Parameters
    ----------
    splits : dict[str, pd.DataFrame], optional
        Raw DataFrames keyed by split name.  If *None*, loads from
        cached Parquet files in ``data/raw/``.

    Returns
    -------
    dict[str, pd.DataFrame]
        Cleaned DataFrames keyed by split name.
    """
    if splits is None:
        splits = {
            name: pd.read_parquet(DATA_RAW_DIR / f"{name}.parquet")
            for name in ["train", "validation", "test"]
        }

    cleaned: dict[str, pd.DataFrame] = {}

    for split_name, df in splits.items():
        df = df.copy()

        # Label name mapping
        df["label_name"] = df["label"].map(LABEL_NAMES)

        # Binarisation
        df["binary_label"] = df["label"].apply(
            lambda x: 1 if x in FAKE_LABELS else 0
        )

        # Text cleaning
        df["statement"] = df["statement"].fillna("").astype(str)
        df["word_count_raw"] = df["statement"].str.split().str.len()
        df["clean_statement"] = df["statement"].apply(_clean_text)
        df["word_count_clean"] = df["clean_statement"].str.split().str.len()

        # Persist
        out_path = DATA_PROCESSED_DIR / f"{split_name}_clean.parquet"
        df.to_parquet(out_path, index=False)
        cleaned[split_name] = df

        # ── Console summary ──────────────────────────────────────────
        fake_count = (df["binary_label"] == 1).sum()
        real_count = (df["binary_label"] == 0).sum()
        print(f"\n--- {split_name} split ({len(df):,} rows) ---")
        print(f"  Fake: {fake_count:,}  ({fake_count / len(df) * 100:.1f}%)")
        print(f"  Real: {real_count:,}  ({real_count / len(df) * 100:.1f}%)")
        print(f"  6-way distribution:")
        for lbl in sorted(LABEL_NAMES.keys()):
            cnt = (df["label"] == lbl).sum()
            print(f"    {LABEL_NAMES[lbl]:>15s}: {cnt:>5,}")
        print(f"  Avg word count (raw):     {df['word_count_raw'].mean():.1f}")
        print(f"  Avg word count (cleaned): {df['word_count_clean'].mean():.1f}")

        # Two example pairs
        print("  Example raw → cleaned:")
        for i in range(min(2, len(df))):
            row = df.iloc[i]
            raw_stmt = row["statement"][:80]
            cln_stmt = row["clean_statement"][:80]
            print(f"    [{i}] RAW:   {raw_stmt}")
            print(f"        CLEAN: {cln_stmt}")

    return cleaned


if __name__ == "__main__":
    preprocess_splits()
