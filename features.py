"""
NarrativeGuard — Feature Engineering
======================================
Fits a TF-IDF vectoriser on the training corpus and transforms all
splits into sparse feature matrices.

Supports coursework section III.1 (Feature Extraction).

Outputs
-------
- **Table 4** — ``tables/table4_feature_config.csv``
- Fitted vectoriser persisted to ``data/processed/tfidf_vectorizer.joblib``

Design rationale
----------------
We use unigrams + bigrams (``ngram_range=(1,2)``) with ``max_features=8000``
and ``min_df=3`` to balance vocabulary coverage against overfitting on rare
tokens.  This follows the BoW/TF-IDF baseline approach validated by
Karim, Asad & Azam (2024/2026) on the ISOT dataset.
"""

from __future__ import annotations

import joblib
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    DATA_PROCESSED_DIR,
    MAX_FEATURES,
    MIN_DF,
    NGRAM_RANGE,
)
from src.utils import save_table


def build_tfidf(
    splits: dict[str, pd.DataFrame] | None = None,
) -> tuple[spmatrix, spmatrix, spmatrix, TfidfVectorizer]:
    """Fit TF-IDF on the training split and transform all splits.

    Parameters
    ----------
    splits : dict, optional
        Cleaned DataFrames.  If *None*, loads from processed Parquet.

    Returns
    -------
    X_train, X_val, X_test : sparse matrices
        TF-IDF feature matrices.
    vectorizer : TfidfVectorizer
        The fitted vectoriser instance.
    """
    if splits is None:
        splits = {
            name: pd.read_parquet(DATA_PROCESSED_DIR / f"{name}_clean.parquet")
            for name in ["train", "validation", "test"]
        }

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
    )

    X_train = vectorizer.fit_transform(splits["train"]["clean_statement"])
    X_val = vectorizer.transform(splits["validation"]["clean_statement"])
    X_test = vectorizer.transform(splits["test"]["clean_statement"])

    # Persist the fitted vectoriser
    vec_path = DATA_PROCESSED_DIR / "tfidf_vectorizer.joblib"
    joblib.dump(vectorizer, vec_path)
    print(f"  ✓ Vectoriser saved to {vec_path}")

    # Print shapes
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_val   shape: {X_val.shape}")
    print(f"  X_test  shape: {X_test.shape}")
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")

    # ── Table 4: Feature configuration ───────────────────────────────
    table4 = pd.DataFrame(
        {
            "Parameter": [
                "Vectoriser",
                "n-gram range",
                "max_features",
                "min_df",
                "Vocabulary size",
            ],
            "Value": [
                "TfidfVectorizer (scikit-learn)",
                str(NGRAM_RANGE),
                str(MAX_FEATURES),
                str(MIN_DF),
                str(len(vectorizer.vocabulary_)),
            ],
            "Rationale": [
                "Standard sparse bag-of-words representation for text classification",
                "Captures single words and two-word phrases (e.g. 'health care')",
                "Limits dimensionality to the 8,000 most informative features",
                "Ignores terms appearing in fewer than 3 documents to reduce noise",
                "Actual vocabulary extracted from training corpus",
            ],
        }
    )
    save_table(table4, "table4_feature_config.csv")
    print("\nTable 4 — Feature Configuration:")
    print(table4.to_string(index=False))

    return X_train, X_val, X_test, vectorizer


if __name__ == "__main__":
    build_tfidf()
