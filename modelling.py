"""
NarrativeGuard — Model Training
================================
Trains three classical ML classifiers on TF-IDF features and
persists all metrics to ``results/model_results.json``.

Supports coursework section III.1 (Modelling).

Models
------
1. Logistic Regression (``class_weight='balanced'``)
2. Linear SVM (``kernel='linear', class_weight='balanced'``)
3. Multinomial Naive Bayes (``alpha=1.0``)

Design rationale
----------------
- ``class_weight='balanced'`` compensates for any imbalance between
  Fake and Real classes (Karim, Asad & Azam, 2024/2026).
- Multinomial NB naturally suits TF-IDF count-like features.
- All random seeds fixed to ``RANDOM_SEED = 42`` for reproducibility.

Optional transformer baseline (not executed by default)
-------------------------------------------------------
A code path gated behind ``RUN_TRANSFORMER_BASELINE`` shows how
sentence-transformer embeddings + Logistic Regression or full
DistilBERT fine-tuning would be integrated.  This mirrors the
94% accuracy reported by Cotroneo, Natella & Orbinato (2026)
on FakeCTI with fine-tuned DistilBERT.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from src.config import (
    DATA_PROCESSED_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    RUN_TRANSFORMER_BASELINE,
)
from src.utils import save_table


def train_and_evaluate(
    X_train: spmatrix,
    y_train: np.ndarray,
    X_test: spmatrix,
    y_test: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Train three classifiers, evaluate on the test set, and persist results.

    Parameters
    ----------
    X_train, X_test : sparse matrices
        TF-IDF feature matrices.
    y_train, y_test : np.ndarray
        Binary labels (1 = Fake, 0 = Real).

    Returns
    -------
    dict[str, dict]
        Per-model dictionary containing accuracy, precision, recall,
        F1-score, training time, confusion matrix, and full
        classification report dict.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "Linear SVM": SVC(
            kernel="linear",
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "Multinomial Naive Bayes": MultinomialNB(alpha=1.0),
    }

    all_results: dict[str, dict[str, Any]] = {}

    for name, clf in models.items():
        print(f"\n  Training {name} …")
        t0 = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=1)
        rec = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        cm = confusion_matrix(y_test, y_pred).tolist()
        report = classification_report(
            y_test, y_pred, target_names=["Real", "Fake"], output_dict=True
        )

        print(f"    Accuracy : {acc:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall   : {rec:.4f}")
        print(f"    F1       : {f1:.4f}")
        print(f"    Time     : {train_time:.2f}s")
        print()
        print(
            classification_report(
                y_test, y_pred, target_names=["Real", "Fake"]
            )
        )

        all_results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "train_time_s": round(train_time, 3),
            "confusion_matrix": cm,
            "classification_report": report,
        }

    # Persist to JSON
    results_path = RESULTS_DIR / "model_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  ✓ Results saved to {results_path}")

    # ── Table 5: Model hyper-parameters ──────────────────────────────
    table5 = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "Linear SVM",
                "Multinomial Naive Bayes",
            ],
            "Vectoriser": [
                "TF-IDF (unigram+bigram, 8000 features)",
                "TF-IDF (unigram+bigram, 8000 features)",
                "TF-IDF (unigram+bigram, 8000 features)",
            ],
            "Key Hyperparameters": [
                "max_iter=1000, class_weight='balanced', solver='lbfgs'",
                "kernel='linear', class_weight='balanced', C=1.0",
                "alpha=1.0 (Laplace smoothing)",
            ],
        }
    )
    save_table(table5, "table5_model_hyperparams.csv")
    print("\nTable 5 — Model Hyperparameters:")
    print(table5.to_string(index=False))

    # ── Optional transformer baseline (NOT executed by default) ──────
    if RUN_TRANSFORMER_BASELINE:
        # -----------------------------------------------------------
        # TRANSFORMER BASELINE — sentence-transformers + LogReg
        # -----------------------------------------------------------
        # This code path demonstrates how a transformer-based approach
        # would be integrated for comparison.  It is gated behind
        # RUN_TRANSFORMER_BASELINE (default False) because:
        #   (a) it requires the sentence-transformers package
        #   (b) encoding 18k statements is CPU/GPU intensive
        #   (c) fine-tuning DistilBERT needs a GPU and significant time
        #
        # Cotroneo, Natella & Orbinato (2026) report 94% attribution
        # accuracy with fine-tuned DistilBERT on FakeCTI; Karim, Asad
        # & Azam (2024/2026) report 99.98% with BERT on ISOT.
        # -----------------------------------------------------------
        try:
            from sentence_transformers import SentenceTransformer

            print("\n  [TRANSFORMER] Encoding with MiniLM …")
            st_model = SentenceTransformer("all-MiniLM-L6-v2")

            train_df = pd.read_parquet(
                DATA_PROCESSED_DIR / "train_clean.parquet"
            )
            test_df = pd.read_parquet(
                DATA_PROCESSED_DIR / "test_clean.parquet"
            )

            X_train_emb = st_model.encode(
                train_df["clean_statement"].tolist(),
                show_progress_bar=True,
                batch_size=64,
            )
            X_test_emb = st_model.encode(
                test_df["clean_statement"].tolist(),
                show_progress_bar=True,
                batch_size=64,
            )

            lr_emb = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_SEED,
            )
            lr_emb.fit(X_train_emb, train_df["binary_label"].values)
            y_pred_emb = lr_emb.predict(X_test_emb)

            acc_emb = accuracy_score(
                test_df["binary_label"].values, y_pred_emb
            )
            print(f"  [TRANSFORMER] MiniLM + LogReg accuracy: {acc_emb:.4f}")

        except ImportError:
            print(
                "  [TRANSFORMER] sentence-transformers not installed — skipping."
            )
    else:
        print(
            "\n  ℹ Transformer baseline skipped (set RUN_TRANSFORMER_BASELINE=True to enable)."
        )

    return all_results


if __name__ == "__main__":
    import joblib

    from src.features import build_tfidf

    X_tr, X_v, X_te, _ = build_tfidf()
    splits = {
        name: pd.read_parquet(DATA_PROCESSED_DIR / f"{name}_clean.parquet")
        for name in ["train", "test"]
    }
    train_and_evaluate(
        X_tr,
        splits["train"]["binary_label"].values,
        X_te,
        splits["test"]["binary_label"].values,
    )
