#!/usr/bin/env python3
"""
NarrativeGuard — Main Pipeline
===============================
Runs the entire NarrativeGuard pipeline end-to-end:

1. Data loading (LIAR2 from HuggingFace)
2. Pre-processing (cleaning + binarisation)
3. Exploratory Data Analysis (Figures 4–5, Table 2)
4. Feature extraction (TF-IDF, Table 4)
5. Architecture diagrams (Figures 1, 2, 3, 6, 7)
6. Model training (LogReg, SVM, NB — Table 5, model_results.json)
7. Evaluation (Figures 8–10, Tables 6–7, reflection)

Usage
-----
    python main.py

All outputs are written to ``figures/``, ``tables/``, and ``results/``.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURES_DIR, RESULTS_DIR, TABLES_DIR
from src.utils import section_banner


def main() -> None:
    """Execute the full NarrativeGuard pipeline."""

    # ── 1. Data Loading ──────────────────────────────────────────────
    section_banner("Stage 1: Data Loading (LIAR2)")
    from src.data_loader import load_liar2
    raw_splits = load_liar2()

    # ── 2. Pre-processing ────────────────────────────────────────────
    section_banner("Stage 2: Pre-processing")
    from src.preprocessing import preprocess_splits
    clean_splits = preprocess_splits(raw_splits)

    # ── 3. Exploratory Data Analysis ─────────────────────────────────
    section_banner("Stage 3: Exploratory Data Analysis")
    from src.eda import run_eda
    run_eda(clean_splits)

    # ── 4. Feature Extraction ────────────────────────────────────────
    section_banner("Stage 4: Feature Extraction (TF-IDF)")
    from src.features import build_tfidf
    X_train, X_val, X_test, vectorizer = build_tfidf(clean_splits)

    # ── 5. Architecture Diagrams ─────────────────────────────────────
    section_banner("Stage 5: Architecture Diagrams")
    from src.architecture_diagram import generate_all_diagrams
    generate_all_diagrams()

    # ── 6. Model Training ────────────────────────────────────
    section_banner("Stage 6: Model Training")
    from src.modelling import train_and_evaluate
    y_train = clean_splits["train"]["binary_label"].values
    y_test = clean_splits["test"]["binary_label"].values
    results = train_and_evaluate(X_train, y_train, X_test, y_test)

    # ── 7. Evaluation ────────────────────────────────────────────────
    section_banner("Stage 7: Evaluation")
    from src.evaluation import run_evaluation
    run_evaluation(results)

    # ── Summary of generated files ───────────────────────────────────
    section_banner("Pipeline Complete — Generated Files")
    print("\nFigures:")
    for f in sorted(FIGURES_DIR.glob("*.png")):
        print(f"  {f}")
    print("\nTables:")
    for f in sorted(TABLES_DIR.glob("*.csv")):
        print(f"  {f}")
    print("\nResults:")
    for f in sorted(RESULTS_DIR.iterdir()):
        print(f"  {f}")
    print()
    print("All done! ✓")


if __name__ == "__main__":
    main()
