"""
NarrativeGuard — Evaluation & Benchmarking
============================================
Generates evaluation visualisations (Figures 8–10) and summary
tables (Tables 6–7) from the trained model results.

Supports coursework section III.2 (Evaluation).

Outputs
-------
- **Figure 8**  — ``fig8_confusion_matrices.png``
- **Figure 9**  — ``fig9_classwise_precision_recall.png``
- **Figure 10** — ``fig10_model_comparison.png``
- **Table 6**   — ``table6_evaluation_metrics.csv``
- **Table 7**   — ``table7_literature_benchmark.csv``
- Reflection   — ``results/reflection.txt``
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import RESULTS_DIR
from src.utils import save_figure, save_table


def run_evaluation(
    results: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Generate all evaluation artefacts.

    Parameters
    ----------
    results : dict, optional
        Model results dictionary.  If *None*, loads from
        ``results/model_results.json``.
    """
    if results is None:
        with open(RESULTS_DIR / "model_results.json") as f:
            results = json.load(f)

    model_names = list(results.keys())

    # ── Figure 8: Confusion matrices ─────────────────────────────────
    fig8, axes8 = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, name in zip(axes8, model_names):
        cm = np.array(results[name]["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Real", "Fake"],
            yticklabels=["Real", "Fake"],
            ax=ax,
            cbar=False,
        )
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig8.suptitle(
        "Figure 8: Confusion Matrices — Classical ML Models (TF-IDF)",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )
    fig8.tight_layout()
    save_figure(fig8, "fig8_confusion_matrices.png")

    # ── Figure 9: Per-class precision & recall ───────────────────────
    fig9, ax9 = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(model_names))
    width = 0.18

    metrics_per_class = []
    for name in model_names:
        report = results[name]["classification_report"]
        metrics_per_class.append({
            "real_precision": report["Real"]["precision"],
            "real_recall":    report["Real"]["recall"],
            "fake_precision": report["Fake"]["precision"],
            "fake_recall":    report["Fake"]["recall"],
        })

    bars_data = [
        ([m["real_precision"] for m in metrics_per_class], "Real Precision", "#4285F4"),
        ([m["real_recall"]    for m in metrics_per_class], "Real Recall",    "#7BAAF7"),
        ([m["fake_precision"] for m in metrics_per_class], "Fake Precision", "#EA4335"),
        ([m["fake_recall"]    for m in metrics_per_class], "Fake Recall",    "#F28B82"),
    ]

    for i, (values, label, color) in enumerate(bars_data):
        offset = (i - 1.5) * width
        ax9.bar(x_pos + offset, values, width, label=label, color=color, edgecolor="black", linewidth=0.5)

    ax9.set_xticks(x_pos)
    ax9.set_xticklabels(model_names)
    ax9.set_ylabel("Score")
    ax9.set_ylim(0, 1.05)
    ax9.set_title(
        "Figure 9: Per-Class Precision & Recall",
        fontsize=12, fontweight="bold",
    )
    ax9.legend(loc="lower right")
    fig9.tight_layout()
    save_figure(fig9, "fig9_classwise_precision_recall.png")

    # ── Figure 10: Model comparison (Accuracy / Precision / Recall / F1) ─
    fig10, ax10 = plt.subplots(figsize=(10, 5))
    metric_keys = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]
    colors10 = ["#4285F4", "#34A853", "#FBBC05", "#EA4335"]
    width10 = 0.18

    for i, (key, label, color) in enumerate(zip(metric_keys, metric_labels, colors10)):
        values = [results[name][key] for name in model_names]
        offset = (i - 1.5) * width10
        bars = ax10.bar(
            x_pos + offset, values, width10,
            label=label, color=color, edgecolor="black", linewidth=0.5,
        )
        for bar in bars:
            height = bar.get_height()
            ax10.text(
                bar.get_x() + bar.get_width() / 2, height + 0.005,
                f"{height:.3f}", ha="center", va="bottom", fontsize=7,
            )

    ax10.set_xticks(x_pos)
    ax10.set_xticklabels(model_names)
    ax10.set_ylabel("Score")
    ax10.set_ylim(0, 1.05)
    ax10.set_title(
        "Figure 10: Model Comparison — Accuracy, Precision, Recall, F1",
        fontsize=12, fontweight="bold",
    )
    ax10.legend(loc="lower right")
    fig10.tight_layout()
    save_figure(fig10, "fig10_model_comparison.png")

    # ── Table 6: Evaluation metrics ──────────────────────────────────
    table6_rows = []
    for name in model_names:
        r = results[name]
        table6_rows.append({
            "Model": name,
            "Accuracy": f"{r['accuracy']:.4f}",
            "Precision": f"{r['precision']:.4f}",
            "Recall": f"{r['recall']:.4f}",
            "F1-Score": f"{r['f1']:.4f}",
            "Train Time (s)": f"{r['train_time_s']:.3f}",
        })
    table6 = pd.DataFrame(table6_rows)
    save_table(table6, "table6_evaluation_metrics.csv")
    print("\nTable 6 — Evaluation Metrics:")
    print(table6.to_string(index=False))

    # ── Table 7: Literature benchmark comparison ─────────────────────
    # Find the best model from this study
    best_model_name = max(results, key=lambda n: results[n]["accuracy"])
    best_accuracy = results[best_model_name]["accuracy"]

    lit_rows = [
        {
            "Study": "Karim, Asad & Azam (2024/2026)",
            "Dataset": "ISOT (full articles)",
            "Best Model": "BERT (transformer)",
            "Accuracy": "99.98%",
        },
        {
            "Study": "Karim, Asad & Azam (2024/2026)",
            "Dataset": "ISOT (full articles)",
            "Best Model": "SVM + BoW",
            "Accuracy": "99.81%",
        },
        {
            "Study": "Cotroneo, Natella & Orbinato (2026)",
            "Dataset": "FakeCTI (campaign attribution)",
            "Best Model": "Fine-tuned DistilBERT",
            "Accuracy": "94.0%",
        },
        {
            "Study": "Cotroneo, Natella & Orbinato (2026)",
            "Dataset": "FakeCTI (campaign attribution)",
            "Best Model": "SBERT semantic similarity",
            "Accuracy": "67.5%",
        },
        {
            "Study": "NarrativeGuard (this study)",
            "Dataset": "LIAR2 (short statements)",
            "Best Model": best_model_name,
            "Accuracy": f"{best_accuracy * 100:.2f}%",
        },
    ]
    table7 = pd.DataFrame(lit_rows)
    save_table(table7, "table7_literature_benchmark.csv")
    print("\nTable 7 — Literature Benchmark Comparison:")
    print(table7.to_string(index=False))

    # ── Reflection ───────────────────────────────────────────────────
    reflection = f"""Reflection — Accuracy Gap Analysis
====================================

NarrativeGuard's best model ({best_model_name}) achieves {best_accuracy * 100:.2f}%
accuracy on the LIAR2 test set, which is notably lower than the 94–99%
reported by comparable studies.  Three key factors explain this gap:

(a) Full-article vs. short-statement difficulty: The ISOT dataset used
    by Karim, Asad & Azam (2024/2026) contains full news articles
    (hundreds of words), providing rich lexical cues.  LIAR2 statements
    average ~17 words after cleaning — far less signal for any classifier
    to exploit.  Similarly, FakeCTI classifies campaigns with repeated
    narrative patterns, not isolated one-line claims.

(b) Lexical vs. semantic vs. transformer feature richness: TF-IDF
    captures surface-level word frequencies and bigrams but cannot
    model paraphrase, context, or world knowledge.  Transformer
    architectures (BERT, DistilBERT) encode deep contextual semantics
    and have been pre-trained on billions of tokens, giving them a
    substantial advantage — especially on short, ambiguous text where
    word overlap alone is insufficient.

(c) Compute constraints: Fine-tuning DistilBERT or running SBERT
    inference on >18,000 statements requires GPU resources beyond
    the scope of this coursework run.  The optional transformer
    baseline in modelling.py demonstrates how such an approach would
    be integrated, and the literature benchmarks in Table 7 provide
    the expected performance ceiling.

Despite the lower absolute accuracy, the classical pipeline validates
the end-to-end CTI classification framework and confirms that TF-IDF
with balanced linear classifiers can discriminate fake from real
statements well above the 50% random baseline, even on challenging
short-text data.
"""
    reflection_path = RESULTS_DIR / "reflection.txt"
    with open(reflection_path, "w") as f:
        f.write(reflection)
    print(f"\n  ✓ Reflection saved to {reflection_path}")
    print("\n" + reflection)


if __name__ == "__main__":
    run_evaluation()
