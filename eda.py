"""
NarrativeGuard — Exploratory Data Analysis
============================================
Produces dataset-level statistics (Table 2) and visual summaries
(Figures 4–5) for the coursework.

Supports coursework section II (Data Analysis).

Outputs
-------
- **Table 2**  — ``tables/table2_dataset_stats.csv``
- **Figure 4** — ``figures/fig4_class_distribution.png``
- **Figure 5** — ``figures/fig5_wordcloud.png``
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

from src.config import DATA_PROCESSED_DIR
from src.utils import save_figure, save_table


def run_eda(
    splits: dict[str, pd.DataFrame] | None = None,
) -> None:
    """Generate all EDA artefacts (Table 2, Figures 4–5)."""
    if splits is None:
        splits = {
            name: pd.read_parquet(DATA_PROCESSED_DIR / f"{name}_clean.parquet")
            for name in ["train", "validation", "test"]
        }

    train = splits["train"]
    val = splits["validation"]
    test = splits["test"]

    # ── Table 2: Dataset statistics ──────────────────────────────────
    total = len(train) + len(val) + len(test)
    fake_train = int((train["binary_label"] == 1).sum())
    real_train = int((train["binary_label"] == 0).sum())

    stats = pd.DataFrame(
        {
            "Metric": [
                "Total statements",
                "Training set size",
                "Validation set size",
                "Test set size",
                "Fake count (train)",
                "Real count (train)",
                "Avg words (raw)",
                "Avg words (cleaned)",
            ],
            "Value": [
                f"{total:,}",
                f"{len(train):,}",
                f"{len(val):,}",
                f"{len(test):,}",
                f"{fake_train:,} ({fake_train / len(train) * 100:.1f}%)",
                f"{real_train:,} ({real_train / len(train) * 100:.1f}%)",
                f"{train['word_count_raw'].mean():.1f}",
                f"{train['word_count_clean'].mean():.1f}",
            ],
        }
    )
    save_table(stats, "table2_dataset_stats.csv")
    print("\nTable 2 — Dataset Statistics:")
    print(stats.to_string(index=False))

    # ── Figure 4: Binary class distribution (training set) ───────────
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    counts = train["binary_label"].value_counts().sort_index()
    labels = ["Real (0)", "Fake (1)"]
    colors = ["#4285F4", "#EA4335"]  # blue for Real, red for Fake
    bars = ax4.bar(labels, [counts.get(0, 0), counts.get(1, 0)], color=colors, edgecolor="black", linewidth=0.5)
    for bar in bars:
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            height + 50,
            f"{int(height):,}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )
    ax4.set_ylabel("Number of Statements")
    ax4.set_title("Figure 4: Class Distribution — Fake vs. Real (Training Set)")
    ax4.set_ylim(0, max(counts) * 1.15)
    fig4.tight_layout()
    save_figure(fig4, "fig4_class_distribution.png")

    # ── Figure 5: Word clouds (Fake vs. Real) ───────────────────────
    fake_text = " ".join(train.loc[train["binary_label"] == 1, "clean_statement"].tolist())
    real_text = " ".join(train.loc[train["binary_label"] == 0, "clean_statement"].tolist())

    wc_fake = WordCloud(
        width=800, height=400, background_color="white",
        colormap="Reds", max_words=150, random_state=42,
    ).generate(fake_text if fake_text.strip() else "empty")

    wc_real = WordCloud(
        width=800, height=400, background_color="white",
        colormap="Blues", max_words=150, random_state=42,
    ).generate(real_text if real_text.strip() else "empty")

    fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(14, 5))
    ax5a.imshow(wc_fake, interpolation="bilinear")
    ax5a.set_title("Fake Statements", fontsize=13)
    ax5a.axis("off")
    ax5b.imshow(wc_real, interpolation="bilinear")
    ax5b.set_title("Real Statements", fontsize=13)
    ax5b.axis("off")
    fig5.suptitle(
        "Figure 5: Word Cloud Comparison — Fake vs. Real Statements",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig5.tight_layout()
    save_figure(fig5, "fig5_wordcloud.png")


if __name__ == "__main__":
    run_eda()
