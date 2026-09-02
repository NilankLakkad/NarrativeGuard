"""
NarrativeGuard — Architecture & Conceptual Diagrams
=====================================================
Generates five conceptual diagrams using matplotlib shapes, boxes,
and arrows.  No external image assets required.

Supports coursework sections I.1 (Literature Review) and I.2 (Methodology).

Outputs
-------
- **Figure 1** — ``fig1_pyramid_of_pain.png``
- **Figure 2** — ``fig2_architecture.png``
- **Figure 3** — ``fig3_preprocessing_workflow.png``
- **Figure 6** — ``fig6_tfidf_pipeline.png``
- **Figure 7** — ``fig7_model_comparison_framework.png``
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.utils import save_figure


# ═════════════════════════════════════════════════════════════════════
# Helper: draw a labelled rounded box at (x, y) with given w, h
# ═════════════════════════════════════════════════════════════════════
def _box(ax, x, y, w, h, text, color="#E8F0FE", fontsize=9, text_color="black"):
    """Draw a rounded rectangle with centred text."""
    fancy = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.15",
        facecolor=color, edgecolor="#333333", linewidth=1.2,
    )
    ax.add_patch(fancy)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=text_color, wrap=True)


def _arrow(ax, x1, y1, x2, y2, color="#333333"):
    """Draw an arrow from (x1,y1) to (x2,y2)."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
    )


# ═════════════════════════════════════════════════════════════════════
# Figure 1: Pyramid of Pain — Traditional CTI vs. Disinformation
# ═════════════════════════════════════════════════════════════════════
def generate_figure1() -> None:
    """Side-by-side Pyramids of Pain.

    Left: Traditional CTI Pyramid (Bianco, 2013).
    Right: Disinformation-oriented adaptation from Cotroneo,
    Natella & Orbinato (2026).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    def _draw_pyramid(ax, layers, title, colors):
        n = len(layers)
        for i, (label, col) in enumerate(zip(layers, colors)):
            # Bottom (i=0) is widest
            level = i
            width = 1.0 - level * (0.8 / n)
            left = 0.5 - width / 2
            bottom = level * (1.0 / n)
            height = 0.9 / n
            rect = mpatches.FancyBboxPatch(
                (left, bottom), width, height,
                boxstyle="round,pad=0.02",
                facecolor=col, edgecolor="#333", linewidth=1.0,
            )
            ax.add_patch(rect)
            ax.text(
                0.5, bottom + height / 2, label,
                ha="center", va="center", fontsize=8,
                fontweight="bold", color="black",
            )
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.05, 1.1)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
        ax.axis("off")

        # Arrow indicating increasing pain
        ax.annotate(
            "Harder to\nevade", xy=(0.95, 0.9), xytext=(0.95, 0.1),
            arrowprops=dict(arrowstyle="->", color="red", lw=2),
            fontsize=8, ha="center", va="bottom", color="red",
        )

    trad_layers = [
        "Hash Values", "IP Addresses", "Domain Names",
        "Network/Host Artifacts", "Tools", "TTPs",
    ]
    disinfo_layers = [
        "Social Media Handles", "Website Domains", "Content Artifacts",
        "Behavioural Patterns", "Recurrent Narratives", "TTPs",
    ]
    greens = ["#C8E6C9", "#A5D6A7", "#81C784", "#66BB6A", "#4CAF50", "#388E3C"]
    reds = ["#FFCDD2", "#EF9A9A", "#E57373", "#EF5350", "#F44336", "#C62828"]

    _draw_pyramid(ax1, trad_layers, "Traditional CTI\nPyramid of Pain", greens)
    _draw_pyramid(ax2, disinfo_layers, "Disinformation-Oriented\nPyramid of Pain", reds)

    fig.suptitle(
        "Figure 1: Pyramid of Pain — Traditional CTI vs. Disinformation CTI\n"
        "(Adapted from Cotroneo, Natella & Orbinato, 2026)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, "fig1_pyramid_of_pain.png")


# ═════════════════════════════════════════════════════════════════════
# Figure 2: System Architecture — end-to-end pipeline
# ═════════════════════════════════════════════════════════════════════
def generate_figure2() -> None:
    """Left-to-right pipeline diagram."""
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis("off")

    boxes = [
        (1.0, 1.0, "Raw Statements\n(LIAR2 Dataset)", "#BBDEFB"),
        (3.0, 1.0, "Pre-processing\n(Clean + Binarise)", "#C8E6C9"),
        (5.0, 1.0, "Feature Extraction\n(TF-IDF)", "#FFF9C4"),
        (7.0, 1.0, "Classification\n(LogReg / SVM / NB)", "#FFCCBC"),
        (9.0, 1.0, "Fake / Real\nCTI Label", "#E1BEE7"),
    ]
    for bx, by, label, col in boxes:
        _box(ax, bx, by, 1.6, 1.0, label, color=col, fontsize=9)

    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 0.85
        x2 = boxes[i + 1][0] - 0.85
        _arrow(ax, x1, 1.0, x2, 1.0)

    ax.set_title(
        "Figure 2: NarrativeGuard System Architecture",
        fontsize=12, fontweight="bold", pad=15,
    )
    save_figure(fig, "fig2_architecture.png")


# ═════════════════════════════════════════════════════════════════════
# Figure 3: Pre-processing Workflow (vertical flowchart)
# ═════════════════════════════════════════════════════════════════════
def generate_figure3() -> None:
    """Vertical flowchart of the five cleaning steps."""
    fig, ax = plt.subplots(figsize=(5, 8))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 10)
    ax.axis("off")

    steps = [
        ("1. Lower-casing", "str.lower()"),
        ("2. URL / Noise Removal", "regex: http\\S+ | [^a-z\\s]"),
        ("3. Whitespace Normalisation", "regex: collapse multiple spaces"),
        ("4. Stopword Removal", "NLTK English stopwords"),
        ("5. Short-token Filtering", "drop tokens with len ≤ 2"),
    ]
    colors = ["#BBDEFB", "#B2DFDB", "#FFF9C4", "#FFCCBC", "#E1BEE7"]

    y_positions = [8.5, 7.0, 5.5, 4.0, 2.5]

    for (title, technique), color, y in zip(steps, colors, y_positions):
        _box(ax, 2.0, y, 3.2, 1.0, f"{title}\n({technique})", color=color, fontsize=8)

    for i in range(len(y_positions) - 1):
        _arrow(ax, 2.0, y_positions[i] - 0.55, 2.0, y_positions[i + 1] + 0.55)

    # Input / output labels
    ax.text(2.0, 9.5, "Raw Statement", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#333")
    _arrow(ax, 2.0, 9.2, 2.0, 9.05)
    ax.text(2.0, 1.5, "Clean Statement", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#333")
    _arrow(ax, 2.0, 2.0, 2.0, 1.8)

    ax.set_title(
        "Figure 3: Pre-processing Workflow",
        fontsize=12, fontweight="bold", pad=15,
    )
    save_figure(fig, "fig3_preprocessing_workflow.png")


# ═════════════════════════════════════════════════════════════════════
# Figure 6: TF-IDF Pipeline with mini heatmap inset
# ═════════════════════════════════════════════════════════════════════
def generate_figure6() -> None:
    """Diagram: raw text → tokenised corpus → TF-IDF matrix → classifier."""
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")

    _box(ax, 1.5, 2.0, 2.2, 1.4, "Raw Text\nCorpus", "#BBDEFB", fontsize=9)
    _box(ax, 4.5, 2.0, 2.2, 1.4, "Tokenised\nCorpus", "#C8E6C9", fontsize=9)
    _box(ax, 12.0, 2.0, 2.2, 1.4, "Classifier\n(LogReg/SVM/NB)", "#FFCCBC", fontsize=9)

    _arrow(ax, 2.7, 2.0, 3.3, 2.0)
    _arrow(ax, 5.7, 2.0, 6.5, 2.0)
    _arrow(ax, 10.2, 2.0, 10.8, 2.0)

    # Mini TF-IDF heatmap inset
    ax_inset = fig.add_axes([0.48, 0.2, 0.2, 0.6])  # [left, bottom, width, height]
    np.random.seed(42)
    example_matrix = np.array([
        [0.42, 0.00, 0.31, 0.15],
        [0.00, 0.55, 0.00, 0.38],
        [0.28, 0.12, 0.00, 0.47],
        [0.00, 0.00, 0.63, 0.00],
    ])
    im = ax_inset.imshow(example_matrix, cmap="YlOrRd", aspect="auto")
    ax_inset.set_xticks(range(4))
    ax_inset.set_xticklabels(["tax", "job", "war", "vote"], fontsize=7)
    ax_inset.set_yticks(range(4))
    ax_inset.set_yticklabels(["Doc1", "Doc2", "Doc3", "Doc4"], fontsize=7)
    ax_inset.set_title("TF-IDF Matrix", fontsize=9, fontweight="bold")

    # Add values to cells
    for i in range(4):
        for j in range(4):
            ax_inset.text(j, i, f"{example_matrix[i, j]:.2f}",
                          ha="center", va="center", fontsize=7,
                          color="white" if example_matrix[i, j] > 0.35 else "black")

    ax.set_title(
        "Figure 6: TF-IDF Vectorisation Pipeline",
        fontsize=12, fontweight="bold", pad=15,
    )
    save_figure(fig, "fig6_tfidf_pipeline.png")


# ═════════════════════════════════════════════════════════════════════
# Figure 7: Model Comparison Framework
# ═════════════════════════════════════════════════════════════════════
def generate_figure7() -> None:
    """TF-IDF features → 3 parallel models → Evaluation & Comparison."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Input box
    _box(ax, 1.5, 2.5, 2.2, 1.2, "TF-IDF\nFeature Matrix", "#FFF9C4", fontsize=9)

    # Three model boxes
    models = [
        (5.0, 4.0, "Logistic\nRegression", "#BBDEFB"),
        (5.0, 2.5, "Linear SVM", "#C8E6C9"),
        (5.0, 1.0, "Multinomial\nNaive Bayes", "#FFCCBC"),
    ]
    for mx, my, label, col in models:
        _box(ax, mx, my, 2.0, 0.9, label, color=col, fontsize=9)
        _arrow(ax, 2.7, 2.5, mx - 1.05, my)

    # Output box
    _box(ax, 8.5, 2.5, 2.2, 1.2, "Evaluation &\nComparison", "#E1BEE7", fontsize=9)
    for _, my, _, _ in models:
        _arrow(ax, 6.05, my, 7.35, 2.5)

    ax.set_title(
        "Figure 7: Model Comparison Framework",
        fontsize=12, fontweight="bold", pad=15,
    )
    save_figure(fig, "fig7_model_comparison_framework.png")


# ═════════════════════════════════════════════════════════════════════
# Public entry point
# ═════════════════════════════════════════════════════════════════════
def generate_all_diagrams() -> None:
    """Generate all five conceptual diagrams (Figures 1, 2, 3, 6, 7)."""
    generate_figure1()
    generate_figure2()
    generate_figure3()
    generate_figure6()
    generate_figure7()


if __name__ == "__main__":
    generate_all_diagrams()
