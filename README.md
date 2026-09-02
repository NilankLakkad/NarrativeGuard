# NarrativeGuard

**NarrativeGuard — Applying Text Mining & AI for Fake News Detection as Cyber Threat Intelligence**

A reproducible Python data-science pipeline that classifies political statements as **Fake** (disinformation-flagged) or **Real** (credible) using classical NLP and machine learning, framed as a Cyber Threat Intelligence (CTI) content-classification problem. The project draws on the concept-based CTI framework introduced by Cotroneo, Natella & Orbinato (2026), who demonstrate that persistent narrative indicators are more resilient threat artefacts than volatile domain or account-level signals. NarrativeGuard operationalises this insight by training TF-IDF-based classifiers (Logistic Regression, Linear SVM, Multinomial Naive Bayes) on the LIAR2 dataset of short political statements and benchmarking the results against transformer-based approaches from the literature.

## Dataset

**LIAR2** (Cheng et al.) — 22,962 political statements with 6-point veracity labels, loaded via the HuggingFace `datasets` library:

```python
from datasets import load_dataset
ds = load_dataset("chengxuphd/liar2")
```

Labels are binarised: `{pants-fire, false, barely-true}` → **Fake (1)**; `{half-true, mostly-true, true}` → **Real (0)**.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

The pipeline runs all stages end-to-end: data loading → pre-processing → EDA → TF-IDF → diagram generation → model training → evaluation. All outputs are written to `figures/`, `tables/`, and `results/`.

## Project Structure

```
NarrativeGuard/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                          # cached HF dataset parquet dumps
│   └── processed/                    # cleaned/binarised parquet files
├── figures/                          # ALL generated PNG figures land here
├── tables/                           # ALL generated tables exported as CSV
├── results/
│   └── model_results.json            # metrics for every model, machine-readable
├── src/
│   ├── __init__.py
│   ├── config.py                     # constants: seeds, paths, label maps, TF-IDF params
│   ├── data_loader.py                # load_dataset + cache to data/raw
│   ├── preprocessing.py              # text cleaning + binarisation -> data/processed
│   ├── eda.py                        # Figures 4, 5 + Table 2 dataset stats
│   ├── features.py                   # TF-IDF vectorisation, Table 4
│   ├── architecture_diagram.py       # Figures 1, 2, 3, 6, 7 (diagrams)
│   ├── modelling.py                  # trains LogReg, SVM, Naive Bayes -> results/
│   ├── evaluation.py                 # Figures 8, 9, 10 + Tables 5, 6, 7
│   └── utils.py                      # shared helpers
├── main.py                           # runs the entire pipeline end-to-end
└── notebooks/
    └── exploration.ipynb             # optional interactive walkthrough
```

## Figures & Tables Reference

| #        | Filename                                 | Description                                                   |
|----------|------------------------------------------|---------------------------------------------------------------|
| Fig. 1   | `fig1_pyramid_of_pain.png`               | Pyramid of Pain — Traditional CTI vs. Disinformation CTI     |
| Fig. 2   | `fig2_architecture.png`                  | NarrativeGuard system architecture (end-to-end pipeline)     |
| Fig. 3   | `fig3_preprocessing_workflow.png`        | Pre-processing workflow (5-step vertical flowchart)          |
| Fig. 4   | `fig4_class_distribution.png`            | Binary class distribution — Fake vs. Real (training set)     |
| Fig. 5   | `fig5_wordcloud.png`                     | Word cloud comparison — Fake vs. Real statements             |
| Fig. 6   | `fig6_tfidf_pipeline.png`                | TF-IDF vectorisation pipeline with example heatmap           |
| Fig. 7   | `fig7_model_comparison_framework.png`    | Model comparison framework (3 parallel classifiers)          |
| Fig. 8   | `fig8_confusion_matrices.png`            | Confusion matrices for all three classifiers                 |
| Fig. 9   | `fig9_classwise_precision_recall.png`    | Per-class precision and recall comparison                    |
| Fig. 10  | `fig10_model_comparison.png`             | Accuracy / Precision / Recall / F1 grouped bar chart         |
| Table 2  | `table2_dataset_stats.csv`               | Dataset statistics (split sizes, class balance, word counts) |
| Table 4  | `table4_feature_config.csv`              | TF-IDF vectoriser configuration and rationale                |
| Table 5  | `table5_model_hyperparams.csv`           | Model hyperparameters summary                                |
| Table 6  | `table6_evaluation_metrics.csv`          | Evaluation metrics (Acc / Prec / Rec / F1 / Time)            |
| Table 7  | `table7_literature_benchmark.csv`        | Literature benchmark comparison                              |

## References

1. Cotroneo, D., Natella, R. and Orbinato, V. (2026). *Elevating Cyber Threat Intelligence against Disinformation Campaigns with LLM-based Concept Extraction and the FakeCTI Dataset.* Journal of Systems and Software, 232.
2. Cinus, F., Minici, M., Luceri, L. and Ferrara, E. (2025). *Exposing Cross-Platform Coordinated Inauthentic Activity in the Run-Up to the 2024 U.S. Election.* ACM Web Conference (WWW '25).
3. Karim, A. A. J., Asad, K. H. M. and Azam, A. (2024, updated 2026). *Strengthening False Information Propagation Detection: Leveraging SVM and Sophisticated Text Vectorization Techniques in Comparison to BERT.* arXiv:2411.12703.
