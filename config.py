"""
NarrativeGuard — Configuration Module
======================================
Centralised constants used across the entire pipeline.
Supports coursework sections I.1 (Literature Review context) and I.2 (Pre-processing parameters).

References
----------
- Cotroneo, D., Natella, R. and Orbinato, V. (2026). Elevating Cyber Threat Intelligence
  against Disinformation Campaigns with LLM-based Concept Extraction and the FakeCTI
  Dataset. Journal of Systems and Software, 232.
"""

from pathlib import Path

# ── Reproducibility ──────────────────────────────────────────────────
RANDOM_SEED: int = 42

# ── Project root (two levels up from this file) ─────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ── Directory paths ──────────────────────────────────────────────────
DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR: Path = PROJECT_ROOT / "figures"
TABLES_DIR: Path = PROJECT_ROOT / "tables"
RESULTS_DIR: Path = PROJECT_ROOT / "results"

# Create directories if they don't exist
for _dir in [DATA_RAW_DIR, DATA_PROCESSED_DIR, FIGURES_DIR, TABLES_DIR, RESULTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── LIAR2 label mapping (verified) ──────────────────────────────────
# 0=pants-fire, 1=false, 2=barely-true, 3=half-true, 4=mostly-true, 5=true
LABEL_NAMES: dict[int, str] = {
    0: "pants-fire",
    1: "false",
    2: "barely-true",
    3: "half-true",
    4: "mostly-true",
    5: "true",
}

# ── Binarisation rule ────────────────────────────────────────────────
# FAKE (1): pants-fire, false, barely-true  →  labels {0, 1, 2}
# REAL (0): half-true, mostly-true, true    →  labels {3, 4, 5}
FAKE_LABELS: set[int] = {0, 1, 2}
REAL_LABELS: set[int] = {3, 4, 5}

BINARY_LABEL_NAMES: dict[int, str] = {0: "Real", 1: "Fake"}

# ── TF-IDF hyper-parameters ─────────────────────────────────────────
MAX_FEATURES: int = 8000
NGRAM_RANGE: tuple[int, int] = (1, 2)
MIN_DF: int = 3

# ── HuggingFace dataset identifier ──────────────────────────────────
HF_DATASET_NAME: str = "chengxuphd/liar2"

# ── Transformer baseline flag (not executed by default) ─────────────
RUN_TRANSFORMER_BASELINE: bool = False
