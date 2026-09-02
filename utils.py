"""
NarrativeGuard — Shared Utilities
==================================
Helper functions for consistent figure/table saving and other
repeated operations across the pipeline.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES_DIR, TABLES_DIR


def save_figure(
    fig: plt.Figure,
    filename: str,
    dpi: int = 150,
    directory: Optional[Path] = None,
) -> Path:
    """Save a matplotlib figure to the figures directory.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object to save.
    filename : str
        Filename (e.g. ``fig4_class_distribution.png``).
    dpi : int, optional
        Resolution in dots per inch (default 150).
    directory : Path, optional
        Override the default ``FIGURES_DIR``.

    Returns
    -------
    Path
        Absolute path to the saved file.
    """
    out_dir = directory or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ Saved figure: {filepath}")
    return filepath


def save_table(df: pd.DataFrame, filename: str, directory: Optional[Path] = None) -> Path:
    """Save a pandas DataFrame to CSV in the tables directory.

    Parameters
    ----------
    df : pd.DataFrame
        The table to export.
    filename : str
        Filename (e.g. ``table2_dataset_stats.csv``).
    directory : Path, optional
        Override the default ``TABLES_DIR``.

    Returns
    -------
    Path
        Absolute path to the saved file.
    """
    out_dir = directory or TABLES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / filename
    df.to_csv(filepath, index=False)
    print(f"  ✓ Saved table:  {filepath}")
    return filepath


def section_banner(title: str) -> None:
    """Print a clearly visible section banner to stdout."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
