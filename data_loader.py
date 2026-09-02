"""
NarrativeGuard — Data Loader
=============================
Downloads the LIAR2 dataset from HuggingFace and caches each split
as a Parquet file under ``data/raw/``.

Supports coursework section I.2 (Data Acquisition).

Dataset
-------
LIAR2 — 22,962 political statements labelled on a 6-point veracity
scale.  Loaded via ``datasets.load_dataset('chengxuphd/liar2')``.
"""

from __future__ import annotations

import pandas as pd
from datasets import load_dataset

from src.config import DATA_RAW_DIR, HF_DATASET_NAME


def load_liar2() -> dict[str, pd.DataFrame]:
    """Download LIAR2 and persist each split as Parquet.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping ``{split_name: DataFrame}`` for *train*, *validation*,
        and *test* splits.
    """
    print(f"Loading dataset '{HF_DATASET_NAME}' from HuggingFace …")
    ds = load_dataset(HF_DATASET_NAME)

    splits: dict[str, pd.DataFrame] = {}
    for split_name in ["train", "validation", "test"]:
        df = ds[split_name].to_pandas()
        out_path = DATA_RAW_DIR / f"{split_name}.parquet"
        df.to_parquet(out_path, index=False)
        splits[split_name] = df
        print(f"  {split_name:>12s}: {len(df):,} rows  →  {out_path}")

    print(f"\nColumns: {list(splits['train'].columns)}")
    return splits


if __name__ == "__main__":
    load_liar2()
