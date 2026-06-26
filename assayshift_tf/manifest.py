from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_MANIFEST_COLUMNS = [
    "dataset_id",
    "source",
    "tf",
    "species",
    "assembly",
    "assay",
    "lab",
    "biosample",
    "cell_type",
    "processed_peak_url",
    "control_url",
    "notes",
    "citation_url",
]


def load_manifest(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def validate_manifest(frame: pd.DataFrame) -> list[str]:
    messages: list[str] = []
    missing = [col for col in REQUIRED_MANIFEST_COLUMNS if col not in frame.columns]
    if missing:
        messages.append(f"missing columns: {', '.join(missing)}")
    if "dataset_id" in frame and frame["dataset_id"].duplicated().any():
        dupes = frame.loc[frame["dataset_id"].duplicated(), "dataset_id"].tolist()
        messages.append(f"duplicate dataset_id values: {dupes}")
    for col in ["tf", "species", "assay", "lab"]:
        if col in frame and frame[col].isna().any():
            messages.append(f"column {col} contains missing values")
    if "processed_peak_url" in frame:
        empty = frame["processed_peak_url"].fillna("").eq("").sum()
        if empty:
            messages.append(f"{empty} rows are missing processed_peak_url")
    return messages
