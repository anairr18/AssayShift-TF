from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


GROUP_COLUMNS = {
    "assay": "assay",
    "lab": "lab",
    "species": "species",
    "tf_family": "tf_family",
    "family": "tf_family",
}


def _stratify_or_none(labels: pd.Series) -> pd.Series | None:
    counts = labels.value_counts()
    if counts.shape[0] != 2:
        return None
    if counts.min() < 2:
        return None
    return labels


def choose_holdout_value(frame: pd.DataFrame, column: str) -> object:
    counts = frame[column].value_counts(dropna=False)
    if counts.shape[0] < 2:
        raise ValueError(f"need at least two groups in {column} for held-out evaluation")
    return counts.index[-1]


def make_split(
    frame: pd.DataFrame,
    split_type: str,
    label_col: str = "label",
    holdout: object | None = None,
    test_size: float = 0.2,
    valid_size: float = 0.2,
    random_state: int = 13,
) -> pd.Series:
    """Return train/valid/test labels for IID or group-held-out evaluation."""
    if split_type == "iid":
        stratify_all = _stratify_or_none(frame[label_col])
        train_valid_idx, test_idx = train_test_split(
            frame.index,
            test_size=test_size,
            stratify=stratify_all,
            random_state=random_state,
        )
        stratify_train_valid = _stratify_or_none(frame.loc[train_valid_idx, label_col])
        train_idx, valid_idx = train_test_split(
            train_valid_idx,
            test_size=valid_size,
            stratify=stratify_train_valid,
            random_state=random_state + 1,
        )
    else:
        column = GROUP_COLUMNS.get(split_type, split_type)
        if column not in frame.columns:
            raise ValueError(f"split column {column!r} is missing")
        holdout = choose_holdout_value(frame, column) if holdout is None else holdout
        test_mask = frame[column].eq(holdout)
        if test_mask.sum() == 0:
            raise ValueError(f"holdout value {holdout!r} not found in {column}")
        train_valid_idx = frame.index[~test_mask]
        test_idx = frame.index[test_mask]
        y_train_valid = frame.loc[train_valid_idx, label_col]
        stratify = _stratify_or_none(y_train_valid)
        train_idx, valid_idx = train_test_split(
            train_valid_idx,
            test_size=valid_size,
            stratify=stratify,
            random_state=random_state + 1,
        )

    split = pd.Series("unused", index=frame.index, dtype=object)
    split.loc[np.asarray(train_idx)] = "train"
    split.loc[np.asarray(valid_idx)] = "valid"
    split.loc[np.asarray(test_idx)] = "test"
    return split
