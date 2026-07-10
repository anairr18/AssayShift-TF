from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


def _as_arrays(y_true: Iterable[float], prob: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(list(y_true), dtype=float)
    p = np.asarray(list(prob), dtype=float)
    if y.shape[0] != p.shape[0]:
        raise ValueError("y_true and prob must have the same length")
    if y.shape[0] == 0:
        raise ValueError("cannot score empty arrays")
    return y, np.clip(p, 1e-7, 1 - 1e-7)


def expected_calibration_error(
    y_true: Iterable[float],
    prob: Iterable[float],
    n_bins: int = 15,
) -> float:
    """Binary ECE over predicted positive-class probabilities."""
    y, p = _as_arrays(y_true, prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not np.any(mask):
            continue
        ece += mask.mean() * abs(float(np.mean(p[mask])) - float(np.mean(y[mask])))
    return float(ece)


def binary_metrics(y_true: Iterable[float], prob: Iterable[float], n_bins: int = 15) -> dict[str, float]:
    y, p = _as_arrays(y_true, prob)
    out: dict[str, float] = {
        "n": float(len(y)),
        "prevalence": float(np.mean(y)),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p, n_bins=n_bins),
    }
    if len(np.unique(y)) == 2:
        out["auroc"] = float(roc_auc_score(y, p))
        out["auprc"] = float(average_precision_score(y, p))
    else:
        out["auroc"] = np.nan
        out["auprc"] = np.nan
    return out


def bootstrap_metric_cis(
    y_true: Iterable[float],
    prob: Iterable[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 13,
    n_bins: int = 15,
) -> pd.DataFrame:
    """Percentile bootstrap confidence intervals for binary prediction metrics."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be non-negative")

    y, p = _as_arrays(y_true, prob)
    estimate = binary_metrics(y, p, n_bins=n_bins)
    metric_names = ("auprc", "auroc", "ece", "brier")
    values: dict[str, list[float]] = {name: [] for name in metric_names}
    rng = np.random.default_rng(random_state)

    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y), size=len(y))
        metrics = binary_metrics(y[idx], p[idx], n_bins=n_bins)
        for name in metric_names:
            value = metrics[name]
            if not pd.isna(value):
                values[name].append(float(value))

    alpha = 1.0 - confidence
    rows: list[dict[str, float | int | str]] = []
    for name in metric_names:
        boot = np.asarray(values[name], dtype=float)
        if boot.size:
            ci_low = float(np.quantile(boot, alpha / 2.0))
            ci_high = float(np.quantile(boot, 1.0 - alpha / 2.0))
        else:
            ci_low = float("nan")
            ci_high = float("nan")
        rows.append(
            {
                "metric": name,
                "estimate": float(estimate[name]) if not pd.isna(estimate[name]) else float("nan"),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_bootstrap": int(n_bootstrap),
                "valid_bootstraps": int(boot.size),
                "confidence": float(confidence),
            }
        )
    return pd.DataFrame(rows)


def group_metrics(
    frame: pd.DataFrame,
    label_col: str = "label",
    prob_col: str = "prob",
    group_cols: Iterable[str] = ("assay", "lab", "species", "tf_family"),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for col in group_cols:
        if col not in frame.columns:
            continue
        for value, group in frame.groupby(col, dropna=False):
            metrics = binary_metrics(group[label_col], group[prob_col])
            rows.append({"group_col": col, "group": value, **metrics})
    return pd.DataFrame(rows)


def worst_group_auprc(group_frame: pd.DataFrame) -> float:
    usable = group_frame.dropna(subset=["auprc"])
    if usable.empty:
        return float("nan")
    return float(usable["auprc"].min())


def selective_metrics(
    y_true: Iterable[float],
    prob: Iterable[float],
    coverages: Iterable[float] = (1.0, 0.8, 0.6, 0.4, 0.2),
) -> pd.DataFrame:
    y, p = _as_arrays(y_true, prob)
    confidence = np.abs(p - 0.5)
    order = np.argsort(-confidence)
    rows: list[dict[str, float]] = []
    for coverage in coverages:
        if not 0 < coverage <= 1:
            raise ValueError("coverage values must be in (0, 1]")
        keep = max(1, int(round(len(y) * coverage)))
        idx = order[:keep]
        rows.append({"coverage": float(coverage), **binary_metrics(y[idx], p[idx])})
    return pd.DataFrame(rows)


@dataclass
class PlattCalibrator:
    """Platt scaling on model probabilities using held-out validation data."""

    model: LogisticRegression | None = None
    identity: bool = False

    def fit(self, prob: Iterable[float], y_true: Iterable[float]) -> "PlattCalibrator":
        y, p = _as_arrays(y_true, prob)
        logits = np.log(p / (1.0 - p)).reshape(-1, 1)
        if len(np.unique(y)) < 2 or len(np.unique(logits)) < 2:
            self.identity = True
            return self
        self.model = LogisticRegression(solver="lbfgs")
        self.model.fit(logits, y.astype(int))
        return self

    def predict(self, prob: Iterable[float]) -> np.ndarray:
        if self.identity:
            return np.clip(np.asarray(list(prob), dtype=float), 1e-7, 1 - 1e-7)
        if self.model is None:
            raise RuntimeError("PlattCalibrator.fit must be called before predict")
        p = np.asarray(list(prob), dtype=float)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        logits = np.log(p / (1.0 - p)).reshape(-1, 1)
        return self.model.predict_proba(logits)[:, 1]


@dataclass
class ProtocolPlattCalibrator:
    """Group-specific Platt scaling with a global fallback for unseen protocols."""

    group_col: str
    global_model: PlattCalibrator | None = None
    group_models: dict[str, PlattCalibrator] | None = None
    group_counts: dict[str, int] | None = None

    def fit(
        self,
        prob: Iterable[float],
        y_true: Iterable[float],
        groups: Iterable[object],
    ) -> "ProtocolPlattCalibrator":
        p = np.asarray(list(prob), dtype=float)
        y = np.asarray(list(y_true), dtype=float)
        g = pd.Series(list(groups)).fillna("").astype(str).to_numpy()
        if not (len(p) == len(y) == len(g)):
            raise ValueError("prob, y_true, and groups must have the same length")
        self.global_model = PlattCalibrator().fit(p, y)
        self.group_models = {}
        self.group_counts = {}
        for group in sorted(set(g.tolist())):
            mask = g == group
            self.group_counts[group] = int(mask.sum())
            self.group_models[group] = PlattCalibrator().fit(p[mask], y[mask])
        return self

    def predict(self, prob: Iterable[float], groups: Iterable[object]) -> np.ndarray:
        if self.global_model is None or self.group_models is None:
            raise RuntimeError("ProtocolPlattCalibrator.fit must be called before predict")
        p = np.asarray(list(prob), dtype=float)
        g = pd.Series(list(groups)).fillna("").astype(str).to_numpy()
        out = self.global_model.predict(p)
        for group, model in self.group_models.items():
            mask = g == group
            if np.any(mask):
                out[mask] = model.predict(p[mask])
        return np.clip(out, 1e-7, 1 - 1e-7)

    def report(self, groups: Iterable[object]) -> pd.DataFrame:
        if self.group_models is None or self.group_counts is None:
            raise RuntimeError("ProtocolPlattCalibrator.fit must be called before report")
        requested = pd.Series(list(groups)).fillna("").astype(str)
        rows = [
            {
                "calibration_group_col": self.group_col,
                "calibration_group": "__global__",
                "fit_n": int(sum(self.group_counts.values())),
                "test_n": int(len(requested)),
                "used_group_specific": False,
            }
        ]
        for group, test_n in requested.value_counts(dropna=False).sort_index().items():
            rows.append(
                {
                    "calibration_group_col": self.group_col,
                    "calibration_group": group,
                    "fit_n": int(self.group_counts.get(group, 0)),
                    "test_n": int(test_n),
                    "used_group_specific": group in self.group_models,
                }
            )
        return pd.DataFrame(rows)
