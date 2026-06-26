from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CATEGORICAL_COLUMNS = ["assay", "lab", "species", "tf", "tf_family"]
NUMERIC_COLUMNS = ["gc", "n_fraction", "length"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    kmer_k: int = 5
    deep_epochs: int = 10
    deep_batch_size: int = 128
    deep_lr: float = 1e-3
    deep_device: str = "auto"
    axis_dropout: float = 0.15
    counterfactual_weight: float = 0.35
    metadata_residual_weight: float = 0.02
    adversarial_weight: float = 0.0
    random_state: int = 13


DEEP_MODEL_KINDS = {"tiny_cnn", "axis_guard_cnn"}
METADATA_MODEL_KINDS = {"kmer_metadata", "axis_guard_cnn"}


def _available_columns(frame: pd.DataFrame, candidates: list[str]) -> list[str]:
    return [col for col in candidates if col in frame.columns]


def model_uses_metadata(spec: ModelSpec) -> bool:
    return spec.kind in METADATA_MODEL_KINDS


def model_spec_from_name(
    name: str,
    *,
    deep_epochs: int | None = None,
    deep_batch_size: int | None = None,
    deep_lr: float | None = None,
    deep_device: str | None = None,
    axis_dropout: float | None = None,
    counterfactual_weight: float | None = None,
    metadata_residual_weight: float | None = None,
    adversarial_weight: float | None = None,
    random_state: int = 13,
) -> ModelSpec:
    aliases = {
        "gc": {"name": "gc_artifact_logreg", "kind": "gc"},
        "gc_artifact_logreg": {"name": "gc_artifact_logreg", "kind": "gc"},
        "kmer": {"name": "kmer_logreg", "kind": "kmer"},
        "kmer_logreg": {"name": "kmer_logreg", "kind": "kmer"},
        "kmer_metadata": {"name": "kmer_metadata_logreg", "kind": "kmer_metadata"},
        "kmer_metadata_logreg": {"name": "kmer_metadata_logreg", "kind": "kmer_metadata"},
        "tiny_cnn": {"name": "tiny_cnn", "kind": "tiny_cnn"},
        "axis_guard_cnn": {"name": "axis_guard_cnn", "kind": "axis_guard_cnn"},
        "picard_tf": {"name": "axis_guard_full", "kind": "axis_guard_cnn", "adversarial_weight": 0.02},
        "axis_guard_full": {"name": "axis_guard_full", "kind": "axis_guard_cnn", "adversarial_weight": 0.02},
        "axis_guard_no_cf": {
            "name": "axis_guard_no_cf",
            "kind": "axis_guard_cnn",
            "counterfactual_weight": 0.0,
            "adversarial_weight": 0.02,
        },
        "axis_guard_no_resid": {
            "name": "axis_guard_no_resid",
            "kind": "axis_guard_cnn",
            "metadata_residual_weight": 0.0,
            "adversarial_weight": 0.02,
        },
        "axis_guard_no_adv": {
            "name": "axis_guard_no_adv",
            "kind": "axis_guard_cnn",
            "adversarial_weight": 0.0,
        },
    }
    key = name.strip().lower()
    if key not in aliases:
        raise ValueError(f"unknown model {name!r}; choose one of {sorted(aliases)}")
    alias: dict[str, Any] = {
        "deep_epochs": 10,
        "deep_batch_size": 128,
        "deep_lr": 1e-3,
        "deep_device": "auto",
        "axis_dropout": 0.15,
        "counterfactual_weight": 0.35,
        "metadata_residual_weight": 0.02,
        "adversarial_weight": 0.0,
        **aliases[key],
    }
    overrides = {
        "deep_epochs": deep_epochs,
        "deep_batch_size": deep_batch_size,
        "deep_lr": deep_lr,
        "deep_device": deep_device,
        "axis_dropout": axis_dropout,
        "counterfactual_weight": counterfactual_weight,
        "metadata_residual_weight": metadata_residual_weight,
        "adversarial_weight": adversarial_weight,
    }
    for field, value in overrides.items():
        if value is not None:
            alias[field] = value
    return ModelSpec(
        alias["name"],
        alias["kind"],
        deep_epochs=alias["deep_epochs"],
        deep_batch_size=alias["deep_batch_size"],
        deep_lr=alias["deep_lr"],
        deep_device=alias["deep_device"],
        axis_dropout=alias["axis_dropout"],
        counterfactual_weight=alias["counterfactual_weight"],
        metadata_residual_weight=alias["metadata_residual_weight"],
        adversarial_weight=alias["adversarial_weight"],
        random_state=random_state,
    )


def build_model(spec: ModelSpec, example_frame: pd.DataFrame):
    if spec.kind in DEEP_MODEL_KINDS:
        from assayshift_tf.deep import DeepModelConfig, TorchProbClassifier

        config = DeepModelConfig(
            epochs=spec.deep_epochs,
            batch_size=spec.deep_batch_size,
            lr=spec.deep_lr,
            device=spec.deep_device,
            axis_dropout=spec.axis_dropout,
            counterfactual_weight=spec.counterfactual_weight,
            metadata_residual_weight=spec.metadata_residual_weight,
            adversarial_weight=spec.adversarial_weight,
            random_state=spec.random_state,
        )
        return TorchProbClassifier(spec.kind, config)

    transformers = []
    if spec.kind in {"kmer", "kmer_metadata"}:
        transformers.append(
            (
                "kmer",
                CountVectorizer(
                    analyzer="char",
                    ngram_range=(spec.kmer_k, spec.kmer_k),
                    lowercase=False,
                    min_df=2,
                ),
                "sequence",
            )
        )
    if spec.kind in {"gc", "kmer_metadata"}:
        numeric = _available_columns(example_frame, NUMERIC_COLUMNS)
        if numeric:
            transformers.append(("numeric", StandardScaler(), numeric))
    if spec.kind == "kmer_metadata":
        categorical = _available_columns(example_frame, CATEGORICAL_COLUMNS)
        if categorical:
            transformers.append(("metadata", OneHotEncoder(handle_unknown="ignore"), categorical))

    if not transformers:
        raise ValueError(f"unknown or unsupported model kind: {spec.kind}")

    features = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)
    classifier = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    return Pipeline([("features", features), ("classifier", classifier)])
