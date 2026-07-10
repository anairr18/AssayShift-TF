from __future__ import annotations

from itertools import permutations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from assayshift_tf.datasets import DemoConfig, make_demo_dataset
from assayshift_tf.features import add_sequence_stats
from assayshift_tf.metrics import (
    PlattCalibrator,
    ProtocolPlattCalibrator,
    binary_metrics,
    bootstrap_metric_cis,
    group_metrics,
    selective_metrics,
    worst_group_auprc,
)
from assayshift_tf.models import ModelSpec, build_model, model_uses_metadata
from assayshift_tf.splits import GROUP_COLUMNS, choose_holdout_value, make_split


@dataclass(frozen=True)
class SplitSpec:
    name: str
    split_type: str
    holdout: object | None = None


@dataclass(frozen=True)
class EvaluationArtifacts:
    results: pd.DataFrame
    groups: pd.DataFrame
    selective: pd.DataFrame
    predictions: pd.DataFrame
    bootstrap_cis: pd.DataFrame
    split_counts: pd.DataFrame
    calibration: pd.DataFrame


@dataclass(frozen=True)
class WindowFilterArtifacts:
    frame: pd.DataFrame
    report: pd.DataFrame


DEFAULT_SPLITS = [
    SplitSpec("iid", "iid"),
    SplitSpec("assay_heldout_cutrun", "assay", "CUT&RUN"),
    SplitSpec("lab_heldout_henikoff", "lab", "Henikoff"),
    SplitSpec("species_heldout_mouse", "species", "Mus musculus"),
    SplitSpec("family_heldout_zinc_finger", "tf_family", "zinc_finger"),
]

DEFAULT_REAL_SPLITS = [
    SplitSpec("iid", "iid"),
    SplitSpec("assay_heldout", "assay"),
    SplitSpec("lab_heldout", "lab"),
    SplitSpec("species_heldout", "species"),
    SplitSpec("family_heldout", "tf_family"),
]

DEFAULT_MODELS = [
    ModelSpec("gc_artifact_logreg", "gc"),
    ModelSpec("kmer_logreg", "kmer"),
    ModelSpec("kmer_metadata_logreg", "kmer_metadata"),
]

REQUIRED_WINDOW_COLUMNS = {"label", "sequence"}
SEQUENCE_STAT_COLUMNS = {"gc", "n_fraction", "length"}
PREDICTION_BASE_COLUMNS = ["example_id", "label", "prob", "split_name", "model", "calibrated"]
PREDICTION_OPTIONAL_COLUMNS = [
    "dataset_id",
    "assay",
    "lab",
    "species",
    "tf",
    "tf_family",
    "biosample",
    "cell_type",
    "assembly",
    "chrom",
    "start",
    "end",
    "gc",
    "n_fraction",
    "length",
]
PAIRWISE_RESULT_METRICS = ("auprc", "auroc", "ece", "brier", "worst_group_auprc")
PAIRWISE_SELECTIVE_METRICS = ("auprc", "auroc", "ece", "brier")
PAIRWISE_METRIC_DIRECTIONS = {
    "auprc": "higher",
    "auroc": "higher",
    "worst_group_auprc": "higher",
    "ece": "lower",
    "brier": "lower",
}


def _pairwise_delta_columns(id_cols: list[str]) -> list[str]:
    return [
        *id_cols,
        "model_a",
        "model_b",
        "metric",
        "direction",
        "model_a_value",
        "model_b_value",
        "delta",
        "delta_favors_model_a",
        "better_model",
    ]


def pairwise_metric_deltas(frame: pd.DataFrame, metrics: tuple[str, ...]) -> pd.DataFrame:
    """Directed model-vs-model metric deltas from an already-scored results table."""
    id_cols = [
        col
        for col in ["seed", "split", "split_type", "holdout", "calibrated", "coverage"]
        if col in frame.columns
    ]
    columns = _pairwise_delta_columns(id_cols)
    if frame.empty or "model" not in frame.columns:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    grouped = frame.groupby(id_cols, dropna=False) if id_cols else [((), frame)]
    for keys, group in grouped:
        key_values = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(id_cols, key_values, strict=False))
        models = list(dict.fromkeys(group["model"].astype(str).tolist()))
        if len(models) < 2:
            continue
        by_model = {str(row.model): row for row in group.itertuples(index=False)}
        for model_a, model_b in permutations(models, 2):
            row_a = by_model[model_a]
            row_b = by_model[model_b]
            for metric in metrics:
                if metric not in group.columns:
                    continue
                value_a = pd.to_numeric(getattr(row_a, metric), errors="coerce")
                value_b = pd.to_numeric(getattr(row_b, metric), errors="coerce")
                delta = float(value_a - value_b) if not (pd.isna(value_a) or pd.isna(value_b)) else float("nan")
                direction = PAIRWISE_METRIC_DIRECTIONS.get(metric, "higher")
                favors_a = (
                    delta > 0
                    if direction == "higher"
                    else delta < 0
                    if direction == "lower"
                    else False
                )
                better_model = ""
                if not pd.isna(delta) and delta != 0:
                    better_model = model_a if favors_a else model_b
                rows.append(
                    {
                        **base,
                        "model_a": model_a,
                        "model_b": model_b,
                        "metric": metric,
                        "direction": direction,
                        "model_a_value": float(value_a) if not pd.isna(value_a) else float("nan"),
                        "model_b_value": float(value_b) if not pd.isna(value_b) else float("nan"),
                        "delta": delta,
                        "delta_favors_model_a": bool(favors_a),
                        "better_model": better_model,
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def summarize_pairwise_deltas(deltas: pd.DataFrame) -> pd.DataFrame:
    """Summarize directed pairwise deltas across seeds."""
    id_cols = [
        col
        for col in ["split", "split_type", "holdout", "calibrated", "coverage", "model_a", "model_b", "metric", "direction"]
        if col in deltas.columns
    ]
    columns = [
        *id_cols,
        "mean_delta",
        "std_delta",
        "min_delta",
        "max_delta",
        "n",
        "n_seeds",
        "model_a_win_rate",
        "mean_delta_favors_model_a",
    ]
    if deltas.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for keys, group in deltas.groupby(id_cols, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(id_cols, key_values, strict=False))
        values = pd.to_numeric(group["delta"], errors="coerce").dropna()
        favors = group.loc[values.index, "delta_favors_model_a"].astype(float) if not values.empty else pd.Series(dtype=float)
        rows.append(
            {
                **base,
                "mean_delta": float(values.mean()) if not values.empty else float("nan"),
                "std_delta": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min_delta": float(values.min()) if not values.empty else float("nan"),
                "max_delta": float(values.max()) if not values.empty else float("nan"),
                "n": int(values.shape[0]),
                "n_seeds": int(group["seed"].nunique()) if "seed" in group.columns else int(values.shape[0]),
                "model_a_win_rate": float(favors.mean()) if not favors.empty else float("nan"),
                "mean_delta_favors_model_a": bool(values.mean() > 0)
                if base.get("direction", "higher") == "higher" and not values.empty
                else bool(values.mean() < 0)
                if base.get("direction") == "lower" and not values.empty
                else False,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def paired_prediction_delta_cis(
    predictions: pd.DataFrame,
    split_meta: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = PAIRWISE_SELECTIVE_METRICS,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 13,
) -> pd.DataFrame:
    """Paired bootstrap CIs for directed model deltas on identical test examples."""
    columns = [
        "split",
        "split_type",
        "holdout",
        "calibrated",
        "model_a",
        "model_b",
        "metric",
        "direction",
        "model_a_value",
        "model_b_value",
        "delta",
        "ci_low",
        "ci_high",
        "n_bootstrap",
        "valid_bootstraps",
        "confidence",
        "delta_favors_model_a",
        "better_model",
    ]
    if predictions.empty or n_bootstrap <= 0:
        return pd.DataFrame(columns=columns)
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")

    meta = split_meta.drop_duplicates("split").set_index("split") if not split_meta.empty else pd.DataFrame()
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(random_state)
    for (split_name, calibrated), group in predictions.groupby(["split_name", "calibrated"], dropna=False):
        models = list(dict.fromkeys(group["model"].astype(str).tolist()))
        if len(models) < 2:
            continue
        model_frames = {
            model: (
                group.loc[group["model"].astype(str).eq(model), ["example_id", "label", "prob"]]
                .drop_duplicates("example_id")
                .rename(columns={"prob": f"prob_{model}"})
            )
            for model in models
        }
        for model_a, model_b in permutations(models, 2):
            aligned = model_frames[model_a].merge(
                model_frames[model_b],
                on=["example_id", "label"],
                how="inner",
            )
            if aligned.empty:
                continue
            y = aligned["label"].to_numpy(dtype=float)
            prob_a = aligned[f"prob_{model_a}"].to_numpy(dtype=float)
            prob_b = aligned[f"prob_{model_b}"].to_numpy(dtype=float)
            estimate_a = binary_metrics(y, prob_a)
            estimate_b = binary_metrics(y, prob_b)
            boot_values: dict[str, list[float]] = {metric: [] for metric in metrics}
            for _ in range(n_bootstrap):
                idx = rng.integers(0, len(y), size=len(y))
                metrics_a = binary_metrics(y[idx], prob_a[idx])
                metrics_b = binary_metrics(y[idx], prob_b[idx])
                for metric in metrics:
                    value_a = metrics_a.get(metric, float("nan"))
                    value_b = metrics_b.get(metric, float("nan"))
                    if not (pd.isna(value_a) or pd.isna(value_b)):
                        boot_values[metric].append(float(value_a - value_b))

            alpha = 1.0 - confidence
            split_type = meta.loc[split_name, "split_type"] if not meta.empty and split_name in meta.index else ""
            holdout = meta.loc[split_name, "holdout"] if not meta.empty and split_name in meta.index else ""
            for metric in metrics:
                value_a = estimate_a.get(metric, float("nan"))
                value_b = estimate_b.get(metric, float("nan"))
                delta = float(value_a - value_b) if not (pd.isna(value_a) or pd.isna(value_b)) else float("nan")
                boot = np.asarray(boot_values[metric], dtype=float)
                ci_low = float(np.quantile(boot, alpha / 2.0)) if boot.size else float("nan")
                ci_high = float(np.quantile(boot, 1.0 - alpha / 2.0)) if boot.size else float("nan")
                direction = PAIRWISE_METRIC_DIRECTIONS.get(metric, "higher")
                favors_a = (
                    delta > 0
                    if direction == "higher"
                    else delta < 0
                    if direction == "lower"
                    else False
                )
                rows.append(
                    {
                        "split": split_name,
                        "split_type": split_type,
                        "holdout": holdout if holdout is not None else "",
                        "calibrated": calibrated,
                        "model_a": model_a,
                        "model_b": model_b,
                        "metric": metric,
                        "direction": direction,
                        "model_a_value": float(value_a) if not pd.isna(value_a) else float("nan"),
                        "model_b_value": float(value_b) if not pd.isna(value_b) else float("nan"),
                        "delta": delta,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "n_bootstrap": int(n_bootstrap),
                        "valid_bootstraps": int(boot.size),
                        "confidence": float(confidence),
                        "delta_favors_model_a": bool(favors_a),
                        "better_model": model_a if favors_a else model_b if not pd.isna(delta) and delta != 0 else "",
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def _predict_positive(model, frame: pd.DataFrame) -> pd.Series:
    return pd.Series(model.predict_proba(frame)[:, 1], index=frame.index)


def _calibration_group_column(frame: pd.DataFrame, split_spec: SplitSpec, requested: str) -> str | None:
    if requested != "auto":
        return requested if requested in frame.columns else None
    if split_spec.split_type in {"assay", "lab", "species", "tf", "tf_family"} and split_spec.split_type in frame.columns:
        return split_spec.split_type
    for col in ("assay", "lab", "species", "tf_family", "tf"):
        if col in frame.columns:
            return col
    return None


def _read_window_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path)
    suffix = table_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(table_path)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(table_path, sep="\t")
    return pd.read_csv(table_path)


def _prepare_window_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_WINDOW_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"window table is missing required columns: {sorted(missing)}")

    out = frame.copy()
    if "example_id" not in out.columns:
        out.insert(0, "example_id", [f"window_{i:08d}" for i in range(len(out))])
    out["label"] = out["label"].astype(int)
    if not set(out["label"].dropna().unique()).issubset({0, 1}):
        raise ValueError("label column must contain binary 0/1 values")
    if not SEQUENCE_STAT_COLUMNS <= set(out.columns):
        out = add_sequence_stats(out)
    return out


def filter_window_frame(
    frame: pd.DataFrame,
    *,
    drop_all_n: bool = False,
    drop_duplicate_sequences: bool = False,
) -> WindowFilterArtifacts:
    """Apply paper-facing leakage/QC filters and report exactly what changed."""
    out = frame.copy()
    rows: list[dict[str, object]] = []

    def add_row(step: str, before: int, after: int) -> None:
        labels = out["label"] if "label" in out.columns else pd.Series(dtype=int)
        positives = int(labels.sum()) if not labels.empty else 0
        rows.append(
            {
                "step": step,
                "before_n": int(before),
                "after_n": int(after),
                "removed_n": int(before - after),
                "positive": positives,
                "negative": int(after - positives),
                "prevalence": float(positives / after) if after else float("nan"),
            }
        )

    add_row("input", len(out), len(out))
    if drop_all_n:
        before = len(out)
        sequence = out["sequence"].fillna("").astype(str).str.upper()
        all_n = sequence.str.fullmatch("N+").fillna(False)
        out = out.loc[~all_n].copy()
        add_row("drop_all_n_sequences", before, len(out))

    if drop_duplicate_sequences:
        before = len(out)
        duplicate = out["sequence"].fillna("").astype(str).duplicated(keep="first")
        out = out.loc[~duplicate].copy()
        add_row("drop_duplicate_sequences", before, len(out))

    out = out.reset_index(drop=True)
    return WindowFilterArtifacts(out, pd.DataFrame(rows))


def _resolve_split_spec(frame: pd.DataFrame, split_spec: SplitSpec) -> SplitSpec:
    if split_spec.split_type == "iid" or split_spec.holdout is not None:
        return split_spec
    column = GROUP_COLUMNS.get(split_spec.split_type, split_spec.split_type)
    if column not in frame.columns:
        raise ValueError(f"split column {column!r} is missing")
    return SplitSpec(split_spec.name, split_spec.split_type, choose_holdout_value(frame, column))


def _split_count_rows(frame: pd.DataFrame, split: pd.Series, split_spec: SplitSpec) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for partition in ("train", "valid", "test"):
        labels = frame.loc[split.eq(partition), "label"]
        n = int(labels.shape[0])
        positives = int(labels.sum()) if n else 0
        rows.append(
            {
                "split": split_spec.name,
                "split_type": split_spec.split_type,
                "holdout": split_spec.holdout if split_spec.holdout is not None else "",
                "partition": partition,
                "n": n,
                "positive": positives,
                "negative": n - positives,
                "prevalence": float(positives / n) if n else float("nan"),
            }
        )
    return rows


def _prediction_columns(frame: pd.DataFrame) -> list[str]:
    return PREDICTION_BASE_COLUMNS + [col for col in PREDICTION_OPTIONAL_COLUMNS if col in frame.columns]


def _mask_leaky_metadata(frame: pd.DataFrame, split_spec: SplitSpec) -> pd.DataFrame:
    out = frame.copy()
    for col in ("assay", "lab", "species", "tf", "tf_family"):
        if col in out.columns and f"__group_{col}" not in out.columns:
            out[f"__group_{col}"] = out[col]
    mask_cols = {
        "assay": ["assay", "lab"],
        "lab": ["lab"],
        "species": ["species", "assembly"],
        "tf": ["tf", "tf_family"],
        "tf_family": ["tf", "tf_family"],
        "family": ["tf", "tf_family"],
    }.get(split_spec.split_type, [])
    for col in mask_cols:
        if col in out.columns:
            out[col] = "masked_for_" + split_spec.split_type
    return out


def parse_split_spec(value: str) -> SplitSpec:
    """Parse CLI split specs.

    Supported forms:
    - iid
    - name:type
    - name:type:holdout
    - name=type
    - name=type:holdout
    """
    if not value:
        raise ValueError("split spec cannot be empty")
    if "=" in value:
        name, rest = value.split("=", 1)
        parts = rest.split(":", 1)
        split_type = parts[0]
        holdout = parts[1] if len(parts) == 2 and parts[1] else None
    else:
        parts = value.split(":", 2)
        if len(parts) == 1:
            name = parts[0]
            split_type = parts[0]
            holdout = None
        elif len(parts) == 2:
            name, split_type = parts
            holdout = None
        else:
            name, split_type, holdout = parts
            holdout = holdout or None
    if not name or not split_type:
        raise ValueError(f"invalid split spec: {value!r}")
    return SplitSpec(name, split_type, holdout)


def evaluate_frame_full(
    frame: pd.DataFrame,
    split_specs: list[SplitSpec] | None = None,
    model_specs: list[ModelSpec] | None = None,
    random_state: int = 13,
    bootstrap_iterations: int = 0,
    ci_confidence: float = 0.95,
    calibration_method: str = "platt",
    calibration_group: str = "auto",
) -> EvaluationArtifacts:
    split_specs = DEFAULT_SPLITS if split_specs is None else split_specs
    model_specs = DEFAULT_MODELS if model_specs is None else model_specs

    result_rows: list[dict[str, object]] = []
    group_rows: list[pd.DataFrame] = []
    selective_rows: list[pd.DataFrame] = []
    prediction_rows: list[pd.DataFrame] = []
    ci_rows: list[pd.DataFrame] = []
    split_count_rows: list[dict[str, object]] = []
    calibration_rows: list[pd.DataFrame] = []

    for split_idx, raw_split_spec in enumerate(split_specs):
        split_spec = _resolve_split_spec(frame, raw_split_spec)
        split = make_split(
            frame,
            split_spec.split_type,
            holdout=split_spec.holdout,
            random_state=random_state,
        )
        split_count_rows.extend(_split_count_rows(frame, split, split_spec))
        train_original = frame.loc[split.eq("train")].copy()
        valid_original = frame.loc[split.eq("valid")].copy()
        test_original = frame.loc[split.eq("test")].copy()

        for model_idx, model_spec in enumerate(model_specs):
            if model_uses_metadata(model_spec):
                model_frame = _mask_leaky_metadata(frame, split_spec)
                train = model_frame.loc[split.eq("train")].copy()
                valid = model_frame.loc[split.eq("valid")].copy()
                test = model_frame.loc[split.eq("test")].copy()
            else:
                train = train_original
                valid = valid_original
                test = test_original

            model = build_model(model_spec, train)
            model.fit(train, train["label"])
            valid_prob = _predict_positive(model, valid)
            test_prob = _predict_positive(model, test)
            group_col = _calibration_group_column(valid_original, split_spec, calibration_group)
            if calibration_method == "protocol_platt" and group_col is not None:
                calibrator = ProtocolPlattCalibrator(group_col).fit(
                    valid_prob,
                    valid_original["label"],
                    valid_original[group_col],
                )
                test_prob_calibrated = calibrator.predict(test_prob, test_original[group_col])
                calibration_report = calibrator.report(test_original[group_col])
            elif calibration_method == "platt":
                calibrator = PlattCalibrator().fit(valid_prob, valid["label"])
                test_prob_calibrated = calibrator.predict(test_prob)
                calibration_report = pd.DataFrame(
                    [
                        {
                            "calibration_group_col": "__global__",
                            "calibration_group": "__global__",
                            "fit_n": int(len(valid_original)),
                            "test_n": int(len(test_original)),
                            "used_group_specific": False,
                        }
                    ]
                )
            else:
                raise ValueError("calibration_method must be one of: platt, protocol_platt")
            calibration_report.insert(0, "model", model_spec.name)
            calibration_report.insert(0, "holdout", split_spec.holdout if split_spec.holdout is not None else "")
            calibration_report.insert(0, "split_type", split_spec.split_type)
            calibration_report.insert(0, "split", split_spec.name)
            calibration_report.insert(0, "calibration_method", calibration_method)
            calibration_rows.append(calibration_report)

            for calibrated, prob in [(False, test_prob), (True, test_prob_calibrated)]:
                pred_frame = test_original.copy()
                pred_frame["prob"] = prob
                pred_frame["split_name"] = split_spec.name
                pred_frame["model"] = model_spec.name
                pred_frame["calibrated"] = calibrated
                prediction_rows.append(pred_frame[_prediction_columns(pred_frame)])

                metrics = binary_metrics(pred_frame["label"], pred_frame["prob"])
                gm = group_metrics(pred_frame)
                result_rows.append(
                    {
                        "split": split_spec.name,
                        "split_type": split_spec.split_type,
                        "holdout": split_spec.holdout if split_spec.holdout is not None else "",
                        "model": model_spec.name,
                        "calibrated": calibrated,
                        **metrics,
                        "worst_group_auprc": worst_group_auprc(gm),
                    }
                )
                gm.insert(0, "calibrated", calibrated)
                gm.insert(0, "model", model_spec.name)
                gm.insert(0, "split", split_spec.name)
                group_rows.append(gm)

                sm = selective_metrics(pred_frame["label"], pred_frame["prob"])
                sm.insert(0, "calibrated", calibrated)
                sm.insert(0, "model", model_spec.name)
                sm.insert(0, "split", split_spec.name)
                selective_rows.append(sm)

                if bootstrap_iterations:
                    ci = bootstrap_metric_cis(
                        pred_frame["label"],
                        pred_frame["prob"],
                        n_bootstrap=bootstrap_iterations,
                        confidence=ci_confidence,
                        random_state=random_state + split_idx * 100_000 + model_idx * 1_000 + int(calibrated) * 17,
                    )
                    ci.insert(0, "calibrated", calibrated)
                    ci.insert(0, "model", model_spec.name)
                    ci.insert(0, "holdout", split_spec.holdout if split_spec.holdout is not None else "")
                    ci.insert(0, "split_type", split_spec.split_type)
                    ci.insert(0, "split", split_spec.name)
                    ci_rows.append(ci)

    results = pd.DataFrame(result_rows)
    groups = pd.concat(group_rows, ignore_index=True) if group_rows else pd.DataFrame()
    selective = pd.concat(selective_rows, ignore_index=True) if selective_rows else pd.DataFrame()
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    bootstrap_cis = pd.concat(ci_rows, ignore_index=True) if ci_rows else pd.DataFrame()
    split_counts = pd.DataFrame(split_count_rows)
    calibration = pd.concat(calibration_rows, ignore_index=True) if calibration_rows else pd.DataFrame()
    return EvaluationArtifacts(results, groups, selective, predictions, bootstrap_cis, split_counts, calibration)


def evaluate_frame(
    frame: pd.DataFrame,
    split_specs: list[SplitSpec] | None = None,
    model_specs: list[ModelSpec] | None = None,
    random_state: int = 13,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    artifacts = evaluate_frame_full(
        frame,
        split_specs=split_specs,
        model_specs=model_specs,
        random_state=random_state,
    )
    return artifacts.results, artifacts.groups, artifacts.selective, artifacts.predictions


def run_demo(
    n_examples: int,
    out_dir: str | Path,
    figures_dir: str | Path,
    random_seed: int = 13,
) -> dict[str, Path]:
    out_path = Path(out_dir)
    fig_path = Path(figures_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig_path.mkdir(parents=True, exist_ok=True)

    frame = make_demo_dataset(DemoConfig(n_examples=n_examples, random_seed=random_seed))
    results, groups, selective, predictions = evaluate_frame(frame, random_state=random_seed)

    paths = {
        "dataset": out_path / "demo_dataset.parquet",
        "results": out_path / "demo_results.csv",
        "groups": out_path / "demo_group_metrics.csv",
        "selective": out_path / "demo_selective.csv",
        "predictions": out_path / "demo_predictions.csv",
        "summary": out_path / "preliminary_results.md",
        "figure": fig_path / "demo_shift_results.png",
    }
    try:
        frame.to_parquet(paths["dataset"], index=False)
    except Exception:
        paths["dataset"] = out_path / "demo_dataset.csv"
        frame.to_csv(paths["dataset"], index=False)
    results.to_csv(paths["results"], index=False)
    groups.to_csv(paths["groups"], index=False)
    selective.to_csv(paths["selective"], index=False)
    predictions.to_csv(paths["predictions"], index=False)

    from assayshift_tf.figures import plot_demo_results
    from assayshift_tf.report import write_preliminary_summary

    plot_demo_results(results, selective, paths["figure"])
    write_preliminary_summary(results, selective, paths["summary"])
    return paths


def run_real_data_evaluation(
    window_table: str | Path,
    out_dir: str | Path,
    figures_dir: str | Path,
    prefix: str = "real",
    split_specs: list[SplitSpec] | None = None,
    model_specs: list[ModelSpec] | None = None,
    random_seed: int = 13,
    bootstrap_iterations: int = 1000,
    ci_confidence: float = 0.95,
    calibration_method: str = "platt",
    calibration_group: str = "auto",
    drop_all_n: bool = False,
    drop_duplicate_sequences: bool = False,
) -> dict[str, Path]:
    out_path = Path(out_dir)
    fig_path = Path(figures_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig_path.mkdir(parents=True, exist_ok=True)

    filter_artifacts = filter_window_frame(
        _prepare_window_frame(_read_window_table(window_table)),
        drop_all_n=drop_all_n,
        drop_duplicate_sequences=drop_duplicate_sequences,
    )
    frame = filter_artifacts.frame
    artifacts = evaluate_frame_full(
        frame,
        split_specs=DEFAULT_REAL_SPLITS if split_specs is None else split_specs,
        model_specs=model_specs,
        random_state=random_seed,
        bootstrap_iterations=bootstrap_iterations,
        ci_confidence=ci_confidence,
        calibration_method=calibration_method,
        calibration_group=calibration_group,
    )

    paths = {
        "results": out_path / f"{prefix}_results.csv",
        "groups": out_path / f"{prefix}_group_metrics.csv",
        "selective": out_path / f"{prefix}_selective.csv",
        "predictions": out_path / f"{prefix}_predictions.csv",
        "bootstrap_cis": out_path / f"{prefix}_bootstrap_cis.csv",
        "pairwise_deltas": out_path / f"{prefix}_pairwise_deltas.csv",
        "pairwise_delta_cis": out_path / f"{prefix}_pairwise_delta_cis.csv",
        "selective_pairwise_deltas": out_path / f"{prefix}_selective_pairwise_deltas.csv",
        "calibration_report": out_path / f"{prefix}_calibration_report.csv",
        "split_counts": out_path / f"{prefix}_split_counts.csv",
        "window_filter_report": out_path / f"{prefix}_window_filter_report.csv",
        "summary": out_path / f"{prefix}_preliminary_results.md",
        "figure": fig_path / f"{prefix}_shift_results.png",
        "reliability_figure": fig_path / f"{prefix}_reliability.png",
        "model_schematic": fig_path / "picard_tf_schematic.png",
    }
    artifacts.results.to_csv(paths["results"], index=False)
    artifacts.groups.to_csv(paths["groups"], index=False)
    artifacts.selective.to_csv(paths["selective"], index=False)
    artifacts.predictions.to_csv(paths["predictions"], index=False)
    artifacts.bootstrap_cis.to_csv(paths["bootstrap_cis"], index=False)
    pairwise_metric_deltas(artifacts.results, PAIRWISE_RESULT_METRICS).to_csv(paths["pairwise_deltas"], index=False)
    paired_prediction_delta_cis(
        artifacts.predictions,
        artifacts.results[["split", "split_type", "holdout"]],
        n_bootstrap=bootstrap_iterations,
        confidence=ci_confidence,
        random_state=random_seed + 77_777,
    ).to_csv(paths["pairwise_delta_cis"], index=False)
    pairwise_metric_deltas(artifacts.selective, PAIRWISE_SELECTIVE_METRICS).to_csv(
        paths["selective_pairwise_deltas"],
        index=False,
    )
    artifacts.calibration.to_csv(paths["calibration_report"], index=False)
    artifacts.split_counts.to_csv(paths["split_counts"], index=False)
    filter_artifacts.report.to_csv(paths["window_filter_report"], index=False)

    from assayshift_tf.figures import plot_picard_schematic, plot_real_results, plot_reliability_curves
    from assayshift_tf.report import write_real_preliminary_summary

    plot_real_results(artifacts.results, artifacts.selective, paths["figure"])
    plot_reliability_curves(artifacts.predictions, paths["reliability_figure"])
    plot_picard_schematic(paths["model_schematic"])
    write_real_preliminary_summary(
        artifacts.results,
        artifacts.selective,
        artifacts.bootstrap_cis,
        artifacts.split_counts,
        paths["summary"],
        source_table=Path(window_table),
    )
    return paths


def summarize_seed_results(frame: pd.DataFrame, metrics: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if frame.empty:
        return pd.DataFrame()
    id_cols = [col for col in ["split", "split_type", "holdout", "model", "calibrated", "coverage"] if col in frame.columns]
    grouped = frame.groupby(id_cols, dropna=False)
    for keys, group in grouped:
        key_values = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(id_cols, key_values, strict=False))
        for metric in metrics:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "mean": float(values.mean()) if not values.empty else float("nan"),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "min": float(values.min()) if not values.empty else float("nan"),
                    "max": float(values.max()) if not values.empty else float("nan"),
                    "n_seeds": int(values.shape[0]),
                }
            )
    return pd.DataFrame(rows)


def run_real_seed_sweep(
    window_table: str | Path,
    out_dir: str | Path,
    prefix: str = "real_seed_sweep",
    split_specs: list[SplitSpec] | None = None,
    model_specs: list[ModelSpec] | None = None,
    seeds: list[int] | None = None,
    bootstrap_iterations: int = 0,
    ci_confidence: float = 0.95,
    calibration_method: str = "platt",
    calibration_group: str = "auto",
    drop_all_n: bool = False,
    drop_duplicate_sequences: bool = False,
) -> dict[str, Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    seeds = [13, 17, 23, 29, 31] if seeds is None else seeds
    filter_artifacts = filter_window_frame(
        _prepare_window_frame(_read_window_table(window_table)),
        drop_all_n=drop_all_n,
        drop_duplicate_sequences=drop_duplicate_sequences,
    )
    frame = filter_artifacts.frame
    paths = {
        "seed_results": out_path / f"{prefix}_seed_results.csv",
        "seed_selective": out_path / f"{prefix}_seed_selective.csv",
        "seed_split_counts": out_path / f"{prefix}_seed_split_counts.csv",
        "seed_bootstrap_cis": out_path / f"{prefix}_seed_bootstrap_cis.csv",
        "seed_pairwise_deltas": out_path / f"{prefix}_seed_pairwise_deltas.csv",
        "seed_pairwise_delta_summary": out_path / f"{prefix}_seed_pairwise_delta_summary.csv",
        "seed_selective_pairwise_deltas": out_path / f"{prefix}_seed_selective_pairwise_deltas.csv",
        "seed_selective_pairwise_delta_summary": out_path / f"{prefix}_seed_selective_pairwise_delta_summary.csv",
        "seed_calibration_report": out_path / f"{prefix}_seed_calibration_report.csv",
        "seed_result_summary": out_path / f"{prefix}_seed_result_summary.csv",
        "seed_selective_summary": out_path / f"{prefix}_seed_selective_summary.csv",
        "window_filter_report": out_path / f"{prefix}_window_filter_report.csv",
    }
    filter_artifacts.report.to_csv(paths["window_filter_report"], index=False)

    result_frames: list[pd.DataFrame] = []
    selective_frames: list[pd.DataFrame] = []
    split_count_frames: list[pd.DataFrame] = []
    ci_frames: list[pd.DataFrame] = []
    calibration_frames: list[pd.DataFrame] = []
    for seed_idx, seed in enumerate(seeds, start=1):
        print(f"[sweep-real] seed {seed_idx}/{len(seeds)} = {seed}", flush=True)
        seeded_specs = None
        if model_specs is not None:
            seeded_specs = [
                ModelSpec(
                    spec.name,
                    spec.kind,
                    kmer_k=spec.kmer_k,
                    deep_epochs=spec.deep_epochs,
                    deep_batch_size=spec.deep_batch_size,
                    deep_lr=spec.deep_lr,
                    deep_device=spec.deep_device,
                    axis_dropout=spec.axis_dropout,
                    counterfactual_mode=spec.counterfactual_mode,
                    counterfactual_weight=spec.counterfactual_weight,
                    metadata_residual_weight=spec.metadata_residual_weight,
                    adversarial_weight=spec.adversarial_weight,
                    deep_objective=spec.deep_objective,
                    group_key=spec.group_key,
                    groupdro_eta=spec.groupdro_eta,
                    protocol_penalty=spec.protocol_penalty,
                    protocol_penalty_weight=spec.protocol_penalty_weight,
                    rc_augment=spec.rc_augment,
                    rc_ensemble=spec.rc_ensemble,
                    embedding_cache=spec.embedding_cache,
                    embedding_head=spec.embedding_head,
                    embedding_include_metadata=spec.embedding_include_metadata,
                    random_state=seed,
                )
                for spec in model_specs
            ]
        artifacts = evaluate_frame_full(
            frame,
            split_specs=DEFAULT_REAL_SPLITS if split_specs is None else split_specs,
            model_specs=seeded_specs,
            random_state=seed,
            bootstrap_iterations=bootstrap_iterations,
            ci_confidence=ci_confidence,
            calibration_method=calibration_method,
            calibration_group=calibration_group,
        )
        for table in (
            artifacts.results,
            artifacts.selective,
            artifacts.split_counts,
            artifacts.bootstrap_cis,
            artifacts.calibration,
        ):
            table.insert(0, "seed", seed)
        result_frames.append(artifacts.results)
        selective_frames.append(artifacts.selective)
        split_count_frames.append(artifacts.split_counts)
        calibration_frames.append(artifacts.calibration)
        if not artifacts.bootstrap_cis.empty:
            ci_frames.append(artifacts.bootstrap_cis)
        partial_results = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
        partial_selective = pd.concat(selective_frames, ignore_index=True) if selective_frames else pd.DataFrame()
        partial_split_counts = pd.concat(split_count_frames, ignore_index=True) if split_count_frames else pd.DataFrame()
        partial_bootstrap_cis = pd.concat(ci_frames, ignore_index=True) if ci_frames else pd.DataFrame()
        partial_calibration = pd.concat(calibration_frames, ignore_index=True) if calibration_frames else pd.DataFrame()
        partial_pairwise = pairwise_metric_deltas(partial_results, PAIRWISE_RESULT_METRICS)
        partial_selective_pairwise = pairwise_metric_deltas(partial_selective, PAIRWISE_SELECTIVE_METRICS)
        partial_results.to_csv(paths["seed_results"], index=False)
        partial_selective.to_csv(paths["seed_selective"], index=False)
        partial_split_counts.to_csv(paths["seed_split_counts"], index=False)
        partial_bootstrap_cis.to_csv(paths["seed_bootstrap_cis"], index=False)
        partial_calibration.to_csv(paths["seed_calibration_report"], index=False)
        partial_pairwise.to_csv(paths["seed_pairwise_deltas"], index=False)
        summarize_pairwise_deltas(partial_pairwise).to_csv(paths["seed_pairwise_delta_summary"], index=False)
        partial_selective_pairwise.to_csv(paths["seed_selective_pairwise_deltas"], index=False)
        summarize_pairwise_deltas(partial_selective_pairwise).to_csv(
            paths["seed_selective_pairwise_delta_summary"],
            index=False,
        )
        summarize_seed_results(
            partial_results,
            ("auprc", "auroc", "ece", "brier", "worst_group_auprc"),
        ).to_csv(paths["seed_result_summary"], index=False)
        summarize_seed_results(partial_selective, ("auprc", "auroc", "ece", "brier")).to_csv(
            paths["seed_selective_summary"],
            index=False,
        )

    results = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    selective = pd.concat(selective_frames, ignore_index=True) if selective_frames else pd.DataFrame()
    split_counts = pd.concat(split_count_frames, ignore_index=True) if split_count_frames else pd.DataFrame()
    bootstrap_cis = pd.concat(ci_frames, ignore_index=True) if ci_frames else pd.DataFrame()
    calibration = pd.concat(calibration_frames, ignore_index=True) if calibration_frames else pd.DataFrame()
    result_summary = summarize_seed_results(
        results,
        ("auprc", "auroc", "ece", "brier", "worst_group_auprc"),
    )
    selective_summary = summarize_seed_results(selective, ("auprc", "auroc", "ece", "brier"))
    pairwise = pairwise_metric_deltas(results, PAIRWISE_RESULT_METRICS)
    pairwise_summary = summarize_pairwise_deltas(pairwise)
    selective_pairwise = pairwise_metric_deltas(selective, PAIRWISE_SELECTIVE_METRICS)
    selective_pairwise_summary = summarize_pairwise_deltas(selective_pairwise)

    results.to_csv(paths["seed_results"], index=False)
    selective.to_csv(paths["seed_selective"], index=False)
    split_counts.to_csv(paths["seed_split_counts"], index=False)
    bootstrap_cis.to_csv(paths["seed_bootstrap_cis"], index=False)
    calibration.to_csv(paths["seed_calibration_report"], index=False)
    pairwise.to_csv(paths["seed_pairwise_deltas"], index=False)
    pairwise_summary.to_csv(paths["seed_pairwise_delta_summary"], index=False)
    selective_pairwise.to_csv(paths["seed_selective_pairwise_deltas"], index=False)
    selective_pairwise_summary.to_csv(paths["seed_selective_pairwise_delta_summary"], index=False)
    result_summary.to_csv(paths["seed_result_summary"], index=False)
    selective_summary.to_csv(paths["seed_selective_summary"], index=False)
    return paths
