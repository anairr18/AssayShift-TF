from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from assayshift_tf.metrics import PlattCalibrator, binary_metrics, group_metrics, selective_metrics, worst_group_auprc
from assayshift_tf.models import ModelSpec, build_model, model_uses_metadata
from assayshift_tf.splits import make_split


@dataclass(frozen=True)
class RealSplitSpec:
    name: str
    split_type: str
    holdout: object | None = None


DEFAULT_REAL_MODELS = [
    ModelSpec("gc_artifact_logreg", "gc"),
    ModelSpec("kmer_logreg", "kmer"),
    ModelSpec("kmer_metadata_logreg", "kmer_metadata"),
]


def load_window_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    required = {"example_id", "sequence", "label", "assay", "lab", "species", "tf", "gc"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"window table is missing required columns: {', '.join(missing)}")
    if "tf_family" not in frame.columns:
        frame["tf_family"] = ""
    return frame


def default_real_splits(frame: pd.DataFrame) -> list[RealSplitSpec]:
    splits = [RealSplitSpec("iid", "iid")]
    if "CUT&RUN" in set(frame["assay"]):
        splits.append(RealSplitSpec("assay_heldout_cutrun", "assay", "CUT&RUN"))
    lab_counts = frame["lab"].value_counts()
    if lab_counts.shape[0] > 1:
        splits.append(RealSplitSpec(f"lab_heldout_{_slug(lab_counts.index[-1])}", "lab", lab_counts.index[-1]))
    families = [x for x in frame.get("tf_family", pd.Series(dtype=str)).dropna().unique() if x]
    if len(families) > 1:
        holdout = pd.Series(families).sort_values().iloc[-1]
        splits.append(RealSplitSpec(f"family_heldout_{_slug(holdout)}", "tf_family", holdout))
    return splits


def _slug(value: object) -> str:
    text = str(value).lower()
    out = []
    for ch in text:
        out.append(ch if ch.isalnum() else "_")
    return "_".join("".join(out).split("_"))[:48]


def _mask_leaky_metadata(frame: pd.DataFrame, split_spec: RealSplitSpec) -> pd.DataFrame:
    out = frame.copy()
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


def _predict_positive(model, frame: pd.DataFrame) -> pd.Series:
    return pd.Series(model.predict_proba(frame)[:, 1], index=frame.index)


def _bootstrap_metrics(
    pred_frame: pd.DataFrame,
    n_bootstrap: int,
    random_state: int,
) -> dict[str, float]:
    metrics = ["auprc", "auroc", "ece", "brier"]
    rows: list[dict[str, float]] = []
    rng = np.random.default_rng(random_state)
    if n_bootstrap <= 0 or pred_frame.empty:
        return {f"{metric}_{suffix}": np.nan for metric in metrics for suffix in ["lo", "hi"]}

    groups = pred_frame["chrom"].astype(str) if "chrom" in pred_frame.columns else pred_frame["example_id"].astype(str)
    unique_groups = groups.drop_duplicates().to_numpy()
    group_to_idx = {group: np.flatnonzero(groups.to_numpy() == group) for group in unique_groups}
    for _ in range(n_bootstrap):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([group_to_idx[group] for group in sampled_groups])
        sample = pred_frame.iloc[idx]
        rows.append(binary_metrics(sample["label"], sample["prob"]))

    boot = pd.DataFrame(rows)
    out: dict[str, float] = {}
    for metric in metrics:
        out[f"{metric}_lo"] = float(boot[metric].quantile(0.025))
        out[f"{metric}_hi"] = float(boot[metric].quantile(0.975))
    return out


def evaluate_real_frame(
    frame: pd.DataFrame,
    split_specs: list[RealSplitSpec] | None = None,
    model_specs: list[ModelSpec] | None = None,
    random_state: int = 13,
    n_bootstrap: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_specs = default_real_splits(frame) if split_specs is None else split_specs
    model_specs = DEFAULT_REAL_MODELS if model_specs is None else model_specs

    result_rows: list[dict[str, object]] = []
    group_rows: list[pd.DataFrame] = []
    selective_rows: list[pd.DataFrame] = []
    prediction_rows: list[pd.DataFrame] = []
    ci_rows: list[dict[str, object]] = []
    split_count_rows: list[dict[str, object]] = []

    for split_spec in split_specs:
        split = make_split(frame, split_spec.split_type, holdout=split_spec.holdout, random_state=random_state)
        split_counts = (
            frame.assign(split=split)
            .groupby(["split", "assay", "tf", "label"], dropna=False)
            .size()
            .reset_index(name="n")
        )
        split_counts.insert(0, "split_name", split_spec.name)
        split_count_rows.append(split_counts)

        train_idx = split.eq("train")
        valid_idx = split.eq("valid")
        test_idx = split.eq("test")
        train_original = frame.loc[train_idx].copy()
        valid_original = frame.loc[valid_idx].copy()
        test_original = frame.loc[test_idx].copy()

        for model_spec in model_specs:
            if model_uses_metadata(model_spec):
                model_frame = _mask_leaky_metadata(frame, split_spec)
                train = model_frame.loc[train_idx].copy()
                valid = model_frame.loc[valid_idx].copy()
                test = model_frame.loc[test_idx].copy()
            else:
                train = train_original
                valid = valid_original
                test = test_original

            model = build_model(model_spec, train)
            model.fit(train, train["label"])
            valid_prob = _predict_positive(model, valid)
            test_prob = _predict_positive(model, test)
            calibrator = PlattCalibrator().fit(valid_prob, valid["label"])
            test_prob_calibrated = calibrator.predict(test_prob)

            for calibrated, prob in [(False, test_prob), (True, test_prob_calibrated)]:
                pred_frame = test_original.copy()
                pred_frame["prob"] = prob
                pred_frame["split_name"] = split_spec.name
                pred_frame["model"] = model_spec.name
                pred_frame["calibrated"] = calibrated
                keep_cols = [
                    "example_id",
                    "label",
                    "prob",
                    "split_name",
                    "model",
                    "calibrated",
                    "assay",
                    "lab",
                    "species",
                    "tf",
                    "tf_family",
                    "gc",
                    "chrom",
                    "negative_strategy",
                ]
                prediction_rows.append(pred_frame[[col for col in keep_cols if col in pred_frame.columns]])

                metrics = binary_metrics(pred_frame["label"], pred_frame["prob"])
                gm = group_metrics(pred_frame, group_cols=("assay", "lab", "species", "tf", "tf_family", "negative_strategy"))
                row = {
                    "split": split_spec.name,
                    "split_type": split_spec.split_type,
                    "holdout": split_spec.holdout if split_spec.holdout is not None else "",
                    "model": model_spec.name,
                    "calibrated": calibrated,
                    **metrics,
                    "worst_group_auprc": worst_group_auprc(gm),
                }
                result_rows.append(row)
                ci_rows.append(
                    {
                        "split": split_spec.name,
                        "model": model_spec.name,
                        "calibrated": calibrated,
                        **_bootstrap_metrics(pred_frame, n_bootstrap, random_state + len(ci_rows)),
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

    return (
        pd.DataFrame(result_rows),
        pd.concat(group_rows, ignore_index=True) if group_rows else pd.DataFrame(),
        pd.concat(selective_rows, ignore_index=True) if selective_rows else pd.DataFrame(),
        pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame(),
        pd.DataFrame(ci_rows),
        pd.concat(split_count_rows, ignore_index=True) if split_count_rows else pd.DataFrame(),
    )


def run_real(
    windows_path: str | Path,
    out_prefix: str | Path,
    figures_dir: str | Path,
    random_state: int = 13,
    n_bootstrap: int = 100,
) -> dict[str, Path]:
    frame = load_window_table(windows_path)
    results, groups, selective, predictions, ci, split_counts = evaluate_real_frame(
        frame, random_state=random_state, n_bootstrap=n_bootstrap
    )

    prefix = Path(out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    figures = Path(figures_dir)
    figures.mkdir(parents=True, exist_ok=True)
    paths = {
        "results": prefix.with_name(prefix.name + "_results.csv"),
        "groups": prefix.with_name(prefix.name + "_group_metrics.csv"),
        "selective": prefix.with_name(prefix.name + "_selective.csv"),
        "predictions": prefix.with_name(prefix.name + "_predictions.csv"),
        "bootstrap_ci": prefix.with_name(prefix.name + "_bootstrap_ci.csv"),
        "split_counts": prefix.with_name(prefix.name + "_split_counts.csv"),
        "summary": prefix.with_name(prefix.name + "_summary.md"),
        "figure": figures / (prefix.name + "_shift_results.png"),
    }
    results.to_csv(paths["results"], index=False)
    groups.to_csv(paths["groups"], index=False)
    selective.to_csv(paths["selective"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    ci.to_csv(paths["bootstrap_ci"], index=False)
    split_counts.to_csv(paths["split_counts"], index=False)

    from assayshift_tf.figures import plot_real_results
    from assayshift_tf.report import write_real_summary

    plot_real_results(results, selective, paths["figure"])
    write_real_summary(results, ci, split_counts, paths["summary"])
    return paths
