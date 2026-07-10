import pandas as pd

from assayshift_tf.benchmark import SplitSpec, filter_window_frame, run_real_data_evaluation, run_real_seed_sweep
from assayshift_tf.features import add_sequence_stats
from assayshift_tf.models import ModelSpec


def _tiny_window_table() -> pd.DataFrame:
    rows = []
    assays = [
        ("TF ChIP-seq", "Snyder", "Homo sapiens"),
        ("CUT&RUN", "Henikoff", "Mus musculus"),
    ]
    for assay, lab, species in assays:
        for i in range(24):
            label = i % 2
            if label:
                sequence = ("ACGT" * 18) + "CACGTG" + ("CGTA" * 18)
            else:
                sequence = ("ATTA" * 18) + "TTAATT" + ("TAAT" * 18)
            rows.append(
                {
                    "example_id": f"{assay.replace(' ', '_')}_{i:03d}",
                    "dataset_id": f"{assay.replace(' ', '_')}_{lab}",
                    "sequence": sequence,
                    "label": label,
                    "assay": assay,
                    "lab": lab,
                    "species": species,
                    "tf": "MAX",
                    "tf_family": "bhlh",
                    "assembly": "GRCh38" if species == "Homo sapiens" else "mm10",
                    "chrom": "chr1",
                    "start": i * 100,
                    "end": i * 100 + len(sequence),
                }
            )
    return add_sequence_stats(pd.DataFrame(rows))


def test_real_data_evaluation_outputs_expected_schemas(tmp_path):
    table = tmp_path / "windows.csv"
    _tiny_window_table().to_csv(table, index=False)

    paths = run_real_data_evaluation(
        table,
        out_dir=tmp_path / "reports",
        figures_dir=tmp_path / "figures",
        prefix="tiny_real",
        split_specs=[
            SplitSpec("tiny_iid", "iid"),
            SplitSpec("assay_cutrun", "assay", "CUT&RUN"),
        ],
        random_seed=7,
        bootstrap_iterations=12,
    )

    for path in paths.values():
        assert path.exists()

    results = pd.read_csv(paths["results"])
    assert {
        "split",
        "split_type",
        "holdout",
        "model",
        "calibrated",
        "n",
        "prevalence",
        "brier",
        "ece",
        "auroc",
        "auprc",
        "worst_group_auprc",
    } <= set(results.columns)
    assert set(results["split"]) == {"tiny_iid", "assay_cutrun"}

    groups = pd.read_csv(paths["groups"])
    assert {"split", "model", "calibrated", "group_col", "group", "auprc", "auroc", "ece", "brier"} <= set(
        groups.columns
    )

    selective = pd.read_csv(paths["selective"])
    assert {"split", "model", "calibrated", "coverage", "n", "auprc", "auroc", "ece", "brier"} <= set(
        selective.columns
    )

    predictions = pd.read_csv(paths["predictions"])
    assert {"example_id", "label", "prob", "split_name", "model", "calibrated", "assay", "gc"} <= set(
        predictions.columns
    )
    assert predictions["prob"].between(0, 1).all()

    cis = pd.read_csv(paths["bootstrap_cis"])
    assert {
        "split",
        "split_type",
        "holdout",
        "model",
        "calibrated",
        "metric",
        "estimate",
        "ci_low",
        "ci_high",
        "n_bootstrap",
        "valid_bootstraps",
        "confidence",
    } <= set(cis.columns)
    assert set(cis["metric"]) == {"auprc", "auroc", "ece", "brier"}

    pairwise = pd.read_csv(paths["pairwise_deltas"])
    assert {
        "split",
        "split_type",
        "holdout",
        "model_a",
        "model_b",
        "metric",
        "direction",
        "model_a_value",
        "model_b_value",
        "delta",
        "delta_favors_model_a",
        "better_model",
    } <= set(pairwise.columns)
    assert not pairwise.empty
    first = pairwise.iloc[0]
    assert first["model_a"] != first["model_b"]
    assert first["delta"] == first["model_a_value"] - first["model_b_value"]

    pairwise_cis = pd.read_csv(paths["pairwise_delta_cis"])
    assert {
        "split",
        "split_type",
        "holdout",
        "model_a",
        "model_b",
        "metric",
        "delta",
        "ci_low",
        "ci_high",
        "valid_bootstraps",
    } <= set(pairwise_cis.columns)
    assert set(pairwise_cis["metric"]) == {"auprc", "auroc", "ece", "brier"}

    selective_pairwise = pd.read_csv(paths["selective_pairwise_deltas"])
    assert {"split", "coverage", "model_a", "model_b", "metric", "delta"} <= set(selective_pairwise.columns)
    assert not selective_pairwise.empty

    calibration = pd.read_csv(paths["calibration_report"])
    assert {
        "calibration_method",
        "split",
        "model",
        "calibration_group_col",
        "calibration_group",
        "fit_n",
        "test_n",
        "used_group_specific",
    } <= set(calibration.columns)
    assert not calibration.empty

    split_counts = pd.read_csv(paths["split_counts"])
    assert {"split", "split_type", "holdout", "partition", "n", "positive", "negative", "prevalence"} <= set(
        split_counts.columns
    )
    assert set(split_counts["partition"]) == {"train", "valid", "test"}

    summary = paths["summary"].read_text(encoding="utf-8")
    assert "Preliminary Real-Data Evaluation" in summary
    assert "deterministic stress-test" not in summary
    assert paths["figure"].stat().st_size > 0
    assert paths["window_filter_report"].exists()


def test_filter_window_frame_drops_all_n_and_duplicate_sequences():
    frame = pd.DataFrame(
        {
            "example_id": ["a", "b", "c", "d"],
            "sequence": ["ACGT", "NNNN", "ACGT", "TGCA"],
            "label": [1, 0, 1, 0],
        }
    )
    filtered = filter_window_frame(
        add_sequence_stats(frame),
        drop_all_n=True,
        drop_duplicate_sequences=True,
    )
    assert filtered.frame["sequence"].tolist() == ["ACGT", "TGCA"]
    assert set(filtered.report["step"]) == {"input", "drop_all_n_sequences", "drop_duplicate_sequences"}
    assert int(filtered.report.loc[filtered.report["step"].eq("drop_all_n_sequences"), "removed_n"].iloc[0]) == 1
    assert int(filtered.report.loc[filtered.report["step"].eq("drop_duplicate_sequences"), "removed_n"].iloc[0]) == 1


def test_seed_sweep_writes_aggregate_outputs(tmp_path):
    table = tmp_path / "windows.csv"
    _tiny_window_table().to_csv(table, index=False)

    paths = run_real_seed_sweep(
        table,
        out_dir=tmp_path / "reports",
        prefix="tiny_sweep",
        split_specs=[SplitSpec("tiny_iid", "iid")],
        model_specs=[ModelSpec("gc_artifact_logreg", "gc")],
        seeds=[3, 5],
        bootstrap_iterations=0,
        drop_all_n=False,
        drop_duplicate_sequences=False,
    )
    for path in paths.values():
        assert path.exists()
    results = pd.read_csv(paths["seed_results"])
    assert set(results["seed"]) == {3, 5}
    summary = pd.read_csv(paths["seed_result_summary"])
    assert {"metric", "mean", "std", "n_seeds"} <= set(summary.columns)
    pairwise_summary = pd.read_csv(paths["seed_pairwise_delta_summary"])
    assert {
        "model_a",
        "model_b",
        "metric",
        "mean_delta",
        "std_delta",
        "n_seeds",
        "model_a_win_rate",
    } <= set(pairwise_summary.columns)
    calibration = pd.read_csv(paths["seed_calibration_report"])
    assert {"seed", "calibration_method", "split", "model", "fit_n", "test_n"} <= set(calibration.columns)
