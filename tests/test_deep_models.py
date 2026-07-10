import numpy as np
import pandas as pd
import pytest

from assayshift_tf.benchmark import SplitSpec, run_real_data_evaluation
from assayshift_tf.deep import DeepModelConfig, TorchProbClassifier, one_hot_encode
from assayshift_tf.features import add_sequence_stats
from assayshift_tf.models import ModelSpec, model_spec_from_name


torch = pytest.importorskip("torch")


def _deep_frame(n: int = 48) -> pd.DataFrame:
    rows = []
    for i in range(n):
        label = int(i % 2 == 0)
        motif = "CACGTG" if label else "TTAATT"
        sequence = ("ACGT" * 8) + motif + ("TGCA" * 8)
        rows.append(
            {
                "example_id": f"x{i}",
                "sequence": sequence,
                "label": label,
                "assay": "CUT&RUN" if i % 4 == 0 else "TF ChIP-seq",
                "lab": "lab_b" if i % 5 == 0 else "lab_a",
                "species": "Homo sapiens",
                "tf": "MAX" if i % 3 else "CTCF",
                "tf_family": "bhlh_zip" if i % 3 else "zinc_finger",
                "biosample": "K562",
                "cell_type": "K562",
                "assembly": "hg19",
                "chrom": "chr1",
                "start": i * 100,
                "end": i * 100 + len(sequence),
                "source_width": 211 if label else np.nan,
                "source_score": 10.0 if label else np.nan,
            }
        )
    return add_sequence_stats(pd.DataFrame(rows))


def _fast_config(counterfactual_mode: str = "mask") -> DeepModelConfig:
    return DeepModelConfig(
        epochs=1,
        batch_size=12,
        conv_channels=8,
        embedding_dim=4,
        hidden_dim=8,
        counterfactual_mode=counterfactual_mode,
        counterfactual_weight=0.1,
        metadata_residual_weight=0.01,
        axis_dropout=0.1,
        random_state=5,
    )


def test_one_hot_encode_ambiguous_bases_are_zero():
    encoded = one_hot_encode(["ACGTN"], max_len=5)
    assert encoded.shape == (1, 4, 5)
    assert encoded[:, :, :4].sum() == 4
    assert encoded[:, :, 4].sum() == 0


@pytest.mark.parametrize("model_kind", ["tiny_cnn", "axis_guard_cnn"])
def test_torch_prob_classifier_contract(model_kind):
    frame = _deep_frame()
    model = TorchProbClassifier(model_kind, _fast_config()).fit(frame, frame["label"])
    prob = model.predict_proba(frame.head(7))
    assert prob.shape == (7, 2)
    assert np.allclose(prob.sum(axis=1), 1.0)
    assert np.all((prob >= 0.0) & (prob <= 1.0))


def test_axis_guard_counterfactual_masks_protocol_not_biology():
    frame = _deep_frame()
    model = TorchProbClassifier("axis_guard_cnn", _fast_config()).fit(frame, frame["label"])
    batch = frame.head(6)
    numeric = torch.tensor(model._transform_numeric(batch), device=model.device_)
    categorical = torch.tensor(model._transform_categorical(batch), dtype=torch.long, device=model.device_)
    cf_numeric, cf_categorical = model._counterfactual_metadata(numeric, categorical, torch)

    assert torch.equal(cf_numeric, torch.zeros_like(numeric))
    tf_idx = model.categorical_columns_.index("tf")
    family_idx = model.categorical_columns_.index("tf_family")
    assert torch.equal(cf_categorical[:, tf_idx], categorical[:, tf_idx])
    assert torch.equal(cf_categorical[:, family_idx], categorical[:, family_idx])
    for column in ("assay", "lab", "species", "biosample", "cell_type", "assembly"):
        idx = model.categorical_columns_.index(column)
        assert torch.equal(cf_categorical[:, idx], torch.zeros_like(cf_categorical[:, idx]))


def test_axis_guard_counterfactual_can_shuffle_protocol_not_biology():
    frame = _deep_frame()
    model = TorchProbClassifier("axis_guard_cnn", _fast_config(counterfactual_mode="shuffle")).fit(frame, frame["label"])
    batch = frame.head(12)
    numeric = torch.tensor(model._transform_numeric(batch), device=model.device_)
    categorical = torch.tensor(model._transform_categorical(batch), dtype=torch.long, device=model.device_)
    torch.manual_seed(0)
    cf_numeric, cf_categorical = model._counterfactual_metadata(numeric, categorical, torch)

    tf_idx = model.categorical_columns_.index("tf")
    family_idx = model.categorical_columns_.index("tf_family")
    assert torch.equal(cf_categorical[:, tf_idx], categorical[:, tf_idx])
    assert torch.equal(cf_categorical[:, family_idx], categorical[:, family_idx])

    assay_idx = model.categorical_columns_.index("assay")
    lab_idx = model.categorical_columns_.index("lab")
    assert not torch.equal(cf_categorical[:, assay_idx], torch.zeros_like(cf_categorical[:, assay_idx]))
    assert not torch.equal(cf_categorical[:, lab_idx], torch.zeros_like(cf_categorical[:, lab_idx]))
    assert torch.equal(torch.sort(cf_categorical[:, assay_idx]).values, torch.sort(categorical[:, assay_idx]).values)
    assert torch.equal(torch.sort(cf_categorical[:, lab_idx]).values, torch.sort(categorical[:, lab_idx]).values)
    assert (
        not torch.equal(cf_categorical[:, assay_idx], categorical[:, assay_idx])
        or not torch.equal(cf_categorical[:, lab_idx], categorical[:, lab_idx])
        or not torch.equal(cf_numeric, numeric)
    )


def test_axis_guard_ignores_leakage_prone_columns():
    frame = _deep_frame()
    model = TorchProbClassifier("axis_guard_cnn", _fast_config()).fit(frame, frame["label"])
    assert "sequence" in model.feature_columns_
    for column in ("label", "example_id", "chrom", "start", "end", "source_width", "source_score"):
        assert column not in model.feature_columns_
    assert {"label", "example_id", "chrom", "start", "end", "source_width", "source_score"} <= set(
        model.leakage_excluded_columns_
    )


def test_real_evaluation_runs_deep_models_end_to_end(tmp_path):
    table = tmp_path / "deep_windows.csv"
    _deep_frame(40).to_csv(table, index=False)
    paths = run_real_data_evaluation(
        table,
        out_dir=tmp_path / "reports",
        figures_dir=tmp_path / "figures",
        prefix="deep_real",
        split_specs=[SplitSpec("tiny_iid", "iid")],
        model_specs=[
            ModelSpec("tiny_cnn", "tiny_cnn", deep_epochs=1, deep_batch_size=12, random_state=3),
            ModelSpec(
                "axis_guard_cnn",
                "axis_guard_cnn",
                deep_epochs=1,
                deep_batch_size=12,
                axis_dropout=0.1,
                counterfactual_weight=0.1,
                random_state=3,
            ),
        ],
        random_seed=3,
        bootstrap_iterations=0,
    )

    results = pd.read_csv(paths["results"])
    assert set(results["model"]) == {"tiny_cnn", "axis_guard_cnn"}
    assert {"auprc", "auroc", "ece", "brier"} <= set(results.columns)
    predictions = pd.read_csv(paths["predictions"])
    assert predictions["prob"].between(0, 1).all()
    assert paths["figure"].stat().st_size > 0


def test_axis_guard_ablation_aliases_have_expected_weights():
    full = model_spec_from_name("axis_guard_full")
    shuffled = model_spec_from_name("axis_guard_cnn", counterfactual_mode="shuffle")
    no_cf = model_spec_from_name("axis_guard_no_cf")
    no_resid = model_spec_from_name("axis_guard_no_resid")
    no_adv = model_spec_from_name("axis_guard_no_adv")

    assert full.name == "axis_guard_full"
    assert full.counterfactual_weight > 0
    assert full.metadata_residual_weight > 0
    assert shuffled.counterfactual_mode == "shuffle"
    assert full.adversarial_weight > 0
    assert no_cf.counterfactual_weight == 0.0
    assert no_resid.metadata_residual_weight == 0.0
    assert no_adv.adversarial_weight == 0.0

