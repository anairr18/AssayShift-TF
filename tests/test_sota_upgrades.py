import numpy as np
import pandas as pd

from assayshift_tf.benchmark import SplitSpec, run_real_data_evaluation
from assayshift_tf.features import add_sequence_stats
from assayshift_tf.models import ModelSpec, build_model, model_spec_from_name


def _upgrade_frame(n: int = 36) -> pd.DataFrame:
    rows = []
    for i in range(n):
        label = int(i % 2 == 0)
        assay = "TF ChIP-seq" if i % 3 else "CUT&RUN"
        rows.append(
            {
                "example_id": f"u{i:03d}",
                "sequence": ("ACGT" * 20) + ("CACGTG" if label else "TTAATT") + ("TGCA" * 20),
                "label": label,
                "assay": assay,
                "lab": "LabA" if i % 4 else "LabB",
                "species": "Homo sapiens",
                "tf": "MAX" if label else "CTCF",
                "tf_family": "bhlh" if label else "zinc_finger",
                "chrom": "chr1",
                "start": i * 10,
                "end": i * 10 + 211,
            }
        )
    return add_sequence_stats(pd.DataFrame(rows))


def test_embedding_head_matches_sklearn_contract(tmp_path):
    frame = _upgrade_frame(20)
    labels = frame["label"].to_numpy(dtype=float)
    embedding = np.column_stack(
        [
            labels,
            1.0 - labels,
            np.linspace(0.0, 1.0, len(frame)),
        ]
    ).astype(np.float32)
    cache = tmp_path / "embeddings.npz"
    np.savez_compressed(cache, example_id=np.asarray(frame["example_id"].astype(str).tolist(), dtype=str), embedding=embedding)

    spec = model_spec_from_name("embedding_head", embedding_cache=cache)
    model = build_model(spec, frame)
    model.fit(frame, frame["label"])
    prob = model.predict_proba(frame)
    assert prob.shape == (len(frame), 2)
    assert np.allclose(prob.sum(axis=1), 1.0)
    assert prob[:, 1].mean() > 0


def test_groupdro_and_alignment_deep_model_smoke():
    frame = _upgrade_frame(24)
    spec = ModelSpec(
        "tiny_groupdro",
        "tiny_cnn",
        deep_epochs=1,
        deep_batch_size=8,
        deep_lr=1e-3,
        deep_device="cpu",
        deep_objective="groupdro",
        group_key="assay",
        groupdro_eta=0.1,
        protocol_penalty="mmd",
        protocol_penalty_weight=0.01,
        rc_augment=True,
        rc_ensemble=True,
        random_state=5,
    )
    model = build_model(spec, frame)
    model.fit(frame, frame["label"])
    prob = model.predict_proba(frame)
    assert prob.shape == (len(frame), 2)
    assert np.isfinite(prob).all()


def test_protocol_platt_evaluation_output(tmp_path):
    table = tmp_path / "windows.csv"
    _upgrade_frame(40).to_csv(table, index=False)
    paths = run_real_data_evaluation(
        table,
        out_dir=tmp_path / "reports",
        figures_dir=tmp_path / "figures",
        prefix="protocol_cal",
        split_specs=[SplitSpec("iid", "iid")],
        model_specs=[ModelSpec("gc_artifact_logreg", "gc")],
        bootstrap_iterations=0,
        calibration_method="protocol_platt",
        calibration_group="assay",
    )
    calibration = pd.read_csv(paths["calibration_report"])
    assert set(calibration["calibration_method"]) == {"protocol_platt"}
    assert "assay" in set(calibration["calibration_group_col"])
    assert calibration["test_n"].sum() >= 1
