import numpy as np
import pandas as pd

from assayshift_tf.metrics import binary_metrics, expected_calibration_error, selective_metrics
from assayshift_tf.splits import make_split


def test_ece_perfect_probabilities_are_small():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.01, 0.02, 0.98, 0.99])
    assert expected_calibration_error(y, p, n_bins=4) < 0.03


def test_binary_metrics_include_core_fields():
    out = binary_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert {"auroc", "auprc", "ece", "brier", "prevalence", "n"} <= set(out)
    assert out["auroc"] == 1.0


def test_selective_metrics_respects_coverage():
    out = selective_metrics([0, 1, 0, 1], [0.1, 0.9, 0.49, 0.51], coverages=[1.0, 0.5])
    assert list(out["n"]) == [4.0, 2.0]


def test_group_holdout_split():
    frame = pd.DataFrame(
        {
            "label": [0, 1, 0, 1, 0, 1],
            "assay": ["A", "A", "A", "B", "B", "B"],
        }
    )
    split = make_split(frame, "assay", holdout="B", valid_size=0.34, random_state=1)
    assert set(split[frame["assay"].eq("B")]) == {"test"}
    assert "train" in set(split)
    assert "valid" in set(split)
