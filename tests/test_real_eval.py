import numpy as np
import pandas as pd

from assayshift_tf.models import ModelSpec
from assayshift_tf.real_eval import RealSplitSpec, evaluate_real_frame


def test_real_evaluator_outputs_ci_and_counts():
    rng = np.random.default_rng(1)
    rows = []
    for i in range(80):
        assay = "CUT&RUN" if i >= 60 else "TF ChIP-seq"
        label = int(i % 4 == 0)
        gc = 0.65 if label else 0.35
        seq = ("GCGC" if label else "ATAT") * 30
        rows.append(
            {
                "example_id": f"x{i}",
                "sequence": seq,
                "label": label,
                "assay": assay,
                "lab": "heldout" if assay == "CUT&RUN" else "trainlab",
                "species": "Homo sapiens",
                "tf": "CTCF" if i % 2 else "MAX",
                "tf_family": "zinc_finger" if i % 2 else "bhlh_zip",
                "gc": gc + float(rng.normal(0, 0.01)),
                "n_fraction": 0.0,
                "length": len(seq),
                "chrom": "chr1" if i < 40 else "chr2",
                "negative_strategy": "gc",
            }
        )
    frame = pd.DataFrame(rows)
    results, groups, selective, predictions, ci, split_counts = evaluate_real_frame(
        frame,
        split_specs=[RealSplitSpec("assay_heldout_cutrun", "assay", "CUT&RUN")],
        model_specs=[ModelSpec("gc_artifact_logreg", "gc")],
        n_bootstrap=3,
    )
    assert {"n", "prevalence", "auprc", "auroc", "ece", "brier"} <= set(results.columns)
    assert {"auprc_lo", "auprc_hi", "ece_lo", "ece_hi"} <= set(ci.columns)
    assert not split_counts.empty
    assert not groups.empty
    assert not selective.empty
    assert not predictions.empty
