# Preliminary Real-Data Evaluation

Source table: `data\processed\encode_k562_grch38_windows.parquet`

This report summarizes baseline evaluation on the supplied real-data window table. Metrics are preliminary integration outputs: they verify the evaluation contract and should be interpreted with the data provenance, negative sampling, and peak-processing choices used to build the table.

Intervals are percentile bootstrap confidence intervals over test-set windows.

## Test Metrics

| Split | Model | AUPRC | AUROC | ECE | Brier | Worst-Group AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| iid | gc_artifact_logreg | 0.672 [0.647, 0.708] | 0.713 [0.689, 0.734] | 0.060 [0.048, 0.088] | 0.217 [0.210, 0.224] | 0.551 |
| iid | kmer_logreg | 0.754 [0.725, 0.785] | 0.778 [0.755, 0.801] | 0.119 [0.103, 0.142] | 0.207 [0.192, 0.221] | 0.707 |
| iid | kmer_metadata_logreg | 0.753 [0.723, 0.791] | 0.778 [0.760, 0.800] | 0.122 [0.106, 0.140] | 0.208 [0.195, 0.219] | 0.698 |
| lab_heldout | gc_artifact_logreg | 0.588 [0.558, 0.615] | 0.666 [0.642, 0.684] | 0.115 [0.097, 0.137] | 0.237 [0.232, 0.243] | 0.588 |
| lab_heldout | kmer_logreg | 0.742 [0.716, 0.766] | 0.755 [0.734, 0.774] | 0.129 [0.112, 0.148] | 0.222 [0.210, 0.234] | 0.713 |
| lab_heldout | kmer_metadata_logreg | 0.743 [0.714, 0.775] | 0.755 [0.736, 0.773] | 0.135 [0.122, 0.159] | 0.224 [0.213, 0.235] | 0.714 |
| family_heldout_zinc_finger | gc_artifact_logreg | 0.722 [0.693, 0.749] | 0.746 [0.723, 0.766] | 0.048 [0.039, 0.069] | 0.207 [0.200, 0.214] | 0.722 |
| family_heldout_zinc_finger | kmer_logreg | 0.732 [0.706, 0.759] | 0.735 [0.714, 0.754] | 0.151 [0.135, 0.175] | 0.234 [0.222, 0.247] | 0.732 |
| family_heldout_zinc_finger | kmer_metadata_logreg | 0.733 [0.704, 0.767] | 0.736 [0.712, 0.759] | 0.151 [0.135, 0.177] | 0.235 [0.221, 0.250] | 0.733 |

## Split Counts

| Split | Type | Holdout | Partition | N | Positive | Negative | Prevalence |
|---|---|---|---|---:|---:|---:|---:|
| iid | iid |  | train | 5120 | 2560 | 2560 | 0.500 |
| iid | iid |  | valid | 1280 | 640 | 640 | 0.500 |
| iid | iid |  | test | 1600 | 800 | 800 | 0.500 |
| lab_heldout | lab | Richard Myers, HAIB | train | 4800 | 2400 | 2400 | 0.500 |
| lab_heldout | lab | Richard Myers, HAIB | valid | 1200 | 600 | 600 | 0.500 |
| lab_heldout | lab | Richard Myers, HAIB | test | 2000 | 1000 | 1000 | 0.500 |
| family_heldout_zinc_finger | tf_family | zinc_finger | train | 4800 | 2400 | 2400 | 0.500 |
| family_heldout_zinc_finger | tf_family | zinc_finger | valid | 1200 | 600 | 600 | 0.500 |
| family_heldout_zinc_finger | tf_family | zinc_finger | test | 2000 | 1000 | 1000 | 0.500 |

## Calibrated Selective Prediction

| Split | Model | Coverage | AUPRC | ECE | Brier |
|---|---:|---:|---:|---:|---:|
| iid | kmer_metadata_logreg | 1.000 | 0.753 | 0.040 | 0.193 |
| iid | kmer_metadata_logreg | 0.800 | 0.772 | 0.041 | 0.179 |
| iid | kmer_metadata_logreg | 0.600 | 0.789 | 0.041 | 0.161 |
| iid | kmer_metadata_logreg | 0.400 | 0.809 | 0.041 | 0.144 |
| iid | kmer_metadata_logreg | 0.200 | 0.843 | 0.042 | 0.105 |
| lab_heldout | kmer_metadata_logreg | 1.000 | 0.743 | 0.055 | 0.204 |
| lab_heldout | kmer_metadata_logreg | 0.800 | 0.761 | 0.059 | 0.193 |
| lab_heldout | kmer_metadata_logreg | 0.600 | 0.782 | 0.068 | 0.180 |
| lab_heldout | kmer_metadata_logreg | 0.400 | 0.797 | 0.046 | 0.153 |
| lab_heldout | kmer_metadata_logreg | 0.200 | 0.808 | 0.028 | 0.107 |
| family_heldout_zinc_finger | kmer_metadata_logreg | 1.000 | 0.733 | 0.052 | 0.210 |
| family_heldout_zinc_finger | kmer_metadata_logreg | 0.800 | 0.756 | 0.056 | 0.202 |
| family_heldout_zinc_finger | kmer_metadata_logreg | 0.600 | 0.776 | 0.063 | 0.191 |
| family_heldout_zinc_finger | kmer_metadata_logreg | 0.400 | 0.802 | 0.065 | 0.168 |
| family_heldout_zinc_finger | kmer_metadata_logreg | 0.200 | 0.828 | 0.074 | 0.144 |

Interpretation guardrail: these outputs do not by themselves establish a biological protocol-shift claim. They are suitable for checking that real prebuilt windows can flow through the split-aware baselines, calibration, selective prediction, bootstrap uncertainty, and reporting stack.
