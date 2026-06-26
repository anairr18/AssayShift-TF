# Preliminary Real-Data Evaluation

Source table: `data\processed\assay_mvp_hg19_windows.parquet`

This report summarizes baseline evaluation on the supplied real-data window table. Metrics are preliminary integration outputs: they verify the evaluation contract and should be interpreted with the data provenance, negative sampling, and peak-processing choices used to build the table.

Intervals are percentile bootstrap confidence intervals over test-set windows.

## Test Metrics

| Split | Model | AUPRC | AUROC | ECE | Brier | Worst-Group AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| iid | gc_artifact_logreg | 0.629 [0.588, 0.653] | 0.569 [0.538, 0.602] | 0.074 [0.056, 0.099] | 0.245 [0.241, 0.248] | 0.601 |
| iid | kmer_logreg | 0.761 [0.730, 0.788] | 0.749 [0.721, 0.767] | 0.171 [0.156, 0.200] | 0.236 [0.222, 0.254] | 0.655 |
| iid | kmer_metadata_logreg | 0.765 [0.742, 0.798] | 0.757 [0.734, 0.786] | 0.159 [0.143, 0.187] | 0.231 [0.213, 0.248] | 0.657 |
| iid | tiny_cnn | 0.831 [0.806, 0.858] | 0.820 [0.799, 0.846] | 0.059 [0.050, 0.083] | 0.174 [0.162, 0.186] | 0.811 |
| iid | axis_guard_cnn | 0.835 [0.812, 0.861] | 0.823 [0.804, 0.847] | 0.096 [0.080, 0.120] | 0.180 [0.165, 0.192] | 0.818 |
| assay_heldout_cutrun | gc_artifact_logreg | 0.619 [0.597, 0.644] | 0.554 [0.532, 0.574] | 0.096 [0.081, 0.115] | 0.248 [0.245, 0.252] | 0.578 |
| assay_heldout_cutrun | kmer_logreg | 0.706 [0.679, 0.730] | 0.706 [0.688, 0.726] | 0.256 [0.239, 0.272] | 0.289 [0.272, 0.301] | 0.688 |
| assay_heldout_cutrun | kmer_metadata_logreg | 0.715 [0.694, 0.736] | 0.713 [0.694, 0.731] | 0.243 [0.229, 0.263] | 0.282 [0.270, 0.298] | 0.694 |
| assay_heldout_cutrun | tiny_cnn | 0.796 [0.775, 0.811] | 0.777 [0.765, 0.792] | 0.061 [0.051, 0.078] | 0.194 [0.188, 0.200] | 0.777 |
| assay_heldout_cutrun | axis_guard_cnn | 0.790 [0.775, 0.810] | 0.765 [0.749, 0.783] | 0.135 [0.121, 0.151] | 0.219 [0.208, 0.229] | 0.777 |

## Split Counts

| Split | Type | Holdout | Partition | N | Positive | Negative | Prevalence |
|---|---|---|---|---:|---:|---:|---:|
| iid | iid |  | train | 3840 | 1920 | 1920 | 0.500 |
| iid | iid |  | valid | 960 | 480 | 480 | 0.500 |
| iid | iid |  | test | 1200 | 600 | 600 | 0.500 |
| assay_heldout_cutrun | assay | CUT&RUN | train | 2400 | 1200 | 1200 | 0.500 |
| assay_heldout_cutrun | assay | CUT&RUN | valid | 600 | 300 | 300 | 0.500 |
| assay_heldout_cutrun | assay | CUT&RUN | test | 3000 | 1500 | 1500 | 0.500 |

## Calibrated Selective Prediction

| Split | Model | Coverage | AUPRC | ECE | Brier |
|---|---:|---:|---:|---:|---:|
| iid | kmer_metadata_logreg | 1.000 | 0.765 | 0.051 | 0.200 |
| iid | kmer_metadata_logreg | 0.800 | 0.791 | 0.058 | 0.189 |
| iid | kmer_metadata_logreg | 0.600 | 0.827 | 0.062 | 0.172 |
| iid | kmer_metadata_logreg | 0.400 | 0.888 | 0.055 | 0.139 |
| iid | kmer_metadata_logreg | 0.200 | 0.939 | 0.022 | 0.073 |
| assay_heldout_cutrun | kmer_metadata_logreg | 1.000 | 0.715 | 0.050 | 0.219 |
| assay_heldout_cutrun | kmer_metadata_logreg | 0.800 | 0.741 | 0.062 | 0.212 |
| assay_heldout_cutrun | kmer_metadata_logreg | 0.600 | 0.766 | 0.063 | 0.199 |
| assay_heldout_cutrun | kmer_metadata_logreg | 0.400 | 0.794 | 0.065 | 0.180 |
| assay_heldout_cutrun | kmer_metadata_logreg | 0.200 | 0.829 | 0.073 | 0.159 |

Interpretation guardrail: these outputs do not by themselves establish a biological protocol-shift claim. They are suitable for checking that real prebuilt windows can flow through the split-aware baselines, calibration, selective prediction, bootstrap uncertainty, and reporting stack.
