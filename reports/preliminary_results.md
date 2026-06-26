# Preliminary Demo Results

These are deterministic stress-test results, not final biological claims. The demo plants a transferable motif signal plus a protocol-specific GC artifact so the analysis can verify the paper's intended failure mode before running on public peak-derived data.

| Split | Model | AUPRC | AUROC | ECE | Brier | Worst-Group AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| iid | gc_artifact_logreg | 0.534 | 0.730 | 0.261 | 0.207 | 0.099 |
| iid | kmer_logreg | 0.510 | 0.757 | 0.181 | 0.195 | 0.160 |
| iid | kmer_metadata_logreg | 0.510 | 0.770 | 0.175 | 0.189 | 0.158 |
| assay_heldout_cutrun | gc_artifact_logreg | 0.105 | 0.010 | 0.721 | 0.604 | 0.091 |
| assay_heldout_cutrun | kmer_logreg | 0.125 | 0.269 | 0.490 | 0.474 | 0.119 |
| assay_heldout_cutrun | kmer_metadata_logreg | 0.125 | 0.269 | 0.477 | 0.471 | 0.125 |
| lab_heldout_henikoff | gc_artifact_logreg | 0.105 | 0.010 | 0.721 | 0.604 | 0.091 |
| lab_heldout_henikoff | kmer_logreg | 0.125 | 0.269 | 0.490 | 0.474 | 0.119 |
| lab_heldout_henikoff | kmer_metadata_logreg | 0.125 | 0.269 | 0.477 | 0.471 | 0.125 |

## Calibrated Selective Prediction: Assay-Held-Out CUT&RUN

| Coverage | AUPRC | ECE | Brier |
|---:|---:|---:|---:|
| 1.000 | 0.125 | 0.283 | 0.258 |
| 0.600 | 0.186 | 0.227 | 0.273 |
| 0.400 | 0.237 | 0.274 | 0.311 |
| 0.200 | 0.363 | 0.386 | 0.398 |

Interpretation: if the public-data run shows the same pattern, the paper can claim that IID ranking metrics hide assay/lab shortcuts, and that calibrated selective prediction improves the retained subset at moderate coverage. The lowest-coverage regime should be paired with explicit OOD scoring before making a stronger reliability claim. Until then, this table is a reproducibility and analysis-contract check.
