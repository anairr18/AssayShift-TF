# Real-Data Preliminary Results

These results are generated from public peak-derived sequence windows. They are suitable for the abstract only after the corresponding window table, manifests, and commands are preserved with the submission artifact.

| Split | Model | Test n | Prevalence | AUPRC [95% CI] | AUROC | ECE | Brier | Worst-Group AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| iid | gc_artifact_logreg | 1600 | 0.500 | 0.672 [0.651, 0.704] | 0.713 | 0.060 | 0.217 | 0.513 |
| iid | kmer_logreg | 1600 | 0.500 | 0.754 [0.686, 0.791] | 0.778 | 0.119 | 0.207 | 0.707 |
| iid | kmer_metadata_logreg | 1600 | 0.500 | 0.753 [0.699, 0.783] | 0.778 | 0.122 | 0.208 | 0.698 |
| lab_heldout_richard_myers__haib | gc_artifact_logreg | 2000 | 0.500 | 0.588 [0.558, 0.619] | 0.666 | 0.115 | 0.237 | 0.588 |
| lab_heldout_richard_myers__haib | kmer_logreg | 2000 | 0.500 | 0.742 [0.718, 0.766] | 0.755 | 0.129 | 0.222 | 0.713 |
| lab_heldout_richard_myers__haib | kmer_metadata_logreg | 2000 | 0.500 | 0.743 [0.709, 0.769] | 0.755 | 0.133 | 0.224 | 0.714 |
| family_heldout_zinc_finger | gc_artifact_logreg | 2000 | 0.500 | 0.722 [0.694, 0.739] | 0.746 | 0.048 | 0.207 | 0.597 |
| family_heldout_zinc_finger | kmer_logreg | 2000 | 0.500 | 0.732 [0.693, 0.761] | 0.735 | 0.151 | 0.234 | 0.645 |
| family_heldout_zinc_finger | kmer_metadata_logreg | 2000 | 0.500 | 0.733 [0.703, 0.757] | 0.736 | 0.152 | 0.235 | 0.644 |

## Split Counts Preview

| Split name | Partition | Assay | TF | Label | n |
|---|---|---|---|---:|---:|
| iid | test | TF ChIP-seq | CTCF | 0 | 103 |
| iid | test | TF ChIP-seq | CTCF | 1 | 109 |
| iid | test | TF ChIP-seq | GATA1 | 0 | 97 |
| iid | test | TF ChIP-seq | GATA1 | 1 | 91 |
| iid | test | TF ChIP-seq | GATA2 | 0 | 102 |
| iid | test | TF ChIP-seq | GATA2 | 1 | 112 |
| iid | test | TF ChIP-seq | JUND | 0 | 93 |
| iid | test | TF ChIP-seq | JUND | 1 | 107 |
| iid | test | TF ChIP-seq | MAX | 0 | 108 |
| iid | test | TF ChIP-seq | MAX | 1 | 105 |
| iid | test | TF ChIP-seq | MYC | 0 | 92 |
| iid | test | TF ChIP-seq | MYC | 1 | 90 |
| iid | test | TF ChIP-seq | REST | 0 | 89 |
| iid | test | TF ChIP-seq | REST | 1 | 96 |
| iid | test | TF ChIP-seq | SPI1 | 0 | 116 |
| iid | test | TF ChIP-seq | SPI1 | 1 | 90 |
| iid | train | TF ChIP-seq | CTCF | 0 | 322 |
| iid | train | TF ChIP-seq | CTCF | 1 | 307 |
| iid | train | TF ChIP-seq | GATA1 | 0 | 328 |
| iid | train | TF ChIP-seq | GATA1 | 1 | 338 |
| iid | train | TF ChIP-seq | GATA2 | 0 | 314 |
| iid | train | TF ChIP-seq | GATA2 | 1 | 320 |
| iid | train | TF ChIP-seq | JUND | 0 | 334 |
| iid | train | TF ChIP-seq | JUND | 1 | 302 |
| iid | train | TF ChIP-seq | MAX | 0 | 303 |
| iid | train | TF ChIP-seq | MAX | 1 | 322 |
| iid | train | TF ChIP-seq | MYC | 0 | 319 |
| iid | train | TF ChIP-seq | MYC | 1 | 320 |
| iid | train | TF ChIP-seq | REST | 0 | 340 |
| iid | train | TF ChIP-seq | REST | 1 | 314 |

Caveat: assay-shift claims require called CUT&RUN peaks or otherwise harmonized labels. Do not cite fragment BEDs as final peaks.
