# Execution Log

## Completed In This Sprint

- Created a runnable `assayshift_tf` Python package with CLI commands for demo evaluation, manifest validation, peak downloading, and FASTA-backed window building.
- Added deterministic protocol-shift demo data, IID/group-held-out splits, k-mer/metadata/GC baselines, Platt calibration, ECE/Brier/worst-group metrics, and selective-prediction curves.
- Generated preliminary demo outputs in `reports/` and `figures/demo_shift_results.png`.
- Drafted paper-facing artifacts: `docs/mlcb_abstract_draft.md`, `docs/paper_outline.md`, and `figures/benchmark_schematic.mmd`.
- Curated and validated `data/manifests/candidate_experiments.csv` with 14 public rows: 11 ENCODE TF ChIP-seq rows plus 3 GEO CUT&RUN rows.
- Downloaded all 14 candidate peak/fragment files into `data/raw/candidate_peaks/`.
- Wrote `reports/peak_inventory.csv` with interval counts and parse status.
- Added pytest coverage for metrics, split behavior, and BED parsing.
- Added `data/manifests/assay_mvp_hg19.csv` and `data/manifests/encode_k562_grch38.csv`.
- Downloaded hg19 from UCSC into `data/references/hg19.fa` and reused local `C:\Users\Aadi Nair\DNA-Diffusion\data\hg38.fa`.
- Called deterministic CUT&RUN peaks from GSE84474 fragments using the GSE84474 no-antibody control:
  - CTCF `GSM2433139`: 35,595 called peaks.
  - MAX `GSM2433145`: 19,877 called peaks.
  - MYC `GSM2433146`: 9,053 called peaks.
- Built real window tables:
  - `data/processed/assay_mvp_hg19_windows.parquet`: 6,000 windows, 3 TFs, ChIP vs called CUT&RUN, GC-matched negatives.
  - `data/processed/encode_k562_grch38_windows.parquet`: 8,000 windows, 8 ENCODE K562 TF ChIP-seq rows, GC+accessibility-matched negatives.
- Generated real result reports:
  - `reports/real_assay_mvp_hg19_preliminary_results.md`
  - `reports/real_encode_k562_grch38_eval_preliminary_results.md`
- Added PICARD-TF / `axis_guard_cnn`, a PyTorch protocol-factorized CNN with counterfactual protocol masking, metadata residual penalty, optional assay/lab adversary, and sklearn-style `fit`/`predict_proba` integration.
- Added `tiny_cnn` as the direct sequence-only deep baseline for the AxisGuard ablation.
- Extended `evaluate-real` with explicit `--model`, deep training, axis-dropout, counterfactual-weight, and adversarial-weight flags.
- Generated AxisGuard submission reports:
  - `reports/real_assay_mvp_hg19_axisguard_preliminary_results.md`
  - `reports/real_encode_k562_grch38_axisguard_preliminary_results.md`
- Added paper figures:
  - `figures/picard_tf_schematic.png`
  - `figures/real_assay_mvp_hg19_axisguard_reliability.png`
  - `figures/real_encode_k562_grch38_axisguard_reliability.png`
- Drafted an 8-page submission narrative in `docs/mlcb_8page_draft.md`.

## Evidence Available Now

- Primary real result: in the hg19 3-TF assay pilot, k-mer-plus-metadata AUPRC drops from 0.765 IID to 0.715 under held-out CUT&RUN, while uncalibrated ECE worsens from 0.159 to 0.244.
- Calibration result: Platt scaling reduces held-out CUT&RUN ECE to 0.054, and selective prediction raises retained-subset AUPRC to 0.824 at 20% coverage.
- Secondary real result: in the GRCh38 ENCODE-only benchmark, k-mer baselines remain fairly stable under lab and zinc-finger family holdout, while GC-only controls are weaker under lab holdout.
- Model result: in the ENCODE GRCh38 lab-held-out split, AxisGuard-CNN improves over the plain CNN from AUPRC 0.702 to 0.731, calibrated ECE 0.092 to 0.032, and calibrated Brier score 0.209 to 0.196.
- Nuanced assay result: in the hg19 held-out CUT&RUN split, the plain CNN remains stronger at full coverage, while AxisGuard-CNN slightly improves the 20% calibrated selective subset, AUPRC 0.959 versus 0.952.

## Remaining Before A Submission Claim

- Add mappability/blacklist filtering to the hg19 assay pilot.
- Add an accessibility-matched negative control for hg19 if a suitable K562 hg19 accessibility track is staged.
- Run a sensitivity analysis over CUT&RUN peak-calling thresholds.
- Add TF protein or motif priors if pursuing a stronger TF-family shift result.
- Add repeated seeds for `tiny_cnn` and `axis_guard_cnn` before making a final full-paper claim.
- Keep abstract claims framed as a preliminary benchmark stress test, not causal protocol disentanglement.
