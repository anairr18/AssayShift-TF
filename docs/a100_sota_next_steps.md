# A100 SOTA Upgrade Plan

This document is the repo-grounded version of the adversarial SOTA plan. The attached deep-research memo was directionally useful, but it did not inspect this repository and therefore invented several command/module names. The real CLI entry point here is:

```bash
python -m assayshift_tf.cli ...
```

The real model aliases currently exposed by the evaluator are:

- `gc`
- `kmer`
- `kmer_metadata`
- `tiny_cnn`
- `axis_guard_cnn`
- `axis_guard_full`
- `axis_guard_no_cf`
- `axis_guard_no_resid`
- `axis_guard_no_adv`
- `picard_tf`
- `embedding_head`
- `embedding_logreg`
- `embedding_mlp`
- `embedding_metadata`

The real A100/HF runner is:

```bash
scripts/run_colab_hf_axisguard.py
scripts/run_colab_a100_hf_axisguard.sh
```

## Honest Current State

AssayShift-TF is strongest right now as a benchmark and robustness-evaluation paper with a guarded-model contribution. It is not yet a universal SOTA TF-binding method.

The cleanest positive result is the primary assay-shift setting: held-out CUT&RUN on `data/processed/assay_mvp_hg19_windows.parquet`, where `axis_guard_full` modestly beats `tiny_cnn` on AUPRC/Brier/ECE in the final evaluator. ENCODE is mixed: AxisGuard helps calibration and sometimes Brier under held-out lab shift, but k-mer baselines can still win raw AUPRC, and the held-out TF-family setting is not a clear AxisGuard win.

The correct paper claim is therefore:

> AssayShift-TF exposes assay/lab/family shift failures that IID metrics hide, and PICARD-TF / AxisGuard provides a protocol-guarded, calibrated model that improves robustness in some assay/lab-shift settings, while the benchmark honestly reveals where classical and plain CNN baselines remain stronger.

Do not claim causal disentanglement, guaranteed assay generalization, or universal SOTA.

## What Already Exists

Implemented:

- sklearn-style evaluator compatibility via `fit` / `predict_proba`.
- `tiny_cnn` and `axis_guard_cnn` in the deep model path.
- AxisGuard losses: counterfactual protocol perturbation, metadata residual penalty, optional adversarial heads.
- CLI flags for deep epochs, batch size, LR, device, dropout, counterfactual mode, counterfactual weight, residual weight, adversarial weight.
- `evaluate-real` for final report generation.
- `sweep-real` for repeated seed sweeps.
- Pairwise model-delta tables for results and selective metrics.
- Paired-bootstrap pairwise delta CIs for `evaluate-real`.
- Seed-sweep pairwise delta summaries for `sweep-real`.
- Frozen HF embedding cache generation via `python -m assayshift_tf.cli embed`.
- Frozen-embedding logistic/MLP heads via `embedding_head`, `embedding_logreg`, `embedding_mlp`, and `embedding_metadata`.
- Reverse-complement augmentation/ensembling for local CNNs via `--rc-augment` and `--rc-ensemble`.
- Protocol-aware calibration via `--calibration protocol_platt`.
- GroupDRO training for deep models via `--deep-objective groupdro`.
- CORAL/MMD protocol alignment via `--protocol-penalty coral|mmd`.
- LoRA/PEFT hooks in `scripts/run_colab_hf_axisguard.py`.
- Colab/A100 CNN sweep script.
- Colab/A100 HF AxisGuard script.

Not yet implemented as first-class repo features:

- Full Enformer/Borzoi feature extraction.
- Automated paper table generation from the new pairwise delta files.

Those are the real SOTA upgrade hooks.

## First A100 Priority: Prove The Existing Gap

Before adding new architecture, run clean seed sweeps. If the narrow AxisGuard win vanishes across seeds, the model story must be downgraded and the paper should be framed mostly as a benchmark.

Use the existing script:

```bash
cd /content/AssayShift-TF
bash scripts/run_colab_a100_axisguard_sweep.sh
```

Expected outputs include files like:

```text
reports/real_assay_mvp_hg19_axisguard_clean_5seed_a100_seed_results.csv
reports/real_assay_mvp_hg19_axisguard_clean_5seed_a100_seed_result_summary.csv
reports/real_assay_mvp_hg19_axisguard_clean_5seed_a100_seed_selective_summary.csv
reports/real_encode_k562_grch38_axisguard_clean_5seed_a100_seed_results.csv
reports/real_encode_k562_grch38_axisguard_clean_5seed_a100_seed_result_summary.csv
reports/real_encode_k562_grch38_axisguard_clean_5seed_a100_seed_selective_summary.csv
```

Decision rule:

- If AxisGuard beats `tiny_cnn` with stable mean and small std on held-out CUT&RUN, keep the AxisGuard model claim.
- If the win is unstable, make the benchmark/calibration framework the main claim.
- If ENCODE family shift stays worse than `tiny_cnn`, report that plainly.

## Second A100 Priority: Final Paper-Facing Evaluator Runs

Run final report generation with all baseline families and bootstrap CIs.

Primary assay-shift command:

```bash
python -m assayshift_tf.cli evaluate-real data/processed/assay_mvp_hg19_windows.parquet \
  --out reports \
  --figures figures \
  --prefix real_assay_mvp_hg19_axisguard_final_a100 \
  --split iid \
  --split "assay_heldout_cutrun=assay:CUT&RUN" \
  --model gc \
  --model kmer \
  --model kmer_metadata \
  --model tiny_cnn \
  --model axis_guard_full \
  --deep-epochs 50 \
  --deep-batch-size 1024 \
  --deep-device cuda \
  --counterfactual-mode mask_or_shuffle \
  --rc-augment \
  --rc-ensemble \
  --calibration protocol_platt \
  --drop-all-n \
  --drop-duplicate-sequences \
  --bootstrap 1000
```

ENCODE stress command:

```bash
python -m assayshift_tf.cli evaluate-real data/processed/encode_k562_grch38_windows.parquet \
  --out reports \
  --figures figures \
  --prefix real_encode_k562_grch38_axisguard_final_a100 \
  --split iid \
  --split "lab_heldout_haib=lab:Richard Myers, HAIB" \
  --split "family_heldout_zinc_finger=tf_family:zinc_finger" \
  --model gc \
  --model kmer \
  --model kmer_metadata \
  --model tiny_cnn \
  --model axis_guard_full \
  --deep-epochs 50 \
  --deep-batch-size 1024 \
  --deep-device cuda \
  --counterfactual-mode mask_or_shuffle \
  --rc-augment \
  --rc-ensemble \
  --calibration protocol_platt \
  --drop-all-n \
  --drop-duplicate-sequences \
  --bootstrap 1000
```

## Frozen-Embedding Head Runs

Cache embeddings once:

```bash
python -m assayshift_tf.cli embed data/processed/assay_mvp_hg19_windows.parquet \
  --out reports/assay_ntv2_250m_embeddings.npz \
  --hf-model InstaDeepAI/nucleotide-transformer-v2-250m-multi-species \
  --device cuda \
  --batch-size 32 \
  --max-tokens 256 \
  --pooling mean \
  --trust-remote-code \
  --drop-all-n \
  --drop-duplicate-sequences
```

Evaluate the frozen representation through the normal benchmark:

```bash
python -m assayshift_tf.cli evaluate-real data/processed/assay_mvp_hg19_windows.parquet \
  --out reports \
  --figures figures \
  --prefix real_assay_ntv2_embedding_head \
  --split iid \
  --split "assay_heldout_cutrun=assay:CUT&RUN" \
  --model embedding_logreg \
  --model embedding_metadata \
  --embedding-cache reports/assay_ntv2_250m_embeddings.npz \
  --calibration protocol_platt \
  --bootstrap 1000 \
  --drop-all-n \
  --drop-duplicate-sequences
```

## GroupDRO / Alignment Runs

```bash
python -m assayshift_tf.cli sweep-real data/processed/encode_k562_grch38_windows.parquet \
  --out reports \
  --prefix real_encode_groupdro_mmd_5seed \
  --split "lab_heldout_haib=lab:Richard Myers, HAIB" \
  --split "family_heldout_zinc_finger=tf_family:zinc_finger" \
  --model tiny_cnn \
  --model axis_guard_full \
  --deep-epochs 50 \
  --deep-batch-size 1024 \
  --deep-device cuda \
  --deep-objective groupdro \
  --group-key protocol \
  --groupdro-eta 0.2 \
  --protocol-penalty mmd \
  --protocol-penalty-weight 0.01 \
  --rc-augment \
  --rc-ensemble \
  --calibration protocol_platt \
  --drop-all-n \
  --drop-duplicate-sequences
```

## Third A100 Priority: Stronger Foundation-Model Arm

The current HF model arm uses Nucleotide Transformer v2 50M. On an A100, the most direct stronger run is the same runner with the 250M checkpoint:

```bash
cd /content/AssayShift-TF
HF_MODEL=InstaDeepAI/nucleotide-transformer-v2-250m-multi-species \
  bash scripts/run_colab_a100_hf_axisguard.sh
```

Optional LoRA run:

```bash
HF_MODEL=InstaDeepAI/nucleotide-transformer-v2-250m-multi-species \
PEFT=lora \
LORA_R=16 \
LORA_ALPHA=32 \
  bash scripts/run_colab_a100_hf_axisguard.sh
```

Treat 500M as experimental only after 250M works. If 250M underperforms k-mer/CNN again, do not feature it as the headline model. Instead use it as an honest result: foundation DNA models do not automatically solve protocol shift.

## What To Implement Next

Ranked by expected value:

1. Run frozen embeddings for NT-v2-250M and DNABERT-2; compare `embedding_logreg` and `embedding_metadata`.
2. Run GroupDRO + MMD on the ENCODE lab/family splits.
3. Run LoRA only if frozen embeddings show signal under shift.
4. Generate paper tables from pairwise deltas and seed summaries.

## Reviewer-Proof Claim Template

Use this only if seed sweeps and bootstrap CIs support it:

> Across leakage-controlled assay/lab/family shifts, AssayShift-TF shows that IID TF-binding performance overstates reliability under protocol shift. AxisGuard improves calibration and selective prediction under held-out assay/lab shifts relative to a capacity-matched CNN, but family shift remains challenging and classical k-mer models remain competitive on raw AUPRC.

Avoid:

- "SOTA TF-binding predictor"
- "causal disentanglement"
- "guaranteed generalization"
- "foundation models solve assay shift"
- any claim based on a single seed

## Venue Framing

Best framing:

- benchmark plus robustness evaluation;
- calibrated/selective prediction under assay/lab/family shift;
- guarded model as a serious ablation, not an overclaimed universal method.

Best-fit venues are computational biology / ML-for-genomics venues, especially if the final paper emphasizes the benchmark and honest failure analysis. A main methods claim requires stable, statistically supported wins with CIs.
