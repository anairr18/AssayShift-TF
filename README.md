# AssayShift-TF

Protocol-aware and uncertainty-calibrated transcription factor binding prediction under assay, lab, species, and family shift.

This repository is a research package for an MLCB-style submission. It contains:

- A benchmark design for TF binding prediction under realistic distribution shifts.
- Data manifest scaffolding for ENCODE/GEO-style peak files.
- Reproducible split logic for IID, assay-held-out, lab-held-out, species-held-out, and TF-family-held-out evaluation.
- Baselines for GC/artifact controls and k-mer logistic regression, with metadata-aware variants.
- `PICARD-TF` / `axis_guard_cnn`, a protocol-factorized CNN that uses counterfactual protocol masking and calibrated selective prediction under shift.
- Repeated-seed ablation sweeps for `tiny_cnn`, `axis_guard_no_cf`, `axis_guard_no_resid`, `axis_guard_no_adv`, and `axis_guard_full`.
- Evaluation-time cleanup flags for all-N windows and exact duplicate sequence strings.
- Calibration, Brier/ECE, worst-group metrics, and selective-prediction/abstention analysis.
- Real preliminary assay-shift and ENCODE shift reports for an 8-page paper draft or 2-page abstract.

## Quickstart

```powershell
cd "C:\Users\Aadi Nair\AssayShift-TF"
python -m assayshift_tf.cli demo --n 6000 --out reports --figures figures
python -m pytest
```

Key outputs:

- `reports/demo_results.csv`
- `reports/demo_selective.csv`
- `reports/preliminary_results.md`
- `figures/demo_shift_results.png`

Submission-facing outputs:

- `reports/real_assay_mvp_hg19_axisguard_preliminary_results.md`
- `figures/real_assay_mvp_hg19_axisguard_shift_results.png`
- `figures/real_assay_mvp_hg19_axisguard_reliability.png`
- `reports/real_encode_k562_grch38_axisguard_preliminary_results.md`
- `figures/real_encode_k562_grch38_axisguard_shift_results.png`
- `figures/real_encode_k562_grch38_axisguard_reliability.png`
- `figures/picard_tf_schematic.png`

## Real-Data Path

1. Curate `data/manifests/candidate_experiments.csv` with public peak files and metadata.
2. Validate required fields:

```powershell
python -m assayshift_tf.cli validate-manifest data\manifests\candidate_experiments.csv
```

3. Download processed BED/narrowPeak files:

```powershell
python -m assayshift_tf.cli download-peaks data\manifests\candidate_experiments.csv --out data\raw\candidate_peaks --index data\raw\candidate_download_index.csv
```

4. Provide FASTA files for the assemblies used in the manifest, then build sequence windows:

```powershell
python -m assayshift_tf.cli build-windows data\manifests\candidate_experiments.csv --download-index data\raw\candidate_download_index.csv --fasta GRCh38=C:\path\to\hg38.fa --fasta hg19=C:\path\to\hg19.fa --fasta mm10=C:\path\to\mm10.fa --out data\processed\windows.parquet
```

5. Evaluate a prebuilt window table with named split specs and real-data output prefixes:

```powershell
python -m assayshift_tf.cli evaluate-real data\processed\windows.parquet --prefix real --out reports --figures figures --split iid --split "assay_cutrun=assay:CUT&RUN" --split lab_henikoff=lab:Henikoff
```

This writes `real_results.csv`, `real_group_metrics.csv`, `real_selective.csv`, `real_predictions.csv`, `real_bootstrap_cis.csv`, `real_split_counts.csv`, `real_preliminary_results.md`, and `real_shift_results.png`.

Run the novel model and deep baseline explicitly:

```powershell
python -m assayshift_tf.cli evaluate-real data\processed\windows.parquet --prefix real_axisguard --out reports --figures figures --split iid --model gc --model kmer --model kmer_metadata --model tiny_cnn --model axis_guard_full --deep-epochs 25 --deep-batch-size 256 --drop-all-n --drop-duplicate-sequences
```

Run the 5-seed ablation suite:

```powershell
.\scripts\run_axisguard_ablation_sweep.ps1
```

On Colab A100:

```bash
bash scripts/run_colab_a100_axisguard_sweep.sh
```

6. Reproduce the hg19 assay-shift pilot:

```powershell
.\scripts\run_assay_mvp_hg19.ps1
.\scripts\run_encode_grch38.ps1
```

The benchmark is intentionally split-aware: the unit of evaluation is not random windows alone, but held-out assay, lab, species, or TF family groups.

Current local status: the hg19 assay pilot and GRCh38 ENCODE benchmark both run end-to-end. CUT&RUN labels are called peaks produced by the deterministic fragment midpoint caller, not raw fragment intervals.

Current single-seed AxisGuard result before the new cleaned 5-seed sweep: in the ENCODE K562 lab-held-out stress test, `axis_guard_cnn` improves over the plain `tiny_cnn` from AUPRC 0.702 to 0.731, calibrated ECE 0.092 to 0.032, and calibrated Brier score 0.209 to 0.196. In the hg19 held-out CUT&RUN pilot, `axis_guard_cnn` does not beat `tiny_cnn` at full coverage, but reaches higher calibrated 20% selective AUPRC, 0.959 versus 0.952.

## Claims This Package Is Built To Test

- IID AUPRC can be high while assay/lab-held-out AUPRC collapses.
- Shifted models can be overconfident even when ranking metrics look acceptable.
- Calibration plus selective prediction can recover reliable high-confidence subsets.
- GC/repeat/accessibility-matched controls distinguish biological sequence signal from assay shortcuts.
- Repeated-seed ablations identify whether protocol guarding, residual control, or adversarial axis removal actually drive the result.

The submission-facing claim should use the real reports, not the deterministic demo.
