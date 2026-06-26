# AssayShift-TF Paper Outline

## Working Thesis

TF-binding predictors need protocol-aware evaluation because random or weakly structured splits can hide failures that appear when assay, lab, species, or TF family changes. AssayShift-TF frames these deployment settings as explicit benchmark splits and evaluates not only discrimination, but also calibration, selective prediction, and grouped failure modes.

## 1. Title and Abstract

**Narrative.** Present AssayShift-TF as a compact benchmark and PICARD-TF as a protocol-factorized model for TF-binding prediction under structured experimental and biological shift. The abstract should emphasize the strongest supported model result: AxisGuard-CNN improves lab-shift reliability over a plain CNN, while assay/family results remain nuanced.

**Evidence needed.**

- Final list of data sources and metadata fields actually parsed.
- One small preliminary result table or figure, once available.
- A clear statement of which splits are implemented in the MVP.

## 2. Introduction

**Narrative.** TF-binding prediction is central for regulatory genomics, but current evaluation can conflate sequence learning with repeated protocols and source-specific biases. In realistic use, users apply predictors across labs, assay protocols, species, and TF families. The paper asks whether models remain accurate, calibrated, and useful when these axes shift.

**Claims that need evidence.**

- Existing random or chromosome splits are insufficient for deployment-like stress tests.
- Metadata axes such as assay, lab, species, and TF family are available often enough to support benchmark construction.
- Calibration and abstention matter for downstream use, not just AUROC/AUPRC.

## 3. Related Work

**Narrative.** Position against TF-binding prediction methods, genomic foundation models, cross-cell-type and cross-species regulatory prediction, dataset shift benchmarks, and uncertainty/calibration work in biology. Keep this section compact for a 2-page abstract; name categories rather than surveying exhaustively.

**Claims that need evidence.**

- Which prior TF benchmarks use random, chromosome, cell-type, or TF-held-out splits.
- Whether any prior work explicitly studies assay or lab/source shift for TF binding.
- Which calibration or selective prediction metrics are standard enough to cite.

## 4. Benchmark Construction

**Narrative.** Describe the input schema: sequence window, binary label, TF, species, assay/protocol, lab/source, cell context if available, TF family, and processing metadata. Explain how positives and negatives are generated and how splits block leakage along the target axis.

**Claims that need evidence.**

- Exact data source names and licenses.
- Genome builds and sequence window length.
- Positive and negative sampling rules.
- Metadata harmonization rules and missing-value handling.
- Number of examples, TFs, families, assays, labs/sources, species, and groups per split.

## 5. Protocol-Aware Splits

**Narrative.** Define each split as a deployment scenario:

- **In-distribution control:** random or chromosome-held-out examples within shared metadata distribution.
- **Assay/protocol shift:** hold out a measurement protocol or assay family.
- **Lab/source shift:** hold out examples from one producing lab or source.
- **Species shift:** train on one species and evaluate on another.
- **TF-family shift:** hold out one or more TF families.

**Claims that need evidence.**

- Split construction avoids leakage from duplicated loci, shared peaks, or identical TF/protocol records where relevant.
- Group sizes are sufficient for stable metrics.
- Held-out groups are biologically and experimentally meaningful.

## 6. Models

**Narrative.** Use simple baselines to make the benchmark interpretable: GC/artifact logistic regression, k-mer/logistic, k-mer-plus-metadata logistic regression, and a plain sequence CNN. Add PICARD-TF / AxisGuard-CNN as the novel model: sequence CNN plus TF/family embeddings, guarded protocol metadata branch, counterfactual protocol masking, metadata residual penalty, optional assay/lab adversary, and calibration/selective prediction.

**Claims that need evidence.**

- Baselines are implemented and reproducible.
- Model capacity and training budgets are comparable.
- Protocol-aware models do not receive unavailable deployment metadata in a way that leaks the answer; held-out assay masks assay and lab, held-out species masks species and assembly, and held-out family masks TF and family.

## 7. Metrics and Reporting

**Narrative.** Report discrimination, calibration, and decision utility under each shift. Main metrics: AUROC, AUPRC, precision at fixed recall, Brier score, ECE, calibration-in-the-large, worst-group AUPRC, and selective prediction risk-coverage.

**Claims that need evidence.**

- Metric definitions and confidence interval method.
- Calibration method, if any, fit only on allowed validation data.
- Selective prediction protocol, including score used for abstention.

## 8. Preliminary Results

**Narrative.** Include the PICARD-TF schematic, the main assay/lab shift result table, reliability curves, and risk-coverage curves. The central model claim should be lab-shift reliability improvement, not universal dominance.

**Claims that need evidence.**

- Real metric table generated from `reports/real_assay_mvp_hg19_axisguard_results.csv` and `reports/real_encode_k562_grch38_axisguard_results.csv`.
- Reproducible scripts in `scripts/run_assay_mvp_hg19.ps1` and `scripts/run_encode_grch38.ps1`.
- Sanity checks for label balance, split counts, and leakage-prone column exclusion.

## 9. Failure Analysis

**Narrative.** Analyze where models fail by TF family, motif strength, GC content, sequence complexity, lab/source, assay, and species. The goal is to separate possible biological extrapolation limits from protocol artifacts.

**Claims that need evidence.**

- Motif or sequence-feature summaries for false positives and false negatives.
- Grouped calibration and AUPRC tables.
- Examples of high-confidence errors, if interpretable.

## 10. Limitations and Scope

**Narrative.** Be direct: metadata is incomplete, sources may be confounded, negative labels are imperfect, species shifts are not purely biological, and the MVP is a benchmark/report artifact rather than a definitive biological model.

**Claims that need evidence.**

- Known missing metadata rates.
- Potential confounders per split.
- Which data sources or species are excluded from the MVP.

## 11. Conclusion

**Narrative.** AssayShift-TF reframes TF-binding prediction evaluation around realistic shift axes and decision reliability. The main takeaway should be a benchmark protocol and preliminary evidence that protocol-aware reporting changes the interpretation of model quality.

**Claims that need evidence.**

- At least one concrete result showing that shift-aware evaluation changes model ranking, calibration assessment, or selective prediction behavior.

## One-Day Execution Checklist

1. Freeze the MVP claim: benchmark/report first, model novelty second.
2. Confirm which data files and metadata fields actually exist locally.
3. Generate or inspect split counts for random/chromosome, assay, lab/source, species, and TF-family axes.
4. Run the smallest reproducible baseline on one in-distribution split and one shift split.
5. Export a metrics table with AUROC, AUPRC, Brier score, ECE, and group counts.
6. Add one calibration or selective prediction plot if the pipeline supports it.
7. Replace the preliminary-result placeholder in `docs/mlcb_abstract_draft.md` with observed numbers only after verifying output files.
8. Render the Mermaid schematic or include it as source in the abstract supplement.
9. Add a limitations sentence tied to any failed or underpowered split.
10. Do a final pass for overclaiming: remove any statement that implies completed experiments without a generated result file.
