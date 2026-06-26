# PICARD-TF: Protocol-Factorized and Uncertainty-Calibrated TF Binding Prediction Under Assay and Lab Shift

## Abstract

Genomic sequence models are commonly evaluated under random or weakly structured splits, but TF-binding labels are produced by measurement pipelines whose assay protocols, laboratories, antibodies, genome builds, and negative-sampling choices can change across deployments. We introduce AssayShift-TF, a benchmark and reporting framework for transcription-factor binding prediction under held-out assay, lab/source, species, and TF-family shifts. We also introduce PICARD-TF, implemented here as `axis_guard_cnn`, a protocol-factorized CNN that combines a DNA sequence encoder, TF/family embeddings, a guarded protocol branch, counterfactual metadata masking, and calibrated selective prediction. In a 3-TF K562 hg19 pilot, k-mer-plus-metadata performance drops from IID AUPRC 0.765 to held-out CUT&RUN AUPRC 0.715, with uncalibrated ECE worsening from 0.159 to 0.243. A plain CNN improves held-out assay ranking, while AxisGuard-CNN improves the most selective held-out CUT&RUN subset, reaching calibrated AUPRC 0.959 at 20% coverage. In an ENCODE K562 GRCh38 stress test, AxisGuard-CNN improves over the plain CNN under held-out HAIB lab shift from AUPRC 0.702 to 0.731, calibrated ECE 0.092 to 0.032, and calibrated Brier score 0.209 to 0.196. These preliminary results support a cautious claim: protocol-factorized guarding can improve reliability under some structured shifts, while the benchmark reveals settings where simpler sequence models remain stronger.

## 1. Introduction

TF-binding prediction is a central task in regulatory genomics, but binary binding labels are not purely biological observations. They are produced by experimental protocols, antibodies, sequencing depth, peak callers, controls, and laboratory-specific processing choices. Models that appear strong under IID splits may therefore learn a mixture of sequence grammar and measurement artifacts. In practical deployment, a model may be trained on ChIP-seq but applied to CUT&RUN, trained on one producing lab and evaluated on another, or asked to generalize to a sparse TF family.

AssayShift-TF reframes TF-binding prediction as a shift-aware and calibration-aware benchmark. The central question is not only whether a model ranks positives above negatives, but whether it remains calibrated and useful when the data-generating protocol changes. This matters for downstream genomics workflows where overconfident errors can waste validation effort or bias biological interpretation.

This paper makes two contributions. First, we build a real-data benchmark pipeline with explicit assay-held-out, lab-held-out, and TF-family-held-out splits, matched negative controls, bootstrap uncertainty, grouped metrics, and selective prediction curves. Second, we add PICARD-TF, a protocol-factorized model that tries to learn binding signal while discouraging direct reliance on protocol shortcuts.

## 2. Related Work

DeepBind, DeepSEA-style CNNs, DNABERT, Nucleotide Transformer, HyenaDNA, Enformer, and related models establish that sequence models can learn regulatory grammar. Cross-species and TF-held-out work shows that biological extrapolation is difficult and that domain-adaptation methods such as DANN, GroupDRO, IRM, and Fishr can help in some settings. Calibration and selective classification provide tools for deciding when predictions should be trusted. AssayShift-TF combines these threads around a concrete deployment problem: TF-binding predictions under assay, lab, and family shift with explicit reliability reporting.

## 3. Benchmark

Each example contains a fixed-length DNA sequence window, binary binding label, TF identity, TF family, assay, lab/source, species, assembly, biosample, cell type, sequence statistics, and provenance fields. Positives are peaks or called peaks; negatives are sampled at equal prevalence and matched by GC or GC+accessibility when available.

The current real-data benchmark has two tracks.

1. **hg19 assay-shift pilot:** K562 CTCF, MAX, and MYC. Training data are ENCODE TF ChIP-seq peaks. Held-out assay data are deterministic CUT&RUN peaks called from GSE84474 fragments using a no-antibody control. The table has 6,000 windows, balanced positive/negative labels, and GC-matched negatives.
2. **GRCh38 ENCODE stress test:** Eight K562 TF ChIP-seq rows with CTCF, MYC, MAX, GATA1, GATA2, REST, JUND, and SPI1. The table has 8,000 windows, balanced labels, and GC+accessibility-matched negatives.

Splits are IID, held-out CUT&RUN assay, held-out HAIB lab/source, and held-out zinc-finger TF family. Direct leakage fields such as `example_id`, genomic coordinates, `source_width`, and `source_score` are never model inputs. For metadata-aware models, held-out split axes and direct correlates are masked, for example assay shift masks assay and lab, and TF-family shift masks TF and family.

## 4. Model

PICARD-TF is implemented as `axis_guard_cnn`. It uses:

- a one-hot DNA sequence Conv1D encoder;
- TF and TF-family embeddings for biological context;
- a protocol branch for assay, lab, species, assembly, biosample, cell type, GC, N fraction, and window length;
- counterfactual consistency loss under masked protocol metadata;
- a residual penalty on protocol-only logits;
- an optional assay/lab adversary with gradient reversal;
- Platt scaling and selective prediction using the existing validation split.

The training objective is:

```text
BCE(label)
+ lambda_cf * consistency(pred(x, metadata), pred(x, masked_protocol_metadata))
+ lambda_resid * residual_protocol_logit^2
+ lambda_adv * assay_lab_adversary_loss
```

We compare against GC logistic regression, k-mer logistic regression, k-mer+metadata logistic regression, and `tiny_cnn`, which shares the sequence CNN backbone but has no protocol guard.

## 5. Metrics

We report AUPRC, AUROC, expected calibration error, Brier score, worst-group AUPRC, bootstrap confidence intervals, split counts, grouped metrics by metadata axis, reliability curves, and selective prediction curves at 100%, 80%, 60%, 40%, and 20% coverage. Calibration is fit only on validation predictions from the same split protocol.

## 6. Results

### 6.1 Held-out CUT&RUN assay shift

In the hg19 pilot, k-mer+metadata performance drops from IID AUPRC 0.765 [0.742, 0.798] to held-out CUT&RUN AUPRC 0.715 [0.694, 0.736]. Uncalibrated ECE worsens from 0.159 [0.143, 0.187] to 0.243 [0.229, 0.263], showing that IID evaluation overstates reliability.

The CNN baselines outperform k-mer models on held-out CUT&RUN. The plain CNN reaches held-out AUPRC 0.796 [0.775, 0.814], calibrated ECE 0.044 [0.041, 0.064], and calibrated Brier 0.193 [0.187, 0.201]. AxisGuard-CNN reaches held-out AUPRC 0.790 [0.769, 0.807], calibrated ECE 0.074 [0.065, 0.092], and calibrated Brier 0.200 [0.194, 0.208]. Thus the guarded model does not win full-coverage assay shift in this pilot.

Selective prediction reveals a smaller positive signal: at 20% coverage on held-out CUT&RUN, AxisGuard-CNN reaches calibrated AUPRC 0.959 and Brier 0.071, versus plain CNN AUPRC 0.952 and Brier 0.077.

### 6.2 Held-out lab/source shift

In the ENCODE GRCh38 benchmark, lab-held-out evaluation is the strongest current model result. Under held-out Richard Myers/HAIB lab shift, AxisGuard-CNN improves over the plain CNN from AUPRC 0.702 [0.677, 0.735] to 0.731 [0.706, 0.766], calibrated ECE 0.092 [0.079, 0.113] to 0.032 [0.027, 0.060], and calibrated Brier 0.209 [0.201, 0.215] to 0.196 [0.187, 0.203].

This supports the paper's main model claim: protocol-factorized guarding can improve reliability and discrimination under lab/source shift.

### 6.3 Held-out TF-family shift

Under held-out zinc-finger family shift, the plain CNN remains strongest at full coverage: AUPRC 0.784 and calibrated ECE 0.029, compared with AxisGuard-CNN AUPRC 0.767 and calibrated ECE 0.024. This is an important negative result. The protocol guard helps lab shift but does not solve TF-family extrapolation, where protein/motif conditioning may be required.

## 7. Limitations

These are preliminary results, not a guarantee of broad biological generalization. The hg19 assay pilot confounds assay and lab because CUT&RUN and ChIP-seq come from different sources. CUT&RUN labels are deterministic called peaks, not gold-standard labels. Both real tables are balanced by construction, and negative labels remain imperfect because unmeasured condition-specific binding may exist. ENCODE-only lab and family shifts are small. The current model uses TF identity/family metadata but not TF protein sequence, so it is not yet a strong unseen-TF biological extrapolator.

## 8. Conclusion

AssayShift-TF provides a rigorous, reproducible benchmark for TF-binding prediction under structured protocol and biological shift. PICARD-TF / AxisGuard-CNN adds a concrete model contribution: protocol-factorized guarding with calibrated selective prediction. The current evidence is promising but nuanced: the model improves lab-shift reliability and some selective assay-shift behavior, while a plain CNN remains stronger in some full-coverage assay and family splits. That honesty strengthens the submission because the benchmark exposes real failure modes rather than hiding them behind IID scores.

## Submission Figures And Artifacts

- Model schematic: `C:\Users\Aadi Nair\AssayShift-TF\figures\picard_tf_schematic.png`
- Primary assay result: `C:\Users\Aadi Nair\AssayShift-TF\reports\real_assay_mvp_hg19_axisguard_results.csv`
- Primary assay reliability: `C:\Users\Aadi Nair\AssayShift-TF\figures\real_assay_mvp_hg19_axisguard_reliability.png`
- Secondary ENCODE result: `C:\Users\Aadi Nair\AssayShift-TF\reports\real_encode_k562_grch38_axisguard_results.csv`
- Secondary ENCODE reliability: `C:\Users\Aadi Nair\AssayShift-TF\figures\real_encode_k562_grch38_axisguard_reliability.png`

