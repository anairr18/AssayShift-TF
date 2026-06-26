# A100 Path From Reviewer-Solid To SOTA-Oriented

## What The New Additions Buy

The repeated seeds, ablations, duplicate/all-N filtering, and careful claim framing make the paper much more defensible. They help answer reviewer questions like:

- Is the AxisGuard result stable across random seeds?
- Which part of the method matters: counterfactual masking, residual penalty, adversary, or just CNN capacity?
- Did duplicate sequence windows or all-N negatives inflate performance?
- Are we overselling weak family/species shift results?

These additions are necessary for a serious MLCB paper. They do not alone make the model SOTA.

## What SOTA Needs On The Colab A100

Use the A100 for representation strength after the benchmark is clean:

1. Run the 5-seed ablation sweep:

```bash
bash scripts/run_colab_a100_axisguard_sweep.sh
```

2. If AxisGuard remains stable, add one stronger backbone:

- frozen DNABERT-2 / Nucleotide Transformer / HyenaDNA embeddings plus the same AxisGuard protocol head;
- then optionally fine-tune only the final transformer block or adapters;
- keep `tiny_cnn` as the efficiency baseline.

3. Add TF biological conditioning for family shift:

- TF protein or DNA-binding-domain embedding;
- motif/PWM prior score when available;
- family-held-out evaluation should be treated as the main test for this extension.

4. Report SOTA only on a clearly named setting:

- "best under lab shift";
- "best calibrated selective predictor";
- "best worst-group AUPRC";
- not "universal SOTA TF-binding predictor" unless it beats strong baselines across assay, lab, and family splits.

## Current Honest Claim

The current repo supports this claim:

> PICARD-TF / AxisGuard-CNN is a protocol-factorized reliability model that improves lab-shift calibration and discrimination over a plain CNN in a real ENCODE K562 stress test, while the benchmark exposes settings where a plain sequence CNN remains stronger.

That is already a meaningful MLCB contribution. The A100 should now be used to test whether a stronger pretrained DNA/protein backbone turns this into a SOTA predictor rather than only a reliability-oriented model.

