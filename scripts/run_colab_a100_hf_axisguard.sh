#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root in Colab after staging data/processed/*.parquet:
#   cd /content/AssayShift-TF
#   bash scripts/run_colab_a100_hf_axisguard.sh

python -m pip install -e ".[deep]"

HF_MODEL="${HF_MODEL:-InstaDeepAI/nucleotide-transformer-v2-50m-multi-species}"
COMMON_ARGS=(
  --hf-model "$HF_MODEL"
  --model-name hf_axis_guard_nt
  --seed 13 --seed 17 --seed 23
  --epochs 8
  --batch-size 16
  --lr 2e-5
  --device cuda
  --pooling attention
  --drop-all-n
  --drop-duplicate-sequences
  --rc-augment
  --rc-ensemble
  --counterfactual-weight 0.2
  --metadata-residual-weight 0.02
  --adversarial-weight 0.02
  --protocol-dropout 0.1
  --grad-clip 1.0
)

python scripts/run_colab_hf_axisguard.py data/processed/encode_k562_grch38_windows.parquet \
  --out reports \
  --prefix hf_axisguard_encode_lab_3seed \
  --split "lab_heldout_haib=lab:Richard Myers, HAIB" \
  "${COMMON_ARGS[@]}"

python scripts/run_colab_hf_axisguard.py data/processed/encode_k562_grch38_windows.parquet \
  --out reports \
  --prefix hf_axisguard_encode_family_3seed \
  --split family_heldout_zinc_finger=tf_family:zinc_finger \
  "${COMMON_ARGS[@]}"

python scripts/run_colab_hf_axisguard.py data/processed/assay_mvp_hg19_windows.parquet \
  --out reports \
  --prefix hf_axisguard_assay_cutrun_3seed \
  --split "assay_heldout_cutrun=assay:CUT&RUN" \
  "${COMMON_ARGS[@]}"
