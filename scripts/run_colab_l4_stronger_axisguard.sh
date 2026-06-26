#!/usr/bin/env bash
set -euo pipefail

# Colab L4/A100 stronger-results run:
#   cd /content/AssayShift-TF
#   bash scripts/run_colab_l4_stronger_axisguard.sh
#
# This uses best-validation checkpointing and lower backbone LR to reduce the
# overfitting seen when the pretrained encoder is fine-tuned for a fixed 8 epochs.

python -m pip install -U -e ".[deep]"

export TOKENIZERS_PARALLELISM=false
HF_MODEL="${HF_MODEL:-InstaDeepAI/nucleotide-transformer-v2-50m-multi-species}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-12}"
PATIENCE="${PATIENCE:-2}"
BACKBONE_LR="${BACKBONE_LR:-5e-6}"
HEAD_LR="${HEAD_LR:-1e-4}"

COMMON_ARGS=(
  --hf-model "$HF_MODEL"
  --trust-remote-code
  --model-name hf_axis_guard_nt_es
  --seed 13 --seed 17 --seed 23
  --epochs "$EPOCHS"
  --early-stopping-patience "$PATIENCE"
  --early-stopping-metric valid_brier
  --batch-size "$BATCH_SIZE"
  --lr "$HEAD_LR"
  --head-lr "$HEAD_LR"
  --backbone-lr "$BACKBONE_LR"
  --weight-decay 0.01
  --device cuda
  --pooling attention
  --dropout 0.2
  --drop-all-n
  --drop-duplicate-sequences
  --rc-augment
  --rc-ensemble
  --counterfactual-weight 0.2
  --metadata-residual-weight 0.02
  --adversarial-weight 0.02
  --protocol-dropout 0.25
  --grad-clip 1.0
)

# Most important manuscript claims first.
python scripts/run_colab_hf_axisguard.py data/processed/assay_mvp_hg19_windows.parquet \
  --out reports \
  --prefix hf_axisguard_assay_cutrun_es_3seed \
  --split "assay_heldout_cutrun=assay:CUT&RUN" \
  "${COMMON_ARGS[@]}"

python scripts/run_colab_hf_axisguard.py data/processed/encode_k562_grch38_windows.parquet \
  --out reports \
  --prefix hf_axisguard_encode_lab_es_3seed \
  --split "lab_heldout_haib=lab:Richard Myers, HAIB" \
  "${COMMON_ARGS[@]}"

# Family shift was weaker in the first run; keep it optional to save L4 time.
if [[ "${RUN_FAMILY:-0}" == "1" ]]; then
  python scripts/run_colab_hf_axisguard.py data/processed/encode_k562_grch38_windows.parquet \
    --out reports \
    --prefix hf_axisguard_encode_family_es_3seed \
    --split family_heldout_zinc_finger=tf_family:zinc_finger \
    "${COMMON_ARGS[@]}"
fi
