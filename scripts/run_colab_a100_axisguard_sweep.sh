#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root in Colab, for example:
#   cd /content/AssayShift-TF
#   bash scripts/run_colab_a100_axisguard_sweep.sh

python -m pip install -e ".[deep]"

python -m assayshift_tf.cli sweep-real data/processed/encode_k562_grch38_windows.parquet \
  --out reports \
  --prefix real_encode_k562_grch38_axisguard_clean_5seed_a100 \
  --split iid \
  --split "lab_heldout_haib=lab:Richard Myers, HAIB" \
  --split family_heldout_zinc_finger=tf_family:zinc_finger \
  --model tiny_cnn \
  --model axis_guard_no_cf \
  --model axis_guard_no_resid \
  --model axis_guard_no_adv \
  --model axis_guard_full \
  --seed 13 --seed 17 --seed 23 --seed 29 --seed 31 \
  --deep-epochs 50 \
  --deep-batch-size 1024 \
  --deep-device cuda \
  --drop-all-n \
  --drop-duplicate-sequences \
  --bootstrap 0

python -m assayshift_tf.cli sweep-real data/processed/assay_mvp_hg19_windows.parquet \
  --out reports \
  --prefix real_assay_mvp_hg19_axisguard_clean_5seed_a100 \
  --split iid \
  --split "assay_heldout_cutrun=assay:CUT&RUN" \
  --model tiny_cnn \
  --model axis_guard_no_cf \
  --model axis_guard_no_resid \
  --model axis_guard_no_adv \
  --model axis_guard_full \
  --seed 13 --seed 17 --seed 23 --seed 29 --seed 31 \
  --deep-epochs 50 \
  --deep-batch-size 1024 \
  --deep-device cuda \
  --drop-all-n \
  --drop-duplicate-sequences \
  --bootstrap 0

