Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location "C:\Users\Aadi Nair\AssayShift-TF"

$models = @(
  "tiny_cnn",
  "axis_guard_no_cf",
  "axis_guard_no_resid",
  "axis_guard_no_adv",
  "axis_guard_full"
)
$modelArgs = @()
foreach ($model in $models) {
  $modelArgs += "--model"
  $modelArgs += $model
}

python -m assayshift_tf.cli sweep-real data\processed\encode_k562_grch38_windows.parquet `
  --out reports `
  --prefix real_encode_k562_grch38_axisguard_clean_5seed `
  --split iid `
  --split "lab_heldout_haib=lab:Richard Myers, HAIB" `
  --split family_heldout_zinc_finger=tf_family:zinc_finger `
  @modelArgs `
  --seed 13 --seed 17 --seed 23 --seed 29 --seed 31 `
  --deep-epochs 25 `
  --deep-batch-size 512 `
  --deep-device auto `
  --drop-all-n `
  --drop-duplicate-sequences `
  --bootstrap 0
