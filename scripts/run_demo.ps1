Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location "C:\Users\Aadi Nair\AssayShift-TF"
python -m assayshift_tf.cli demo --n 6000 --out reports --figures figures
