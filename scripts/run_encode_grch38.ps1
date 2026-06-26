Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location "C:\Users\Aadi Nair\AssayShift-TF"

python -m assayshift_tf.cli validate-manifest data\manifests\encode_k562_grch38.csv --strict
python -m assayshift_tf.cli download-peaks data\manifests\encode_k562_grch38.csv --out data\raw\encode_k562_grch38_peaks --index data\raw\encode_k562_grch38_download_index.csv

if (-not (Test-Path data\raw\accessibility\ENCFF912COY.bed.gz)) {
  curl.exe -L --retry 3 --retry-delay 5 -o data\raw\accessibility\ENCFF912COY.bed.gz "https://www.encodeproject.org/files/ENCFF912COY/@@download/ENCFF912COY.bed.gz"
}

python -m assayshift_tf.cli build-windows data\manifests\encode_k562_grch38.csv --download-index data\raw\encode_k562_grch38_download_index.csv --fasta GRCh38="C:\Users\Aadi Nair\DNA-Diffusion\data\hg38.fa" --out data\processed\encode_k562_grch38_windows.parquet --max-peaks-per-dataset 500 --negatives-per-positive 1 --window-size 211 --negative-strategy gc_accessibility --accessibility-bed data\raw\accessibility\ENCFF912COY.bed.gz --candidate-pool 12 --seed 13
python -m assayshift_tf.cli evaluate-real data\processed\encode_k562_grch38_windows.parquet --out reports --figures figures --prefix real_encode_k562_grch38_axisguard --split iid --split "lab_heldout_haib=lab:Richard Myers, HAIB" --split family_heldout_zinc_finger=tf_family:zinc_finger --model gc --model kmer --model kmer_metadata --model tiny_cnn --model axis_guard_full --deep-epochs 25 --deep-batch-size 256 --drop-all-n --drop-duplicate-sequences --bootstrap 100 --seed 13
