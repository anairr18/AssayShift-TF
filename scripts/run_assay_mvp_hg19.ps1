Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location "C:\Users\Aadi Nair\AssayShift-TF"

python -m assayshift_tf.cli validate-manifest data\manifests\assay_mvp_hg19.csv --strict
python -m assayshift_tf.cli download-peaks data\manifests\assay_mvp_hg19.csv --out data\raw\assay_mvp_hg19_inputs --index data\raw\assay_mvp_hg19_download_index.csv

if (-not (Test-Path data\raw\controls\GSM2433147_no_antibody.bed.gz)) {
  curl.exe -L --retry 3 --retry-delay 5 -o data\raw\controls\GSM2433147_no_antibody.bed.gz "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM2433147&format=file&file=GSM2433147%5FCut%5Fand%5FRun%5Fno%5Fantibody%5Fsize%5Fselected%2Ebed%2Egz"
}

if (-not (Test-Path data\references\hg19.fa)) {
  if (-not (Test-Path data\references\hg19.fa.gz)) {
    curl.exe -L --retry 3 --retry-delay 5 -o data\references\hg19.fa.gz "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz"
  }
  python -c "import gzip, shutil; f_in=gzip.open('data/references/hg19.fa.gz','rb'); f_out=open('data/references/hg19.fa','wb'); shutil.copyfileobj(f_in,f_out); f_in.close(); f_out.close()"
}

python -m assayshift_tf.cli call-cutrun-peaks data\raw\assay_mvp_hg19_inputs\GSM2433139__download --control data\raw\controls\GSM2433147_no_antibody.bed.gz --out data\processed\cutrun_peaks\GSM2433139_CTCF_called.bed --source GSM2433139_CTCF_CUTRUN --max-gap 75 --min-fragments 20 --peak-padding 50
python -m assayshift_tf.cli call-cutrun-peaks data\raw\assay_mvp_hg19_inputs\GSM2433145__download --control data\raw\controls\GSM2433147_no_antibody.bed.gz --out data\processed\cutrun_peaks\GSM2433145_MAX_called.bed --source GSM2433145_MAX_CUTRUN --max-gap 75 --min-fragments 20 --peak-padding 50
python -m assayshift_tf.cli call-cutrun-peaks data\raw\assay_mvp_hg19_inputs\GSM2433146__download --control data\raw\controls\GSM2433147_no_antibody.bed.gz --out data\processed\cutrun_peaks\GSM2433146_MYC_called.bed --source GSM2433146_MYC_CUTRUN --max-gap 75 --min-fragments 20 --peak-padding 50

$rows = @(
  @{dataset_id="ENCSR000EGM"; status="ok"; path="data\raw\assay_mvp_hg19_inputs\ENCSR000EGM__ENCFF002CWL.bed.gz"; bytes=(Get-Item data\raw\assay_mvp_hg19_inputs\ENCSR000EGM__ENCFF002CWL.bed.gz).Length},
  @{dataset_id="ENCSR000EFV"; status="ok"; path="data\raw\assay_mvp_hg19_inputs\ENCSR000EFV__ENCFF002CXD.bed.gz"; bytes=(Get-Item data\raw\assay_mvp_hg19_inputs\ENCSR000EFV__ENCFF002CXD.bed.gz).Length},
  @{dataset_id="ENCSR000EGS"; status="ok"; path="data\raw\assay_mvp_hg19_inputs\ENCSR000EGS__ENCFF002CWF.bed.gz"; bytes=(Get-Item data\raw\assay_mvp_hg19_inputs\ENCSR000EGS__ENCFF002CWF.bed.gz).Length},
  @{dataset_id="GSM2433139"; status="ok"; path="data\processed\cutrun_peaks\GSM2433139_CTCF_called.bed"; bytes=(Get-Item data\processed\cutrun_peaks\GSM2433139_CTCF_called.bed).Length},
  @{dataset_id="GSM2433145"; status="ok"; path="data\processed\cutrun_peaks\GSM2433145_MAX_called.bed"; bytes=(Get-Item data\processed\cutrun_peaks\GSM2433145_MAX_called.bed).Length},
  @{dataset_id="GSM2433146"; status="ok"; path="data\processed\cutrun_peaks\GSM2433146_MYC_called.bed"; bytes=(Get-Item data\processed\cutrun_peaks\GSM2433146_MYC_called.bed).Length}
)
$rows | ForEach-Object { [pscustomobject]$_ } | Export-Csv data\processed\assay_mvp_hg19_called_index.csv -NoTypeInformation

python -m assayshift_tf.cli build-windows data\manifests\assay_mvp_hg19.csv --download-index data\processed\assay_mvp_hg19_called_index.csv --fasta hg19=data\references\hg19.fa --out data\processed\assay_mvp_hg19_windows.parquet --max-peaks-per-dataset 500 --negatives-per-positive 1 --window-size 211 --negative-strategy gc --candidate-pool 24 --seed 13
python -m assayshift_tf.cli evaluate-real data\processed\assay_mvp_hg19_windows.parquet --out reports --figures figures --prefix real_assay_mvp_hg19_axisguard --split iid --split "assay_heldout_cutrun=assay:CUT&RUN" --model gc --model kmer --model kmer_metadata --model tiny_cnn --model axis_guard_full --deep-epochs 25 --deep-batch-size 256 --drop-all-n --drop-duplicate-sequences --bootstrap 100 --seed 13
