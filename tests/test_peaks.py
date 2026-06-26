import gzip

import pandas as pd

from assayshift_tf.peaks import build_sequence_table
from assayshift_tf.peaks import call_cutrun_peaks, read_bed


def test_read_bed_gz(tmp_path):
    path = tmp_path / "x.bed.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("chr1\t10\t20\tpeak1\t100\t.\t5\t1\t1\t4\n")
    frame = read_bed(path)
    assert list(frame.columns) == [
        "chrom",
        "start",
        "end",
        "name",
        "score",
        "strand",
        "signal_value",
        "p_value",
        "q_value",
        "summit",
    ]
    assert frame.loc[0, "chrom"] == "chr1"
    assert frame.loc[0, "start"] == 10
    assert frame.loc[0, "summit"] == 4


def test_gc_matched_negatives_exclude_positive_padding(tmp_path):
    fasta = tmp_path / "toy.fa"
    fasta.write_text(">chr1\n" + "ACGT" * 3000 + "\n")
    bed = tmp_path / "peaks.bed"
    bed.write_text("chr1\t100\t120\tp1\t100\t.\t1\t1\t1\t10\n")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "dataset_id": "toy",
                "source": "test",
                "tf": "CTCF",
                "species": "Homo sapiens",
                "assembly": "toy",
                "assay": "TF ChIP-seq",
                "lab": "lab",
                "biosample": "K562",
                "cell_type": "cell line",
                "processed_peak_url": "",
                "control_url": "",
                "notes": "",
                "citation_url": "",
            }
        ]
    ).to_csv(manifest, index=False)
    downloads = tmp_path / "downloads.csv"
    pd.DataFrame([{"dataset_id": "toy", "status": "ok", "path": str(bed)}]).to_csv(downloads, index=False)

    out = tmp_path / "windows.csv"
    frame = build_sequence_table(
        manifest,
        downloads,
        {"toy": fasta},
        out,
        max_peaks_per_dataset=1,
        negatives_per_positive=1,
        window_size=20,
        negative_strategy="gc",
        exclude_padding=50,
        candidate_pool=8,
    )

    assert set(frame["label"]) == {0, 1}
    neg = frame[frame["label"].eq(0)].iloc[0]
    assert not (50 <= neg.start <= 170)
    assert neg.negative_strategy == "gc"
    assert frame.loc[frame["label"].eq(1), "tf_family"].iloc[0] == "zinc_finger"


def test_call_cutrun_peaks_clusters_midpoints_and_counts_control():
    fragments = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1", "chr1", "chr2"],
            "start": [95, 100, 106, 300, 50],
            "end": [105, 110, 116, 310, 60],
        }
    )
    control = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1", "chr2"],
            "start": [90, 100, 117, 40],
            "end": [96, 101, 125, 70],
        }
    )

    peaks = call_cutrun_peaks(
        fragments,
        control_fragments=control,
        source="GSM_TEST",
        max_gap=12,
        min_fragments=2,
        peak_padding=5,
    )

    assert len(peaks) == 1
    peak = peaks.iloc[0]
    assert list(peaks.columns[:10]) == [
        "chrom",
        "start",
        "end",
        "name",
        "score",
        "strand",
        "signal_value",
        "p_value",
        "q_value",
        "summit",
    ]
    assert peak["chrom"] == "chr1"
    assert peak["start"] == 95
    assert peak["end"] == 117
    assert peak["name"] == "GSM_TEST_peak_1"
    assert peak["summit"] == 10
    assert peak["fragment_count"] == 3
    assert peak["control_overlap_count"] == 2
    assert peak["width"] == 22
    assert peak["source"] == "GSM_TEST"


def test_call_cutrun_peaks_aggregates_duplicate_midpoints():
    fragments = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [8, 9, 100],
            "end": [12, 11, 104],
        }
    )

    peaks = call_cutrun_peaks(
        fragments,
        source="CUT&RUN",
        max_gap=0,
        min_fragments=2,
        peak_padding=0,
    )

    assert len(peaks) == 1
    assert peaks.loc[0, "start"] == 10
    assert peaks.loc[0, "end"] == 11
    assert peaks.loc[0, "summit"] == 0
    assert peaks.loc[0, "fragment_count"] == 2
    assert peaks.loc[0, "control_overlap_count"] == 0
    assert peaks.loc[0, "name"] == "CUT_RUN_peak_1"
