from __future__ import annotations

import gzip
import shutil
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from pyfaidx import Fasta

from assayshift_tf.features import add_sequence_stats, gc_fraction
from assayshift_tf.manifest import load_manifest


BED_COLUMNS = [
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

CUTRUN_PEAK_COLUMNS = [
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
    "fragment_count",
    "control_overlap_count",
    "width",
    "source",
]

TF_FAMILY = {
    "CTCF": "zinc_finger",
    "REST": "zinc_finger",
    "MYC": "bhlh_zip",
    "MAX": "bhlh_zip",
    "GATA1": "gata",
    "GATA2": "gata",
    "JUND": "bzip",
    "SPI1": "ets",
    "FOXA1": "forkhead",
    "ESR1": "nuclear_receptor",
}


def _download_name(url: str, dataset_id: str) -> str:
    path = urlparse(url).path
    name = Path(path).name
    if not name:
        name = f"{dataset_id}.bed"
    return f"{dataset_id}__{name}"


def download_peak_files(manifest_path: str | Path, out_dir: str | Path) -> pd.DataFrame:
    manifest = load_manifest(manifest_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for row in manifest.itertuples(index=False):
        url = getattr(row, "processed_peak_url", "")
        dataset_id = getattr(row, "dataset_id")
        if not isinstance(url, str) or not url.strip():
            rows.append({"dataset_id": dataset_id, "status": "skipped_missing_url", "path": ""})
            continue
        dest = out / _download_name(url, dataset_id)
        if not dest.exists():
            with urllib.request.urlopen(url) as response, dest.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        rows.append({"dataset_id": dataset_id, "status": "ok", "path": str(dest), "bytes": dest.stat().st_size})
    return pd.DataFrame(rows)


def read_bed(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    path = Path(path)
    with path.open("rb") as probe:
        is_gzip = probe.read(2) == b"\x1f\x8b"
    opener = gzip.open if is_gzip else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        frame = pd.read_csv(handle, sep="\t", header=None, comment="#", nrows=nrows)
    frame = frame.iloc[:, : min(frame.shape[1], len(BED_COLUMNS))]
    frame.columns = BED_COLUMNS[: frame.shape[1]]
    frame["start"] = frame["start"].astype(int)
    frame["end"] = frame["end"].astype(int)
    return frame


def _coerce_fragment_frame(fragments: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(fragments, pd.DataFrame):
        frame = fragments.copy()
    else:
        frame = read_bed(fragments)
    missing = {"chrom", "start", "end"} - set(frame.columns)
    if missing:
        raise ValueError(f"fragment table is missing columns: {', '.join(sorted(missing))}")
    frame = frame.loc[:, ["chrom", "start", "end"]].copy()
    frame["start"] = pd.to_numeric(frame["start"], errors="coerce")
    frame["end"] = pd.to_numeric(frame["end"], errors="coerce")
    frame = frame.dropna(subset=["chrom", "start", "end"])
    frame["chrom"] = frame["chrom"].astype(str)
    frame["start"] = frame["start"].astype(int)
    frame["end"] = frame["end"].astype(int)
    return frame[frame["end"] > frame["start"]].reset_index(drop=True)


def _aggregate_fragment_midpoints(fragments: pd.DataFrame | str | Path) -> pd.DataFrame:
    frame = _coerce_fragment_frame(fragments)
    columns = ["chrom", "midpoint", "fragment_count"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["midpoint"] = ((frame["start"] + frame["end"]) // 2).astype(int)
    return (
        frame.groupby(["chrom", "midpoint"], sort=True)
        .size()
        .reset_index(name="fragment_count")
        .loc[:, columns]
        .reset_index(drop=True)
    )


def _control_overlap_index(
    control_fragments: pd.DataFrame | str | Path | None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if control_fragments is None:
        return {}
    frame = _coerce_fragment_frame(control_fragments)
    index: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for chrom, group in frame.groupby("chrom", sort=False):
        starts = np.sort(group["start"].to_numpy(dtype=np.int64))
        ends = np.sort(group["end"].to_numpy(dtype=np.int64))
        index[str(chrom)] = (starts, ends)
    return index


def _count_control_overlaps(
    control_index: dict[str, tuple[np.ndarray, np.ndarray]],
    chrom: str,
    start: int,
    end: int,
) -> int:
    if chrom not in control_index:
        return 0
    starts, ends = control_index[chrom]
    starts_before_peak_end = np.searchsorted(starts, end, side="left")
    ends_before_or_at_peak_start = np.searchsorted(ends, start, side="right")
    return int(starts_before_peak_end - ends_before_or_at_peak_start)


def _peak_name_prefix(source: str) -> str:
    prefix = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(source).strip())
    return prefix or "cutrun"


def _append_cutrun_peak(
    rows: list[dict[str, object]],
    chrom: str,
    midpoint_start: int,
    midpoint_end: int,
    fragment_count: int,
    control_index: dict[str, tuple[np.ndarray, np.ndarray]],
    source: str,
    min_fragments: int,
    peak_padding: int,
) -> None:
    if fragment_count < min_fragments:
        return
    start = max(0, midpoint_start - peak_padding)
    end = midpoint_end + peak_padding + 1
    center = (midpoint_start + midpoint_end) // 2
    summit = max(0, min(end - start - 1, center - start))
    rows.append(
        {
            "chrom": chrom,
            "start": start,
            "end": end,
            "name": f"{_peak_name_prefix(source)}_peak_{len(rows) + 1}",
            "score": fragment_count,
            "strand": ".",
            "signal_value": fragment_count,
            "p_value": 0.0,
            "q_value": 0.0,
            "summit": summit,
            "fragment_count": fragment_count,
            "control_overlap_count": _count_control_overlaps(control_index, chrom, start, end),
            "width": end - start,
            "source": source,
        }
    )


def call_cutrun_peaks(
    fragments: pd.DataFrame | str | Path,
    control_fragments: pd.DataFrame | str | Path | None = None,
    *,
    source: str = "CUT&RUN",
    max_gap: int = 75,
    min_fragments: int = 2,
    peak_padding: int = 50,
) -> pd.DataFrame:
    """Call deterministic CUT&RUN peaks from BED-like fragment intervals.

    Fragment intervals are converted to midpoint support, identical midpoints
    are aggregated, and nearby midpoint bins are greedily clustered per
    chromosome. Output is BED-like: the first ten columns are compatible with
    the local BED/narrowPeak reader, followed by fragment count,
    no-antibody/control overlap count, width, and source metadata.
    """
    if max_gap < 0:
        raise ValueError("max_gap must be non-negative")
    if min_fragments < 1:
        raise ValueError("min_fragments must be at least 1")
    if peak_padding < 0:
        raise ValueError("peak_padding must be non-negative")

    midpoint_counts = _aggregate_fragment_midpoints(fragments)
    if midpoint_counts.empty:
        return pd.DataFrame(columns=CUTRUN_PEAK_COLUMNS)

    control_index = _control_overlap_index(control_fragments)
    rows: list[dict[str, object]] = []

    for chrom, group in midpoint_counts.groupby("chrom", sort=True):
        cluster_start: int | None = None
        cluster_end: int | None = None
        cluster_count = 0
        previous_midpoint: int | None = None

        for midpoint_row in group.itertuples(index=False):
            midpoint = int(midpoint_row.midpoint)
            count = int(midpoint_row.fragment_count)
            if previous_midpoint is not None and midpoint - previous_midpoint > max_gap:
                if cluster_start is not None and cluster_end is not None:
                    _append_cutrun_peak(
                        rows,
                        str(chrom),
                        cluster_start,
                        cluster_end,
                        cluster_count,
                        control_index,
                        source,
                        min_fragments,
                        peak_padding,
                    )
                cluster_start = midpoint
                cluster_count = 0
            elif cluster_start is None:
                cluster_start = midpoint

            cluster_end = midpoint
            cluster_count += count
            previous_midpoint = midpoint

        if cluster_start is not None and cluster_end is not None:
            _append_cutrun_peak(
                rows,
                str(chrom),
                cluster_start,
                cluster_end,
                cluster_count,
                control_index,
                source,
                min_fragments,
                peak_padding,
            )

    return pd.DataFrame(rows, columns=CUTRUN_PEAK_COLUMNS)


def write_cutrun_peaks(peaks: pd.DataFrame, out_path: str | Path, include_header: bool = False) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    peaks.to_csv(out, sep="\t", index=False, header=include_header)
    return out


def _peak_window(start: int, end: int, summit: object, window_size: int) -> tuple[int, int]:
    if pd.notna(summit):
        center = int(start) + int(float(summit))
    else:
        center = (int(start) + int(end)) // 2
    half = window_size // 2
    return max(0, center - half), max(window_size, center + (window_size - half))


def _fasta_lengths(fasta: Fasta) -> dict[str, int]:
    return {name: len(fasta[name]) for name in fasta.keys()}


def _negative_interval(
    chrom: str,
    chrom_len: int,
    window_size: int,
    rng: np.random.Generator,
) -> tuple[int, int]:
    if chrom_len <= window_size:
        return 0, chrom_len
    start = int(rng.integers(0, chrom_len - window_size))
    return start, start + window_size


def _sequence_from_fasta(fasta: Fasta, chrom: str, start: int, end: int) -> str:
    return str(fasta[chrom][int(start) : int(end)])


def _load_exclusion_intervals(path: str | Path | None, padding: int = 0) -> dict[str, list[tuple[int, int]]]:
    if path is None:
        return {}
    bed = read_bed(path)
    out: dict[str, list[tuple[int, int]]] = {}
    for row in bed.itertuples(index=False):
        out.setdefault(row.chrom, []).append((max(0, int(row.start) - padding), int(row.end) + padding))
    for chrom in out:
        out[chrom].sort()
    return out


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _overlaps_any(start: int, end: int, intervals: list[tuple[int, int]]) -> bool:
    for lo, hi in intervals:
        if hi <= start:
            continue
        if lo >= end:
            return False
        return True
    return False


def _positive_exclusions(
    merged: pd.DataFrame,
    downloads: pd.DataFrame,
    padding: int,
) -> dict[str, list[tuple[int, int]]]:
    exclusions: dict[str, list[tuple[int, int]]] = {}
    for dataset in merged.merge(downloads, on="dataset_id", how="inner").itertuples(index=False):
        if getattr(dataset, "status") != "ok":
            continue
        try:
            bed = read_bed(getattr(dataset, "path"))
        except Exception:
            continue
        for row in bed.itertuples(index=False):
            exclusions.setdefault(row.chrom, []).append((max(0, int(row.start) - padding), int(row.end) + padding))
    return {chrom: _merge_intervals(values) for chrom, values in exclusions.items()}


def _accessibility_intervals(path: str | Path | None) -> dict[str, list[tuple[int, int]]]:
    if path is None:
        return {}
    bed = read_bed(path)
    out: dict[str, list[tuple[int, int]]] = {}
    for row in bed.itertuples(index=False):
        if int(row.end) > int(row.start):
            out.setdefault(row.chrom, []).append((int(row.start), int(row.end)))
    return {chrom: _merge_intervals(values) for chrom, values in out.items()}


def _sample_accessible_interval(
    chrom: str,
    chrom_len: int,
    window_size: int,
    intervals: dict[str, list[tuple[int, int]]],
    rng: np.random.Generator,
) -> tuple[int, int]:
    choices = intervals.get(chrom, [])
    usable = [(start, end) for start, end in choices if end - start >= window_size]
    if not usable:
        return _negative_interval(chrom, chrom_len, window_size, rng)
    start, end = usable[int(rng.integers(0, len(usable)))]
    neg_start = int(rng.integers(start, end - window_size + 1))
    return neg_start, neg_start + window_size


def _sample_matched_negative(
    fasta: Fasta,
    chrom: str,
    chrom_len: int,
    window_size: int,
    positive_gc: float,
    rng: np.random.Generator,
    exclusions: dict[str, list[tuple[int, int]]],
    negative_strategy: str,
    accessibility: dict[str, list[tuple[int, int]]],
    candidate_pool: int,
) -> tuple[int, int, str]:
    best: tuple[float, int, int, str] | None = None
    chrom_exclusions = exclusions.get(chrom, [])
    for _ in range(max(1, candidate_pool)):
        if negative_strategy == "gc_accessibility":
            start, end = _sample_accessible_interval(chrom, chrom_len, window_size, accessibility, rng)
        else:
            start, end = _negative_interval(chrom, chrom_len, window_size, rng)
        if end > chrom_len or _overlaps_any(start, end, chrom_exclusions):
            continue
        seq = _sequence_from_fasta(fasta, chrom, start, end)
        if "N" in seq.upper():
            continue
        if negative_strategy in {"gc", "gc_accessibility"}:
            diff = abs(gc_fraction(seq) - positive_gc)
        else:
            return start, end, seq
        if best is None or diff < best[0]:
            best = (diff, start, end, seq)
    if best is None:
        for _ in range(200):
            start, end = _negative_interval(chrom, chrom_len, window_size, rng)
            if not _overlaps_any(start, end, chrom_exclusions):
                return start, end, _sequence_from_fasta(fasta, chrom, start, end)
        start, end = _negative_interval(chrom, chrom_len, window_size, rng)
        return start, end, _sequence_from_fasta(fasta, chrom, start, end)
    return best[1], best[2], best[3]


def build_sequence_table(
    manifest_path: str | Path,
    downloaded_files: str | Path,
    fasta_by_assembly: dict[str, str | Path],
    out_path: str | Path,
    max_peaks_per_dataset: int = 2000,
    negatives_per_positive: int = 1,
    window_size: int = 211,
    random_seed: int = 13,
    negative_strategy: str = "random",
    accessibility_bed: str | Path | None = None,
    blacklist_bed: str | Path | None = None,
    exclude_padding: int = 500,
    candidate_pool: int = 24,
) -> pd.DataFrame:
    """Build a positive/negative sequence table from peak BED files.

    Supported negative strategies are random, gc, and gc_accessibility. All
    strategies avoid positive intervals from every manifest row plus optional
    blacklist intervals.
    """
    manifest = load_manifest(manifest_path)
    downloads = pd.read_csv(downloaded_files)
    merged = manifest.merge(downloads, on="dataset_id", how="inner")
    rng = np.random.default_rng(random_seed)
    fasta_cache: dict[str, Fasta] = {}
    rows: list[dict[str, object]] = []
    if negative_strategy not in {"random", "gc", "gc_accessibility"}:
        raise ValueError("negative_strategy must be one of random, gc, gc_accessibility")
    positive_exclusions = _positive_exclusions(manifest, downloads, padding=exclude_padding)
    blacklist = _load_exclusion_intervals(blacklist_bed, padding=0)
    for chrom, intervals in blacklist.items():
        positive_exclusions.setdefault(chrom, []).extend(intervals)
        positive_exclusions[chrom] = _merge_intervals(positive_exclusions[chrom])
    accessibility = _accessibility_intervals(accessibility_bed)

    for dataset in merged.itertuples(index=False):
        if getattr(dataset, "status") != "ok":
            continue
        assembly = getattr(dataset, "assembly")
        if assembly not in fasta_by_assembly:
            continue
        if assembly not in fasta_cache:
            fasta_cache[assembly] = Fasta(str(fasta_by_assembly[assembly]), as_raw=True, sequence_always_upper=True)
        fasta = fasta_cache[assembly]
        chrom_lengths = _fasta_lengths(fasta)
        peaks = read_bed(getattr(dataset, "path"))
        peaks = peaks[peaks["chrom"].isin(chrom_lengths)].copy()
        if peaks.empty:
            continue
        if len(peaks) > max_peaks_per_dataset:
            peaks = peaks.sample(max_peaks_per_dataset, random_state=random_seed)

        for j, peak in enumerate(peaks.itertuples(index=False)):
            start, end = _peak_window(peak.start, peak.end, getattr(peak, "summit", np.nan), window_size)
            if end > chrom_lengths[peak.chrom]:
                continue
            seq = _sequence_from_fasta(fasta, peak.chrom, start, end)
            base = {
                "dataset_id": dataset.dataset_id,
                "tf": dataset.tf,
                "tf_family": TF_FAMILY.get(str(dataset.tf).upper(), ""),
                "species": dataset.species,
                "assembly": dataset.assembly,
                "assay": dataset.assay,
                "lab": dataset.lab,
                "biosample": dataset.biosample,
                "cell_type": dataset.cell_type,
                "negative_strategy": negative_strategy,
            }
            rows.append(
                {
                    **base,
                    "example_id": f"{dataset.dataset_id}_pos_{j}",
                    "chrom": peak.chrom,
                    "start": start,
                    "end": end,
                    "label": 1,
                    "sequence": seq,
                    "source_width": int(peak.end) - int(peak.start),
                    "source_score": getattr(peak, "score", np.nan),
                }
            )
            for k in range(negatives_per_positive):
                neg_start, neg_end, neg_seq = _sample_matched_negative(
                    fasta=fasta,
                    chrom=peak.chrom,
                    chrom_len=chrom_lengths[peak.chrom],
                    window_size=window_size,
                    positive_gc=gc_fraction(seq),
                    rng=rng,
                    exclusions=positive_exclusions,
                    negative_strategy=negative_strategy,
                    accessibility=accessibility,
                    candidate_pool=candidate_pool,
                )
                rows.append(
                    {
                        **base,
                        "example_id": f"{dataset.dataset_id}_neg_{j}_{k}",
                        "chrom": peak.chrom,
                        "start": neg_start,
                        "end": neg_end,
                        "label": 0,
                        "sequence": neg_seq,
                    }
                )

    frame = add_sequence_stats(pd.DataFrame(rows)) if rows else pd.DataFrame()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".parquet":
        frame.to_parquet(out, index=False)
    else:
        frame.to_csv(out, index=False)
    return frame
