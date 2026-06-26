from __future__ import annotations

from pathlib import Path

import pandas as pd


def _fmt(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.3f}"


def write_preliminary_summary(results: pd.DataFrame, selective: pd.DataFrame, path: str | Path) -> None:
    key_models = ["gc_artifact_logreg", "kmer_logreg", "kmer_metadata_logreg"]
    rows = []
    for split in ["iid", "assay_heldout_cutrun", "lab_heldout_henikoff"]:
        for model in key_models:
            row = results[
                (results["split"].eq(split))
                & (results["model"].eq(model))
                & (~results["calibrated"])
            ]
            if row.empty:
                continue
            r = row.iloc[0]
            rows.append(
                f"| {split} | {model} | {_fmt(r.auprc)} | {_fmt(r.auroc)} | {_fmt(r.ece)} | {_fmt(r.brier)} | {_fmt(r.worst_group_auprc)} |"
            )

    sel = selective[
        (selective["split"].eq("assay_heldout_cutrun"))
        & (selective["model"].eq("kmer_logreg"))
        & (selective["calibrated"])
        & (selective["coverage"].isin([1.0, 0.6, 0.4, 0.2]))
    ]
    sel_rows = [
        f"| {_fmt(r.coverage)} | {_fmt(r.auprc)} | {_fmt(r.ece)} | {_fmt(r.brier)} |"
        for r in sel.itertuples()
    ]

    text = "\n".join(
        [
            "# Preliminary Demo Results",
            "",
            "These are deterministic stress-test results, not final biological claims. The demo plants a transferable motif signal plus a protocol-specific GC artifact so the analysis can verify the paper's intended failure mode before running on public peak-derived data.",
            "",
            "| Split | Model | AUPRC | AUROC | ECE | Brier | Worst-Group AUPRC |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "## Calibrated Selective Prediction: Assay-Held-Out CUT&RUN",
            "",
            "| Coverage | AUPRC | ECE | Brier |",
            "|---:|---:|---:|---:|",
            *sel_rows,
            "",
            "Interpretation: if the public-data run shows the same pattern, the paper can claim that IID ranking metrics hide assay/lab shortcuts, and that calibrated selective prediction improves the retained subset at moderate coverage. The lowest-coverage regime should be paired with explicit OOD scoring before making a stronger reliability claim. Until then, this table is a reproducibility and analysis-contract check.",
            "",
        ]
    )
    Path(path).write_text(text, encoding="utf-8")


def _ci_fmt(
    bootstrap_cis: pd.DataFrame,
    split: object,
    model: object,
    calibrated: object,
    metric: str,
    fallback: float,
) -> str:
    if bootstrap_cis.empty:
        return _fmt(fallback)
    row = bootstrap_cis[
        (bootstrap_cis["split"].eq(split))
        & (bootstrap_cis["model"].eq(model))
        & (bootstrap_cis["calibrated"].eq(calibrated))
        & (bootstrap_cis["metric"].eq(metric))
    ]
    if row.empty:
        return _fmt(fallback)
    r = row.iloc[0]
    if pd.isna(r.ci_low) or pd.isna(r.ci_high):
        return _fmt(fallback)
    return f"{_fmt(fallback)} [{_fmt(r.ci_low)}, {_fmt(r.ci_high)}]"


def write_real_preliminary_summary(
    results: pd.DataFrame,
    selective: pd.DataFrame,
    bootstrap_cis: pd.DataFrame,
    split_counts: pd.DataFrame,
    path: str | Path,
    source_table: str | Path | None = None,
) -> None:
    rows = []
    for r in results[~results["calibrated"]].itertuples(index=False):
        rows.append(
            f"| {r.split} | {r.model} | "
            f"{_ci_fmt(bootstrap_cis, r.split, r.model, r.calibrated, 'auprc', r.auprc)} | "
            f"{_ci_fmt(bootstrap_cis, r.split, r.model, r.calibrated, 'auroc', r.auroc)} | "
            f"{_ci_fmt(bootstrap_cis, r.split, r.model, r.calibrated, 'ece', r.ece)} | "
            f"{_ci_fmt(bootstrap_cis, r.split, r.model, r.calibrated, 'brier', r.brier)} | "
            f"{_fmt(r.worst_group_auprc)} |"
        )

    count_rows = []
    for r in split_counts.itertuples(index=False):
        count_rows.append(
            f"| {r.split} | {r.split_type} | {r.holdout} | {r.partition} | {int(r.n)} | {int(r.positive)} | {int(r.negative)} | {_fmt(r.prevalence)} |"
        )

    selected_model = None
    if not selective.empty:
        selected_model = "kmer_metadata_logreg" if "kmer_metadata_logreg" in set(selective["model"]) else str(selective["model"].iloc[0])
    sel = selective[
        (selective["model"].eq(selected_model))
        & (selective["calibrated"])
        & (selective["coverage"].isin([1.0, 0.8, 0.6, 0.4, 0.2]))
    ] if selected_model else pd.DataFrame()
    sel_rows = [
        f"| {r.split} | {r.model} | {_fmt(r.coverage)} | {_fmt(r.auprc)} | {_fmt(r.ece)} | {_fmt(r.brier)} |"
        for r in sel.itertuples(index=False)
    ]

    source = f"`{source_table}`" if source_table is not None else "the supplied window table"
    text = "\n".join(
        [
            "# Preliminary Real-Data Evaluation",
            "",
            f"Source table: {source}",
            "",
            "This report summarizes baseline evaluation on the supplied real-data window table. Metrics are preliminary integration outputs: they verify the evaluation contract and should be interpreted with the data provenance, negative sampling, and peak-processing choices used to build the table.",
            "",
            "Intervals are percentile bootstrap confidence intervals over test-set windows.",
            "",
            "## Test Metrics",
            "",
            "| Split | Model | AUPRC | AUROC | ECE | Brier | Worst-Group AUPRC |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "## Split Counts",
            "",
            "| Split | Type | Holdout | Partition | N | Positive | Negative | Prevalence |",
            "|---|---|---|---|---:|---:|---:|---:|",
            *count_rows,
            "",
            "## Calibrated Selective Prediction",
            "",
            "| Split | Model | Coverage | AUPRC | ECE | Brier |",
            "|---|---:|---:|---:|---:|---:|",
            *sel_rows,
            "",
            "Interpretation guardrail: these outputs do not by themselves establish a biological protocol-shift claim. They are suitable for checking that real prebuilt windows can flow through the split-aware baselines, calibration, selective prediction, bootstrap uncertainty, and reporting stack.",
            "",
        ]
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


def write_real_summary(
    results: pd.DataFrame,
    ci: pd.DataFrame,
    split_counts: pd.DataFrame,
    path: str | Path,
) -> None:
    key = results.merge(ci, on=["split", "model", "calibrated"], how="left")
    key = key[~key["calibrated"]].copy()
    rows = []
    for r in key.itertuples(index=False):
        rows.append(
            "| {split} | {model} | {n:.0f} | {prev} | {auprc} [{lo}, {hi}] | {auroc} | {ece} | {brier} | {worst} |".format(
                split=r.split,
                model=r.model,
                n=r.n,
                prev=_fmt(r.prevalence),
                auprc=_fmt(r.auprc),
                lo=_fmt(getattr(r, "auprc_lo", float("nan"))),
                hi=_fmt(getattr(r, "auprc_hi", float("nan"))),
                auroc=_fmt(r.auroc),
                ece=_fmt(r.ece),
                brier=_fmt(r.brier),
                worst=_fmt(r.worst_group_auprc),
            )
        )

    count_rows = []
    for r in split_counts.head(30).itertuples(index=False):
        count_rows.append(f"| {r.split_name} | {r.split} | {r.assay} | {r.tf} | {r.label} | {r.n} |")

    text = "\n".join(
        [
            "# Real-Data Preliminary Results",
            "",
            "These results are generated from public peak-derived sequence windows. They are suitable for the abstract only after the corresponding window table, manifests, and commands are preserved with the submission artifact.",
            "",
            "| Split | Model | Test n | Prevalence | AUPRC [95% CI] | AUROC | ECE | Brier | Worst-Group AUPRC |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "## Split Counts Preview",
            "",
            "| Split name | Partition | Assay | TF | Label | n |",
            "|---|---|---|---|---:|---:|",
            *count_rows,
            "",
            "Caveat: assay-shift claims require called CUT&RUN peaks or otherwise harmonized labels. Do not cite fragment BEDs as final peaks.",
            "",
        ]
    )
    Path(path).write_text(text, encoding="utf-8")
