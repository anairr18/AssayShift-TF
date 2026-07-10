from __future__ import annotations

import argparse
from pathlib import Path

from assayshift_tf.benchmark import (
    _prepare_window_frame,
    _read_window_table,
    filter_window_frame,
    parse_split_spec,
    run_demo,
    run_real_data_evaluation,
    run_real_seed_sweep,
)
from assayshift_tf.embeddings import write_hf_embedding_cache
from assayshift_tf.manifest import load_manifest, validate_manifest
from assayshift_tf.models import model_spec_from_name
from assayshift_tf.peaks import build_sequence_table, call_cutrun_peaks, download_peak_files, write_cutrun_peaks
from assayshift_tf.real_eval import run_real


def _cmd_demo(args: argparse.Namespace) -> int:
    paths = run_demo(args.n, args.out, args.figures, random_seed=args.seed)
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


def _model_specs_from_args(args: argparse.Namespace, seed: int):
    if not args.model:
        return None
    return [
        model_spec_from_name(
            model,
            deep_epochs=args.deep_epochs,
            deep_batch_size=args.deep_batch_size,
            deep_lr=args.deep_lr,
            deep_device=args.deep_device,
            axis_dropout=args.axis_dropout,
            counterfactual_mode=args.counterfactual_mode,
            counterfactual_weight=args.counterfactual_weight,
            metadata_residual_weight=args.metadata_residual_weight,
            adversarial_weight=args.adversarial_weight,
            deep_objective=args.deep_objective,
            group_key=args.group_key,
            groupdro_eta=args.groupdro_eta,
            protocol_penalty=args.protocol_penalty,
            protocol_penalty_weight=args.protocol_penalty_weight,
            rc_augment=args.rc_augment,
            rc_ensemble=args.rc_ensemble,
            embedding_cache=args.embedding_cache,
            embedding_head=args.embedding_head,
            embedding_include_metadata=args.embedding_include_metadata,
            random_state=seed,
        )
        for model in args.model
    ]


def _cmd_evaluate_real(args: argparse.Namespace) -> int:
    model_specs = _model_specs_from_args(args, args.seed)
    paths = run_real_data_evaluation(
        args.window_table,
        args.out,
        args.figures,
        prefix=args.prefix,
        split_specs=args.split,
        model_specs=model_specs,
        random_seed=args.seed,
        bootstrap_iterations=args.bootstrap,
        ci_confidence=args.confidence,
        calibration_method=args.calibration,
        calibration_group=args.calibration_group,
        drop_all_n=args.drop_all_n,
        drop_duplicate_sequences=args.drop_duplicate_sequences,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


def _cmd_sweep_real(args: argparse.Namespace) -> int:
    seeds = args.seed or [13, 17, 23, 29, 31]
    paths = run_real_seed_sweep(
        args.window_table,
        args.out,
        prefix=args.prefix,
        split_specs=args.split,
        model_specs=_model_specs_from_args(args, seeds[0]),
        seeds=seeds,
        bootstrap_iterations=args.bootstrap,
        ci_confidence=args.confidence,
        calibration_method=args.calibration,
        calibration_group=args.calibration_group,
        drop_all_n=args.drop_all_n,
        drop_duplicate_sequences=args.drop_duplicate_sequences,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


def _cmd_embed(args: argparse.Namespace) -> int:
    filter_artifacts = filter_window_frame(
        _prepare_window_frame(_read_window_table(args.window_table)),
        drop_all_n=args.drop_all_n,
        drop_duplicate_sequences=args.drop_duplicate_sequences,
    )
    path = write_hf_embedding_cache(
        filter_artifacts.frame,
        args.out,
        hf_model=args.hf_model,
        batch_size=args.batch_size,
        device=args.device,
        max_tokens=args.max_tokens,
        pooling=args.pooling,
        trust_remote_code=args.trust_remote_code,
        limit=args.limit,
    )
    print(f"embedding_cache: {path}")
    return 0


def _cmd_validate_manifest(args: argparse.Namespace) -> int:
    frame = load_manifest(args.path)
    messages = validate_manifest(frame)
    if messages:
        print("Manifest validation issues:")
        for message in messages:
            print(f"- {message}")
        return 1 if args.strict else 0
    print(f"Manifest OK: {args.path} ({len(frame)} rows)")
    return 0


def _cmd_download_peaks(args: argparse.Namespace) -> int:
    downloads = download_peak_files(args.manifest, args.out)
    args.index.parent.mkdir(parents=True, exist_ok=True)
    downloads.to_csv(args.index, index=False)
    print(f"Wrote download index: {args.index}")
    print(downloads["status"].value_counts(dropna=False).to_string())
    return 0


def _parse_fasta_args(values: list[str]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--fasta values must look like ASSEMBLY=path.fa")
        assembly, path = value.split("=", 1)
        mapping[assembly] = Path(path)
    return mapping


def _cmd_build_windows(args: argparse.Namespace) -> int:
    fasta_by_assembly = _parse_fasta_args(args.fasta)
    frame = build_sequence_table(
        manifest_path=args.manifest,
        downloaded_files=args.download_index,
        fasta_by_assembly=fasta_by_assembly,
        out_path=args.out,
        max_peaks_per_dataset=args.max_peaks_per_dataset,
        negatives_per_positive=args.negatives_per_positive,
        window_size=args.window_size,
        random_seed=args.seed,
        negative_strategy=args.negative_strategy,
        accessibility_bed=args.accessibility_bed,
        blacklist_bed=args.blacklist_bed,
        exclude_padding=args.exclude_padding,
        candidate_pool=args.candidate_pool,
    )
    print(f"Wrote sequence table: {args.out} ({len(frame)} rows)")
    return 0


def _cmd_run_real(args: argparse.Namespace) -> int:
    paths = run_real(
        windows_path=args.windows,
        out_prefix=args.out_prefix,
        figures_dir=args.figures,
        random_state=args.seed,
        n_bootstrap=args.bootstrap,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


def _cmd_call_cutrun_peaks(args: argparse.Namespace) -> int:
    peaks = call_cutrun_peaks(
        args.fragments,
        control_fragments=args.control,
        source=args.source,
        max_gap=args.max_gap,
        min_fragments=args.min_fragments,
        peak_padding=args.peak_padding,
    )
    write_cutrun_peaks(peaks, args.out, include_header=args.header)
    print(f"Wrote CUT&RUN peaks: {args.out} ({len(peaks)} peaks)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="assayshift-tf")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="Run deterministic protocol-shift demo benchmark.")
    demo.add_argument("--n", type=int, default=6000, help="Number of synthetic windows.")
    demo.add_argument("--seed", type=int, default=13)
    demo.add_argument("--out", type=Path, default=Path("reports"))
    demo.add_argument("--figures", type=Path, default=Path("figures"))
    demo.set_defaults(func=_cmd_demo)

    real = sub.add_parser("evaluate-real", help="Evaluate a prebuilt real-data window table.")
    real.add_argument("window_table", type=Path, help="CSV, TSV, or Parquet table with label and sequence columns.")
    real.add_argument("--out", type=Path, default=Path("reports"))
    real.add_argument("--figures", type=Path, default=Path("figures"))
    real.add_argument("--prefix", default="real", help="Prefix for output files.")
    real.add_argument(
        "--split",
        action="append",
        type=parse_split_spec,
        help="Named split spec: iid, name:type, name:type:holdout, or name=type:holdout. May be repeated.",
    )
    real.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap replicates for metric CIs.")
    real.add_argument("--confidence", type=float, default=0.95, help="Bootstrap CI confidence level.")
    real.add_argument("--calibration", choices=["platt", "protocol_platt"], default="platt")
    real.add_argument("--calibration-group", default="auto", help="Group column for protocol_platt, or auto.")
    real.add_argument("--seed", type=int, default=13)
    real.add_argument(
        "--model",
        action="append",
        help=(
            "Model to run; repeatable. Choices include gc, kmer, kmer_metadata, tiny_cnn, "
            "axis_guard_cnn, axis_guard_no_cf, axis_guard_no_resid, axis_guard_no_adv, axis_guard_full, "
            "embedding_head, embedding_logreg, embedding_mlp, embedding_metadata. "
            "Defaults to the fast logistic baselines."
        ),
    )
    real.add_argument("--deep-epochs", type=int, default=None, help="Training epochs for deep models.")
    real.add_argument("--deep-batch-size", type=int, default=None, help="Batch size for deep models.")
    real.add_argument("--deep-lr", type=float, default=None, help="Learning rate for deep models.")
    real.add_argument("--deep-device", default=None, help="Deep model device: auto, cpu, or cuda.")
    real.add_argument("--axis-dropout", type=float, default=None, help="Protocol metadata dropout for axis_guard_cnn.")
    real.add_argument(
        "--counterfactual-mode",
        choices=["mask", "shuffle", "mask_or_shuffle"],
        default=None,
        help="Counterfactual protocol perturbation for axis_guard_cnn.",
    )
    real.add_argument(
        "--counterfactual-weight",
        type=float,
        default=None,
        help="Prediction-consistency penalty weight for counterfactual protocol perturbation.",
    )
    real.add_argument(
        "--metadata-residual-weight",
        type=float,
        default=None,
        help="Penalty weight for protocol-branch residual logits in axis_guard_cnn.",
    )
    real.add_argument(
        "--adversarial-weight",
        type=float,
        default=None,
        help="Optional gradient-reversal assay/lab adversary weight for axis_guard_cnn.",
    )
    real.add_argument("--deep-objective", choices=["erm", "groupdro"], default=None)
    real.add_argument("--group-key", default=None, help="Group column for GroupDRO/alignment, or protocol.")
    real.add_argument("--groupdro-eta", type=float, default=None, help="Exponentiated-gradient step for GroupDRO.")
    real.add_argument("--protocol-penalty", choices=["none", "coral", "mmd"], default=None)
    real.add_argument("--protocol-penalty-weight", type=float, default=None)
    real.add_argument("--rc-augment", action="store_true", help="Augment deep-model training rows with reverse complements.")
    real.add_argument("--rc-ensemble", action="store_true", help="Average deep-model predictions with reverse complements.")
    real.add_argument("--embedding-cache", type=Path, default=None, help="NPZ cache for embedding_head models.")
    real.add_argument("--embedding-head", choices=["logreg", "mlp"], default=None)
    real.add_argument(
        "--embedding-include-metadata",
        action="store_true",
        default=None,
        help="Append safe metadata features to frozen embeddings for embedding_head models.",
    )
    real.add_argument("--drop-all-n", action="store_true", help="Drop windows whose sequence is entirely N bases.")
    real.add_argument(
        "--drop-duplicate-sequences",
        action="store_true",
        help="Drop exact duplicate sequence strings before splitting to reduce sequence leakage.",
    )
    real.set_defaults(func=_cmd_evaluate_real)

    sweep = sub.add_parser("sweep-real", help="Run repeated-seed real-data evaluations and aggregate metrics.")
    sweep.add_argument("window_table", type=Path, help="CSV, TSV, or Parquet table with label and sequence columns.")
    sweep.add_argument("--out", type=Path, default=Path("reports"))
    sweep.add_argument("--prefix", default="real_seed_sweep", help="Prefix for sweep output files.")
    sweep.add_argument(
        "--split",
        action="append",
        type=parse_split_spec,
        help="Named split spec: iid, name:type, name:type:holdout, or name=type:holdout. May be repeated.",
    )
    sweep.add_argument("--seed", action="append", type=int, help="Seed to run; repeatable. Defaults to 5 seeds.")
    sweep.add_argument("--bootstrap", type=int, default=0, help="Bootstrap replicates per seed; usually 0 for sweeps.")
    sweep.add_argument("--confidence", type=float, default=0.95, help="Bootstrap CI confidence level.")
    sweep.add_argument("--calibration", choices=["platt", "protocol_platt"], default="platt")
    sweep.add_argument("--calibration-group", default="auto", help="Group column for protocol_platt, or auto.")
    sweep.add_argument(
        "--model",
        action="append",
        help=(
            "Model to run; repeatable. For ablations use tiny_cnn, axis_guard_no_cf, "
            "axis_guard_no_resid, axis_guard_no_adv, axis_guard_full."
        ),
    )
    sweep.add_argument("--deep-epochs", type=int, default=None, help="Training epochs for deep models.")
    sweep.add_argument("--deep-batch-size", type=int, default=None, help="Batch size for deep models.")
    sweep.add_argument("--deep-lr", type=float, default=None, help="Learning rate for deep models.")
    sweep.add_argument("--deep-device", default=None, help="Deep model device: auto, cpu, or cuda.")
    sweep.add_argument("--axis-dropout", type=float, default=None, help="Protocol metadata dropout for axis_guard_cnn.")
    sweep.add_argument(
        "--counterfactual-mode",
        choices=["mask", "shuffle", "mask_or_shuffle"],
        default=None,
        help="Counterfactual protocol perturbation for axis_guard_cnn.",
    )
    sweep.add_argument("--counterfactual-weight", type=float, default=None)
    sweep.add_argument("--metadata-residual-weight", type=float, default=None)
    sweep.add_argument("--adversarial-weight", type=float, default=None)
    sweep.add_argument("--deep-objective", choices=["erm", "groupdro"], default=None)
    sweep.add_argument("--group-key", default=None, help="Group column for GroupDRO/alignment, or protocol.")
    sweep.add_argument("--groupdro-eta", type=float, default=None, help="Exponentiated-gradient step for GroupDRO.")
    sweep.add_argument("--protocol-penalty", choices=["none", "coral", "mmd"], default=None)
    sweep.add_argument("--protocol-penalty-weight", type=float, default=None)
    sweep.add_argument("--rc-augment", action="store_true", help="Augment deep-model training rows with reverse complements.")
    sweep.add_argument("--rc-ensemble", action="store_true", help="Average deep-model predictions with reverse complements.")
    sweep.add_argument("--embedding-cache", type=Path, default=None, help="NPZ cache for embedding_head models.")
    sweep.add_argument("--embedding-head", choices=["logreg", "mlp"], default=None)
    sweep.add_argument(
        "--embedding-include-metadata",
        action="store_true",
        default=None,
        help="Append safe metadata features to frozen embeddings for embedding_head models.",
    )
    sweep.add_argument("--drop-all-n", action="store_true", help="Drop windows whose sequence is entirely N bases.")
    sweep.add_argument(
        "--drop-duplicate-sequences",
        action="store_true",
        help="Drop exact duplicate sequence strings before splitting to reduce sequence leakage.",
    )
    sweep.set_defaults(func=_cmd_sweep_real)

    embed = sub.add_parser("embed", help="Cache frozen HF DNA sequence embeddings keyed by example_id.")
    embed.add_argument("window_table", type=Path, help="CSV, TSV, or Parquet table with example_id and sequence columns.")
    embed.add_argument("--out", type=Path, required=True, help="Output .npz embedding cache.")
    embed.add_argument("--hf-model", required=True, help="HF encoder/model id.")
    embed.add_argument("--batch-size", type=int, default=32)
    embed.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    embed.add_argument("--max-tokens", type=int, default=256)
    embed.add_argument("--pooling", choices=["mean", "cls", "max"], default="mean")
    embed.add_argument("--trust-remote-code", action="store_true")
    embed.add_argument("--limit", type=int, default=None, help="Optional first-N rows smoke-test limit.")
    embed.add_argument("--drop-all-n", action="store_true", help="Drop windows whose sequence is entirely N bases.")
    embed.add_argument("--drop-duplicate-sequences", action="store_true")
    embed.set_defaults(func=_cmd_embed)

    manifest = sub.add_parser("validate-manifest", help="Validate dataset manifest columns.")
    manifest.add_argument("path", type=Path)
    manifest.add_argument("--strict", action="store_true", help="Return nonzero on warnings.")
    manifest.set_defaults(func=_cmd_validate_manifest)

    download = sub.add_parser("download-peaks", help="Download processed peak files in a manifest.")
    download.add_argument("manifest", type=Path)
    download.add_argument("--out", type=Path, default=Path("data/raw/peaks"))
    download.add_argument("--index", type=Path, default=Path("data/raw/download_index.csv"))
    download.set_defaults(func=_cmd_download_peaks)

    windows = sub.add_parser("build-windows", help="Build FASTA-backed positive/negative sequence windows.")
    windows.add_argument("manifest", type=Path)
    windows.add_argument("--download-index", type=Path, default=Path("data/raw/download_index.csv"))
    windows.add_argument("--fasta", action="append", required=True, help="Assembly FASTA mapping, e.g. GRCh38=hg38.fa")
    windows.add_argument("--out", type=Path, default=Path("data/processed/windows.parquet"))
    windows.add_argument("--max-peaks-per-dataset", type=int, default=2000)
    windows.add_argument("--negatives-per-positive", type=int, default=1)
    windows.add_argument("--window-size", type=int, default=211)
    windows.add_argument("--seed", type=int, default=13)
    windows.add_argument("--negative-strategy", choices=["random", "gc", "gc_accessibility"], default="random")
    windows.add_argument("--accessibility-bed", type=Path)
    windows.add_argument("--blacklist-bed", type=Path)
    windows.add_argument("--exclude-padding", type=int, default=500)
    windows.add_argument("--candidate-pool", type=int, default=24)
    windows.set_defaults(func=_cmd_build_windows)

    cutrun = sub.add_parser("call-cutrun-peaks", help="Call deterministic peaks from CUT&RUN fragment BED.")
    cutrun.add_argument("fragments", type=Path)
    cutrun.add_argument("--control", type=Path, default=None, help="Optional no-antibody/control fragment BED.")
    cutrun.add_argument("--out", type=Path, required=True)
    cutrun.add_argument("--source", default="CUT&RUN")
    cutrun.add_argument("--max-gap", type=int, default=75, help="Maximum gap between midpoint bins in one cluster.")
    cutrun.add_argument("--min-fragments", type=int, default=2, help="Minimum fragment support for a called peak.")
    cutrun.add_argument("--peak-padding", type=int, default=50, help="Bases to pad around clustered midpoint span.")
    cutrun.add_argument("--header", action="store_true", help="Include a header row in the BED-like output.")
    cutrun.set_defaults(func=_cmd_call_cutrun_peaks)

    real = sub.add_parser("run-real", help="Evaluate baselines on a real window table.")
    real.add_argument("windows", type=Path)
    real.add_argument("--out-prefix", type=Path, default=Path("reports/real"))
    real.add_argument("--figures", type=Path, default=Path("figures"))
    real.add_argument("--bootstrap", type=int, default=100)
    real.add_argument("--seed", type=int, default=13)
    real.set_defaults(func=_cmd_run_real)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
