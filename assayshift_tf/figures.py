from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_demo_results(results: pd.DataFrame, selective: pd.DataFrame, path: str | Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    base = results[~results["calibrated"]].copy()
    cal = results[results["calibrated"]].copy()
    label_map = {
        "iid": "IID",
        "assay_heldout_cutrun": "Assay\nCUT&RUN",
        "lab_heldout_henikoff": "Lab\nHenikoff",
        "species_heldout_mouse": "Species\nMouse",
        "family_heldout_zinc_finger": "Family\nZinc finger",
    }
    base["split_label"] = base["split"].map(label_map).fillna(base["split"])
    cal["split_label"] = cal["split"].map(label_map).fillna(cal["split"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    sns.barplot(data=base, x="split_label", y="auprc", hue="model", ax=axes[0])
    axes[0].set_title("Ranking Under Shift")
    axes[0].set_ylabel("AUPRC")
    axes[0].set_xlabel("")

    sns.barplot(data=cal, x="split_label", y="ece", hue="model", ax=axes[1])
    axes[1].set_title("Calibration After Platt Scaling")
    axes[1].set_ylabel("ECE")
    axes[1].set_xlabel("")

    subset = selective[
        (selective["split"].isin(["iid", "assay_heldout_cutrun"]))
        & (selective["model"].eq("kmer_logreg"))
        & (selective["calibrated"])
    ]
    sns.lineplot(data=subset, x="coverage", y="auprc", hue="split", marker="o", ax=axes[2])
    axes[2].invert_xaxis()
    axes[2].set_title("Selective Prediction")
    axes[2].set_ylabel("AUPRC")
    axes[2].set_xlabel("Coverage kept")

    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes[:2]:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    line_legend = axes[2].get_legend()
    if line_legend is not None:
        line_legend.set_title("")
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _split_label(value: object) -> str:
    text = str(value).replace("_heldout", " held-out").replace("_", " ")
    if len(text) > 16:
        return text.replace(" held-out ", "\nheld-out ")
    return text


def _save_empty_figure(path: str | Path, message: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_real_results(results: pd.DataFrame, selective: pd.DataFrame, path: str | Path) -> None:
    """Plot real-data evaluation outputs without demo-specific labels."""
    if results.empty:
        _save_empty_figure(path, "No real-data evaluation results available")
        return

    sns.set_theme(style="whitegrid", context="notebook")
    base = results[~results["calibrated"]].copy()
    cal = results[results["calibrated"]].copy()
    if base.empty:
        _save_empty_figure(path, "No uncalibrated real-data evaluation results available")
        return

    split_order = list(dict.fromkeys(results["split"].astype(str)))[:6]
    label_map = {split: _split_label(split) for split in split_order}
    base = base[base["split"].astype(str).isin(split_order)].copy()
    cal = cal[cal["split"].astype(str).isin(split_order)].copy()
    base["split_label"] = base["split"].astype(str).map(label_map)
    cal["split_label"] = cal["split"].astype(str).map(label_map)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    sns.barplot(data=base, x="split_label", y="auprc", hue="model", order=list(label_map.values()), ax=axes[0])
    axes[0].set_title("Real-Data Ranking Under Shift")
    axes[0].set_ylabel("AUPRC")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=20)

    sns.barplot(data=cal, x="split_label", y="ece", hue="model", order=list(label_map.values()), ax=axes[1])
    axes[1].set_title("Real-Data Calibration")
    axes[1].set_ylabel("ECE")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=20)

    if selective.empty:
        axes[2].text(0.5, 0.5, "No selective metrics", ha="center", va="center", transform=axes[2].transAxes)
    else:
        preferred_split = "assay_heldout_cutrun" if "assay_heldout_cutrun" in set(selective["split"].astype(str)) else split_order[0]
        subset = selective[selective["split"].astype(str).eq(preferred_split) & selective["calibrated"]].copy()
        if "kmer_logreg" in set(subset.get("model", [])):
            model_subset = subset[subset["model"].eq("kmer_logreg")]
            if not model_subset.empty:
                subset = model_subset
        if not subset.empty:
            sns.lineplot(data=subset, x="coverage", y="auprc", hue="model", marker="o", ax=axes[2])
            axes[2].invert_xaxis()
        else:
            axes[2].text(0.5, 0.5, "No calibrated selective metrics", ha="center", va="center", transform=axes[2].transAxes)
    if selective.empty:
        preferred_split = "selected split"
    axes[2].set_title(f"Selective Prediction: {_split_label(preferred_split)}")
    axes[2].set_ylabel("AUPRC")
    axes[2].set_xlabel("Coverage kept")

    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes[:2]:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    line_legend = axes[2].get_legend()
    if line_legend is not None:
        line_legend.set_title("")
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(3, len(labels)), frameon=False, bbox_to_anchor=(0.5, -0.04))

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _calibration_bins(frame: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    rows = []
    edges = pd.interval_range(start=0.0, end=1.0, periods=n_bins, closed="right")
    work = frame.copy()
    work["bin"] = pd.cut(work["prob"].clip(0.0, 1.0), bins=n_bins, labels=False, include_lowest=True)
    for (split, model, calibrated, bin_id), group in work.groupby(
        ["split_name", "model", "calibrated", "bin"], dropna=False
    ):
        if pd.isna(bin_id) or group.empty:
            continue
        rows.append(
            {
                "split": split,
                "model": model,
                "calibrated": bool(calibrated),
                "bin": int(bin_id),
                "confidence": float(group["prob"].mean()),
                "accuracy": float(group["label"].mean()),
                "n": int(len(group)),
                "interval": str(edges[int(bin_id)]) if int(bin_id) < len(edges) else "",
            }
        )
    return pd.DataFrame(rows)


def plot_reliability_curves(predictions: pd.DataFrame, path: str | Path, max_models: int = 5) -> None:
    """Plot calibration curves before and after Platt scaling for real-data predictions."""
    if predictions.empty:
        _save_empty_figure(path, "No prediction rows available for reliability curves")
        return

    sns.set_theme(style="whitegrid", context="notebook")
    preferred_splits = [
        split
        for split in ["assay_heldout_cutrun", "lab_heldout_haib", "family_heldout_zinc_finger", "iid"]
        if split in set(predictions["split_name"].astype(str))
    ]
    if not preferred_splits:
        preferred_splits = list(dict.fromkeys(predictions["split_name"].astype(str)))[:2]
    preferred_splits = preferred_splits[:2]
    models = list(dict.fromkeys(predictions["model"].astype(str)))[:max_models]
    work = predictions[
        predictions["split_name"].astype(str).isin(preferred_splits) & predictions["model"].astype(str).isin(models)
    ].copy()
    curves = _calibration_bins(work)
    if curves.empty:
        _save_empty_figure(path, "No populated calibration bins")
        return

    fig, axes = plt.subplots(1, len(preferred_splits), figsize=(6.2 * len(preferred_splits), 5), squeeze=False)
    axes = axes[0]
    for ax, split in zip(axes, preferred_splits):
        subset = curves[curves["split"].astype(str).eq(split)]
        sns.lineplot(
            data=subset,
            x="confidence",
            y="accuracy",
            hue="model",
            style="calibrated",
            markers=True,
            dashes=True,
            ax=ax,
        )
        ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"Reliability: {_split_label(split)}")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed positive rate")
        legend = ax.get_legend()
        if legend is not None:
            legend.set_title("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_picard_schematic(path: str | Path) -> None:
    """Draw a compact PICARD-TF model schematic for paper drafts."""
    sns.set_theme(style="white", context="notebook")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axis("off")

    boxes = {
        "seq": (0.05, 0.62, 0.18, 0.18, "DNA window\none-hot"),
        "cnn": (0.31, 0.62, 0.2, 0.18, "sequence CNN\nbiological latent"),
        "tf": (0.05, 0.28, 0.18, 0.18, "TF + family\nembeddings"),
        "bio": (0.58, 0.62, 0.2, 0.18, "binding head\nlatent occupancy"),
        "protocol": (0.31, 0.28, 0.2, 0.18, "assay/lab/species\nprotocol branch"),
        "guard": (0.58, 0.28, 0.2, 0.18, "counterfactual guard\nmetadata masking"),
        "out": (0.83, 0.45, 0.14, 0.18, "calibrated\nprobability"),
    }
    colors = {
        "seq": "#E8F1FA",
        "cnn": "#D7E8D5",
        "tf": "#F3E6C8",
        "bio": "#D7E8D5",
        "protocol": "#F5D6C6",
        "guard": "#EFE2F3",
        "out": "#D9E6F2",
    }
    for key, (x, y, w, h, label) in boxes.items():
        ax.add_patch(
            plt.Rectangle((x, y), w, h, facecolor=colors[key], edgecolor="#30343B", linewidth=1.2, joinstyle="round")
        )
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11)

    def arrow(src: str, dst: str, yoff_src: float = 0.5, yoff_dst: float = 0.5) -> None:
        sx, sy, sw, sh, _ = boxes[src]
        dx, dy, _, dh, _ = boxes[dst]
        ax.annotate(
            "",
            xy=(dx, dy + dh * yoff_dst),
            xytext=(sx + sw, sy + sh * yoff_src),
            arrowprops=dict(arrowstyle="->", lw=1.4, color="#30343B"),
        )

    arrow("seq", "cnn")
    arrow("tf", "bio", 0.65, 0.35)
    arrow("cnn", "bio")
    arrow("bio", "out")
    arrow("protocol", "guard")
    arrow("guard", "out")
    arrow("protocol", "out", 0.7, 0.25)
    ax.text(
        0.44,
        0.12,
        "Training losses: BCE(label) + counterfactual consistency + metadata residual penalty + optional assay/lab adversary",
        ha="center",
        va="center",
        fontsize=10,
        color="#30343B",
    )
    ax.text(0.5, 0.93, "PICARD-TF / AxisGuard-CNN", ha="center", va="center", fontsize=15, weight="bold")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
