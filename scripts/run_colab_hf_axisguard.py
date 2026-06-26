from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

from assayshift_tf.benchmark import (
    _mask_leaky_metadata,
    _prepare_window_frame,
    _read_window_table,
    filter_window_frame,
    parse_split_spec,
)
from assayshift_tf.deep import BIOLOGICAL_CATEGORICAL_COLUMNS, NUMERIC_METADATA_COLUMNS, PROTOCOL_CATEGORICAL_COLUMNS
from assayshift_tf.metrics import PlattCalibrator, expected_calibration_error, selective_metrics
from assayshift_tf.splits import make_split


RC_TRANS = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def ensure_nucleotide_transformer_compat() -> None:
    # The NT v2 remote modeling file was authored against an older Transformers
    # utility surface. Newer Colab images keep prune_linear_layer but no longer
    # export this helper, so provide the original implementation before dynamic
    # remote-code import executes.
    import transformers.pytorch_utils as pytorch_utils

    if hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):
        return

    def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
        mask = torch.ones(n_heads, head_size)
        heads = set(heads) - already_pruned_heads
        for head in heads:
            head = head - sum(1 if pruned_head < head else 0 for pruned_head in already_pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        return heads, index

    pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices


def load_hf_encoder(hf_model: str, *, trust_remote_code: bool) -> nn.Module:
    ensure_nucleotide_transformer_compat()
    config = AutoConfig.from_pretrained(hf_model, trust_remote_code=trust_remote_code)
    for name, value in {
        "is_decoder": False,
        "add_cross_attention": False,
        "output_attentions": False,
        "output_hidden_states": False,
        "use_return_dict": True,
    }.items():
        if not hasattr(config, name):
            setattr(config, name, value)
    # Nucleotide Transformer v2 registers its custom GLU ESM implementation for
    # AutoModelForMaskedLM, not generic AutoModel. Load the official class and
    # unwrap the encoder so the checkpoint shapes match the remote model code.
    masked_lm = AutoModelForMaskedLM.from_pretrained(hf_model, config=config, trust_remote_code=trust_remote_code)
    if hasattr(masked_lm, "esm"):
        return masked_lm.esm
    if hasattr(masked_lm, "get_encoder"):
        return masked_lm.get_encoder()
    if hasattr(masked_lm, "base_model"):
        return masked_lm.base_model
    raise TypeError(f"Could not locate an encoder module inside {type(masked_lm).__name__}")


def reverse_complement(sequence: str) -> str:
    return str(sequence).translate(RC_TRANS)[::-1].upper()


@dataclass
class CategoryEncoder:
    columns: list[str]
    maps: dict[str, dict[str, int]]

    @classmethod
    def fit(cls, frame: pd.DataFrame, columns: list[str]) -> "CategoryEncoder":
        maps: dict[str, dict[str, int]] = {}
        for col in columns:
            vocab = {"<UNK>": 0}
            for value in sorted(frame[col].fillna("").astype(str).unique()):
                if value and value not in vocab:
                    vocab[value] = len(vocab)
            maps[col] = vocab
        return cls(columns, maps)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        cols = []
        for col in self.columns:
            mapping = self.maps[col]
            cols.append(frame[col].fillna("").astype(str).map(lambda value: mapping.get(value, 0)).to_numpy())
        return np.column_stack(cols).astype(np.int64) if cols else np.zeros((len(frame), 0), dtype=np.int64)


class WindowDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer,
        categories: CategoryEncoder,
        max_tokens: int,
        *,
        reverse_complement_input: bool = False,
        adversary_frame: pd.DataFrame | None = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        adversary_source = self.frame if adversary_frame is None else adversary_frame.reset_index(drop=True)
        sequences = self.frame["sequence"].astype(str)
        if reverse_complement_input:
            sequences = sequences.map(reverse_complement)
        tokens = tokenizer(
            sequences.tolist(),
            padding=True,
            truncation=True,
            max_length=max_tokens,
            return_tensors="pt",
        )
        self.input_ids = tokens["input_ids"]
        self.attention_mask = tokens.get("attention_mask", torch.ones_like(self.input_ids))
        self.categories = torch.tensor(categories.transform(self.frame), dtype=torch.long)
        self.adv_categories = torch.tensor(categories.transform(adversary_source), dtype=torch.long)
        self.numeric = torch.tensor(
            self.frame[[col for col in NUMERIC_METADATA_COLUMNS if col in self.frame.columns]]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        self.labels = torch.tensor(self.frame["label"].to_numpy(dtype=np.float32), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "categories": self.categories[idx],
            "adv_categories": self.adv_categories[idx],
            "numeric": self.numeric[idx],
            "label": self.labels[idx],
        }


class HFAxisGuard(nn.Module):
    def __init__(
        self,
        hf_model: str,
        category_columns: list[str],
        category_cardinalities: list[int],
        numeric_dim: int,
        embedding_dim: int = 16,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        pooling: str = "attention",
        adversarial_columns: list[str] | None = None,
        freeze_backbone: bool = False,
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = load_hf_encoder(hf_model, trust_remote_code=trust_remote_code)
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
        hidden = getattr(self.encoder.config, "hidden_size", None) or getattr(self.encoder.config, "d_model")
        if pooling not in {"mean", "cls", "max", "attention"}:
            raise ValueError("pooling must be one of: mean, cls, max, attention")
        self.pooling = pooling
        self.attention_pool = nn.Sequential(nn.Linear(hidden, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.category_columns = category_columns
        self.bio_indices = [i for i, col in enumerate(category_columns) if col in BIOLOGICAL_CATEGORICAL_COLUMNS]
        self.protocol_indices = [i for i, col in enumerate(category_columns) if col in PROTOCOL_CATEGORICAL_COLUMNS]
        self.embeddings = nn.ModuleList([nn.Embedding(cardinality, embedding_dim) for cardinality in category_cardinalities])
        bio_dim = hidden + len(self.bio_indices) * embedding_dim
        protocol_dim = numeric_dim + len(self.protocol_indices) * embedding_dim
        self.bio_head = nn.Sequential(nn.Linear(bio_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.protocol_head = (
            nn.Sequential(
                nn.Linear(protocol_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            if protocol_dim
            else None
        )
        self.adversaries = nn.ModuleDict()
        for col in adversarial_columns or []:
            if col in category_columns:
                idx = category_columns.index(col)
                self.adversaries[col] = nn.Linear(hidden, category_cardinalities[idx])

    def _pool(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        if self.pooling == "cls":
            return hidden[:, 0, :]
        mask = attention_mask.unsqueeze(-1).float()
        if self.pooling == "max":
            masked = hidden.masked_fill(mask.eq(0), -1e4)
            return masked.max(dim=1).values
        if self.pooling == "attention":
            score = self.attention_pool(hidden).masked_fill(mask.eq(0), -1e4)
            weights = torch.softmax(score, dim=1)
            return (hidden * weights).sum(dim=1)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def forward(self, input_ids, attention_mask, categories, numeric, adversarial_weight: float = 1.0):
        seq = self._pool(input_ids, attention_mask)
        embeds = [emb(categories[:, i]) for i, emb in enumerate(self.embeddings)]
        bio = torch.cat([seq] + [embeds[i] for i in self.bio_indices], dim=1)
        bio_logit = self.bio_head(bio).squeeze(-1)
        residual = torch.zeros_like(bio_logit)
        if self.protocol_head is not None:
            protocol_parts = []
            if numeric.shape[1]:
                protocol_parts.append(numeric)
            protocol_parts.extend(embeds[i] for i in self.protocol_indices)
            residual = self.protocol_head(torch.cat(protocol_parts, dim=1)).squeeze(-1)
        adv_logits = {}
        if self.adversaries:
            reversed_seq = GradientReverse.apply(seq, float(adversarial_weight))
            adv_logits = {name: head(reversed_seq) for name, head in self.adversaries.items()}
        return {
            "logit": bio_logit + residual,
            "bio_logit": bio_logit,
            "residual_logit": residual,
            "adv_logits": adv_logits,
        }


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.weight = weight
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.weight * grad_output, None


def _binary_row(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return {
        "n": float(len(y)),
        "prevalence": float(y.mean()),
        "auprc": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "ece": float(expected_calibration_error(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def _predict(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            output = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["categories"].to(device),
                batch["numeric"].to(device),
            )
            logits = output["logit"]
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            labels.append(batch["label"].numpy())
    return np.concatenate(labels), np.concatenate(probs)


def _predict_rc_ensemble(
    model: nn.Module,
    frame: pd.DataFrame,
    tokenizer,
    categories: CategoryEncoder,
    max_tokens: int,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    loader_fwd = DataLoader(WindowDataset(frame, tokenizer, categories, max_tokens), batch_size=batch_size)
    loader_rc = DataLoader(
        WindowDataset(frame, tokenizer, categories, max_tokens, reverse_complement_input=True),
        batch_size=batch_size,
    )
    y_fwd, p_fwd = _predict(model, loader_fwd, device)
    y_rc, p_rc = _predict(model, loader_rc, device)
    if not np.array_equal(y_fwd, y_rc):
        raise RuntimeError("forward and reverse-complement labels are misaligned")
    return y_fwd, (p_fwd + p_rc) / 2.0


def _counterfactual_metadata(categories: torch.Tensor, numeric: torch.Tensor, category_columns: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    cf_categories = categories.clone()
    for idx, col in enumerate(category_columns):
        if col in PROTOCOL_CATEGORICAL_COLUMNS:
            cf_categories[:, idx] = 0
    return cf_categories, torch.zeros_like(numeric)


def _apply_protocol_dropout(
    categories: torch.Tensor,
    numeric: torch.Tensor,
    category_columns: list[str],
    dropout: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if dropout <= 0:
        return categories, numeric
    cat = categories.clone()
    for idx, col in enumerate(category_columns):
        if col in PROTOCOL_CATEGORICAL_COLUMNS:
            mask = torch.rand(cat.shape[0], device=cat.device) < float(dropout)
            cat[mask, idx] = 0
    num = numeric
    if numeric.shape[1]:
        mask = torch.rand((numeric.shape[0], 1), device=numeric.device) < float(dropout)
        num = torch.where(mask, torch.zeros_like(numeric), numeric)
    return cat, num


def run_one(args: argparse.Namespace, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    frame = _prepare_window_frame(_read_window_table(args.window_table))
    filtered = filter_window_frame(
        frame,
        drop_all_n=args.drop_all_n,
        drop_duplicate_sequences=args.drop_duplicate_sequences,
    ).frame
    split_spec = parse_split_spec(args.split)
    split = make_split(filtered, split_spec.split_type, holdout=split_spec.holdout, random_state=seed)
    model_frame = _mask_leaky_metadata(filtered, split_spec)
    train = model_frame.loc[split.eq("train")].copy()
    valid = model_frame.loc[split.eq("valid")].copy()
    test = model_frame.loc[split.eq("test")].copy()
    train_original = filtered.loc[split.eq("train")].copy()
    test_original = filtered.loc[split.eq("test")].copy()
    if args.rc_augment:
        train_rc = train.copy()
        train_rc["sequence"] = train_rc["sequence"].map(reverse_complement)
        train = pd.concat([train, train_rc], ignore_index=True)
        train_original_rc = train_original.copy()
        train_original_rc["sequence"] = train_original_rc["sequence"].map(reverse_complement)
        train_original = pd.concat([train_original, train_original_rc], ignore_index=True)
    numeric_cols = [col for col in NUMERIC_METADATA_COLUMNS if col in filtered.columns]
    for col in numeric_cols:
        values = pd.to_numeric(train[col], errors="coerce")
        mean = float(values.mean()) if not values.dropna().empty else 0.0
        std = float(values.std(ddof=0)) if not values.dropna().empty else 1.0
        std = std if std > 1e-8 else 1.0
        for target in (train, valid, test):
            target[col] = (pd.to_numeric(target[col], errors="coerce").fillna(mean) - mean) / std

    category_columns = [col for col in (*BIOLOGICAL_CATEGORICAL_COLUMNS, *PROTOCOL_CATEGORICAL_COLUMNS) if col in filtered.columns]
    categories = CategoryEncoder.fit(train_original, category_columns)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=args.trust_remote_code)
    train_loader = DataLoader(
        WindowDataset(train, tokenizer, categories, args.max_tokens, adversary_frame=train_original),
        batch_size=args.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(WindowDataset(valid, tokenizer, categories, args.max_tokens), batch_size=args.batch_size)
    test_loader = DataLoader(WindowDataset(test, tokenizer, categories, args.max_tokens), batch_size=args.batch_size)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; refusing to fall back to CPU for an A100 run")
    device = args.device
    adversarial_cols = [col for col in ("assay", "lab") if col in category_columns]
    model = HFAxisGuard(
        args.hf_model,
        category_columns,
        [len(categories.maps[col]) for col in category_columns],
        numeric_dim=len([col for col in NUMERIC_METADATA_COLUMNS if col in filtered.columns]),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        pooling=args.pooling,
        adversarial_columns=adversarial_cols,
        freeze_backbone=args.freeze_backbone,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    labels = train["label"].to_numpy(dtype=np.float32)
    pos_weight = torch.tensor(max((len(labels) - labels.sum()) / max(labels.sum(), 1), 1.0), device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    ce_loss = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in train_loader:
            batch_categories = batch["categories"].to(device)
            batch_numeric = batch["numeric"].to(device)
            batch_categories, batch_numeric = _apply_protocol_dropout(
                batch_categories,
                batch_numeric,
                category_columns,
                args.protocol_dropout,
            )
            output = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch_categories,
                batch_numeric,
                adversarial_weight=float(args.adversarial_weight),
            )
            logits = output["logit"]
            labels = batch["label"].to(device)
            loss = loss_fn(logits, labels)
            cf_categories, cf_numeric = _counterfactual_metadata(
                batch_categories,
                batch_numeric,
                category_columns,
            )
            cf_output = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                cf_categories,
                cf_numeric,
                adversarial_weight=float(args.adversarial_weight),
            )
            cf_logits = cf_output["logit"]
            loss = loss + float(args.counterfactual_weight) * torch.mean(
                (torch.sigmoid(logits) - torch.sigmoid(cf_logits)) ** 2
            )
            loss = loss + float(args.metadata_residual_weight) * torch.mean(output["residual_logit"] ** 2)
            if args.adversarial_weight > 0 and output.get("adv_logits"):
                raw_categories = batch["adv_categories"].to(device)
                for col, adv_logits in output["adv_logits"].items():
                    col_idx = category_columns.index(col)
                    loss = loss + float(args.adversarial_weight) * ce_loss(adv_logits, raw_categories[:, col_idx])
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"seed={seed} epoch={epoch + 1}/{args.epochs} loss={np.mean(losses):.4f}", flush=True)

    if args.rc_ensemble:
        valid_y, valid_p = _predict_rc_ensemble(
            model,
            valid,
            tokenizer,
            categories,
            args.max_tokens,
            args.batch_size,
            device,
        )
        test_y, test_p = _predict_rc_ensemble(
            model,
            test,
            tokenizer,
            categories,
            args.max_tokens,
            args.batch_size,
            device,
        )
    else:
        valid_y, valid_p = _predict(model, valid_loader, device)
        test_y, test_p = _predict(model, test_loader, device)
    calibrator = PlattCalibrator().fit(valid_p, valid_y)
    test_p_cal = calibrator.predict(test_p)
    result_rows = []
    pred_rows = []
    for calibrated, probs in [(False, test_p), (True, test_p_cal)]:
        result_rows.append(
            {
                "seed": seed,
                "split": split_spec.name,
                "model": args.model_name,
                "calibrated": calibrated,
                **_binary_row(test_y, probs),
            }
        )
        pred = test_original.copy()
        pred["seed"] = seed
        pred["split_name"] = split_spec.name
        pred["model"] = args.model_name
        pred["calibrated"] = calibrated
        pred["prob"] = probs
        pred_rows.append(pred[["seed", "example_id", "label", "prob", "split_name", "model", "calibrated", "assay", "lab", "tf", "tf_family"]])
    selective = selective_metrics(test_y, test_p_cal)
    selective.insert(0, "calibrated", True)
    selective.insert(0, "model", args.model_name)
    selective.insert(0, "split", split_spec.name)
    selective.insert(0, "seed", seed)
    return pd.DataFrame(result_rows), pd.concat(pred_rows, ignore_index=True), selective


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("window_table", type=Path)
    parser.add_argument("--out", type=Path, default=Path("reports"))
    parser.add_argument("--prefix", default="hf_axisguard")
    parser.add_argument("--split", required=True, help="Single split spec, e.g. lab_heldout_haib=lab:Richard Myers, HAIB")
    parser.add_argument("--hf-model", default="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species")
    parser.add_argument("--model-name", default="hf_axis_guard")
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling", choices=["mean", "cls", "max", "attention"], default="attention")
    parser.add_argument("--counterfactual-weight", type=float, default=0.2)
    parser.add_argument("--metadata-residual-weight", type=float, default=0.02)
    parser.add_argument("--protocol-dropout", type=float, default=0.1)
    parser.add_argument("--adversarial-weight", type=float, default=0.02)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--rc-augment", action="store_true", help="Duplicate training rows with reverse complements.")
    parser.add_argument("--rc-ensemble", action="store_true", help="Average forward and reverse-complement probabilities.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--drop-all-n", action="store_true")
    parser.add_argument("--drop-duplicate-sequences", action="store_true")
    args = parser.parse_args()
    args.seed = args.seed or [13]

    result_frames = []
    pred_frames = []
    selective_frames = []
    args.out.mkdir(parents=True, exist_ok=True)
    result_path = args.out / f"{args.prefix}_results.csv"
    pred_path = args.out / f"{args.prefix}_predictions.csv"
    selective_path = args.out / f"{args.prefix}_selective.csv"
    for seed_idx, seed in enumerate(args.seed, start=1):
        print(f"[hf-axisguard] seed {seed_idx}/{len(args.seed)} = {seed}", flush=True)
        results, predictions, selective = run_one(args, seed)
        result_frames.append(results)
        pred_frames.append(predictions)
        selective_frames.append(selective)
        pd.concat(result_frames, ignore_index=True).to_csv(result_path, index=False)
        pd.concat(pred_frames, ignore_index=True).to_csv(pred_path, index=False)
        pd.concat(selective_frames, ignore_index=True).to_csv(selective_path, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
