from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


BASE_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}
RC_TRANS = str.maketrans("ACGTNacgtn", "TGCANtgcan")
BIOLOGICAL_CATEGORICAL_COLUMNS = ("tf", "tf_family")
PROTOCOL_CATEGORICAL_COLUMNS = ("assay", "lab", "species", "biosample", "cell_type", "assembly")
NUMERIC_METADATA_COLUMNS = ("gc", "n_fraction", "length")
LEAKAGE_PRONE_COLUMNS = ("label", "example_id", "chrom", "start", "end", "source_width", "source_score")


def reverse_complement(sequence: str) -> str:
    return str(sequence).translate(RC_TRANS)[::-1].upper()


def one_hot_encode(sequences: list[str], max_len: int | None = None) -> np.ndarray:
    """Return N x 4 x L one-hot DNA tensors for CNN baselines."""
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    arr = np.zeros((len(sequences), 4, max_len), dtype=np.float32)
    for i, seq in enumerate(sequences):
        for j, base in enumerate(str(seq).upper()[:max_len]):
            idx = BASE_TO_INDEX.get(base)
            if idx is not None:
                arr[i, idx, j] = 1.0
    return arr


@dataclass(frozen=True)
class DeepModelConfig:
    epochs: int = 10
    batch_size: int = 128
    lr: float = 1e-3
    device: str = "auto"
    random_state: int = 13
    max_len: int | None = None
    conv_channels: int = 64
    embedding_dim: int = 8
    hidden_dim: int = 64
    dropout: float = 0.2
    axis_dropout: float = 0.15
    counterfactual_mode: str = "mask"
    counterfactual_weight: float = 0.35
    metadata_residual_weight: float = 0.02
    adversarial_weight: float = 0.0
    objective: str = "erm"
    group_key: str = "protocol"
    groupdro_eta: float = 0.2
    protocol_penalty: str = "none"
    protocol_penalty_weight: float = 0.0
    rc_augment: bool = False
    rc_ensemble: bool = False
    weight_decay: float = 1e-4
    verbose: bool = False


class SequenceMetadataDataset:
    """Tensor-ready view of a sequence window table."""

    def __init__(self, frame: pd.DataFrame, encoder: "TorchProbClassifier", y: Any | None = None) -> None:
        self.sequence = one_hot_encode(frame["sequence"].astype(str).tolist(), encoder.max_len_)
        self.numeric = encoder._transform_numeric(frame)
        self.categorical = encoder._transform_categorical(frame)
        self.group = encoder._transform_group(frame)
        self.y = None if y is None else np.asarray(y, dtype=np.float32)

    def __len__(self) -> int:
        return int(self.sequence.shape[0])

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.int64, np.float32]:
        y = np.float32(0.0) if self.y is None else np.float32(self.y[idx])
        return self.sequence[idx], self.numeric[idx], self.categorical[idx], np.int64(self.group[idx]), y


def _require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - exercised only without optional dependency.
        raise ImportError("tiny_cnn and axis_guard_cnn require the optional 'torch' dependency") from exc
    return torch, nn


def _resolve_device(torch, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def _build_torch_modules():
    torch, nn = _require_torch()

    class GradientReverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight):
            ctx.weight = weight
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output):
            return -ctx.weight * grad_output, None

    class SequenceEncoder(nn.Module):
        def __init__(self, conv_channels: int, dropout: float) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(4, conv_channels, kernel_size=15, padding=7),
                nn.BatchNorm1d(conv_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(conv_channels, conv_channels, kernel_size=9, padding=4),
                nn.BatchNorm1d(conv_channels),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1),
                nn.Flatten(),
                nn.Dropout(dropout),
            )

        def forward(self, sequence):
            return self.net(sequence)

    class TinySequenceCNN(nn.Module):
        def __init__(self, conv_channels: int, dropout: float) -> None:
            super().__init__()
            self.encoder = SequenceEncoder(conv_channels, dropout)
            self.classifier = nn.Linear(conv_channels, 1)

        def forward(self, sequence, numeric=None, categorical=None):
            features = self.encoder(sequence)
            return {
                "logit": self.classifier(features).squeeze(-1),
                "bio_logit": None,
                "residual_logit": None,
                "features": features,
            }

    class AxisGuardNet(nn.Module):
        def __init__(
            self,
            conv_channels: int,
            numeric_dim: int,
            categorical_columns: list[str],
            categorical_cardinalities: list[int],
            embedding_dim: int,
            hidden_dim: int,
            dropout: float,
            adversarial_columns: list[str],
        ) -> None:
            super().__init__()
            self.categorical_columns = list(categorical_columns)
            self.bio_indices = [
                i for i, col in enumerate(categorical_columns) if col in BIOLOGICAL_CATEGORICAL_COLUMNS
            ]
            self.protocol_indices = [
                i for i, col in enumerate(categorical_columns) if col in PROTOCOL_CATEGORICAL_COLUMNS
            ]
            self.encoder = SequenceEncoder(conv_channels, dropout)
            self.embeddings = nn.ModuleList(
                [nn.Embedding(cardinality, embedding_dim) for cardinality in categorical_cardinalities]
            )

            bio_dim = conv_channels + len(self.bio_indices) * embedding_dim
            protocol_dim = numeric_dim + len(self.protocol_indices) * embedding_dim
            self.bio_head = nn.Sequential(
                nn.Linear(bio_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
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
            for col in adversarial_columns:
                if col in categorical_columns:
                    idx = categorical_columns.index(col)
                    self.adversaries[col] = nn.Linear(conv_channels, categorical_cardinalities[idx])

        def _embed_columns(self, categorical):
            if categorical.shape[1] == 0:
                return []
            return [embedding(categorical[:, i]) for i, embedding in enumerate(self.embeddings)]

        def forward(self, sequence, numeric, categorical, adversarial_weight: float = 1.0):
            seq_features = self.encoder(sequence)
            embeddings = self._embed_columns(categorical)
            bio_parts = [seq_features] + [embeddings[i] for i in self.bio_indices]
            bio_features = torch.cat(bio_parts, dim=1)
            bio_logit = self.bio_head(bio_features).squeeze(-1)

            residual_logit = torch.zeros_like(bio_logit)
            if self.protocol_head is not None:
                protocol_parts = []
                if numeric.shape[1]:
                    protocol_parts.append(numeric)
                protocol_parts.extend(embeddings[i] for i in self.protocol_indices)
                if protocol_parts:
                    protocol_features = torch.cat(protocol_parts, dim=1)
                    residual_logit = self.protocol_head(protocol_features).squeeze(-1)

            adv_logits = {}
            if self.adversaries:
                reversed_features = GradientReverse.apply(seq_features, float(adversarial_weight))
                adv_logits = {name: head(reversed_features) for name, head in self.adversaries.items()}

            return {
                "logit": bio_logit + residual_logit,
                "bio_logit": bio_logit,
                "residual_logit": residual_logit,
                "adv_logits": adv_logits,
                "features": seq_features,
            }

    return torch, nn, TinySequenceCNN, AxisGuardNet


class TorchProbClassifier:
    """Small PyTorch binary classifier with an sklearn-like API."""

    def __init__(self, model_kind: str, config: DeepModelConfig | None = None) -> None:
        if model_kind not in {"tiny_cnn", "axis_guard_cnn"}:
            raise ValueError(f"unsupported deep model kind: {model_kind}")
        self.model_kind = model_kind
        self.config = config or DeepModelConfig()
        if self.config.counterfactual_mode not in {"mask", "shuffle", "mask_or_shuffle"}:
            raise ValueError("counterfactual_mode must be one of 'mask', 'shuffle', or 'mask_or_shuffle'")
        if self.config.objective not in {"erm", "groupdro"}:
            raise ValueError("objective must be one of 'erm' or 'groupdro'")
        if self.config.protocol_penalty not in {"none", "coral", "mmd"}:
            raise ValueError("protocol_penalty must be one of 'none', 'coral', or 'mmd'")
        self.classes_ = np.array([0, 1])

    def fit(self, X: pd.DataFrame, y: Any) -> "TorchProbClassifier":
        torch, nn, TinySequenceCNN, AxisGuardNet = _build_torch_modules()
        torch.manual_seed(int(self.config.random_state))
        np.random.seed(int(self.config.random_state))
        self.device_ = _resolve_device(torch, self.config.device)
        train_frame = X.copy()
        train_y = np.asarray(y, dtype=np.float32)
        if self.config.rc_augment:
            rc_frame = train_frame.copy()
            rc_frame["sequence"] = rc_frame["sequence"].map(reverse_complement)
            train_frame = pd.concat([train_frame, rc_frame], ignore_index=True)
            train_y = np.concatenate([train_y, train_y])

        self.max_len_ = self.config.max_len or int(train_frame["sequence"].astype(str).map(len).max())
        self.numeric_columns_ = (
            [col for col in NUMERIC_METADATA_COLUMNS if col in train_frame.columns] if self.model_kind == "axis_guard_cnn" else []
        )
        self.categorical_columns_ = (
            [
                col
                for col in (*BIOLOGICAL_CATEGORICAL_COLUMNS, *PROTOCOL_CATEGORICAL_COLUMNS)
                if col in train_frame.columns
            ]
            if self.model_kind == "axis_guard_cnn"
            else []
        )
        self.feature_columns_ = ["sequence", *self.numeric_columns_, *self.categorical_columns_]
        self.leakage_excluded_columns_ = [col for col in LEAKAGE_PRONE_COLUMNS if col in train_frame.columns]
        self._fit_metadata_encoders(train_frame)
        self._fit_group_encoder(train_frame)

        dataset = SequenceMetadataDataset(train_frame, self, train_y)
        generator = torch.Generator()
        generator.manual_seed(int(self.config.random_state))
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=max(1, int(self.config.batch_size)),
            shuffle=True,
            generator=generator,
        )

        if self.model_kind == "tiny_cnn":
            self.model_ = TinySequenceCNN(self.config.conv_channels, self.config.dropout)
        else:
            adversarial_cols = [col for col in ("assay", "lab") if col in self.categorical_columns_]
            self.model_ = AxisGuardNet(
                conv_channels=self.config.conv_channels,
                numeric_dim=len(self.numeric_columns_),
                categorical_columns=self.categorical_columns_,
                categorical_cardinalities=[len(self.category_maps_[col]) for col in self.categorical_columns_],
                embedding_dim=self.config.embedding_dim,
                hidden_dim=self.config.hidden_dim,
                dropout=self.config.dropout,
                adversarial_columns=adversarial_cols,
            )
        self.model_.to(self.device_)
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=float(self.config.lr),
            weight_decay=float(self.config.weight_decay),
        )
        labels = train_y
        positives = max(float(labels.sum()), 1.0)
        negatives = max(float(labels.shape[0] - labels.sum()), 1.0)
        bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(negatives / positives, device=self.device_),
            reduction="none",
        )
        ce = nn.CrossEntropyLoss()
        groupdro_weights = torch.ones(len(self.group_map_), device=self.device_) / max(len(self.group_map_), 1)

        self.model_.train()
        for epoch in range(max(1, int(self.config.epochs))):
            losses = []
            for sequence, numeric, categorical, group, target in loader:
                sequence = sequence.to(self.device_)
                numeric = numeric.to(self.device_)
                categorical = categorical.to(self.device_, dtype=torch.long)
                group = group.to(self.device_, dtype=torch.long)
                target = target.to(self.device_)
                if self.model_kind == "axis_guard_cnn":
                    numeric_in, categorical_in = self._apply_axis_dropout(numeric, categorical, torch)
                    output = self.model_(
                        sequence,
                        numeric_in,
                        categorical_in,
                        adversarial_weight=float(self.config.adversarial_weight),
                    )
                else:
                    output = self.model_(sequence, numeric, categorical)
                supervised_loss = bce(output["logit"], target)
                loss, groupdro_weights = self._reduce_supervised_loss(supervised_loss, group, groupdro_weights, torch)

                if self.config.protocol_penalty != "none" and self.config.protocol_penalty_weight > 0:
                    loss = loss + float(self.config.protocol_penalty_weight) * self._alignment_penalty(
                        output["features"],
                        group,
                        torch,
                    )

                if self.model_kind == "axis_guard_cnn":
                    cf_numeric, cf_categorical = self._counterfactual_metadata(numeric, categorical, torch)
                    cf_output = self.model_(
                        sequence,
                        cf_numeric,
                        cf_categorical,
                        adversarial_weight=float(self.config.adversarial_weight),
                    )
                    consistency = torch.mean((torch.sigmoid(output["logit"]) - torch.sigmoid(cf_output["logit"])) ** 2)
                    residual_penalty = torch.mean(output["residual_logit"] ** 2)
                    loss = loss + float(self.config.counterfactual_weight) * consistency
                    loss = loss + float(self.config.metadata_residual_weight) * residual_penalty
                    if self.config.adversarial_weight > 0 and output.get("adv_logits"):
                        for col, logits in output["adv_logits"].items():
                            col_idx = self.categorical_columns_.index(col)
                            loss = loss + float(self.config.adversarial_weight) * ce(logits, categorical[:, col_idx])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            if self.config.verbose:
                mean_loss = float(np.mean(losses)) if losses else float("nan")
                print(f"{self.model_kind} epoch {epoch + 1}/{self.config.epochs}: loss={mean_loss:.4f}")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.config.rc_ensemble:
            forward = self._predict_proba_once(X)[:, 1]
            rc = X.copy()
            rc["sequence"] = rc["sequence"].map(reverse_complement)
            reverse = self._predict_proba_once(rc)[:, 1]
            p1 = np.clip((forward + reverse) / 2.0, 1e-7, 1.0 - 1e-7)
            return np.column_stack([1.0 - p1, p1])
        return self._predict_proba_once(X)

    def _predict_proba_once(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("TorchProbClassifier.fit must be called before predict_proba")
        torch, _, _, _ = _build_torch_modules()
        dataset = SequenceMetadataDataset(X, self)
        loader = torch.utils.data.DataLoader(dataset, batch_size=max(1, int(self.config.batch_size)), shuffle=False)
        probs: list[np.ndarray] = []
        self.model_.eval()
        with torch.no_grad():
            for sequence, numeric, categorical, _, _ in loader:
                sequence = sequence.to(self.device_)
                numeric = numeric.to(self.device_)
                categorical = categorical.to(self.device_, dtype=torch.long)
                output = self.model_(sequence, numeric, categorical)
                prob = torch.sigmoid(output["logit"]).detach().cpu().numpy()
                probs.append(prob)
        p1 = np.concatenate(probs) if probs else np.asarray([], dtype=np.float32)
        p1 = np.clip(p1.astype(float), 1e-7, 1.0 - 1e-7)
        return np.column_stack([1.0 - p1, p1])

    def _fit_metadata_encoders(self, frame: pd.DataFrame) -> None:
        self.numeric_means_: dict[str, float] = {}
        self.numeric_stds_: dict[str, float] = {}
        for col in self.numeric_columns_:
            values = pd.to_numeric(frame[col], errors="coerce")
            mean = float(values.mean()) if not values.dropna().empty else 0.0
            std = float(values.std(ddof=0)) if not values.dropna().empty else 1.0
            self.numeric_means_[col] = mean
            self.numeric_stds_[col] = std if std > 1e-8 else 1.0

        self.category_maps_: dict[str, dict[str, int]] = {}
        for col in self.categorical_columns_:
            values = frame[col].fillna("").astype(str)
            vocab = {"<UNK>": 0}
            for value in sorted(values.unique()):
                if value and value not in vocab:
                    vocab[value] = len(vocab)
            self.category_maps_[col] = vocab

    def _transform_numeric(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.numeric_columns_:
            return np.zeros((len(frame), 0), dtype=np.float32)
        columns = []
        for col in self.numeric_columns_:
            values = pd.to_numeric(frame[col], errors="coerce").fillna(self.numeric_means_[col])
            columns.append(((values.to_numpy(dtype=np.float32) - self.numeric_means_[col]) / self.numeric_stds_[col]))
        return np.column_stack(columns).astype(np.float32)

    def _transform_categorical(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.categorical_columns_:
            return np.zeros((len(frame), 0), dtype=np.int64)
        columns = []
        for col in self.categorical_columns_:
            mapping = self.category_maps_[col]
            values = frame[col].fillna("").astype(str).map(lambda value: mapping.get(value, 0))
            columns.append(values.to_numpy(dtype=np.int64))
        return np.column_stack(columns).astype(np.int64)

    def _group_values(self, frame: pd.DataFrame) -> pd.Series:
        key = str(self.config.group_key or "protocol")
        if key == "protocol":
            preferred = ["__group_assay", "__group_lab", "__group_species"]
            fallback = ["assay", "lab", "species"]
            cols = [col for col in preferred if col in frame.columns]
            if not cols:
                cols = [col for col in fallback if col in frame.columns]
            if not cols:
                return pd.Series(["global"] * len(frame), index=frame.index)
            return frame[cols].fillna("").astype(str).agg("|".join, axis=1)
        source_col = f"__group_{key}"
        if source_col in frame.columns:
            return frame[source_col].fillna("").astype(str)
        if key in frame.columns:
            return frame[key].fillna("").astype(str)
        return pd.Series(["global"] * len(frame), index=frame.index)

    def _fit_group_encoder(self, frame: pd.DataFrame) -> None:
        values = self._group_values(frame)
        vocab = {"<UNK>": 0}
        for value in sorted(values.unique()):
            if value and value not in vocab:
                vocab[value] = len(vocab)
        self.group_map_ = vocab

    def _transform_group(self, frame: pd.DataFrame) -> np.ndarray:
        values = self._group_values(frame)
        return values.map(lambda value: self.group_map_.get(value, 0)).to_numpy(dtype=np.int64)

    def _reduce_supervised_loss(self, per_example_loss, group, groupdro_weights, torch):
        if self.config.objective != "groupdro":
            return per_example_loss.mean(), groupdro_weights
        present = torch.unique(group)
        group_losses = []
        for group_id in present:
            mask = group.eq(group_id)
            group_losses.append(per_example_loss[mask].mean())
        losses = torch.stack(group_losses)
        with torch.no_grad():
            groupdro_weights[present] = groupdro_weights[present] * torch.exp(float(self.config.groupdro_eta) * losses.detach())
            groupdro_weights = groupdro_weights / groupdro_weights.sum().clamp_min(1e-12)
        weights = groupdro_weights[present]
        return (weights * losses).sum() / weights.sum().clamp_min(1e-12), groupdro_weights

    def _alignment_penalty(self, features, group, torch):
        unique = torch.unique(group)
        if unique.numel() < 2:
            return torch.zeros((), device=features.device)
        penalties = []
        for i, group_a in enumerate(unique):
            feat_a = features[group.eq(group_a)]
            for group_b in unique[i + 1 :]:
                feat_b = features[group.eq(group_b)]
                if self.config.protocol_penalty == "mmd":
                    penalties.append(torch.mean((feat_a.mean(dim=0) - feat_b.mean(dim=0)) ** 2))
                elif feat_a.shape[0] > 1 and feat_b.shape[0] > 1:
                    centered_a = feat_a - feat_a.mean(dim=0, keepdim=True)
                    centered_b = feat_b - feat_b.mean(dim=0, keepdim=True)
                    cov_a = centered_a.T @ centered_a / max(feat_a.shape[0] - 1, 1)
                    cov_b = centered_b.T @ centered_b / max(feat_b.shape[0] - 1, 1)
                    penalties.append(torch.mean((cov_a - cov_b) ** 2))
        if not penalties:
            return torch.zeros((), device=features.device)
        return torch.stack(penalties).mean()

    def _protocol_categorical_indices(self) -> list[int]:
        return [i for i, col in enumerate(self.categorical_columns_) if col in PROTOCOL_CATEGORICAL_COLUMNS]

    def _counterfactual_metadata(self, numeric, categorical, torch):
        mode = self.config.counterfactual_mode
        if mode == "mask_or_shuffle":
            mode = "shuffle" if bool(torch.rand((), device=categorical.device) < 0.5) else "mask"

        cf_numeric = numeric.clone()
        cf_categorical = categorical.clone()
        protocol_indices = self._protocol_categorical_indices()
        if mode == "shuffle" and categorical.shape[0] > 1:
            permutation = torch.randperm(categorical.shape[0], device=categorical.device)
            if numeric.shape[1]:
                cf_numeric = numeric[permutation].clone()
            for idx in protocol_indices:
                cf_categorical[:, idx] = categorical[permutation, idx]
        else:
            cf_numeric = torch.zeros_like(numeric)
            for idx in protocol_indices:
                cf_categorical[:, idx] = 0
        return cf_numeric, cf_categorical

    def _apply_axis_dropout(self, numeric, categorical, torch):
        if self.model_kind != "axis_guard_cnn" or self.config.axis_dropout <= 0:
            return numeric, categorical
        numeric_out = numeric
        categorical_out = categorical.clone()
        if numeric.shape[1]:
            numeric_mask = torch.rand((numeric.shape[0], 1), device=numeric.device) < float(self.config.axis_dropout)
            numeric_out = torch.where(numeric_mask, torch.zeros_like(numeric), numeric)
        for idx in self._protocol_categorical_indices():
            mask = torch.rand(categorical.shape[0], device=categorical.device) < float(self.config.axis_dropout)
            categorical_out[mask, idx] = 0
        return numeric_out, categorical_out
