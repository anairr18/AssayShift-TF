from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SAFE_CATEGORICAL_METADATA = ("assay", "lab", "species", "tf", "tf_family", "biosample", "cell_type", "assembly")
SAFE_NUMERIC_METADATA = ("gc", "n_fraction", "length")


def _require_embedding_deps():
    try:
        import torch
        from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional A100/HF path.
        raise ImportError("embedding cache generation requires the optional deep dependencies") from exc
    return torch, AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer


def _ensure_nucleotide_transformer_compat() -> None:
    try:
        import torch
        import transformers.pytorch_utils as pytorch_utils
    except ImportError:  # pragma: no cover - optional A100/HF path.
        return

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


def _load_hf_encoder(hf_model: str, *, trust_remote_code: bool):
    torch, AutoConfig, AutoModel, AutoModelForMaskedLM, _ = _require_embedding_deps()
    _ensure_nucleotide_transformer_compat()
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
    try:
        return AutoModel.from_pretrained(hf_model, config=config, trust_remote_code=trust_remote_code)
    except Exception:
        masked_lm = AutoModelForMaskedLM.from_pretrained(hf_model, config=config, trust_remote_code=trust_remote_code)
        if hasattr(masked_lm, "esm"):
            return masked_lm.esm
        if hasattr(masked_lm, "get_encoder"):
            return masked_lm.get_encoder()
        if hasattr(masked_lm, "base_model"):
            return masked_lm.base_model
        raise TypeError(f"Could not locate an encoder module inside {type(masked_lm).__name__}")


def _pool_hidden(hidden, attention_mask, pooling: str):
    if pooling == "cls":
        return hidden[:, 0, :]
    mask = attention_mask.unsqueeze(-1).float()
    if pooling == "max":
        return hidden.masked_fill(mask.eq(0), -1e4).max(dim=1).values
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def write_hf_embedding_cache(
    frame: pd.DataFrame,
    out_path: str | Path,
    *,
    hf_model: str,
    batch_size: int = 32,
    device: str = "auto",
    max_tokens: int = 256,
    pooling: str = "mean",
    trust_remote_code: bool = False,
    limit: int | None = None,
) -> Path:
    """Write frozen HF sequence embeddings keyed by example_id."""
    if pooling not in {"mean", "cls", "max"}:
        raise ValueError("pooling must be one of: mean, cls, max")
    if "example_id" not in frame.columns or "sequence" not in frame.columns:
        raise ValueError("embedding cache input must contain example_id and sequence columns")

    torch, _, _, _, AutoTokenizer = _require_embedding_deps()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    work = frame.copy()
    if limit is not None:
        work = work.head(int(limit)).copy()
    requested_device = device
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for embedding generation but is not available")

    tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=trust_remote_code)
    model = _load_hf_encoder(hf_model, trust_remote_code=trust_remote_code).to(requested_device)
    model.eval()
    embeddings: list[np.ndarray] = []
    sequences = work["sequence"].astype(str).tolist()
    with torch.no_grad():
        for start in range(0, len(sequences), max(1, int(batch_size))):
            batch = sequences[start : start + max(1, int(batch_size))]
            tokens = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_tokens,
                return_tensors="pt",
            )
            input_ids = tokens["input_ids"].to(requested_device)
            attention_mask = tokens.get("attention_mask", torch.ones_like(input_ids)).to(requested_device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
            pooled = _pool_hidden(hidden, attention_mask, pooling)
            embeddings.append(pooled.detach().cpu().numpy().astype(np.float32))
            print(f"[embed] {min(start + len(batch), len(sequences))}/{len(sequences)}", flush=True)

    metadata = {
        "hf_model": hf_model,
        "pooling": pooling,
        "max_tokens": int(max_tokens),
        "n": int(len(work)),
        "dim": int(embeddings[0].shape[1]) if embeddings else 0,
    }
    np.savez_compressed(
        out,
        example_id=np.asarray(work["example_id"].astype(str).tolist(), dtype=str),
        embedding=np.vstack(embeddings).astype(np.float32) if embeddings else np.zeros((0, 0), dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    return out


@dataclass
class EmbeddingHeadConfig:
    cache_path: str | Path
    head: str = "logreg"
    include_metadata: bool = False
    random_state: int = 13


class EmbeddingHeadClassifier:
    """Frozen-embedding classifier with sklearn-style fit/predict_proba."""

    def __init__(self, config: EmbeddingHeadConfig) -> None:
        self.config = config
        if self.config.head not in {"logreg", "mlp"}:
            raise ValueError("embedding head must be one of: logreg, mlp")
        self.classes_ = np.array([0, 1])
        self._load_cache()

    def _load_cache(self) -> None:
        data = np.load(self.config.cache_path, allow_pickle=False)
        self.cache_ids_ = data["example_id"].astype(str)
        self.cache_embeddings_ = data["embedding"].astype(np.float32)
        if len(set(self.cache_ids_.tolist())) != len(self.cache_ids_):
            raise ValueError("embedding cache contains duplicate example_id values")
        self.cache_index_ = {example_id: idx for idx, example_id in enumerate(self.cache_ids_)}

    def _embedding_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        if "example_id" not in frame.columns:
            raise ValueError("embedding_head requires an example_id column for cache alignment")
        ids = frame["example_id"].astype(str).tolist()
        missing = [example_id for example_id in ids if example_id not in self.cache_index_]
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(f"embedding cache is missing {len(missing)} example_id values, e.g. {preview}")
        return self.cache_embeddings_[[self.cache_index_[example_id] for example_id in ids]]

    def _metadata_columns(self, frame: pd.DataFrame) -> tuple[list[str], list[str]]:
        categorical = [col for col in SAFE_CATEGORICAL_METADATA if col in frame.columns]
        numeric = [col for col in SAFE_NUMERIC_METADATA if col in frame.columns]
        return categorical, numeric

    def _features(self, frame: pd.DataFrame, *, fit: bool = False):
        embeddings = self._embedding_matrix(frame)
        if not self.config.include_metadata:
            return embeddings
        categorical, numeric = self._metadata_columns(frame)
        if fit:
            transformers = []
            if numeric:
                transformers.append(("numeric", StandardScaler(), numeric))
            if categorical:
                transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore"), categorical))
            self.metadata_transformer_ = ColumnTransformer(transformers=transformers, remainder="drop")
            metadata = self.metadata_transformer_.fit_transform(frame)
        else:
            metadata = self.metadata_transformer_.transform(frame)
        try:
            from scipy import sparse
        except ImportError:  # pragma: no cover - sklearn normally installs scipy.
            return np.hstack([embeddings, np.asarray(metadata)])
        return sparse.hstack([sparse.csr_matrix(embeddings), metadata], format="csr")

    def fit(self, X: pd.DataFrame, y: Any) -> "EmbeddingHeadClassifier":
        features = self._features(X, fit=True)
        if self.config.head == "mlp":
            self.model_ = MLPClassifier(
                hidden_layer_sizes=(128,),
                alpha=1e-4,
                max_iter=300,
                random_state=int(self.config.random_state),
            )
        else:
            self.model_ = LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="liblinear",
                random_state=int(self.config.random_state),
            )
        self.model_.fit(features, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("EmbeddingHeadClassifier.fit must be called before predict_proba")
        return self.model_.predict_proba(self._features(X, fit=False))
