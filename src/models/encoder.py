"""
Encoder wrapper for all 5 locked models.

Reads model config from configs/models.yaml and applies per-model quirks
(prefixes, prompt names, pooling) automatically. Eval scripts call
encode_queries() and encode_docs() and contain zero model-specific logic.

Model-specific quirks handled here:
  - nomic_v15: prepends "search_query: " / "search_document: " to all texts
  - pplx_embed_v1_06b: uses prompt_name="query" / "document" via sentence-transformers
  - minilm_l6, bge_small, gte_modernbert_base: plain encode, normalize=True
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseEncoder(ABC):
    """Abstract interface that all encoders must implement."""

    @abstractmethod
    def encode_docs(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        """Encode documents for indexing. Returns float32 array of shape (N, dim)."""

    @abstractmethod
    def encode_queries(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        """Encode queries for retrieval. Returns float32 array of shape (N, dim)."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimensionality. BM25 returns 0."""

    @property
    @abstractmethod
    def model_key(self) -> str:
        """Identifier matching the key in models.yaml."""


# ---------------------------------------------------------------------------
# Dense encoder (SentenceTransformer-based)
# ---------------------------------------------------------------------------


class SentenceTransformerEncoder(BaseEncoder):
    """
    Wraps sentence-transformers for all dense models.

    Handles per-model quirks:
      - nomic_v15: manually prepends "search_query: " / "search_document: "
      - pplx_embed_v1_06b: uses model.encode(..., prompt_name="query"/"document")
      - others: plain model.encode(..., normalize_embeddings=True)

    MRL truncation: if truncate_dim is given and model supports MRL,
    the output is truncated (and re-normalized) to truncate_dim dimensions.
    """

    def __init__(
        self,
        model_key: str,
        model_cfg: dict,
        device: str = "cpu",
        model_path: str | None = None,
        truncate_dim: int | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._model_key = model_key
        self._model_cfg = model_cfg
        self._device = device

        # Resolve model source
        hf_id: str = model_cfg["hf_id"]
        load_path = model_path if model_path else hf_id

        # Trust remote code for models that require it
        trust_remote_code: bool = model_cfg.get("trust_remote_code", False) or (
            model_key in ("gte_modernbert_base", "nomic_v15")
        )

        logger.info("Loading %s from %s (device=%s)", model_key, load_path, device)

        load_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if device != "cpu":
            load_kwargs["device"] = device

        # Fix #4: FP16 inference on MPS/CUDA — Apple Silicon AMX has dedicated
        # FP16 matrix-multiply hardware; this roughly doubles encode throughput.
        if device in ("mps", "cuda"):
            load_kwargs["model_kwargs"] = {"torch_dtype": torch.float16}

        self._model = SentenceTransformer(load_path, **load_kwargs)

        # Fix #5: torch.compile on inner transformer — fuses Metal kernels,
        # reduces Python→GPU call overhead. ~15-30% encode speedup after warmup.
        # Use dynamic=True because sentence lengths vary.
        # Skip for trust_remote_code models (e.g. gte_modernbert, nomic) — their
        # custom architectures emit float64 ops that MPS Inductor doesn't support.
        if (
            device in ("mps", "cuda")
            and hasattr(torch, "compile")
            and not trust_remote_code
        ):
            try:
                self._model[0].auto_model = torch.compile(
                    self._model[0].auto_model,
                    mode="reduce-overhead",
                    dynamic=True,
                )
                logger.info("torch.compile applied to %s (mode=reduce-overhead)", model_key)
            except Exception as e:
                logger.warning("torch.compile failed for %s, falling back to eager: %s", model_key, e)

        # Determine effective dimension
        dims_cfg = model_cfg.get("dims")
        if isinstance(dims_cfg, list) and len(dims_cfg) > 0:
            full_dim = int(dims_cfg[0])
        else:
            full_dim = self._model.get_sentence_embedding_dimension() or 0

        if truncate_dim is not None and dims_cfg and truncate_dim in dims_cfg:
            self._dim = truncate_dim
        else:
            self._dim = full_dim

        self._truncate_dim = truncate_dim if (truncate_dim and truncate_dim < full_dim) else None

        # Detect encoding strategy
        pooling = model_cfg.get("pooling", "mean")
        self._use_prompt_name: bool = pooling == "eos_token"  # pplx only

        # Nomic requires explicit text prefixes (not prompt_name)
        raw_qp = model_cfg.get("query_prefix")
        raw_dp = model_cfg.get("doc_prefix")
        # Only apply as text prefix when it is a real word-prefix (not a system-prompt key)
        self._query_prefix: str | None = None
        self._doc_prefix: str | None = None
        if not self._use_prompt_name and raw_qp and "system_prompt" not in str(raw_qp):
            self._query_prefix = str(raw_qp) + ": "
        if not self._use_prompt_name and raw_dp and "system_prompt" not in str(raw_dp):
            self._doc_prefix = str(raw_dp) + ": "

        logger.info(
            "%s loaded | dim=%d | use_prompt_name=%s | query_prefix=%r | doc_prefix=%r",
            model_key,
            self._dim,
            self._use_prompt_name,
            self._query_prefix,
            self._doc_prefix,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_prefix(self, texts: list[str], prefix: str | None) -> list[str]:
        if prefix is None:
            return texts
        return [prefix + t for t in texts]

    @torch.inference_mode()
    def _encode(
        self,
        texts: list[str],
        batch_size: int,
        prompt_name: str | None = None,
        desc: str = "Encoding",
    ) -> np.ndarray:
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        encode_kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "normalize_embeddings": True,
            "show_progress_bar": False,  # We use rich instead
        }
        if prompt_name is not None:
            encode_kwargs["prompt_name"] = prompt_name

        n_batches = max(1, len(texts) // batch_size + (1 if len(texts) % batch_size else 0))
        all_vecs: list[np.ndarray] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(desc, total=len(texts))
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                vecs = self._model.encode(batch, **encode_kwargs)
                all_vecs.append(vecs.astype(np.float32))
                progress.advance(task, len(batch))

        result = np.concatenate(all_vecs, axis=0)

        if self._truncate_dim is not None:
            result = result[:, : self._truncate_dim]
            # Re-normalize after truncation
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            result = (result / norms).astype(np.float32)

        return result

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def encode_docs(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        """Encode documents with appropriate prefixes/prompt_name."""
        if self._use_prompt_name:
            return self._encode(texts, batch_size, prompt_name="document", desc="Encoding docs")
        prefixed = self._apply_prefix(texts, self._doc_prefix)
        return self._encode(prefixed, batch_size, desc="Encoding docs")

    def encode_queries(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        """Encode queries with appropriate prefixes/prompt_name."""
        if self._use_prompt_name:
            return self._encode(texts, batch_size, prompt_name="query", desc="Encoding queries")
        prefixed = self._apply_prefix(texts, self._query_prefix)
        return self._encode(prefixed, batch_size, desc="Encoding queries")

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_key(self) -> str:
        return self._model_key

    @property
    def model(self):
        """Underlying SentenceTransformer model (for fine-tuning)."""
        return self._model


# ---------------------------------------------------------------------------
# BM25 (FTS) Placeholder
# ---------------------------------------------------------------------------


class BM25FTSPlaceholder(BaseEncoder):
    """
    Placeholder for BM25 when using LanceDB native FTS.
    BM25 doesn't need an encoder class since tokenization and indexing 
    are handled by LanceDB/Tantivy.
    """

    def __init__(self, model_key: str) -> None:
        self._model_key = model_key

    def encode_docs(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        return np.empty((0,), dtype=np.float32)

    def encode_queries(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        return np.empty((0,), dtype=np.float32)

    @property
    def dim(self) -> int:
        return 0

    @property
    def model_key(self) -> str:
        return self._model_key


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def load_encoder(
    model_key: str,
    model_cfg: dict,
    device: str = "cpu",
    model_path: str | None = None,
    truncate_dim: int | None = None,
) -> BaseEncoder:
    """
    Load the appropriate encoder for a given model key.

    Parameters
    ----------
    model_key : str
        One of the 5 locked model keys from models.yaml.
    model_cfg : dict
        The dict for this specific model from models.yaml.
    device : str
        "cpu", "cuda", or "mps". Ignored for BM25.
    model_path : str | None
        If given, load a fine-tuned model from this local path.
    truncate_dim : int | None
        For MRL models, truncate embeddings to this dimension.

    Returns
    -------
    BaseEncoder
    """
    model_type = model_cfg.get("type", "dense")
    if model_type == "bm25":
        return BM25FTSPlaceholder(model_key=model_key)

    return SentenceTransformerEncoder(
        model_key=model_key,
        model_cfg=model_cfg,
        device=device,
        model_path=model_path,
        truncate_dim=truncate_dim,
    )
