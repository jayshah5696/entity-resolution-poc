"""Pydantic config models for Modal fine-tuning.

Loads configs/finetune_modal.yaml and validates all fields at import time.
Models inherit from shared defaults unless they provide overrides.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class ModalInfra(BaseModel):
    """Modal infrastructure settings."""

    app_name: str = "entity-resolution-finetune"
    volume_name: str = "entity-resolution-checkpoints"
    gpu: str = "A10G"
    timeout_min: int = 90


class TrainDefaults(BaseModel):
    """Shared training defaults — models inherit these unless they override."""

    epochs: int = 3
    batch_size: int = 256
    grad_accum: int = 1
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    logging_steps: int = 50
    save_steps: int = 500
    save_total_limit: int = 2
    mnrl_scale: float = 20.0
    dataloader_num_workers: int = 4
    curriculum: list[float] = Field(default=[0.10, 0.30, 0.50])
    seed: int = 42
    bf16: bool = True
    fp16: bool = False
    grad_checkpointing: bool = False


class ModelConfig(BaseModel):
    """Per-model config. Training fields override defaults when present."""

    # Required
    hf_id: str
    dims: list[int]

    # Model loading
    trust_remote_code: bool = False
    mrl_native: bool = False
    query_prefix: Optional[str] = None
    doc_prefix: Optional[str] = None

    # Training overrides (None = use default)
    batch_size: Optional[int] = None
    grad_accum: Optional[int] = None
    epochs: Optional[int] = None
    learning_rate: Optional[float] = None
    warmup_ratio: Optional[float] = None
    weight_decay: Optional[float] = None
    logging_steps: Optional[int] = None
    save_steps: Optional[int] = None
    save_total_limit: Optional[int] = None
    mnrl_scale: Optional[float] = None
    dataloader_num_workers: Optional[int] = None
    curriculum: Optional[list[float]] = None
    seed: Optional[int] = None
    bf16: Optional[bool] = None
    fp16: Optional[bool] = None
    grad_checkpointing: Optional[bool] = None

    @model_validator(mode="after")
    def validate_dims(self) -> "ModelConfig":
        if not self.dims:
            raise ValueError("dims must be non-empty")
        if self.dims != sorted(self.dims, reverse=True):
            raise ValueError(f"dims must be sorted descending, got {self.dims}")
        return self


class ResolvedModelConfig(BaseModel):
    """Fully resolved config for a model — defaults merged with overrides."""

    # Model identity
    model_key: str
    hf_id: str
    dims: list[int]
    trust_remote_code: bool
    mrl_native: bool
    query_prefix: Optional[str]
    doc_prefix: Optional[str]

    # Training (all resolved — no Nones)
    epochs: int
    batch_size: int
    grad_accum: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    logging_steps: int
    save_steps: int
    save_total_limit: int
    mnrl_scale: float
    dataloader_num_workers: int
    curriculum: list[float]
    seed: int
    bf16: bool
    fp16: bool
    grad_checkpointing: bool

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class FinetuneModalConfig(BaseModel):
    """Top-level config loaded from configs/finetune_modal.yaml."""

    modal: ModalInfra = Field(default_factory=ModalInfra)
    hf_dataset_repo: str = "jayshah5696/entity-resolution-triplets"
    hf_model_prefix: str = "jayshah5696/er"
    wandb_project: str = "entity-resolution-poc"
    wandb_entity: str = "jayshah5696"
    defaults: TrainDefaults = Field(default_factory=TrainDefaults)
    models: dict[str, ModelConfig] = Field(default_factory=dict)

    def resolve(self, model_key: str) -> ResolvedModelConfig:
        """Merge defaults with per-model overrides to produce a fully resolved config."""
        if model_key not in self.models:
            raise KeyError(
                f"Model '{model_key}' not in config. Available: {list(self.models.keys())}"
            )
        model = self.models[model_key]
        defaults = self.defaults

        # For each training field: use model override if set, else default
        resolved = {}
        for field_name in TrainDefaults.model_fields:
            model_val = getattr(model, field_name, None)
            resolved[field_name] = (
                model_val if model_val is not None else getattr(defaults, field_name)
            )

        return ResolvedModelConfig(
            model_key=model_key,
            hf_id=model.hf_id,
            dims=model.dims,
            trust_remote_code=model.trust_remote_code,
            mrl_native=model.mrl_native,
            query_prefix=model.query_prefix,
            doc_prefix=model.doc_prefix,
            **resolved,
        )

    @property
    def model_keys(self) -> list[str]:
        return list(self.models.keys())


def load_config(path: str | Path | None = None) -> FinetuneModalConfig:
    """Load and validate config from YAML file.

    Args:
        path: Path to YAML file. If None, auto-discovers from common locations.
    """
    if path is None:
        # Try common locations
        candidates = [
            Path("configs/finetune_modal.yaml"),
            Path(__file__).resolve().parents[2] / "configs" / "finetune_modal.yaml",
            Path("/configs/finetune_modal.yaml"),  # Inside Modal container
            Path.cwd() / "configs" / "finetune_modal.yaml",  # Fallback for uv run
        ]
        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break
        if path is None:
            raise FileNotFoundError(
                f"Config not found in: {[str(c) for c in candidates]}"
            )

    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    return FinetuneModalConfig.model_validate(raw)
