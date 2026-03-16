"""Pydantic config models for Phase 2 cross-encoder training.

Loads configs/crossencoder_models.yaml and configs/crossencoder.yaml.
Mirrors the pattern from src/models/finetune_config.py (Phase 1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# crossencoder_models.yaml models
# ---------------------------------------------------------------------------


class CEModelEntry(BaseModel):
    """Single cross-encoder model entry."""

    hf_id: str
    params_m: int
    license: str
    context_tokens: int
    finetune: bool
    note: str
    beir_ndcg10: Optional[str | float] = None  # YAML uses "~50" for approx values
    mteb_r: Optional[float] = None


class CEModelsConfig(BaseModel):
    """Top-level config for crossencoder_models.yaml."""

    models: dict[str, CEModelEntry]
    finetune_targets: list[str]
    zero_shot_references: list[str]
    hf_model_prefix: str
    hf_dataset: str


# ---------------------------------------------------------------------------
# crossencoder.yaml models
# ---------------------------------------------------------------------------


class DataConfig(BaseModel):
    """Data section of crossencoder.yaml."""

    train: str
    val: str
    test: str
    serialization: str
    top_k_stage1: int
    label_col: str
    pos_neg_ratio: str


class TrainingConfig(BaseModel):
    """Training hyperparameters."""

    epochs: int
    batch_size: int
    lr: float
    warmup_steps: int
    weight_decay: float
    max_length: int
    fp16: bool
    bf16: bool
    dataloader_num_workers: int
    seed: int


class LossConfig(BaseModel):
    """Loss schedule config."""

    phase1_epochs: int
    phase1_loss: str
    phase2_epochs: int
    phase2_loss: str


class CurriculumEntry(BaseModel):
    """Single epoch curriculum entry."""

    hard_pct: float
    random_pct: float


class HardNegativeFilterConfig(BaseModel):
    """Hard negative pre-filter settings."""

    min_margin: float
    stock_ce_for_filter: str


class EvalConfig(BaseModel):
    """Evaluation strategy config."""

    strategy: str
    eval_steps: int
    save_steps: int
    load_best_model_at_end: bool
    metric_for_best_model: str
    greater_is_better: bool


class ModalConfig(BaseModel):
    """Modal infrastructure config."""

    gpu: str
    timeout_min: int
    volume: str
    python_version: str


class WandbConfig(BaseModel):
    """W&B config."""

    project: str
    entity: str
    tags: list[str]


class ThresholdCalibrationConfig(BaseModel):
    """Threshold calibration settings."""

    method: str
    fallback: float


class HFHubConfig(BaseModel):
    """HuggingFace Hub push config."""

    push_after_training: bool


class CETrainingConfig(BaseModel):
    """Top-level config for crossencoder.yaml."""

    data: DataConfig
    training: TrainingConfig
    loss: LossConfig
    hard_negative_curriculum: dict[str, CurriculumEntry]
    hard_negative_filter: HardNegativeFilterConfig
    eval: EvalConfig
    modal: ModalConfig
    wandb: WandbConfig
    threshold_calibration: ThresholdCalibrationConfig
    hf_hub: HFHubConfig


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _find_config(filename: str, explicit_path: str | Path | None = None) -> Path:
    """Find a config file from common locations."""
    if explicit_path is not None:
        p = Path(explicit_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Config not found at: {p}")

    candidates = [
        Path("configs") / filename,
        Path(__file__).resolve().parents[2] / "configs" / filename,
        Path("/configs") / filename,  # Inside Modal container
        Path.cwd() / "configs" / filename,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Config '{filename}' not found in: {[str(c) for c in candidates]}")


def load_ce_models_config(path: str | Path | None = None) -> CEModelsConfig:
    """Load and validate crossencoder_models.yaml."""
    config_path = _find_config("crossencoder_models.yaml", path)
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return CEModelsConfig.model_validate(raw)


def load_ce_training_config(path: str | Path | None = None) -> CETrainingConfig:
    """Load and validate crossencoder.yaml."""
    config_path = _find_config("crossencoder.yaml", path)
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return CETrainingConfig.model_validate(raw)
