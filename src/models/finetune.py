"""
finetune.py -- Fine-tune embedding models with Matryoshka + MNRL loss.

Uses sentence-transformers v3 SentenceTransformerTrainer API.
Supports curriculum training: hard negative ratio increases each epoch.
Saves a training_manifest.json alongside the model checkpoint.

Usage:
    uv run python src/models/finetune.py \\
        --model gte_modernbert_base \\
        --serialization pipe \\
        --triplets data/triplets/triplets.parquet \\
        --output-dir models/gte_modernbert_base_pipe_ft \\
        --models-config configs/models.yaml \\
        --finetune-config configs/finetune.yaml \\
        --device mps
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import yaml
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Curriculum dataset helpers
# ---------------------------------------------------------------------------


def build_epoch_dataset(
    triplets_df: pl.DataFrame,
    hard_neg_ratio: float,
    anchor_col: str,
    positive_col: str,
    negative_col: str,
    seed: int = 42,
) -> "datasets.Dataset":
    """
    Build a HuggingFace Dataset for one training epoch.

    Samples the requested ratio of hard negatives (negative_source == "hard")
    vs random negatives. The total size equals the full dataset size.

    Parameters
    ----------
    triplets_df : polars.DataFrame
        Full triplets dataframe.
    hard_neg_ratio : float
        Fraction of triplets that should use hard negatives (0.0 to 1.0).
    anchor_col, positive_col, negative_col : str
        Column names for anchor, positive, and negative texts.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    datasets.Dataset with columns: anchor, positive, negative
    """
    from datasets import Dataset

    n_total = len(triplets_df)

    hard_df = triplets_df.filter(pl.col("negative_source") == "hard")
    random_df = triplets_df.filter(pl.col("negative_source") != "hard")

    n_hard_target = min(int(n_total * hard_neg_ratio), len(hard_df))
    n_random_target = n_total - n_hard_target

    # Sample hard negatives
    hard_sample = (
        hard_df.sample(n=n_hard_target, shuffle=True, seed=seed)
        if n_hard_target > 0 and len(hard_df) > 0
        else hard_df.clear()
    )

    # Sample random negatives (with replacement if needed)
    if n_random_target > 0 and len(random_df) > 0:
        if n_random_target <= len(random_df):
            random_sample = random_df.sample(n=n_random_target, shuffle=True, seed=seed + 1)
        else:
            # Need more random negatives than available: oversample
            reps = (n_random_target // len(random_df)) + 1
            random_sample = pl.concat([random_df] * reps).sample(
                n=n_random_target, shuffle=True, seed=seed + 1
            )
    else:
        random_sample = random_df.clear()

    combined = pl.concat([hard_sample, random_sample]).sample(
        fraction=1.0, shuffle=True, seed=seed + 2
    )

    # Apply prefixes to anchor (query) and positive/negative (docs) if needed
    return Dataset.from_dict(
        {
            "anchor": combined[anchor_col].to_list(),
            "positive": combined[positive_col].to_list(),
            "negative": combined[negative_col].to_list(),
        }
    )


def apply_text_prefix(
    df: pl.DataFrame,
    col: str,
    prefix: str,
    new_col: str | None = None,
) -> pl.DataFrame:
    """Add a string prefix to all values in a text column."""
    target_col = new_col or col
    return df.with_columns(
        (pl.lit(prefix) + pl.col(col)).alias(target_col)
    )


# ---------------------------------------------------------------------------
# Curriculum callback
# ---------------------------------------------------------------------------


def make_curriculum_callback(
    triplets_df: pl.DataFrame,
    curriculum_ratios: list[float],
    anchor_col: str,
    positive_col: str,
    negative_col: str,
    seed: int = 42,
):
    """
    Returns a HuggingFace TrainerCallback that swaps the training dataset
    at the start of each epoch to match the curriculum hard_neg_ratio.

    The callback holds a reference to the trainer and mutates
    trainer.train_dataset between epochs.
    """
    from transformers import TrainerCallback

    class CurriculumCallback(TrainerCallback):
        def __init__(self):
            self.trainer_ref = None  # Set after trainer construction

        def on_epoch_begin(self, args, state, control, **kwargs):
            epoch_idx = int(state.epoch)
            if epoch_idx == 0:
                return  # Epoch 1 dataset was set at construction time
            if epoch_idx < len(curriculum_ratios):
                ratio = curriculum_ratios[epoch_idx]
                console.print(
                    f"[bold cyan]Curriculum: epoch {epoch_idx + 1} hard_neg_ratio={ratio:.0%}"
                )
                new_dataset = build_epoch_dataset(
                    triplets_df=triplets_df,
                    hard_neg_ratio=ratio,
                    anchor_col=anchor_col,
                    positive_col=positive_col,
                    negative_col=negative_col,
                    seed=seed + epoch_idx,
                )
                if self.trainer_ref is not None:
                    self.trainer_ref.train_dataset = new_dataset
                    console.print(
                        f"[green]Swapped dataset for epoch {epoch_idx + 1}: "
                        f"{len(new_dataset):,} triplets"
                    )

    return CurriculumCallback()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune an embedding model with Matryoshka + MNRL loss."
    )
    parser.add_argument("--model", required=True, help="Model key from models.yaml")
    parser.add_argument(
        "--serialization", required=True, choices=["pipe", "kv"], help="Serialization format"
    )
    parser.add_argument("--triplets", required=True, help="Parquet file with training triplets")
    parser.add_argument("--output-dir", required=True, help="Output directory for model checkpoint")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--finetune-config", default="configs/finetune.yaml")
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda", "mps"], help="Compute device"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ---- Load configs ----
    with open(args.models_config) as f:
        all_model_cfg = yaml.safe_load(f)
    with open(args.finetune_config) as f:
        ft_cfg = yaml.safe_load(f)

    if args.model not in all_model_cfg:
        console.print(f"[red]Model '{args.model}' not found in {args.models_config}")
        sys.exit(1)

    model_cfg = all_model_cfg[args.model]
    hf_id: str = model_cfg["hf_id"]
    mrl_native: bool = bool(model_cfg.get("mrl_native", False))
    trust_remote_code: bool = args.model in ("gte_modernbert_base", "nomic_v15")

    # Determine Matryoshka dims
    if mrl_native:
        matryoshka_dims: list[int] = all_model_cfg.get(
            "matryoshka_dims_finetune", [768, 512, 256, 128, 64]
        )
    else:
        # bge_small: add MRL via MatryoshkaLoss with bge dims
        matryoshka_dims = all_model_cfg.get(
            "matryoshka_dims_bge", [384, 256, 128, 64]
        )

    # Hyperparams from finetune.yaml
    epochs: int = int(ft_cfg.get("epochs", 3))
    batch_size: int = int(ft_cfg.get("batch_size", 256))
    learning_rate: float = float(ft_cfg.get("learning_rate", 2e-5))
    warmup_ratio: float = float(ft_cfg.get("warmup_ratio", 0.1))
    weight_decay: float = float(ft_cfg.get("weight_decay", 0.01))
    seed: int = int(ft_cfg.get("seed", args.seed))
    logging_steps: int = int(ft_cfg.get("logging_steps", 50))
    save_steps: int = int(ft_cfg.get("save_steps", 1000))
    save_total_limit: int = int(ft_cfg.get("save_total_limit", 3))
    mnrl_scale: float = float(ft_cfg.get("mnrl_scale", 20.0))
    gradient_accumulation_steps: int = int(ft_cfg.get("gradient_accumulation_steps", 1))
    gradient_checkpointing: bool = bool(ft_cfg.get("gradient_checkpointing", False))
    dataloader_num_workers: int = int(ft_cfg.get("dataloader_num_workers", 4))

    # Curriculum config
    curriculum_cfg = ft_cfg.get("curriculum", {})
    curriculum_ratios: list[float] = [
        float(curriculum_cfg.get("epoch1_hard_neg_ratio", 0.10)),
        float(curriculum_cfg.get("epoch2_hard_neg_ratio", 0.30)),
        float(curriculum_cfg.get("epoch3_hard_neg_ratio", 0.50)),
    ]
    # Extend to cover all epochs if needed
    while len(curriculum_ratios) < epochs:
        curriculum_ratios.append(curriculum_ratios[-1])

    # Text prefix handling for nomic
    raw_qp = model_cfg.get("query_prefix")
    raw_dp = model_cfg.get("doc_prefix")
    query_prefix: str | None = None
    doc_prefix: str | None = None
    if raw_qp and "system_prompt" not in str(raw_qp):
        query_prefix = str(raw_qp) + ": "
    if raw_dp and "system_prompt" not in str(raw_dp):
        doc_prefix = str(raw_dp) + ": "

    # ---- Load triplets ----
    console.print(f"[bold cyan]Loading triplets from {args.triplets}...")
    triplets_df = pl.read_parquet(args.triplets)
    console.print(f"[cyan]Loaded {len(triplets_df):,} triplets. Columns: {triplets_df.columns}")

    # Select text columns based on serialization
    col_suffix = args.serialization  # "pipe" or "kv"
    anchor_col = f"anchor_text_{col_suffix}"
    positive_col = f"positive_text_{col_suffix}"
    negative_col = f"negative_text_{col_suffix}"

    # Validate columns
    for col in (anchor_col, positive_col, negative_col):
        if col not in triplets_df.columns:
            console.print(f"[red]Column '{col}' not found. Available: {triplets_df.columns}")
            sys.exit(1)

    # Apply prefixes to triplet text columns
    if query_prefix:
        triplets_df = triplets_df.with_columns(
            (pl.lit(query_prefix) + pl.col(anchor_col)).alias(anchor_col)
        )
    if doc_prefix:
        triplets_df = triplets_df.with_columns(
            (pl.lit(doc_prefix) + pl.col(positive_col)).alias(positive_col),
            (pl.lit(doc_prefix) + pl.col(negative_col)).alias(negative_col),
        )

    console.print(f"[cyan]Using columns: {anchor_col}, {positive_col}, {negative_col}")
    console.print(f"[cyan]Curriculum ratios (per epoch): {curriculum_ratios[:epochs]}")

    # ---- Check for hard negatives column ----
    if "negative_source" not in triplets_df.columns:
        console.print(
            "[yellow]'negative_source' column not found; disabling curriculum "
            "(treating all negatives as random)."
        )
        # Add a dummy column so the curriculum code doesn't fail
        triplets_df = triplets_df.with_columns(
            pl.lit("random").alias("negative_source")
        )

    # ---- Load SentenceTransformer model ----
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss

    console.print(f"[bold cyan]Loading model: {hf_id} (device={args.device})...")
    model = SentenceTransformer(
        hf_id,
        trust_remote_code=trust_remote_code,
        device=args.device if args.device != "cpu" else None,
    )
    console.print(f"[green]Model loaded. Embedding dim: {model.get_sentence_embedding_dimension()}")
    # Detect actual device AFTER loading -- ST may auto-select MPS even when device=None
    actual_device_type = next(model.parameters()).device.type
    use_mps = actual_device_type == "mps"

    # ---- Setup loss ----
    inner_loss = MultipleNegativesRankingLoss(model=model, scale=mnrl_scale)
    loss = MatryoshkaLoss(
        model=model,
        loss=inner_loss,
        matryoshka_dims=matryoshka_dims,
    )
    console.print(f"[cyan]Loss: MatryoshkaLoss(MNRL) | matryoshka_dims={matryoshka_dims}")

    # ---- Build epoch 1 dataset ----
    initial_ratio = curriculum_ratios[0]
    console.print(
        f"[cyan]Building epoch 1 dataset (hard_neg_ratio={initial_ratio:.0%})..."
    )
    train_dataset = build_epoch_dataset(
        triplets_df=triplets_df,
        hard_neg_ratio=initial_ratio,
        anchor_col=anchor_col,
        positive_col=positive_col,
        negative_col=negative_col,
        seed=seed,
    )
    console.print(f"[green]Epoch 1 dataset: {len(train_dataset):,} triplets")

    # ---- Training arguments ----
    from sentence_transformers import SentenceTransformerTrainer
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # MPS-specific settings: no fp16/bf16 (not supported on MPS via Trainer)
    # use_mps already detected above from actual model device
    steps_per_epoch = len(train_dataset) // batch_size
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    train_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        seed=seed,
        fp16=False,
        bf16=False,
        dataloader_num_workers=dataloader_num_workers if not use_mps else 0,
        dataloader_pin_memory=False,  # Disable for MPS compatibility
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        report_to=[],  # Disable wandb/tensorboard for clean output; user can enable manually
        load_best_model_at_end=False,  # Disable to avoid needing eval_dataset
        run_name=f"{args.model}_{args.serialization}_ft",
    )

    # ---- Curriculum callback ----
    curriculum_cb = make_curriculum_callback(
        triplets_df=triplets_df,
        curriculum_ratios=curriculum_ratios,
        anchor_col=anchor_col,
        positive_col=positive_col,
        negative_col=negative_col,
        seed=seed,
    )

    # ---- Build trainer ----
    trainer = SentenceTransformerTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        loss=loss,
        callbacks=[curriculum_cb],
    )

    # Give the callback a reference to the trainer so it can swap datasets
    curriculum_cb.trainer_ref = trainer

    # ---- Train ----
    console.print(
        f"\n[bold green]Starting fine-tuning: {args.model} | "
        f"{len(train_dataset):,} triplets | {epochs} epochs | "
        f"batch={batch_size} | lr={learning_rate}"
    )

    t_start = time.perf_counter()
    trainer.train()
    training_time_sec = time.perf_counter() - t_start

    console.print(f"[bold green]Training complete in {training_time_sec / 60:.1f} min.")

    # ---- Save model ----
    model.save_pretrained(str(output_dir))
    console.print(f"[green]Model saved -> {output_dir}")

    # ---- Save training manifest ----
    manifest = {
        "model_key": args.model,
        "base_model": hf_id,
        "serialization": args.serialization,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "matryoshka_dims": matryoshka_dims,
        "triplet_count": len(triplets_df),
        "curriculum_ratios": curriculum_ratios[:epochs],
        "training_time_sec": round(training_time_sec, 2),
        "training_time_min": round(training_time_sec / 60, 2),
        "output_dir": str(output_dir),
        "device": args.device,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    manifest_path = output_dir / "training_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    console.print(f"[green]Training manifest saved -> {manifest_path}")
    console.print("[bold green]Fine-tuning complete.")


if __name__ == "__main__":
    main()
