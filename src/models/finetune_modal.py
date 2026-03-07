"""
finetune_modal.py -- Fine-tune embedding models on Modal A10G GPUs.

Runs all 3 fine-tune targets in parallel. Each job:
  1. Downloads triplets from HuggingFace Hub (dataset repo)
  2. Fine-tunes with MatryoshkaLoss + MNRL + curriculum hard negatives
  3. Logs all metrics to Weights & Biases
  4. Pushes fine-tuned model to HuggingFace Hub under jayshah5696/
  5. Saves checkpoint to Modal Volume for resume-on-failure

Usage (from your M3, one-time data upload):
    python src/models/finetune_modal.py upload-data

Usage (launch all 3 models in parallel):
    modal run src/models/finetune_modal.py::run_all

Usage (single model, useful for debugging):
    modal run src/models/finetune_modal.py::finetune_one --model-key gte_modernbert_base

Usage (resume a crashed run):
    modal run src/models/finetune_modal.py::finetune_one --model-key gte_modernbert_base --resume

Secrets required in Modal dashboard (modal.com -> Secrets):
  - "huggingface"  : HF_TOKEN=<your token>
  - "wandb"        : WANDB_API_KEY=<your key>

HF dataset repo: jayshah5696/entity-resolution-triplets  (created by upload-data)
HF model repos:  jayshah5696/er-{model_key}-pipe-ft      (created by each run)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal primitives
# ---------------------------------------------------------------------------

APP_NAME = "entity-resolution-finetune"
VOLUME_NAME = "entity-resolution-checkpoints"
HF_DATASET_REPO = "jayshah5696/entity-resolution-triplets"
HF_MODEL_PREFIX = "jayshah5696/er"
WANDB_PROJECT = "entity-resolution-poc"
WANDB_ENTITY = "jayshah5696"

# GPU: A10G -- 24GB VRAM, good for 149M models at batch=256. ~$0.90/hr on Modal.
GPU = "A10G"
TIMEOUT = 60 * 90  # 90 min per job (3 epochs on 600K triplets ~ 30-40min on A10G)

# Persistent volume for checkpoints -- survives container crashes
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
CHECKPOINT_ROOT = Path("/checkpoints")

# Container image -- pin versions to match your local env
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2",
        "sentence-transformers>=3.0",
        "accelerate>=1.1.0",
        "transformers>=4.40",
        "datasets>=2.18",
        "polars>=0.20",
        "pyarrow>=15.0",
        "huggingface-hub>=0.21",
        "wandb>=0.16",
        "pyyaml>=6.0",
        "rich>=13.7",
        "numpy>=1.26",
        "tqdm>=4.66",
    )
)

app = modal.App(APP_NAME, image=image)

# ---------------------------------------------------------------------------
# Model registry -- mirrors configs/models.yaml (inlined to avoid file deps)
# ---------------------------------------------------------------------------

MODELS = {
    "minilm_l6": {
        "hf_id": "sentence-transformers/all-MiniLM-L6-v2",
        "dims": [384, 256, 128, 64],
        "mrl_native": False,
        "trust_remote_code": False,
        "query_prefix": None,
        "doc_prefix": None,
    },
    "bge_small": {
        "hf_id": "BAAI/bge-small-en-v1.5",
        "dims": [384, 256, 128, 64],
        "mrl_native": False,
        "trust_remote_code": False,
        "query_prefix": None,
        "doc_prefix": None,
    },
    "gte_modernbert_base": {
        "hf_id": "Alibaba-NLP/gte-modernbert-base",
        "dims": [768, 512, 256, 128, 64],
        "mrl_native": True,
        "trust_remote_code": True,
        "query_prefix": None,
        "doc_prefix": None,
    },
    "nomic_v15": {
        "hf_id": "nomic-ai/nomic-embed-text-v1.5",
        "dims": [768, 512, 256, 128, 64],
        "mrl_native": True,
        "trust_remote_code": True,
        "query_prefix": "search_query",
        "doc_prefix": "search_document",
    },
    "pplx_embed_v1_06b": {
        "hf_id": "perplexity-ai/pplx-embed-v1-0.6b",
        "dims": [1536, 768, 256, 128, 64],
        "mrl_native": True,
        "trust_remote_code": True,
        "query_prefix": None,  # system prompts not used during training
        "doc_prefix": None,
    },
}

# Finetune hyperparams (matches finetune.yaml)
FINETUNE_CFG = {
    "epochs": 3,
    "batch_size": 256,          # A10G handles this fine, no cap needed
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "logging_steps": 50,
    "save_steps": 500,          # More frequent on GPU (fast)
    "save_total_limit": 2,
    "mnrl_scale": 20.0,
    "gradient_accumulation_steps": 1,
    "dataloader_num_workers": 4,
    "curriculum": [0.10, 0.30, 0.50],  # hard_neg_ratio per epoch
    "seed": 42,
}

# ---------------------------------------------------------------------------
# Core training function (runs inside Modal container)
# ---------------------------------------------------------------------------

@app.function(
    gpu=modal.gpu.A10G(),  # default; pplx gets A100 via per-call override below
    timeout=TIMEOUT,
    volumes={str(CHECKPOINT_ROOT): volume},
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb"),
    ],
    retries=1,
)
def finetune_one(model_key: str, resume: bool = False) -> dict:
    """
    Fine-tune a single embedding model on A10G.
    Returns a dict with training metrics and HF repo URL.
    """
    import json
    import time
    from datetime import datetime, timezone

    import polars as pl
    import wandb
    from datasets import Dataset
    from huggingface_hub import HfApi
    from rich.console import Console
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
    from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments
    from transformers import TrainerCallback

    console = Console()
    cfg = MODELS[model_key]
    ft = FINETUNE_CFG
    hf_token = os.environ["HF_TOKEN"]
    # Disable XetHub -- use standard LFS upload path inside Modal container
    os.environ["HF_HUB_DISABLE_XET"] = "1"

    hf_output_repo = f"{HF_MODEL_PREFIX}-{model_key.replace('_', '-')}-pipe-ft"
    checkpoint_dir = CHECKPOINT_ROOT / model_key
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold cyan]Starting: {model_key} | GPU={GPU} | resume={resume}")

    # ---- W&B init ----
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=f"{model_key}_pipe_ft",
        config={
            "model_key": model_key,
            "hf_id": cfg["hf_id"],
            "serialization": "pipe",
            "gpu": GPU,
            **ft,
        },
        resume="allow" if resume else None,
        tags=[model_key, "pipe", "matryoshka", "mnrl", "modal"],
    )

    # ---- Download triplets from HF Hub ----
    console.print(f"[cyan]Downloading triplets from {HF_DATASET_REPO}...")
    from huggingface_hub import hf_hub_download
    triplets_path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename="triplets.parquet",
        repo_type="dataset",
        token=hf_token,
    )
    triplets_df = pl.read_parquet(triplets_path)
    console.print(f"[green]Loaded {len(triplets_df):,} triplets")

    # ---- Column setup ----
    anchor_col = "anchor_text_pipe"
    positive_col = "positive_text_pipe"
    negative_col = "negative_text_pipe"

    for col in (anchor_col, positive_col, negative_col):
        if col not in triplets_df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {triplets_df.columns}")

    # Apply nomic prefixes
    if cfg["query_prefix"]:
        prefix = cfg["query_prefix"] + ": "
        triplets_df = triplets_df.with_columns(
            (pl.lit(prefix) + pl.col(anchor_col)).alias(anchor_col)
        )
    if cfg["doc_prefix"]:
        prefix = cfg["doc_prefix"] + ": "
        triplets_df = triplets_df.with_columns(
            (pl.lit(prefix) + pl.col(positive_col)).alias(positive_col),
            (pl.lit(prefix) + pl.col(negative_col)).alias(negative_col),
        )

    # ---- Ensure negative_source col exists ----
    if "negative_source" not in triplets_df.columns:
        triplets_df = triplets_df.with_columns(pl.lit("random").alias("negative_source"))

    # ---- Build epoch dataset helper ----
    def build_dataset(hard_neg_ratio: float, seed: int) -> Dataset:
        n_total = len(triplets_df)
        hard_df = triplets_df.filter(pl.col("negative_source") == "hard")
        random_df = triplets_df.filter(pl.col("negative_source") != "hard")
        n_hard = min(int(n_total * hard_neg_ratio), len(hard_df))
        n_rand = n_total - n_hard
        hard_sample = hard_df.sample(n=n_hard, shuffle=True, seed=seed) if n_hard > 0 else hard_df.clear()
        if n_rand > 0 and len(random_df) > 0:
            if n_rand <= len(random_df):
                rand_sample = random_df.sample(n=n_rand, shuffle=True, seed=seed + 1)
            else:
                reps = (n_rand // len(random_df)) + 1
                rand_sample = pl.concat([random_df] * reps).sample(n=n_rand, shuffle=True, seed=seed + 1)
        else:
            rand_sample = random_df.clear()
        combined = pl.concat([hard_sample, rand_sample]).sample(fraction=1.0, shuffle=True, seed=seed + 2)
        return Dataset.from_dict({
            "anchor": combined[anchor_col].to_list(),
            "positive": combined[positive_col].to_list(),
            "negative": combined[negative_col].to_list(),
        })

    # ---- Curriculum callback ----
    curriculum_ratios = ft["curriculum"]
    seed = ft["seed"]

    class CurriculumCallback(TrainerCallback):
        def __init__(self):
            self.trainer_ref = None

        def on_epoch_begin(self, args, state, control, **kwargs):
            epoch_idx = int(state.epoch)
            if epoch_idx == 0:
                return
            if epoch_idx < len(curriculum_ratios):
                ratio = curriculum_ratios[epoch_idx]
                console.print(f"[bold cyan]Curriculum epoch {epoch_idx + 1}: hard_neg_ratio={ratio:.0%}")
                new_ds = build_dataset(ratio, seed=seed + epoch_idx)
                if self.trainer_ref is not None:
                    self.trainer_ref.train_dataset = new_ds
                    console.print(f"[green]Swapped dataset: {len(new_ds):,} triplets")

    curriculum_cb = CurriculumCallback()

    # ---- Load model ----
    console.print(f"[bold cyan]Loading model: {cfg['hf_id']}...")
    model = SentenceTransformer(
        cfg["hf_id"],
        trust_remote_code=cfg["trust_remote_code"],
        token=hf_token,
    )
    embed_dim = model.get_sentence_embedding_dimension()
    console.print(f"[green]Model loaded. dim={embed_dim}")

    # ---- Loss ----
    inner_loss = MultipleNegativesRankingLoss(model=model, scale=ft["mnrl_scale"])
    loss = MatryoshkaLoss(model=model, loss=inner_loss, matryoshka_dims=cfg["dims"])
    console.print(f"[cyan]Loss: MatryoshkaLoss(MNRL) | dims={cfg['dims']}")

    # ---- Epoch 1 dataset ----
    train_dataset = build_dataset(curriculum_ratios[0], seed=seed)
    console.print(f"[green]Epoch 1 dataset: {len(train_dataset):,} triplets")

    # ---- Training args ----
    epochs = ft["epochs"]
    # pplx is 600M params -- reduce batch to fit A10G 24GB with fp16
    batch_size = 64 if model_key == "pplx_embed_v1_06b" else ft["batch_size"]
    grad_accum = 4 if model_key == "pplx_embed_v1_06b" else ft["gradient_accumulation_steps"]
    # effective batch stays 256 for pplx (64 * 4)
    steps_per_epoch = len(train_dataset) // batch_size
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(total_steps * ft["warmup_ratio"]))

    # Check for existing checkpoint to resume from
    resume_from = None
    if resume:
        checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"))
        if checkpoints:
            resume_from = str(checkpoints[-1])
            console.print(f"[yellow]Resuming from: {resume_from}")

    train_args = SentenceTransformerTrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=ft["learning_rate"],
        warmup_steps=warmup_steps,
        weight_decay=ft["weight_decay"],
        logging_steps=ft["logging_steps"],
        save_steps=ft["save_steps"],
        save_total_limit=ft["save_total_limit"],
        seed=seed,
        fp16=True,   # A10G supports fp16, 2x speedup
        bf16=False,
        dataloader_num_workers=ft["dataloader_num_workers"],
        dataloader_pin_memory=True,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=False,
        report_to=["wandb"],
        load_best_model_at_end=False,
        run_name=f"{model_key}_pipe_ft",
    )

    # ---- Trainer ----
    trainer = SentenceTransformerTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        loss=loss,
        callbacks=[curriculum_cb],
    )
    curriculum_cb.trainer_ref = trainer

    # ---- Train ----
    console.print(
        f"\n[bold green]Training: {model_key} | "
        f"{len(train_dataset):,} triplets | {epochs} epochs | "
        f"batch={batch_size} | lr={ft['learning_rate']} | fp16=True"
    )
    t_start = time.perf_counter()
    trainer.train(resume_from_checkpoint=resume_from)
    training_time_sec = time.perf_counter() - t_start
    console.print(f"[bold green]Training complete in {training_time_sec / 60:.1f} min.")

    # ---- Commit checkpoints to volume ----
    volume.commit()

    # ---- Push to HF Hub ----
    console.print(f"[bold cyan]Pushing to HF Hub: {hf_output_repo}...")
    model.save_pretrained(str(checkpoint_dir / "final"))
    api = HfApi(token=hf_token)
    api.create_repo(repo_id=hf_output_repo, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(
        folder_path=str(checkpoint_dir / "final"),
        repo_id=hf_output_repo,
        repo_type="model",
        commit_message=f"Fine-tuned {cfg['hf_id']} on entity-resolution triplets (pipe, {epochs} epochs)",
    )
    hf_url = f"https://huggingface.co/{hf_output_repo}"
    console.print(f"[bold green]Model pushed -> {hf_url}")

    # ---- Training manifest ----
    manifest = {
        "model_key": model_key,
        "base_model": cfg["hf_id"],
        "serialization": "pipe",
        "hf_repo": hf_output_repo,
        "hf_url": hf_url,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": ft["learning_rate"],
        "matryoshka_dims": cfg["dims"],
        "triplet_count": len(triplets_df),
        "curriculum_ratios": curriculum_ratios[:epochs],
        "training_time_sec": round(training_time_sec, 2),
        "training_time_min": round(training_time_sec / 60, 2),
        "gpu": GPU,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(checkpoint_dir / "training_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    volume.commit()

    # ---- Log final metrics to W&B ----
    wandb.log({
        "training_time_min": manifest["training_time_min"],
        "hf_repo": hf_output_repo,
    })
    wandb.finish()

    console.print(f"[bold green]Done: {model_key} -> {hf_url}")
    return manifest


# ---------------------------------------------------------------------------
# Launch all 3 fine-tune targets in parallel
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def run_all():
    """Launch all 5 models in parallel on Modal.

    GPU allocation:
      minilm_l6            -> A10G (22M params,  fast ~5min)
      bge_small            -> A10G (33M params,  fast ~10min)
      gte_modernbert_base  -> A10G (149M params, ~25min)
      nomic_v15            -> A10G (137M params, ~25min)
      pplx_embed_v1_06b   -> A10G (600M params, ~60min, fp16 batch=64)

    All 5 run simultaneously. Total wall time ~ 60min. Cost ~$5-7.
    """
    targets = list(MODELS.keys())
    print(f"Launching {len(targets)} parallel fine-tune jobs: {targets}")
    print(f"W&B project: {WANDB_PROJECT} | HF prefix: {HF_MODEL_PREFIX}")
    print("Monitor at: https://wandb.ai/jayshah5696/entity-resolution-poc")
    print()

    # starmap launches all 5 simultaneously
    results = list(finetune_one.starmap(
        [(key, False) for key in targets]
    ))

    print("\n" + "=" * 60)
    print("ALL 5 RUNS COMPLETE")
    print("=" * 60)
    for r in results:
        print(f"  {r['model_key']:30s} -> {r['hf_url']}")
        print(f"    Training time: {r['training_time_min']:.1f} min")
    print()
    print("Next: run offline eval with:")
    print("  python src/eval/run_eval.py --model <key> --index-dir results/indexes/...")


# ---------------------------------------------------------------------------
# Data upload: DO NOT use Modal for this.
# Just run directly on your M3:
#
#   hf upload jayshah5696/entity-resolution-triplets \
#       data/triplets/triplets.parquet triplets.parquet --repo-type dataset
#
# Or use the standalone script:
#   python src/models/upload_triplets.py
# ---------------------------------------------------------------------------
