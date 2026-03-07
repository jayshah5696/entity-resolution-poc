# finetune_modal.py -- Fine-tune all 5 embedding models on Modal A10G GPUs in parallel.
#
# Usage:
#   uv run python3 -m modal run src/models/finetune_modal.py::run_all
#   uv run python3 -m modal run src/models/finetune_modal.py::finetune_one --model-key gte_modernbert_base
#   uv run python3 -m modal run src/models/finetune_modal.py::finetune_one --model-key gte_modernbert_base --resume
#
# Secrets required in Modal (modal.com -> Secrets):
#   "huggingface" : HF_TOKEN=<your token>
#   "wandb"       : WANDB_API_KEY=<your key>
#
# HF model repos: jayshah5696/er-{model_key}-pipe-ft

from __future__ import annotations

import os
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal primitives
# ---------------------------------------------------------------------------

APP_NAME = "entity-resolution-finetune"
VOLUME_NAME = "entity-resolution-checkpoints"

# GPU: A10G -- 24GB VRAM, good for 149M models at batch=256. ~$0.90/hr on Modal.
GPU = "A10G"
TIMEOUT = 60 * 90  # 90 min per job (3 epochs on 600K triplets ~ 30-40min on A10G)

# Persistent volume for checkpoints -- survives container crashes
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
CHECKPOINT_ROOT = Path("/checkpoints")

# Container image: debian_slim + Python 3.11 + uv for fast installs.
# Using .uv_pip_install() (Modal v1.1.0+) -- much faster than pip.
# torch 2.5.1 installed via CUDA 12.1 wheel index to get GPU support.
# flash-attn installed from pre-built wheel (v2.7.2.post1, cu12, torch2.5, cxx11abi=FALSE)
# -- no 20+ min compilation, instant install from ~190 MB wheel.
# Python 3.11 chosen because flash-attn publishes the widest wheel coverage for 3.11.
# torch 2.5.1 chosen because transformers >=4.48 requires torch >=2.4 (LRScheduler import).

# Pre-built flash-attn wheel pinned for reproducibility:
_FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/"
    "flash_attn-2.7.2.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
)

# Robust path resolution for the configs dir (works when __file__ is inside modal/tmp paths)
try:
    _repo_root = Path(__file__).resolve().parents[2]
except IndexError:
    # If __file__ lacks 3 levels of parents, fallback to cwd (which is the repo root for uv run)
    _repo_root = Path.cwd()

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .uv_pip_install(
        "torch==2.5.1",
        extra_options="--index-url https://download.pytorch.org/whl/cu121",
    )
    .uv_pip_install(
        _FLASH_ATTN_WHEEL,
    )
    .uv_pip_install(
        "sentence-transformers>=3.3",
        "accelerate>=1.2.0",
        "transformers>=4.47",
        "datasets>=2.18",
        "einops>=0.7",
        "scipy>=1.12",
        "polars>=0.20",
        "pyarrow>=15.0",
        "huggingface-hub>=0.27",
        "wandb>=0.16",
        "pyyaml>=6.0",
        "rich>=13.7",
        "numpy>=1.26",
        "tqdm>=4.66",
    )
    .add_local_python_source("src")
    .add_local_dir(
        local_path=str(_repo_root / "configs"),
        remote_path="/configs"
    )
)

app = modal.App(APP_NAME, image=image)

# ---------------------------------------------------------------------------
# Load Config
# ---------------------------------------------------------------------------

# We do this at the module level so the modal app configuration (e.g., volume, app)
# can use the config values during deployment.
try:
    from src.models.finetune_config import load_config
    CONFIG = load_config()
except Exception as e:
    # Handle the fact that this script might be evaluated within the Modal cloud
    # environment where 'configs/finetune_modal.yaml' isn't initially present
    # at module load time (it gets mounted later for the function).
    print(f"Warning: Could not load config at module level: {e}")
    CONFIG = None

# Fallbacks for app definition in case config isn't available locally
APP_NAME = getattr(CONFIG.modal, "app_name", "entity-resolution-finetune") if CONFIG else "entity-resolution-finetune"
VOLUME_NAME = getattr(CONFIG.modal, "volume_name", "entity-resolution-checkpoints") if CONFIG else "entity-resolution-checkpoints"
GPU = getattr(CONFIG.modal, "gpu", "A10G") if CONFIG else "A10G"
TIMEOUT = getattr(CONFIG.modal, "timeout_min", 90) * 60 if CONFIG else 90 * 60

# ---------------------------------------------------------------------------
# Core training function (runs inside Modal container)
# ---------------------------------------------------------------------------

@app.function(
    gpu=GPU,
    timeout=TIMEOUT,
    volumes={str(CHECKPOINT_ROOT): volume},
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb"),
    ],
    retries=0,  # fail fast -- retries=1 would restart from scratch, wasting compute
)
def finetune_one(model_key: str, resume: bool = False) -> dict:
    """
    Fine-tune a single embedding model.
    Returns a dict with training metrics and HF repo URL.
    """
    import json
    import time
    from datetime import datetime, timezone

    import polars as pl
    import wandb
    from datasets import Dataset
    from huggingface_hub import HfApi, hf_hub_download
    from rich.console import Console
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
    from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments
    from src.models.finetune_config import load_config
    
    console = Console()
    
    # Load config from the mounted directory inside Modal
    config = load_config("/configs/finetune_modal.yaml")
    cfg = config.resolve(model_key)
    
    hf_token = os.environ["HF_TOKEN"]
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    
    hf_output_repo = f"{config.hf_model_prefix}-{model_key.replace('_', '-')}-pipe-ft"
    checkpoint_dir = CHECKPOINT_ROOT / model_key
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold cyan]Starting: {model_key} | GPU={GPU} | resume={resume}")

    # ---- W&B init ----
    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=f"{model_key}_pipe_ft",
        config=cfg.model_dump(),
        resume="allow" if resume else None,
        tags=[model_key, "pipe", "matryoshka", "mnrl", "modal"],
    )

    # ---- Download triplets from HF Hub ----
    console.print(f"[cyan]Downloading triplets from {config.hf_dataset_repo}...")
    triplets_path = hf_hub_download(
        repo_id=config.hf_dataset_repo,
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
    if cfg.query_prefix:
        prefix = cfg.query_prefix + ": "
        triplets_df = triplets_df.with_columns(
            (pl.lit(prefix) + pl.col(anchor_col)).alias(anchor_col)
        )
    if cfg.doc_prefix:
        prefix = cfg.doc_prefix + ": "
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

    curriculum_ratios = cfg.curriculum
    seed = cfg.seed

    # ---- Curriculum trainer subclass ----
    # NOTE: mutating trainer.train_dataset mid-training does NOT work -- HF Trainer
    # calls get_train_dataloader() ONCE before the loop. Must override it instead.
    class CurriculumTrainer(SentenceTransformerTrainer):
        def __init__(self, *args, curriculum_ratios, seed, build_dataset_fn, **kwargs):
            super().__init__(*args, **kwargs)
            self._curriculum_ratios = curriculum_ratios
            self._seed = seed
            self._build_fn = build_dataset_fn
            self._epoch_idx = 0

        def get_train_dataloader(self):
            ratio = self._curriculum_ratios[min(self._epoch_idx, len(self._curriculum_ratios) - 1)]
            console.print(f"[bold cyan]Curriculum epoch {self._epoch_idx + 1}: hard_neg_ratio={ratio:.0%}")
            self.train_dataset = self._build_fn(ratio, seed=self._seed + self._epoch_idx)
            console.print(f"[green]Dataset: {len(self.train_dataset):,} triplets")
            self._epoch_idx += 1
            return super().get_train_dataloader()

    # ---- Load model ----
    console.print(f"[bold cyan]Loading model: {cfg.hf_id}...")
    model = SentenceTransformer(
        cfg.hf_id,
        trust_remote_code=cfg.trust_remote_code,
        token=hf_token,
    )
    console.print(f"[green]Model loaded. dim={model.get_sentence_embedding_dimension()}")

    # ---- Loss ----
    inner_loss = MultipleNegativesRankingLoss(model=model, scale=cfg.mnrl_scale)
    loss = MatryoshkaLoss(model=model, loss=inner_loss, matryoshka_dims=cfg.dims)
    console.print(f"[cyan]Loss: MatryoshkaLoss(MNRL) | dims={cfg.dims}")

    # ---- Epoch 1 dataset (also used as placeholder for trainer init) ----
    train_dataset = build_dataset(curriculum_ratios[0], seed=seed)
    console.print(f"[green]Initial dataset: {len(train_dataset):,} triplets")

    # ---- Training args ----
    # warmup steps based on optimizer steps (accounts for grad_accum)
    optimizer_steps_per_epoch = len(train_dataset) // cfg.effective_batch_size
    total_optimizer_steps = optimizer_steps_per_epoch * cfg.epochs
    warmup_steps = max(1, int(total_optimizer_steps * cfg.warmup_ratio))

    # Check for existing checkpoint to resume from -- sort numerically not lexicographically
    resume_from = None
    if resume:
        checkpoints = sorted(
            checkpoint_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if checkpoints:
            resume_from = str(checkpoints[-1])
            console.print(f"[yellow]Resuming from: {resume_from}")

    train_args = SentenceTransformerTrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        seed=cfg.seed,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_pin_memory=True,
        gradient_accumulation_steps=cfg.grad_accum,
        gradient_checkpointing=cfg.grad_checkpointing,
        report_to=["wandb"],
        load_best_model_at_end=False,
        run_name=f"{model_key}_pipe_ft",
    )

    # ---- Trainer (curriculum-aware subclass) ----
    trainer = CurriculumTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        loss=loss,
        curriculum_ratios=curriculum_ratios,
        seed=seed,
        build_dataset_fn=build_dataset,
    )

    # ---- Train ----
    console.print(
        f"\n[bold green]Training: {model_key} | {cfg.epochs} epochs | "
        f"batch={cfg.batch_size} x accum={cfg.grad_accum} (eff={cfg.effective_batch_size}) | "
        f"lr={cfg.learning_rate} | bf16={cfg.bf16}"
    )
    t_start = time.perf_counter()
    trainer.train(resume_from_checkpoint=resume_from)
    training_time_sec = time.perf_counter() - t_start
    console.print(f"[bold green]Training complete in {training_time_sec / 60:.1f} min.")

    # ---- Save final model to volume FIRST, then push to HF ----
    # Order matters: commit before upload so final model is safe if upload fails
    console.print(f"[bold cyan]Saving final model to volume...")
    model.save_pretrained(str(checkpoint_dir / "final"))
    volume.commit()  # final model persisted before attempting HF upload

    # ---- Push to HF Hub ----
    console.print(f"[bold cyan]Pushing to HF Hub: {hf_output_repo}...")
    api = HfApi(token=hf_token)
    api.create_repo(repo_id=hf_output_repo, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(
        folder_path=str(checkpoint_dir / "final"),
        repo_id=hf_output_repo,
        repo_type="model",
        commit_message=f"Fine-tuned {cfg.hf_id} on entity-resolution triplets (pipe, {cfg.epochs} epochs)",
    )
    hf_url = f"https://huggingface.co/{hf_output_repo}"
    console.print(f"[bold green]Model pushed -> {hf_url}")

    # ---- Training manifest ----
    manifest = {
        "model_key": model_key,
        "base_model": cfg.hf_id,
        "serialization": "pipe",
        "hf_repo": hf_output_repo,
        "hf_url": hf_url,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "matryoshka_dims": cfg.dims,
        "triplet_count": len(triplets_df),
        "curriculum_ratios": curriculum_ratios[:cfg.epochs],
        "training_time_sec": round(training_time_sec, 2),
        "training_time_min": round(training_time_sec / 60, 2),
        "gpu": getattr(config.modal, "gpu", "A10G"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(checkpoint_dir / "training_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    volume.commit()  # persist manifest

    # ---- Log final metrics to W&B ----
    wandb.log({
        "training_time_min": manifest["training_time_min"],
        "hf_repo": hf_output_repo,
    })
    wandb.finish()

    console.print(f"[bold green]Done: {model_key} -> {hf_url}")
    return manifest


# ---------------------------------------------------------------------------
# Launch all 5 fine-tune targets in parallel
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def run_all():
    """Launch all configured models in parallel on Modal.

    GPU allocation: defined in configs/finetune_modal.yaml
    """
    if not CONFIG:
        print("Error: Could not load config. Cannot run_all().")
        return

    targets = CONFIG.model_keys
    print(f"Launching {len(targets)} parallel fine-tune jobs: {targets}")
    print(f"W&B project: {CONFIG.wandb_project} | HF prefix: {CONFIG.hf_model_prefix}")
    print(f"Monitor at: https://wandb.ai/{CONFIG.wandb_entity}/{CONFIG.wandb_project}")
    print()

    # starmap launches all targets simultaneously
    # return_exceptions=True: one failure does NOT cancel other jobs
    results = list(finetune_one.starmap(
        [(key, False) for key in targets],
        return_exceptions=True,
        wrap_returned_exceptions=False,
    ))

    print("\n" + "=" * 60)
    print("ALL 5 RUNS COMPLETE")
    print("=" * 60)
    for r in results:
        if isinstance(r, Exception):
            print(f"  FAILED: {r}")
        else:
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
