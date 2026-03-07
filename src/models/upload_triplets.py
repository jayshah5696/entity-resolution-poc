"""
upload_triplets.py -- Upload triplets to HuggingFace Hub directly (no Modal).

Usage (from repo root):
    python src/models/upload_triplets.py

Or use hf CLI directly (2026 standard):
    hf auth login
    hf upload jayshah5696/entity-resolution-triplets \
        data/triplets/triplets.parquet triplets.parquet --repo-type dataset
"""

import sys
from pathlib import Path
from huggingface_hub import HfApi

TRIPLETS_PATH = "data/triplets/triplets.parquet"
HF_DATASET_REPO = "jayshah5696/entity-resolution-triplets"

path = Path(TRIPLETS_PATH)
if not path.exists():
    print(f"ERROR: {TRIPLETS_PATH} not found. Run from repo root.")
    sys.exit(1)

size_mb = path.stat().st_size / 1024 / 1024
print(f"Uploading {TRIPLETS_PATH} ({size_mb:.1f} MB) -> {HF_DATASET_REPO}")

api = HfApi()  # uses cached token from: hf auth login
api.create_repo(repo_id=HF_DATASET_REPO, repo_type="dataset", exist_ok=True, private=False)
api.upload_file(
    path_or_fileobj=TRIPLETS_PATH,
    path_in_repo="triplets.parquet",
    repo_id=HF_DATASET_REPO,
    repo_type="dataset",
    commit_message="Upload entity resolution triplets (600K, pipe+kv)",
)
print(f"Done -> https://huggingface.co/datasets/{HF_DATASET_REPO}")
