"""
build_index.py -- Encode all index profiles and store in LanceDB (or BM25 pickle).

Usage:
    uv run python src/eval/build_index.py \\
        --model gte_modernbert_base \\
        --serialization pipe \\
        --quantization fp32 \\
        --index-profiles data/processed/index.parquet \\
        --output-dir results/indexes/gte_modernbert_base_pipe_fp32 \\
        --models-config configs/models.yaml \\
        --batch-size 128 \\
        --device mps

For BM25:
    uv run python src/eval/build_index.py \\
        --model bm25_baseline \\
        --serialization pipe \\
        --index-profiles data/processed/index.parquet \\
        --output-dir results/indexes/bm25_pipe \\
        --models-config configs/models.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

# Ensure src is importable when run with uv run python
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.serialize import serialize
from src.models.encoder import BM25Encoder, load_encoder

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------


def apply_quantization(vecs: np.ndarray, mode: str) -> np.ndarray:
    """
    Apply quantization to a float32 embedding matrix.

    Parameters
    ----------
    vecs : np.ndarray, shape (N, D), dtype float32
        Input embeddings (assumed L2-normalized).
    mode : str
        "fp32"   -- no-op, return as-is.
        "int8"   -- scale each vector to [-128, 127], store as float32.
        "binary" -- sign quantization: values become +1.0 or -1.0.

    Returns
    -------
    np.ndarray, dtype float32
    """
    if mode == "fp32":
        return vecs.astype(np.float32)

    if mode == "int8":
        # Scale per-vector to fill [-128, 127]
        abs_max = np.abs(vecs).max(axis=1, keepdims=True)
        abs_max = np.where(abs_max == 0, 1.0, abs_max)
        scaled = np.round(vecs / abs_max * 127).clip(-128, 127).astype(np.float32)
        return scaled

    if mode == "binary":
        return np.sign(vecs).astype(np.float32)

    raise ValueError(f"Unknown quantization mode: {mode!r}. Use 'fp32', 'int8', or 'binary'.")


# ---------------------------------------------------------------------------
# LanceDB helpers
# ---------------------------------------------------------------------------


def create_lance_table(db, table_name: str, dim: int):
    """Create a LanceDB table with entity_id, text, vector columns."""
    import pyarrow as pa

    schema = pa.schema(
        [
            pa.field("entity_id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ]
    )
    # Drop existing table if present
    try:
        db.drop_table(table_name)
    except Exception:
        pass
    table = db.create_table(table_name, schema=schema)
    return table


def build_lance_ann_index(table, dim: int) -> None:
    """
    Create an approximate nearest-neighbor index on the LanceDB table.
    Tries IVF_PQ first; falls back gracefully if unavailable.
    """
    n_rows = table.count_rows()
    if n_rows < 256:
        console.print("[yellow]Too few rows for ANN index, skipping.")
        return

    # Choose partition count based on corpus size
    num_partitions = min(256, max(8, n_rows // 4000))
    num_sub_vectors = min(64, max(1, dim // 16))

    console.print(
        f"[bold cyan]Building ANN index (IVF_PQ) "
        f"partitions={num_partitions} sub_vectors={num_sub_vectors}..."
    )
    try:
        table.create_index(
            metric="cosine",
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
        )
        console.print("[green]ANN index created.")
    except Exception as exc:
        console.print(f"[yellow]ANN index creation failed ({exc}); index will use brute force.")


# ---------------------------------------------------------------------------
# BM25 index building
# ---------------------------------------------------------------------------


def build_bm25_index(
    encoder: BM25Encoder,
    entity_ids: list[str],
    texts: list[str],
    output_dir: Path,
) -> None:
    """Tokenize texts, build BM25Okapi, save to disk."""
    console.print(f"[bold cyan]Building BM25 index for {len(texts):,} records...")

    encoder.set_entity_ids(entity_ids)
    encoder.encode_docs(texts)  # builds internal BM25 index

    bm25_path = output_dir / "bm25.pkl"
    entity_ids_path = output_dir / "entity_ids.json"
    encoder.save(str(bm25_path), str(entity_ids_path))

    console.print(f"[green]BM25 index saved -> {bm25_path}")
    console.print(f"[green]Entity IDs saved -> {entity_ids_path}")


# ---------------------------------------------------------------------------
# Dense index building
# ---------------------------------------------------------------------------


def build_dense_index(
    encoder,
    entity_ids: list[str],
    texts: list[str],
    output_dir: Path,
    quantization: str,
    batch_size: int,
    write_batch_size: int = 10_000,
) -> None:
    """Encode texts in batches, quantize, write to LanceDB."""
    import lancedb

    dim = encoder.dim
    console.print(
        f"[bold cyan]Encoding {len(texts):,} records | dim={dim} | quant={quantization}"
    )

    # Encode all documents
    all_vecs = encoder.encode_docs(texts, batch_size=batch_size)
    console.print(f"[green]Encoding complete. Shape: {all_vecs.shape}")

    # Apply quantization
    all_vecs = apply_quantization(all_vecs, quantization)
    console.print(f"[cyan]Quantization '{quantization}' applied.")

    # Open / create LanceDB
    db = lancedb.connect(str(output_dir))
    table = create_lance_table(db, "index", dim)

    # Write in batches
    n = len(entity_ids)
    n_batches = (n + write_batch_size - 1) // write_batch_size
    console.print(f"[bold cyan]Writing {n:,} rows to LanceDB in {n_batches} batches...")

    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Writing to LanceDB", total=n)
        for i in range(0, n, write_batch_size):
            batch_eids = entity_ids[i : i + write_batch_size]
            batch_texts = texts[i : i + write_batch_size]
            batch_vecs = all_vecs[i : i + write_batch_size]

            rows = [
                {
                    "entity_id": eid,
                    "text": t,
                    "vector": v.tolist(),
                }
                for eid, t, v in zip(batch_eids, batch_texts, batch_vecs)
            ]
            table.add(rows)
            progress.advance(task, len(batch_eids))

    console.print(f"[green]All {n:,} rows written.")

    # Build ANN index
    build_lance_ann_index(table, dim)


# ---------------------------------------------------------------------------
# Index size calculation
# ---------------------------------------------------------------------------


def get_dir_size_mb(path: Path) -> float:
    """Recursively compute directory size in MB."""
    total = 0
    for root, dirs, files in os.walk(path):
        for fname in files:
            fp = Path(root) / fname
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total / (1024 * 1024)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode index profiles and store in LanceDB or BM25 pickle."
    )
    parser.add_argument("--model", required=True, help="Model key from models.yaml")
    parser.add_argument(
        "--serialization", required=True, choices=["pipe", "kv"], help="Serialization format"
    )
    parser.add_argument(
        "--quantization",
        default="fp32",
        choices=["fp32", "int8", "binary"],
        help="Quantization mode (dense models only)",
    )
    parser.add_argument(
        "--index-profiles",
        required=True,
        help="Parquet file with index profiles",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for index")
    parser.add_argument(
        "--models-config", default="configs/models.yaml", help="Path to models.yaml"
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Encoding batch size")
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda", "mps"], help="Compute device"
    )
    parser.add_argument(
        "--model-path", default=None, help="Local path to fine-tuned model checkpoint"
    )
    parser.add_argument(
        "--truncate-dim",
        type=int,
        default=None,
        help="Truncate MRL embeddings to this many dimensions",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ---- Load configs ----
    with open(args.models_config) as f:
        all_cfg = yaml.safe_load(f)

    if args.model not in all_cfg:
        console.print(f"[red]Model '{args.model}' not found in {args.models_config}")
        sys.exit(1)

    model_cfg = all_cfg[args.model]
    is_bm25 = model_cfg.get("type") == "bm25"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load index profiles ----
    console.print(f"[bold]Loading index profiles from {args.index_profiles}...")
    df = pl.read_parquet(args.index_profiles)
    console.print(f"[cyan]Loaded {len(df):,} profiles. Columns: {df.columns}")

    # ---- Serialize profiles ----
    console.print(f"[bold cyan]Serializing profiles with format='{args.serialization}'...")
    records = df.to_dicts()
    texts = [serialize(r, args.serialization) for r in records]
    entity_ids = df["entity_id"].to_list()
    console.print(f"[green]Serialization done. Example: {texts[0]!r}")

    # ---- Load encoder ----
    console.print(f"[bold cyan]Loading encoder: {args.model}...")
    encoder = load_encoder(
        model_key=args.model,
        model_cfg=model_cfg,
        device=args.device,
        model_path=args.model_path,
        truncate_dim=args.truncate_dim,
    )

    # ---- Build index ----
    t_start = time.perf_counter()

    if is_bm25:
        build_bm25_index(
            encoder=encoder,
            entity_ids=entity_ids,
            texts=texts,
            output_dir=output_dir,
        )
    else:
        build_dense_index(
            encoder=encoder,
            entity_ids=entity_ids,
            texts=texts,
            output_dir=output_dir,
            quantization=args.quantization,
            batch_size=args.batch_size,
        )

    elapsed = time.perf_counter() - t_start
    console.print(f"[bold green]Index built in {elapsed:.1f}s")

    # ---- Save metadata ----
    index_size_mb = get_dir_size_mb(output_dir)
    hf_id = model_cfg.get("hf_id", "")
    dim = encoder.dim
    if args.truncate_dim:
        dim = args.truncate_dim

    metadata = {
        "model_key": args.model,
        "hf_id": hf_id,
        "serialization": args.serialization,
        "quantization": args.quantization if not is_bm25 else "none",
        "dim": dim,
        "n_records": len(entity_ids),
        "index_size_mb": round(index_size_mb, 2),
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "device": args.device,
        "model_path": args.model_path,
        "build_time_sec": round(elapsed, 2),
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    console.print(f"[bold green]Metadata saved -> {metadata_path}")
    console.print(f"[bold green]Index size: {index_size_mb:.1f} MB")
    console.print("[bold green]Done.")


if __name__ == "__main__":
    main()
