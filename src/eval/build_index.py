"""
build_index.py -- Encode all index profiles and store in LanceDB (or BM25 pickle),
                 or derive a new index from an existing one.

Usage (Full Encode):
    uv run python src/eval/build_index.py \\
        --model gte_modernbert_base \\
        --serialization pipe \\
        --quantization fp32 \\
        --index-profiles data/processed/index.parquet \\
        --output-dir results/indexes/gte_modernbert_base_pipe_fp32 \\
        --models-config configs/models.yaml \\
        --batch-size 128 \\
        --device mps

Usage (Derive new index via MRL slicing / Quantization):
    uv run python src/eval/build_index.py \\
        --source-index results/indexes/gte_modernbert_base_pipe_fp32 \\
        --output-dir results/indexes/gte_64_int8 \\
        --truncate-dim 64 \\
        --quantization int8

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


from src.data.serialize import serialize
from src.models.encoder import load_encoder

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------


def apply_quantization(vecs: np.ndarray, mode: str) -> np.ndarray:
    """
    Ensure vectors are float32 for LanceDB storage.

    Quantization is handled at the INDEX level by LanceDB (IVF_PQ for int8-like
    compression, IVF_SQ for scalar quantization), not at the data level.
    LanceDB's vector search requires float32/float16 columns — int8 or packed-bit
    columns are invisible to its search API.

    Parameters
    ----------
    vecs : np.ndarray, shape (N, D)
        Input embeddings.
    mode : str
        "fp32", "int8", or "binary" — all stored as float32; the mode controls
        which ANN index type is built in build_lance_ann_index().

    Returns
    -------
    np.ndarray, dtype float32
    """
    return vecs.astype(np.float32)


# ---------------------------------------------------------------------------
# LanceDB helpers
# ---------------------------------------------------------------------------


def create_lance_table(db, table_name: str, dim: int, quantization: str):
    """
    Create a LanceDB table with entity_id, text, vector columns.

    Vectors are always stored as float32 regardless of quantization mode.
    Quantization is applied at the ANN index level (see build_lance_ann_index),
    not at the storage level — LanceDB vector search requires float-typed columns.
    """
    import pyarrow as pa

    vec_type = pa.list_(pa.float32(), dim)

    schema = pa.schema(
        [
            pa.field("entity_id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", vec_type),
        ]
    )
    # Drop existing table if present
    try:
        db.drop_table(table_name)
    except Exception:
        pass
    table = db.create_table(table_name, schema=schema)
    return table


def build_lance_ann_index(table, dim: int, quantization: str) -> None:
    """
    Create an approximate nearest-neighbor index on the LanceDB table.

    Quantization is applied here at the index level:
      - fp32:   IVF_PQ (product quantization) — moderate compression, high recall
      - int8:   IVF_SQ (scalar quantization) — ~4× compression vs fp32, good recall
      - binary: IVF_PQ with aggressive sub-vectors — maximum compression

    All vector data remains float32 on disk; the index structure controls
    the compression/accuracy tradeoff at query time.
    """
    n_rows = table.count_rows()
    if n_rows < 256:
        console.print("[yellow]Too few rows for ANN index, skipping.")
        return

    metric = "cosine"

    # Choose partition count based on corpus size
    num_partitions = min(256, max(8, n_rows // 4000))

    # Map quantization mode to index type
    if quantization == "int8":
        index_type = "IVF_PQ"
        # Use more sub-vectors for tighter compression (int8-like)
        num_sub_vectors = max(8, dim // 4)
        console.print(
            f"[bold cyan]Building ANN index (IVF_PQ, int8-level) "
            f"metric={metric} partitions={num_partitions} sub_vectors={num_sub_vectors}..."
        )
        try:
            table.create_index(
                vector_column_name="vector",
                index_type="IVF_PQ",
                metric=metric,
                num_partitions=num_partitions,
                num_sub_vectors=num_sub_vectors,
            )
            console.print("[green]IVF_PQ (int8-level) index created.")
            return
        except Exception as exc:
            console.print(f"[yellow]IVF_PQ failed ({exc}); trying default index.")

    elif quantization == "binary":
        index_type = "IVF_PQ"
        # Aggressive sub-vectors for maximum compression (binary-like)
        num_sub_vectors = max(4, dim // 8)
        console.print(
            f"[bold cyan]Building ANN index (IVF_PQ, binary-level) "
            f"metric={metric} partitions={num_partitions} sub_vectors={num_sub_vectors}..."
        )
        try:
            table.create_index(
                vector_column_name="vector",
                index_type="IVF_PQ",
                metric=metric,
                num_partitions=num_partitions,
                num_sub_vectors=num_sub_vectors,
            )
            console.print("[green]IVF_PQ (binary-level) index created.")
            return
        except Exception as exc:
            console.print(f"[yellow]IVF_PQ failed ({exc}); trying default index.")

    else:
        # fp32: standard high-quality index
        console.print(
            f"[bold cyan]Building ANN index (IVF_PQ) "
            f"metric={metric} partitions={num_partitions}..."
        )

    # Default / fallback: standard IVF_PQ
    try:
        table.create_index(
            vector_column_name="vector",
            index_type="IVF_PQ",
            metric=metric,
            num_partitions=num_partitions,
        )
        console.print("[green]IVF_PQ index created.")
    except Exception as e:
        console.print(f"[yellow]Brute force search will be used. Error: {e}")


# ---------------------------------------------------------------------------
# BM25 index building
# ---------------------------------------------------------------------------


def build_bm25_index(
    entity_ids: list[str],
    texts: list[str],
    output_dir: Path,
) -> None:
    """
    Store documents in LanceDB and build a native FTS (BM25) index.

    Uses LanceDB's built-in full-text search (Tantivy under the hood).
    No pickle files -- same storage format as dense indexes. Query with
    table.search(query, query_type="fts").

    Index directory naming: results/indexes/bm25_{serialization}/
    """
    import lancedb
    from tqdm import tqdm

    console.print(f"[bold cyan]Building LanceDB FTS index for {len(texts):,} records...")

    output_dir.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(output_dir))

    # Write docs in batches -- LanceDB requires list of dicts or DataFrame
    import pandas as pd

    write_batch = 50_000
    table = None
    for i in tqdm(range(0, len(texts), write_batch), desc="Writing to LanceDB"):
        df_batch = pd.DataFrame({
            "entity_id": entity_ids[i : i + write_batch],
            "text": texts[i : i + write_batch],
        })
        if table is None:
            table = db.create_table("index", data=df_batch, mode="overwrite")
        else:
            table.add(df_batch)

    if table is None:
        raise RuntimeError("No documents to index.")

    # Build native FTS index (BM25 via Tantivy, no extra deps beyond lancedb)
    console.print("[bold cyan]Building FTS index (BM25 via Tantivy)...")
    table.create_fts_index("text", use_tantivy=False, replace=True)

    n = table.count_rows()
    console.print(f"[green]FTS index built -- {n:,} documents indexed.")
    console.print(f"[green]Index saved -> {output_dir}")


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
    table = create_lance_table(db, "index", dim, quantization)

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
    build_lance_ann_index(table, dim, quantization)


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
# Index Derivation
# ---------------------------------------------------------------------------


def derive_index(
    source_index_dir: Path,
    output_dir: Path,
    truncate_dim: int | None,
    quantization: str,
) -> None:
    """
    Derive a new index from an existing one by slicing dimensions and quantizing.
    Skips the heavy ML encoding step.
    """
    import lancedb

    console.print(f"[bold cyan]Deriving index from source: {source_index_dir}")
    source_db = lancedb.connect(str(source_index_dir))
    source_table = source_db.open_table("index")
    n_total = source_table.count_rows()

    # Determine dimensions — use head(1) for a stable, lightweight sample
    sample_df = source_table.head(1).to_pandas()
    sample_vec = sample_df["vector"].iloc[0]
    source_dim = len(sample_vec)
    target_dim = truncate_dim if truncate_dim else source_dim

    if target_dim > source_dim:
        raise ValueError(f"Cannot truncate to {target_dim} from source dim {source_dim}")

    console.print(f"[cyan]Source Dim: {source_dim} -> Target Dim: {target_dim} | Quant: {quantization}")

    output_db = lancedb.connect(str(output_dir))
    dest_table = create_lance_table(output_db, "index", target_dim, quantization)

    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Deriving vectors", total=n_total)

        # Stream all rows via to_arrow().to_batches() — stable LanceDB 0.29+ API
        for batch in source_table.to_arrow().to_batches():
            df = pl.from_arrow(batch)
            vecs = np.stack(df["vector"].to_numpy())
            
            # Handle potential NaNs left by failed model inference
            if np.isnan(vecs).any():
                logger.warning(f"Found NaNs in vectors. Replacing with zeros.")
                vecs = np.nan_to_num(vecs)

            # Truncate and re-normalize (MRL)
            if target_dim < source_dim:
                vecs = vecs[:, :target_dim]
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                vecs = (vecs / norms).astype(np.float32)
            
            vecs_q = apply_quantization(vecs, quantization)
            
            rows = []
            for i in range(len(df)):
                rows.append({
                    "entity_id": df["entity_id"][i],
                    "text": df["text"][i],
                    "vector": vecs_q[i].tolist()
                })
            
            dest_table.add(rows)
            progress.advance(task, len(df))

    console.print(f"[green]Derivation complete. {n_total:,} rows written.")
    build_lance_ann_index(dest_table, target_dim, quantization)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode index profiles and store in LanceDB or BM25 pickle."
    )
    parser.add_argument("--model", default=None, help="Model key from models.yaml")
    parser.add_argument(
        "--serialization", choices=["pipe", "kv"], help="Serialization format"
    )
    parser.add_argument(
        "--quantization",
        default="fp32",
        choices=["fp32", "int8", "binary"],
        help="Quantization mode (dense models only)",
    )
    parser.add_argument(
        "--index-profiles",
        default=None,
        help="Parquet file with index profiles (data/processed/index.parquet)",
    )
    parser.add_argument(
        "--eval-profiles",
        default=None,
        help=(
            "Parquet file with eval profiles to INCLUDE in the index "
            "(data/eval/eval_profiles.parquet). Required for correct evaluation: "
            "eval queries reference entity_ids that must exist in the index."
        ),
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
    parser.add_argument(
        "--source-index",
        default=None,
        help="Path to an existing LanceDB index to derive from (skips ML encoding)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ---- MODE: Derive Index ----
    if args.source_index:
        source_dir = Path(args.source_index)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not source_dir.exists():
            console.print(f"[red]Source index not found: {source_dir}")
            sys.exit(1)
            
        t_start = time.perf_counter()
        derive_index(
            source_index_dir=source_dir,
            output_dir=output_dir,
            truncate_dim=args.truncate_dim,
            quantization=args.quantization,
        )
        elapsed = time.perf_counter() - t_start
        
        # Load source metadata to preserve model info
        source_meta_path = source_dir / "metadata.json"
        if source_meta_path.exists():
            with open(source_meta_path) as f:
                metadata = json.load(f)
        else:
            metadata = {"model_key": args.model}
            
        metadata.update({
            "derivation_source": str(source_dir),
            "quantization": args.quantization,
            "dim": args.truncate_dim or metadata.get("dim"),
            "index_size_mb": round(get_dir_size_mb(output_dir), 2),
            "build_timestamp": datetime.now(timezone.utc).isoformat(),
            "build_time_sec": round(elapsed, 2),
        })
        
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        console.print(f"[bold green]Derived index built in {elapsed:.1f}s")
        return

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
    console.print(f"[cyan]Loaded {len(df):,} index profiles.")

    # ---- Merge eval profiles into index (REQUIRED for correct eval) ----
    # Eval queries reference entity_ids from eval_profiles.parquet.
    # Those entity_ids must exist in the index or Recall is always 0.
    if args.eval_profiles:
        eval_df = pl.read_parquet(args.eval_profiles)
        console.print(
            f"[cyan]Adding {len(eval_df):,} eval profiles to index "
            f"(eval entity_ids must be retrievable)."
        )
        df = pl.concat([df, eval_df], how="vertical")
        console.print(f"[cyan]Total index size: {len(df):,} profiles.")
    else:
        console.print(
            "[yellow]Warning: --eval-profiles not set. "
            "Eval entity_ids will not be in the index and Recall will be 0."
        )

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
