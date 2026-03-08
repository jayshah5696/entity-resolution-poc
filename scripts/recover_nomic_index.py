"""
recover_nomic_index.py -- Recovery script for the interrupted nomic_v15 run.

This script opens the existing un-indexed LanceDB table that finished encoding,
builds the IVF_HNSW index on it, and writes the missing metadata.json file so 
that `run_eval.py` can be executed.
"""
import lancedb
import json
from pathlib import Path
from datetime import datetime, timezone
import sys

from src.eval.build_index import build_lance_ann_index, get_dir_size_mb

def main():
    output_dir = Path("results/indexes/nomic_v15_ft_pipe")
    if not output_dir.exists():
        print(f"Directory {output_dir} not found.")
        sys.exit(1)
        
    print(f"Connecting to LanceDB at {output_dir}...")
    db = lancedb.connect(str(output_dir))
    table = db.open_table("index")
    
    print(f"Found {table.count_rows():,} rows. Building IVF_HNSW index...")
    # Using the fixed function which passes down vector_column_name
    build_lance_ann_index(table, dim=768, quantization="fp32")
    
    print("Building metadata.json...")
    metadata = {
        "model_key": "nomic_v15",
        "hf_id": "nomic-ai/nomic-embed-text-v1.5",
        "serialization": "pipe",
        "quantization": "fp32",
        "dim": 768,
        "n_records": table.count_rows(),
        "index_size_mb": round(get_dir_size_mb(output_dir), 2),
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "device": "mps",
        "model_path": "jayshah5696/er-nomic-v15-pipe-ft",
        "build_time_sec": 0,
        "note": "Recovered from interrupted build"
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    print("Recovery complete. You can now run the run_eval.py command!")

if __name__ == "__main__":
    main()
