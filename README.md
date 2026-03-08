# Entity Resolution POC

Research project testing whether fine-tuned Matryoshka embeddings beat BM25 for structured people-record matching at 500M scale.

> **New here?** Read [`docs/UNDERSTANDING.md`](docs/UNDERSTANDING.md) — it's the single-source-of-truth covering everything about this project.

## Documentation

| Doc | Purpose |
|-----|--------|
| [`docs/UNDERSTANDING.md`](docs/UNDERSTANDING.md) | Complete project overview — read this first |
| [`docs/research-design.md`](docs/research-design.md) | Hypotheses, architecture, ablation plan, timeline |
| [`docs/dataset-design.md`](docs/dataset-design.md) | Full data spec — schema, corruptions, triplets, eval sets |
| [`docs/evaluation-protocol.md`](docs/evaluation-protocol.md) | Metric definitions, latency methodology, result format |
| [`docs/decisions.md`](docs/decisions.md) | 5 Architecture Decision Records (ADRs) |
| [`docs/experiment-log.md`](docs/experiment-log.md) | Per-experiment tracking template |

## The Problem

The production system uses BM25 to match people across 500M contact records. It fails on dirty data. Lexical token overlap breaks on abbreviated names, typos, missing fields, and swapped email domains.

This project tests whether a dense retriever, fine-tuned on our corruption distribution, improves recall without exceeding production memory constraints.

## Approach

- **Pure embedding retrieval vs BM25**: No hybrid approach for baseline comparisons.
- **Models**: Five models evaluated: one lexical baseline and four embedding models ranging from 22M to 600M parameters.
- **Training**: Fine-tuning uses Matryoshka Representation Learning (MRL) and Multiple Negatives Ranking Loss (MNRL) on synthetic triplets. Corruptions mirror production errors.
- **Inference**: Two-stage retrieval. Binary 64-dim HNSW for ANN (4GB for 500M records), followed by full FP32 768-dim re-rank on top-100 candidates.

## Quickstart

**Prerequisites:** Python 3.12+, uv, ~50GB disk.

### 1. Setup
```bash
git clone https://github.com/jayshah5696/entity-resolution-poc
cd entity-resolution-poc
uv sync
```

### 2. Data Generation
Generate profiles, training triplets, and evaluation queries.
```bash
# Generate 1.2M base profiles and split into index / triplets / eval (~20 min on M3)
uv run python src/data/generate.py --config configs/dataset.yaml --output-dir data/

# Build training triplets from the 200K triplet_source split (~10 min)
uv run python src/data/triplets.py \
    --config configs/dataset.yaml \
    --profiles data/processed/triplet_source.parquet \
    --output-dir data/triplets/

# Build eval query set (10K queries across 6 buckets, ~2 min)
uv run python src/data/eval_set.py \
    --config configs/dataset.yaml \
    --eval-profiles data/eval/eval_profiles.parquet \
    --output-dir data/eval/
```

### 3. Evaluation

**Quantization & Index Derivation**
When building a dense index, use the `--quantization` flag to control memory usage. To test multiple dimensions (MRL) and quantizations without re-running the heavy ML encoding, you can "derive" a new index directly from an existing one:

```bash
# 1. Build the base index (heavy ML encode - run once)
uv run python src/eval/build_index.py --model gte_modernbert_base --serialization pipe --quantization fp32 --index-profiles data/processed/index.parquet --eval-profiles data/eval/eval_profiles.parquet --output-dir results/indexes/gte_modernbert_base_pipe_fp32 --device mps

# 2. Derive a 64-dim int8 index (instant CPU-bound slice and quantize)
uv run python src/eval/build_index.py --source-index results/indexes/gte_modernbert_base_pipe_fp32 --output-dir results/indexes/gte_64_int8 --truncate-dim 64 --quantization int8

# 3. Evaluate the derived index
uv run python src/eval/run_eval.py --model gte_modernbert_base --index-dir results/indexes/gte_64_int8 --eval-queries data/eval/eval_queries.parquet --output results/gte_64_int8.json --serialization pipe
```

```bash
# Build BM25 index and run evaluation
uv run python src/eval/build_index.py --model bm25_baseline --serialization pipe --index-profiles data/processed/index.parquet --eval-profiles data/eval/eval_profiles.parquet --output-dir results/indexes/bm25_pipe --models-config configs/models.yaml
uv run python src/eval/run_bm25.py --index-dir results/indexes/bm25_pipe --eval-queries data/eval/eval_queries.parquet --output results/001_bm25_pipe.json --serialization pipe --experiment-id 001

# Build Dense embedding index and run evaluation (FP32)
uv run python src/eval/build_index.py --model gte_modernbert_base --serialization pipe --quantization fp32 --index-profiles data/processed/index.parquet --eval-profiles data/eval/eval_profiles.parquet --output-dir results/indexes/gte_modernbert_base_pipe_fp32 --device mps
uv run python src/eval/run_eval.py --model gte_modernbert_base --index-dir results/indexes/gte_modernbert_base_pipe_fp32 --eval-queries data/eval/eval_queries.parquet --output results/004_gte_modernbert_pipe_fp32.json --serialization pipe --experiment-id 004

# Evaluate Fine-Tuned pplx-embed model (background)
nohup bash -c 'uv run python src/eval/build_index.py --model pplx_embed_v1_06b --model-path jayshah5696/er-pplx-embed-v1-06b-pipe-ft --serialization pipe --quantization fp32 --index-profiles data/processed/index.parquet --eval-profiles data/eval/eval_profiles.parquet --output-dir results/indexes/pplx_embed_v1_06b_ft_pipe --device mps && uv run python src/eval/run_eval.py --model pplx_embed_v1_06b --model-path jayshah5696/er-pplx-embed-v1-06b-pipe-ft --index-dir results/indexes/pplx_embed_v1_06b_ft_pipe --eval-queries data/eval/eval_queries.parquet --output results/pplx_embed_v1_06b_ft_pipe.json --serialization pipe --experiment-id pplx_embed_v1_06b_ft' > eval_pplx_embed_ft.log 2>&1 &

# Aggregate results
uv run python src/eval/aggregate.py --results-dir results/ --output-csv results/master_results.csv --output-report results/report.md
```

### 4. Fine-Tuning (Modal)
Fine-tune all 5 models in parallel on Modal A10G.
```bash
# Push training data to HuggingFace Hub (one-time local setup)
export HF_HUB_DISABLE_XET=1
hf auth login
hf upload jayshah5696/entity-resolution-triplets data/triplets/triplets.parquet triplets.parquet --repo-type dataset

# Run the training
modal run src/models/finetune_modal.py::run_all
```

### 5. Tests
```bash
uv run pytest tests/ -v
```

## Models

| # | Model | Params | Dims | MRL | License | Role |
|---|-------|--------|------|-----|---------|------|
| 1 | BM25 (rank_bm25) | -- | -- | -- | Apache | Lexical baseline |
| 2 | all-MiniLM-L6-v2 | 22M | 384 | No | Apache | Absolute floor |
| 3 | bge-small-en-v1.5 | 33M | 384 | Yes | MIT | Efficiency baseline |
| 4 | gte-modernbert-base | 149M | 768 | Yes | Apache | Primary candidate |
| 5 | nomic-embed-text-v1.5 | 137M | 768 | Yes | Apache | MRL reference |
| 6 | pplx-embed-v1-0.6b | 600M | 1536 | Yes | Apache | Zero-shot ceiling |

*Note: `nomic` requires `search_query:` / `search_document:` prefixes. `pplx` uses separate system prompts. See `docs/UNDERSTANDING.md`.*

## Repository

```
configs/        model registry, dataset config, finetune hyperparams, eval settings
docs/           research design, dataset spec, evaluation protocol, model notes
src/
  data/         profile generation, corruption engine, serialization, triplet building
  models/       embedding wrappers and BM25
  eval/         retrieval evaluation harness
  utils/        nicknames, config loading
experiments/    per-experiment tracking
data/           raw profiles, processed records, triplets, eval sets
models/         fine-tuned checkpoints
results/        JSON per experiment + master CSV
tests/          pytest suite
```

## Experiment Log

| ID  | Name                                | Status  | Key Result |
|-----|-------------------------------------|---------|------------|
| 001 | BM25 baseline (pipe, 1M index)      | pending |            |
| 002 | Nomic v1.5 zero-shot (pipe)         | pending |            |
| 003 | Nomic v1.5 fine-tuned (pipe)        | pending |            |
| 004 | Nomic v1.5 fine-tuned (KV)          | pending |            |
| 005 | BGE-small fine-tuned (pipe)         | pending |            |
| 006 | Binary two-stage (64D ANN + 768D rerank) | pending |       |
| 007 | Dimensionality ablation (all dims)  | pending |            |

*Update `experiments/00N_name/notes.md` and this table after each run.*
