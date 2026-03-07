# Entity Resolution POC — Beating BM25 with Pure Embeddings

> Research project: can fine-tuned Matryoshka embeddings outperform BM25 for structured people-record search at 500M scale?

---

## Overview

This repository contains the full research scaffold for a people-entity resolution system built on dense embeddings. The core question is whether a fine-tuned embedding model — using Matryoshka Representation Learning (MRL) + Multiple Negatives Ranking Loss (MNRL) — can beat a strong BM25 baseline on a structured 5-field person schema under realistic data corruption conditions, while remaining production-viable at 500M record scale.

The answer we're testing: **yes, with the right two-stage architecture and domain-specific fine-tuning.**

---

## Problem Statement

People search is deceptively hard. A production B2B data platform might store 500 million contact records and receive queries like:

```
first_name: Jon   last_name: Smyth   company: Goog   email: j.smith@gmail.com   country: US
```

The true match is `Jonathan Smith | Google | jonathan.smith@google.com | USA`. BM25 will fail here — token overlap is low, abbreviations are not stemmed to their expansions, and the email domain mismatch destroys recall. A well-trained dense retriever that has seen these corruption patterns can model the semantic proximity directly in embedding space.

**The structured-to-structured matching problem** is distinct from document retrieval. Records are short (< 50 tokens), fields are semantically typed, field presence/absence carries information, and the corruption distribution is known and learnable. This is the regime where BM25's term-frequency assumptions break down fastest.

**Scale reality check:** At 500M records with 768-dim float32 embeddings, you need 1.5TB of RAM just for the index. That's not production. Binary 64-dim embeddings collapse that to ~4GB — that's a server. The architecture must exploit MRL's ability to produce useful representations at multiple dimensionalities.

---

## Architecture

### Two-Stage Retrieval

```
Query Record (possibly corrupted/partial)
         │
         ▼
  Serialization Layer
  (pipe or key-value format)
         │
         ▼
 ┌───────────────────┐
 │  Stage 1: ANN     │
 │  64-dim binary    │
 │  HNSW index       │   ← ~24GB for 500M records (production-viable)
 │  Recall target:   │
 │  top-100 cands    │
 └───────┬───────────┘
         │ 100 candidates
         ▼
 ┌───────────────────┐
 │  Stage 2: Re-rank │
 │  768-dim FP32     │
 │  dot product      │   ← only 100 vectors loaded per query
 │  exact scoring    │
 └───────┬───────────┘
         │ top-K results
         ▼
      Final Results
```

### Memory Math (500M records)

| Format         | Dims | Bytes/vec | Total RAM  | Viable? |
|----------------|------|-----------|------------|---------|
| FP32           | 768  | 3,072     | ~1,536 GB  | No      |
| INT8           | 768  | 768       | ~384 GB    | Barely  |
| FP32           | 64   | 256       | ~128 GB    | Maybe   |
| Binary         | 64   | 8         | ~4 GB      | Yes     |
| Binary         | 768  | 96        | ~48 GB     | Yes     |

Target: binary 64-dim for Stage 1 ANN, full FP32 768-dim for the re-rank of top-100 candidates (768-dim × 100 vectors = 307KB per query — trivial).

### Why MRL

Matryoshka Representation Learning trains a single model to produce embeddings where the first 64 dimensions are themselves a useful representation, the first 128 are better, and so on up to the full 768. This means one fine-tuned model serves both stages without separate model training.

### Serialization

Records are serialized to a single string before embedding. Two formats under test:

**Pipe format:**
```
Jonathan Smith | Google Inc | jonathan.smith@google.com | USA
```

**Key-value format:**
```
first_name: Jonathan last_name: Smith company: Google Inc email: jonathan.smith@google.com country: USA
```

Missing fields are explicitly represented: `first_name: [MISSING]` — this teaches the model that absence is informative, not an error.

---

## Repo Structure

```
entity-resolution-poc/
├── README.md
├── pyproject.toml              # uv-managed dependencies
├── configs/
│   ├── dataset.yaml            # data generation config
│   ├── models.yaml             # model registry
│   ├── finetune.yaml           # training hyperparameters
│   └── eval.yaml               # evaluation settings
├── docs/
│   ├── research-design.md      # full research design
│   ├── dataset-design.md       # dataset spec and corruption types
│   ├── models.md               # model roster deep-dive
│   └── evaluation-protocol.md # metrics, index params, latency
├── src/
│   ├── data/                   # dataset generation & corruption
│   ├── models/                 # embedding wrappers & BM25
│   ├── eval/                   # retrieval evaluation harness
│   └── utils/                  # serialization, config, logging
├── experiments/
│   ├── 001_bm25_baseline/
│   ├── 002_nomic_zeroshot/
│   ├── 003_nomic_finetuned_pipe/
│   ├── 004_nomic_finetuned_kv/
│   ├── 005_bge_base_finetuned/
│   ├── 006_binary_twostage/
│   └── 007_ablation_dims/
├── data/
│   ├── raw/                    # generated base profiles (parquet)
│   ├── processed/              # serialized records
│   ├── triplets/               # training triplets (parquet)
│   └── eval/                   # evaluation sets per bucket
├── models/                     # fine-tuned model checkpoints
├── results/                    # JSON per experiment + master CSV
└── notebooks/
    └── results_viz.ipynb       # results visualization (only notebook)
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- ~50GB disk space for data + models

### Setup

```bash
git clone <repo-url> entity-resolution-poc
cd entity-resolution-poc

# Install all dependencies
uv sync

# Verify install
uv run python -c "import sentence_transformers, faiss, rank_bm25; print('OK')"
```

### Run Scripts in Order

```bash
# Step 1: Generate base profiles (takes ~10 min for 1.2M records)
uv run python src/data/generate_profiles.py --config configs/dataset.yaml

# Step 2: Build corruption + triplets
uv run python src/data/build_triplets.py --config configs/dataset.yaml

# Step 3: Build eval sets (6 buckets, 10K total)
uv run python src/data/build_eval.py --config configs/dataset.yaml

# Step 4: Run BM25 baseline
uv run python src/eval/run_eval.py \
    --experiment experiments/001_bm25_baseline/config.json \
    --eval-config configs/eval.yaml

# Step 5: Run zero-shot embedding baselines
uv run python src/eval/run_eval.py \
    --experiment experiments/002_nomic_zeroshot/config.json \
    --eval-config configs/eval.yaml

# Step 6: Fine-tune (long — kick off before leaving)
uv run python src/models/finetune.py --config configs/finetune.yaml

# Step 7: Eval fine-tuned model
uv run python src/eval/run_eval.py \
    --experiment experiments/003_nomic_finetuned_pipe/config.json \
    --eval-config configs/eval.yaml

# Step 8: Aggregate and visualize results
uv run python src/eval/aggregate_results.py --results-dir results/
# Then open notebooks/results_viz.ipynb
```

---

## Experiment Log

| ID  | Description                          | Status  | Key Result |
|-----|--------------------------------------|---------|------------|
| 001 | BM25 Baseline (pipe, 1M index)       | pending |            |
| 002 | Nomic v1.5 Zero-shot (pipe)          | pending |            |
| 003 | Nomic v1.5 Fine-tuned (pipe format)  | pending |            |
| 004 | Nomic v1.5 Fine-tuned (KV format)    | pending |            |
| 005 | BGE-Base Fine-tuned (pipe format)    | pending |            |
| 006 | Binary Two-Stage (64D→768D re-rank)  | pending |            |
| 007 | Ablation: Dimensionality (all dims)  | pending |            |

**Update this table after each experiment.** Key Result should be Recall@1 on the `pristine` bucket vs BM25 baseline delta (e.g., `+12.3pp R@1 pristine, +31.2pp R@1 typo_name`).

---

## Contributing / Documenting Results

### After Each Experiment

1. Fill in `experiments/00N_name/notes.md` — hypothesis, setup, results, observations, next steps.
2. Update the Experiment Log table in this README.
3. Commit with message format: `exp(00N): <one-line result summary>`

### Result Files

Each experiment produces:
- `results/exp_00N_<name>.json` — full metrics across all buckets and top-K values
- Appended row in `results/master_results.csv`

### Config Changes

If you change a config mid-experiment, copy the original config into the experiment's directory before modifying. Configs in `configs/` should reflect the current best settings, not historical ones.

### Reproducibility

All randomness is seeded via `random_seed: 42` in `configs/dataset.yaml`. Always pass `--seed 42` to scripts that accept it. Document any deviations in the experiment notes.

---

## Research Context

- **Task:** Structured people-entity resolution (B2B scale)
- **Schema:** `first_name | last_name | company | email | country`
- **Scale target:** 500M records
- **Baseline to beat:** BM25 (rank_bm25, k1=1.5, b=0.75)
- **Primary metric:** Recall@1 and Recall@10 across 6 corruption buckets
- **Hardware:** Apple M3 Pro (MPS) — no CUDA, FP16 disabled, MPS mixed precision enabled
- **Expected runtime:** ~4-6h fine-tuning, ~2h eval across all experiments

See `docs/research-design.md` for the full research design rationale.
