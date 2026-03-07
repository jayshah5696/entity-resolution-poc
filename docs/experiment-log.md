# Experiment Log

Detailed log of every experiment run. Update this file after each run.

For each experiment: fill in the config, run the scripts, record the actual results, write observations.

See README.md for the summary table. This file has the full details.

---

## Template

### EXP-00N: Name

**Date:**
**Status:** pending / running / done / failed

**Config:**
- Model:
- Serialization:
- Mode: zero_shot / finetuned
- Quantization:
- Dims:
- Index size:

**Command run:**
```bash
uv run python src/eval/build_index.py ...
uv run python src/eval/run_eval.py ...
```

**Results:**

| Bucket | R@1 | R@5 | R@10 | MRR@10 | NDCG@10 |
|--------|-----|-----|------|--------|---------|
| pristine | | | | | |
| missing_firstname | | | | | |
| missing_email_company | | | | | |
| typo_name | | | | | |
| domain_mismatch | | | | | |
| swapped_attributes | | | | | |
| **overall** | | | | | |

**Latency:** p50= ms, p95= ms, p99= ms

**Observations:**

**Next steps:**

---

## EXP-001: BM25 Baseline

**Date:** pending
**Status:** pending

**Config:**
- Model: BM25 (rank_bm25, k1=1.5, b=0.75)
- Serialization: pipe
- Mode: zero_shot
- Index size: 1,000,000

**Command run:**
```bash
uv run python src/eval/run_bm25.py \
    --config configs/dataset.yaml \
    --eval-queries data/eval/eval_queries.parquet \
    --index-dir data/processed/index.parquet \
    --output results/001_bm25_pipe.json \
    --serialization pipe
```

**Results:** pending

**Observations:** pending

---

## EXP-002: MiniLM Zero-Shot

**Date:** pending
**Status:** pending

**Config:**
- Model: all-MiniLM-L6-v2 (22M)
- Serialization: pipe
- Mode: zero_shot
- Quantization: fp32
- Dims: 384
- Index size: 1,000,000

**Commands:**
```bash
uv run python src/eval/build_index.py \
    --model minilm_l6 \
    --serialization pipe \
    --index-profiles data/processed/index.parquet \
    --output-dir results/indexes/minilm_l6_pipe_fp32

uv run python src/eval/run_eval.py \
    --model minilm_l6 \
    --index-dir results/indexes/minilm_l6_pipe_fp32 \
    --eval-queries data/eval/eval_queries.parquet \
    --output results/002_minilm_pipe_fp32.json
```

**Results:** pending

**Observations:** pending

---

## EXP-003: pplx-embed-v1-0.6b Zero-Shot

**Date:** pending
**Status:** pending

**Config:**
- Model: pplx-embed-v1-0.6b (600M, SOTA as of March 2026)
- Serialization: pipe
- Mode: zero_shot
- Quantization: fp32
- Dims: 1536
- Index size: 1,000,000

**Commands:**
```bash
uv run python src/eval/build_index.py \
    --model pplx_embed_v1_06b \
    --serialization pipe \
    --index-profiles data/processed/index.parquet \
    --output-dir results/indexes/pplx_embed_v1_06b_pipe_fp32

uv run python src/eval/run_eval.py \
    --model pplx_embed_v1_06b \
    --index-dir results/indexes/pplx_embed_v1_06b_pipe_fp32 \
    --eval-queries data/eval/eval_queries.parquet \
    --output results/003_pplx_pipe_fp32.json
```

**Results:** pending

**Observations:** pending

---

## EXP-004: gte-modernbert-base Zero-Shot

**Date:** pending
**Status:** pending

**Config:**
- Model: gte-modernbert-base (149M)
- Serialization: pipe
- Mode: zero_shot
- Quantization: fp32
- Dims: 768

**Commands:**
```bash
uv run python src/eval/build_index.py \
    --model gte_modernbert_base \
    --serialization pipe \
    --index-profiles data/processed/index.parquet \
    --output-dir results/indexes/gte_modernbert_base_pipe_fp32

uv run python src/eval/run_eval.py \
    --model gte_modernbert_base \
    --index-dir results/indexes/gte_modernbert_base_pipe_fp32 \
    --eval-queries data/eval/eval_queries.parquet \
    --output results/004_gte_modernbert_pipe_fp32.json
```

**Results:** pending

---

## EXP-005: gte-modernbert-base Fine-Tuned (pipe)

**Date:** pending
**Status:** pending

**Config:**
- Model: gte-modernbert-base fine-tuned on entity triplets
- Serialization: pipe
- Mode: finetuned
- Quantization: fp32
- Dims: 768 (also 256, 64 from MRL checkpoints)

**Commands:**
```bash
# Fine-tune first (run Saturday night)
uv run python src/models/finetune.py \
    --config configs/finetune.yaml \
    --model gte_modernbert_base \
    --serialization pipe \
    --triplets data/triplets/triplets.parquet \
    --output-dir models/gte_modernbert_base_pipe_ft

# Then build index and eval
uv run python src/eval/build_index.py \
    --model gte_modernbert_base \
    --model-path models/gte_modernbert_base_pipe_ft \
    --serialization pipe \
    --index-profiles data/processed/index.parquet \
    --output-dir results/indexes/gte_modernbert_base_pipe_ft_fp32

uv run python src/eval/run_eval.py \
    --model gte_modernbert_base \
    --index-dir results/indexes/gte_modernbert_base_pipe_ft_fp32 \
    --eval-queries data/eval/eval_queries.parquet \
    --output results/005_gte_modernbert_pipe_ft_fp32.json
```

**Results:** pending

---

## EXP-006: bge-small-en-v1.5 Fine-Tuned (pipe)

**Date:** pending
**Status:** pending

**Config:**
- Model: bge-small-en-v1.5 fine-tuned, MRL added via MatryoshkaLoss
- Serialization: pipe
- Mode: finetuned
- Quantization: fp32
- Dims: 384 (also 256, 128, 64 from MRL)

**Commands:**
```bash
uv run python src/models/finetune.py \
    --config configs/finetune.yaml \
    --model bge_small \
    --serialization pipe \
    --triplets data/triplets/triplets.parquet \
    --output-dir models/bge_small_pipe_ft

uv run python src/eval/build_index.py \
    --model bge_small \
    --model-path models/bge_small_pipe_ft \
    --serialization pipe \
    --index-profiles data/processed/index.parquet \
    --output-dir results/indexes/bge_small_pipe_ft_fp32

uv run python src/eval/run_eval.py \
    --model bge_small \
    --index-dir results/indexes/bge_small_pipe_ft_fp32 \
    --eval-queries data/eval/eval_queries.parquet \
    --output results/006_bge_small_pipe_ft_fp32.json
```

**Results:** pending

---

## EXP-007: nomic-embed-text-v1.5 Fine-Tuned (pipe)

**Date:** pending
**Status:** pending

**Config:**
- Model: nomic-embed-text-v1.5 fine-tuned
- Serialization: pipe (with search_query:/search_document: prefixes applied automatically)
- Mode: finetuned
- Quantization: fp32
- Dims: 768

**Commands:**
```bash
uv run python src/models/finetune.py \
    --config configs/finetune.yaml \
    --model nomic_v15 \
    --serialization pipe \
    --triplets data/triplets/triplets.parquet \
    --output-dir models/nomic_v15_pipe_ft

uv run python src/eval/build_index.py \
    --model nomic_v15 \
    --model-path models/nomic_v15_pipe_ft \
    --serialization pipe \
    --index-profiles data/processed/index.parquet \
    --output-dir results/indexes/nomic_v15_pipe_ft_fp32

uv run python src/eval/run_eval.py \
    --model nomic_v15 \
    --index-dir results/indexes/nomic_v15_pipe_ft_fp32 \
    --eval-queries data/eval/eval_queries.parquet \
    --output results/007_nomic_v15_pipe_ft_fp32.json
```

**Results:** pending

---

## Aggregate + Report

After all experiments:

```bash
uv run python src/eval/aggregate.py \
    --results-dir results/ \
    --output-csv results/master_results.csv \
    --output-report results/report.md
```

Then open results/report.md for the Monday summary.
