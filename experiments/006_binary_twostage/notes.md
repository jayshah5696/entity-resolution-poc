# Experiment 006: Binary Two-Stage Retrieval

**Status:** pending
**Created:** 2025-01-01
**Run by:** —

---

## Hypothesis

A two-stage retrieval pipeline using the fine-tuned nomic-embed-v1.5:
- **Stage 1:** Binary 64-dim HNSW, retrieve top-100 candidates
- **Stage 2:** FP32 768-dim dot-product re-rank of 100 candidates

...will achieve ≥ 95% of the Recall@10 achieved by single-stage FP32 768-dim retrieval (Experiment 003), while using 32× less memory for Stage 1.

Memory math:
- Single-stage FP32 768-dim: 3.07 GB per 1M records → ~1,536 GB at 500M (dead)
- Two-stage Binary 64-dim Stage 1: 0.008 GB per 1M records → ~4 GB at 500M (viable)

Quality hypothesis: The first 64 MRL dims of the fine-tuned model encode enough signal to identify the true match in the top-100 with ≥ 95% probability. Stage 2 re-rank then precisely orders those 100 candidates.

---

## Setup

**Prereq:** Experiment 003 model must be trained first.

**Command:**
```bash
uv run python src/eval/run_eval.py \
    --experiment experiments/006_binary_twostage/config.json \
    --eval-config configs/eval.yaml \
    --output results/exp_006_binary_twostage.json
```

**Implementation notes:**
- Stage 1: extract 64-dim prefix from 768-dim embeddings, binarize by sign
- Stage 2: fetch full 768-dim embeddings for top-100 candidates, cosine re-rank
- Need to store both 64-dim binary index AND full 768-dim FP32 embeddings (searchable by record ID)
- FP32 store: numpy memmap or HDF5 file for O(1) ID-based lookup

---

## Results

*To be filled after running.*

### Stage 1 ANN Quality (Candidates Coverage)

| Bucket | True match in top-100 (%) |
|--------|--------------------------|
| pristine | — |
| missing_firstname | — |
| missing_email_company | — |
| typo_name | — |
| domain_mismatch | — |
| swapped_attributes | — |

### Final Results After Stage 2 Re-rank

| Bucket | R@1 | R@10 | vs exp003 FP32 R@1 | vs exp003 FP32 R@10 |
|--------|-----|------|--------------------|--------------------|
| pristine | — | — | — | — |
| missing_firstname | — | — | — | — |
| missing_email_company | — | — | — | — |
| typo_name | — | — | — | — |
| domain_mismatch | — | — | — | — |
| swapped_attributes | — | — | — | — |

### Memory Comparison (1M index)

| Stage | Index Type | Size (GB) | Extrapolated 500M (GB) |
|-------|-----------|----------|----------------------|
| Stage 1 binary | Binary HNSW 64-dim | — | — |
| Stage 2 FP32 | On-disk store | — | — |
| Single-stage FP32 | HNSW 768-dim | — | — |

### Latency

| Stage | p50 (ms) | p95 (ms) | p99 (ms) |
|-------|----------|----------|----------|
| Stage 1 ANN | — | — | — |
| Stage 2 re-rank | — | — | — |
| Total pipeline | — | — | — |

---

## Observations

*To be filled after running.*

Key questions:
1. Is true match in binary top-100 at ≥ 95% rate across all buckets?
2. Does Stage 2 re-rank fully recover from Stage 1 misses?
3. Is the total pipeline latency under 30ms p99?
4. What's the quality loss at Recall@1? (Expect < 3pp vs full precision)

---

## Next Steps

- [ ] If binary Stage 1 achieves ≥ 95% candidate coverage: binary two-stage is the recommended production architecture
- [ ] If coverage < 90% on some bucket: try increasing Stage 1 top-k to 200 or 500
- [ ] Document memory savings in README and research-design.md
