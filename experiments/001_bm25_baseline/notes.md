# Experiment 001: BM25 Baseline

**Status:** pending
**Created:** 2025-01-01
**Run by:** —

---

## Hypothesis

BM25 with standard parameters (k1=1.5, b=0.75) on pipe-serialized person records will achieve strong Recall@1 on pristine data (~0.82-0.88) but degrade sharply on corrupted records, particularly:
- typo_name: expected R@1 ~0.30-0.40 (different tokens = zero overlap)
- missing_email_company: expected R@1 ~0.10-0.20 (only name + country tokens remain)
- swapped_attributes: expected R@1 ~0.05-0.15 (token positions scrambled)

This experiment establishes the baseline delta that all dense models must exceed to justify the complexity of embedding-based retrieval.

---

## Setup

**Command:**
```bash
uv run python src/eval/run_eval.py \
    --experiment experiments/001_bm25_baseline/config.json \
    --eval-config configs/eval.yaml \
    --output results/exp_001_bm25_baseline.json
```

**Data:** `data/processed/index_profiles.parquet` (1M records, pipe serialization)
**Eval:** `data/eval/eval_*.parquet` (6 buckets, ~1667 queries each)

**Index construction:**
- BM25Okapi from rank_bm25
- Tokenization: lowercase + whitespace split
- Pipe separators removed before tokenization
- [MISSING] preserved as single token

---

## Results

*To be filled after running.*

| Bucket | R@1 | R@5 | R@10 | MRR@10 | NDCG@10 |
|--------|-----|-----|------|--------|---------|
| pristine | — | — | — | — | — |
| missing_firstname | — | — | — | — | — |
| missing_email_company | — | — | — | — | — |
| typo_name | — | — | — | — | — |
| domain_mismatch | — | — | — | — | — |
| swapped_attributes | — | — | — | — | — |

**Latency (1M index):**
- p50: — ms
- p95: — ms
- p99: — ms

---

## Observations

*To be filled after running.*

Key questions to answer:
1. Does pristine R@1 match the expected 0.82-0.88 range?
2. Which bucket shows the biggest drop from pristine?
3. What's the common-name collision rate? (e.g., how often does "John Smith" fail because of other "John Smith" entries)
4. Is BM25 latency comfortably under 100ms at 1M scale?

---

## Next Steps

- [ ] Run experiment 002 (nomic zero-shot) to measure embedding gap without fine-tuning
- [ ] Compare per-bucket results to understand failure mode distribution
- [ ] If pristine R@1 < 0.80, investigate tokenization — may need to split email local part differently
