# Phase 2 Understanding — Cross-Encoder Reranking

> Read this before any Phase 2 implementation. For Phase 1 architecture, see `docs/UNDERSTANDING.md`.

## Status

**Phase 1:** Complete. All models fine-tuned, all experiments run, blog published.  
**Phase 2:** In progress on branch `phase2`. See `plan.md` for full spec.

## Architecture

```
Query Record (dirty/partial)
         ↓
  STAGE 1 (Phase 1, UNCHANGED)
  GTE-ModernBERT FT bi-encoder
  LanceDB 64D binary HNSW → top-50 candidates (~10-15ms)
         ↓
  STAGE 2 (Phase 2, NEW)
  Cross-Encoder (gte-reranker-modernbert-base FT)
  COL/VAL pair serialization
  Runs on top-50 → reranked list + P(match) score (~30-50ms)
         ↓
  Threshold → MATCH / NON-MATCH
```

## Key Design Decisions

- **Existing indexes untouched.** The Phase 1 LanceDB indexes are Stage 1. No re-embedding.
- **COL/VAL format for CE input only.** Bi-encoder still uses pipe format.
- **Strict data separation.** Phase 2 training data uses real sources (GLEIF/EDGAR/O*NET), zero overlap with Phase 1 Faker triplets.
- **Four CE models.** Two fine-tuned (gte_reranker, granite_reranker), two zero-shot reference (minilm_reranker, bge_reranker_m3).

## Pinned Stack (same as Phase 1)

```
Python:               3.11 (Modal) / 3.12 (local)
torch:                2.6.0+cu124 (Modal)
sentence-transformers: >=5.0,<6
transformers:         ==4.57.6  ← PINNED
flash-attn:           2.7.3 (pre-built wheel on Modal)
Package manager:      uv
```

## File Map (Phase 2 new files only)

```
plan.md                              ← authoritative Phase 2 spec
AGENTS.md                            ← AI assistant instructions
configs/
  crossencoder_models.yaml           ← 4 CE models locked
  crossencoder.yaml                  ← training hyperparameters
data/phase2/
  raw/                               ← downloaded source files (gitignored)
  base_pool.parquet                  ← 50K real entity records (gitignored)
  ce_train.parquet                   ← cross-encoder training pairs (gitignored)
  ce_val.parquet                     ← validation pairs (gitignored)
  ce_test.parquet                    ← LOCKED test set (gitignored)
src/
  data/
    phase2_sources.py                ← download + parse GLEIF/EDGAR/ONET/Census/SSA
    phase2_pool.py                   ← build 50K base entity pool
    phase2_corrupt.py                ← 28 positive + 7 negative corruption types
    phase2_negatives.py              ← 5 negative mining strategies
    phase2_boundary.py               ← bi-encoder boundary-zone mining + LLM judge
    phase2_split.py                  ← train/val/test split + dedup
  models/
    crossencoder.py                  ← CE inference wrapper
    finetune_ce_modal.py             ← Modal training for CE models
    upload_ce_pairs.py               ← push dataset to HF Hub
  eval/
    run_reranker.py                  ← Stage 1 + Stage 2 pipeline eval script
    metrics_phase2.py                ← F1@threshold, PR curve, recall_retention
experiments/
  008-014/                           ← Phase 2 experiments
tests/
  data/test_corrupt_phase2.py        ← all 28 corruption types
  data/test_negatives.py             ← all 7 negative strategies
  data/test_split.py                 ← critical invariant tests
  models/test_crossencoder.py        ← CE inference tests
  eval/test_run_reranker.py          ← integration tests
writing/
  BLOG_POST_PHASE2.md               ← phase 2 blog (write after results)
  PAPER_PHASE2_ADDENDUM.md          ← paper methodology + results sections
```

## Known Gotchas

See `AGENTS.md` for the complete list. Most critical for Phase 2:
- `HF_HUB_DISABLE_XET=1` before any HF push
- `warmup_steps` not `warmup_ratio` (deprecated in ST 3.4+)
- CE threshold must be calibrated on val set — never use 0.5 as default
- `ce_test.parquet` is LOCKED from day one — do not reference in any training code

## Experiments

| Exp ID | System | Status |
|--------|--------|--------|
| 008 | BM25 + MiniLM reranker ZS | pending |
| 009 | BM25 + GTE CE FT | pending |
| 010 | GTE FT + MiniLM ZS | pending |
| 011 | GTE FT + GTE CE ZS | pending |
| 012 | GTE FT + GTE CE FT ← **main result** | pending |
| 013 | GTE FT + Granite FT | pending |
| 014 | GTE FT + BGE-M3 ZS | pending |
