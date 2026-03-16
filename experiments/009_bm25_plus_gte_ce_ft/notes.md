# EXP-009: BM25 + gte_reranker FT

## Hypothesis
[Fill before running]

## Setup
- Stage 1: BM25
- Stage 2: gte_reranker FT
- CE input: COL/VAL format
- top_k_stage1: 50
- Eval: all 6 buckets, 10K test set

## Run Command
```bash
python src/eval/run_reranker.py \
  --stage1-model bm25_baseline \
  --stage1-index experiments/001_bm25_baseline \
  --reranker gte_reranker \
  --reranker-model-path jayshah5696/er-ce-gte-reranker-ft \
  --eval-queries data/eval/ \
  --output results/009_bm25_plus_gte_ce_ft.json \
  --experiment-id 009
```

## Key Results
| Bucket | Stage 1 R@10 | + Reranker nDCG@10 | F1 | Delta |
|--------|-------------|-------------------|-----|-------|
| pristine | | | | |
| missing_firstname | | | | |
| missing_email_company | | | | |
| typo_name | | | | |
| domain_mismatch | | | | |
| swapped_attributes | | | | |
| **OVERALL** | | | | |

Stage 1 latency p50: — ms  
Stage 2 latency p50: — ms  
Recall retention: —  

## Interpretation
[Fill after running]

## Commit
`exp(009): [one-line result]`
