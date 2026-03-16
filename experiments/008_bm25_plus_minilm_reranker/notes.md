# EXP-008: BM25 + minilm_reranker ZS

## Hypothesis
[Fill before running]

## Setup
- Stage 1: BM25
- Stage 2: minilm_reranker ZS
- CE input: COL/VAL format
- top_k_stage1: 50
- Eval: all 6 buckets, 10K test set

## Run Command
```bash
python src/eval/run_reranker.py \
  --stage1-model bm25_baseline \
  --stage1-index experiments/001_bm25_baseline \
  --reranker minilm_reranker \
  # zero-shot: no --reranker-model-path \
  --eval-queries data/eval/ \
  --output results/008_bm25_plus_minilm_reranker.json \
  --experiment-id 008
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
`exp(008): [one-line result]`
