# EXP-014: GTE-ModernBERT FT + bge_reranker_m3 ZS

## Hypothesis
[Fill before running]

## Setup
- Stage 1: GTE-ModernBERT FT
- Stage 2: bge_reranker_m3 ZS
- CE input: COL/VAL format
- top_k_stage1: 50
- Eval: all 6 buckets, 10K test set

## Run Command
```bash
python src/eval/run_reranker.py \
  --stage1-model gte_modernbert_base \
  --stage1-index experiments/007_ablation_dims/indexes/gte_modernbert_base_pipe_ft_fp32 \
  --reranker bge_reranker_m3 \
  # zero-shot: no --reranker-model-path \
  --eval-queries data/eval/ \
  --output results/014_gte_ft_plus_bge_m3_zs.json \
  --experiment-id 014
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
`exp(014): [one-line result]`
