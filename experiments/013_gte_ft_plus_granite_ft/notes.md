# EXP-013: GTE-ModernBERT FT + granite_reranker FT

## Hypothesis
[Fill before running]

## Setup
- Stage 1: GTE-ModernBERT FT
- Stage 2: granite_reranker FT
- CE input: COL/VAL format
- top_k_stage1: 50
- Eval: all 6 buckets, 10K test set

## Run Command
```bash
python src/eval/run_reranker.py \
  --stage1-model gte_modernbert_base \
  --stage1-index experiments/007_ablation_dims/indexes/gte_modernbert_base_pipe_ft_fp32 \
  --reranker granite_reranker \
  --reranker-model-path jayshah5696/er-ce-granite-reranker-ft \
  --eval-queries data/eval/ \
  --output results/013_gte_ft_plus_granite_ft.json \
  --experiment-id 013
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
`exp(013): [one-line result]`
