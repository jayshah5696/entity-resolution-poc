# Experiment 005: BGE-Base Fine-tuned (Pipe Format)

**Status:** pending
**Created:** 2025-01-01
**Run by:** —

---

## Hypothesis

BGE-base-en-v1.5, after fine-tuning with identical config to nomic-embed-v1.5 (experiment 003), will:

- Slightly underperform nomic-embed-v1.5 on all buckets
- The gap will be small (< 5pp on most buckets) because fine-tuning quality matters more than architecture at this scale
- However, nomic's ModernBERT architecture (RoPE, Flash Attention, longer context) may give a meaningful advantage on harder corruption buckets

The story here is: **does architecture matter when data + loss function are the same?**

If BGE-base matches nomic within 2pp: architecture doesn't matter much, prioritize license/efficiency.
If BGE-base lags by 5pp+: ModernBERT gives real benefit, stick with nomic.

---

## Setup

**Fine-tune command:**
```bash
# Edit configs/finetune.yaml: change base_model to BAAI/bge-base-en-v1.5
# Or pass override:
uv run python src/models/finetune.py \
    --config configs/finetune.yaml \
    --base-model BAAI/bge-base-en-v1.5 \
    --run-name bge_base_pipe_ep3_bs256
```

**Eval command:**
```bash
uv run python src/eval/run_eval.py \
    --experiment experiments/005_bge_base_finetuned/config.json \
    --eval-config configs/eval.yaml \
    --output results/exp_005_bge_base_finetuned.json
```

**Note on MRL:** BGE-base does not have native MRL. The MRL wrapper in sentence-transformers will be applied during fine-tuning. Sub-dim quality at 64D may be lower than nomic until fine-tuning adapts the representations.

---

## Results

*To be filled after running.*

| Bucket | R@1 | R@5 | R@10 | MRR@10 | vs nomic FT (exp003) |
|--------|-----|-----|------|--------|---------------------|
| pristine | — | — | — | — | — |
| missing_firstname | — | — | — | — | — |
| missing_email_company | — | — | — | — | — |
| typo_name | — | — | — | — | — |
| domain_mismatch | — | — | — | — | — |
| swapped_attributes | — | — | — | — | — |

**Architecture comparison summary:**

| Model | Avg R@1 (6 buckets) | License | MRL Native |
|-------|--------------------|---------|-----------| 
| nomic-embed-v1.5 FT | — | Apache2 | Yes |
| bge-base-en-v1.5 FT | — | MIT | No |

---

## Observations

*To be filled after running.*

---

## Next Steps

- [ ] If BGE-base within 3pp of nomic on all buckets: recommend BGE-base for its MIT license
- [ ] If BGE-base lags significantly: document ModernBERT architecture advantage
- [ ] Run BGE-base zero-shot comparison as bonus (fast, no training needed)
