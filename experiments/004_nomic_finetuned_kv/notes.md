# Experiment 004: Nomic v1.5 Fine-tuned (Key-Value Format)

**Status:** pending
**Created:** 2025-01-01
**Run by:** —

---

## Hypothesis

Key-value serialization (`first_name: Jonathan last_name: Smith ...`) compared to pipe format (`Jonathan Smith | Google | ...`) will:

- **Post fine-tuning:** Produce slightly lower quality than pipe format. Rationale: KV format is more verbose (more tokens), the model must learn to ignore the key labels and focus on values, and the positional signal is diluted.

- **Exception — swapped_attributes bucket:** KV format may outperform pipe on swapped_attributes because explicit field labels are preserved even when values are swapped. The model can use label context to resolve the swap.

- **vs Zero-shot models:** KV format should work better for zero-shot models (labels provide semantic grounding from pre-training). This experiment tests the post-fine-tuning behavior.

---

## Setup

**Fine-tune command:**
```bash
uv run python src/models/finetune.py \
    --config configs/finetune.yaml \
    --serialization keyvalue \
    --run-name nomic_v15_kv_ep3_bs256
```

**Eval command:**
```bash
uv run python src/eval/run_eval.py \
    --experiment experiments/004_nomic_finetuned_kv/config.json \
    --eval-config configs/eval.yaml \
    --output results/exp_004_nomic_finetuned_kv.json
```

**Important:** Ensure the eval set uses KV-serialized queries when comparing to this model.

---

## Results

*To be filled after running.*

| Bucket | R@1 | R@5 | R@10 | MRR@10 | vs exp003 (pipe) |
|--------|-----|-----|------|--------|-----------------|
| pristine | — | — | — | — | — |
| missing_firstname | — | — | — | — | — |
| missing_email_company | — | — | — | — | — |
| typo_name | — | — | — | — | — |
| domain_mismatch | — | — | — | — | — |
| swapped_attributes | — | — | — | — | — |

**Key comparison: Experiment 003 (pipe) vs 004 (KV):**

| Bucket | Pipe R@1 | KV R@1 | Winner | Delta |
|--------|----------|--------|--------|-------|
| pristine | — | — | — | — |
| swapped_attributes | — | — | — | — |

---

## Observations

*To be filled after running.*

Key questions:
1. Does pipe or KV win overall? (Sum of R@1 across all buckets)
2. Does KV win specifically on swapped_attributes? (The hypothesis)
3. Is the performance gap large enough to matter in production, or is it < 2pp (noise)?
4. Does KV tokenize to significantly more tokens? (Check mean token count pipe vs KV)

---

## Next Steps

- [ ] Based on pipe vs KV winner, choose that format as the standard for subsequent experiments
- [ ] If KV > pipe on most buckets: re-run experiment 003 training considerations
- [ ] Summarize serialization recommendation in research-design.md
