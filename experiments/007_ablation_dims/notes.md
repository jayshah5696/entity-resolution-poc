# Experiment 007: Ablation — MRL Dimensionality

**Status:** pending
**Created:** 2025-01-01
**Run by:** —

---

## Hypothesis

For the fine-tuned nomic-embed-v1.5 (pipe format), the quality-vs-memory curve across MRL dims will show:

1. **Diminishing returns above 256 dims:** Going from 256→768 adds < 3pp Recall@1 but 3× the memory.
2. **Elbow at 128 dims:** The best quality/memory tradeoff — acceptable quality loss with 6× memory reduction.
3. **64-dim binary is viable for production Stage 1:** ≥ 90% of 768-dim recall with 384× memory reduction.

Expected Recall@1 pattern (pristine bucket):
- 768-dim FP32: ~0.891 (reference = exp003 result)
- 512-dim FP32: ~0.885 (< 1pp loss)
- 256-dim FP32: ~0.872 (~2pp loss)
- 128-dim FP32: ~0.855 (~4pp loss)
- 64-dim FP32: ~0.832 (~6pp loss)
- 64-dim binary: ~0.810 (~8pp loss from 768-dim FP32)

---

## Setup

**Command:**
```bash
uv run python src/eval/ablation_dims.py \
    --experiment experiments/007_ablation_dims/config.json \
    --eval-config configs/eval.yaml \
    --output-dir results/exp_007/
```

This script iterates through all dim × quantization combinations and outputs:
- Individual JSON per combination: `results/exp_007/dim_{d}_{quant}.json`
- Summary CSV: `results/exp_007/summary.csv`

**Prereq:** Experiment 003 model must be trained.

---

## Results

*To be filled after running.*

### Recall@1 Across Dims (Pristine Bucket)

| Dims | FP32 R@1 | Binary R@1 | FP32 RAM (1M) | Binary RAM (1M) | FP32 p99 (ms) |
|------|----------|------------|--------------|----------------|--------------|
| 768 | — | — | 3.07 GB | 0.096 GB | — |
| 512 | — | — | 2.05 GB | 0.064 GB | — |
| 256 | — | — | 1.02 GB | 0.032 GB | — |
| 128 | — | — | 0.51 GB | 0.016 GB | — |
| 64 | — | — | 0.26 GB | 0.008 GB | — |

### Recall@1 Across Dims (typo_name Bucket)

| Dims | FP32 R@1 | Binary R@1 | Quality @ 64D Binary vs 768D FP32 |
|------|----------|------------|----------------------------------|
| 768 | — | — | 100% (reference) |
| 512 | — | — | — |
| 256 | — | — | — |
| 128 | — | — | — |
| 64 | — | — | — |

### Recommended Operating Points

| Use Case | Recommended Config | Rationale |
|----------|-------------------|-----------|
| Max quality | 768-dim FP32 | Reference ceiling |
| Quality + memory balance | 256-dim FP32 | Best elbow |
| Production Stage 1 ANN | 64-dim binary | 4GB / 1M records |
| Edge deployment | 128-dim binary | 16MB / 1M records |

---

## Observations

*To be filled after running.*

Key questions:
1. Where exactly is the quality elbow (biggest drop per additional compression step)?
2. Is 64-dim FP32 > 768-dim binary? (Compare accuracy at same memory budget)
3. Do harder buckets (missing_email_company, typo_name) show larger quality degradation at lower dims?
4. What's the latency benefit of lower dims? (HNSW search is faster at lower dims)

---

## Next Steps

- [ ] Update research-design.md with the actual quality-vs-memory curve
- [ ] Use results to finalize production architecture recommendation (which dim for Stage 1, which for Stage 2)
- [ ] If 128-dim binary looks strong, test it as Stage 1 instead of 64-dim in a follow-up
