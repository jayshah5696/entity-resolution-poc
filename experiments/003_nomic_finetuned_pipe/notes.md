# Experiment 003: Nomic v1.5 Fine-tuned (Pipe Format)

**Status:** pending
**Created:** 2025-01-01
**Run by:** —

---

## Hypothesis

Domain-specific fine-tuning of nomic-embed-text-v1.5 on entity resolution triplets (pipe format, MRL + MNRL loss, curriculum hard negatives) will produce a model that:

1. Meets or beats BM25 on pristine records (R@1 ≥ 0.87)
2. Substantially beats zero-shot on all corruption buckets:
   - typo_name R@1 ≥ 0.78 (delta: +20-25pp vs zero-shot)
   - missing_email_company R@1 ≥ 0.65 (delta: +25-30pp vs zero-shot)
   - domain_mismatch R@1 ≥ 0.82
3. Shows the value of corruption-pattern-specific training

---

## Setup

**Fine-tune command (run first):**
```bash
uv run python src/models/finetune.py \
    --config configs/finetune.yaml \
    --run-name nomic_v15_pipe_ep3_bs256
```

**Expected training time:** 4-6 hours on Apple M3 Pro (MPS)
**Expected final model path:** models/nomic_v15_pipe_ep3_bs256/

**Eval command:**
```bash
uv run python src/eval/run_eval.py \
    --experiment experiments/003_nomic_finetuned_pipe/config.json \
    --eval-config configs/eval.yaml \
    --output results/exp_003_nomic_finetuned_pipe.json
```

---

## Results

*To be filled after running.*

| Bucket | R@1 | R@5 | R@10 | MRR@10 | vs BM25 | vs zero-shot |
|--------|-----|-----|------|--------|---------|-------------|
| pristine | — | — | — | — | — | — |
| missing_firstname | — | — | — | — | — | — |
| missing_email_company | — | — | — | — | — | — |
| typo_name | — | — | — | — | — | — |
| domain_mismatch | — | — | — | — | — | — |
| swapped_attributes | — | — | — | — | — | — |

**Training details:**
- Final training loss: —
- Best eval metric (Recall@10, pristine): —
- Steps to convergence: —
- WandB run link: —

**Latency:**
- Embedding p99: — ms
- HNSW retrieval p99: — ms
- Total pipeline p99: — ms

---

## Observations

*To be filled after running.*

Key questions:
1. Is the fine-tuned model better than zero-shot by the expected margin (~20pp on typo)?
2. Did curriculum hard negatives help? (Check WandB eval curve — look for improvement after epoch when hard neg ratio increases)
3. Any signs of overfitting? (training loss << eval recall)
4. Does pipe format generalization hold? (model fine-tuned on pipe, can it handle slight serialization variations?)

---

## Next Steps

- [ ] Run experiment 004 (KV format) with identical training config to compare formats
- [ ] Extract 64-dim sub-embeddings and run experiment 006 (binary two-stage)
- [ ] Check WandB curves: if epoch 2-3 flat, training may have converged early
