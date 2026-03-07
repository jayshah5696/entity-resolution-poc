# Experiment 002: Nomic v1.5 Zero-Shot (Pipe)

**Status:** pending
**Created:** 2025-01-01
**Run by:** —

---

## Hypothesis

nomic-embed-text-v1.5, without any fine-tuning, will outperform BM25 on corrupted record buckets (typo_name, missing_email_company) because:
1. It has seen name variations, abbreviations, and typos during pre-training on web text.
2. Its dense representations capture semantic similarity beyond token overlap.

However, it may slightly underperform BM25 on pristine records because:
1. BM25 is an exact/near-exact retrieval method — pristine records have full token overlap.
2. The zero-shot model was not trained to prioritize structured field matching.

Expected deltas vs BM25 (Experiment 001):
- pristine: 0 to -5pp (slight BM25 advantage)
- typo_name: +15 to +25pp (embedding advantage)
- missing_email_company: +20 to +35pp (embedding advantage)
- domain_mismatch: +10 to +20pp

---

## Setup

**Command:**
```bash
uv run python src/eval/run_eval.py \
    --experiment experiments/002_nomic_zeroshot/config.json \
    --eval-config configs/eval.yaml \
    --output results/exp_002_nomic_zeroshot.json
```

**Note:** Embedding 1M records with nomic v1.5 will take ~30-60 min on MPS. Run while unattended.

**Key question during setup:** Does the model output need `trust_remote_code=True`? Yes.

---

## Results

*To be filled after running.*

| Bucket | R@1 | R@5 | R@10 | MRR@10 | vs BM25 R@1 |
|--------|-----|-----|------|--------|------------|
| pristine | — | — | — | — | — |
| missing_firstname | — | — | — | — | — |
| missing_email_company | — | — | — | — | — |
| typo_name | — | — | — | — | — |
| domain_mismatch | — | — | — | — | — |
| swapped_attributes | — | — | — | — | — |

**Latency (1M index, HNSW):**
- Embedding p99: — ms
- Retrieval p99: — ms
- Total p99: — ms

**Index build time:** — min
**Index RAM:** — GB

---

## Observations

*To be filled after running.*

Key questions:
1. Does zero-shot outperform BM25 on any bucket? By how much?
2. Is pristine performance worse than BM25 (expected)?
3. Does this justify running fine-tuning? (If zero-shot already massively beats BM25, fine-tuning delta may be smaller than expected)

---

## Next Steps

- [ ] If zero-shot is already strong (R@1 typo > 0.70), document and update expectations for fine-tuned model
- [ ] Start fine-tuning run (Experiment 003) — kick off overnight
- [ ] Compare embedding time — if > 60 min for 1M records, consider batch size optimization
