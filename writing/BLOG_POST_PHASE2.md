# Phase 2 Blog Post — PLACEHOLDER

**Status:** Placeholder. Write after experiments 008-014 are complete.

## Working Title

"The Reranker Gap: Why Dense Retrieval Alone Isn't Enough for B2B Entity Resolution"

or

"Building a Two-Stage Entity Matcher: From Bi-Encoder Recall to Cross-Encoder Precision"

## Narrative Arc (pre-planned)

1. **One-paragraph Phase 1 recap** — link to Phase 1 post. GTE-ModernBERT beat BM25. But R@10 isn't the whole story.
2. **The precision problem** — retrieving the right answer in top-10 is different from ranking it #1. Walk through what cosine similarity misses vs cross-field attention.
3. **The BM25 + reranker straw man** — why most teams stop here, and why the recall ceiling kills it on the `missing_email_company` case.
4. **What a cross-encoder actually sees** — Mermaid diagram. COL/VAL field tokens. Cross-field attention visualization.
5. **The data pipeline** — why Faker isn't enough for Phase 2. What GLEIF/O*NET/Census gives you. Rule-based corruption taxonomy.
6. **Results** — fill in after experiments. Comparison table (8 systems × 6 buckets).
7. **Phase 2.5 preview** — MarginMSE distillation, bi-encoder improvement loop.
8. **Modal bill** — total training cost for Phase 2.

## Key Figures to Include

- [ ] Two-stage pipeline Mermaid diagram (in plan.md)
- [ ] Option B data pipeline Mermaid diagram (in plan.md)
- [ ] 8×6 comparison table (fill after experiments)
- [ ] PR curve for best model
- [ ] Latency breakdown: Stage 1 vs Stage 2

## Target Publication

After Phase 2 experiments complete and paper addendum drafted.
