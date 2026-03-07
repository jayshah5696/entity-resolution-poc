# Model Lock — March 2026

**Status: LOCKED. Do not add models without team discussion.**

## The 5 Locked Models

### Evaluation Matrix

| # | Model | Params | Dims | License | MRL | MTEB Retrieval | Mode | Role |
|---|-------|--------|------|---------|-----|----------------|------|------|
| 1 | BM25 | — | — | Apache | — | baseline | zero-shot | Lexical baseline |
| 2 | all-MiniLM-L6-v2 | 22M | 384 | Apache | No | 56.3 | zero-shot | Absolute floor |
| 3 | bge-small-en-v1.5 | 33M | 384 | MIT | via FT | ~58 | zero-shot + fine-tune | Efficiency story |
| 4 | gte-modernbert-base | 149M | 768 | Apache | Yes | 66.4 | zero-shot + fine-tune | Primary result |
| 5 | nomic-embed-text-v1.5 | 137M | 768 | Apache | Yes | 62.3 | zero-shot + fine-tune | MRL reference |
| 6 | pplx-embed-v1-0.6b | 600M | 1536 | Apache | Yes | 62.41 | zero-shot | SOTA ceiling |

## Why These 5

**BM25** — the system to beat. If embeddings can't beat BM25 on typo/partial queries, nothing else matters.

**all-MiniLM-L6-v2** — 22M params, Apache. Zero-shot only. Shows the baseline failure mode of off-the-shelf models on entity records. Provides the floor that makes fine-tuning look good.

**bge-small-en-v1.5** — 33M params, MIT license. The narrative model. Proves that a tiny MIT-licensed model fine-tuned on entity triplets beats BM25. This is the production argument: you don't need big models, you need the right training data. Add MRL via MatryoshkaLoss during fine-tuning.

**gte-modernbert-base** — 149M params, Apache. Primary fine-tune target. Best MTEB-per-parameter of any 2025 model under 200M. ModernBERT backbone with RoPE + Flash Attention. MRL native. Short structured text (entity records are 10-30 tokens) is exactly its sweet spot.

**nomic-embed-text-v1.5** — 137M params, Apache. MRL baked in from training (not post-hoc). Established 2024 MRL baseline. Compare fine-tuned nomic vs fine-tuned gte-modernbert to isolate architecture sensitivity. IMPORTANT: requires search_query:/search_document: prefixes.

**pplx-embed-v1-0.6b** — 600M params, Apache. Current SOTA <1B as of March 2026. #1 MTEB Retrieval among all <1B models (62.41 NDCG@10). Zero-shot only on M3 Pro — too large to fine-tune safely. Sets the quality ceiling: if fine-tuned bge-small approaches this zero-shot, the data argument is proven.

## What This Answers Monday

The narrative arc:

1. BM25 crushes pristine queries, fails on typos/partial fields
2. Off-the-shelf MiniLM also fails — it's not a model size problem
3. Zero-shot pplx-embed-v1 (SOTA) also fails on entity-specific corruptions
4. Fine-tuned bge-small (33M, MIT) beats BM25 on typo/partial buckets → **training data is the unlock**
5. Fine-tuned gte-modernbert beats everything on all buckets → **right architecture + right data = production-ready**

## Critical Implementation Notes

### Inference Quirks Per Model

**nomic-v1.5**: MUST prepend prefixes
- Queries: `"search_query: " + text`
- Docs: `"search_document: " + text`
- Skipping this loses 3-5% recall. Handle in eval harness via model config.

**pplx-embed-v1-0.6b**: Uses separate system prompts for query vs doc
- Query system prompt: "You are a helpful assistant that retrieves relevant documents for a given query."
- Doc system prompt: "You are a helpful assistant that represents documents for retrieval."
- Architecture: decoder-only, EOS token pooling
- Use sentence-transformers with `prompt_name="query"` / `prompt_name="document"`

**bge-small**: No prefix needed. CLS token pooling.

**gte-modernbert-base**: No prefix needed. CLS token pooling.

**all-MiniLM-L6-v2**: No prefix needed. Mean pooling.

### MRL Dimensions

Fine-tuning produces checkpoints at all dims simultaneously via MatryoshkaLoss:
- gte-modernbert + nomic: [768, 512, 256, 128, 64]
- bge-small: [384, 256, 128, 64]

Use 64D binary HNSW for first-stage ANN retrieval → full precision re-rank on top-1000 candidates.

### Quantization Ablation (Sunday, if time allows)
For each fine-tuned model at each dim: FP32 → INT8 → Binary
Report recall drop and index size reduction.

## Cut Models and Why

| Model | Why Cut |
|-------|---------|
| mxbai-embed-large-v1 | 335M, OOM on M3 Pro fine-tuning. pplx-0.6b covers quality ceiling. |
| bge-base-en-v1.5 | gte-modernbert is strictly better at same param count. |
| nomic-embed-text-v2-moe | MoE fine-tuning is complex. Not weekend-feasible. |
| granite-30m/125m | No MRL native. bge-small covers the small model slot. |
| snowflake-arctic-embed-m-v1.5 | Good model but redundant with gte-modernbert. Cut for time. |
| pplx-embed-v1-8b | 8B won't run on M3 Pro. |
| jina-embeddings-v3 | CC BY-NC license. |

## Weekend Execution

**Saturday AM**: Dataset generation + quality pipeline (1.2M profiles, triplets, eval set)
**Saturday PM**: BM25 baseline + eval harness. All 6 buckets, all metrics.
**Saturday evening**: Launch fine-tuning: bge-small first (fastest), then gte-modernbert overnight.
**Sunday AM**: Fine-tuning done. Build FAISS HNSW indexes. Run zero-shot eval on all 5 models.
**Sunday PM**: Run fine-tuned eval. Quantization ablation. Results viz notebook. Write-up for Monday.
