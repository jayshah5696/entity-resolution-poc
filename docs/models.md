# Model Roster

Complete analysis of all embedding models evaluated in this research. All selected models are Apache 2.0 or MIT licensed with open weights.

## Quick Comparison Table

| Model | Params | Dims | License | MRL Native | MTEB Avg | Released | Notes |
|-------|--------|------|---------|-----------|----------|----------|-------|
| BM25 | — | — | Apache | — | baseline | — | Lexical floor |
| all-MiniLM-L6-v2 | 22M | 384 | Apache | No | 56.3 | 2021 | Floor baseline only |
| bge-small-en-v1.5 | 33M | 384 | MIT | No (add via FT) | ~62 | 2024 | Smallest production model |
| granite-embedding-30m | 30M | 768 | MIT | No | ~58 | 2025-Q1 | Ultra-small reference |
| granite-embedding-125m | 125M | 768 | Apache | No | ~61 | 2025-Q1 | IBM enterprise quality |
| bge-base-en-v1.5 | 109M | 768 | MIT | No (add via FT) | 64.3 | 2024 | Solid MIT baseline |
| nomic-embed-text-v1.5 | 137M | 768 | Apache | YES | 62.3 | 2024-02 | MRL baked in |
| snowflake-arctic-embed-m-v1.5 | 109M | 768 | Apache | YES | 64.6 | 2024-Q4 | MRL is core design |
| nomic-embed-text-v2-moe | 475M (137M active) | 768 | Apache | YES | ~63 | 2025-Q1 | MoE efficiency |
| **gte-modernbert-base** | **149M** | **768** | **Apache** | **YES** | **66.4** | **2025-01** | **TOP PICK** |
| mxbai-embed-large-v1 | 335M | 1024 | Apache | YES | 64.7 | 2024-Q1 | Quality ceiling |

---

## The Perplexity AI Situation

Jay mentioned Perplexity released an embedding model "last week." After a thorough research sweep:

**Perplexity AI has NOT released an open-source embedding model.**

What they actually have:
- `perplexity-ai/r1-1776` on HuggingFace — a DeepSeek-R1 reasoning fine-tune (language model, MIT, released Feb 2025). NOT an embedding model.
- A proprietary embedding endpoint in their Sonar API, model internally referenced as "r-embed-v1" in some API examples. API-only, closed weights, no public specs for dimensions or architecture.

**Verdict**: Excluded. Cannot download weights, cannot reproduce, cannot compare. If they open-source it later, add it to this roster.

---

## Primary Candidates (Deep Dive)

### gte-modernbert-base — TOP PICK
**HuggingFace**: `Alibaba-NLP/gte-modernbert-base`  
**Released**: January 2025  
**Params**: 149M | **Dims**: 768 | **License**: Apache 2.0  
**MRL Native**: Yes | **MTEB Avg**: 66.4  

Architecture: Built on ModernBERT — the first major BERT re-architecture since 2019. Key improvements:
- Rotary position embeddings (RoPE) instead of absolute positions → better at arbitrary token positions
- Flash Attention 2 → 2-4x faster than standard attention at same params
- 8192 token context (overkill for entity records, but means no truncation ever)
- Alternating local + global attention → efficient for short text (our use case)

Why it wins for entity resolution specifically:
- Character-level tokenization (WordPiece on ModernBERT vocab) is well-calibrated for name variations
- Bidirectional attention = full field-to-field awareness in a single record string
- Short text (entity records are 10-30 tokens) is where ModernBERT's local attention is fastest
- MRL means we can compress to 64D for HNSW first-stage retrieval without a separate training run
- Best MTEB/param ratio of any 2025 model under 200M params

Weaknesses: Slightly larger than bge-small. ModernBERT is newer so less community fine-tuning knowledge.

---

### nomic-embed-text-v1.5 — MRL REFERENCE
**HuggingFace**: `nomic-ai/nomic-embed-text-v1.5`  
**Released**: February 2024  
**Params**: 137M | **Dims**: 768 | **License**: Apache 2.0  
**MRL Native**: Yes | **MTEB Avg**: 62.3  

The original open-source MRL model. MRL was trained into the model from the start (not post-hoc), meaning the first 64 dimensions are genuinely a good embedding, not a truncated bad one.

Important quirk: Requires `search_query:` prefix on queries and `search_document:` prefix on documents at inference time. If you skip this, recall drops ~3-5%. Must be consistent in training and eval.

Why include alongside gte-modernbert: It's the established MRL baseline. Fine-tuning it creates a clean before/after comparison. Also 137M vs 149M — nearly identical cost, different architecture. The diff in results tells you something about architecture sensitivity.

---

### snowflake-arctic-embed-m-v1.5 — MRL DESIGNED FOR RETRIEVAL
**HuggingFace**: `Snowflake/snowflake-arctic-embed-m-v1.5`  
**Released**: Q4 2024  
**Params**: 109M | **Dims**: 768 | **License**: Apache 2.0  
**MRL Native**: Yes | **MTEB Avg**: 64.6  

Snowflake trained this specifically for retrieval pipelines where dimension reduction is needed. The MRL implementation is reportedly among the cleanest — dimension compression causes less recall drop than other models. Good for the quantization ablation experiment.

---

### nomic-embed-text-v2-moe — MoE EFFICIENCY
**HuggingFace**: `nomic-ai/nomic-embed-text-v2-moe`  
**Released**: Q1 2025  
**Params**: 475M total, ~137M active | **Dims**: 768 | **License**: Apache 2.0  
**MRL Native**: Yes | **MTEB Avg**: ~63  

Mixture of Experts architecture: 8 experts, only 2 active per token at inference. Net compute ≈ 137M params despite 475M stored. Interesting data point: does MoE help entity resolution where different experts might specialize in name vs company vs email tokens?

Caveat: 475M param storage on M3 Pro is fine (model fits in memory) but fine-tuning all experts is expensive. Likely treat this as zero-shot eval only or LoRA fine-tune.

---

### bge-small-en-v1.5 — THE EFFICIENCY STORY
**HuggingFace**: `BAAI/bge-small-en-v1.5`  
**Released**: Q1 2024  
**Params**: 33M | **Dims**: 384 | **License**: MIT  
**MRL Native**: No (add via MatryoshkaLoss during fine-tuning)  

The narrative model. At 33M params with MIT license, if fine-tuning on entity triplets gets this to competitive recall against BM25, that's the production argument: you don't need a large model, you need the right training data. This is the model that makes the Monday story land.

MRL addition during fine-tuning:
```python
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
inner_loss = MultipleNegativesRankingLoss(model)
loss = MatryoshkaLoss(model, inner_loss, matryoshka_dims=[384, 256, 128, 64])
```
One fine-tuning run gives you all dimension checkpoints.

---

### mxbai-embed-large-v1 — QUALITY CEILING
**HuggingFace**: `mixedbread-ai/mxbai-embed-large-v1`  
**Params**: 335M | **Dims**: 1024 | **License**: Apache 2.0  
**MRL Native**: Yes | **MTEB Avg**: 64.7  

Largest model in roster. Use zero-shot as the quality ceiling — if fine-tuned bge-small approaches mxbai-large zero-shot, you've proven the training data argument decisively. Don't fine-tune this on M3 Pro (335M + MNRL with large batch will OOM).

---

### IBM Granite Embedding — 2025 NEWCOMERS
**granite-embedding-125m-english**: `ibm-granite/granite-embedding-125m-english` — 125M, 768D, Apache 2.0  
**granite-embedding-30m-english**: `ibm-granite/granite-embedding-30m-english` — 30M, 768D, MIT  

Released Q1 2025. IBM's enterprise embedding series. No MRL native. Include granite-30m as your absolute smallest data point in the size ablation — 30M params at 768D is architecturally interesting (bge-small is 33M at 384D, but granite-30m is 768D at similar param count via width-vs-depth tradeoffs).

---

## Excluded Models

### Jina Embeddings v3
**Why excluded**: CC BY-NC 4.0 license. Not commercial-friendly. Cannot include in a work research project.

### Perplexity (any model)
**Why excluded**: No open-source embedding model exists. See section above.

### ColBERTv2
**Why excluded from retrieval**: 128D per token × ~20 tokens/record × 500M records = ~500TB storage. Impractical. Keep as re-ranker discussion point only.

### SPLADE v2
**Why excluded from primary eval**: 30K-dim sparse vectors at 500M records. Not POC-able on M3 Pro. Document as "interesting future direction" in the paper.

---

## Fine-Tuning Priority

Order to fine-tune given M3 Pro constraint (run weekends):

1. **bge-small (33M)** — cheapest, fastest, proves training data argument
2. **gte-modernbert-base (149M)** — best 2025 architecture, primary result
3. **nomic-v1.5 (137M)** — MRL reference comparison
4. Skip fine-tuning mxbai-large (OOM risk), nomic-v2-moe (MoE fine-tuning complex)

Zero-shot eval only (no fine-tuning):
- mxbai-embed-large-v1 (quality ceiling)
- nomic-embed-text-v2-moe (MoE reference)
- granite-embedding-30m (floor reference)
- all-MiniLM-L6-v2 (absolute floor)

---

## Memory Footprint at 1M Records (Index Size)

| Model | Dims | FP32 | INT8 | Binary |
|-------|------|------|------|--------|
| bge-small | 384 | 1.5GB | 384MB | 48MB |
| gte-modernbert | 768 | 3.1GB | 768MB | 96MB |
| nomic-v1.5 | 768 | 3.1GB | 768MB | 96MB |
| mxbai-large | 1024 | 4.1GB | 1GB | 128MB |

At 500M records (extrapolated):

| Model | Dims | FP32 | INT8 | Binary |
|-------|------|------|------|--------|
| bge-small | 384 | 768GB ❌ | 192GB ⚠️ | 24GB ✅ |
| gte-modernbert | 768 | 1.5TB ❌ | 384GB ❌ | 48GB ✅ |
| gte-modernbert MRL 64D | 64 | 128GB ❌ | 32GB ✅ | 4GB ✅ |

Two-stage with gte-modernbert: 64D binary HNSW (4GB) for ANN → 768D FP32 on 1000 candidates → production-viable.
