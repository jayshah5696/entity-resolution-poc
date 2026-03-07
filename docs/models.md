# Model Roster: Deep-Dive Reference

**Project:** entity-resolution-poc
**Purpose:** Architecture-level reference for all models under evaluation.

---

## Comparison Table

| Model | Params | Native Dims | MRL Native | License | MTEB Avg | Max Seq Len | Architecture |
|-------|--------|-------------|------------|---------|----------|-------------|--------------|
| BM25 (rank_bm25) | — | — | — | MIT | — | ∞ | Lexical/TF-IDF |
| nomic-embed-text-v1.5 | 137M | 768 | Yes | Apache2 | 62.28 | 8192 | ModernBERT |
| bge-small-en-v1.5 | 33M | 384 | No | MIT | 51.68 | 512 | BERT-small |
| bge-base-en-v1.5 | 109M | 768 | No | MIT | 53.25 | 512 | BERT-base |
| mxbai-embed-large-v1 | 335M | 1024 | Yes | Apache2 | 64.68 | 512 | RoBERTa-large |
| all-MiniLM-L6-v2 | 22M | 384 | No | Apache2 | 44.54 | 256 | MiniLM-L6 |
| gte-modernbert-base | 149M | 768 | Yes | Apache2 | 63.48 | 8192 | ModernBERT |
| arctic-embed-m-v1.5 | 109M | 768 | Yes | Apache2 | 57.24 | 512 | BERT-based |

**MRL Native:** Does the base pre-trained checkpoint already support Matryoshka sub-embeddings? If Yes, sub-dims are immediately usable without fine-tuning. If No, MRL must be trained from scratch with MRL wrapper.

**MTEB Avg:** MTEB English benchmark average score. Higher is better general-purpose embedding quality. Note: MTEB is NOT entity resolution-specific — a lower MTEB score doesn't mean worse entity resolution performance after fine-tuning.

---

## 1. BM25 Baseline

### Overview

BM25 (Best Match 25) is a probabilistic retrieval function derived from the BM (Best Match) family of ranking functions. It is the standard lexical baseline for all retrieval tasks.

**Library:** `rank_bm25` (Python) — `BM25Okapi` implementation

### Algorithm

Given a query Q with terms {q₁, q₂, ..., qₙ} and a document D in corpus C:

```
Score(D, Q) = Σᵢ IDF(qᵢ) × [f(qᵢ, D) × (k₁ + 1)] / [f(qᵢ, D) + k₁ × (1 - b + b × |D| / avgdl)]

Where:
  f(qᵢ, D) = term frequency of qᵢ in D
  |D|       = document length in tokens
  avgdl     = average document length across corpus
  k₁        = 1.5 (term frequency saturation)
  b         = 0.75 (length normalization)
  IDF(qᵢ)  = log((N - n(qᵢ) + 0.5) / (n(qᵢ) + 0.5))
```

### Parameters Used

| Parameter | Value | Effect |
|-----------|-------|--------|
| k1 | 1.5 | Controls term frequency saturation. Lower = faster saturation. |
| b | 0.75 | Controls length normalization. 0 = no normalization, 1 = full. |
| tokenization | whitespace + lowercase | Preserves dots (email local parts), @ symbol as a sub-token effect |

### Strengths for This Task

- Exact token match: If query says "Google" and index says "Google", high score — perfect for pristine records.
- Fast: O(1) per token after inverted index construction. Extremely low latency.
- No model needed: No GPU, no download, no warmup time.
- Interpretable: Score decomposition is fully transparent.

### Weaknesses for This Task

- Zero-overlap = zero-score: "Jon" vs "Jonathan" → 0 overlap → 0 contribution. Abbreviations destroy recall.
- Typo blindness: "Smyth" vs "Smith" → different tokens, no credit.
- No field semantics: BM25 doesn't know that "first_name: Jon" and "first_name: Jonathan" are semantically similar — it treats "first_name" as just another token.
- Short document issues: At 5-field records (typically 8-25 tokens), length normalization is noisy. avgdl is small, variance is high.
- Missing field problem: `[MISSING]` becomes a high-IDF token (rare in corpus → high weight) — but it's a marker, not semantically meaningful. Queries with many `[MISSING]` tokens get dominated by this token in scoring.

### Expected Performance

| Bucket | Expected R@1 |
|--------|-------------|
| pristine | 0.82-0.88 |
| missing_firstname | 0.60-0.70 |
| missing_email_company | 0.10-0.20 |
| typo_name | 0.30-0.40 |
| domain_mismatch | 0.50-0.60 |
| swapped_attributes | 0.05-0.15 |

---

## 2. nomic-embed-text-v1.5

### Overview

Produced by Nomic AI, nomic-embed-text-v1.5 is a ModernBERT-based text embedding model with native Matryoshka Representation Learning support. It is the **primary fine-tuning target** for this research.

**HuggingFace:** `nomic-ai/nomic-embed-text-v1.5`
**License:** Apache 2.0
**Params:** 137M
**Architecture:** ModernBERT encoder (rotary position embeddings, Flash Attention 2, efficient attention)

### Architecture Details

ModernBERT replaces traditional BERT attention with:
- **Rotary Position Embeddings (RoPE):** Better handling of position information; extends naturally to long sequences (up to 8192 tokens).
- **Flash Attention 2:** Memory-efficient attention implementation; enables larger batch sizes.
- **Alternating local/global attention:** Every other layer uses local attention (window=128 tokens) rather than full global attention, reducing compute on long sequences.
- **Removed token type embeddings:** Simplified architecture, standard for modern encoders.
- **Trained with MRL:** The model was trained with Matryoshka loss during pre-training, so sub-vectors (first 64 dims, first 128 dims, etc.) are immediately meaningful without additional training.

### Embedding Dimensions (MRL)

| Dims | Quality (relative) | Memory per 1M vecs |
|------|-------------------|-------------------|
| 768 | 100% (reference) | 3.07 GB (FP32) |
| 512 | ~98.5% | 2.05 GB |
| 256 | ~96% | 1.02 GB |
| 128 | ~92% | 0.51 GB |
| 64 | ~86% | 0.26 GB |

*Quality estimates from Nomic's ablation on BEIR benchmark — actual task performance may vary.*

### Why This Model for Entity Resolution

1. **Native MRL:** Sub-vectors are immediately usable. We can use 64-dim for Stage 1 ANN and 768-dim for Stage 2 re-rank without any additional training for MRL (though our fine-tuning will further optimize the MRL structure for our schema).

2. **Long context window (8192):** Our records are at most ~50 tokens. No truncation issues, ever. This matters when we serialize records with verbose key-value format.

3. **Strong starting point (MTEB 62.28):** A good zero-shot baseline before fine-tuning means we need fewer training samples to reach peak performance.

4. **Active development + good ST integration:** Well-supported by `sentence-transformers`, which is our training framework.

5. **Apache2 license:** No restrictions on commercial deployment.

### Trust Remote Code

Nomic models require `trust_remote_code=True` when loading. This is expected — the custom modeling code adds the RoPE modifications. Code should be reviewed before production deployment.

### Strengths for Entity Resolution

- Long context window handles verbose records without truncation
- Native MRL ideal for two-stage architecture
- Strong semantic understanding for abbreviation/typo robustness
- `[MISSING]` tokens are interpretable as OOV-ish tokens, not noise

### Weaknesses for Entity Resolution

- Slightly larger than bge-base (137M vs 109M) — slower inference on MPS
- `trust_remote_code=True` required — slight security consideration for production
- Overkill seq_len for our 5-field records (8192 >> 50 tokens) but no performance downside

---

## 3. bge-small-en-v1.5

### Overview

BAAI's BGE-small is a compact BERT-small encoder intended for efficient inference. Part of the BGE (BAAI General Embedding) family.

**HuggingFace:** `BAAI/bge-small-en-v1.5`
**License:** MIT
**Params:** 33M
**Architecture:** BERT-small (12 layers → 6 layers, 512 hidden → 384 hidden)

### Architecture Details

Standard BERT architecture with reduced depth (6 layers) and width (384 hidden dims). Trained with:
- Contrastive learning on a large text corpus
- Hard negative mining
- Knowledge distillation from larger BGE models

No native MRL support. If MRL is desired, the MRL wrapper must be applied during fine-tuning, adding additional loss terms.

### Use Case in This Research

**Speed/quality tradeoff baseline.** At 33M params, this model is ~4× smaller than nomic v1.5 and should be ~4× faster at inference. If its post-fine-tuning quality is comparable to larger models, it's a strong production choice for latency-sensitive deployments.

### Embedding Dimensions

Native: 384. MRL would target [384, 256, 128, 64].

### MTEB Score

51.68 — lower than nomic/bge-base. Reflects smaller model capacity.

### Strengths for Entity Resolution

- Very fast inference (33M params, 384-dim output)
- MIT license (most permissive)
- Well-studied — lots of prior art on this architecture
- Low memory footprint for the full-precision index

### Weaknesses for Entity Resolution

- Smaller model = less capacity to learn corruption patterns
- No native MRL — additional training overhead
- 384-dim max (vs 768-dim for larger models) limits maximum representational quality
- 512 max seq len — fine for our records, but tighter than nomic/gte

---

## 4. bge-base-en-v1.5

### Overview

BAAI's BGE-base, the standard-size member of the BGE family. Direct comparison target vs nomic-embed-v1.5 at approximately the same parameter count.

**HuggingFace:** `BAAI/bge-base-en-v1.5`
**License:** MIT
**Params:** 109M
**Architecture:** BERT-base (12 layers, 768 hidden dims)

### Architecture Details

Standard BERT-base architecture. Trained with:
- Contrastive learning on a large text corpus including MS-MARCO, NLI datasets, and web-crawled pairs
- Hard negative mining (likely BM25-mined negatives from MS-MARCO)
- Self-distillation from BGE-large

No native MRL. MRL wrapper required for fine-tuning with MRL.

### Use Case in This Research

**Architecture comparison vs nomic-embed-v1.5.** Both are ~109-137M params with 768-dim output. BGE-base uses traditional BERT (sinusoidal position embeddings, no Flash Attention), while nomic uses ModernBERT (RoPE, Flash Attention). This comparison tests whether the architectural improvements in ModernBERT translate to better entity resolution.

### Strengths for Entity Resolution

- Well-calibrated 768-dim embeddings
- MIT license
- Strong MTEB for its size class
- No `trust_remote_code` required

### Weaknesses for Entity Resolution

- No native MRL — must train MRL from scratch (higher fine-tuning cost)
- Traditional BERT attention — less efficient for long sequences (though our sequences are short)
- 512 max seq len — sufficient for our records

---

## 5. mxbai-embed-large-v1

### Overview

Mixedbread AI's flagship embedding model based on RoBERTa-large. The largest model in this roster and the MTEB leader among our candidates.

**HuggingFace:** `mixedbread-ai/mxbai-embed-large-v1`
**License:** Apache 2.0
**Params:** 335M
**Architecture:** RoBERTa-large (24 layers, 1024 hidden dims)

### Architecture Details

RoBERTa-large is a robustly trained BERT variant:
- 24 transformer layers (vs 12 in BERT-base)
- 1024 hidden dimensions (vs 768)
- Dynamic masking for MLM pre-training
- Trained on more data with larger batches than original BERT

Mixedbread fine-tuned this with:
- Native MRL training with Matryoshka loss
- Multiple Negatives Ranking Loss on large-scale retrieval datasets
- MRL dimensions: [1024, 512, 256, 128, 64]

### Use Case in This Research

**Quality ceiling.** At 335M params, this is the most expensive but potentially highest quality model. If fine-tuned nomic-embed-v1.5 (137M) matches or beats it, we have an excellent efficiency story. If mxbai-large is significantly better, it guides the production model choice.

**Note:** Fine-tuning this model on MPS with batch_size=256 may cause OOM. Reduce to batch_size=64-128 if needed.

### Strengths for Entity Resolution

- Highest MTEB score in roster (64.68)
- Largest capacity — best at learning subtle disambiguation
- Native MRL — all sub-dims work immediately
- RoBERTa-large is the dominant architecture for many retrieval benchmarks

### Weaknesses for Entity Resolution

- 335M params = slow inference (~2.5× nomic-v1.5)
- 1024-dim output makes FP32 index larger (1024/768 = 33% more memory)
- Fine-tuning expensive (time + memory)
- 512 max seq len (fine for our use case)

---

## 6. all-MiniLM-L6-v2

### Overview

The ubiquitous sentence-transformers reference model. Highly optimized for speed.

**HuggingFace:** `sentence-transformers/all-MiniLM-L6-v2`
**License:** Apache 2.0
**Params:** 22M
**Architecture:** MiniLM-L6 (6 layers, 384 hidden dims via knowledge distillation)

### Architecture Details

Knowledge-distilled from a larger model (MiniLM-L12) using a teacher-student setup:
- 6 transformer layers (vs 12 in BERT-base)
- 384 hidden dimensions
- Trained on 1 billion sentence pairs
- Distillation preserves attention distributions from larger model

**Known limitation:** 256 max sequence length. Fine for our 5-field records but worth noting.

### Use Case in This Research

**Lower bound reference.** This is the model you use when you need something fast and don't care too much about quality. If our fine-tuned approach can't beat zero-shot MiniLM-L6 on a specialized task, something is fundamentally wrong with the approach.

### Strengths for Entity Resolution

- Extremely fast (22M params, 384-dim)
- Very well-studied — huge amount of prior work
- No trust remote code requirement
- Can embed 1M records in minutes, not hours

### Weaknesses for Entity Resolution

- 256 max seq len (our records are typically <50 tokens, so fine)
- No MRL support
- 22M params may not have enough capacity for complex corruption patterns
- MTEB 44.54 — significantly below the 60+ range of larger models

---

## 7. gte-modernbert-base

### Overview

Alibaba DAMO Academy's GTE (General Text Embeddings) model built on ModernBERT. A strong competitor to nomic-embed-v1.5 using the same underlying architecture.

**HuggingFace:** `Alibaba-NLP/gte-modernbert-base`
**License:** Apache 2.0
**Params:** 149M
**Architecture:** ModernBERT encoder (same as nomic-embed-v1.5 base architecture)

### Architecture Details

Shares the ModernBERT architecture with nomic-embed-v1.5:
- Rotary Position Embeddings (RoPE)
- Flash Attention 2
- Alternating local/global attention
- 8192 token context window
- Native Matryoshka training during pre-training

**Key difference from nomic:** Alibaba's training data and training recipe. GTE models have historically been strong on retrieval benchmarks and are widely used in production.

### Use Case in This Research

**Direct architectural comparison vs nomic-embed-v1.5.** Both use ModernBERT, similar param count, similar dims. The difference is the pre-training corpus and fine-tuning recipe. This isolates the effect of pre-training data choices on downstream entity resolution quality.

### MTEB Score

63.48 — marginally higher than nomic-embed-v1.5 (62.28) on general benchmarks. Interesting to see if this translates to better entity resolution.

### Strengths for Entity Resolution

- Same ModernBERT architecture as nomic → same efficiency benefits
- Slightly higher MTEB ceiling
- 8192 context window
- Native MRL
- Apache2 license

### Weaknesses for Entity Resolution

- Slightly larger (149M vs 137M for nomic) — negligible difference in practice
- Less community deployment experience than nomic in sentence-transformers ecosystem
- May require model-specific pooling configuration

---

## 8. snowflake-arctic-embed-m-v1.5

### Overview

Snowflake's arctic-embed-m is explicitly designed for production retrieval workloads. Interesting addition because Snowflake is itself a data platform company and may have optimized for structured data retrieval patterns.

**HuggingFace:** `Snowflake/snowflake-arctic-embed-m-v1.5`
**License:** Apache 2.0
**Params:** 109M
**Architecture:** BERT-based (modified for retrieval; specific architecture not fully disclosed)

### Architecture Details

Based on a BERT-style encoder with Snowflake's proprietary retrieval-focused modifications:
- Trained on curated retrieval datasets with hard negative mining
- Native Matryoshka support (added in v1.5)
- Separate query and passage encoders (asymmetric model — query encoder is different from document encoder)
- 512 max seq len

**Important:** arctic-embed models may use an asymmetric query/document encoder approach. Verify the correct API for querying vs indexing when running evals (may need to use `encode_queries()` vs `encode_corpus()`).

### Use Case in This Research

**Production-focused comparison.** Snowflake built this for production retrieval. If it performs well on our task with minimal fine-tuning, it suggests the model has implicitly learned structured data patterns. Also a good benchmark for "what does a model trained by a data platform company produce?"

### Strengths for Entity Resolution

- Native MRL (v1.5 addition)
- Production-focused training — likely robust to noisy data
- Same parameter scale as bge-base (109M) — good comparison point
- Apache2 license
- Well-maintained by a well-resourced company

### Weaknesses for Entity Resolution

- 512 max seq len (sufficient but not generous)
- Asymmetric encoder architecture may require care in eval code
- Less MTEB score transparency than nomic/bge
- May not be asymmetric — verify before running eval

---

## Implementation Notes

### Loading Models in Code

```python
from sentence_transformers import SentenceTransformer

# nomic requires trust_remote_code
nomic = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1.5",
    trust_remote_code=True
)

# others do not
bge = SentenceTransformer("BAAI/bge-base-en-v1.5")

# arctic may need special tokenizer settings
arctic = SentenceTransformer("Snowflake/snowflake-arctic-embed-m-v1.5")
# Check if encode_queries / encode_documents asymmetry applies
```

### MRL Truncation

For models with native MRL, truncate embeddings before indexing:

```python
import numpy as np

def get_64d_embedding(full_embedding: np.ndarray) -> np.ndarray:
    """Get 64-dim sub-embedding from MRL model output."""
    sub = full_embedding[:64]
    # L2 normalize the sub-embedding (important for cosine similarity to work correctly)
    norm = np.linalg.norm(sub)
    return sub / norm if norm > 0 else sub
```

### Adding New Models

To add a new model to the roster:
1. Add entry to `configs/models.yaml` with all required fields.
2. Ensure `sentence-transformers` can load it (test with `SentenceTransformer(hf_id)`).
3. Add an experiment config in `experiments/00N_model_name/config.json`.
4. Run eval with `src/eval/run_eval.py`.
