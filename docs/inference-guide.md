# Model Inference Guide

Per-model encoding details. The encoder wrapper in `src/models/encoder.py` reads from `configs/models.yaml` and handles these automatically. This doc explains why each model needs different handling.

---

## BM25 (rank_bm25)

Not a neural model. No embeddings. Tokenize text by whitespace, build inverted index, score with BM25(k1=1.5, b=0.75).

Encoding: none. Index build = tokenize all documents and call BM25Okapi([tokens]).
Query: tokenize query, call bm25.get_top_n(query_tokens, corpus, n=top_k).

Storage: pickled BM25Okapi object at `results/indexes/bm25_{fmt}/index.pkl`. Also store entity_id list so results can be mapped back to ground truth.

Serialization sensitivity: BM25 is purely lexical, so pipe vs KV format changes which tokens appear. KV adds prefix tokens (fn:, ln:, org:) which BM25 will match on -- this could help or hurt depending on query format. Test both.

---

## all-MiniLM-L6-v2 (sentence-transformers/all-MiniLM-L6-v2)

Pooling: mean pooling over all token embeddings.
Query prefix: none.
Doc prefix: none.
Normalization: L2-normalize before storing (cosine similarity = dot product on normalized vectors).

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(texts, normalize_embeddings=True, batch_size=256)
```

Same encode function for queries and docs.

---

## bge-small-en-v1.5 (BAAI/bge-small-en-v1.5)

Pooling: CLS token.
Query prefix: none (BGE recommends "Represent this sentence: " for some tasks but NOT for symmetric retrieval -- entity resolution is symmetric so no prefix).
Doc prefix: none.
Normalization: L2-normalize.

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
embeddings = model.encode(texts, normalize_embeddings=True, batch_size=256)
```

Fine-tuning note: MRL not native. Add via MatryoshkaLoss wrapping MNRL during fine-tune:
```python
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
inner = MultipleNegativesRankingLoss(model)
loss = MatryoshkaLoss(model, inner, matryoshka_dims=[384, 256, 128, 64])
```

---

## gte-modernbert-base (Alibaba-NLP/gte-modernbert-base)

Pooling: CLS token.
Query prefix: none.
Doc prefix: none.
Normalization: L2-normalize.

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("Alibaba-NLP/gte-modernbert-base", trust_remote_code=True)
embeddings = model.encode(texts, normalize_embeddings=True, batch_size=128)
```

Note: trust_remote_code=True required. Batch size 128 on M3 Pro (768D x 128 x float32 = ~38MB per batch, safe).

MRL native: Yes. Truncate embeddings to desired dim after encoding. All dims are valid.

---

## nomic-embed-text-v1.5 (nomic-ai/nomic-embed-text-v1.5)

Pooling: mean pooling.
Query prefix: REQUIRED -- prepend "search_query: " to every query at inference.
Doc prefix: REQUIRED -- prepend "search_document: " to every document at inference.
Normalization: L2-normalize.

Skipping the prefix loses 3-5% recall. This applies at both zero-shot and fine-tuned eval. During fine-tuning, prefixes must also be applied to training data.

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

query_embeddings = model.encode(
    ["search_query: " + q for q in queries],
    normalize_embeddings=True, batch_size=128
)
doc_embeddings = model.encode(
    ["search_document: " + d for d in docs],
    normalize_embeddings=True, batch_size=128
)
```

MRL native: Yes. Truncate after encoding.

---

## pplx-embed-v1-0.6b (perplexity-ai/pplx-embed-v1-0.6b)

Architecture: decoder-only transformer (Qwen2.5-0.5B base). Uses EOS token pooling (last token representation).
Pooling: EOS token (last token of sequence).
Normalization: L2-normalize after extraction.

DIFFERENT system prompts for queries vs documents:
- Query system prompt: "You are a helpful assistant that retrieves relevant documents for a given query."
- Doc system prompt: "You are a helpful assistant that represents documents for retrieval."

Zero-shot only on M3 Pro. 600M params -- fine-tuning would require gradient checkpointing and is not planned for this POC.

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("perplexity-ai/pplx-embed-v1-0.6b", trust_remote_code=True)

query_embeddings = model.encode(queries, prompt_name="query", normalize_embeddings=True, batch_size=32)
doc_embeddings = model.encode(docs, prompt_name="document", normalize_embeddings=True, batch_size=32)
```

Note: batch_size=32 on M3 Pro (1536D x 32 x float32 x 600M param model -- watch memory).

Paper: https://research.perplexity.ai/articles/pplx-embed-state-of-the-art-embedding-models-for-web-scale-retrieval

---

## Quantization

Applied after encoding, before storing in LanceDB.

FP32: default, no transformation.
INT8: scale each vector to [-128, 127] range per-vector. Use numpy: `(vec / np.max(np.abs(vec)) * 127).astype(np.int8)`. Recall loss ~1%.
Binary: sign binarization. `(vec > 0).astype(np.uint8)`. Pack with np.packbits. Recall loss ~3-5%. Compare with Hamming distance (or XOR + popcount).

For this POC, start with FP32 only. Add INT8 and binary as a Sunday afternoon ablation if time allows.

---

## Encoder Wrapper Contract

`src/models/encoder.py` exposes:

```python
def load_encoder(model_key: str, models_config: dict) -> BaseEncoder:
    """Returns an encoder object for the given model key."""

class BaseEncoder:
    def encode_docs(self, texts: list[str], batch_size: int = 128) -> np.ndarray: ...
    def encode_queries(self, texts: list[str], batch_size: int = 128) -> np.ndarray: ...
    def dim(self) -> int: ...
    def model_key(self) -> str: ...
```

The encoder reads query_prefix and doc_prefix from models.yaml and applies them automatically. Eval scripts call encode_docs() and encode_queries() and never contain model-specific logic.
