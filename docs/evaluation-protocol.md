# Evaluation Protocol

**Project:** entity-resolution-poc
**Version:** v1
**Status:** Reference spec — all eval scripts implement this exactly

---

## 1. Metric Definitions

### 1.1 Recall@K

**Definition:** For a set of queries Q, Recall@K is the fraction of queries where the true match appears in the top-K retrieved results.

**Formula:**
```
Recall@K = (1 / |Q|) × Σ_{q ∈ Q} 𝟙[true_match(q) ∈ top_K_results(q)]

Where:
  |Q|              = number of queries
  true_match(q)    = the single correct answer for query q
  top_K_results(q) = the K highest-scored results for query q
  𝟙[·]             = indicator function (1 if true, 0 if false)
```

**Python implementation:**
```python
def recall_at_k(retrieved: list[str], relevant: str, k: int) -> float:
    """
    Args:
        retrieved: Ordered list of retrieved document IDs (best first)
        relevant: The single true match document ID
        k: Cutoff rank
    Returns:
        1.0 if relevant is in retrieved[:k], else 0.0
    """
    return float(relevant in retrieved[:k])

def mean_recall_at_k(results: list[tuple[list[str], str]], k: int) -> float:
    """Average recall@k over all queries."""
    return np.mean([recall_at_k(ret, rel, k) for ret, rel in results])
```

**Why it's the primary metric:** Entity resolution has exactly one true match per query. Recall@1 directly measures "did we get it right?" — the clearest success signal. Recall@10 measures "is it findable in the top-10?" — relevant for two-stage architectures where Stage 1 must have the true match in its candidates.

### 1.2 Precision@K

**Definition:** For binary relevance (one true match per query), Precision@K is the fraction of the top-K results that are relevant.

**Formula:**
```
Precision@K = (1 / |Q|) × Σ_{q ∈ Q} (|relevant(q) ∩ top_K(q)| / K)

For our task (one true match per query):
Precision@K = Recall@K / K
```

**Python implementation:**
```python
def precision_at_k(retrieved: list[str], relevant: str, k: int) -> float:
    """Precision@K for single-relevant-document queries."""
    return float(relevant in retrieved[:k]) / k
```

**When it matters:** Precision@K penalizes larger K values. Precision@1 = Recall@1 (the match is either the top result or not). Precision@10 = Recall@10 / 10. Useful for understanding answer density in the retrieved set.

### 1.3 MRR@K (Mean Reciprocal Rank)

**Definition:** The average of the reciprocal rank of the true match, truncated at K. MRR rewards retrieving the true match at rank 1 vs rank 10.

**Formula:**
```
MRR@K = (1 / |Q|) × Σ_{q ∈ Q} (1 / rank(true_match(q)))

Where:
  rank(true_match(q)) = position of true match in retrieved results (1-indexed)
                        = ∞ (or 0 contribution) if not in top-K
```

**Python implementation:**
```python
def reciprocal_rank_at_k(retrieved: list[str], relevant: str, k: int) -> float:
    """Reciprocal rank, truncated at K."""
    if relevant in retrieved[:k]:
        rank = retrieved.index(relevant) + 1  # 1-indexed
        return 1.0 / rank
    return 0.0

def mrr_at_k(results: list[tuple[list[str], str]], k: int) -> float:
    """Mean Reciprocal Rank@K."""
    return np.mean([reciprocal_rank_at_k(ret, rel, k) for ret, rel in results])
```

**When it matters:** MRR captures rank quality within the top-K, not just the binary "was it found?" signal of Recall@K. An MRR@10 of 0.90 means on average the true match is near rank 1. MRR@10 of 0.50 means true matches are often appearing around rank 2.

**MRR@1 = Recall@1 = Precision@1** (all equivalent for single-relevant queries).

### 1.4 NDCG@K (Normalized Discounted Cumulative Gain)

**Definition:** NDCG measures quality of ranked results with graded relevance. For binary relevance (our task), it simplifies considerably.

**Formula:**
```
DCG@K = Σ_{i=1}^{K} rel_i / log₂(i + 1)

IDCG@K = max possible DCG@K = rel_1 / log₂(2) = 1.0 / 1.0 = 1.0
          (for binary relevance, ideal is the true match at rank 1)

NDCG@K = DCG@K / IDCG@K = DCG@K (since IDCG@K = 1.0 for binary relevance)

For our task:
  rel_i = 1 if position i contains the true match, else 0
  NDCG@K = 1 / log₂(rank + 1) if true match at rank ≤ K, else 0
```

**Python implementation:**
```python
def ndcg_at_k(retrieved: list[str], relevant: str, k: int) -> float:
    """NDCG@K for single-relevant-document queries."""
    if relevant in retrieved[:k]:
        rank = retrieved.index(relevant) + 1  # 1-indexed
        return 1.0 / np.log2(rank + 1)
    return 0.0

def mean_ndcg_at_k(results: list[tuple[list[str], str]], k: int) -> float:
    """Mean NDCG@K."""
    return np.mean([ndcg_at_k(ret, rel, k) for ret, rel in results])
```

**Note on NDCG vs MRR:** For binary relevance with one relevant document, NDCG@K and MRR@K are monotonically related but not identical. MRR uses 1/rank; NDCG uses 1/log₂(rank+1). NDCG penalizes lower ranks less aggressively than MRR. Report both for completeness.

---

## 2. Evaluation Buckets — Per-Bucket Descriptions

### Bucket Design Principle

Each bucket isolates a specific failure mode. Running all 6 buckets answers: *where does each model fail, and by how much?*

### Bucket 1: pristine

**Description:** No corruption applied. Query record is identical to the indexed record.

**Query construction:**
```
first_name: Jonathan, last_name: Smith, company: Google Inc, email: jonathan.smith@google.com, country: USA
→ Serialized: "Jonathan Smith | Google Inc | jonathan.smith@google.com | USA"
```

**What failure here means:** If a model fails on pristine records, it has fundamental retrieval quality problems. This is the floor — every model should score highly here.

**BM25 expected behavior:** Strong. All tokens match. Only failure mode is a different "Jonathan Smith" at Google (common name collision) outranking the true match.

**Dense model expected behavior:** Strong. Embeddings collapse near-identical strings to nearly identical vectors.

**N per bucket:** 1,667

### Bucket 2: missing_firstname

**Description:** The first_name field is dropped (set to [MISSING]).

**Query construction:**
```
first_name: [MISSING], last_name: Smith, company: Google Inc, email: jonathan.smith@google.com, country: USA
→ Pipe: "[MISSING] | Smith | Google Inc | jonathan.smith@google.com | USA"
→ KV:   "first_name: [MISSING] last_name: Smith company: Google Inc email: jonathan.smith@google.com country: USA"
```

**What failure here means:** The model over-weights first_name. Without it, recall drops significantly. This tests whether the model learned to use the remaining fields (email contains "jonathan" as a hint, company provides context).

**BM25 expected behavior:** Moderate. Falls back to last_name + company + email tokens. Should still work for non-ambiguous last names. Fails for "Smith" + generic company.

**Dense model expected behavior:** Should use email local part ("jonathan.smith") to infer first name. Fine-tuned model trained on field_drop corruptions should handle this well.

**N per bucket:** 1,667

### Bucket 3: missing_email_company

**Description:** Both email and company are dropped. Only name and country remain.

**Query construction:**
```
first_name: Jonathan, last_name: Smith, company: [MISSING], email: [MISSING], country: USA
→ Pipe: "Jonathan Smith | [MISSING] | [MISSING] | USA"
```

**What failure here means:** The two most discriminating fields are gone. The model must find "Jonathan Smith from USA" in a corpus of millions — there could be hundreds of "Jonathan Smith | USA" records.

**BM25 expected behavior:** Poor (R@1 ~0.10-0.20). Without email or company, BM25 relies on name tokens only. Common names cause massive ranking failures.

**Dense model expected behavior:** Better than BM25 due to semantic name understanding, but still hard. The key differentiator: dense models trained on field_drop_double corruptions learn to boost recall by matching on country context and name semantics together.

**N per bucket:** 1,667

### Bucket 4: typo_name

**Description:** Levenshtein-1 or Levenshtein-2 applied to first_name or last_name.

**Query construction (Lev-1 example):**
```
first_name: Jonathen, last_name: Smith → "Jonathen Smith | Google Inc | jonathan.smith@google.com | USA"
```

**Query construction (Lev-2 example):**
```
first_name: Jonathan, last_name: Smyht → "Jonathan Smyht | Google Inc | jonathan.smith@google.com | USA"
```

**Distribution:** 50% Lev-1, 50% Lev-2. Applied to first_name (50%) or last_name (50%) per query.

**What failure here means:** The model can't handle minor spelling variations. This is the most common real-world corruption (mistyped data entry, OCR errors).

**BM25 expected behavior:** Poor (R@1 ~0.30-0.40). "Jonathen" is a different token from "Jonathan" — 0 exact match. BM25 falls back to company + email tokens. If email is intact, BM25 can still find the match via email scoring, but typo name reduces score.

**Dense model expected behavior:** Good (R@1 target ≥0.78). Embedding models trained on typos learn that "Smyth" and "Smith" are similar. Token-level representations capture character n-gram similarity implicitly.

**N per bucket:** 1,667

### Bucket 5: domain_mismatch

**Description:** Email domain is swapped (work→personal, or different domain).

**Query construction:**
```
email: "jonathan.smith@gmail.com"  (was: jonathan.smith@google.com)
→ "Jonathan Smith | Google Inc | jonathan.smith@gmail.com | USA"
```

**What failure here means:** The model over-weights email domain as a distinguishing feature. The local part ("jonathan.smith") IS informative, but models that hardcode domain matching fail here.

**BM25 expected behavior:** Moderate (R@1 ~0.50-0.60). The tokens "jonathan.smith" still match. "gmail" and "google" don't match, but other tokens do. Score should still be fairly high if company name matches.

**Dense model expected behavior:** Good (R@1 target ≥0.82). Embeddings should learn that "jonathan.smith@gmail.com" and "jonathan.smith@google.com" are similar because the local part is identical and the name matches. The model sees "gmail" and "google" as domain synonyms in context.

**N per bucket:** 1,667

### Bucket 6: swapped_attributes

**Description:** Two field values are swapped in the query record.

**Query construction (first_name ↔ company):**
```
first_name: "Google Inc", company: "Jonathan", last_name: Smith, email: jonathan.smith@google.com, country: USA
→ "Google Inc Smith | Jonathan | jonathan.smith@google.com | USA"
```

**Swap pairs used (sampled uniformly):**
- first_name ↔ company
- first_name ↔ country
- last_name ↔ company
- company ↔ email (email in company field, company in email field)

**What failure here means:** The model relies on field position/label as a strong prior. When a person's name appears in the company position, the model gets confused. This tests schema robustness.

**BM25 expected behavior:** Poor (R@1 ~0.05-0.15). Token "Google Inc" in the name position doesn't help find "Google Inc" in the company position — they cancel out in some implementations.

**Dense model expected behavior:** Moderate after fine-tuning (R@1 target ≥0.50). Harder for dense models trained on pipe format (position is semantic). Easier for key-value format with explicit labels (labels are preserved on both sides of the swap).

**N per bucket:** 1,665

---

## 3. Index Construction Parameters

### FAISS HNSW (Primary Index)

HNSW (Hierarchical Navigable Small World) is the production-grade ANN algorithm used for dense retrieval. It provides excellent recall-speed tradeoffs.

**Construction parameters:**

```python
import faiss

def build_hnsw_index(embeddings: np.ndarray, dim: int) -> faiss.IndexHNSWFlat:
    """
    Build HNSW index for cosine similarity (inner product on L2-normalized vectors).

    Args:
        embeddings: L2-normalized embedding matrix, shape (n, dim)
        dim: Embedding dimensionality

    Returns:
        FAISS HNSW index ready for search
    """
    # Use inner product metric — cosine similarity = inner product on L2-normalized vectors
    index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    # M=32: each node connected to 32 neighbors in the graph
    # Higher M → better recall, more RAM for graph structure (~M × 4 bytes per vector extra)

    index.hnsw.efConstruction = 200
    # efConstruction: exploration depth during index construction
    # Higher → more accurate graph, slower build, same memory
    # 200 is a good default for production-quality graphs

    index.hnsw.efSearch = 100
    # efSearch: exploration depth at query time
    # Higher → better recall, slower query
    # 100 recommended for efConstruction=200

    # Normalize embeddings before adding (for cosine similarity via inner product)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index
```

**Parameter rationale:**

| Param | Value | Tradeoff |
|-------|-------|----------|
| M | 32 | Connections per node. 32 = good recall, ~128 bytes overhead per vector. Increase to 64 for maximum recall at 2× overhead. |
| efConstruction | 200 | 200 = production quality. Build time ~3× vs efConstruction=40. Worth it. |
| efSearch | 100 | 100 = ~99% recall at M=32 for typical datasets. Increase to 200 for maximum recall. |
| Metric | Inner Product | With L2-normalized vectors: inner product = cosine similarity. |

### FAISS Flat (Exact Search, Quality Ceiling)

```python
def build_flat_index(embeddings: np.ndarray, dim: int) -> faiss.IndexFlatIP:
    """Exact inner product search — no approximation, quality ceiling."""
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index
```

Used to establish the quality ceiling — what's the best possible recall at this dimensionality? HNSW should achieve ≥ 99% of flat index recall at K=10.

### Binary Index (Stage 1 ANN)

```python
def build_binary_index(embeddings: np.ndarray) -> faiss.IndexBinaryHNSW:
    """
    Binary HNSW index for Stage 1 retrieval.
    Input embeddings are binarized (sign of each float → 0/1 bit).
    """
    # Binarize embeddings
    binary_codes = np.packbits((embeddings > 0).astype(np.uint8), axis=1)
    # dim/8 bytes per vector (e.g., 64-dim → 8 bytes)

    d_bits = embeddings.shape[1]  # number of bits = number of dims
    index = faiss.IndexBinaryHNSW(d_bits, 32)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 100
    index.add(binary_codes)
    return index
```

---

## 4. BM25 Index Construction and Tokenization

### Tokenization

```python
import re

def tokenize_for_bm25(text: str) -> list[str]:
    """
    Tokenize a serialized record for BM25 indexing.

    Strategy: lowercase + split on whitespace + remove pipe separators.
    We keep email local parts and domain as sub-tokens (dot-separated)
    because "jonathan.smith" contains useful signal that would be lost
    if we split on punctuation.

    Special handling: [MISSING] is kept as a single token (useful signal
    that a field is absent, though BM25 will weight it oddly).
    """
    # Lowercase
    text = text.lower()
    # Remove pipe separators (artifact of pipe format, not content)
    text = text.replace(' | ', ' ')
    # Keep [MISSING] as a single token
    text = text.replace('[missing]', '__missing__')
    # Tokenize on whitespace
    tokens = text.split()
    # Restore missing token
    tokens = ['[MISSING]' if t == '__missing__' else t for t in tokens]
    return tokens

def build_bm25_index(corpus: list[str], k1: float = 1.5, b: float = 0.75) -> BM25Okapi:
    """
    Build BM25 index over serialized record corpus.

    Args:
        corpus: List of serialized record strings
        k1: Term frequency saturation parameter (1.5 = slightly aggressive saturation)
        b: Length normalization (0.75 = moderate normalization)
    """
    from rank_bm25 import BM25Okapi
    tokenized_corpus = [tokenize_for_bm25(doc) for doc in corpus]
    return BM25Okapi(tokenized_corpus, k1=k1, b=b)
```

**Tokenization rationale:**

- **Lowercase:** Standard. "Google" and "google" should match.
- **Whitespace splitting:** Preserves email structure. "jonathan.smith@google.com" stays as one token — BM25 can match it exactly against another occurrence.
- **Keep [MISSING] as token:** Even though BM25 handles this poorly (high IDF for rare token), it's consistent with how the dense model sees the corpus.
- **No stemming:** Stemming would conflate "Google" → "googl" which is unusual. The specific form of company names and names matters.
- **No stopword removal:** "Inc" and "LLC" are kept — they're part of company names.

### BM25 Configuration

```yaml
# from configs/eval.yaml
bm25:
  k1: 1.5
  b: 0.75
  algorithm: BM25Okapi
```

**k1=1.5:** Standard choice. Controls how much term frequency affects score. At k1=1.5, doubling the term frequency increases score by ~50% (not double). Higher values give more weight to repeated terms.

**b=0.75:** Standard choice. Controls length normalization. At b=0.75, a document twice as long as average is moderately penalized. For our short, fixed-schema records, b has less effect than in general text retrieval.

---

## 5. Latency Measurement Methodology

### Goal

Measure per-query retrieval latency at realistic operating conditions. Report p50, p95, p99 latencies in milliseconds.

### Measurement Protocol

```python
import time
import numpy as np

def measure_query_latency(
    index,
    queries: list[np.ndarray],
    k: int = 10,
    warmup_n: int = 100,
    measure_n: int = 1000,
) -> dict[str, float]:
    """
    Measure per-query retrieval latency.

    Protocol:
    1. Run warmup_n queries (JIT compilation, cache warming) — discard timings
    2. Run measure_n queries, record per-query wall-clock time
    3. Report p50, p95, p99 in milliseconds

    Args:
        index: FAISS index (or BM25 index with search() method)
        queries: List of query embeddings (or tokenized query texts for BM25)
        k: Top-K results to retrieve
        warmup_n: Number of warmup queries (discarded)
        measure_n: Number of queries to time

    Returns:
        Dict with p50_ms, p95_ms, p99_ms keys
    """
    # Ensure we have enough queries
    assert len(queries) >= warmup_n + measure_n, \
        f"Need {warmup_n + measure_n} queries, got {len(queries)}"

    # Warmup — discard
    for q in queries[:warmup_n]:
        _ = index.search(q.reshape(1, -1), k)

    # Measurement — record wall-clock per query
    latencies_ms = []
    for q in queries[warmup_n:warmup_n + measure_n]:
        t0 = time.perf_counter()
        _ = index.search(q.reshape(1, -1), k)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000)

    return {
        'p50_ms':  float(np.percentile(latencies_ms, 50)),
        'p95_ms':  float(np.percentile(latencies_ms, 95)),
        'p99_ms':  float(np.percentile(latencies_ms, 99)),
        'mean_ms': float(np.mean(latencies_ms)),
        'min_ms':  float(np.min(latencies_ms)),
        'max_ms':  float(np.max(latencies_ms)),
        'n':       measure_n,
    }
```

**Measurement notes:**

1. **Single-query latency, not batch throughput.** Production entity resolution receives one query at a time per user. Batch latency is irrelevant.
2. **100-query warmup.** First queries are slower due to JIT compilation (Python), cache warming, and potential OS scheduling effects. Discarding 100 warmup queries gives stable measurements.
3. **1000 measured queries.** At n=1000, p99 is the 10th worst query — statistically meaningful.
4. **`time.perf_counter()` not `time.time()`.** `perf_counter` has nanosecond resolution; `time.time()` has lower resolution on some platforms.
5. **Repeat 3× and take median of medians.** Single measurement runs can be affected by background processes. Median of 3 runs is robust.
6. **Separate latency for embedding (query → vector) vs retrieval (vector → results).** Report both separately. Embedding latency depends on model; retrieval latency depends on index.

**Latency targets (1M record index):**

| Component | Target p99 |
|-----------|-----------|
| Query embedding (768-dim) | < 20ms |
| Stage 1 binary HNSW search | < 5ms |
| Stage 2 FP32 re-rank (100 cands) | < 2ms |
| Total two-stage pipeline | < 30ms |
| BM25 search | < 50ms (scales with index size) |

---

## 6. Result Storage Format

### Per-Experiment JSON

Each experiment produces one JSON file at `results/exp_{id}_{name}.json`.

**Schema:**

```json
{
  "experiment_id": "003",
  "experiment_name": "nomic_finetuned_pipe",
  "model": "nomic_v15_finetuned_pipe",
  "serialization": "pipe",
  "dims": 768,
  "quantization": "fp32",
  "index_size": 1000000,
  "dataset_version": "v1",
  "timestamp": "2025-01-04T14:23:10Z",
  "hardware": "Apple M3 Pro, 36GB RAM",
  "git_commit": "abc1234",

  "metrics": {
    "pristine": {
      "recall_at_1": 0.891,
      "recall_at_5": 0.962,
      "recall_at_10": 0.978,
      "precision_at_1": 0.891,
      "precision_at_5": 0.192,
      "precision_at_10": 0.098,
      "mrr_at_10": 0.921,
      "ndcg_at_10": 0.918,
      "n_queries": 1667
    },
    "missing_firstname": { "..." : "..." },
    "missing_email_company": { "..." : "..." },
    "typo_name": { "..." : "..." },
    "domain_mismatch": { "..." : "..." },
    "swapped_attributes": { "..." : "..." }
  },

  "latency": {
    "embedding_p50_ms": 8.3,
    "embedding_p95_ms": 12.1,
    "embedding_p99_ms": 18.4,
    "retrieval_p50_ms": 3.2,
    "retrieval_p95_ms": 4.8,
    "retrieval_p99_ms": 6.1,
    "total_p50_ms": 11.5,
    "total_p95_ms": 16.9,
    "total_p99_ms": 24.5,
    "n_queries_measured": 1000,
    "warmup_queries": 100
  },

  "index_info": {
    "index_type": "hnsw",
    "M": 32,
    "efConstruction": 200,
    "efSearch": 100,
    "ram_gb": 3.07,
    "build_time_s": 180
  },

  "notes": "Fine-tuned 3 epochs with curriculum hard negatives. Pipe format."
}
```

### Master Results CSV

`results/master_results.csv` aggregates all experiments. Append a new row after each experiment. Never delete rows (append-only log).

**Columns:**
```
experiment_id, model, serialization, dims, quantization, index_size, bucket,
recall_at_1, recall_at_5, recall_at_10, precision_at_1, precision_at_5, precision_at_10,
mrr_at_10, ndcg_at_10, p50_latency_ms, p95_latency_ms, p99_latency_ms,
index_ram_gb, timestamp, notes
```

Each bucket is a separate row. An experiment with 6 buckets adds 6 rows to the CSV.

---

## 7. Statistical Significance

### Minimum Sample Sizes

At our target n=1667 per bucket, the confidence interval for Recall@K is:

```
For binary metric (0/1 per query), 95% CI width ≈ 2 × 1.96 × sqrt(p × (1-p) / n)

At p=0.80 (expected Recall@1 for fine-tuned model), n=1667:
  CI width ≈ 2 × 1.96 × sqrt(0.80 × 0.20 / 1667)
           ≈ 2 × 1.96 × 0.0098
           ≈ ±1.9%

→ If two models differ by > 4pp in Recall@1, the difference is statistically significant.
→ If the difference is < 2pp, we cannot conclude one is better.
```

**Rule of thumb:** Report differences as "practically significant" only if:
- Absolute delta in Recall@K > 3pp, AND
- The direction is consistent across at least 3 of 6 buckets

### Statistical Test (Optional)

For borderline differences, use a paired McNemar test:

```python
from scipy.stats import chi2_contingency

def mcnemar_test(model_a_correct: list[bool], model_b_correct: list[bool]) -> float:
    """
    McNemar's test for paired binary outcomes.
    Tests H0: P(A correct, B wrong) == P(A wrong, B correct)
    Returns p-value.
    """
    both_correct = sum(a and b for a, b in zip(model_a_correct, model_b_correct))
    a_only = sum(a and not b for a, b in zip(model_a_correct, model_b_correct))
    b_only = sum(not a and b for a, b in zip(model_a_correct, model_b_correct))
    neither = sum(not a and not b for a, b in zip(model_a_correct, model_b_correct))

    table = [[both_correct, a_only], [b_only, neither]]
    _, p_value, _, _ = chi2_contingency(table, correction=True)
    return p_value
```

**When to use:** When two models have Recall@1 within 5pp of each other on any bucket, run McNemar's test. p < 0.05 confirms significant difference.

**Note:** With n=1667 per bucket, we have sufficient power to detect 3pp differences. At n=10000, we could detect ~1pp differences at 95% confidence — relevant for production decisions but overkill for this research POC.

### Minimum N Recommendation

| Goal | Minimum n per bucket |
|------|---------------------|
| Research comparison (±3pp) | 1,000 |
| Production decision (±1pp) | 10,000 |
| System monitoring (±0.3pp) | 100,000 |

Our eval at 1667 per bucket is sufficient for research-quality comparisons. For production deployment decisions, collect a larger eval set from real production data.
