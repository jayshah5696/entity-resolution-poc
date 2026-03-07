# Research Design: Entity Resolution with Fine-tuned Matryoshka Embeddings

**Project:** entity-resolution-poc
**Author:** jayshah5696
**Status:** Active Research

---

## 1. Motivation and Problem Statement

### The Entity Resolution Problem at Scale

Entity resolution (ER) is the task of determining whether two or more records refer to the same real-world entity. In the people-data domain — think B2B contact databases, CRM deduplication, identity graphs, and people-search APIs — this reduces to: *given a query record describing a person, find all matching records in a corpus of potentially hundreds of millions of records.*

This is fundamentally different from traditional information retrieval. In document retrieval:
- Queries are natural language (semantic intent matters)
- Documents are long (BM25's length normalization does useful work)
- There are many relevant documents per query (gradations of relevance)

In structured people-entity resolution:
- **Queries are structured records** — 5 typed fields, total token count typically 10-40 tokens
- **Indexing targets are also structured records** — same schema, same brevity
- **Exactly one true match exists** in the corpus per query (or zero — no match)
- **Data corruption is systematic and learnable** — abbreviations, typos, field drops follow known distributions
- **Scale is extreme** — 500M records is a real B2B data platform scale

### Why BM25 Fails Here

BM25 works by computing term-frequency × inverse-document-frequency scores. For entity resolution, this breaks down in several ways:

1. **Abbreviations:** "Jon" vs "Jonathan" have zero token overlap. BM25 score = 0 for the name component. Dense embeddings trained on name-understanding data know these are related.

2. **Typos:** "Smyth" vs "Smith" — different tokens. BM25 assigns no credit. An embedding model that has seen Levenshtein-1 corruptions during training can recover this similarity.

3. **Field drops:** If `email` and `company` are missing, BM25 sees a very short document (< 5 tokens) and the IDF weighting becomes noisy. Dense models trained with explicit `[MISSING]` tokens learn that absence is a signal, not noise.

4. **Domain swaps:** "j.smith@gmail.com" vs "j.smith@google.com" — the character overlap is high, but "gmail" and "google" are different tokens. BM25 would score this as a strong match (it has "j.smith" in common), while we want the model to treat this as a soft positive (same person, different email).

5. **Short document problem:** All records are short. BM25's b parameter (length normalization) was designed for documents with varying lengths — applied to 5-field records, it adds noise.

### The Structured-to-Structured Matching Distinction

Most retrieval research optimizes for query-to-document matching. Our task is structured-to-structured. Both query and corpus items have the same schema. The model needs to learn:

- Field semantics: knowing that `first_name: Jon` and `first_name: Jonathan` are highly similar
- Cross-field relationships: that `email: j.smith@google.com` is evidence for `company: Google`
- Corruption robustness: that missing a field should reduce confidence but not destroy recall
- Hard negative discrimination: "Jane Smith | Google" is NOT the same person as "John Smith | Google", even though 3 of 5 fields are similar

### Scale Target: 500M Records

Production B2B databases like ZoomInfo, Salesforce Data, Apollo, or Clearbit operate at 500M+ records. The retrieval system must:

- Fit index in memory (or on fast NVMe) on commodity server hardware
- Return results in < 100ms for p99 (interactive people search)
- Scale horizontally without model changes

This constraint rules out naive dense retrieval (FP32 768-dim embeddings at 500M records = 1.5TB RAM — not viable). The architecture must exploit quantization and dimensionality reduction while preserving retrieval quality.

---

## 2. Core Hypothesis

> **H1:** A domain-specific fine-tuned embedding model using Matryoshka Representation Learning and Multiple Negatives Ranking Loss will achieve higher Recall@1 and Recall@10 than BM25 on corrupted and partial person records, while matching BM25 on pristine records.

> **H2:** Binary quantization of 64-dim MRL sub-embeddings produces an ANN index that, combined with full-precision re-ranking of top-100 candidates, achieves 95%+ of the Recall@10 of full-precision exact search.

> **H3:** Pipe serialization (`Jon Smith | Google | j.smith@google.com | USA`) outperforms key-value serialization (`first_name: Jon last_name: Smith ...`) for fine-tuned models because it's more compact and the model learns field order as a positional signal.

> **H4 (counter-hypothesis):** Key-value serialization outperforms pipe for zero-shot models because explicit field labels provide semantic context the zero-shot model can leverage without fine-tuning.

### What Success Looks Like

| Metric | Condition | Target |
|--------|-----------|--------|
| Recall@1 | pristine | ≥ BM25 |
| Recall@1 | typo_name | > BM25 + 20pp |
| Recall@10 | missing_email_company | > BM25 + 30pp |
| Recall@10 | swapped_attributes | > BM25 + 25pp |
| p99 latency | 1M index, two-stage | < 50ms |
| Index RAM | 500M records | < 32GB |

---

## 3. Two-Stage Architecture

### Architecture Diagram

```
Query: "Jon Smyth | Googl | jsmith@gmail.com | USA"
(corrupted record — 3 corruptions applied)
         │
         ▼ serialize()
"Jon Smyth | Googl | jsmith@gmail.com | USA"
         │
         ▼ embed_stage1()   [64-dim, from MRL]
[v₁, v₂, ..., v₆₄]  →  binarize()  →  [b₁, b₂, ..., b₆₄]  (8 bytes)
         │
         ▼ HNSW ANN search (binary index, efSearch=100)
top-100 candidate IDs: [id₇₇₃₄₂, id₁₁₂₃₄, ..., id₄₄₅₆₇]
         │
         ▼ fetch 100 full embeddings from FP32 store (768-dim)
         │   (307KB fetched — trivial)
         ▼ cosine_similarity(query_768d, candidates_768d)
reranked: [(id₁₁₂₃₄, 0.987), (id₇₇₃₄₂, 0.923), ...]
         │
         ▼ return top-K
True match: id₁₁₂₃₄ ("Jonathan Smith | Google Inc | jonathan.smith@google.com | USA")
FOUND at rank 1 ✓
```

### Stage 1: Binary 64-dim HNSW

- **Index type:** FAISS BinaryHNSW or BinaryIVF
- **Embedding dims:** 64 (first 64 dims of MRL model output)
- **Quantization:** Binary (sign of each float → 1 bit) → 8 bytes per vector
- **Recall target:** ≥ 95% of true matches in top-100 candidates
- **Construction:** efConstruction=200, M=32
- **Search:** efSearch=100

### Stage 2: Full Precision Re-rank

- **Embedding dims:** 768 (full model output)
- **Storage:** FP32, off hot path (fetched per query, not kept in GPU RAM)
- **Metric:** Cosine similarity (dot product on L2-normalized vectors)
- **Input:** 100 candidates from Stage 1
- **Output:** Top-K final results

### Memory Analysis (500M records)

```
Stage 1 Binary Index:
  500M records × 64 dims × 1 bit = 500M × 8 bytes = 4.0 GB
  Plus HNSW graph overhead (~2× raw vectors): ~8 GB total
  → Production-viable on any server

Stage 2 FP32 Store (full embeddings on disk/NVMe):
  500M records × 768 dims × 4 bytes = 1,536 GB
  → Stored on NVMe, only 100 vectors fetched per query
  → Per-query fetch: 100 × 768 × 4 = 307 KB (< 1ms on NVMe)
  → Acceptable for production

Comparison table (Stage 1 index only):
  Format        │ Dims │ Bytes/vec │ 500M total
  ──────────────┼──────┼───────────┼───────────
  FP32          │  768 │   3,072   │  1,536 GB  ← dead
  FP32          │  768 │   3,072   │  1,536 GB  ← dead
  INT8          │  768 │     768   │    384 GB  ← arguable
  FP32          │   64 │     256   │    128 GB  ← heavy
  Binary        │  768 │      96   │     48 GB  ← okay
  Binary        │   64 │       8   │      4 GB  ← production
```

**Decision: Binary 64-dim for Stage 1 ANN, FP32 768-dim on disk for Stage 2 re-rank.**

---

## 4. Dataset Design

### Base Profile Generation

- **Tool:** Faker (Python) with locale-aware generation
- **Scale:** 1.2M unique profiles generated; 1M used for index; 200K for triplet sourcing; 10K for eval
- **Schema:** `first_name, last_name, company, email, country`
- **Country distribution:** Skewed toward USA (60%) reflecting B2B data platform reality
- **Email:** 70% work email (`f.last@company.com` patterns), 30% personal

### Quality Pipeline (7 Steps)

Before any record enters the training or eval pipeline:

1. **Deduplication:** Exact dedupe on email. Near-dedupe on (first_name, last_name, company) using Levenshtein distance < 2.
2. **Email validation:** Well-formed regex check. Domain must be in valid domain list or follow `<company_slug>.com` pattern.
3. **Name validation:** At least 2 characters. No all-numeric names. Common special chars (hyphens in hyphenated last names) allowed.
4. **Company normalization:** Strip legal suffixes (Inc., LLC, Ltd.) to create canonical form for near-duplicate detection.
5. **Country canonicalization:** Map all country inputs to ISO 3166-1 alpha-3 format. Filter to distribution countries.
6. **Cross-field consistency check:** Email domain must loosely match company name (edit distance or common-domain check). Work emails assigned to work-email profiles.
7. **Triplet validity assertion:** For each (anchor, positive, negative) triplet: assert anchor ≠ positive (by ID), assert positive IS the true match, assert negative IS NOT the true match, assert corruption was actually applied to anchor.

### Corruption Types

See detailed specs in `docs/dataset-design.md`. Summary:

| Type | Prob | Fields | Example |
|------|------|--------|---------|
| abbreviation | 0.15 | first_name | Jonathan → Jon |
| truncation | 0.10 | last_name, company | Smith → Smi |
| levenshtein_1 | 0.20 | first_name, last_name | Smith → Smyth |
| levenshtein_2 | 0.10 | first_name, last_name, company | Jonathan → Johnathen |
| field_drop_single | 0.20 | any | email dropped |
| field_drop_double | 0.10 | any two | email + company dropped |
| domain_swap | 0.10 | email | @google.com → @gmail.com |
| company_abbrev | 0.05 | company | Google Inc → Goog |
| case_mutation | 0.05 | first_name, last_name, company | JOHN SMITH |
| nickname | 0.05 | first_name | William → Bill |

### Hard Negative Mining Strategy

**Round 1 (structural):** For each anchor, find profiles sharing the first 4 characters of company name but with different last names. These are "same company, different person" negatives — realistic confusion cases.

**Round 2 (BM25-mined):** Run BM25 retrieval against the full index. Take top-50 BM25 results that are NOT the true match. These are high-overlap records that a lexical model already finds confusing — exactly what we need to push the dense model to discriminate.

---

## 5. Serialization Formats

### Why Serialization Matters

The embedding model receives a single string. How we convert a 5-field struct to a string determines:

- What signals are available to the model
- How missing fields are represented
- How the model handles field-order positional information
- Whether zero-shot models can leverage field labels

### Pipe Format

```
Jonathan Smith | Google Inc | jonathan.smith@google.com | USA
```

Missing field:
```
Jonathan Smith | [MISSING] | jonathan.smith@google.com | USA
```

Two fields missing:
```
[MISSING] | Smith | [MISSING] | USA
```

**Rationale:** Compact. Field order is a positional signal. The model learns that position-1 is always first+last name, position-2 is always company, etc. After fine-tuning, the model can use position to distinguish "John Smith" (a name at position-1) from "Smith John" (a non-standard entry at position-1 — flag as unusual). Explicit `[MISSING]` tokens teach the model that absence is informative.

### Key-Value Format

```
first_name: Jonathan last_name: Smith company: Google Inc email: jonathan.smith@google.com country: USA
```

Missing field:
```
first_name: [MISSING] last_name: Smith company: Google Inc email: jonathan.smith@google.com country: USA
```

**Rationale:** Self-describing. Zero-shot models (no fine-tuning) know what "first_name" means from pre-training on web text. Explicit labels reduce ambiguity for models that haven't seen the pipe format before. The trade-off is verbosity (longer tokens → more compute, closer to seq_len limits) and potential over-reliance on the labels rather than the values.

### Hypothesis on Format

Post fine-tuning: pipe wins (fewer tokens, model learns the structure). Zero-shot: key-value wins (labels provide prior). Testing both will confirm.

---

## 6. Model Roster

### Summary Table

| Model | Params | Max Dims | MRL Native | License | MTEB |
|-------|--------|----------|------------|---------|------|
| BM25 | — | — | — | MIT | — |
| nomic-embed-v1.5 | 137M | 768 | Yes | Apache2 | 62.28 |
| bge-small-en-v1.5 | 33M | 384 | No | MIT | 51.68 |
| bge-base-en-v1.5 | 109M | 768 | No | MIT | 53.25 |
| mxbai-embed-large | 335M | 1024 | Yes | Apache2 | 64.68 |
| all-MiniLM-L6-v2 | 22M | 384 | No | Apache2 | 44.54 |
| gte-modernbert-base | 149M | 768 | Yes | Apache2 | 63.48 |
| arctic-embed-m-v1.5 | 109M | 768 | Yes | Apache2 | 57.24 |

See `docs/models.md` for architecture-level deep dives.

### Primary Target

**nomic-embed-text-v1.5** is the primary fine-tuning target because:
1. Native MRL — one model for all dimensionalities
2. 8192 token context — no truncation issues
3. Strong MTEB baseline to start from
4. Apache2 license — no restrictions on use
5. Active community + good sentence-transformers integration

---

## 7. Ablation Plan

### Tier 1 Ablations (Must Run — Core Hypotheses)

| Ablation | Variable | Fixed | Answers |
|----------|----------|-------|---------|
| T1-A | BM25 vs zero-shot embed | nomic pipe | Do embeddings need fine-tuning? |
| T1-B | Zero-shot vs fine-tuned | nomic pipe | How much does domain fine-tuning help? |
| T1-C | Pipe vs KV serialization | nomic finetuned | Which format is better? |
| T1-D | 768 vs 64 dims | nomic finetuned | MRL quality at low dims? |
| T1-E | FP32 vs binary quantization | nomic 64d | Quantization quality loss? |

### Tier 2 Ablations (Run If Time)

| Ablation | Variable | Fixed | Answers |
|----------|----------|-------|---------|
| T2-A | nomic vs bge-base | finetuned pipe | Architecture matters? |
| T2-B | Curriculum vs flat hard neg ratio | nomic finetuned | Training stability? |
| T2-C | Two-stage vs single-stage | nomic binary 64d | Does re-rank help? |
| T2-D | HNSW vs flat exact search | nomic finetuned 768d | ANN quality loss? |

### Tier 3 Ablations (If Exceptional Time)

| Ablation | Variable | Fixed | Answers |
|----------|----------|-------|---------|
| T3-A | Hard neg weight 1x vs 5x vs 10x | nomic pipe | Optimal negative weight? |
| T3-B | Batch size 64 vs 128 vs 256 | nomic pipe | In-batch negative quality? |
| T3-C | mxbai-large vs nomic (no finetune) | zero-shot pipe | Ceiling without FT? |

---

## 8. Evaluation Protocol

### Metrics

All metrics computed per (bucket, top-K, model) combination.

**Recall@K:** Is the true match in the top-K returned results?
```
Recall@K = (1/|Q|) × Σ_q 𝟙[true_match(q) ∈ top_K(q)]
```
*Primary metric* — directly measures retrieval success for entity resolution.

**Precision@K:** Since each query has exactly one true match, Precision@K = Recall@K / K. Useful for understanding result density.

**MRR@K:** Mean Reciprocal Rank — rewards finding the match at rank 1 vs rank 10.
```
MRR@K = (1/|Q|) × Σ_q (1 / rank(true_match(q))) if rank ≤ K, else 0
```

**NDCG@K:** Normalized Discounted Cumulative Gain — standard IR metric.

### 6 Evaluation Buckets

| Bucket | Corruption Applied | Tests |
|--------|-------------------|-------|
| pristine | None | Baseline retrieval on clean data |
| missing_firstname | first_name → [MISSING] | Partial record recall (name-only miss) |
| missing_email_company | email + company → [MISSING] | Severe partial record (2 fields gone) |
| typo_name | Levenshtein-1/2 on first or last name | Typo robustness |
| domain_mismatch | Email domain swapped | Email domain confusion |
| swapped_attributes | Two field values swapped | Schema confusion (rare but real) |

### Index Construction

FAISS HNSW (exact settings):
- M=32, efConstruction=200 for build
- efSearch=100 at query time
- Metric: inner product (vectors are L2-normalized → equivalent to cosine)

### Latency Measurement

1. Build index with full n_index records
2. Serialize and embed all 1000 latency test queries
3. Run 100 warmup queries (discarded)
4. Run 1000 measured queries
5. Record per-query wall-clock time (Python `time.perf_counter()`)
6. Report p50, p95, p99 in milliseconds
7. Repeat 3× runs and take median of medians

---

## 9. Weekend Execution Timeline

### Saturday AM (4h): Data + BM25 Baseline

- 09:00 — Run `generate_profiles.py` (10 min, ~1.2M records)
- 09:15 — Run `build_triplets.py` (20 min, round1 + round2)
- 09:40 — Run `build_eval.py` (5 min)
- 09:50 — Run BM25 baseline eval (exp 001) — 30 min
- 10:30 — Run zero-shot nomic eval (exp 002) — 45 min (embedding 1M records)
- 11:30 — **Checkpoint: compare BM25 vs zero-shot, set expectations**

### Saturday PM (3h): Fine-tuning Kickoff

- 12:00 — Start nomic pipe fine-tuning (exp 003, ~4-6h on MPS)
- 12:00 — In parallel: write up experiment notes for 001 and 002
- 13:00 — Monitor training via wandb; adjust if loss is diverging
- 15:00 — **Mid-training checkpoint:** check eval metrics at step 1000-1500

### Saturday Evening (2h): Eval + KV Fine-tuning

- 17:00 — Fine-tune should be done; run exp 003 eval
- 18:00 — Start nomic KV fine-tuning (exp 004)
- 18:30 — Fill in experiment notes; update README experiment log
- 20:00 — **Checkpoint: pipe vs KV training comparison**

### Sunday AM (4h): Remaining Experiments

- 08:00 — KV fine-tuning should be done; run exp 004 eval
- 08:30 — Run binary two-stage eval (exp 006) — uses exp 003 model
- 09:30 — Run dimensionality ablation (exp 007) — test all dims for nomic finetuned
- 11:00 — **Checkpoint: all primary experiments done**

### Sunday PM (3h): Analysis + Write-up

- 12:00 — Open `notebooks/results_viz.ipynb`, generate all plots
- 13:00 — Write final experiment notes for all 7 experiments
- 14:00 — Update README experiment log with final results
- 15:00 — **Monday story ready**

---

## 10. Expected Outcomes and Monday Story

### Expected Results

Based on prior work on structured-data embedding and MRL:

| Model | pristine R@1 | typo_name R@1 | missing_email_company R@10 |
|-------|-------------|----------------|---------------------------|
| BM25 | ~0.85 | ~0.35 | ~0.15 |
| Nomic zero-shot | ~0.70 | ~0.55 | ~0.40 |
| Nomic finetuned | ~0.87 | ~0.80+ | ~0.75+ |
| Two-stage binary | ~0.84 | ~0.77+ | ~0.72+ |

### The Monday Story

The punchline for stakeholders is:

1. **BM25 is the current standard, but it's brittle.** On clean data it works. On real-world corrupted data (which is most of production), it falls apart. Recall@1 on typos drops to ~35%.

2. **Zero-shot embeddings are better, but not enough.** Without domain fine-tuning, they understand semantic similarity but not the specific corruption patterns in our data. ~55% on typos — better, not great.

3. **Fine-tuned embeddings with domain data are the answer.** Training on our specific corruption distribution with hard negatives from our corpus pushes typo Recall@1 to 80%+. The model has learned what "Jon" → "Jonathan" means in the context of our 5-field schema.

4. **And it fits in production.** Binary 64-dim index = 4GB for 500M records. Two-stage architecture preserves 95%+ of quality at 32× memory reduction. p99 latency < 50ms on commodity hardware.

5. **One model, one deployment.** MRL means we don't need a separate small model for Stage 1 — the first 64 dims of the same fine-tuned model serve both stages.

**The ask on Monday:** Let us run this on a 5M sample of production data to validate the corruption distribution assumptions, then scale to full production index.
