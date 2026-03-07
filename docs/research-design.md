# Research Design: Entity Resolution with Fine-tuned Matryoshka Embeddings

**Project:** entity-resolution-poc
**Author:** jayshah5696
**Status:** Active

---

## 1. Problem Statement

### What entity resolution actually is here

Entity resolution is: given a query record describing a person, find all matching records in a corpus of potentially hundreds of millions of records. This is not document retrieval.

In document retrieval:
- Queries are natural language
- Documents are long (BM25's length normalization does useful work)
- Many relevant documents exist per query

In structured people-entity resolution:
- Queries are 5-field structured records (10-40 tokens total)
- Corpus items have the same schema
- Exactly one true match exists per query (or zero)
- Corruption is systematic and learnable -- abbreviations, typos, field drops follow known distributions
- Scale is 500M+ records

### Why BM25 fails

BM25 computes TF-IDF scores over tokens. For entity resolution this breaks in five ways:

1. **Abbreviations.** "Jon" vs "Jonathan" -- zero token overlap, BM25 score is 0. An embedding model that has seen this corruption pattern handles it.

2. **Typos.** "Smyth" vs "Smith" -- different tokens, no credit from BM25. An embedding model trained on Levenshtein-1 corruptions recovers this.

3. **Field drops.** Missing email + company leaves a very short document (under 5 tokens). IDF weighting becomes noisy. Dense models trained with explicit empty-field tokens learn that absence is a signal.

4. **Domain swaps.** "j.smith@gmail.com" vs "j.smith@google.com" -- high character overlap, but BM25 treats "gmail" and "google" as unrelated. We want the model to recognize same-person, different-email-domain as a soft positive.

5. **Short document problem.** All records are short. BM25's b parameter was designed for documents with varying lengths. Applying it to 5-field records adds noise.

### Structured-to-structured matching

Most retrieval research optimizes query-to-document. Here both sides have the same schema. The model needs to learn:
- Field semantics: `first_name: Jon` and `first_name: Jonathan` are highly similar
- Cross-field relationships: `email: j.smith@google.com` is evidence for `company: Google`
- Corruption robustness: missing a field reduces confidence but should not destroy recall
- Hard negative discrimination: "Jane Smith | Google" is NOT the same person as "John Smith | Google"

### Scale: 500M records

Production B2B databases (ZoomInfo, Apollo, Clearbit) operate at 500M+ records. The retrieval system must:
- Fit the index in memory on commodity server hardware
- Return results in under 100ms at p99
- Scale horizontally without model changes

FP32 768-dim embeddings at 500M records = 1.5TB RAM. Not viable. The architecture must exploit quantization and MRL.

---

## 2. Hypotheses

**H1:** A fine-tuned embedding model using MRL + MNRL achieves higher Recall@1 and Recall@10 than BM25 on corrupted and partial records, while matching BM25 on pristine records.

**H2:** Binary 64-dim MRL sub-embeddings, combined with full-precision re-ranking of top-100 candidates, achieves 95%+ of the Recall@10 of full-precision exact search.

**H3:** Pipe serialization outperforms KV for fine-tuned models -- more compact, model learns field order as positional signal.

**H4 (counter):** KV serialization outperforms pipe for zero-shot models -- explicit field labels give the model semantic prior without fine-tuning.

### Success targets

| Metric | Condition | Target |
|--------|-----------|--------|
| Recall@1 | pristine | >= BM25 |
| Recall@1 | typo_name | > BM25 + 20pp |
| Recall@10 | missing_email_company | > BM25 + 30pp |
| Recall@10 | swapped_attributes | > BM25 + 25pp |
| p99 latency | 1M index, two-stage | < 50ms |
| Index RAM | 500M records | < 32GB |

---

## 3. Two-Stage Architecture

```
Query: "Jon Smyth | Googl | jsmith@gmail.com | USA"
(3 corruptions applied)
         |
         v serialize()
"Jon Smyth | Googl | jsmith@gmail.com | USA"
         |
         v embed_stage1()  [64-dim, from MRL]
[v1, v2, ..., v64]  ->  binarize()  ->  [b1, b2, ..., b64]  (8 bytes)
         |
         v HNSW ANN search (binary index, efSearch=100)
top-100 candidate IDs
         |
         v fetch 100 full embeddings from FP32 store (768-dim)
         |   (307KB per query -- trivial)
         v cosine_similarity(query_768d, candidates_768d)
reranked: [(id11234, 0.987), (id77342, 0.923), ...]
         |
         v return top-K
True match found at rank 1
```

### Stage 1: Binary 64-dim HNSW

- Index type: FAISS BinaryHNSW or BinaryIVF
- Dims: 64 (first 64 dims of MRL model output)
- Quantization: binary (sign of each float), 8 bytes per vector
- Recall target: >= 95% of true matches in top-100
- Build: efConstruction=200, M=32
- Search: efSearch=100

### Stage 2: Full precision re-rank

- Dims: 768 (full model output)
- Storage: FP32, off hot path (fetched per query)
- Metric: cosine similarity (dot product on L2-normalized vectors)
- Input: 100 candidates from Stage 1
- Output: top-K final results

### Memory at 500M records

```
Stage 1 binary index:
  500M x 64 dims x 1 bit = 500M x 8 bytes = 4.0 GB
  Plus HNSW graph overhead (~2x raw vectors): ~8 GB total

Stage 2 FP32 store (on disk/NVMe):
  500M x 768 dims x 4 bytes = 1,536 GB
  Only 100 vectors fetched per query: 307 KB (< 1ms on NVMe)
```

Format comparison for Stage 1 index only:

| Format | Dims | Bytes/vec | 500M total | Viable |
|--------|------|-----------|------------|--------|
| FP32 | 768 | 3,072 | 1,536 GB | No |
| INT8 | 768 | 768 | 384 GB | Barely |
| FP32 | 64 | 256 | 128 GB | Heavy |
| Binary | 768 | 96 | 48 GB | OK |
| Binary | 64 | 8 | 4 GB | Yes |

Decision: binary 64-dim for Stage 1 ANN, FP32 768-dim on disk for Stage 2 re-rank.

### Why MRL

Matryoshka Representation Learning trains a single model such that the first 64 dimensions are a useful representation on their own, the first 128 are better, and so on up to 768. One fine-tuned model serves both stages -- no separate small model for ANN retrieval.

---

## 4. Dataset Design

### Base profile generation

- Tool: Faker (Python), locale-aware
- Scale: 1.2M unique profiles; 1M for index, 200K for triplet sourcing, 10K for eval
- Schema: first_name, last_name, company, email, country
- Country distribution: 60% USA (reflects B2B reality)
- Email: 70% work email (`f.last@company.com`), 30% personal

### Quality pipeline (7 steps)

Each record passes through before entering training or eval:

1. **Deduplication.** Exact dedupe on email. Near-dedupe on (first_name, last_name, company) using Levenshtein distance < 2.
2. **Email validation.** Well-formed regex. Domain must be in valid domain list or follow `<company_slug>.com` pattern.
3. **Name validation.** At least 2 characters. No all-numeric names.
4. **Company normalization.** Strip legal suffixes (Inc., LLC, Ltd.) to create canonical form for near-duplicate detection.
5. **Country canonicalization.** Map all country inputs to ISO 3166-1 alpha-3. Filter to distribution countries.
6. **Cross-field consistency check.** Email domain must loosely match company name (edit distance or known-domain check).
7. **Triplet validity assertion.** For each (anchor, positive, negative): assert anchor != positive by ID, assert corruption was applied to anchor, assert negative is not the true match.

### Corruption types

| Type | Prob | Fields | Example |
|------|------|--------|---------|
| abbreviation | 0.15 | first_name | Jonathan -> Jon |
| truncation | 0.10 | last_name, company | Smith -> Smi |
| levenshtein_1 | 0.20 | first_name, last_name | Smith -> Smyth |
| levenshtein_2 | 0.10 | first_name, last_name, company | Jonathan -> Johnathen |
| field_drop_single | 0.20 | any | email dropped |
| field_drop_double | 0.10 | any two | email + company dropped |
| domain_swap | 0.10 | email | @google.com -> @gmail.com |
| company_abbrev | 0.05 | company | Google Inc -> Goog |
| case_mutation | 0.05 | first_name, last_name, company | JOHN SMITH |
| nickname | 0.05 | first_name | William -> Bill |

See `docs/dataset-design.md` for full specs and `src/data/corrupt.py` for implementation.

### Hard negative mining

**Round 1 (structural):** For each anchor, find profiles sharing the first 4 characters of company name but with a different last name. "Same company, different person" -- realistic confusion cases.

**Round 2 (BM25-mined):** Run BM25 against the full index, take top-50 non-match results. High-overlap records that a lexical model already finds confusing -- exactly what the dense model needs to discriminate.

---

## 5. Serialization Formats

Records are serialized to a single string before embedding. How this is done determines what signals are available to the model.

### Pipe format

```
Jonathan Smith | Google Inc | jonathan.smith@google.com | USA
```

Missing field (slot preserved):
```
 | Smith | Google Inc | jonathan.smith@google.com | USA
```

Two fields missing:
```
 | Smith |  | jonathan.smith@google.com | USA
```

Field order is a positional signal. After fine-tuning, the model knows position-0 is always name, position-1 is company, etc. Compact -- fewer tokens than KV.

### KV format

```
fn:Jonathan ln:Smith org:Google Inc em:jonathan.smith@google.com co:USA
```

Missing field:
```
fn: ln:Smith org:Google Inc em:jonathan.smith@google.com co:USA
```

Self-describing. Zero-shot models can leverage field labels from web pre-training. More verbose than pipe -- longer token sequences, potential truncation concerns at very long values.

### Expected result

Post fine-tuning: pipe wins (model learns structure, saves tokens). Zero-shot: KV wins (labels reduce ambiguity). The experiments confirm or deny this.

See `src/data/serialize.py` and `tests/test_serialize.py`.

---

## 6. Model Roster

| Model | Params | Dims | MRL | License | MTEB Retrieval | Mode |
|-------|--------|------|-----|---------|----------------|------|
| BM25 | -- | -- | -- | Apache | -- | baseline |
| all-MiniLM-L6-v2 | 22M | 384 | No | Apache | 56.3 | zero-shot |
| bge-small-en-v1.5 | 33M | 384 | via FT | MIT | ~58 | zero-shot + fine-tune |
| gte-modernbert-base | 149M | 768 | Yes | Apache | 66.4 | zero-shot + fine-tune |
| nomic-embed-text-v1.5 | 137M | 768 | Yes | Apache | 62.3 | zero-shot + fine-tune |
| pplx-embed-v1-0.6b | 600M | 1536 | Yes | Apache | 62.41 | zero-shot |

**Primary fine-tune target: gte-modernbert-base.** Best MTEB-per-parameter of any model under 200M as of early 2025. ModernBERT backbone (RoPE, Flash Attention). MRL native. Short structured text is its sweet spot.

**nomic-v1.5** is the MRL reference point. Fine-tuned nomic vs fine-tuned gte-modernbert isolates architecture sensitivity. Important: must prepend `search_query:` to queries and `search_document:` to docs at inference -- dropping these loses 3-5% recall.

**bge-small** (33M, MIT) is the narrative model: proves a tiny MIT-licensed model fine-tuned on entity triplets beats BM25. The production argument -- you don't need large models, you need the right training data.

**pplx-embed-v1-0.6b** (600M) is the SOTA ceiling as of March 2026, #1 MTEB Retrieval among all models under 1B. Zero-shot only on M3 Pro -- too large to fine-tune on this hardware. If fine-tuned bge-small approaches pplx zero-shot, the training-data argument is proven.

**all-MiniLM-L6-v2** is the floor -- shows that off-the-shelf small models fail on entity resolution before any fine-tuning.

See `docs/model-lock.md` for inference quirks per model and quantization dimensions.

---

## 7. Ablation Plan

### Tier 1 -- must run (core hypotheses)

| Ablation | Variable | Fixed | Answers |
|----------|----------|-------|---------|
| T1-A | BM25 vs zero-shot embed | nomic pipe | Do embeddings need fine-tuning at all? |
| T1-B | Zero-shot vs fine-tuned | nomic pipe | How much does domain fine-tuning help? |
| T1-C | Pipe vs KV serialization | nomic fine-tuned | Which format wins? |
| T1-D | 768 vs 64 dims | nomic fine-tuned | MRL quality at low dims? |
| T1-E | FP32 vs binary quantization | nomic 64d | Quantization quality loss? |

### Tier 2 -- run if time

| Ablation | Variable | Fixed | Answers |
|----------|----------|-------|---------|
| T2-A | nomic vs bge-base | fine-tuned pipe | Architecture sensitivity? |
| T2-B | Curriculum vs flat hard neg ratio | nomic fine-tuned | Training stability? |
| T2-C | Two-stage vs single-stage | nomic binary 64d | Does re-rank help? |
| T2-D | HNSW vs flat exact search | nomic fine-tuned 768d | ANN quality loss? |

### Tier 3 -- if exceptional time

| Ablation | Variable | Fixed | Answers |
|----------|----------|-------|---------|
| T3-A | Hard neg weight 1x vs 5x vs 10x | nomic pipe | Optimal negative weight? |
| T3-B | Batch size 64 vs 128 vs 256 | nomic pipe | In-batch negative quality? |
| T3-C | mxbai-large vs nomic (no finetune) | zero-shot pipe | Quality ceiling without fine-tuning? |

---

## 8. Evaluation Protocol

### Metrics

All metrics computed per (bucket, top-K, model).

**Recall@K:** Is the true match in the top-K results?
```
Recall@K = (1/|Q|) x sum_q 1[true_match(q) in top_K(q)]
```
Primary metric -- directly measures retrieval success for entity resolution.

**MRR@K:** Mean Reciprocal Rank. Rewards rank 1 over rank 10.
```
MRR@K = (1/|Q|) x sum_q (1 / rank(true_match(q))) if rank <= K, else 0
```

**NDCG@K:** Normalized Discounted Cumulative Gain. Standard IR metric, included for comparability with published work.

**Precision@K:** Since each query has exactly one true match, Precision@K = Recall@K / K. Useful for understanding result density.

### Six evaluation buckets

| Bucket | Corruption | Tests |
|--------|------------|-------|
| pristine | None | Baseline recall on clean data |
| missing_firstname | first_name dropped | Partial record, name-only drop |
| missing_email_company | email + company dropped | Severe partial (2 fields gone) |
| typo_name | Levenshtein-1/2 on first or last name | Typo robustness |
| domain_mismatch | Email domain swapped to personal domain | Email domain confusion |
| swapped_attributes | first_name and last_name swapped | Schema confusion |

### Index construction

FAISS HNSW settings:
- M=32, efConstruction=200 for build
- efSearch=100 at query time
- Metric: inner product on L2-normalized vectors (equivalent to cosine)

### Latency measurement

1. Build index on full n_index records
2. Serialize + embed all 1000 latency test queries
3. Run 100 warmup queries (discarded)
4. Run 1000 measured queries
5. Record per-query wall-clock time (`time.perf_counter()`)
6. Report p50, p95, p99 in milliseconds
7. Repeat 3 runs, take median of medians

---

## 9. Weekend Execution Timeline

### Saturday AM (4h): Data + BM25 baseline

- 09:00 -- `generate_profiles.py` (10 min, ~1.2M records)
- 09:15 -- `build_triplets.py` (20 min, round 1 + round 2 hard negatives)
- 09:40 -- `build_eval.py` (5 min)
- 09:50 -- BM25 baseline eval (exp 001), 30 min
- 10:30 -- Zero-shot nomic eval (exp 002), 45 min (embedding 1M records)
- 11:30 -- Checkpoint: compare BM25 vs zero-shot, set expectations

### Saturday PM (3h): Fine-tuning kickoff

- 12:00 -- Start nomic pipe fine-tuning (exp 003, ~4-6h on MPS)
- 12:00 -- In parallel: write experiment notes for 001 and 002
- 13:00 -- Monitor training via wandb; adjust if loss diverging
- 15:00 -- Mid-training checkpoint: check eval at steps 1000-1500

### Saturday evening (2h): Eval + KV fine-tuning

- 17:00 -- Fine-tune done; run exp 003 eval
- 18:00 -- Start nomic KV fine-tuning (exp 004)
- 18:30 -- Fill in experiment notes; update README experiment log
- 20:00 -- Checkpoint: pipe vs KV training comparison

### Sunday AM (4h): Remaining experiments

- 08:00 -- KV fine-tuning done; run exp 004 eval
- 08:30 -- Run binary two-stage eval (exp 006) using exp 003 model
- 09:30 -- Run dimensionality ablation (exp 007) for all dims on nomic fine-tuned
- 11:00 -- Checkpoint: all primary experiments done

### Sunday PM (3h): Analysis + write-up

- 12:00 -- Open `notebooks/results_viz.ipynb`, generate all plots
- 13:00 -- Write final experiment notes for all 7 experiments
- 14:00 -- Update README experiment log with final results
- 15:00 -- Monday story ready

---

## 10. Expected Outcomes and Monday Story

### Expected numbers

| Model | pristine R@1 | typo_name R@1 | missing_email_company R@10 |
|-------|-------------|----------------|---------------------------|
| BM25 | ~0.85 | ~0.35 | ~0.15 |
| Nomic zero-shot | ~0.70 | ~0.55 | ~0.40 |
| Nomic fine-tuned | ~0.87 | ~0.80+ | ~0.75+ |
| Two-stage binary | ~0.84 | ~0.77+ | ~0.72+ |

### The Monday story

1. **BM25 is brittle.** On clean data it works. On real-world corrupted queries -- which is most of production -- Recall@1 on typos drops to ~35%. That's the current system.

2. **Zero-shot embeddings are better but not enough.** Without domain fine-tuning they understand semantic similarity but not the specific corruption patterns in our data. ~55% on typos -- improvement, not a fix.

3. **Fine-tuned embeddings on domain data are the answer.** Training on our specific corruption distribution with hard negatives from our corpus pushes typo Recall@1 to 80%+. The model has learned what "Jon" -> "Jonathan" means within our 5-field schema.

4. **It fits in production.** Binary 64-dim index = 4GB for 500M records. Two-stage architecture preserves 95%+ of quality at 32x memory reduction. p99 latency < 50ms on commodity hardware.

5. **One model, one deployment.** MRL means no separate small model for Stage 1 -- the first 64 dims of the same fine-tuned model serve both stages.

**The ask:** Let us run this on a 5M sample of production data to validate the corruption distribution assumptions, then scale to the full index.
