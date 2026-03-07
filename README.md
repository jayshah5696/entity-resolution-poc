# entity-resolution-poc

Research project testing whether fine-tuned Matryoshka embeddings can beat BM25 for structured people-record matching at 500M scale.

---

## Problem

The current production system uses BM25 for people-entity resolution across ~500M contact records. It works fine on clean data. On real-world queries -- abbreviated names, typos, missing fields, swapped email domains -- recall drops sharply. "Jon" vs "Jonathan", "Smyth" vs "Smith", a gmail address instead of a work one: BM25 token overlap fails all three.

The question: can a fine-tuned dense retriever trained on our specific corruption distribution fix this, while still fitting in production memory?

---

## Approach

Pure embedding retrieval vs BM25, no hybrid. Five models evaluated: one lexical baseline and four embedding models ranging from 22M to 600M parameters.

Fine-tuning uses Matryoshka Representation Learning (MRL) + Multiple Negatives Ranking Loss (MNRL) on synthetic triplets built from ~1.2M generated profiles. Corruptions applied to anchors mirror what we see in production: abbreviations, Levenshtein-1 typos, field drops, domain swaps, nickname substitutions.

Two-stage retrieval at inference: binary 64-dim HNSW for ANN (4GB for 500M records), then full FP32 768-dim re-rank on top-100 candidates.

---

## Locked Models

Five models, no additions without team discussion.

| # | Model | Params | Dims | MRL | License | Role |
|---|-------|--------|------|-----|---------|------|
| 1 | BM25 (rank_bm25) | -- | -- | -- | Apache | Lexical baseline |
| 2 | all-MiniLM-L6-v2 | 22M | 384 | No | Apache | Absolute floor |
| 3 | bge-small-en-v1.5 | 33M | 384 | via FT | MIT | Efficiency story |
| 4 | gte-modernbert-base | 149M | 768 | Yes | Apache | Primary result |
| 5 | nomic-embed-text-v1.5 | 137M | 768 | Yes | Apache | MRL reference |
| 6 | pplx-embed-v1-0.6b | 600M | 1536 | Yes | Apache | SOTA ceiling (zero-shot only) |

Fine-tune targets: bge-small, gte-modernbert-base, nomic-v1.5. MiniLM and pplx are zero-shot only.

Note: nomic requires `search_query:` / `search_document:` prefixes at inference. pplx uses separate system prompts for query vs doc (decoder-only, EOS token pooling). See `docs/model-lock.md`.

---

## Repo Layout

```
configs/        model registry, dataset config, finetune hyperparams, eval settings
docs/           research design, dataset spec, evaluation protocol, model notes
src/
  data/         profile generation, corruption engine, serialization, triplet building
  models/       embedding wrappers and BM25
  eval/         retrieval evaluation harness
  utils/        nicknames, config loading
experiments/    per-experiment config + notes (001 through 007)
data/           raw profiles, processed records, triplets, eval sets (parquet)
models/         fine-tuned checkpoints
results/        JSON per experiment + master CSV
notebooks/      results_viz.ipynb
tests/          pytest suite for data pipeline
```

---

## How to Run

**Prerequisites:** Python 3.11+, ~50GB disk.

```bash
git clone https://github.com/jayshah5696/entity-resolution-poc
cd entity-resolution-poc
pip install -r requirements.txt  # or: pip install faker polars pyarrow python-levenshtein rank-bm25 rich pyyaml typer pytest
```

Scripts use `src.*` imports so always run from the repo root with `PYTHONPATH=.`:

```bash
# 1. Generate 1.2M base profiles and split into index/triplets/eval (~20-30 min on M3 Pro)
python src/data/generate.py \
    --config configs/dataset.yaml \
    --output-dir data/

# 2. Build training triplets from the 200K triplet_source split (~10-15 min)
PYTHONPATH=. python src/data/triplets.py \
    --config configs/dataset.yaml \
    --profiles data/processed/triplet_source.parquet \
    --output-dir data/triplets/

# 3. Build eval query set -- 6 buckets x 10K = 60K queries (~2-3 min)
PYTHONPATH=. python src/data/eval_set.py \
    --config configs/dataset.yaml \
    --eval-profiles data/eval/eval_profiles.parquet \
    --output-dir data/eval/
```

BM25 baseline + eval harness, fine-tuning, and results aggregation scripts are next (in progress).

Tests:

```bash
PYTHONPATH=. pytest tests/ -v
```

---

## Experiment Log

| ID  | Name                                | Status  | Key Result |
|-----|-------------------------------------|---------|------------|
| 001 | BM25 baseline (pipe, 1M index)      | pending |            |
| 002 | Nomic v1.5 zero-shot (pipe)         | pending |            |
| 003 | Nomic v1.5 fine-tuned (pipe)        | pending |            |
| 004 | Nomic v1.5 fine-tuned (KV)          | pending |            |
| 005 | BGE-small fine-tuned (pipe)         | pending |            |
| 006 | Binary two-stage (64D ANN + 768D rerank) | pending |       |
| 007 | Dimensionality ablation (all dims)  | pending |            |

Update this table after each experiment. Key Result = Recall@1 on pristine bucket vs BM25 delta (e.g. "+12.3pp R@1 pristine, +31.2pp R@1 typo_name").

After each experiment:
1. Fill in `experiments/00N_name/notes.md`
2. Update this table
3. Commit: `exp(00N): <one-line result>`

---

## Notes

### Serialization Formats

Two formats under test for how records are converted to strings before embedding:

**Pipe** -- compact, position carries meaning:
```
Jonathan Smith | Google Inc | jonathan.smith@google.com | USA
```
Missing field (empty slot preserved):
```
 | Smith | Google Inc | jonathan.smith@google.com | USA
```

**KV** -- self-describing, better for zero-shot:
```
fn:Jonathan ln:Smith org:Google Inc em:jonathan.smith@google.com co:USA
```
Missing field:
```
fn: ln:Smith org:Google Inc em:jonathan.smith@google.com co:USA
```

Hypothesis: pipe wins after fine-tuning (fewer tokens, model learns structure). KV wins zero-shot (labels give the model prior knowledge). Testing both confirms.

See `src/data/serialize.py` for implementation and `tests/test_serialize.py` for round-trip tests.

### Eval Buckets

Six corruption categories, each 10K queries:

| Bucket | What changes | Tests |
|--------|--------------|-------|
| pristine | nothing | baseline recall on clean data |
| missing_firstname | first_name dropped | partial record -- name-only miss |
| missing_email_company | email + company dropped | severe partial (2 fields gone) |
| typo_name | Levenshtein-1/2 on first or last name | typo robustness |
| domain_mismatch | email domain swapped to personal domain | gmail vs work email |
| swapped_attributes | first_name and last_name swapped | schema confusion |

See `src/data/corrupt.py` and `tests/test_corrupt.py`.
