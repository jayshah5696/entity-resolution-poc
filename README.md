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

**Prerequisites:** Python 3.11+, [uv](https://github.com/astral-sh/uv), ~50GB disk.

```bash
git clone https://github.com/jayshah5696/entity-resolution-poc
cd entity-resolution-poc
uv sync
```

`uv sync` installs all deps from pyproject.toml and installs `src` as a package so imports work. Run all scripts with `uv run`:

Steps 1-3 generate data. They must all complete before any eval or fine-tuning step.

```bash
# 1. Generate 1.2M base profiles and split into index / triplets / eval (~20-30 min on M3 Pro)
#    Outputs: data/processed/index.parquet, data/processed/triplet_source.parquet,
#             data/eval/eval_profiles.parquet
uv run python src/data/generate.py --config configs/dataset.yaml --output-dir data/

# 2. Build training triplets from the 200K triplet_source split (~10-15 min)
#    Outputs: data/triplets/triplets.parquet
uv run python src/data/triplets.py \
    --config configs/dataset.yaml \
    --profiles data/processed/triplet_source.parquet \
    --output-dir data/triplets/

# 3. Build eval query set -- 6 buckets x 10K = 60K queries (~2-3 min)
#    Outputs: data/eval/eval_queries.parquet, data/eval/eval_queries_{bucket}.parquet
#    REQUIRED before any eval script (run_bm25.py, run_eval.py)
uv run python src/data/eval_set.py \
    --config configs/dataset.yaml \
    --eval-profiles data/eval/eval_profiles.parquet \
    --output-dir data/eval/
```

```bash
# 4. Build BM25 index (~5-10 min for 1M records)
# --eval-profiles is required: eval entity_ids must exist in the index
uv run python src/eval/build_index.py \
    --model bm25_baseline \
    --serialization pipe \
    --index-profiles data/processed/index.parquet \
    --eval-profiles data/eval/eval_profiles.parquet \
    --output-dir results/indexes/bm25_pipe \
    --models-config configs/models.yaml

# 5. Run BM25 eval (all 6 buckets)
uv run python src/eval/run_bm25.py \
    --index-dir results/indexes/bm25_pipe \
    --eval-queries data/eval/eval_queries.parquet \
    --output results/001_bm25_pipe.json \
    --serialization pipe \
    --experiment-id 001

# 6. Build dense embedding index (~10-30 min per model)
uv run python src/eval/build_index.py \
    --model gte_modernbert_base \
    --serialization pipe \
    --quantization fp32 \
    --index-profiles data/processed/index.parquet \
    --eval-profiles data/eval/eval_profiles.parquet \
    --output-dir results/indexes/gte_modernbert_base_pipe_fp32 \
    --device mps

# 7. Run dense model eval
uv run python src/eval/run_eval.py \
    --model gte_modernbert_base \
    --index-dir results/indexes/gte_modernbert_base_pipe_fp32 \
    --eval-queries data/eval/eval_queries.parquet \
    --output results/004_gte_modernbert_pipe_fp32.json \
    --serialization pipe \
    --experiment-id 004

# 8. Fine-tune all 5 models in parallel on Modal A10G (~60 min, ~$5-7 total)
#
#    One-time setup (do once):
#      uv run python3 -m modal secret create huggingface HF_TOKEN=<your_hf_token>
#      uv run python3 -m modal secret create wandb WANDB_API_KEY=<your_wandb_key>
#
#    One-time data upload to HF Hub (runs locally on M3, no Modal):
#      export HF_HUB_DISABLE_XET=1   # disable XetHub chunked upload -- use standard LFS
#      hf auth login
#      hf upload jayshah5696/entity-resolution-triplets \
#          data/triplets/triplets.parquet triplets.parquet --repo-type dataset
#      # Creates: https://huggingface.co/datasets/jayshah5696/entity-resolution-triplets
#
#    Launch all 5 jobs in parallel:
uv run python3 -m modal run src/models/finetune_modal.py::run_all
#    Monitor: https://wandb.ai/jayshah5696/entity-resolution-poc
#    Models pushed to: https://huggingface.co/jayshah5696
#
#    Single model (debug or re-run one):
#      uv run python3 -m modal run src/models/finetune_modal.py::finetune_one --model-key gte_modernbert_base
#
#    Resume a crashed run (picks up from last checkpoint):
#      uv run python3 -m modal run src/models/finetune_modal.py::finetune_one --model-key gte_modernbert_base --resume
#
#    Local M3 fallback (slow, ~15h per model -- only if Modal unavailable):
#      uv run python src/models/finetune.py \
#          --model gte_modernbert_base \
#          --serialization pipe \
#          --triplets data/triplets/triplets.parquet \
#          --output-dir models/gte_modernbert_base_pipe_ft

# 9. Aggregate all results into CSV + Markdown report
uv run python src/eval/aggregate.py \
    --results-dir results/ \
    --output-csv results/master_results.csv \
    --output-report results/report.md
```

Tests:

```bash
uv run pytest tests/ -v
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
