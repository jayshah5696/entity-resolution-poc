# Entity Resolution POC — Complete Understanding

> **For the next developer:** This document is the single source of truth for everything about this project. Read it fully before touching a single file.

---

## 1. WHY THIS EXISTS

### The Problem

6sense has ~500 million B2B contact records. When a user searches for a person, the production system uses **BM25 (lexical retrieval)**. BM25 works fine on clean, exact data. It falls apart on:

- Abbreviated names: "Jon" vs "Jonathan", "Mike" vs "Michael"
- Typos: "Smyth" vs "Smith", "Googel" vs "Google"
- Missing fields: only have email, no name
- Domain swaps: `jay@startup.com` vs `jay@gmail.com` (same person)
- Swapped fields: first/last name transposed in the record

**The hypothesis:** A dense embedding model fine-tuned on this specific corruption distribution will outperform BM25 on messy real-world queries, while still fitting in production memory at 500M scale.

**This POC answers:** Can we prove this hypothesis with synthetic data before committing to production training?

---

## 2. WHAT WE'RE BUILDING

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        OFFLINE (BUILD)                          │
│                                                                  │
│  1.2M synthetic profiles                                        │
│       │                                                          │
│       ├──► 1M index records ──► LanceDB index (per model)       │
│       ├──► 200K triplet source ──► 600K training triplets       │
│       └──► 10K eval queries (6 corruption buckets × ~1667)      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING (Modal)                            │
│                                                                  │
│  triplets.parquet (HF Hub) ──► 5 parallel fine-tune jobs        │
│                                    A10G GPU × 5, ~60 min        │
│                                    MatryoshkaLoss + MNRL         │
│                                    Curriculum hard negatives     │
│                                    ──► pushed to HF Hub          │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                       EVAL (Local M3)                            │
│                                                                  │
│  Build ANN index ──► query each bucket ──► metrics per model    │
│                                           Recall@1, @10, MRR    │
│                                           vs BM25 baseline       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Production Vision (if POC succeeds)

```
Query: "Jon Smith | Google | jon.smith@gmail.com"
         │
         ▼
    Embedding model (64-dim binary)
         │
         ▼
    HNSW ANN search → top 100 candidates   [4GB for 500M records]
         │
         ▼
    Re-rank with full 768-dim FP32         [CPU, fast]
         │
         ▼
    Return top-K matches
```

---

## 3. THE DATA PIPELINE

### Step 1: Profile Generation (`src/data/generate.py`)

Generates 1.2M synthetic B2B contact profiles with realistic distributions:
- Names (first, last) with nickname variants
- Companies (real company names)
- Email addresses (work and personal domains)
- Countries

Split into:
- **1M index records** → what we search against
- **200K triplet source** → what we train from
- **10K eval profiles** → held out for evaluation (~1,667 per corruption bucket)

### Step 2: Triplet Building (`src/data/triplets.py`)

For each profile in the triplet source, creates `(anchor, positive, negative)` triplets:

```
anchor:   corrupted version of profile (what a query looks like)
positive: the clean canonical record (ground truth match)
negative: a different profile (hard or random)
```

Corruptions applied to anchors (10 types in `corrupt.py`):
- **Abbreviation**: Jonathan → Jon
- **Nickname substitution**: William → Bill, Robert → Bob
- **Truncation**: Schmidt → Schm
- **Levenshtein-1/2 typos**: Smith → Smyth
- **Field drops (single/double)**: remove first_name, or remove email+company
- **Domain swaps**: work email → gmail/yahoo
- **Company abbreviation**: Google Inc → Goog
- **Case mutation**: JONATHAN SMITH
- **Field swaps**: first_name ↔ last_name

Output: `data/triplets/triplets.parquet` — 600K rows, ~50MB

Each row has columns for both serialization formats:
- `anchor_text_pipe`, `positive_text_pipe`, `negative_text_pipe`
- `anchor_text_kv`, `positive_text_kv`, `negative_text_kv`
- `negative_source`: "hard" or "random"

### Step 3: Eval Set (`src/data/eval_set.py`)

Creates 6 corruption buckets × ~1,667 = 10K evaluation queries:

| Bucket | Corruption | What it tests |
|--------|-----------|---------------|
| `pristine` | None | Baseline recall on clean data |
| `missing_firstname` | Drop first name | Name-only queries |
| `missing_email_company` | Drop email + company | Severely partial records |
| `typo_name` | Levenshtein-1/2 | Typo robustness |
| `domain_mismatch` | Swap work→personal email | Gmail vs work email |
| `swapped_attributes` | Swap first↔last name | Schema confusion |

### Serialization Formats

Two ways to convert a record to a string:

**Pipe** (compact, position-based):
```
Jonathan Smith | Google Inc | jonathan.smith@google.com | USA
```

**KV** (self-describing, better zero-shot):
```
fn:Jonathan ln:Smith org:Google Inc em:jonathan.smith@google.com co:USA
```

**Hypothesis:** Pipe wins post-fine-tuning (fewer tokens, model learns format). KV wins zero-shot. Currently running pipe only.

---

## 4. THE MODELS

### Model Lock (March 2026) — 5 models, no additions

```
┌──────────────────────┬────────┬────────────────────┬─────────────┬──────────────────────────┐
│ Model                │ Params │ Dims (MRL)         │ Role        │ Fine-tune?               │
├──────────────────────┼────────┼────────────────────┼─────────────┼──────────────────────────┤
│ BM25 (rank_bm25)     │ --     │ --                 │ Baseline    │ No (lexical)             │
│ all-MiniLM-L6-v2     │ 22M    │ 384                │ Floor       │ No (zero-shot only)      │
│ bge-small-en-v1.5    │ 33M    │ 384 (→64 via FT)   │ Efficiency  │ YES — adds MRL via FT    │
│ gte-modernbert-base  │ 149M   │ 768 (→64 native)   │ Primary ★   │ YES — MRL native         │
│ nomic-embed-text-v1.5│ 137M   │ 768 (→64 native)   │ MRL ref     │ YES — MRL native         │
│ pplx-embed-v1-0.6b   │ 600M   │ 1536 (→64 native)  │ SOTA ceil   │ No (zero-shot only)      │
└──────────────────────┴────────┴────────────────────┴─────────────┴──────────────────────────┘
```

### Critical Model Notes

**nomic-embed-text-v1.5:**
- MUST prepend `"search_query: "` to ALL queries at inference
- MUST prepend `"search_document: "` to ALL docs at inference
- Dropping this prefix loses 3-5% recall
- HF: `nomic-ai/nomic-embed-text-v1.5`

**pplx-embed-v1-0.6b:**
- Decoder-only architecture (Qwen2.5-0.5B base)
- Uses EOS token pooling, NOT mean pooling
- Separate system prompts for query vs doc at inference
- Released March 2026 — SOTA <1B on MTEB retrieval
- HF: `perplexity-ai/pplx-embed-v1-0.6b`
- During training: system prompts dropped, plain MNRL

**gte-modernbert-base:**
- ModernBERT backbone (RoPE, Flash Attention)
- Native MRL support — preferred primary target
- `trust_remote_code=True` required
- `requires_trust_remote_code` for Alibaba-NLP models

**bge-small:**
- Does NOT have native MRL — added via `MatryoshkaLoss` wrapper during fine-tuning
- MIT license (important for production use)
- The "efficiency story": proves small + domain-adapted > large zero-shot

---

## 5. TRAINING APPROACH

### Loss Function

```
MatryoshkaLoss(
    MultipleNegativesRankingLoss(scale=20.0)
)
```

- **MNRL**: In-batch negatives. With batch=256, each sample gets 255 negatives "for free"
- **MatryoshkaLoss**: Trained simultaneously at dims [768, 512, 256, 128, 64]
- Result: One model usable at any dimensionality tradeoff

### Curriculum Hard Negatives

```
Epoch 1: 10% hard negatives, 90% random
Epoch 2: 30% hard negatives, 70% random
Epoch 3: 50% hard negatives, 50% random
```

Hard negatives are profiles that look similar but are different people (same industry, similar name, etc.). Too many early causes unstable training.

### Key Hyperparameters

| Param | Value | Why |
|-------|-------|-----|
| batch_size | 256 (bge/gte/nomic), 8 (pplx) | MNRL: larger = more in-batch negatives |
| learning_rate | 2e-5 | Standard for embedding fine-tuning |
| epochs | 3 | Sufficient with curriculum |
| warmup_steps | 10% of optimizer steps | Not warmup_ratio (deprecated in ST v3.4+) |
| fp16 | True for A10G, False for pplx (bf16) | A10G supports both; pplx needs bf16 |
| grad_checkpointing | True for pplx only | 600M model needs it to fit A10G |
| grad_accum | 32 for pplx (eff batch=256), 1 for others | Maintains effective batch size |

---

## 6. EVALUATION

### Primary Metrics

- **Recall@1**: Does the correct record appear in the top-1 result?
- **Recall@10**: Does the correct record appear in the top-10?
- **MRR**: Mean Reciprocal Rank

Reported per bucket + aggregate.

### Comparison Structure

```
For each model × each bucket:
    Recall@1, Recall@10, MRR
    vs BM25 baseline delta (±pp)
```

The key question: **Does fine-tuning help in hard buckets (typo, domain_mismatch) without hurting pristine recall?**

---

## 7. INFRASTRUCTURE

### Local Development (M3 Pro, 18GB)

```
Tool    | Use
--------|----------------------------------------------------
uv      | Package management (NOT pip, NOT conda)
Python  | 3.12 (NOT 3.14 -- PyTorch wheels don't exist yet)
LanceDB | ANN index (vector search + FTS)
hf CLI  | Use `hf` not `huggingface-cli` (2026 standard)
```

**MPS (Apple Silicon GPU) notes:**
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` before any import
- `dataloader_num_workers=0` mandatory (fork fails with MPS tensors)
- Batch cap ~32 for 768-dim models (65536 output channel limit)
- Training on M3 is ~15h per model — use Modal instead

### Modal GPU Training

```
modal run src/models/finetune_modal.py::run_all
```

**Image setup (current state — March 2026):**
```python
modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .uv_pip_install("torch==2.6.0",
                    extra_options="--index-url https://download.pytorch.org/whl/cu124")
    .uv_pip_install(FLASH_ATTN_WHEEL)   # v2.7.3, pre-built for torch2.6+cu12+cp311
    .uv_pip_install("sentence-transformers>=5.0,<6",
                    "transformers==4.57.6", ...)
```

**Why this setup:**
- `debian_slim` + `uv_pip_install` = fastest build (~2 min vs 20+ min with flash-attn compile)
- `python_version="3.11"` — flash-attn publishes the widest pre-built wheel coverage for 3.11
- `torch` installed first separately with CUDA index so uv resolves correctly
- flash-attn installed from pre-built GitHub wheel — no compilation, instant install

**Dependency version constraints (verified March 7, 2026):**

The PPLX model (`perplexity-ai/pplx-embed-v1-0.6b`) has custom code downloaded from HF Hub at runtime (`trust_remote_code=True`). These files impose hard constraints on the entire dependency stack:

| PPLX import | Source file | First available in |
|---|---|---|
| `from transformers import Qwen3Model` | `modeling.py` | transformers **4.51.0** |
| `from transformers.masking_utils import create_causal_mask` | `modeling.py` | transformers **4.53.0** |
| `from transformers.utils import TransformersKwargs` | `modeling.py` | transformers **4.54.0** |
| `create_causal_mask(or_mask_function=...)` | `modeling.py` | torch **≥2.6** |
| `from sentence_transformers.models import Module` | `st_quantize.py` | sentence-transformers **5.0.0** |

**Pinned stack (all verified against source code):**

| Package | Version | Why this version |
|---|---|---|
| Python | 3.11 | flash-attn pre-built wheels available |
| torch | ==2.6.0 (cu124) | PPLX `create_causal_mask(or_mask_function=...)` needs ≥2.6 |
| flash-attn | 2.7.3+cu12torch2.6cxx11abiFALSE | ABI-matched to torch 2.6; `cxx11abiFALSE` matches pip-installed torch |
| sentence-transformers | ≥5.0,<6 → 5.2.3 | PPLX `st_quantize.py` needs `Module` class (added in 5.0) |
| transformers | ==4.57.6 (pinned) | Last stable 4.x; has all PPLX deps; avoids transformers 5.x breaking changes |
| accelerate | ≥1.2.0 | Compatible with all models |

**Why pin transformers to 4.57.6 (not 5.x)?**
- `transformers` 5.0 (released Feb 2026) is a major breaking release — removed deprecations, changed `apply_chat_template` return type, removed legacy config file saving.
- We have 5 models with different architectures. Pinning to last stable 4.x avoids untested breakage.
- PPLX's custom code works with 4.57.6 (all required imports verified).
- `sentence-transformers` 5.2.3 requires `transformers>=4.41,<6` — 4.57.6 satisfies this.

**Why sentence-transformers ≥5.0 (not 3.x or 4.x)?**
- PPLX's `st_quantize.py` does `from sentence_transformers.models import Module`.
- `Module` class does NOT exist in sentence-transformers 3.x or 4.x (verified against source: v3.3.1, v3.4.1, v4.0.0–v4.1.0).
- `Module` was added in sentence-transformers 5.0.0.

**Other 4 models have NO custom code:**
- MiniLM-L6, BGE-small: standard BERT, `trust_remote_code=false`
- GTE-ModernBERT, Nomic v1.5: `trust_remote_code=true` in config but no `.py` files on HF Hub — they use standard transformers model classes
- These models work with any transformers 4.x/5.x and any sentence-transformers version.

**⚠ PPLX is the constraint driver.** If you add a 6th model with `trust_remote_code=true`, check its HF repo for custom `.py` files and verify their imports against the pinned stack.

**Modal secrets required:**
```bash
modal secret create huggingface HF_TOKEN=<token>
modal secret create wandb WANDB_API_KEY=<key>
```

**Checkpoints stored in:** Modal Volume `entity-resolution-checkpoints`
```
/checkpoints/
  bge_small/checkpoint-500/, .../final/
  gte_modernbert_base/checkpoint-500/, .../final/
  nomic_v15/checkpoint-500/, .../final/
  minilm_l6/.../final/
  pplx_embed_v1_06b/.../final/
```

**Models pushed to HF Hub:**
```
jayshah5696/er-bge-small-pipe-ft
jayshah5696/er-gte-modernbert-base-pipe-ft
jayshah5696/er-nomic-v15-pipe-ft
jayshah5696/er-minilm-l6-pipe-ft
jayshah5696/er-pplx-embed-v1-06b-pipe-ft
```

**Training dataset on HF Hub:**
```
jayshah5696/entity-resolution-triplets
```

### W&B Monitoring

Project: `entity-resolution-poc` under `jayshah5696`
URL: https://wandb.ai/jayshah5696/entity-resolution-poc

---

## 8. CURRENT STATE (March 7, 2026)

### What's Done

- [x] 1.2M profile generation pipeline
- [x] 600K triplet building with curriculum hard negatives
- [x] 60K eval set (6 corruption buckets)
- [x] BM25 eval harness (`run_bm25.py`)
- [x] Dense eval harness (`run_eval.py`) with LanceDB ANN
- [x] `finetune.py` — local training script (MPS-compatible)
- [x] `finetune_modal.py` — parallel Modal training for all 5 models
- [x] Training data uploaded to HF Hub
- [x] Modal secrets configured
- [x] W&B project configured

### What's In Progress

- [ ] **Modal fine-tuning run** — image build was stuck on flash-attn compilation. Fixed now (switched to debian_slim + uv_pip_install, dropped flash-attn). Next step: `git pull && modal run src/models/finetune_modal.py::run_all`

### What's Pending After Fine-tuning

- [ ] Build LanceDB indexes for all 5 fine-tuned models
- [ ] Run eval across all models × all 6 buckets
- [ ] Aggregate results → `results/master_results.csv`
- [ ] Fill experiment log in README

---

## 9. KEY TECHNICAL DECISIONS & WHY

### Why NOT hybrid retrieval?

Pure embedding vs pure BM25. Testing clean separation to understand the embedding contribution before mixing.

### Why Matryoshka dims?

Production constraint: 500M records at 768-dim FP32 = ~1.5TB. At 64-dim binary = ~4GB. MRL lets us pick the tradeoff at inference time without retraining.

### Why synthetic data?

Can't use production data (PII). Synthetic profiles with realistic corruption distributions allow controlled experiments.

### Why NOT LoRA for fine-tuning?

Full fine-tuning for a POC. LoRA adds complexity. If compute becomes a constraint in production training, revisit.

### Why `pipe` serialization first?

Fewer tokens = faster training and inference. The model learns positional structure. KV is better zero-shot but that's not our use case post-fine-tuning.

### Why flash-attn via pre-built GitHub wheel?

Modal's pypi mirror serves flash-attn source tarball regardless of wheel URLs. Compiling takes 20+ min per image build. Solution: install directly from the pre-built GitHub release wheel (`flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl`). Instant install, no compilation. The wheel filename encodes the exact ABI contract: torch version, CUDA version, C++ ABI flag, and Python version — all must match or you get silent segfaults or `undefined symbol` errors at import time.

### Why `uv_pip_install` over `pip_install` in Modal?

Modal added `.uv_pip_install()` in SDK v1.1.0. It's 3x faster than pip. Image build goes from ~5 min to ~2 min.

### warmup_ratio vs warmup_steps?

`warmup_ratio` is deprecated in sentence-transformers v3.4+. Must compute `warmup_steps` manually:
```python
warmup_steps = max(1, int((len(dataset) // (batch_size * grad_accum)) * epochs * warmup_ratio))
```

Note: divide by `grad_accum` — this counts optimizer steps, not gradient steps.

---

## 10. KNOWN ISSUES & GOTCHAS

### LanceDB

```python
# ALWAYS add this before .select() or you get deprecation spam:
table.search(vec).limit(k).disable_scoring_autoprojection().select(["entity_id"]).to_pandas()
```

### HuggingFace Upload

```bash
# ALWAYS set this before uploading to HF:
export HF_HUB_DISABLE_XET=1
# Without it, uploads hang with XetHub chunked protocol errors
```

### Modal Curriculum Callback

The old pattern of mutating `trainer.train_dataset` inside `on_epoch_begin` is **broken** — HF Trainer calls `get_train_dataloader()` once before the loop. The fix is a `CurriculumTrainer` subclass that overrides `get_train_dataloader()`.

### Checkpoint Resume Sort

```python
# WRONG (lexicographic — checkpoint-1000 sorts before checkpoint-200):
sorted(dir.glob("checkpoint-*"))[-1]

# CORRECT (numeric):
sorted(dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))[-1]
```

### pplx Memory on A10G

- batch=8, grad_accum=32 (effective 256)
- bf16=True (safer dynamic range than fp16 for decoders)
- gradient_checkpointing=True (mandatory for 600M params)

---

## 11. FILE MAP

```
entity-resolution-poc/
├── UNDERSTANDING.md          ← YOU ARE HERE
├── README.md                 ← Commands for running everything
├── pyproject.toml            ← Python 3.12, deps via uv
│
├── configs/
│   ├── models.yaml           ← LOCKED model registry (6 entries: BM25 + 5 embedding)
│   ├── finetune.yaml         ← Training hyperparameters
│   ├── eval.yaml             ← Evaluation harness config (metrics, quantization, latency)
│   └── dataset.yaml          ← Data generation config
│
├── src/
│   ├── data/
│   │   ├── generate.py       ← Creates 1.2M synthetic profiles
│   │   ├── triplets.py       ← Builds (anchor, positive, negative) triplets
│   │   ├── corrupt.py        ← 10 corruption strategies
│   │   ├── serialize.py      ← pipe and kv format converters
│   │   └── eval_set.py       ← Creates 10K eval queries (6 buckets)
│   │
│   ├── models/
│   │   ├── finetune.py       ← LOCAL training (MPS, slow)
│   │   ├── finetune_modal.py ← MODAL training (GPU, fast) ← primary
│   │   ├── upload_triplets.py← Uploads data to HF Hub
│   │   └── encoder.py        ← Model loading wrappers
│   │
│   └── eval/
│       ├── build_index.py    ← Builds LanceDB ANN indexes
│       ├── run_bm25.py       ← BM25 evaluation
│       ├── run_eval.py       ← Dense model evaluation
│       ├── metrics.py        ← Recall@K, MRR calculations
│       └── aggregate.py      ← Combines results → CSV + report
│
├── tests/
│   ├── test_corrupt.py       ← 58 tests for corruption logic
│   ├── test_serialize.py     ← 35 tests for serialization
│   ├── test_nicknames.py     ← 30 tests for nickname lookup
│   └── test_metrics.py       ← 80 tests for retrieval metrics
│
├── docs/
│   ├── UNDERSTANDING.md      ← YOU ARE HERE
│   ├── dataset-design.md     ← Full data spec (schema, corruptions, triplets)
│   ├── evaluation-protocol.md← Metric definitions, eval buckets, latency methodology
│   ├── decisions.md          ← 5 Architecture Decision Records (ADRs)
│   └── experiment-log.md     ← Per-experiment tracking template
│
└── data/                     ← Generated (not in git)
    ├── processed/
    │   ├── index_profiles.parquet  ← 1M records to search against
    │   └── triplet_source.parquet
    ├── triplets/
    │   └── triplets.parquet  ← 600K training triplets
    └── eval/
        └── eval_{bucket}.parquet × 6  ← Per-bucket eval queries (~1667 each)
```

---

## 12. WHAT THE NEXT DEVELOPER NEEDS TO DO

### If Modal training just finished:

```bash
# 1. Verify models are on HF Hub
# https://huggingface.co/jayshah5696 — should see 5 er-*-pipe-ft repos

# 2. For each fine-tuned model, build LanceDB index
uv run python src/eval/build_index.py \
    --model gte_modernbert_base \
    --hf-model jayshah5696/er-gte-modernbert-base-pipe-ft \
    --serialization pipe \
    --quantization fp32 \
    --index-profiles data/processed/index.parquet \
    --eval-profiles data/eval/eval_profiles.parquet \
    --output-dir results/indexes/gte_modernbert_base_pipe_ft_fp32

# 3. Run eval for each model
uv run python src/eval/run_eval.py \
    --model gte_modernbert_base \
    --index-dir results/indexes/gte_modernbert_base_pipe_ft_fp32 \
    --eval-queries data/eval/eval_queries.parquet \
    --output results/gte_modernbert_base_ft.json \
    --serialization pipe

# 4. Aggregate all results
uv run python src/eval/aggregate.py \
    --results-dir results/ \
    --output-csv results/master_results.csv \
    --output-report results/report.md
```

### If Modal training failed / still needs to run:

```bash
git pull
modal run src/models/finetune_modal.py::run_all
# Monitor: https://wandb.ai/jayshah5696/entity-resolution-poc
```

### If you need to resume a crashed model:

```bash
modal run src/models/finetune_modal.py::finetune_one \
    --model-key gte_modernbert_base \
    --resume
```

---

## 13. ENVIRONMENT SETUP

```bash
# Clone
git clone https://github.com/jayshah5696/entity-resolution-poc
cd entity-resolution-poc

# Install (Python 3.12 required -- NOT 3.14)
uv sync

# Verify
uv run python -c "import torch; print(torch.__version__)"

# For HF uploads
export HF_HUB_DISABLE_XET=1
hf auth login

# For Modal
modal secret create huggingface HF_TOKEN=<your_token>
modal secret create wandb WANDB_API_KEY=<your_key>
modal token list  # verify you're authenticated
```

---

## 14. DEPENDENCIES GOTCHA LIST (March 2026)

| Package | Correct version | Common mistake |
|---------|----------------|----------------|
| Python | 3.12 | 3.14 has no PyTorch wheels |
| sentence-transformers | >=3.3 | v2.x has completely different API |
| transformers | >=4.47 | ModernBERT needs 4.47+ |
| accelerate | >=1.2.0 | Required by HF Trainer |
| HF CLI | use `hf` | `huggingface-cli` is old 2024 name |
| Modal | use `modal.App` | `modal.Stub` is removed |
| LanceDB | `.disable_scoring_autoprojection()` | Missing it = deprecation flood |
| warmup | `warmup_steps` | `warmup_ratio` deprecated in ST 3.4+ |

---

*Last updated: March 7, 2026*
*Project: jayshah5696/entity-resolution-poc*
*W&B: https://wandb.ai/jayshah5696/entity-resolution-poc*
*HF Models: https://huggingface.co/jayshah5696*
