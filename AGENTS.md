# AGENTS.md — Entity Resolution POC

AI coding assistant instructions for this repository. Read this before touching any file.

---

## Project in One Paragraph

This is a research POC testing whether fine-tuned dense retrieval beats BM25 for B2B entity resolution at 500M record scale. Phase 1 (complete) fine-tuned five bi-encoder embedding models using Matryoshka Representation Learning (MRL) on synthetic B2B contact triplets, then ran a 103-experiment ablation. GTE-ModernBERT (149M) won: R@10=0.975 overall, +10.2pp over BM25 on the hardest corruption bucket. Phase 2 (in progress, branch `phase2`) adds a cross-encoder reranker above the existing retrieval layer.

---

## Current Branch State

- `main` — Phase 1 complete. Tag: `v1.0-phase1`. Do not break anything here.
- `phase2` — Active development. All Phase 2 code goes here.

**Check which branch you are on before editing any file.**

---

## Repo Conventions (Non-Negotiable)

### Language and Runtime
- Python 3.12 locally (`.python-version` file)
- Python 3.11 on Modal (pinned in image build)
- Package manager: `uv` (never `pip install` directly — use `uv add`)
- Types: use type hints everywhere. Pydantic for config models.

### Code Style
- Formatter: `black` (line length 100)
- Linter: `ruff`
- Run before committing: `black src/ tests/ && ruff check src/ tests/`

### CLI Patterns
- Data scripts: `typer` CLI (`python src/data/phase2_pool.py --config configs/... --output-dir data/phase2/`)
- Eval scripts: `argparse` CLI (matching Phase 1 pattern in `run_eval.py`)
- No interactive prompts in scripts — all parameters via CLI args or config files

### Data and I/O
- DataFrames: **Polars** (not pandas). Use `polars.scan_parquet()` for large files.
- File format: parquet for data, YAML for configs, JSON for manifests and results
- Display: **rich** `Console`, `Progress`, `Table` everywhere — no bare print statements for status output
- Configs: always load via YAML; use Pydantic to validate schema

### HuggingFace
- Always set `HF_HUB_DISABLE_XET=1` before any `push_to_hub()` call — without this, uploads hang
- Models: `jayshah5696/er-{model_key}-pipe-ft` (Phase 1), `jayshah5696/er-ce-{model_key}-ft` (Phase 2)
- Dataset: `jayshah5696/entity-resolution-triplets` (Phase 1), `jayshah5696/entity-resolution-ce-pairs` (Phase 2)

### Testing
- TDD always: write the test first, then implement
- All tests in `tests/` with structure mirroring `src/`
- Run: `pytest tests/ -q` — must be zero failures before any commit
- Key invariant tests live in `tests/data/test_split.py` — never skip these

### Experiments
- Every experiment gets a zero-padded 3-digit ID: 008, 009, 010...
- Directory: `experiments/00N_descriptive_name/`
- Each must have: `config.json`, `notes.md`
- Results JSON: `results/00N_descriptive_name.json`
- Commit convention: `exp(00N): {one-line result, e.g., "GTE FT + CE FT: MRR=0.934, +1.7pp vs Stage1"}`

### Commits (General)
- `feat: ` — new feature
- `fix: ` — bug fix
- `chore: ` — tooling, config, structure
- `docs: ` — documentation only
- `exp(00N): ` — experiment result
- `test: ` — tests only

---

## Key Files to Read Before ANY Changes

| File | Why |
|---|---|
| `docs/UNDERSTANDING.md` | Architecture, known issues, pinned stack versions — read this first |
| `docs/decisions.md` | ADR-001 to ADR-008 — why things are the way they are |
| `docs/PHASE2_UNDERSTANDING.md` | Phase 2 equivalent — read before any Phase 2 work |
| `plan.md` | Phase 2 implementation plan — the authoritative spec |
| `configs/crossencoder_models.yaml` | The 4 locked CE models — do not modify without updating lock date |
| `configs/crossencoder.yaml` | Training hyperparameters — touch only with specific justification |

---

## What NOT To Touch (Phase 1 Frozen Files)

These files are **locked** in Phase 2. Do not modify:
```
configs/models.yaml          — Phase 1 model registry
configs/eval.yaml            — Phase 1 eval config
src/data/generate.py         — Phase 1 Faker-based generator
src/data/triplets.py         — Phase 1 triplet generator
src/data/corrupt.py          — Phase 1 corruption functions
src/models/finetune_modal.py — Phase 1 Modal training
src/eval/build_index.py      — Phase 1 index builder
src/eval/run_eval.py         — Phase 1 dense eval harness
src/eval/run_bm25.py         — Phase 1 BM25 eval
experiments/001-007/         — Phase 1 results
```

If you think you need to change these, stop and discuss first.

---

## Running Things

### Local dev (M3 Pro)
```bash
# Activate environment
source .venv/bin/activate  # or: uv sync && uv shell

# Data pipeline (Phase 2)
python src/data/phase2_sources.py --output-dir data/phase2/raw/
python src/data/phase2_pool.py --config configs/crossencoder.yaml --output-dir data/phase2/
python src/data/phase2_corrupt.py --pool data/phase2/base_pool.parquet --output-dir data/phase2/
python src/data/phase2_negatives.py --pool data/phase2/base_pool.parquet --output-dir data/phase2/
python src/data/phase2_boundary.py --pool data/phase2/base_pool.parquet --output-dir data/phase2/
python src/data/phase2_split.py --output-dir data/phase2/

# Upload CE training pairs to HF
HF_HUB_DISABLE_XET=1 python src/models/upload_ce_pairs.py

# Run CE training on Modal (both models in parallel)
modal run src/models/finetune_ce_modal.py::run_all

# Run eval experiment
python src/eval/run_reranker.py \
  --stage1-model gte_modernbert_base \
  --stage1-index experiments/007_ablation_dims/indexes/gte_modernbert_base_pipe_ft_fp32 \
  --reranker gte_reranker \
  --reranker-model-path jayshah5696/er-ce-gte-reranker-ft \
  --eval-queries data/eval/ \
  --output results/012_gte_ft_plus_gte_ce_ft.json \
  --experiment-id 012
```

### Tests
```bash
pytest tests/ -q                          # all tests
pytest tests/data/ -v                     # data pipeline tests
pytest tests/data/test_split.py -v        # critical invariant tests
pytest tests/models/ -v                   # CE model tests
pytest tests/eval/ -v                     # eval pipeline tests
```

### Phase 1 eval (reference only, do not rerun)
```bash
# Don't re-run these — results already in results/*.json
# But if you need to verify Phase 1 infra works:
python src/eval/run_eval.py --model gte_modernbert_base --index-dir experiments/007_ablation_dims/indexes/gte_modernbert_base_pipe_ft_fp32 --eval-queries data/eval/ --output /tmp/verify.json
```

---

## Known Gotchas

### LanceDB
```python
# ALWAYS add this to any table.search() call:
results = table.search(query_vec).limit(k).disable_scoring_autoprojection().to_list()
# Without it: DeprecationWarning flood and incorrect _distance column behavior
```

### HuggingFace uploads
```bash
# ALWAYS set before push:
export HF_HUB_DISABLE_XET=1
# Without it: upload hangs indefinitely on large models
```

### warmup_ratio is deprecated
```python
# WRONG (sentence-transformers >= 3.4):
args = SentenceTransformerTrainingArguments(warmup_ratio=0.1)

# CORRECT:
total_steps = (len(train_dataset) // batch_size) * num_epochs
args = SentenceTransformerTrainingArguments(warmup_steps=int(total_steps * 0.1))
```

### CrossEncoder in sentence-transformers v5
```python
# CrossEncoder is now CrossEncoderModel in ST >= 5.0
# See: https://sbert.net/docs/cross_encoder/usage/usage.html
from sentence_transformers import CrossEncoder  # still works as alias
# But use CrossEncoderTrainer for training, not the old .fit() method
```

### Checkpoint sorting (numeric not lexicographic)
```python
# WRONG — sorts "checkpoint-1000" before "checkpoint-200":
checkpoints = sorted(Path(checkpoint_dir).iterdir())

# CORRECT:
import re
checkpoints = sorted(
    Path(checkpoint_dir).iterdir(),
    key=lambda p: int(re.search(r'\d+', p.name).group())
)
```

### Modal image — pinned stack
```python
# DO NOT change these versions without testing OOM behavior:
# torch: 2.6.0+cu124
# flash-attn: 2.7.3 (pre-built wheel — DO NOT pip install, use the wheel URL)
# sentence-transformers: >=5.0,<6
# transformers: ==4.57.6  ← PINNED, not latest
# Python: 3.11 on Modal (not 3.12)
```

### MPS local development
```python
# For CE inference on M3 (local), batch_size must stay ≤ 32
# For Modal (A10G), batch_size can be 64+ for CE
# Set PYTORCH_ENABLE_MPS_FALLBACK=1 in your shell for any ops not yet MPS-supported
```

---

## Serialization Reference

### Phase 1: Bi-encoder input (pipe format — unchanged)
```
"Jay Smith | Acme Inc | jay.smith@acme.com | VP Engineering | USA"
```

### Phase 2: Cross-encoder input (COL/VAL format — new)
```
"[CLS] COL fn VAL Jay COL ln VAL Smith COL org VAL Acme Inc COL title VAL VP Engineering COL country VAL USA [SEP] COL fn VAL J. COL ln VAL Smith COL org VAL Acme COL title VAL Vice President Engineering COL country VAL USA [SEP]"
```

The record pair is separated by `[SEP]`. Both records use COL/VAL. The colval_serialize() function in `src/data/phase2_corrupt.py` handles this.

---

## Results Schema (ADR-003, Phase 2 Extension)

All results JSONs must follow this schema. Phase 2 files add the `stage2_*` keys:

```json
{
  "experiment_id": "012",
  "model": "gte_modernbert_base",
  "serialization": "pipe",
  "mode": "fine_tuned",
  "quantization": "fp32",
  "dims": 768,
  "stage2_reranker": "gte_reranker",
  "stage2_mode": "fine_tuned",
  "stage2_top_k_input": 50,
  "overall": {
    "recall_at_1": 0.0,
    "recall_at_5": 0.0,
    "recall_at_10": 0.0,
    "mrr_at_10": 0.0,
    "ndcg_at_10": 0.0,
    "f1_at_threshold": 0.0,
    "threshold": 0.0
  },
  "per_bucket": {
    "pristine": {"recall_at_10": 0.0, "mrr_at_10": 0.0, "ndcg_at_10": 0.0},
    "missing_firstname": {},
    "missing_email_company": {},
    "typo_name": {},
    "domain_mismatch": {},
    "swapped_attributes": {}
  },
  "stage1_latency_ms": {"p50": 0.0, "p95": 0.0, "p99": 0.0},
  "stage2_latency_ms": {"p50": 0.0, "p95": 0.0, "p99": 0.0},
  "recall_retention": 0.0,
  "index_size_mb": 0.0,
  "n_records": 1000000,
  "timestamp": "2026-03-15T00:00:00Z"
}
```

---

## Phase 2 Model Keys

| model_key | HF ID | Role |
|---|---|---|
| `minilm_reranker` | cross-encoder/ms-marco-MiniLM-L12-v2 | Latency baseline, zero-shot |
| `gte_reranker` | Alibaba-NLP/gte-reranker-modernbert-base | Primary fine-tune target |
| `bge_reranker_m3` | BAAI/bge-reranker-v2-m3 | Zero-shot ceiling |
| `granite_reranker` | ibm-granite/granite-embedding-reranker-english-r2 | Fine-tune candidate |

Fine-tuned models pushed to:
- `jayshah5696/er-ce-gte-reranker-ft`
- `jayshah5696/er-ce-granite-reranker-ft`

---

## Contacts / Resources

- W&B project: https://wandb.ai/jayshah5696/entity-resolution-poc
- HF org: jayshah5696
- Phase 1 blog: writing/BLOG_POST.md
- Research notes (full): Obsidian vault → Assitant/Research/ER-Phase2-Reranker-Research.md
- Phase 2 plan: plan.md (authoritative)
