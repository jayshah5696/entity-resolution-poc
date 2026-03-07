# Holistic Project Feedback — entity-resolution-poc

**Review date:** 2026-03-07  
**Reviewer context:** Full codebase audit — every source file, config, doc, test, and experiment directory inspected.

---

## Overall Assessment

**Grade: B+** — This is a well-designed research project with strong fundamentals. The research design, documentation, and data pipeline are notably thorough for a POC. The issues below are all fixable in a few hours and none undermine the core research logic.

### What's Working Well

- **Research design docs** are publication-grade — `research-design.md`, `evaluation-protocol.md`, and `dataset-design.md` are excellent
- **4 YAML configs** externalize all parameters — no magic numbers in scripts
- **Data pipeline** (`corrupt.py`, `serialize.py`, `triplets.py`, `eval_set.py`) is clean, well-typed, and tested
- **Encoder abstraction** (`BaseEncoder` → `SentenceTransformerEncoder` / `BM25Encoder`) is the right pattern
- **5 ADRs** documenting key architecture decisions
- **Rich CLI output** with progress bars and summary tables
- **`.gitignore`** properly keeps data/models out of git while preserving structure via `.gitkeep`

---

## 🔴 Critical Issues (Fix First)

### 1. `compute_metrics` silently skips Recall@1

In `src/eval/metrics.py`, the default `ks` is `[5, 10]` — **not** `[1, 5, 10]`. This means `recall_at_1` and `ndcg_at_1` are never computed unless the caller explicitly passes `ks=[1, 5, 10]`.

```python
# Line 102 — current
if ks is None:
    ks = [5, 10]  # ← missing 1

# Should be
if ks is None:
    ks = [1, 5, 10]
```

The docstring promises `recall_at_1` in the return dict, but the default args don't deliver it. Your eval scripts may be passing `ks` explicitly, but this is a latent bug for any new caller. The `aggregate.py` METRIC_KEYS list includes `recall_at_1`, which will error or produce NaN on results that used the default.

### 2. `.gitignore` globally ignores `*.json` but result JSONs need tracking

Line 29 of `.gitignore`:
```
*.json
```

This blanket rule ignores **all** `.json` files project-wide, including `results/001_bm25_pipe.json`, experiment `config.json` files, and manifests. Lines 103-104 try to un-ignore with `!results/*.json`, but the global `*.json` on line 29 takes precedence at the directory level.

**Fix:** Remove line 29 (`*.json`) and add specific ignores where needed, or restructure to only ignore JSON in certain directories.

> **CAUTION:** Your experiment configs (`experiments/*/config.json`), eval manifests (`data/eval/eval_manifest.json`), and result JSONs may all be silently ignored by git. Run `git status --ignored` to verify.

### 3. Python version mismatch across 3 locations

| File | Version |
|------|---------|
| `.python-version` | `3.12` |
| `pyproject.toml` `requires-python` | `>=3.12,<3.13` |
| `pyproject.toml` `[tool.ruff]` `target-version` | `py311` |
| `pyproject.toml` `[tool.black]` `target-version` | `py311` |
| `pyproject.toml` `[tool.mypy]` `python_version` | `3.11` |

Ruff, Black, and mypy think they're targeting 3.11. This means they won't flag 3.12-only features or catch 3.11-incompatible code. Align everything to `3.12`.

---

## 🟡 Stale Items (Experiment Tracking)

### 4. All experiment logs show `pending` despite completed runs

Two result JSONs exist (`001_bm25_pipe.json`, `002_gte_modernbert_pipe_fp32.json`), but:

| Location | Status | Should Be |
|----------|--------|-----------|
| README experiment table | All 7 `pending` | 001 + at least one embedding = `done` |
| `docs/experiment-log.md` | All `pending` | Same |
| `experiments/001_bm25_baseline/notes.md` | `pending`, all results `—` | Fill in actual results |
| `experiments/001_bm25_baseline/config.json` | `"status": "pending"` | `"done"` |

### 5. README experiment numbering doesn't match actual result files

README says experiment 004 is "Nomic v1.5 fine-tuned (KV)", but the actual result file `002_gte_modernbert_pipe_fp32.json` is a GTE experiment. The experiment IDs in the README don't match the file naming. You appear to have run experiments in a different order than originally planned.

### 6. Experiment notes use stale commands

The commands in `experiments/001_bm25_baseline/notes.md` reference `--experiment` and `--eval-config` flags that don't exist in the actual scripts (which use `--index-dir`, `--eval-queries`, `--serialization`, etc.). The notes were written before the scripts were finalized.

---

## 🟡 Code Quality Issues

### 7. `sys.path.insert` hacks in 4 files

These files manually manipulate `sys.path`:

- `src/eval/build_index.py:50`
- `src/eval/run_bm25.py:42`
- `src/eval/run_eval.py:44`
- `src/models/finetune.py:39`

Since `pyproject.toml` already configures `src` as a package (via `[tool.hatch.build.targets.wheel]`), `uv run` and `uv sync` should put `src` on the path automatically. The `sys.path.insert` is a workaround that should be unnecessary with `uv run python src/eval/build_index.py`. Remove them and verify imports still work via `uv run`.

### 8. Mixed CLI frameworks — typer vs argparse

| Files using **typer** | Files using **argparse** |
|---|---|
| `generate.py`, `triplets.py`, `eval_set.py` | `build_index.py`, `run_bm25.py`, `run_eval.py`, `aggregate.py`, `finetune.py` |

This isn't a bug, but it's an inconsistency. The data pipeline uses typer (with `@app.command()` decorators), while eval/model scripts use raw argparse. Pick one. Typer is already a dependency, so standardizing on it would reduce boilerplate.

### 9. Legacy typing imports (`Dict`, `List`, `Optional`)

Since you're on Python 3.12, you should use builtin generics:

```diff
# nicknames.py
-from typing import Dict, List
-NICKNAMES: Dict[str, List[str]] = {
+NICKNAMES: dict[str, list[str]] = {

# corrupt.py, metrics.py
-from typing import Optional
# Use X | None instead (already done in most of the codebase)
```

Files needing cleanup: `nicknames.py`, `corrupt.py`, `metrics.py`.

Most files already use `X | None` syntax (good). These 3 are the stragglers.

### 10. Duplicated dev dependency group

`pyproject.toml` defines dev deps in **two** places:

```toml
[project.optional-dependencies]   # PEP 621
dev = ["ipykernel>=6.29", "jupyterlab>=4.1", "black>=24.0", "ruff>=0.3", "mypy>=1.9"]

[dependency-groups]                # PEP 735 (uv-native)
dev = ["ipykernel>=6.29", "jupyterlab>=4.1", "black>=24.0", "ruff>=0.3", "mypy>=1.9"]
```

Since you're using `uv`, keep only `[dependency-groups]` and remove `[project.optional-dependencies]` dev group. `uv sync --group dev` uses `[dependency-groups]`.

### 11. `pytest` is in main dependencies, not dev

`pytest>=8.0` is listed in `[project.dependencies]` (line 35). It should be in `[dependency-groups] dev` only — production code should never import pytest.

---

## 🟡 Test Coverage Gaps

| Module | Test File | Coverage |
|--------|-----------|----------|
| `src/data/corrupt.py` | ✅ `test_corrupt.py` | 344 lines, 58 tests |
| `src/data/serialize.py` | ✅ `test_serialize.py` | 247 lines, 35 tests |
| `src/utils/nicknames.py` | ✅ `test_nicknames.py` | 168 lines, 30 tests |
| `src/eval/metrics.py` | ✅ `test_metrics.py` | 401 lines, 80 tests |
| `src/data/generate.py` | ❌ No tests | Profile generation + quality pipeline untested |
| `src/data/triplets.py` | ❌ No tests | Triplet building untested |
| `src/models/encoder.py` | ❌ No tests | Encoder abstraction untested |
| `src/eval/build_index.py` | ❌ No tests | Index building untested |
| `src/eval/run_eval.py` | ❌ No tests | Eval harness untested |
| `src/eval/run_bm25.py` | ❌ No tests | BM25 eval untested |
| `src/eval/aggregate.py` | ❌ No tests | Report aggregation untested |

The tested modules are **extremely well tested** (80 tests for metrics alone!). But the untested modules are the ones most likely to have integration bugs. At minimum, add smoke tests for:
- `generate.py` — generate 100 profiles, verify schema + count
- `encoder.py` — load each model config, verify `dim()` and `model_key()`
- `aggregate.py` — feed a sample JSON, verify CSV output schema

---

## 🟢 Minor / Cosmetic

### 12. `upload_triplets.py` has top-level execution

`src/models/upload_triplets.py` runs code at import time (lines 25-42 are module-level, no `if __name__ == "__main__"` guard). Importing this module triggers a file check and exits. Wrap in `main()`.

### 13. `finetune.yaml` hardcodes `nomic-embed-text-v1.5` as base model

`configs/finetune.yaml` line 7: `base_model: nomic-ai/nomic-embed-text-v1.5`. But the actual fine-tune script accepts `--model` and reads from `models.yaml`. This config file's `base_model` field is misleading — either make it the single source of truth or note that the CLI overrides it.

### 14. README lists 6 models in the table (rows 29-36) but says "Five models"

Line 27 says "Five models, no additions without team discussion" but the table has 6 rows (BM25 + 5 embedding models). Minor wording issue.

---

## Prioritized Fix List

| Priority | Item | Time | Impact |
|----------|------|------|--------|
| P0 | Fix `compute_metrics` default ks to include `1` | 2 min | Prevents missing R@1 in future runs |
| P0 | Fix `.gitignore` JSON rule | 5 min | Ensures experiment configs and results are tracked |
| P1 | Update experiment statuses in README + logs + notes | 20 min | Credibility — shows accurate project state |
| P1 | Align Python versions in pyproject.toml tools | 5 min | Correct linting/type checking behavior |
| P2 | Remove duplicated dev deps + move pytest to dev | 5 min | Clean pyproject.toml |
| P2 | Remove `sys.path.insert` hacks | 10 min | Proper package installation |
| P2 | Modernize typing imports | 5 min | 3.12 best practices |
| P3 | Standardize CLI framework (typer vs argparse) | 30 min | Consistency |
| P3 | Add smoke tests for untested modules | 1-2h | Safety net for ongoing experiments |
| P3 | Wrap `upload_triplets.py` in `main()` guard | 2 min | Import safety |

---

## Recommendations for Ongoing Experiment Phase

1. **After each experiment run:** update `experiments/00N/notes.md`, `experiments/00N/config.json` status, and the README experiment table. This is documented in ADR-005 but not being followed yet.

2. **Consider adding a `Makefile`** with targets like `make generate`, `make eval-bm25`, `make eval-gte`, `make report`. Your README instructions are great, but a Makefile prevents copy-paste errors and documents the exact command sequences.

3. **Pin dependency upper bounds** — `sentence-transformers>=3.3` with no ceiling is risky for reproducibility. Consider `sentence-transformers>=3.3,<4.0` or similar.

4. **The `eval.yaml` references FAISS HNSW parameters** but the actual code uses LanceDB (per ADR-002). The FAISS config section in `eval.yaml` (lines 69-86) is now dead config. Either remove it or add a note that it's for reference if migrating back to FAISS.

5. **`results/indexes/` has 380 children** — that's a lot of index data. Make sure the `.gitignore` is catching all of these (it should be via `*.lance`).
