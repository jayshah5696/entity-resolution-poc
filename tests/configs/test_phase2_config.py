"""
Tests for Phase 2 cross-encoder config files.

Validates:
  - crossencoder_models.yaml loads and has correct structure
  - crossencoder.yaml loads and has valid training parameters
  - Pydantic config models validate correctly
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CE_MODELS_PATH = REPO_ROOT / "configs" / "crossencoder_models.yaml"
CE_CONFIG_PATH = REPO_ROOT / "configs" / "crossencoder.yaml"


# ── crossencoder_models.yaml ──────────────────────────────────────────────────


class TestCrossEncoderModelsYAML:
    """Validate the locked CE model registry."""

    @pytest.fixture(autouse=True)
    def load_yaml(self):
        assert CE_MODELS_PATH.exists(), f"Missing: {CE_MODELS_PATH}"
        with open(CE_MODELS_PATH) as f:
            self.cfg = yaml.safe_load(f)

    def test_models_key_exists(self):
        assert "models" in self.cfg

    def test_exactly_four_models(self):
        assert len(self.cfg["models"]) == 4

    def test_expected_model_keys(self):
        expected = {"minilm_reranker", "gte_reranker", "bge_reranker_m3", "granite_reranker"}
        assert set(self.cfg["models"].keys()) == expected

    def test_exactly_two_finetune_targets(self):
        assert "finetune_targets" in self.cfg
        assert len(self.cfg["finetune_targets"]) == 2
        assert set(self.cfg["finetune_targets"]) == {"gte_reranker", "granite_reranker"}

    def test_zero_shot_references(self):
        assert "zero_shot_references" in self.cfg
        assert set(self.cfg["zero_shot_references"]) == {"minilm_reranker", "bge_reranker_m3"}

    @pytest.mark.parametrize(
        "model_key",
        ["minilm_reranker", "gte_reranker", "bge_reranker_m3", "granite_reranker"],
    )
    def test_required_fields_per_model(self, model_key):
        model = self.cfg["models"][model_key]
        required = ["hf_id", "params_m", "license", "context_tokens", "finetune", "note"]
        for field in required:
            assert field in model, f"Model '{model_key}' missing field '{field}'"

    @pytest.mark.parametrize(
        "model_key",
        ["minilm_reranker", "gte_reranker", "bge_reranker_m3", "granite_reranker"],
    )
    def test_license_is_apache(self, model_key):
        model = self.cfg["models"][model_key]
        assert model["license"] == "apache-2.0", f"Model '{model_key}' has non-Apache license"

    def test_finetune_targets_are_marked_true(self):
        for key in self.cfg["finetune_targets"]:
            assert self.cfg["models"][key]["finetune"] is True

    def test_zero_shot_refs_are_marked_false(self):
        for key in self.cfg["zero_shot_references"]:
            assert self.cfg["models"][key]["finetune"] is False

    def test_hf_model_prefix(self):
        assert self.cfg["hf_model_prefix"] == "jayshah5696/er-ce"

    def test_hf_dataset(self):
        assert self.cfg["hf_dataset"] == "jayshah5696/entity-resolution-ce-pairs"

    def test_params_within_budget(self):
        for key, model in self.cfg["models"].items():
            assert model["params_m"] <= 600, f"{key}: {model['params_m']}M exceeds budget"

    def test_finetune_targets_under_200m(self):
        for key in self.cfg["finetune_targets"]:
            params = self.cfg["models"][key]["params_m"]
            assert params <= 200, f"Fine-tune target {key}: {params}M exceeds 200M limit"


# ── crossencoder.yaml ─────────────────────────────────────────────────────────


class TestCrossEncoderTrainingYAML:
    """Validate the CE training hyperparameters."""

    @pytest.fixture(autouse=True)
    def load_yaml(self):
        assert CE_CONFIG_PATH.exists(), f"Missing: {CE_CONFIG_PATH}"
        with open(CE_CONFIG_PATH) as f:
            self.cfg = yaml.safe_load(f)

    def test_top_level_sections(self):
        required = [
            "data", "training", "loss", "hard_negative_curriculum",
            "eval", "modal", "wandb", "hf_hub",
        ]
        for section in required:
            assert section in self.cfg, f"Missing top-level section: {section}"

    def test_training_epochs(self):
        assert self.cfg["training"]["epochs"] == 5

    def test_training_batch_size(self):
        assert self.cfg["training"]["batch_size"] == 64

    def test_warmup_steps_not_ratio(self):
        """warmup_ratio is deprecated in ST 3.4+; must use warmup_steps."""
        assert "warmup_steps" in self.cfg["training"]
        assert "warmup_ratio" not in self.cfg["training"]

    def test_bf16_for_modal(self):
        assert self.cfg["training"]["bf16"] is True

    def test_learning_rate(self):
        assert self.cfg["training"]["lr"] == 2e-5

    def test_max_length(self):
        assert self.cfg["training"]["max_length"] == 512

    def test_loss_has_two_phases(self):
        assert self.cfg["loss"]["phase1_loss"] == "bce"
        assert self.cfg["loss"]["phase2_loss"] == "lambda_rank"
        total = self.cfg["loss"]["phase1_epochs"] + self.cfg["loss"]["phase2_epochs"]
        assert total == 5

    def test_curriculum_has_5_epochs(self):
        curriculum = self.cfg["hard_negative_curriculum"]
        assert len(curriculum) == 5
        for i in range(1, 6):
            key = f"epoch_{i}"
            assert key in curriculum, f"Missing curriculum for {key}"
            entry = curriculum[key]
            assert abs(entry["hard_pct"] + entry["random_pct"] - 1.0) < 1e-6

    def test_curriculum_hard_pct_increases(self):
        curriculum = self.cfg["hard_negative_curriculum"]
        hard_pcts = [curriculum[f"epoch_{i}"]["hard_pct"] for i in range(1, 6)]
        for i in range(1, len(hard_pcts)):
            assert hard_pcts[i] >= hard_pcts[i - 1], f"Hard% decreased at epoch {i + 1}"

    def test_eval_metric_for_best(self):
        assert self.cfg["eval"]["metric_for_best_model"] == "ndcg_at_10"
        assert self.cfg["eval"]["greater_is_better"] is True

    def test_modal_gpu(self):
        assert self.cfg["modal"]["gpu"] == "A10G"

    def test_modal_volume_separate_from_phase1(self):
        assert self.cfg["modal"]["volume"] == "entity-resolution-ce-checkpoints"

    def test_serialization_colval(self):
        assert self.cfg["data"]["serialization"] == "colval"

    def test_data_paths(self):
        assert self.cfg["data"]["train"] == "data/phase2/ce_train.parquet"
        assert self.cfg["data"]["val"] == "data/phase2/ce_val.parquet"
        assert self.cfg["data"]["test"] == "data/phase2/ce_test.parquet"

    def test_hard_negative_filter(self):
        assert self.cfg["hard_negative_filter"]["min_margin"] == 3.0

    def test_threshold_calibration_method(self):
        assert self.cfg["threshold_calibration"]["method"] == "f1_maximize"


# ── Pydantic config models ───────────────────────────────────────────────────


class TestPydanticConfigModels:
    """Validate that the Pydantic config models load correctly."""

    def test_ce_models_config_loads(self):
        from src.models.ce_config import load_ce_models_config

        cfg = load_ce_models_config()
        assert len(cfg.models) == 4
        assert len(cfg.finetune_targets) == 2

    def test_ce_training_config_loads(self):
        from src.models.ce_config import load_ce_training_config

        cfg = load_ce_training_config()
        assert cfg.training.epochs == 5
        assert cfg.training.batch_size == 64

    def test_resolve_finetune_model(self):
        from src.models.ce_config import load_ce_models_config

        cfg = load_ce_models_config()
        gte = cfg.models["gte_reranker"]
        assert gte.hf_id == "Alibaba-NLP/gte-reranker-modernbert-base"
        assert gte.finetune is True

    def test_resolve_zero_shot_model(self):
        from src.models.ce_config import load_ce_models_config

        cfg = load_ce_models_config()
        minilm = cfg.models["minilm_reranker"]
        assert minilm.finetune is False
