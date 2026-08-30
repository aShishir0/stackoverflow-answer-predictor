"""
Loads all trained models and supporting JSON artifacts exactly once at API startup,
and holds them in memory for fast repeated predictions.

Expected files inside MODELS_DIR (default: ../models relative to this file):
  classification_model.txt
  regression_model_lower.txt
  regression_model_median.txt
  regression_model_upper.txt
  model_metadata.json
  regression_metadata.json
  tag_rate_lookup.json
  platform_context.json
"""
from dotenv import load_dotenv
load_dotenv()

import json
import os
from pathlib import Path

import lightgbm as lgb
import shap

MODELS_DIR = Path(os.getenv("MODELS_DIR", Path(__file__).resolve().parent.parent / "models"))


class ModelBundle:
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir

        self.clf_model = lgb.Booster(model_file=str(models_dir / "classification_model.txt"))
        self.reg_lower = lgb.Booster(model_file=str(models_dir / "regression_model_lower.txt"))
        self.reg_median = lgb.Booster(model_file=str(models_dir / "regression_model_median.txt"))
        self.reg_upper = lgb.Booster(model_file=str(models_dir / "regression_model_upper.txt"))

        with open(models_dir / "model_metadata.json") as f:
            self.clf_meta = json.load(f)
        with open(models_dir / "regression_metadata.json") as f:
            self.reg_meta = json.load(f)
        with open(models_dir / "tag_rate_lookup.json") as f:
            self.tag_rates = json.load(f)
        with open(models_dir / "platform_context.json") as f:
            self.platform_context = json.load(f)

        # Built once at startup and reused per request - TreeExplainer is fast enough
        # for this to be practical on every prediction without a noticeable slowdown.
        self.clf_explainer = shap.TreeExplainer(self.clf_model)

    @property
    def feature_cols(self):
        return self.clf_meta["feature_cols"]

    @property
    def categorical_features(self):
        return self.clf_meta["categorical_features"]

    @property
    def final_threshold(self):
        return self.clf_meta["final_threshold"]

    @property
    def body_length_bucket_edges(self):
        return self.clf_meta["body_length_bucket_edges"]

    @property
    def body_length_bucket_categories(self):
        return self.clf_meta["body_length_bucket_categories"]

    @property
    def feature_cols_reg(self):
        return self.reg_meta["feature_cols_reg"]

    @property
    def categorical_features_reg(self):
        return self.reg_meta["categorical_features_reg"]


_bundle: ModelBundle | None = None


def get_model_bundle() -> ModelBundle:
    """Singleton accessor - loads once, reused across all requests."""
    global _bundle
    if _bundle is None:
        _bundle = ModelBundle()
    return _bundle
