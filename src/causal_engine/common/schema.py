"""Pydantic config contracts loaded from configs/*.yaml.

Every pipeline script reads one of these instead of hardcoding dataset paths
or hyperparameters, so a new dataset or method is a new YAML file, not a
code change.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class UpliftConfig(BaseModel):
    run_id: str
    data_path: str | None = None  # None -> fall back to the synthetic generator
    n_synthetic: int = 20_000
    n_features: int = 10
    treatment_col: str = "treatment"
    outcome_col: str = "outcome"
    feature_cols: list[str] | None = None  # None -> inferred (all non-treatment/outcome cols)
    method: str = "t_learner"  # "t_learner" | "x_learner"
    test_size: float = 0.3
    random_state: int = 42
    treat_fraction: float = Field(0.3, description="Fraction of the eligible population to treat under the learned policy")


class OPEConfig(BaseModel):
    run_id: str
    data_path: str | None = None  # None -> fall back to the synthetic logged-bandit generator
    n_synthetic: int = 20_000
    n_features: int = 10
    target_policy: str = "uplift_policy"  # "uplift_policy" | "treat_all" | "treat_none" | "random"
    uplift_policy_path: str = "results/uplift_synthetic_v1/policy.joblib"
    dm_train_fraction: float = Field(0.5, description="Fraction of logged data used to fit the direct-method model for the DR estimator")
    n_bootstrap: int = 500
    random_state: int = 42


class GeoliftConfig(BaseModel):
    run_id: str
    n_donor_geos: int = 20
    n_pre_periods: int = 52
    n_post_periods: int = 8
    true_effect_pct: float = 0.08
    random_state: int = 42


def load_config(path: str | Path, model: type[BaseModel]) -> BaseModel:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return model.model_validate(raw)
