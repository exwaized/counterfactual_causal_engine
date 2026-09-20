"""Target policies to vet against logged data. `load_target_policy` is the
seam that connects phase 3 back to phase 1: point it at the `policy.joblib`
a `pipelines/01_train_uplift_policy.py` run saved, and OPE evaluates that
exact deployable policy, no retraining.
"""
from __future__ import annotations

import numpy as np

from causal_engine.uplift.policy import UpliftPolicy


def treat_all(X: np.ndarray) -> np.ndarray:
    return np.ones(len(X), dtype=bool)


def treat_none(X: np.ndarray) -> np.ndarray:
    return np.zeros(len(X), dtype=bool)


def random_policy(X: np.ndarray, rate: float = 0.5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.binomial(1, rate, size=len(X)).astype(bool)


def load_uplift_policy_decider(path: str):
    policy = UpliftPolicy.load(path)
    return policy.decide


def get_target_policy(name: str, uplift_policy_path: str | None = None):
    """Returns a callable X -> bool array of treat decisions."""
    if name == "treat_all":
        return treat_all
    if name == "treat_none":
        return treat_none
    if name == "random":
        return random_policy
    if name == "uplift_policy":
        if uplift_policy_path is None:
            raise ValueError("uplift_policy_path is required for target_policy='uplift_policy'")
        return load_uplift_policy_decider(uplift_policy_path)
    raise ValueError(f"Unknown target policy: {name!r}")
