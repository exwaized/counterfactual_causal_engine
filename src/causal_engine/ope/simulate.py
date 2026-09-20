"""Synthetic logged-bandit data: what a production targeting policy leaves
behind. Unlike uplift's randomized generator, treatment here is assigned by
a non-uniform *logging policy* -- e.g. an ops rule that treats people with
high spend history more often, unrelated to where the actual treatment
effect is highest. That's the realistic starting point for off-policy
evaluation: you have logs from whatever policy was already running, not a
clean 50/50 split.

Reuses `dgp_baseline_and_cate` from common/io.py -- the same ground-truth
reward function behind the uplift module's data -- so a policy trained in
uplift/ is being vetted here against the same underlying business reality.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from causal_engine.common.io import dgp_baseline_and_cate


def generate_logged_bandit_data(
    n: int = 20_000,
    n_features: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n, n_features))
    baseline, true_cate = dgp_baseline_and_cate(X)

    # Logging policy: a proxy score correlated with, but not equal to, the
    # true CATE driver -- realistic for a pre-existing production rule that
    # was never actually optimized for treatment effect.
    proxy_score = X[:, 0] + 0.3 * rng.normal(size=n)
    propensity = 1.0 / (1.0 + np.exp(-(1.5 * proxy_score - 0.5)))
    propensity = np.clip(propensity, 0.05, 0.95)  # keep overlap -- no near-zero/one propensities

    action = rng.binomial(1, propensity)
    noise = rng.normal(scale=1.0, size=n)
    reward = baseline + action * true_cate + noise

    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(n_features)])
    df["propensity"] = propensity  # P(action=1 | x) under the logging policy
    df["action"] = action
    df["reward"] = reward
    df["true_cate"] = true_cate  # ground truth, evaluation only
    df["baseline"] = baseline  # ground truth, evaluation only
    return df


def true_policy_value(df: pd.DataFrame, target_actions: np.ndarray) -> float:
    """Exact expected reward under the target policy, using the known
    generating functions directly (no sampling noise) -- the ground truth
    an OPE estimator is trying to recover from noisy logs alone."""
    return float(np.mean(df["baseline"].values + target_actions * df["true_cate"].values))
