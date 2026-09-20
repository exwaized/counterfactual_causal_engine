"""Data loading for the uplift module.

Real datasets (Hillstrom, Criteo) are loaded from a local CSV if
`data_path` is set in the config. No CSV is bundled or auto-downloaded
(avoids depending on a specific mirror URL staying alive) — drop the file
into data/raw/ yourself and point the config at it.

When no `data_path` is given, `generate_synthetic_uplift` builds a
randomized-treatment dataset with a known, injected treatment-effect
function. That known ground truth (`true_cate`, kept out of the feature
set) is what lets tests assert the uplift models actually recover
heterogeneity instead of just "running without crashing".
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def dgp_baseline_and_cate(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The shared ground-truth reward model behind every synthetic dataset in
    this project (uplift's randomized campaign, ope's logged bandit data):
    same baseline and same heterogeneous treatment effect function, driven
    only by features 0 and 1. Keeping this in one place means a policy
    trained in uplift/ is being vetted, in ope/, against the *same* business
    reality it was trained on -- not a different simulated world.
    """
    baseline = 1.0 + 0.5 * X[:, 0] - 0.3 * X[:, 2] + 0.2 * X[:, 3]
    true_cate = 2.0 * np.clip(X[:, 0], 0, None) + 1.5 * (X[:, 1] > 0.5).astype(float) - 0.5
    return baseline, true_cate


def generate_synthetic_uplift(
    n: int = 20_000,
    n_features: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n, n_features))

    # Randomized assignment -> naive treated-vs-control comparisons are
    # unbiased here, same as a real A/B test / randomized email campaign.
    treatment = rng.binomial(1, 0.5, size=n)

    baseline, true_cate = dgp_baseline_and_cate(X)
    noise = rng.normal(scale=1.0, size=n)
    outcome = baseline + treatment * true_cate + noise

    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(n_features)])
    df["treatment"] = treatment
    df["outcome"] = outcome
    df["true_cate"] = true_cate  # ground truth, for evaluation only — never a model feature
    return df


def load_uplift_data(data_path: str | None, n_synthetic: int, n_features: int, random_state: int) -> pd.DataFrame:
    if data_path is None:
        return generate_synthetic_uplift(n=n_synthetic, n_features=n_features, random_state=random_state)
    df = pd.read_csv(data_path)
    return df


def feature_columns(df: pd.DataFrame, treatment_col: str, outcome_col: str, explicit: list[str] | None) -> list[str]:
    if explicit is not None:
        return explicit
    excluded = {treatment_col, outcome_col, "true_cate"}
    return [c for c in df.columns if c not in excluded]
