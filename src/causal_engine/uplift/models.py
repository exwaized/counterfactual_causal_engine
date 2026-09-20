"""Meta-learners for CATE (conditional average treatment effect) estimation.

Implemented directly on scikit-learn rather than depending on causalml/EconML
so this runs anywhere without fighting compiled-extension builds (causalml in
particular is a common pain point on Windows). EconML is left as an optional
extra (see pyproject.toml) for anyone who wants causal forests / DML on top
of this.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression


class CATEModel(Protocol):
    def fit(self, X: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> "CATEModel": ...
    def predict_cate(self, X: np.ndarray) -> np.ndarray: ...


def _default_regressor() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05)


class TLearner:
    """Two independent outcome models, one per arm. CATE = mu1(x) - mu0(x).

    Simple and low-variance when each arm has enough data, but can't share
    statistical strength across arms — biased if the two models overfit
    differently.
    """

    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator or _default_regressor()
        self.mu0 = clone(self.base_estimator)
        self.mu1 = clone(self.base_estimator)

    def fit(self, X: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> "TLearner":
        treatment = np.asarray(treatment).astype(bool)
        self.mu0.fit(X[~treatment], outcome[~treatment])
        self.mu1.fit(X[treatment], outcome[treatment])
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        return self.mu1.predict(X) - self.mu0.predict(X)


class XLearner:
    """T-learner base, then a second stage that imputes individual treatment
    effects and fits a model on those, combined via the propensity score.

    Better than a plain T-learner when the two arms are imbalanced in size,
    since the second stage lets the larger arm's model inform the smaller
    arm's effect estimate.
    """

    def __init__(self, base_estimator=None, propensity_model=None):
        self.base_estimator = base_estimator or _default_regressor()
        self.mu0 = clone(self.base_estimator)
        self.mu1 = clone(self.base_estimator)
        self.tau0 = clone(self.base_estimator)
        self.tau1 = clone(self.base_estimator)
        self.propensity_model = propensity_model or LogisticRegression(max_iter=1000)

    def fit(self, X: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> "XLearner":
        treatment = np.asarray(treatment).astype(bool)

        # Stage 1: outcome models per arm (same as T-learner)
        self.mu0.fit(X[~treatment], outcome[~treatment])
        self.mu1.fit(X[treatment], outcome[treatment])

        # Stage 2: imputed treatment effects
        d1 = outcome[treatment] - self.mu0.predict(X[treatment])
        d0 = self.mu1.predict(X[~treatment]) - outcome[~treatment]
        self.tau1.fit(X[treatment], d1)
        self.tau0.fit(X[~treatment], d0)

        # Propensity model, to weight the two second-stage estimates
        self.propensity_model.fit(X, treatment.astype(int))
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        g = self.propensity_model.predict_proba(X)[:, 1]
        tau1_pred = self.tau1.predict(X)
        tau0_pred = self.tau0.predict(X)
        return g * tau0_pred + (1 - g) * tau1_pred


def get_model(method: str) -> CATEModel:
    if method == "t_learner":
        return TLearner()
    if method == "x_learner":
        return XLearner()
    raise ValueError(f"Unknown uplift method: {method!r}. Expected 't_learner' or 'x_learner'.")
