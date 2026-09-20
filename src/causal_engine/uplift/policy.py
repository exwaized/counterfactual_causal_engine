"""Turns a fitted CATE model into a deployable targeting policy: a
budget-constrained threshold rule, serializable so the OPE module (module 2)
can load it and evaluate it against logged data without retraining anything.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from causal_engine.uplift.models import CATEModel


class UpliftPolicy:
    """Treat the top `treat_fraction` of a population, ranked by predicted CATE."""

    def __init__(self, cate_model: CATEModel, treat_fraction: float):
        if not 0 < treat_fraction <= 1:
            raise ValueError("treat_fraction must be in (0, 1]")
        self.cate_model = cate_model
        self.treat_fraction = treat_fraction
        self.threshold_: float | None = None

    def calibrate(self, X_calibration: np.ndarray) -> "UpliftPolicy":
        """Set the score threshold so `treat_fraction` of this reference
        population would be treated. Calibrate on a held-out set, not the
        set you'll report policy value on, to avoid an optimistic bias."""
        scores = self.cate_model.predict_cate(X_calibration)
        self.threshold_ = float(np.quantile(scores, 1 - self.treat_fraction))
        return self

    def decide(self, X: np.ndarray) -> np.ndarray:
        if self.threshold_ is None:
            raise RuntimeError("Call .calibrate(...) before .decide(...)")
        scores = self.cate_model.predict_cate(X)
        return scores >= self.threshold_

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "UpliftPolicy":
        return joblib.load(path)
