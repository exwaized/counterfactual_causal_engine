"""The synthetic control QP: a convex combination of donor geos that best
tracks the treated geo's pre-period trend (see the derivation discussed
alongside this module -- constrained least squares, solved to the global
optimum since it's a convex problem, no early-stopping/tolerance choice).
"""
from __future__ import annotations

import numpy as np


def rmspe(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def _project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection of v onto {w : w >= 0, sum(w) = 1} (Duchi et al. 2008)."""
    n = len(v)
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    idx = np.arange(1, n + 1)
    cond = u - css / idx > 0
    rho = idx[cond][-1]
    theta = css[cond][-1] / rho
    return np.maximum(v - theta, 0.0)


def _solve_simplex_least_squares(A: np.ndarray, b: np.ndarray, n_iter: int = 20_000, tol: float = 1e-12) -> np.ndarray:
    """min ||Aw - b||^2 s.t. w on the probability simplex, via FISTA
    (accelerated projected gradient). Donor geos here are highly collinear
    (they share one common macro factor), which makes this objective's
    curvature very uneven -- a general nonlinear solver (SLSQP) can stall or
    fail its line search on that, and plain projected gradient descent's safe
    step size (1/Lipschitz) converges too slowly to be usable at that
    collinearity. FISTA keeps the same no-line-search robustness (step size
    fixed below 1/Lipschitz, exact simplex projection every step) but adds
    Nesterov momentum, giving O(1/k^2) instead of O(1/k) convergence.
    """
    n_donors = A.shape[1]
    spectral_norm = np.linalg.norm(A, ord=2)
    lipschitz = 2.0 * spectral_norm**2
    step = 1.0 / lipschitz if lipschitz > 0 else 1e-6

    w = np.full(n_donors, 1.0 / n_donors)
    y = w.copy()
    t = 1.0
    prev_obj = np.inf
    for _ in range(n_iter):
        grad = 2.0 * A.T @ (A @ y - b)
        w_new = _project_to_simplex(y - step * grad)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t**2)) / 2.0
        y = w_new + ((t - 1.0) / t_new) * (w_new - w)

        obj = float(np.sum((A @ w_new - b) ** 2))
        if abs(prev_obj - obj) < tol:
            w = w_new
            break
        w, t, prev_obj = w_new, t_new, obj
    return w


class SyntheticControl:
    """Fits donor weights on the pre-period, then projects a synthetic
    (counterfactual, no-treatment) path for the full period."""

    def __init__(self):
        self.weights_: np.ndarray | None = None
        self.donor_columns_: list[str] | None = None
        self.pre_rmspe_: float | None = None

    def fit(self, pre_treated: np.ndarray, pre_donors: np.ndarray, donor_columns: list[str]) -> "SyntheticControl":
        """pre_treated: (T0,) treated geo's pre-period outcomes.
        pre_donors: (T0, J) donor geos' pre-period outcomes, columns matching donor_columns.
        """
        self.weights_ = _solve_simplex_least_squares(pre_donors, pre_treated)
        self.donor_columns_ = list(donor_columns)
        self.pre_rmspe_ = rmspe(pre_treated, pre_donors @ self.weights_)
        return self

    def predict(self, donors: np.ndarray) -> np.ndarray:
        """donors: (T, J) donor outcomes over any period, same column order as fit()."""
        if self.weights_ is None:
            raise RuntimeError("Call .fit(...) first")
        return donors @ self.weights_

    def weight_table(self) -> dict[str, float]:
        return dict(zip(self.donor_columns_, self.weights_.tolist()))
