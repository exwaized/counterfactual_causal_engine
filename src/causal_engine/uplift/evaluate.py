"""Uplift evaluation: Qini curve / AUUC, and policy-value simulation.

Both require the *actual* treatment assignment to have been (conditionally)
random — same requirement as a standard A/B test — otherwise "treated did
better" just reflects who was selected for treatment, not the treatment's
effect. The synthetic generator satisfies this; a real dataset must too
(Hillstrom and Criteo's public uplift set both are randomized).
"""
from __future__ import annotations

import numpy as np

from causal_engine.common.effects import EffectEstimate


def qini_curve(cate_scores: np.ndarray, treatment: np.ndarray, outcome: np.ndarray, n_bins: int = 100):
    """Returns (fractions, qini_y, random_y, auuc).

    qini_y[k] = cumulative incremental outcome among the top-scoring
    fraction[k] of the population, i.e. sum(outcome | treated in top-k) -
    sum(outcome | control in top-k) rescaled by group size so the two arms
    are comparable even though they're not equal-sized within the slice.
    auuc = area between the model's Qini curve and the random-targeting
    diagonal — the metric a targeting model is judged on.
    """
    order = np.argsort(-cate_scores)
    treatment_sorted = np.asarray(treatment)[order].astype(bool)
    outcome_sorted = np.asarray(outcome)[order]
    n = len(order)

    fractions = np.linspace(1.0 / n_bins, 1.0, n_bins)
    qini_y = np.empty(n_bins)
    for i, frac in enumerate(fractions):
        k = max(int(round(frac * n)), 1)
        t_mask = treatment_sorted[:k]
        c_mask = ~t_mask
        n_t, n_c = t_mask.sum(), c_mask.sum()
        y_t = outcome_sorted[:k][t_mask].sum()
        y_c = outcome_sorted[:k][c_mask].sum()
        qini_y[i] = y_t - y_c * (n_t / n_c) if n_c > 0 else y_t

    random_y = fractions * qini_y[-1]
    auuc = float(np.trapezoid(qini_y, fractions) - np.trapezoid(random_y, fractions))
    return fractions, qini_y, random_y, auuc


def auuc_estimate(cate_scores: np.ndarray, treatment: np.ndarray, outcome: np.ndarray, n_bins: int = 100) -> EffectEstimate:
    fractions, qini_y, random_y, auuc = qini_curve(cate_scores, treatment, outcome, n_bins)
    return EffectEstimate(
        method="auuc",
        point_estimate=auuc,
        n=len(cate_scores),
        metadata={"n_bins": n_bins},
    )


def policy_value(policy_treat: np.ndarray, treatment: np.ndarray, outcome: np.ndarray, p_treat: float = 0.5) -> EffectEstimate:
    """Inverse-propensity-weighted estimate of the average outcome *if
    everyone had received the treatment decision the policy prescribes*.

    Only uses units where the actual (randomized) treatment happened to
    match the policy's decision, reweighted by 1/p(actual treatment) so the
    estimate stays unbiased. Same IPW logic the OPE module will reuse for
    evaluating a targeting policy against logged (non-uniform) propensities.
    """
    treatment = np.asarray(treatment).astype(bool)
    policy_treat = np.asarray(policy_treat).astype(bool)
    outcome = np.asarray(outcome)

    match = treatment == policy_treat
    weight = np.where(treatment, 1.0 / p_treat, 1.0 / (1.0 - p_treat))
    contrib = np.where(match, weight * outcome, 0.0)

    point = float(contrib.mean())
    se = float(contrib.std(ddof=1) / np.sqrt(len(contrib)))
    return EffectEstimate(
        method="policy_value_ipw",
        point_estimate=point,
        ci_lower=point - 1.96 * se,
        ci_upper=point + 1.96 * se,
        std_error=se,
        n=len(contrib),
    )
