"""Compares each estimator's off-policy value against the known ground
truth (computable here because the data is synthetic) -- the rigorous way
to validate an OPE estimator, same spirit as Open Bandit Dataset's
random-logging-policy benchmark: don't just check the number looks
plausible, check it against a value you independently know is right.
"""
from __future__ import annotations

import pandas as pd

from causal_engine.common.effects import EffectEstimate


def compare_to_ground_truth(estimates: dict[str, EffectEstimate], true_value: float) -> pd.DataFrame:
    rows = []
    for name, est in estimates.items():
        rows.append(
            {
                "estimator": name,
                "point_estimate": est.point_estimate,
                "ci_lower": est.ci_lower,
                "ci_upper": est.ci_upper,
                "true_value": true_value,
                "abs_error": abs(est.point_estimate - true_value),
                "ci_covers_truth": est.ci_lower <= true_value <= est.ci_upper,
                "std_error": est.std_error,
            }
        )
    return pd.DataFrame(rows).sort_values("abs_error").reset_index(drop=True)
