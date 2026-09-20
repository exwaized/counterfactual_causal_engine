"""Placebo-based inference for synthetic control.

With only a handful of geos, there's no valid normal-theory standard error.
Instead: refit the exact same procedure pretending each *control* geo was
the treated one, and see how unusual the real treated geo's post/pre RMSPE
ratio is against that placebo distribution. This is Abadie, Diamond &
Hainmueller (2010)'s rank permutation test, and it's what Meta's GeoLift
package and most industry geo-lift tooling actually run for significance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from causal_engine.geolift.synthetic_control import SyntheticControl, rmspe


def _fit_one(wide: pd.DataFrame, target_id: str, n_pre: int) -> tuple[float, float, np.ndarray]:
    donor_cols = [c for c in wide.columns if c != target_id]
    pre = wide.iloc[:n_pre]
    full = wide

    sc = SyntheticControl().fit(pre[target_id].values, pre[donor_cols].values, donor_cols)
    synthetic_full = sc.predict(full[donor_cols].values)

    pre_rmspe = rmspe(full[target_id].values[:n_pre], synthetic_full[:n_pre])
    post_rmspe = rmspe(full[target_id].values[n_pre:], synthetic_full[n_pre:])
    return pre_rmspe, post_rmspe, synthetic_full


def run_placebo_tests(wide: pd.DataFrame, treated_id: str, n_pre: int) -> pd.DataFrame:
    rows = []
    for geo_id in wide.columns:
        pre_rmspe, post_rmspe, _ = _fit_one(wide, geo_id, n_pre)
        ratio = post_rmspe / pre_rmspe if pre_rmspe > 0 else np.inf
        rows.append(
            {
                "geo_id": geo_id,
                "pre_rmspe": pre_rmspe,
                "post_rmspe": post_rmspe,
                "post_pre_ratio": ratio,
                "is_treated": geo_id == treated_id,
            }
        )
    return pd.DataFrame(rows).sort_values("post_pre_ratio", ascending=False).reset_index(drop=True)


def placebo_p_value(placebo_df: pd.DataFrame) -> float:
    """Fraction of geos (treated included) whose ratio is >= the treated
    geo's ratio -- a low value means the treated geo's post-period gap is
    unusual relative to what happens when you apply the same procedure to
    geos that received no treatment."""
    treated_ratio = placebo_df.loc[placebo_df["is_treated"], "post_pre_ratio"].iloc[0]
    return float((placebo_df["post_pre_ratio"] >= treated_ratio).mean())
