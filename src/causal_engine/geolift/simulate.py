"""Synthetic geo panel generator with a known, injected treatment effect.

No public geo-marketing dataset ships ground-truth incremental lift, so the
primary validation for synthetic control here is recovery: inject a known
% lift into one geo's post-period outcomes, then check the method recovers
something close to it. Geos are correlated through a shared stochastic
common factor (macro trend/seasonality shocks) with geo-specific loadings,
levels, and trends -- similar but not identical series, same as real
regional sales/signups data, so the donor pool is a nontrivial fit problem
rather than one control geo being a trivial stand-in for another.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_geo_panel(
    n_donor_geos: int = 20,
    n_pre_periods: int = 52,
    n_post_periods: int = 8,
    true_effect_pct: float = 0.08,
    random_state: int = 42,
) -> tuple[pd.DataFrame, str]:
    rng = np.random.default_rng(random_state)
    n_geos = n_donor_geos + 1
    T = n_pre_periods + n_post_periods
    geo_ids = [f"geo_{i}" for i in range(n_geos)]
    treated_id = geo_ids[0]

    common_factor = np.cumsum(rng.normal(scale=1.0, size=T))
    base_levels = rng.uniform(80, 120, size=n_geos)
    trends = rng.uniform(-0.1, 0.3, size=n_geos)
    loadings = rng.uniform(0.5, 1.5, size=n_geos)

    rows = []
    for gi, gid in enumerate(geo_ids):
        for t in range(T):
            y = base_levels[gi] + trends[gi] * t + loadings[gi] * common_factor[t] + rng.normal(scale=2.0)
            is_post = t >= n_pre_periods
            is_treated_geo = gid == treated_id
            if is_post and is_treated_geo:
                y = y * (1 + true_effect_pct)
            rows.append(
                {
                    "geo_id": gid,
                    "period": t,
                    "is_post": is_post,
                    "is_treated_geo": is_treated_geo,
                    "outcome": y,
                }
            )
    return pd.DataFrame(rows), treated_id


def to_wide(panel: pd.DataFrame) -> pd.DataFrame:
    """Long (geo_id, period, outcome) -> wide (period index, one column per geo)."""
    return panel.pivot(index="period", columns="geo_id", values="outcome").sort_index()
