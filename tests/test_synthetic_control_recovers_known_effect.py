import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from causal_engine.geolift.inference import placebo_p_value, run_placebo_tests
from causal_engine.geolift.simulate import generate_geo_panel, to_wide
from causal_engine.geolift.synthetic_control import SyntheticControl


def _fit(true_effect_pct=0.08, n_pre=52, n_post=8, n_donors=20, seed=1):
    panel, treated_id = generate_geo_panel(
        n_donor_geos=n_donors, n_pre_periods=n_pre, n_post_periods=n_post,
        true_effect_pct=true_effect_pct, random_state=seed,
    )
    wide = to_wide(panel)
    donor_cols = [c for c in wide.columns if c != treated_id]
    sc = SyntheticControl().fit(wide[treated_id].values[:n_pre], wide[donor_cols].values[:n_pre], donor_cols)
    synthetic_full = sc.predict(wide[donor_cols].values)
    actual_full = wide[treated_id].values
    estimated = (actual_full[n_pre:].sum() - synthetic_full[n_pre:].sum()) / synthetic_full[n_pre:].sum()
    return estimated, sc, wide, treated_id


def test_weights_are_a_valid_convex_combination():
    _, sc, _, _ = _fit()
    assert abs(sc.weights_.sum() - 1.0) < 1e-6
    assert (sc.weights_ >= -1e-9).all()


def test_recovers_known_injected_effect_on_average():
    # A single seed's point estimate is noisy -- a mediocre pre-period donor
    # fit on one draw can miss by a lot, which is exactly why practitioners
    # lean on pre-RMSPE / placebo tests rather than trusting any one run.
    # The right claim to test is that the estimator is unbiased *on average*.
    true_effect = 0.08
    estimates = [_fit(true_effect_pct=true_effect, seed=s)[0] for s in range(1, 16)]
    import numpy as np

    mean_estimate = np.mean(estimates)
    assert abs(mean_estimate - true_effect) < 0.015


def test_good_pre_fit_recovers_effect_tightly():
    # On a seed with a tight pre-period fit (low pre_rmspe), the post-period
    # estimate should be close to the true effect -- this is the condition
    # under which a single-run point estimate is actually trustworthy.
    true_effect = 0.08
    estimated, sc, _, _ = _fit(true_effect_pct=true_effect, seed=2)
    assert sc.pre_rmspe_ < 3.0
    assert abs(estimated - true_effect) < 0.02


def test_no_effect_case_is_not_falsely_significant():
    """With true_effect_pct=0, the treated geo is just another control geo --
    placebo p-value should NOT reliably come out small."""
    estimated, sc, wide, treated_id = _fit(true_effect_pct=0.0, seed=7)
    placebo_df = run_placebo_tests(wide, treated_id, n_pre=52)
    p = placebo_p_value(placebo_df)
    assert p > 0.1
