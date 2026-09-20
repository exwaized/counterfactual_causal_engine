import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from sklearn.model_selection import train_test_split

from causal_engine.common.io import generate_synthetic_uplift
from causal_engine.uplift.evaluate import auuc_estimate
from causal_engine.uplift.models import TLearner, XLearner


def _fit_and_score(model_cls, n=8000, seed=0):
    df = generate_synthetic_uplift(n=n, n_features=6, random_state=seed)
    feat_cols = [c for c in df.columns if c not in ("treatment", "outcome", "true_cate")]
    train, test = train_test_split(df, test_size=0.3, random_state=seed)

    model = model_cls().fit(train[feat_cols].values, train["treatment"].values, train["outcome"].values)
    cate_pred = model.predict_cate(test[feat_cols].values)

    auuc = auuc_estimate(cate_pred, test["treatment"].values, test["outcome"].values)
    corr = np.corrcoef(cate_pred, test["true_cate"].values)[0, 1]
    return auuc.point_estimate, corr


def test_t_learner_beats_random_targeting():
    auuc, corr = _fit_and_score(TLearner)
    assert auuc > 0, "T-learner's Qini curve should beat random targeting on data with real heterogeneity"
    assert corr > 0.3, "predicted CATE should correlate with the known true CATE"


def test_x_learner_beats_random_targeting():
    auuc, corr = _fit_and_score(XLearner)
    assert auuc > 0
    assert corr > 0.3


def test_synthetic_generator_has_randomized_treatment():
    df = generate_synthetic_uplift(n=5000, random_state=1)
    # Treatment should be ~independent of features -> no arm should be
    # trivially separable, otherwise downstream Qini/policy-value math
    # (which assumes randomization) would be invalid.
    treat_rate = df["treatment"].mean()
    assert 0.45 < treat_rate < 0.55
