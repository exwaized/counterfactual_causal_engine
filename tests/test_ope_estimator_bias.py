import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from causal_engine.ope.estimators import dr_estimate, ips_estimate, snips_estimate
from causal_engine.ope.simulate import generate_logged_bandit_data, true_policy_value

N_FEATURES = 10
SEED = 42


def _logged_data():
    df = generate_logged_bandit_data(n=20_000, n_features=N_FEATURES, random_state=SEED)
    X = df[[f"x{i}" for i in range(N_FEATURES)]].values
    return df, X, df["action"].values, df["propensity"].values, df["reward"].values


def _all_estimates(target_action, X, action, propensity, reward):
    return {
        "ips": ips_estimate(target_action, action, propensity, reward, n_boot=300, seed=SEED),
        "snips": snips_estimate(target_action, action, propensity, reward, n_boot=300, seed=SEED),
        "dr": dr_estimate(target_action, action, propensity, reward, X, n_boot=300, seed=SEED),
    }


def test_logged_reward_matches_true_logging_policy_value():
    # Sanity check on the simulator itself before trusting any estimator on it.
    df, X, action, propensity, reward = _logged_data()
    true_logging_value = true_policy_value(df, propensity)
    assert abs(reward.mean() - true_logging_value) < 0.05


def test_estimators_recover_true_value_for_treat_all():
    df, X, action, propensity, reward = _logged_data()
    target_action = np.ones(len(df), dtype=int)
    true_value = true_policy_value(df, target_action.astype(float))

    for name, est in _all_estimates(target_action, X, action, propensity, reward).items():
        assert abs(est.point_estimate - true_value) < 0.05, f"{name} missed true value by too much"
        assert est.ci_lower <= true_value <= est.ci_upper, f"{name}'s 95% CI should cover the true value"


def test_estimators_recover_true_value_under_partial_overlap():
    # A target policy that disagrees with the logging policy on a lot of
    # units (only x0 > 0.2 gets treated) -- overlap is imperfect but not
    # degenerate (propensities are clipped to [0.05, 0.95] in the generator).
    df, X, action, propensity, reward = _logged_data()
    target_action = (X[:, 0] > 0.2).astype(int)
    true_value = true_policy_value(df, target_action.astype(float))

    for name, est in _all_estimates(target_action, X, action, propensity, reward).items():
        assert abs(est.point_estimate - true_value) < 0.05, f"{name} missed true value by too much"


def test_disagreeing_policies_have_smaller_effective_sample_size():
    # A policy far from the logging policy should get less "support" from
    # the logged data (fewer rows where the actions actually match) --
    # exactly the overlap problem that limits OPE in practice. Derive which
    # constant policy is the "majority" one from the data itself, rather
    # than assuming a direction -- a constant policy predicting whatever the
    # logging policy did more often is guaranteed to match more rows.
    df, X, action, propensity, reward = _logged_data()
    majority_action = int(round(action.mean()))
    majority_policy = np.full(len(df), majority_action)
    minority_policy = 1 - majority_policy

    ips_majority = ips_estimate(majority_policy, action, propensity, reward, n_boot=50, seed=SEED)
    ips_minority = ips_estimate(minority_policy, action, propensity, reward, n_boot=50, seed=SEED)

    assert ips_majority.metadata["effective_sample_size"] > ips_minority.metadata["effective_sample_size"]
