"""Off-policy value estimators: given logs (x, action, propensity, reward)
from a *logging* policy, estimate the average reward a *target* policy
would have gotten, without ever running it.

All three share the same core trick: reweight logged rewards by how much
more (or less) likely the target policy is to have taken the logged action
than the logging policy was. Here the target policies are all deterministic
(uplift-style treat/no-treat rules), so the importance weight collapses to
"1/propensity if the logged action matches the target policy's action,
else 0" -- no logged reward can inform the target policy's value for a
context where the two policies disagree on the action.

CIs are bootstrap (resampling rows), not a derived analytic formula --
correct for IPS/SNIPS/DR alike without deriving three different delta-method
variances by hand, at the cost of being a bit slower.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from causal_engine.common.effects import EffectEstimate
from causal_engine.uplift.models import TLearner


def _action_propensity(action: np.ndarray, propensity_treat: np.ndarray) -> np.ndarray:
    return np.where(action == 1, propensity_treat, 1.0 - propensity_treat)


def _bootstrap_ci(values_fn, n: int, n_boot: int, seed: int) -> tuple[float, float, float]:
    """values_fn(idx) -> scalar estimate on the resampled index set idx.
    Returns (se, ci_lower, ci_upper) from the bootstrap distribution."""
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[b] = values_fn(idx)
    se = float(boot.std(ddof=1))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return se, float(lo), float(hi)


def ips_estimate(target_action: np.ndarray, action: np.ndarray, propensity: np.ndarray, reward: np.ndarray, n_boot: int = 500, seed: int = 0) -> EffectEstimate:
    n = len(reward)
    p = _action_propensity(action, propensity)
    match = (target_action == action).astype(float)
    weight = match / p
    contrib = weight * reward

    point = float(contrib.mean())

    def boot_fn(idx):
        return contrib[idx].mean()

    se, lo, hi = _bootstrap_ci(boot_fn, n, n_boot, seed)
    return EffectEstimate(
        method="ips", point_estimate=point, ci_lower=lo, ci_upper=hi, std_error=se, n=n,
        metadata={"effective_sample_size": float(match.sum())},
    )


def snips_estimate(target_action: np.ndarray, action: np.ndarray, propensity: np.ndarray, reward: np.ndarray, n_boot: int = 500, seed: int = 0) -> EffectEstimate:
    n = len(reward)
    p = _action_propensity(action, propensity)
    match = (target_action == action).astype(float)
    weight = match / p

    def boot_fn(idx):
        w, r = weight[idx], reward[idx]
        return float(np.sum(w * r) / np.sum(w)) if w.sum() > 0 else 0.0

    point = boot_fn(np.arange(n))
    se, lo, hi = _bootstrap_ci(boot_fn, n, n_boot, seed)
    return EffectEstimate(
        method="snips", point_estimate=point, ci_lower=lo, ci_upper=hi, std_error=se, n=n,
        metadata={"effective_sample_size": float(match.sum())},
    )


def dr_estimate(
    target_action: np.ndarray,
    action: np.ndarray,
    propensity: np.ndarray,
    reward: np.ndarray,
    X: np.ndarray,
    dm_train_fraction: float = 0.5,
    n_boot: int = 500,
    seed: int = 0,
) -> EffectEstimate:
    """Doubly robust: a direct-method reward model (mu0/mu1, same T-learner
    machinery as the uplift module) corrects for the fact that IPS only
    learns from rows where the logged action happened to match the target
    policy. Fit the direct-method model on a held-out training fold so the
    correction term isn't evaluated on the same rows it was fit on.
    """
    n = len(reward)
    idx = np.arange(n)
    train_idx, eval_idx = train_test_split(idx, train_size=dm_train_fraction, random_state=seed)

    dm = TLearner().fit(X[train_idx], action[train_idx], reward[train_idx])
    mu0_eval = dm.mu0.predict(X[eval_idx])
    mu1_eval = dm.mu1.predict(X[eval_idx])

    a_eval = action[eval_idx]
    t_eval = target_action[eval_idx]
    p_eval = _action_propensity(a_eval, propensity[eval_idx])
    r_eval = reward[eval_idx]

    q_logged = np.where(a_eval == 1, mu1_eval, mu0_eval)
    q_target = np.where(t_eval == 1, mu1_eval, mu0_eval)
    match = (t_eval == a_eval).astype(float)

    contrib = q_target + (match / p_eval) * (r_eval - q_logged)
    n_eval = len(contrib)
    point = float(contrib.mean())

    def boot_fn(idx2):
        return contrib[idx2].mean()

    se, lo, hi = _bootstrap_ci(boot_fn, n_eval, n_boot, seed)
    return EffectEstimate(
        method="dr", point_estimate=point, ci_lower=lo, ci_upper=hi, std_error=se, n=n_eval,
        metadata={"dm_train_size": len(train_idx), "effective_sample_size": float(match.sum())},
    )
