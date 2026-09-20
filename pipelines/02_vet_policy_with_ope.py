"""Phase 3: vet a target policy against logged data from a different,
non-uniform logging policy -- offline, before spending an A/B test slot.

By default the target policy is the one phase 1 trained and saved
(results/uplift_synthetic_v1/policy.joblib) -- run
`pipelines/01_train_uplift_policy.py` first, or set target_policy to
treat_all/treat_none/random in the config to skip that dependency.

Usage:
    python pipelines/02_vet_policy_with_ope.py configs/ope_synthetic.yaml

Writes results/<run_id>/{summary.json, ope_comparison.png}.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json

from causal_engine.common.schema import OPEConfig, load_config
from causal_engine.common.viz import plot_ope_comparison
from causal_engine.ope.estimators import dr_estimate, ips_estimate, snips_estimate
from causal_engine.ope.evaluate import compare_to_ground_truth
from causal_engine.ope.policies import get_target_policy
from causal_engine.ope.simulate import generate_logged_bandit_data, true_policy_value


def main(config_path: str) -> None:
    cfg: OPEConfig = load_config(config_path, OPEConfig)
    results_dir = Path("results") / cfg.run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    if cfg.data_path is None:
        df = generate_logged_bandit_data(n=cfg.n_synthetic, n_features=cfg.n_features, random_state=cfg.random_state)
    else:
        import pandas as pd

        df = pd.read_csv(cfg.data_path)

    feat_cols = [f"x{i}" for i in range(cfg.n_features)]
    X = df[feat_cols].values
    action = df["action"].values
    propensity = df["propensity"].values
    reward = df["reward"].values

    decide = get_target_policy(cfg.target_policy, cfg.uplift_policy_path)
    target_action = decide(X).astype(int)

    print(f"[{cfg.run_id}] n={len(df)}, target_policy={cfg.target_policy}, target treats {target_action.mean():.1%} of population")

    true_value_target = true_policy_value(df, target_action.astype(float))
    true_value_logging = true_policy_value(df, propensity)  # expected value of the policy that actually generated the logs
    empirical_logged_reward = float(reward.mean())

    estimates = {
        "ips": ips_estimate(target_action, action, propensity, reward, n_boot=cfg.n_bootstrap, seed=cfg.random_state),
        "snips": snips_estimate(target_action, action, propensity, reward, n_boot=cfg.n_bootstrap, seed=cfg.random_state),
        "dr": dr_estimate(target_action, action, propensity, reward, X, dm_train_fraction=cfg.dm_train_fraction, n_boot=cfg.n_bootstrap, seed=cfg.random_state),
    }
    for est in estimates.values():
        print(f"  {est}")

    comparison = compare_to_ground_truth(estimates, true_value_target)
    print(comparison.to_string(index=False))
    plot_ope_comparison(comparison, true_value_target, results_dir / "ope_comparison.png")

    print(f"\n  true target-policy value:  {true_value_target:.4f}")
    print(f"  true logging-policy value: {true_value_logging:.4f}  (empirical logged reward: {empirical_logged_reward:.4f})")
    verdict = "WORTH an A/B test" if comparison.loc[comparison['estimator'] == 'dr', 'point_estimate'].iloc[0] > true_value_logging else "NOT clearly better -- do not spend an A/B test slot yet"
    print(f"  DR-based verdict: target policy looks {verdict}")

    summary = {
        "run_id": cfg.run_id,
        "config": cfg.model_dump(),
        "true_value_target_policy": true_value_target,
        "true_value_logging_policy": true_value_logging,
        "empirical_logged_reward": empirical_logged_reward,
        "estimates": {k: v.to_dict() for k, v in estimates.items()},
        "comparison": comparison.to_dict(orient="records"),
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"Saved results to {results_dir}/")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/ope_synthetic.yaml"
    main(config_path)
