"""Phase 1: train a CATE model on (synthetic or real) randomized campaign
data, evaluate it, and save a deployable targeting policy.

Usage:
    python pipelines/01_train_uplift_policy.py configs/uplift_synthetic.yaml

Writes results/<run_id>/{summary.json, qini_curve.png, policy.joblib}.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running without `pip install -e .` first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json

from sklearn.model_selection import train_test_split

from causal_engine.common.io import feature_columns, load_uplift_data
from causal_engine.common.schema import UpliftConfig, load_config
from causal_engine.common.viz import plot_qini_curve
from causal_engine.uplift.evaluate import auuc_estimate, policy_value, qini_curve
from causal_engine.uplift.models import get_model
from causal_engine.uplift.policy import UpliftPolicy


def main(config_path: str) -> None:
    cfg: UpliftConfig = load_config(config_path, UpliftConfig)
    results_dir = Path("results") / cfg.run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    df = load_uplift_data(cfg.data_path, cfg.n_synthetic, cfg.n_features, cfg.random_state)
    feat_cols = feature_columns(df, cfg.treatment_col, cfg.outcome_col, cfg.feature_cols)

    train_df, test_df = train_test_split(df, test_size=cfg.test_size, random_state=cfg.random_state)
    X_train, T_train, Y_train = train_df[feat_cols].values, train_df[cfg.treatment_col].values, train_df[cfg.outcome_col].values
    X_test, T_test, Y_test = test_df[feat_cols].values, test_df[cfg.treatment_col].values, test_df[cfg.outcome_col].values

    print(f"[{cfg.run_id}] train={len(train_df)} test={len(test_df)} features={feat_cols} method={cfg.method}")

    model = get_model(cfg.method)
    model.fit(X_train, T_train, Y_train)

    cate_scores_test = model.predict_cate(X_test)
    auuc = auuc_estimate(cate_scores_test, T_test, Y_test)
    fractions, qini_y, random_y, _ = qini_curve(cate_scores_test, T_test, Y_test)
    plot_qini_curve(fractions, qini_y, random_y, auuc.point_estimate, results_dir / "qini_curve.png")
    print(auuc)

    # Calibrate the deployable policy on the training split (not test, to
    # avoid an optimistic threshold), then report its value on test.
    policy = UpliftPolicy(model, cfg.treat_fraction).calibrate(X_train)
    policy.save(results_dir / "policy.joblib")

    v_policy = policy_value(policy.decide(X_test), T_test, Y_test)
    v_treat_all = policy_value(_np_true_like(T_test), T_test, Y_test)
    v_treat_none = policy_value(_np_false_like(T_test), T_test, Y_test)
    for label, est in [("policy", v_policy), ("treat_all", v_treat_all), ("treat_none", v_treat_none)]:
        est.metadata["label"] = label
        print(f"  {label:12s} -> {est}")

    summary = {
        "run_id": cfg.run_id,
        "config": cfg.model_dump(),
        "auuc": auuc.to_dict(),
        "policy_value": v_policy.to_dict(),
        "treat_all_value": v_treat_all.to_dict(),
        "treat_none_value": v_treat_none.to_dict(),
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"Saved results to {results_dir}/")


def _np_true_like(a):
    import numpy as np

    return np.ones_like(a, dtype=bool)


def _np_false_like(a):
    import numpy as np

    return np.zeros_like(a, dtype=bool)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/uplift_synthetic.yaml"
    main(config_path)
