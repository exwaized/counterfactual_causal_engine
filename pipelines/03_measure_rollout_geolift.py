"""Phase 2: measure a geo-level campaign's incremental effect with synthetic
control, on a synthetic panel with a known injected effect.

Usage:
    python pipelines/03_measure_rollout_geolift.py configs/geolift_synthetic.yaml

Writes results/<run_id>/{summary.json, geo_trajectory.png}.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json

from causal_engine.common.effects import EffectEstimate
from causal_engine.common.schema import GeoliftConfig, load_config
from causal_engine.common.viz import plot_geo_trajectory
from causal_engine.geolift.inference import placebo_p_value, run_placebo_tests
from causal_engine.geolift.simulate import generate_geo_panel, to_wide
from causal_engine.geolift.synthetic_control import SyntheticControl


def main(config_path: str) -> None:
    cfg: GeoliftConfig = load_config(config_path, GeoliftConfig)
    results_dir = Path("results") / cfg.run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    panel, treated_id = generate_geo_panel(
        n_donor_geos=cfg.n_donor_geos,
        n_pre_periods=cfg.n_pre_periods,
        n_post_periods=cfg.n_post_periods,
        true_effect_pct=cfg.true_effect_pct,
        random_state=cfg.random_state,
    )
    wide = to_wide(panel)
    donor_cols = [c for c in wide.columns if c != treated_id]
    n_pre = cfg.n_pre_periods

    print(f"[{cfg.run_id}] {len(donor_cols)} donor geos, {n_pre} pre-periods, {cfg.n_post_periods} post-periods, true effect={cfg.true_effect_pct:.1%}")

    sc = SyntheticControl().fit(
        wide[treated_id].values[:n_pre],
        wide[donor_cols].values[:n_pre],
        donor_cols,
    )
    synthetic_full = sc.predict(wide[donor_cols].values)
    actual_full = wide[treated_id].values

    plot_geo_trajectory(wide.index.values, actual_full, synthetic_full, n_pre, results_dir / "geo_trajectory.png")

    actual_post = actual_full[n_pre:]
    synthetic_post = synthetic_full[n_pre:]
    estimated_pct_lift = float((actual_post.sum() - synthetic_post.sum()) / synthetic_post.sum())

    placebo_df = run_placebo_tests(wide, treated_id, n_pre)
    p_value = placebo_p_value(placebo_df)

    effect = EffectEstimate(
        method="synthetic_control_pct_lift",
        point_estimate=estimated_pct_lift,
        n=len(donor_cols) + 1,
        metadata={
            "true_effect_pct": cfg.true_effect_pct,
            "recovery_error_pp": (estimated_pct_lift - cfg.true_effect_pct) * 100,
            "placebo_p_value": p_value,
            "pre_period_rmspe": sc.pre_rmspe_,
            "top_donor_weights": dict(sorted(sc.weight_table().items(), key=lambda kv: -kv[1])[:5]),
        },
    )
    print(effect)
    print(f"  true effect:      {cfg.true_effect_pct:+.2%}")
    print(f"  estimated effect: {estimated_pct_lift:+.2%}")
    print(f"  placebo p-value:  {p_value:.3f}")

    summary = {
        "run_id": cfg.run_id,
        "config": cfg.model_dump(),
        "effect": effect.to_dict(),
        "placebo_ratios": placebo_df.to_dict(orient="records"),
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"Saved results to {results_dir}/")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/geolift_synthetic.yaml"
    main(config_path)
