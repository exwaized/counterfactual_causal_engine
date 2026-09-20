"""Phase 4: run the full chain end-to-end and produce one combined report.

    uplift (who to target)
      -> ope (is this policy worth an A/B test, using only logged data?)
      -> geolift (the policy gets rolled out as a broad geo-level campaign --
         can't target individuals within a geo at that scale -- did it
         actually work?)

The geo rollout's injected effect size isn't an arbitrary config number: it's
derived from what OPE's doubly-robust estimator said the policy would do
relative to the current production policy, using only pre-existing logged
data (empirical_logged_reward), exactly the number a team would actually
have in hand before deciding to roll something out. Geolift then measures
whether the (simulated) rollout matches that expectation.

Usage:
    python pipelines/04_full_pipeline.py

Writes results/full_pipeline_v1/{summary.json, report.html} plus the
three modules' own results/ subdirectories.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import yaml

UPLIFT_CONFIG = "configs/uplift_synthetic.yaml"
OPE_CONFIG = "configs/ope_synthetic.yaml"
GEOLIFT_ROLLOUT_CONFIG = "configs/geolift_rollout.yaml"


def run_step(script: str, config: str) -> None:
    print(f"\n=== running {script} {config} ===")
    subprocess.run([sys.executable, script, config], cwd=ROOT, check=True)


def derive_rollout_effect_pct(ope_summary: dict) -> float:
    """What OPE's DR estimate implies the rollout should do to the topline
    metric, relative to what's running today -- the number a team would
    actually act on, not an oracle "true" value."""
    dr_point = ope_summary["estimates"]["dr"]["point_estimate"]
    baseline = ope_summary["empirical_logged_reward"]
    pct = (dr_point - baseline) / baseline
    return float(np.clip(pct, -0.5, 0.5))


def write_geolift_rollout_config(effect_pct: float) -> None:
    base = yaml.safe_load(Path("configs/geolift_synthetic.yaml").read_text())
    base["run_id"] = "geolift_rollout_v1"
    base["true_effect_pct"] = effect_pct
    Path(GEOLIFT_ROLLOUT_CONFIG).write_text(yaml.safe_dump(base, sort_keys=False))
    print(f"Derived rollout effect from OPE: {effect_pct:+.2%} -> wrote {GEOLIFT_ROLLOUT_CONFIG}")


def build_report(uplift_s: dict, ope_s: dict, geo_s: dict, results_dir: Path) -> str:
    dr = ope_s["estimates"]["dr"]
    return f"""<!doctype html>
<title>Causal decision engine — full pipeline run</title>
<meta charset="utf-8">
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
  h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 0.3rem; margin-top: 2.5rem; }}
  img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
  table {{ border-collapse: collapse; }}
  td, th {{ padding: 0.3rem 0.8rem; text-align: left; border-bottom: 1px solid #eee; }}
  code {{ background: #f2f2f2; padding: 0.1rem 0.3rem; border-radius: 3px; }}
</style>

<h1>Causal decision engine — full pipeline run</h1>
<p>uplift (who to target) &rarr; ope (worth an A/B test?) &rarr; geolift (did the rollout work?)</p>

<h2>1. Uplift — who to target</h2>
<p>Trained a {uplift_s['config']['method']} on randomized campaign data, targeting the top
{uplift_s['config']['treat_fraction']:.0%} of the population by predicted CATE.
AUUC = {uplift_s['auuc']['point_estimate']:.2f} (Qini curve above random targeting).</p>
<img src="../{uplift_s['run_id']}/qini_curve.png">

<h2>2. OPE — is it worth an A/B test?</h2>
<p>Vetted that policy against logs from a <em>different</em>, non-uniform production policy,
using only historical data. Doubly-robust estimate: <b>{dr['point_estimate']:.4f}</b>
[{dr['ci_lower']:.4f}, {dr['ci_upper']:.4f}] vs. the current policy's empirical value
<b>{ope_s['empirical_logged_reward']:.4f}</b> &mdash; estimated
<b>{(dr['point_estimate']/ope_s['empirical_logged_reward'] - 1):+.2%}</b> relative lift, before
running any live experiment.</p>
<img src="../{ope_s['run_id']}/ope_comparison.png">

<h2>3. Geolift — did the (simulated) rollout work?</h2>
<p>The vetted policy got rolled out as a broad geo-level campaign. Injected effect
(derived from step 2's DR estimate, clipped to &plusmn;50%): <b>{geo_s['config']['true_effect_pct']:+.2%}</b>.
Synthetic control's recovered estimate: <b>{geo_s['effect']['point_estimate']:+.2%}</b>,
placebo p-value <b>{geo_s['effect']['metadata']['placebo_p_value']:.3f}</b>.</p>
<img src="../{geo_s['run_id']}/geo_trajectory.png">

<h2>Bottom line</h2>
<table>
<tr><th>Stage</th><th>Question</th><th>Answer</th></tr>
<tr><td>Uplift</td><td>Who to target?</td><td>Top {uplift_s['config']['treat_fraction']:.0%} by predicted CATE</td></tr>
<tr><td>OPE</td><td>Worth testing?</td><td>{'Yes' if dr['point_estimate'] > ope_s['empirical_logged_reward'] else 'No'} (DR est. {(dr['point_estimate']/ope_s['empirical_logged_reward'] - 1):+.2%} vs. current)</td></tr>
<tr><td>Geolift</td><td>Did it work?</td><td>{geo_s['effect']['point_estimate']:+.2%} measured (p={geo_s['effect']['metadata']['placebo_p_value']:.3f})</td></tr>
</table>
"""


def main() -> None:
    run_step("pipelines/01_train_uplift_policy.py", UPLIFT_CONFIG)
    run_step("pipelines/02_vet_policy_with_ope.py", OPE_CONFIG)

    ope_summary = json.loads(Path("results/ope_synthetic_v1/summary.json").read_text())
    effect_pct = derive_rollout_effect_pct(ope_summary)
    write_geolift_rollout_config(effect_pct)

    run_step("pipelines/03_measure_rollout_geolift.py", GEOLIFT_ROLLOUT_CONFIG)

    uplift_summary = json.loads(Path("results/uplift_synthetic_v1/summary.json").read_text())
    geolift_summary = json.loads(Path("results/geolift_rollout_v1/summary.json").read_text())

    results_dir = Path("results/full_pipeline_v1")
    results_dir.mkdir(parents=True, exist_ok=True)
    report_html = build_report(uplift_summary, ope_summary, geolift_summary, results_dir)
    (results_dir / "report.html").write_text(report_html, encoding="utf-8")

    combined = {
        "uplift_run": uplift_summary["run_id"],
        "ope_run": ope_summary["run_id"],
        "geolift_run": geolift_summary["run_id"],
        "derived_rollout_effect_pct": effect_pct,
        "geolift_recovered_effect_pct": geolift_summary["effect"]["point_estimate"],
        "geolift_placebo_p_value": geolift_summary["effect"]["metadata"]["placebo_p_value"],
    }
    (results_dir / "summary.json").write_text(json.dumps(combined, indent=2))

    print(f"\n=== full pipeline complete ===")
    print(json.dumps(combined, indent=2))
    print(f"\nOpen {results_dir / 'report.html'} in a browser for the combined report.")


if __name__ == "__main__":
    main()
