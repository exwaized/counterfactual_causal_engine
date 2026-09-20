# counterfactual — a causal decision engine

Three causal-inference modules that chain into one pipeline:

```
UPLIFT (individual)  ->  OPE (policy vetting)  ->  GEOLIFT (aggregate measurement)
"who to target"            "is this targeting        "did the real rollout
                             policy worth an            actually move the
                             A/B test slot?"             needle?"
```

- **`uplift/`** — CATE estimation (T-learner, X-learner) from randomized campaign
  data, turned into a budget-constrained targeting policy. Evaluated with
  Qini curves / AUUC and an IPW policy-value estimate.
- **`ope/`** — off-policy evaluation: estimate whether a new targeting policy
  would beat the current production policy, using only logged data, before
  spending a live A/B test on it. IPS, SNIPS, and doubly-robust estimators,
  validated against a known ground-truth policy value.
- **`geolift/`** — synthetic control for measuring a campaign's aggregate
  effect when individual-level randomization isn't possible (e.g. a
  geo-level rollout). Validated against a synthetic panel with a known
  injected effect, with placebo-based significance testing (Abadie, Diamond
  & Hainmueller 2010's rank permutation test).

## Status

All four phases are built. Each module (`uplift/`, `ope/`, `geolift/`) is
independently runnable and tested against a known ground truth, and
`pipelines/04_full_pipeline.py` chains all three: trains a targeting policy,
vets it offline against a different production policy's logs, derives the
expected rollout lift from that vetting (not an arbitrary number), and
measures a simulated geo-level rollout against it -- ending in one combined
HTML report.

## Setup

```
pip install -e .
```

## Run phase 1

```
python pipelines/01_train_uplift_policy.py configs/uplift_synthetic.yaml
```

Writes `results/uplift_synthetic_v1/{summary.json, qini_curve.png, policy.joblib}`.

To run on the real Hillstrom or Criteo uplift datasets instead: drop the CSV
into `data/raw/`, copy `configs/uplift_hillstrom.yaml`, and point `data_path`
at it (column-mapping notes are in that file).

## Run phase 2

```
python pipelines/03_measure_rollout_geolift.py configs/geolift_synthetic.yaml
```

Writes `results/geolift_synthetic_v1/{summary.json, geo_trajectory.png}` —
the estimated % lift, its placebo-test p-value, and the recovery error
against the config's known `true_effect_pct`. A single run's point estimate
is noisy by nature (see `pre_period_rmspe` in the summary — a high value
means don't trust that run's extrapolation); `tests/` checks the estimator
is unbiased on average across seeds, not exact on any one run.

## Run phase 3

Run phase 1 first (it saves the policy phase 3 vets by default), then:

```
python pipelines/02_vet_policy_with_ope.py configs/ope_synthetic.yaml
```

Writes `results/ope_synthetic_v1/{summary.json, ope_comparison.png}` —
IPS/SNIPS/DR estimates (with bootstrap 95% CIs) of the phase-1 policy's
value under a *different*, non-uniform logging policy's data, compared
against the true value (known because the data is synthetic) and against
the logging policy's own true value, plus a verdict on whether the new
policy looks worth an A/B test. Point `target_policy` in the config at
`treat_all`/`treat_none`/`random` to vet a baseline instead.

## Run the full chained pipeline

```
python pipelines/04_full_pipeline.py
```

Runs phases 1-3 in order via subprocess, derives the geo-rollout's injected
effect from OPE's doubly-robust estimate vs. the current production
policy's empirical value (clipped to &plusmn;50%, written to
`configs/geolift_rollout.yaml` so the derivation is inspectable), then
measures that simulated rollout with synthetic control. Writes
`results/full_pipeline_v1/{summary.json, report.html}` -- open `report.html`
in a browser for the combined write-up with all three plots and a bottom-line
table.

## Tests

```
pytest
```

Each module's tests check recovery of a known ground truth, not just that
the code runs: `test_uplift_qini_beats_random.py` (AUUC > 0, predicted CATE
correlates with the synthetic generator's true CATE), `test_synthetic_control_recovers_known_effect.py`
(estimated lift is unbiased on average against the injected `true_effect_pct`),
`test_ope_estimator_bias.py` (IPS/SNIPS/DR all land within 5pp of the true
policy value, computed directly from the same ground-truth reward model
uplift's data uses).

## Layout

```
configs/          one YAML per experiment run (dataset, method, hyperparams)
data/{raw,processed,synthetic}/
src/causal_engine/
  common/          shared EffectEstimate result type, config loading, plotting
  uplift/          models.py (T/X-learner), policy.py, evaluate.py (Qini/AUUC)
  ope/             estimators.py (IPS/SNIPS/DR), policies.py, simulate.py (logged bandit data)
  geolift/         synthetic_control.py (FISTA QP solver), inference.py (placebo tests), simulate.py
pipelines/         01 uplift, 02 ope (vets 01's policy), 03 geolift, 04 chains 01-03 + report.html
results/<run_id>/  auto-saved JSON + plots per run
tests/
```
