# Validation-aware ore-grade prediction

This repository contains the corrected, leakage-safe experiment pipeline for manuscript `MMEX-D-26-00799`.

## What changed

- The modeling unit is the original assayed interval. Discretizing an interval no longer creates additional observations.
- Unassayed drillhole segments are never assigned the global mean grade.
- Drillhole trajectories are reconstructed from collar and downhole survey records with minimum-curvature interpolation.
- Target-derived neighborhood features are excluded.
- Random-interval, held-out-drillhole, and spatial-block validation use the same features and model definitions.
- Hyperparameter selection, residual construction, variogram fitting, and fusion-weight estimation are confined to outer-training data.
- Residual kriging uses out-of-fold XGBoost residuals.
- Convex fusion weights are estimated from training-only out-of-fold predictions and constrained to `[0, 1]`.

## Models

1. Training-mean reference
2. Local ordinary kriging with fold-specific exponential variograms
3. XGBoost
4. XGBoost plus ordinary kriging of training-only out-of-fold residuals
5. Training-only convex fusion of XGBoost and ordinary kriging

## Data availability

The raw drilling, survey, lithology, and assay records contain confidential mine information and are not distributed in this repository. The pipeline expects the local files described in `data/README.md`. Only aggregate, non-sensitive validation summaries are intended for publication.

The public `results/` directory contains aggregate metrics and publication figures that do not expose interval coordinates, drillhole identifiers, or row-level grades. The former processed table is represented only by count-based audit statistics.

## Reproduce the analysis

```bash
python3 -m pip install -r requirements.txt
python3 run_analysis.py \
  --data-dir /path/to/local/raw_data \
  --output-dir /path/to/local/results
```

The default configuration is stored in `config/analysis.json`. Sensitive interval-level outputs are written under the supplied output directory and must not be committed.

Run the validation-safety tests with:

```bash
python3 -m pytest -q
```

## Main revised findings

| Validation task | XGBoost RMSE | Fusion RMSE | XGBoost R2 | Fusion R2 |
|---|---:|---:|---:|---:|
| Random interval | 5.206 | 5.204 | 0.387 | 0.387 |
| Held-out drillhole | 5.967 | 5.925 | 0.194 | 0.206 |
| Spatial block | 5.823 | 5.781 | 0.233 | 0.244 |

The small fusion advantage is descriptive: it is much smaller than fold-to-fold variability, and fusion relies predominantly on XGBoost. Ordinary kriging and out-of-fold residual kriging do not improve the strict-validation result for this deposit. See `VALIDATION_PROTOCOL.md` for the reusable methodological contract and `results/README.md` for the publication-safe outputs.

## Broad applicability

The reusable contribution is the validation contract rather than a deposit-specific winning algorithm. Any drillhole-based comparison can apply the same sequence: preserve the original assay support, reconstruct sample geometry without target-derived features, define validation groups from the intended prediction task, and fit every adaptive component exclusively inside the outer-training data.

## Repository status

Historical scripts are isolated under `legacy/` to preserve the publication audit trail. They do not implement the corrected validation design and must not be used to reproduce the revised manuscript.
