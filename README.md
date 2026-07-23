# Validation-aware ore-grade prediction

This repository contains the corrected, leakage-safe experiment pipeline for manuscript `MMEX-D-26-00799`, including the frozen Sijiaying/Jiajiabao one-shot blind replication and the seven-scenario failure-mechanism supplement.

## What changed

- The modeling unit is the original assayed interval. Discretizing an interval no longer creates additional observations.
- Unassayed drillhole segments are never assigned the global mean grade.
- Drillhole trajectories are reconstructed from collar and downhole survey records with minimum-curvature interpolation.
- Target-derived neighborhood features are excluded.
- Random-interval, held-out-drillhole, and spatial-block validation use the same features and model definitions.
- Hyperparameter selection, residual construction, variogram fitting, and fusion-weight estimation are confined to outer-training data.
- Residual kriging uses out-of-fold XGBoost residuals.
- Convex fusion weights are estimated from training-only out-of-fold predictions and constrained to `[0, 1]`.
- External replication freezes the source hash, code, contiguous blocks, blind block and learner before the blind target is opened.
- Target-bearing planning products are excluded unless authoritative, pre-assay lineage and prediction-time availability can be demonstrated.
- Failure-mechanism simulations retain every scenario, replicate and adverse result.

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

| Validation task | Contract | R2 | RMSE | Far-quartile R2 |
|---|---|---:|---:|---:|
| Qidashan strict spatial | Locked G4 CatBoost | 0.1783 | 9.265 | -0.0035 |
| Qidashan strict spatial | Cross-fitted residual kriging | -0.4455 | 12.289 | -1.0830 |
| Sijiaying development blocks | Geometry + interval lithology | 0.4613 | 5.151 | 0.3511 |
| Sijiaying one-shot blind block | Geometry + interval lithology | 0.1777 | 5.830 | 0.4087 |

The Sijiaying development increment passed the frozen +0.10 gate, but the one-shot blind result does not support a high-accuracy claim and is not reusable for model selection. Across seven failure mechanisms, cross-fitted residual kriging was not a robust rescue strategy. A deliberately nondeployable leakage control reached mean R2 = 0.5288, showing why target-bearing products require provenance and availability audits. See `SUPPLEMENTARY_VALIDATION.md`, `VALIDATION_PROTOCOL.md` and `results/README.md`.

## Broad applicability

The reusable contribution is the validation contract and the mapped boundary of useful information, rather than a deposit-specific winning algorithm. Any drillhole-based comparison can apply the same sequence: preserve the original assay support, reconstruct sample geometry without target-derived features, define validation groups from the intended prediction task, fit every adaptive component exclusively inside the outer-training data, freeze a genuinely unseen block, and report the blind result regardless of sign.

## Repository status

Historical scripts are isolated under `legacy/` to preserve the publication audit trail. They do not implement the corrected validation design and must not be used to reproduce the revised manuscript.
