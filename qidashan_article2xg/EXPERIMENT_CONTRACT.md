# Frozen experimental contract

## Prediction unit and outer splits

- One raw assay interval is one prediction row.
- Aliased or duplicate collar names belonging to the same trajectory are grouped as one physical drillhole.
- `drillhole`: all intervals from a held-out physical drillhole remain outside training.
- `spatial_block`: contiguous horizontal drillhole regions are held out.
- The same five outer folds are reused for every learner and feature contract.

## Prohibited information

TFe, FeO, SFe, magnetic susceptibility, ratios derived from assays, target-derived neighbourhood summaries, and target-derived domain labels are not predictors. Exact lithology from a held-out physical drillhole is masked before lithology propagation in the new-hole task.

All imputing, clipping, centering, scaling, model selection, and mutual-information feature selection are fitted on outer-training data only. Outer-test metrics are used once for final comparison.

## Feature contracts

| Code | Contract | Increment relative to G0 |
|---|---|---|
| G0 | Current fold-safe geology | Geometry, horizontal/section CAD and training-only propagated lithology |
| G1 | G0 + CAD solid | 19 target-free reconstructed-solid features |
| G2 | G0 + current-channel CVOCA | Complex amplitude, phase and support from current target-free channels |
| G3 | G0 + solid + all CVOCA | Complete representation increment |
| G4 | G0 + solid + selected CVOCA | 32 complex features independently selected in each outer-training fold |

## Primary and diagnostic metrics

- Primary: pooled outer-fold R².
- Error: RMSE, MAE, and mean bias.
- Extrapolation: R² in the farthest training-distance quartile.
- Tail behaviour: bias above the outer-training 90th grade percentile.
- Geological support: performance reported inside and outside the audited CAD-solid coverage.

No minimum R² threshold is imposed on the observed results. In particular, R²=0.8 is a future information-acquisition target, not a parameter of this experiment.

