# Leakage-controlled validation protocol

This protocol separates evidence construction from algorithm comparison. It is designed for drillhole-based grade prediction but can be adapted to other spatial sampling programs.

## 1. Preserve observational support

- Treat each laboratory assay interval as one observation.
- Do not count discretized child points as independent labels.
- Do not replace unassayed locations with a global or local mean grade.
- Retain interval length for support-aware diagnostics, not as a default predictor.

## 2. Reconstruct predictors without the target

- Reconstruct three-dimensional interval midpoints from collar and downhole survey records.
- Join lithology or geological attributes by interval containment.
- Exclude target-neighborhood means, grade-defined domains, grade-informed components, and any transformation fitted before the outer split.

## 3. Match validation to the prediction task

- Random-interval folds measure interpolation among sampled drillholes.
- Held-out-drillhole folds measure transfer to unseen holes.
- Contiguous spatial blocks measure transfer toward unsampled sectors.
- Keep all descendants of one physical assay interval in the same fold.

## 4. Nest every adaptive operation

For each outer fold, fit preprocessing, hyperparameters, variograms, residual models, and ensemble weights using outer-training data only. Residual kriging must use out-of-fold residuals from the outer-training subset. Fusion weights must use paired inner out-of-fold component predictions rather than outer-test labels.

## 5. Report more than a pooled winner

Report pooled out-of-fold RMSE, MAE, R2, bias, calibration, length-weighted errors, fold variability, distance-to-training diagnostics, and training-defined upper-tail performance. A method should not be described as superior when its gain is smaller than fold variability or disappears under the validation task relevant to use.

## 6. Confidentiality boundary

Public artifacts may include code, configuration, aggregate metrics, and aggregate figures. Keep raw collar coordinates, drillhole identifiers, interval-level grades, and interval-level predictions outside version control unless the data owner explicitly authorizes release.
