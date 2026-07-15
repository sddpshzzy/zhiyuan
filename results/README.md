# Publication-safe results

This directory contains aggregate outputs from the corrected experiment.

## Included

- `aggregate/data_audit.csv`: support-correction checks and summary statistics
- `aggregate/legacy_data_audit_public.csv`: count-only audit of the former processed table
- `aggregate/pooled_metrics.csv`: pooled out-of-fold performance
- `aggregate/fold_metrics.csv`: fold-level performance without hole identifiers or coordinates
- `aggregate/feature_ablation_pooled_metrics.csv`: strict-validation feature-group ablation
- `aggregate/fusion_weights.csv`: training-only fusion weights
- `aggregate/variogram_parameters.csv`: fold-selected aggregate variogram settings
- `aggregate/timings.csv`: fold execution times
- `figures/`: aggregate comparison, ablation, distance, and variability figures

## Excluded

Raw workbooks, source manifests, clean interval tables, drillhole-level metrics, interval-level predictions, coordinate plots, and observed-versus-predicted row clouds are intentionally excluded because the mine data are confidential.
