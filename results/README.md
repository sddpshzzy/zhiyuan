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
- `supplementary/sijiaying_external_metrics.csv`: development and one-shot blind aggregate metrics
- `supplementary/failure_simulation_summary.csv`: seven scenarios by deployable/nondeployable method
- `supplementary/three_dimensional_entity_eligibility.csv`: provenance-based entity decisions
- `supplementary/independent_validation.csv`: 27 independent recomputation checks

## Excluded

Raw workbooks and MDB files, clean interval tables, hole-level partitions, drillhole-level metrics, interval-level predictions, coordinate plots, and observed-versus-predicted row clouds are intentionally excluded because the mine data are confidential.
