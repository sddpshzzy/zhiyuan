# Frozen external validation and failure-mechanism supplement

This supplement implements the manuscript's final validation layer without publishing confidential mine coordinates, row-level assays, or the source MDB files.

## One-shot Sijiaying/Jiajiabao experiment

The source schema, source hash, code, five contiguous spatial blocks, selected blind block and learner settings were frozen before blind grades were read. The four development blocks contained 45 physical holes and 1,819 intervals; the blind block contained 21 physical holes and 447 intervals.

Original interval lithology increased development R2 from 0.0567 to 0.4613 (Delta R2 = +0.4046), passing the predeclared +0.10 gate. The blind target was then opened exactly once without refitting. Geometry plus lithology achieved R2 = 0.1777, RMSE = 5.830% TFe and farthest-quartile R2 = 0.4087. The grouped-bootstrap increment was positive relative to IDW but inconclusive relative to a training-mean baseline. The opened blind block must not be reused for model selection.

## Failure-mechanism simulation

Seven frozen scenarios were run with 30 replicates each: stationary covariance, anisotropy misspecification, domain mean shift, fault offset, support mismatch, sparse/far extrapolation and target leakage. Every scenario, replicate and method is retained.

Ordinary kriging was useful only when covariance was transferable or when the simulated discontinuity remained recoverable from training support. Cross-fitted residual kriging was worse than coordinate CatBoost in all seven scenarios, with a mean Delta R2 of -0.4366. The deliberately nondeployable leakage control reached mean R2 = 0.5288 and was positive in all 30 repetitions, demonstrating how target-bearing information can manufacture apparent accuracy.

## Public files

- `prepare_sijiaying_blind_partition.py`: constructs the target-hidden partition and freezes code hashes.
- `run_sijiaying_blind_external.py`: development gate and one-shot blind release.
- `run_failure_mechanism_simulation.py`: seven-scenario fixed-replicate simulation.
- `validate_supplementary_experiments.py`: independent numerical and leakage checks.
- `src/sijiaying_external.py`: trajectory, assay-support and interval-lithology alignment.
- `src/local_residual_kriging.py`: local variogram and ordinary-kriging implementation.
- `results/supplementary/`: aggregate publication-safe metrics only.

## Private-data binding

1. Install `mdbtools` and the Python requirements.
2. Copy `config/experiment.example.json` to `config/experiment.json` and insert the private Qidashan MDB path.
3. Copy `config/sijiaying_source_selection_freeze_20260723.example.json` to `config/sijiaying_source_selection_freeze_20260723.json`, insert the private Sijiaying MDB path and verify its SHA-256. The destination is ignored by Git.
4. For an audited rerun, generate the partition before opening any target and preserve the generated sentinel and manifests.

The source databases, hole-level partitions, coordinate-bearing tables and row-level observed grades are intentionally excluded from Git.
