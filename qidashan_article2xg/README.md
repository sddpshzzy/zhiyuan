# Qidashan Article 2XG reproducibility package

This directory accompanies the revised manuscript **“Geological representation under spatial extrapolation: leakage-controlled comparison of a CAD-constrained ore-domain reconstruction and complex downhole convolution for iron-grade prediction.”**

It contains the target-free TP-CVOCA-style mathematical feature extractor, generic validation-contract checks, synthetic tests, aggregate metrics, and non-sensitive figures. The public package is deliberately separated from the confidential mine data.

## Main result

Under frozen five-fold held-out physical-drillhole and contiguous spatial-block validation, neither the reconstructed CAD-solid features nor the TP-CVOCA-style features raised strict TFe prediction to high-accuracy levels. The best pooled drillhole R² was 0.273 (CatBoost, current fold-safe geology), while the best pooled spatial-block R² was 0.175 (CatBoost, solid plus training-selected complex features). These are audit results, not targets to be tuned upward after seeing the outer test folds.

## What is public

- `src/article2xg/cvoca_features.py`: target-free complex overlapping depth convolution using actual downhole depth lags.
- `src/article2xg/validation_contracts.py`: reusable leakage checks for grouped outer validation.
- `examples/synthetic_demo.py`: runnable synthetic example; it contains no mine observations.
- `tests/`: unit tests for row preservation, finite outputs, target exclusion, and group separation.
- `results/`: non-sensitive pooled metrics and quality-check summaries.
- `figures/`: figures derived only from aggregate metrics or feature names.

## What is not public

Raw assays, interval coordinates, drillhole surveys, lithology logs, CAD drawings, reconstructed meshes, interval-level predictions, and source-file hashes are excluded because they contain confidential production and geological information. The CAD-constrained solid described in the paper is a target-free reconstruction from audited official horizontal maps; it is **not** represented as a mine-approved native resource-model wireframe.

## CVOCA terminology

The implementation is a **TP-CVOCA-style mathematical adaptation** inspired by complex-valued overlapping convolution. It is not an optical accelerator implementation and is not claimed to reproduce the original hardware or network. The transform uses 15, 35, and 75 m Gaussian windows, actual midpoint-depth lags, and returns amplitude, phase sine/cosine, and effective support. Clipping, centering, scaling, and optional feature selection must be fitted within each outer-training fold.

## Quick start

```bash
python -m pip install -r qidashan_article2xg/requirements.txt
PYTHONPATH=qidashan_article2xg/src python qidashan_article2xg/examples/synthetic_demo.py
PYTHONPATH=qidashan_article2xg/src pytest -q qidashan_article2xg/tests
python qidashan_article2xg/plot_aggregate_results.py
```

The synthetic demo is expected to report 24 input rows and 21 complex features. It also verifies that changing the synthetic `TFe` column leaves the generated features unchanged.

## Experimental contracts

The five preregistered feature contracts are documented in [EXPERIMENT_CONTRACT.md](EXPERIMENT_CONTRACT.md). The outer-test data are not used for feature construction, scaling, selection, tuning, or geological-domain reconstruction.

## References for the adaptation

- Bai et al. (2025), *Nature Communications* 16, 292. https://doi.org/10.1038/s41467-024-55321-8
- Lu et al. (2026), *Scientific Reports* 16, 9059. https://doi.org/10.1038/s41598-026-39144-9

The repository-level license applies to this directory. The aggregate results are supplied for verification of the manuscript tables; they do not grant access to the underlying mine data.

