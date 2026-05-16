# ArcticGRO Δ14C-POC prediction pipeline

This repository contains a reproducible Python workflow for predicting particulate organic carbon radiocarbon (Δ14C-POC) in six large Arctic rivers: Ob, Yenisey, Lena, Kolyma, Yukon, and Mackenzie.

The design separates:

1. **Between-river baseline differences** from HydroATLAS catchment attributes and REAL river-migration summaries.
2. **Within-river seasonal/daily tuning** from paired ArcticGRO water temperature, discharge, total suspended sediment, POC%, hydrograph stage, and daily hydrograph derivatives.
3. **Daily source-access terms** representing joint thaw–entrainment bank access and surface erosion/runoff access.

δ13C variables are intentionally excluded from Δ14C prediction and are reserved for source interpretation or mixing plots.

## Repository layout

```text
data/raw/                 # preferred location for raw input files
data/processed/           # cleaned and modeling-ready generated tables
scripts/                  # numbered workflow scripts
outputs/tables/           # inventories, audits, metrics, equations
outputs/figures/          # diagnostic plots
outputs/reports/          # final report in Markdown, DOCX, and PDF when dependencies are available
```

The scripts first look for raw files in `data/raw/` and then fall back to the repository root for compatibility with the current file placement.

## Python version and installation

Recommended: Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Raw input files

Place these files in `data/raw/` or leave them in the repository root:

- `ArcticGRO-compilation.csv`
- `Ob_20240322.xlsx`
- `Yenisey_20240322.xlsx`
- `Lena_20240322.xlsx`
- `Kolyma_20240322.xlsx`
- `Yukon_20240322.xlsx`
- `Mackenzie_20240322.xlsx`
- `ArcticGRO-HydroAtlas.xlsx`
- `REAL_reach.csv`
- `POC_Summ.csv`
- reference PDFs, if available: HydroATLAS technical documentation, Geyman et al. 2024, and Wang et al. 2024.

## Running the workflow

Run all steps:

```bash
python scripts/run_all.py
```

Or run staged phases:

```bash
python scripts/01_load_and_clean.py
python scripts/02_discharge_stage.py
python scripts/03_catchment_features.py
python scripts/04_derive_source_access.py
python scripts/05_model_validate.py
python scripts/06_figures.py
python scripts/07_report.py
```

## Expected outputs

Key generated outputs include:

- `outputs/tables/data_inventory.csv`: raw file inventory with row/column/header checks.
- `data/processed/arcticgro_clean.csv`: harmonized sample table.
- `data/processed/daily_discharge_climatology.csv`: smoothed day-of-year hydrograph variables.
- `outputs/tables/stage_boundaries.csv`: river-specific discharge-stage summaries.
- `data/processed/modeling_table.csv`: ArcticGRO samples merged with HydroATLAS and REAL features.
- `data/processed/modeling_table_features.csv`: modeling table with source-access indices.
- `outputs/tables/cleaned_modeling_table.csv`: user-facing cleaned modeling table with descriptive HydroATLAS names and mechanism indices.
- `outputs/tables/predictor_audit.csv`: predictor screening report documenting retained/removed variables, collinearity, and double-counting risk.
- `outputs/tables/model_metrics.csv`: leave-one-river-out and blocked river-year validation metrics.
- `data/processed/model_predictions.csv`: out-of-fold predictions and residuals.
- `outputs/tables/empirical_equation_coefficients.csv`: interpretable Ridge approximation coefficients with bootstrap intervals.
- `outputs/tables/feature_importance.csv`: random-forest permutation importance summary.
- `outputs/figures/*.png`: measured-vs-predicted, residual, seasonal, model comparison, and correlation diagnostics.
- `outputs/reports/final_ml_prediction_report.md`, plus DOCX/PDF when optional report dependencies are installed.

## Modeling and validation notes

- The target is `Delta14C_POC`, derived from `POC-14C`.
- `POC_pct` comes from ArcticGRO `POC` and is treated as OC% of suspended sediment.
- `log10_TSS` and `log10_Discharge` are used instead of raw TSS/discharge in ML tiers to avoid raw/log duplication.
- Deterministic concentration or flux products are not included when their components are retained.
- δ13C is not used as a predictor of Δ14C.
- Main transferability validation is leave-one-river-out cross-validation.
- Operational monitored-river validation is blocked by river-year.
- Random row-split validation is not reported as main evidence.
- Compared models include interpretable linear regression, polynomial Ridge, penalized-spline additive Ridge, Random Forest, ExtraTrees, Gradient Boosting, and a hierarchical component model.
- The hierarchical model writes catchment baseline, seasonal/daily tuning, and residual correction components in `data/processed/model_predictions.csv`.
- The empirical equation is an interpretable approximation; nonlinear and hierarchical validation models remain the prediction models.

## Data limitations

- The REAL migration table is not a full geometry dataset. Spatial clipping is therefore not performed; matching is audited in `outputs/tables/real_migration_match_audit.csv` and may use basin/Pfafstetter identifiers or a documented global fallback.
- The HydroATLAS-enhanced ArcticGRO table contains only six catchment rows, so catchment-only machine learning has limited degrees of freedom and should be interpreted as a structured baseline rather than a fully general global model.
- `POC_Summ.csv` is reserved for global context or optional pretraining/comparison and is not mixed into six-river validation by default.
