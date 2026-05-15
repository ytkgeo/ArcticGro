#!/usr/bin/env python
"""Fit tiered Δ14C-POC models using leave-one-river-out and blocked river-year validation."""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from common import PROCESSED, TABLES, ensure_dirs, TARGET

warnings.filterwarnings("ignore", category=UserWarning)

TIERS = {
    "M0_river_mean_baseline": ["River"],
    "M1_catchment_baseline": [
        "prm_pc_sse",
        "run_mm_syr",
        "pre_mm_syr",
        "ero_kh_sav",
        "cly_pc_sav",
        "slt_pc_sav",
        "soc_th_sav",
        "GSoil14C_mean",
        "GSoilRC_mean",
        "NPP_mean",
        "GPP_mean",
        "migration_positive_erosion_median",
        "migration_positive_erosion_p75",
        "migration_width_median",
        "migration_mean_sediment_discharge_median",
    ],
    "M2_plus_daily_hydrograph": [
        "prm_pc_sse",
        "run_mm_syr",
        "pre_mm_syr",
        "ero_kh_sav",
        "cly_pc_sav",
        "slt_pc_sav",
        "soc_th_sav",
        "GSoil14C_mean",
        "GSoilRC_mean",
        "NPP_mean",
        "GPP_mean",
        "migration_positive_erosion_median",
        "migration_positive_erosion_p75",
        "migration_width_median",
        "migration_mean_sediment_discharge_median",
        "Temp",
        "Qstar",
        "dQdt",
        "rising_limb",
        "falling_limb",
        "discharge_stage",
    ],
    "M3_plus_TSS_POC": [
        "prm_pc_sse",
        "run_mm_syr",
        "pre_mm_syr",
        "ero_kh_sav",
        "cly_pc_sav",
        "slt_pc_sav",
        "soc_th_sav",
        "GSoil14C_mean",
        "GSoilRC_mean",
        "NPP_mean",
        "GPP_mean",
        "migration_positive_erosion_median",
        "migration_positive_erosion_p75",
        "migration_width_median",
        "migration_mean_sediment_discharge_median",
        "Temp",
        "Qstar",
        "dQdt",
        "rising_limb",
        "falling_limb",
        "discharge_stage",
        "log10_TSS",
        "POC_pct",
        "log10_Discharge",
    ],
    "M4_plus_source_access": [
        "prm_pc_sse",
        "run_mm_syr",
        "pre_mm_syr",
        "ero_kh_sav",
        "cly_pc_sav",
        "slt_pc_sav",
        "soc_th_sav",
        "GSoil14C_mean",
        "GSoilRC_mean",
        "NPP_mean",
        "GPP_mean",
        "migration_positive_erosion_median",
        "migration_positive_erosion_p75",
        "migration_width_median",
        "migration_mean_sediment_discharge_median",
        "Temp",
        "Qstar",
        "dQdt",
        "rising_limb",
        "falling_limb",
        "discharge_stage",
        "log10_TSS",
        "POC_pct",
        "log10_Discharge",
        "thaw_potential_scaled",
        "entrainment_potential",
        "bank_access_gate",
        "old_bank_access_index",
        "surface_erosion_index",
    ],
}

ALGORITHMS = {
    "ridge": Ridge(alpha=10.0),
    "polynomial_ridge": Pipeline(
        [
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("ridge", Ridge(alpha=25.0)),
        ]
    ),
    "random_forest": RandomForestRegressor(
        n_estimators=300, min_samples_leaf=3, random_state=42
    ),
    "extra_trees": ExtraTreesRegressor(
        n_estimators=300, min_samples_leaf=3, random_state=42
    ),
    "gradient_boosting": GradientBoostingRegressor(random_state=42),
    "svr": SVR(C=10.0, epsilon=20.0),
}


def available_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    return [f for f in features if f in df.columns]


def build_pipeline(df: pd.DataFrame, features: list[str], estimator):
    cat = [
        c
        for c in features
        if df[c].dtype == object or c in {"River", "discharge_stage"}
    ]
    num = [c for c in features if c not in cat]
    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat,
            ),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocess", pre), ("model", clone(estimator))])


def metrics(y, pred) -> dict:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = pred - y
    out = {
        "n": len(y),
        "r2": r2_score(y, pred) if len(np.unique(y)) > 1 else np.nan,
        "rmse": mean_squared_error(y, pred) ** 0.5,
        "mae": mean_absolute_error(y, pred),
        "bias": float(np.mean(err)),
        "residual_std": float(np.std(err, ddof=1)) if len(err) > 1 else np.nan,
        "median_error": float(np.median(err)),
        "p95_absolute_error": float(np.percentile(np.abs(err), 95)),
        "overprediction_rate": float(np.mean(err > 0)),
    }
    if len(y) > 2 and np.std(pred) > 0 and np.std(y) > 0:
        slope, intercept = np.polyfit(pred, y, 1)
        out.update(
            {
                "calibration_slope": slope,
                "calibration_intercept": intercept,
                "pearson_r": stats.pearsonr(y, pred).statistic,
                "spearman_rho": stats.spearmanr(y, pred).statistic,
            }
        )
    else:
        out.update(
            {
                "calibration_slope": np.nan,
                "calibration_intercept": np.nan,
                "pearson_r": np.nan,
                "spearman_rho": np.nan,
            }
        )
    return out


def river_mean_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    means = train.groupby("River")[TARGET].mean()
    global_mean = train[TARGET].mean()
    return test["River"].map(means).fillna(global_mean).to_numpy()


def evaluate_scheme(df: pd.DataFrame, scheme: str) -> tuple[list[dict], pd.DataFrame]:
    rows, preds = [], []
    if scheme == "leave_one_river_out":
        splits = [
            (f"heldout_{r}", df["River"] != r, df["River"] == r)
            for r in sorted(df["River"].dropna().unique())
        ]
    else:
        splits = []
        for r, sub in df.groupby("River"):
            years = sorted(sub["Year"].dropna().unique())
            if len(years) < 2:
                continue
            for yr in years:
                test_idx = (df["River"] == r) & (df["Year"] == yr)
                train_idx = ~test_idx & df[TARGET].notna()
                if test_idx.sum() > 0 and train_idx.sum() > 5:
                    splits.append((f"{r}_{int(yr)}", train_idx, test_idx))
    for split_name, train_mask, test_mask in splits:
        train, test = df[train_mask].dropna(subset=[TARGET]), df[test_mask].dropna(
            subset=[TARGET]
        )
        if train.empty or test.empty:
            continue
        for tier, feats in TIERS.items():
            feats = available_features(df, feats)
            if tier.startswith("M0"):
                pred = river_mean_predict(train, test)
                rows.append(
                    {
                        "validation_scheme": scheme,
                        "split": split_name,
                        "model_tier": tier,
                        "algorithm": "river_mean",
                        **metrics(test[TARGET], pred),
                    }
                )
                preds.append(
                    pd.DataFrame(
                        {
                            "validation_scheme": scheme,
                            "split": split_name,
                            "model_tier": tier,
                            "algorithm": "river_mean",
                            "River": test["River"].values,
                            "Year": test["Year"].values,
                            "observed": test[TARGET].values,
                            "predicted": pred,
                        }
                    )
                )
                continue
            for alg, est in ALGORITHMS.items():
                pipe = build_pipeline(df, feats, est)
                try:
                    pipe.fit(train[feats], train[TARGET])
                    pred = pipe.predict(test[feats])
                    rows.append(
                        {
                            "validation_scheme": scheme,
                            "split": split_name,
                            "model_tier": tier,
                            "algorithm": alg,
                            **metrics(test[TARGET], pred),
                        }
                    )
                    preds.append(
                        pd.DataFrame(
                            {
                                "validation_scheme": scheme,
                                "split": split_name,
                                "model_tier": tier,
                                "algorithm": alg,
                                "River": test["River"].values,
                                "Year": test["Year"].values,
                                "discharge_stage": test.get(
                                    "discharge_stage", pd.Series(index=test.index)
                                ).values,
                                "TSS": test.get(
                                    "TSS", pd.Series(index=test.index)
                                ).values,
                                "Qstar": test.get(
                                    "Qstar", pd.Series(index=test.index)
                                ).values,
                                "Julian_Day": test.get(
                                    "Julian_Day", pd.Series(index=test.index)
                                ).values,
                                "observed": test[TARGET].values,
                                "predicted": pred,
                            }
                        )
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "validation_scheme": scheme,
                            "split": split_name,
                            "model_tier": tier,
                            "algorithm": alg,
                            "n": len(test),
                            "error": str(exc),
                        }
                    )
    return rows, pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()


def empirical_equation(df: pd.DataFrame) -> None:
    feats = available_features(df, TIERS["M4_plus_source_access"])
    num_feats = [f for f in feats if pd.api.types.is_numeric_dtype(df[f])]
    d = df.dropna(subset=[TARGET]).copy()
    pipe = build_pipeline(d, num_feats, Ridge(alpha=10.0))
    pipe.fit(d[num_feats], d[TARGET])
    model = pipe.named_steps["model"]
    coefs = model.coef_ if hasattr(model, "coef_") else np.full(len(num_feats), np.nan)
    pd.DataFrame(
        {"feature": num_feats, "standardized_coefficient": coefs[: len(num_feats)]}
    ).to_csv(TABLES / "empirical_equation_standardized_coefficients.csv", index=False)
    equation = "Delta14C_POC_z ≈ " + " + ".join(
        [
            f"({coef:.3g} × {feat}_z)"
            for feat, coef in zip(num_feats, coefs[: len(num_feats)])
        ]
    )
    (TABLES / "empirical_equation.txt").write_text(
        equation
        + "\nNonlinear ML models are the prediction models; this Ridge equation is an interpretable approximation.\n"
    )


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(PROCESSED / "modeling_table_features.csv")
    df = df[df[TARGET].notna()].copy()
    all_rows, all_preds = [], []
    for scheme in ["leave_one_river_out", "blocked_river_year"]:
        rows, preds = evaluate_scheme(df, scheme)
        all_rows.extend(rows)
        all_preds.append(preds)
    pd.DataFrame(all_rows).to_csv(TABLES / "model_metrics.csv", index=False)
    pred_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    if not pred_df.empty:
        pred_df["residual"] = pred_df["predicted"] - pred_df["observed"]
        pred_df.to_csv(PROCESSED / "model_predictions.csv", index=False)
    empirical_equation(df)
    print(f"Wrote {TABLES/'model_metrics.csv'} and {PROCESSED/'model_predictions.csv'}")


if __name__ == "__main__":
    main()
