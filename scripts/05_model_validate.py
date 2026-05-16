#!/usr/bin/env python
"""Fit tiered Δ14C-POC models with rigorous river-blocked validation."""

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
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    PolynomialFeatures,
    SplineTransformer,
    StandardScaler,
)

from common import PROCESSED, TABLES, TARGET, descriptive_name, ensure_dirs

warnings.filterwarnings("ignore", category=UserWarning)

CATCHMENT_FEATURES = [
    "catchment_old_carbon_baseline_index",
    "prm_pc_sse",
    "wetland_extent_total",
    "lithology_old_sedimentary_fraction",
    "soc_th_sav",
    "GSoil14C_mean",
    "GSoilRC_mean",
    "ero_kh_sav",
    "slp_dg_sav",
    "sgr_dk_sav",
    "migration_positive_erosion_median",
    "migration_positive_erosion_p75",
    "migration_width_median",
    "migration_mean_sediment_discharge_median",
]
DAILY_FEATURES = [
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
]

TIERS = {
    "M0_river_mean_baseline": ["River"],
    "M1_catchment_baseline": CATCHMENT_FEATURES,
    "M2_plus_daily_hydrograph": CATCHMENT_FEATURES
    + ["Temp", "Qstar", "dQdt", "rising_limb", "falling_limb", "discharge_stage"],
    "M3_plus_TSS_POC": CATCHMENT_FEATURES
    + [
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
    "M4_plus_source_access": CATCHMENT_FEATURES + DAILY_FEATURES,
}

ALGORITHMS = {
    "interpretable_linear_regression": LinearRegression(),
    "polynomial_ridge": Pipeline(
        [
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("ridge", Ridge(alpha=25.0)),
        ]
    ),
    "penalized_spline_additive_ridge": Pipeline(
        [
            ("splines", SplineTransformer(n_knots=5, degree=3, include_bias=False)),
            ("ridge", Ridge(alpha=10.0)),
        ]
    ),
    "random_forest": RandomForestRegressor(
        n_estimators=400, min_samples_leaf=3, random_state=42
    ),
    "extra_trees": ExtraTreesRegressor(
        n_estimators=400, min_samples_leaf=3, random_state=42
    ),
    "gradient_boosting": GradientBoostingRegressor(random_state=42),
}


def available_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    return [f for f in features if f in df.columns]


def feature_types(df: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    cat = [
        c
        for c in features
        if df[c].dtype == object or c in {"River", "discharge_stage"}
    ]
    num = [c for c in features if c not in cat]
    return num, cat


def build_pipeline(df: pd.DataFrame, features: list[str], estimator) -> Pipeline:
    num, cat = feature_types(df, features)
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
        "median_error": float(np.median(err)),
        "residual_std": float(np.std(err, ddof=1)) if len(err) > 1 else np.nan,
        "p95_absolute_error": float(np.percentile(np.abs(err), 95)),
        "overprediction_rate": float(np.mean(err > 0)),
        "underprediction_rate": float(np.mean(err < 0)),
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


def river_mean_predict(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    means = train.groupby("River")[TARGET].mean()
    global_mean = train[TARGET].mean()
    pred = test["River"].map(means).fillna(global_mean).to_numpy()
    return pd.DataFrame(
        {
            "catchment_baseline_component": pred,
            "seasonal_daily_tuning_component": 0.0,
            "residual_correction_component": 0.0,
            "predicted": pred,
        },
        index=test.index,
    )


def hierarchical_predict(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Δ14C = catchment baseline + daily tuning + river residual calibration."""
    catchment = available_features(train, CATCHMENT_FEATURES)
    daily = available_features(train, DAILY_FEATURES)
    baseline = build_pipeline(train, catchment, Ridge(alpha=10.0))
    baseline.fit(train[catchment], train[TARGET])
    train_base = baseline.predict(train[catchment])
    test_base = baseline.predict(test[catchment])

    residual_target = train[TARGET].to_numpy() - train_base
    tuning = build_pipeline(train, daily, Ridge(alpha=10.0))
    tuning.fit(train[daily], residual_target)
    train_tuning = tuning.predict(train[daily])
    test_tuning = tuning.predict(test[daily])

    train_remaining = train[TARGET].to_numpy() - train_base - train_tuning
    river_resid = (
        pd.Series(train_remaining, index=train.index).groupby(train["River"]).mean()
    )
    correction = test["River"].map(river_resid).fillna(0.0).to_numpy()
    pred = test_base + test_tuning + correction
    return pd.DataFrame(
        {
            "catchment_baseline_component": test_base,
            "seasonal_daily_tuning_component": test_tuning,
            "residual_correction_component": correction,
            "predicted": pred,
        },
        index=test.index,
    )


def split_masks(
    df: pd.DataFrame, scheme: str
) -> list[tuple[str, pd.Series, pd.Series]]:
    if scheme == "leave_one_river_out":
        return [
            (f"heldout_{r}", df["River"] != r, df["River"] == r)
            for r in sorted(df["River"].dropna().unique())
        ]
    splits = []
    for river, sub in df.groupby("River"):
        years = sorted(sub["Year"].dropna().unique())
        if len(years) < 2:
            continue
        for year in years:
            test_idx = (df["River"] == river) & (df["Year"] == year)
            train_idx = ~test_idx & df[TARGET].notna()
            if test_idx.sum() > 0 and train_idx.sum() > 5:
                splits.append((f"{river}_{int(year)}", train_idx, test_idx))
    return splits


def prediction_frame(
    scheme: str,
    split_name: str,
    tier: str,
    algorithm: str,
    test: pd.DataFrame,
    components: pd.DataFrame,
) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "validation_scheme": scheme,
            "split": split_name,
            "model_tier": tier,
            "algorithm": algorithm,
            "River": test["River"].values,
            "Year": test["Year"].values,
            "Date_parsed": test.get("Date_parsed", pd.Series(index=test.index)).values,
            "discharge_stage": test.get(
                "discharge_stage", pd.Series(index=test.index)
            ).values,
            "TSS": test.get("TSS", pd.Series(index=test.index)).values,
            "Discharge": test.get("Discharge", pd.Series(index=test.index)).values,
            "Temp": test.get("Temp", pd.Series(index=test.index)).values,
            "Qstar": test.get("Qstar", pd.Series(index=test.index)).values,
            "Julian_Day": test.get("Julian_Day", pd.Series(index=test.index)).values,
            "observed": test[TARGET].values,
        }
    )
    return pd.concat(
        [base.reset_index(drop=True), components.reset_index(drop=True)], axis=1
    )


def evaluate_scheme(df: pd.DataFrame, scheme: str) -> tuple[list[dict], pd.DataFrame]:
    rows, preds = [], []
    for split_name, train_mask, test_mask in split_masks(df, scheme):
        train = df[train_mask].dropna(subset=[TARGET]).copy()
        test = df[test_mask].dropna(subset=[TARGET]).copy()
        if train.empty or test.empty:
            continue
        for tier, feats in TIERS.items():
            feats = available_features(df, feats)
            if tier.startswith("M0"):
                components = river_mean_predict(train, test)
                rows.append(
                    {
                        "validation_scheme": scheme,
                        "split": split_name,
                        "model_tier": tier,
                        "algorithm": "river_mean",
                        **metrics(test[TARGET], components["predicted"]),
                    }
                )
                preds.append(
                    prediction_frame(
                        scheme, split_name, tier, "river_mean", test, components
                    )
                )
                continue
            for alg, est in ALGORITHMS.items():
                pipe = build_pipeline(df, feats, est)
                try:
                    pipe.fit(train[feats], train[TARGET])
                    pred = pipe.predict(test[feats])
                    components = pd.DataFrame(
                        {
                            "catchment_baseline_component": np.nan,
                            "seasonal_daily_tuning_component": np.nan,
                            "residual_correction_component": np.nan,
                            "predicted": pred,
                        },
                        index=test.index,
                    )
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
                        prediction_frame(
                            scheme, split_name, tier, alg, test, components
                        )
                    )
                except (ValueError, TypeError, FloatingPointError) as exc:
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
        components = hierarchical_predict(train, test)
        rows.append(
            {
                "validation_scheme": scheme,
                "split": split_name,
                "model_tier": "M5_hierarchical_operational",
                "algorithm": "hierarchical_ridge_components",
                **metrics(test[TARGET], components["predicted"]),
            }
        )
        preds.append(
            prediction_frame(
                scheme,
                split_name,
                "M5_hierarchical_operational",
                "hierarchical_ridge_components",
                test,
                components,
            )
        )
    return rows, pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()


def best_model_key(metrics_df: pd.DataFrame) -> tuple[str, str]:
    valid = metrics_df[metrics_df["rmse"].notna()].copy()
    valid = valid[valid["validation_scheme"] == "blocked_river_year"]
    if valid.empty:
        valid = metrics_df[metrics_df["rmse"].notna()].copy()
    grouped = (
        valid.groupby(["model_tier", "algorithm"], as_index=False)["rmse"]
        .mean()
        .sort_values("rmse")
    )
    if grouped.empty:
        return "M5_hierarchical_operational", "hierarchical_ridge_components"
    row = grouped.iloc[0]
    return str(row["model_tier"]), str(row["algorithm"])


def empirical_equation(df: pd.DataFrame) -> None:
    feats = available_features(df, CATCHMENT_FEATURES + DAILY_FEATURES)
    num_feats = [f for f in feats if pd.api.types.is_numeric_dtype(df[f])]
    d = df.dropna(subset=[TARGET]).copy()
    x = d[num_feats].copy()
    y = d[TARGET].copy()
    pipe = build_pipeline(d, num_feats, Ridge(alpha=10.0))
    pipe.fit(x, y)
    model = pipe.named_steps["model"]
    coefs = model.coef_ if hasattr(model, "coef_") else np.full(len(num_feats), np.nan)
    intercept = model.intercept_ if hasattr(model, "intercept_") else np.nan
    x_imputed = x.fillna(x.median(numeric_only=True))
    means = x_imputed.mean(axis=0).to_numpy()
    stds = x_imputed.std(axis=0, ddof=0).replace(0, np.nan).to_numpy()
    standardized_coefs = coefs[: len(num_feats)]
    unstandardized_coefs = standardized_coefs / stds
    unstandardized_intercept = intercept - np.nansum(standardized_coefs * means / stds)
    coef = pd.DataFrame(
        {
            "feature": num_feats,
            "descriptive_name": [descriptive_name(f) for f in num_feats],
            "standardized_coefficient": standardized_coefs,
            "unstandardized_coefficient_approx": unstandardized_coefs,
        }
    )
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(200):
        idx = rng.choice(d.index.to_numpy(), size=len(d), replace=True)
        bd = d.loc[idx]
        bpipe = build_pipeline(bd, num_feats, Ridge(alpha=10.0))
        bpipe.fit(bd[num_feats], bd[TARGET])
        boot.append(bpipe.named_steps["model"].coef_[: len(num_feats)])
    if boot:
        arr = np.vstack(boot)
        coef["bootstrap_ci_low"] = np.nanpercentile(arr, 2.5, axis=0)
        coef["bootstrap_ci_high"] = np.nanpercentile(arr, 97.5, axis=0)
    coef.to_csv(TABLES / "empirical_equation_coefficients.csv", index=False)
    equation = "Delta14C_POC_z ≈ " + " + ".join(
        [
            f"({coef_val:.3g} × {feat}_z)"
            for feat, coef_val in zip(num_feats, coefs[: len(num_feats)])
        ]
    )
    (TABLES / "empirical_equation.txt").write_text(
        f"Standardized equation:\n{equation}\n\n"
        f"Approximate unstandardized intercept: {unstandardized_intercept:.6g}\n"
        "Approximate unstandardized coefficients are provided in empirical_equation_coefficients.csv.\n\n"
        "This empirical equation is an interpretable approximation and is expected to be less accurate than the best nonlinear or hierarchical validation model.\n"
    )


def write_feature_importance(df: pd.DataFrame) -> None:
    d = df.dropna(subset=[TARGET]).copy()
    feats = available_features(d, TIERS["M4_plus_source_access"])
    model = build_pipeline(
        d, feats, RandomForestRegressor(n_estimators=400, random_state=42)
    )
    model.fit(d[feats], d[TARGET])
    result = permutation_importance(
        model,
        d[feats],
        d[TARGET],
        n_repeats=20,
        random_state=42,
        scoring="neg_root_mean_squared_error",
    )
    pd.DataFrame(
        {
            "feature": feats,
            "descriptive_name": [descriptive_name(f) for f in feats],
            "importance_mean": result.importances_mean[: len(feats)],
            "importance_std": result.importances_std[: len(feats)],
        }
    ).sort_values("importance_mean", ascending=False).to_csv(
        TABLES / "feature_importance.csv", index=False
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
    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(TABLES / "model_metrics.csv", index=False)
    pred_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    if not pred_df.empty:
        pred_df["residual"] = pred_df["predicted"] - pred_df["observed"]
        pred_df.to_csv(PROCESSED / "model_predictions.csv", index=False)
        tier, algorithm = best_model_key(metrics_df)
        pred_df[
            (pred_df["model_tier"] == tier) & (pred_df["algorithm"] == algorithm)
        ].to_csv(PROCESSED / "best_model_predictions.csv", index=False)
    empirical_equation(df)
    write_feature_importance(df)
    print(f"Wrote {TABLES/'model_metrics.csv'} and {PROCESSED/'model_predictions.csv'}")


if __name__ == "__main__":
    main()
