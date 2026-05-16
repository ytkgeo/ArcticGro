#!/usr/bin/env python
"""Generate publication-quality diagnostics for the ArcticGRO Δ14C workflow."""

from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from common import FIGURES, PROCESSED, TABLES, TARGET, ensure_dirs

sns.set_theme(style="whitegrid", context="paper")


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=250, bbox_inches="tight")
    plt.close()


def best_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    metrics = pd.read_csv(TABLES / "model_metrics.csv")
    metrics = metrics[
        metrics["validation_scheme"].eq("blocked_river_year") & metrics["rmse"].notna()
    ]
    if metrics.empty:
        return pred
    best = (
        metrics.groupby(["model_tier", "algorithm"], as_index=False)["rmse"]
        .mean()
        .sort_values("rmse")
        .iloc[0]
    )
    return pred[
        (pred["model_tier"] == best["model_tier"])
        & (pred["algorithm"] == best["algorithm"])
    ].copy()


def workflow_diagram() -> None:
    labels = [
        "Raw ArcticGRO\n+ discharge",
        "HydroATLAS\n+ REAL migration",
        "Clean modeling\ntable",
        "Predictor audit\n(no leakage)",
        "Tiered + hierarchical\nmodels",
        "Blocked validation\n+ diagnostics",
    ]
    xy = [
        (0.05, 0.60),
        (0.05, 0.20),
        (0.32, 0.40),
        (0.55, 0.40),
        (0.76, 0.40),
        (0.76, 0.08),
    ]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    for (x, y), label in zip(xy, labels):
        ax.add_patch(Rectangle((x, y), 0.18, 0.18, fc="#e8f1fb", ec="#22577a", lw=1.5))
        ax.text(x + 0.09, y + 0.09, label, ha="center", va="center", fontsize=9)
    arrows = [(0, 2), (1, 2), (2, 3), (3, 4), (4, 5)]
    for start, end in arrows:
        sx, sy = xy[start]
        ex, ey = xy[end]
        ax.add_patch(
            FancyArrowPatch(
                (sx + 0.18, sy + 0.09),
                (ex, ey + 0.09),
                arrowstyle="->",
                mutation_scale=12,
                lw=1.3,
                color="#333333",
            )
        )
    savefig("01_workflow_diagram.png")


def measured_vs_predicted(p: pd.DataFrame) -> None:
    lims = [
        min(p["observed"].min(), p["predicted"].min()),
        max(p["observed"].max(), p["predicted"].max()),
    ]
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=p, x="observed", y="predicted", hue="River", s=45)
    plt.plot(lims, lims, "k--", lw=1)
    plt.xlabel("Measured Δ14C-POC (‰)")
    plt.ylabel("Predicted Δ14C-POC (‰)")
    savefig("07_measured_vs_predicted_all_rivers.png")

    g = sns.relplot(
        data=p,
        x="observed",
        y="predicted",
        col="River",
        col_wrap=3,
        hue="River",
        height=3,
        facet_kws={"sharex": True, "sharey": True},
    )
    for ax in g.axes.flat:
        ax.plot(lims, lims, "k--", lw=0.8)
    g.set_axis_labels("Measured Δ14C-POC (‰)", "Predicted Δ14C-POC (‰)")
    g.figure.savefig(FIGURES / "08_measured_vs_predicted_by_river.png", dpi=250)
    plt.close(g.figure)


def residual_diagnostics(p: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=p, x="River", y="residual")
    plt.axhline(0, color="k", ls="--", lw=1)
    plt.ylabel("Prediction residual (predicted - measured, ‰)")
    savefig("09_residuals_by_river.png")

    if "discharge_stage" in p:
        plt.figure(figsize=(10, 4))
        sns.boxplot(data=p, x="discharge_stage", y="residual")
        plt.xticks(rotation=30, ha="right")
        plt.axhline(0, color="k", ls="--", lw=1)
        savefig("10_residuals_by_discharge_stage.png")

    diagnostics = [
        ("predicted", "residuals_vs_predicted.png", "Predicted Δ14C-POC (‰)"),
        ("TSS", "residuals_vs_TSS.png", "Total suspended sediment (mg/L)"),
        ("Discharge", "residuals_vs_discharge.png", "Discharge (m³/s)"),
        ("Temp", "residuals_vs_water_temperature.png", "Water temperature (°C)"),
    ]
    for col, fname, xlabel in diagnostics:
        if col in p:
            plt.figure(figsize=(6, 4))
            sns.scatterplot(data=p, x=col, y="residual", hue="River", s=35)
            plt.axhline(0, color="k", ls="--", lw=1)
            plt.xlabel(xlabel)
            plt.ylabel("Residual (‰)")
            savefig(fname)


def seasonal_timeseries(p: pd.DataFrame) -> None:
    if "Julian_Day" not in p:
        return
    g = sns.relplot(
        data=p.sort_values("Julian_Day"),
        x="Julian_Day",
        y="observed",
        col="River",
        col_wrap=3,
        kind="scatter",
        height=3,
    )
    for ax, river in zip(g.axes.flat, sorted(p["River"].dropna().unique())):
        sub = p[p["River"] == river].sort_values("Julian_Day")
        ax.plot(sub["Julian_Day"], sub["predicted"], color="tab:red", lw=1.2)
    g.set_axis_labels("Day of year", "Δ14C-POC (‰)")
    g.figure.savefig(
        FIGURES / "11_timeseries_predicted_vs_measured_by_river.png", dpi=250
    )
    plt.close(g.figure)


def audit_and_stage_figures(df: pd.DataFrame) -> None:
    audit = pd.read_csv(TABLES / "predictor_audit.csv")
    status_col = (
        "retained_or_removed" if "retained_or_removed" in audit else "keep_remove"
    )
    plt.figure(figsize=(7, 4))
    sns.countplot(data=audit, y="variable_scale", hue=status_col)
    plt.xlabel("Candidate predictor count")
    plt.ylabel("Variable scale")
    savefig("02_predictor_audit_summary.png")

    if "discharge_stage" in df:
        plt.figure(figsize=(9, 4))
        sns.scatterplot(data=df, x="Julian_Day", y="Qstar", hue="discharge_stage", s=18)
        plt.xlabel("Day of year")
        plt.ylabel("Normalized discharge Q*")
        savefig("03_discharge_stage_classification.png")


def mechanism_figures(df: pd.DataFrame) -> None:
    if "catchment_old_carbon_baseline_index" in df:
        plt.figure(figsize=(7, 4))
        order = sorted(df["River"].dropna().unique())
        sns.barplot(
            data=df.drop_duplicates("River"),
            x="River",
            y="catchment_old_carbon_baseline_index",
            order=order,
        )
        plt.ylabel("Catchment old-carbon baseline index")
        savefig("04_catchment_baseline_index_by_river.png")
    for col, fname, ylabel in [
        (
            "old_bank_access_index",
            "05_daily_old_bank_access_index_by_river.png",
            "Daily old-bank access index",
        ),
        (
            "surface_erosion_index",
            "06_surface_erosion_runoff_index_by_river.png",
            "Surface-erosion/runoff access index",
        ),
    ]:
        if col in df:
            plt.figure(figsize=(9, 4))
            sns.scatterplot(data=df, x="Julian_Day", y=col, hue="River", s=20)
            plt.xlabel("Day of year")
            plt.ylabel(ylabel)
            savefig(fname)


def model_figures(p: pd.DataFrame, df: pd.DataFrame) -> None:
    metrics = pd.read_csv(TABLES / "model_metrics.csv")
    plt.figure(figsize=(12, 5))
    sns.barplot(
        data=metrics[metrics["rmse"].notna()],
        x="model_tier",
        y="rmse",
        hue="algorithm",
        errorbar=None,
    )
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("RMSE (‰)")
    savefig("14_model_comparison.png")

    plt.figure(figsize=(6, 5))
    sns.regplot(data=p, x="predicted", y="observed", scatter_kws={"s": 25})
    plt.xlabel("Predicted Δ14C-POC (‰)")
    plt.ylabel("Measured Δ14C-POC (‰)")
    savefig("15_calibration_plot.png")

    if "feature_importance.csv" in [x.name for x in TABLES.glob("*.csv")]:
        imp = pd.read_csv(TABLES / "feature_importance.csv").head(15)
        plt.figure(figsize=(8, 5))
        sns.barplot(data=imp, y="descriptive_name", x="importance_mean", orient="h")
        plt.xlabel("Permutation importance (RMSE increase)")
        plt.ylabel("")
        savefig("12_feature_importance.png")

    for col in ["old_bank_access_index", "surface_erosion_index", "log10_TSS"]:
        if col in df and TARGET in df:
            plt.figure(figsize=(6, 4))
            sns.regplot(data=df, x=col, y=TARGET, lowess=True, scatter_kws={"s": 20})
            plt.ylabel("Measured Δ14C-POC (‰)")
            savefig(f"13_partial_response_{col}.png")

    if "residual" in p:
        p2 = p.copy()
        spread = p2["residual"].std()
        p2["pi_low"] = p2["predicted"] - 1.96 * spread
        p2["pi_high"] = p2["predicted"] + 1.96 * spread
        p2 = p2.sort_values("predicted").reset_index(drop=True)
        x = np.arange(len(p2))
        plt.figure(figsize=(9, 4))
        plt.fill_between(
            x, p2["pi_low"], p2["pi_high"], alpha=0.25, label="Approx. 95% interval"
        )
        plt.scatter(x, p2["observed"], s=18, label="Measured")
        plt.plot(x, p2["predicted"], color="tab:red", label="Predicted")
        plt.xlabel("Samples sorted by prediction")
        plt.ylabel("Δ14C-POC (‰)")
        plt.legend()
        savefig("16_uncertainty_interval_diagnostics.png")


def main() -> None:
    ensure_dirs()
    workflow_diagram()
    pred = pd.read_csv(PROCESSED / "model_predictions.csv")
    p = best_predictions(pred)
    df = pd.read_csv(PROCESSED / "modeling_table_features.csv")
    if p.empty:
        return
    measured_vs_predicted(p)
    residual_diagnostics(p)
    seasonal_timeseries(p)
    audit_and_stage_figures(df)
    mechanism_figures(df)
    model_figures(p, df)
    print(f"Wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
