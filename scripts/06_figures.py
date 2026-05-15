#!/usr/bin/env python
"""Generate diagnostic figures for the ArcticGRO Δ14C prediction pipeline."""

from __future__ import annotations
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from common import PROCESSED, TABLES, FIGURES, ensure_dirs

sns.set_theme(style="whitegrid")


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=200)
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
    ]


def main() -> None:
    ensure_dirs()
    pred = pd.read_csv(PROCESSED / "model_predictions.csv")
    p = best_predictions(pred)
    if p.empty:
        return
    lims = [
        min(p["observed"].min(), p["predicted"].min()),
        max(p["observed"].max(), p["predicted"].max()),
    ]
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=p, x="observed", y="predicted", hue="River")
    plt.plot(lims, lims, "k--")
    plt.xlabel("Measured Δ14C-POC (‰)")
    plt.ylabel("Predicted Δ14C-POC (‰)")
    savefig("measured_vs_predicted_all_rivers.png")
    g = sns.relplot(
        data=p,
        x="observed",
        y="predicted",
        col="River",
        col_wrap=3,
        hue="River",
        height=3,
    )
    g.set_axis_labels("Measured Δ14C-POC (‰)", "Predicted Δ14C-POC (‰)")
    g.figure.savefig(FIGURES / "measured_vs_predicted_by_river.png", dpi=200)
    plt.close(g.figure)
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=p, x="River", y="residual")
    plt.axhline(0, color="k", ls="--")
    plt.ylabel("Prediction residual (predicted - measured, ‰)")
    savefig("residuals_by_river.png")
    if "discharge_stage" in p:
        plt.figure(figsize=(10, 4))
        sns.boxplot(data=p, x="discharge_stage", y="residual")
        plt.xticks(rotation=30, ha="right")
        plt.axhline(0, color="k", ls="--")
        savefig("residuals_by_discharge_stage.png")
    for col, fname, xlabel in [
        ("TSS", "residuals_vs_TSS.png", "Total suspended sediment (mg/L)"),
        ("Qstar", "residuals_vs_Qstar.png", "Normalized discharge state Q*"),
    ]:
        if col in p:
            plt.figure(figsize=(6, 4))
            sns.scatterplot(data=p, x=col, y="residual", hue="River")
            plt.axhline(0, color="k", ls="--")
            plt.xlabel(xlabel)
            savefig(fname)
    if "Julian_Day" in p:
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
            ax.plot(sub["Julian_Day"], sub["predicted"], color="tab:red", lw=1)
        g.set_axis_labels("Day of year", "Δ14C-POC (‰)")
        g.figure.savefig(
            FIGURES / "seasonal_predicted_vs_measured_by_river.png", dpi=200
        )
        plt.close(g.figure)
    audit = pd.read_csv(TABLES / "predictor_audit.csv")
    kept = audit.loc[audit["keep_remove"] == "keep", "column_name"].tolist()
    df = pd.read_csv(PROCESSED / "modeling_table_features.csv")
    corr = df[[c for c in kept if c in df.columns]].corr(numeric_only=True)
    if not corr.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, cmap="vlag", center=0)
        savefig("predictor_audit_correlation_matrix.png")
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
    savefig("model_rmse_comparison.png")
    print(f"Wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
