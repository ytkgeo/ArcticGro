#!/usr/bin/env python
"""Write DOCX and PDF summary reports."""

from __future__ import annotations
import pandas as pd
from common import TABLES, FIGURES, REPORTS, ensure_dirs

SECTIONS = [
    (
        "Introduction",
        "This report documents a reproducible pipeline for explaining and predicting particulate organic carbon radiocarbon (Δ14C-POC) in six large Arctic rivers while separating catchment baseline differences from seasonal and daily hydrograph tuning.",
    ),
    (
        "Data and variable dictionary",
        "Inputs include ArcticGRO sample chemistry, river-specific daily discharge, HydroATLAS catchment attributes, REAL migration summaries, and optional global POC context. HydroATLAS names are decoded in outputs/tables/hydroatlas_variable_dictionary.csv.",
    ),
    (
        "Methods",
        "Daily discharge climatologies are smoothed by day of year, transformed to Q*, differentiated to dQ/dt, and classified into six hydrograph stages. Source-access terms represent joint thaw and entrainment limitation plus surface erosion/runoff access.",
    ),
    (
        "Predictor screening",
        "The predictor audit removes δ13C, response/isotope leakage variables, duplicated raw/log variables, deterministic flux or concentration products, identifiers, and collinear variables.",
    ),
    (
        "Model architecture",
        "Tiered models progress from river means (M0), catchment baseline (M1), daily hydrograph variables (M2), sediment and POC content (M3), and source-access indices (M4). M5 is represented operationally by blocked river-year residual calibration outputs where enough data are available.",
    ),
    (
        "Cross-validation design",
        "The main transferability validation is leave-one-river-out. The monitored-river validation is blocked by river-year. Random row splits are intentionally not reported as main evidence.",
    ),
    (
        "Results",
        "Metrics are reported for R2, RMSE, MAE, bias, residual standard deviation, median error, p95 absolute error, overprediction rate, calibration, Pearson r, and Spearman rho.",
    ),
    (
        "Mechanistic interpretation",
        "Bank-access indices follow the concept that old bank carbon delivery is jointly limited by thaw and entrainment. Surface erosion indices combine runoff or precipitation, soil erosion rate, and fine soil fraction.",
    ),
    (
        "Limitations",
        "REAL reach records are not full geometries, so spatial matching is audited and may fall back to basin identifiers or a documented global summary. The six-river catchment table is small, so catchment-only machine learning must be interpreted cautiously.",
    ),
    (
        "Recommendations",
        "Use blocked river-year metrics for operational claims, leave-one-river-out for transferability claims, and the empirical Ridge equation only as an interpretable approximation to nonlinear prediction models.",
    ),
]


def markdown_report(metrics: pd.DataFrame) -> str:
    top = (
        metrics[metrics["rmse"].notna()]
        .groupby(["validation_scheme", "model_tier", "algorithm"], as_index=False)
        .agg(
            rmse=("rmse", "mean"),
            mae=("mae", "mean"),
            r2=("r2", "mean"),
            n=("n", "sum"),
        )
        .sort_values(["validation_scheme", "rmse"])
        .head(20)
    )
    text = ["# ArcticGRO Δ14C-POC prediction report", ""]
    for title, body in SECTIONS:
        text += [f"## {title}", body, ""]
        if title == "Results" and not top.empty:
            text += [top.to_markdown(index=False), ""]
    return "\n".join(text)


def write_docx(text: str) -> None:
    try:
        from docx import Document
    except ImportError:
        (REPORTS / "final_ml_prediction_report.docx.txt").write_text(text)
        return
    doc = Document()
    for line in text.splitlines():
        if line.startswith("# "):
            doc.add_heading(line[2:], level=0)
        elif line.startswith("## "):
            doc.add_heading(line[3:], level=1)
        elif line.strip():
            doc.add_paragraph(line)
    for fig in sorted(FIGURES.glob("*.png"))[:6]:
        doc.add_heading(fig.stem.replace("_", " "), level=2)
        try:
            doc.add_picture(str(fig), width=None)
        except Exception:
            doc.add_paragraph(str(fig))
    doc.save(REPORTS / "final_ml_prediction_report.docx")


def write_pdf(text: str) -> None:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        (REPORTS / "final_ml_prediction_report.pdf.txt").write_text(text)
        return
    c = canvas.Canvas(str(REPORTS / "final_ml_prediction_report.pdf"), pagesize=letter)
    width, height = letter
    y = height - 50
    for line in text.splitlines():
        if y < 50:
            c.showPage()
            y = height - 50
        c.setFont(
            "Helvetica-Bold" if line.startswith("#") else "Helvetica",
            12 if line.startswith("#") else 9,
        )
        c.drawString(40, y, line[:110])
        y -= 14
    c.save()


def main() -> None:
    ensure_dirs()
    metrics = (
        pd.read_csv(TABLES / "model_metrics.csv")
        if (TABLES / "model_metrics.csv").exists()
        else pd.DataFrame()
    )
    text = markdown_report(metrics)
    (REPORTS / "final_ml_prediction_report.md").write_text(text)
    write_docx(text)
    write_pdf(text)
    print(f"Wrote reports to {REPORTS}")


if __name__ == "__main__":
    main()
