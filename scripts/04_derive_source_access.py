#!/usr/bin/env python
"""Derive source-access indices and predictor audit before modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import PROCESSED, TABLES, TARGET, descriptive_name, ensure_dirs, scale_0_1

PROTECTED_PATTERNS = [
    "13C",
    "delta13C",
    "POC-13C",
    "DOC-14C",
    "DOC-fm",
    "POC-fm",
    "Fm_POC",
    "Delta14C",
    "POC-14C",
]
RAW_LOG_PAIRS = {"TSS": "log10_TSS", "Discharge": "log10_Discharge"}
CORE_KEEP = [
    "Temp",
    "POC_pct",
    "log10_TSS",
    "log10_Discharge",
    "Qstar",
    "dQdt",
    "rising_limb",
    "falling_limb",
    "thaw_potential_scaled",
    "entrainment_potential",
    "bank_access_gate",
    "old_bank_access_index",
    "surface_erosion_index",
]
CATCHMENT_KEEP = [
    "catchment_old_carbon_baseline_index",
    "prm_pc_sse",
    "run_mm_syr",
    "pre_mm_syr",
    "ero_kh_sav",
    "slp_dg_sav",
    "sgr_dk_sav",
    "cly_pc_sav",
    "slt_pc_sav",
    "soc_th_sav",
    "wetland_extent_total",
    "lithology_old_sedimentary_fraction",
    "GSoil14C_mean",
    "GSoilRC_mean",
    "NPP_mean",
    "GPP_mean",
    "migration_positive_erosion_median",
    "migration_positive_erosion_p75",
    "migration_width_median",
    "migration_mean_sediment_discharge_median",
]
PREFERRED_CLEAN_COLUMNS = [
    "River",
    "Date_parsed",
    "Year",
    "Julian_Day",
    "Discharge",
    "Temp",
    "TSS",
    "POC_pct",
    "Delta14C_POC",
    "Fm_POC",
    "Qstar",
    "dQdt",
    "rising_limb",
    "falling_limb",
    "discharge_stage",
    "bank_access_gate",
    "old_bank_access_index",
    "surface_erosion_index",
    "catchment_old_carbon_baseline_index",
]


def safe_numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def derive(df: pd.DataFrame) -> pd.DataFrame:
    """Add mechanism-based daily and catchment indices without using isotope leakage."""
    out = df.copy()
    tss = safe_numeric(out, "TSS")
    discharge = safe_numeric(out, "Discharge")
    out["log10_TSS"] = np.where(tss > 0, np.log10(tss), np.nan)
    out["log10_Discharge"] = np.where(discharge > 0, np.log10(discharge), np.nan)

    temp = safe_numeric(out, "Temp")
    out["thaw_potential"] = temp.clip(lower=0)
    out["thaw_potential_scaled"] = out.groupby("River")["thaw_potential"].transform(
        scale_0_1
    )
    flow_basis = safe_numeric(out, "Qstar").where(
        safe_numeric(out, "Qstar").notna(), out["log10_Discharge"]
    )
    out["entrainment_potential"] = flow_basis.groupby(out["River"]).transform(scale_0_1)
    out["bank_access_gate"] = np.minimum(
        out["thaw_potential_scaled"], out["entrainment_potential"]
    )

    permafrost = safe_numeric(out, "prm_pc_sse") / 100.0
    migration = scale_0_1(safe_numeric(out, "migration_positive_erosion_p75")).fillna(
        scale_0_1(safe_numeric(out, "migration_positive_erosion_median"))
    )
    out["migration_potential_scaled"] = migration
    out["old_bank_access_index"] = (
        out["bank_access_gate"]
        * permafrost.fillna(permafrost.median())
        * out["migration_potential_scaled"].fillna(
            out["migration_potential_scaled"].median()
        )
    )

    fine_soil = (
        safe_numeric(out, "cly_pc_sav") + safe_numeric(out, "slt_pc_sav")
    ) / 100.0
    runoff = scale_0_1(safe_numeric(out, "run_mm_syr")).fillna(
        scale_0_1(safe_numeric(out, "pre_mm_syr"))
    )
    erosion = scale_0_1(safe_numeric(out, "ero_kh_sav"))
    out["surface_erosion_index"] = runoff * erosion * fine_soil

    wet_cols = [c for c in out.columns if c.startswith("wet_pc_")]
    out["wetland_extent_total"] = (
        out[wet_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        if wet_cols
        else np.nan
    )
    lithology_cols = [c for c in ["su", "ss", "sm", "sc"] if c in out.columns]
    out["lithology_old_sedimentary_fraction"] = (
        out[lithology_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        if lithology_cols
        else np.nan
    )

    old_soil_age = scale_0_1(safe_numeric(out, "GSoilRC_mean"))
    low_soil_delta14c = 1 - scale_0_1(safe_numeric(out, "GSoil14C_mean"))
    soil_oc = scale_0_1(safe_numeric(out, "soc_th_sav"))
    permafrost_scaled = scale_0_1(safe_numeric(out, "prm_pc_sse"))
    wetland_scaled = scale_0_1(safe_numeric(out, "wetland_extent_total"))
    lithology_scaled = scale_0_1(
        safe_numeric(out, "lithology_old_sedimentary_fraction")
    )
    erosion_scaled = scale_0_1(safe_numeric(out, "ero_kh_sav"))
    migration_scaled = out["migration_potential_scaled"]
    out["catchment_old_carbon_baseline_index"] = pd.concat(
        [
            old_soil_age,
            low_soil_delta14c,
            soil_oc,
            permafrost_scaled,
            wetland_scaled,
            lithology_scaled,
            erosion_scaled,
            migration_scaled,
        ],
        axis=1,
    ).mean(axis=1)

    npp = scale_0_1(safe_numeric(out, "NPP_mean"))
    warm = out["thaw_potential_scaled"]
    low_tss = 1 - scale_0_1(out["log10_TSS"])
    high_poc = scale_0_1(safe_numeric(out, "POC_pct"))
    out["aquatic_biomass_score"] = pd.concat(
        [wetland_scaled, npp, warm, low_tss, high_poc], axis=1
    ).mean(axis=1)
    out["aquatic_biomass_score_note"] = (
        "diagnostic composite only; not a measured endmember fraction"
    )
    return out


def audit(df: pd.DataFrame) -> pd.DataFrame:
    """Create a transparent predictor audit enforcing no-leakage/no-double-counting rules."""
    numeric = df.select_dtypes(include=[np.number]).copy()
    retained: list[str] = []
    rows = []
    candidates = [c for c in numeric.columns if c != TARGET]
    keep_set = set(CORE_KEEP + CATCHMENT_KEEP)
    for col in candidates:
        reason = "candidate retained"
        keep = col in keep_set
        double_counting_risk = "no"
        if any(p in col for p in PROTECTED_PATTERNS):
            keep, reason = (
                False,
                "removed: isotope/radiocarbon response or δ13C source tracer; not allowed as Δ14C predictor",
            )
        elif col in RAW_LOG_PAIRS:
            double_counting_risk = "yes: raw/log duplicate"
            keep, reason = (
                False,
                f"removed: raw version duplicated by retained {RAW_LOG_PAIRS[col]}",
            )
        elif col in {"POC_concentration", "POC_flux", "TSS_flux"}:
            double_counting_risk = "yes: deterministic product or flux"
            keep, reason = (
                False,
                "removed: deterministic product/flux would double-count components",
            )
        elif col.endswith("_catchment") or col in {
            "FID",
            "SORT",
            "Unnamed: 72",
            "HYBAS_ID",
            "PFAF_ID",
        }:
            keep, reason = (
                False,
                "removed: identifier/bookkeeping or duplicate catchment field",
            )
        elif col not in keep_set:
            keep, reason = (
                False,
                "removed: not in pre-specified parsimonious tier feature set",
            )
        corr_target = (
            numeric[[col, TARGET]].corr().iloc[0, 1]
            if TARGET in numeric and numeric[col].notna().sum() > 2
            else np.nan
        )
        highest_corr = np.nan
        highest_corr_predictor = ""
        if retained and keep:
            cmat = numeric[[col] + retained].corr().loc[col, retained].abs()
            if len(cmat):
                highest_corr = cmat.max()
                highest_corr_predictor = str(cmat.idxmax())
            if pd.notna(highest_corr) and highest_corr > 0.95:
                keep, reason = (
                    False,
                    f"removed: |r| > 0.95 with retained predictor {highest_corr_predictor}",
                )
                double_counting_risk = "yes: highly collinear retained predictor"
        if keep:
            retained.append(col)
        scale = (
            "sample/daily-level"
            if col in CORE_KEEP
            or col in {"Qstar", "dQdt", "rising_limb", "falling_limb"}
            else "catchment-level" if col in CATCHMENT_KEEP else "global/metadata"
        )
        source = (
            "ArcticGRO sample + daily discharge"
            if scale == "sample/daily-level"
            else "HydroATLAS + REAL migration"
        )
        rows.append(
            {
                "original_column_name": col,
                "descriptive_name": descriptive_name(col),
                "source_dataset": source,
                "variable_scale": scale,
                "retained_or_removed": "retained" if keep else "removed",
                "reason_for_removal": "" if keep else reason,
                "correlation_with_target": corr_target,
                "highest_abs_correlation_with_retained_predictor": highest_corr,
                "highest_correlated_retained_predictor": highest_corr_predictor,
                "double_counting_risk": double_counting_risk,
            }
        )
    return pd.DataFrame(rows)


def write_cleaned_modeling_table(df: pd.DataFrame) -> None:
    rename = {
        "River": "river name",
        "Date_parsed": "date",
        "Year": "year",
        "Julian_Day": "Julian day",
        "Discharge": "discharge (m3 s-1)",
        "Temp": "water temperature (deg C)",
        "TSS": "total suspended sediment (mg L-1)",
        "POC_pct": "POC percent of suspended sediment",
        "Delta14C_POC": "Delta14C-POC (per mil)",
        "Fm_POC": "Fm-POC",
        "Qstar": "normalized discharge Qstar",
        "dQdt": "daily discharge derivative dQ/dt",
        "rising_limb": "rising limb indicator",
        "falling_limb": "falling limb indicator",
        "discharge_stage": "six-stage discharge classification",
        "bank_access_gate": "thaw-entrainment bank-access gate",
        "old_bank_access_index": "daily old-bank access index",
        "surface_erosion_index": "surface-erosion runoff access index",
        "catchment_old_carbon_baseline_index": "catchment old-carbon baseline index",
    }
    selected = [c for c in PREFERRED_CLEAN_COLUMNS if c in df.columns]
    catchment_keep = [
        c for c in CATCHMENT_KEEP if c in df.columns and c not in selected
    ]
    migration_keep = [
        c for c in df.columns if c.startswith("migration_") and c not in selected
    ]
    clean = df[selected + catchment_keep + migration_keep].copy()
    clean = clean.rename(
        columns={**rename, **{c: descriptive_name(c) for c in catchment_keep}}
    )
    clean.to_csv(TABLES / "cleaned_modeling_table.csv", index=False)


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(PROCESSED / "modeling_table.csv", parse_dates=["Date_parsed"])
    derived = derive(df)
    derived.to_csv(PROCESSED / "modeling_table_features.csv", index=False)
    write_cleaned_modeling_table(derived)
    aud = audit(derived)
    aud.to_csv(TABLES / "predictor_audit.csv", index=False)
    print(
        f"Wrote {PROCESSED/'modeling_table_features.csv'}, "
        f"{TABLES/'cleaned_modeling_table.csv'}, and {TABLES/'predictor_audit.csv'}"
    )


if __name__ == "__main__":
    main()
