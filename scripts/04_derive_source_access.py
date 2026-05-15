#!/usr/bin/env python
"""Derive source-access indices and predictor audit before modeling."""

from __future__ import annotations
import numpy as np
import pandas as pd
from common import PROCESSED, TABLES, ensure_dirs, scale_0_1, descriptive_name, TARGET

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
]


def derive(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log10_TSS"] = np.where(
        pd.to_numeric(out.get("TSS"), errors="coerce") > 0,
        np.log10(pd.to_numeric(out.get("TSS"), errors="coerce")),
        np.nan,
    )
    out["log10_Discharge"] = np.where(
        pd.to_numeric(out.get("Discharge"), errors="coerce") > 0,
        np.log10(pd.to_numeric(out.get("Discharge"), errors="coerce")),
        np.nan,
    )
    temp = pd.to_numeric(out.get("Temp"), errors="coerce")
    out["thaw_potential"] = temp.clip(lower=0)
    out["thaw_potential_scaled"] = out.groupby("River")["thaw_potential"].transform(
        scale_0_1
    )
    flow_basis = out["Qstar"].where(out["Qstar"].notna(), out["log10_Discharge"])
    out["entrainment_potential"] = flow_basis.groupby(out["River"]).transform(scale_0_1)
    out["bank_access_gate"] = np.minimum(
        out["thaw_potential_scaled"], out["entrainment_potential"]
    )
    permafrost = pd.to_numeric(out.get("prm_pc_sse"), errors="coerce") / 100.0
    migration = scale_0_1(
        pd.to_numeric(out.get("migration_positive_erosion_p75"), errors="coerce")
    ).fillna(
        scale_0_1(
            pd.to_numeric(out.get("migration_positive_erosion_median"), errors="coerce")
        )
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
        pd.to_numeric(out.get("cly_pc_sav"), errors="coerce")
        + pd.to_numeric(out.get("slt_pc_sav"), errors="coerce")
    ) / 100.0
    runoff = scale_0_1(pd.to_numeric(out.get("run_mm_syr"), errors="coerce")).fillna(
        scale_0_1(pd.to_numeric(out.get("pre_mm_syr"), errors="coerce"))
    )
    erosion = scale_0_1(pd.to_numeric(out.get("ero_kh_sav"), errors="coerce"))
    out["surface_erosion_index"] = runoff * erosion * fine_soil
    wet_cols = [c for c in out.columns if c.startswith("wet_pc_")]
    wet = (
        out[wet_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        if wet_cols
        else pd.Series(np.nan, index=out.index)
    )
    npp = scale_0_1(pd.to_numeric(out.get("NPP_mean"), errors="coerce"))
    warm = out["thaw_potential_scaled"]
    low_tss = 1 - scale_0_1(out["log10_TSS"])
    high_poc = scale_0_1(pd.to_numeric(out.get("POC_pct"), errors="coerce"))
    out["aquatic_biomass_score"] = pd.concat(
        [scale_0_1(wet), npp, warm, low_tss, high_poc], axis=1
    ).mean(axis=1)
    out["aquatic_biomass_score_note"] = (
        "diagnostic composite only; not a measured endmember fraction"
    )
    return out


def audit(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number]).copy()
    retained = []
    rows = []
    candidates = [c for c in numeric.columns if c != TARGET]
    keep_set = set(CORE_KEEP + CATCHMENT_KEEP)
    for col in candidates:
        reason = "candidate retained"
        keep = col in keep_set
        if any(p in col for p in PROTECTED_PATTERNS):
            keep, reason = (
                False,
                "removed: isotope/radiocarbon response or δ13C source tracer; not allowed as Δ14C predictor",
            )
        elif col in RAW_LOG_PAIRS:
            keep, reason = (
                False,
                f"removed: raw version duplicated by retained {RAW_LOG_PAIRS[col]}",
            )
        elif col in {"POC_concentration", "POC_flux", "TSS_flux"}:
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
        high_corr = np.nan
        if retained and keep:
            cmat = numeric[[col] + retained].corr().loc[col, retained].abs()
            high_corr = cmat.max() if len(cmat) else np.nan
            if pd.notna(high_corr) and high_corr > 0.95:
                keep, reason = (
                    False,
                    "removed: |r| > 0.95 with an already retained predictor",
                )
        if keep:
            retained.append(col)
        scale = (
            "daily"
            if col in {"Qstar", "dQdt", "rising_limb", "falling_limb"}
            else (
                "sample"
                if col in CORE_KEEP
                else "catchment" if col in CATCHMENT_KEEP else "global/metadata"
            )
        )
        source = (
            "ArcticGRO/daily discharge"
            if scale in {"daily", "sample"}
            else "HydroATLAS/REAL"
        )
        rows.append(
            {
                "column_name": col,
                "real_descriptive_name": descriptive_name(col),
                "data_source": source,
                "scale": scale,
                "keep_remove": "keep" if keep else "remove",
                "reason": reason,
                "correlation_with_target": corr_target,
                "highest_abs_correlation_with_retained_predictor": high_corr,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(PROCESSED / "modeling_table.csv", parse_dates=["Date_parsed"])
    derived = derive(df)
    derived.to_csv(PROCESSED / "modeling_table_features.csv", index=False)
    aud = audit(derived)
    aud.to_csv(TABLES / "predictor_audit.csv", index=False)
    print(
        f"Wrote {PROCESSED/'modeling_table_features.csv'} and {TABLES/'predictor_audit.csv'}"
    )


if __name__ == "__main__":
    main()
