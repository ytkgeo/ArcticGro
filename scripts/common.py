"""Shared utilities for the ArcticGRO Δ14C-POC prediction pipeline."""

from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
REPORTS = ROOT / "outputs" / "reports"

RIVERS = ["Ob", "Yenisey", "Lena", "Kolyma", "Yukon", "Mackenzie"]
DISCHARGE_FILES = {river: f"{river}_20240322.xlsx" for river in RIVERS}

TARGET = "Delta14C_POC"

RIVER_ALIASES = {
    "obi": "Ob",
    "ob": "Ob",
    "ob'": "Ob",
    "enisey": "Yenisey",
    "yenisei": "Yenisey",
    "yenisey": "Yenisey",
    "lena": "Lena",
    "kolyma": "Kolyma",
    "yukon": "Yukon",
    "mackenzie": "Mackenzie",
    "macKenzie": "Mackenzie",
}

HYDRO_DESCRIPTIONS = {
    "SUB_AREA": "sub-basin area (km²)",
    "UP_AREA": "total upstream drainage area (km²)",
    "slp_dg_sav": "average terrain slope (degrees)",
    "sgr_dk_sav": "average stream gradient (dm/km)",
    "sgr_dg_sav": "average stream gradient (degrees)",
    "dis_m3_pyr": "natural discharge, annual average (m³/s)",
    "dis_m3_pmn": "natural discharge, annual minimum (m³/s)",
    "dis_m3_pmx": "natural discharge, annual maximum (m³/s)",
    "run_mm_syr": "land surface runoff, annual average (mm)",
    "pre_mm_syr": "mean annual precipitation (mm)",
    "pet_mm_syr": "potential evapotranspiration, annual (mm)",
    "aet_mm_syr": "actual evapotranspiration, annual (mm)",
    "snw_pc_syr": "snow cover extent, annual average (%)",
    "snw_pc_smx": "snow cover extent, annual maximum (%)",
    "wet_pc_sg1": "grouped wetland extent class 1 (%)",
    "wet_pc_sg2": "grouped wetland extent class 2 (%)",
    "for_pc_sse": "forest cover extent (%)",
    "gla_pc_sse": "glacier extent (%)",
    "prm_pc_sse": "permafrost extent (%)",
    "cly_pc_sav": "soil clay fraction, average (%)",
    "slt_pc_sav": "soil silt fraction, average (%)",
    "snd_pc_sav": "soil sand fraction, average (%)",
    "soc_th_sav": "soil organic carbon content (t/ha)",
    "swc_pc_syr": "soil water content, annual average (%)",
    "ero_kh_sav": "soil erosion rate (kg/ha/yr)",
    "catchment_old_carbon_baseline_index": "catchment old-carbon baseline index",
    "wetland_extent_total": "total wetland extent across GLWD classes (%)",
    "lithology_old_sedimentary_fraction": "old-carbon-prone sedimentary and unconsolidated lithology fraction",
    "GSoil14C_mean": "mean soil Δ14C",
    "GSoil14C_stdev": "soil Δ14C standard deviation",
    "GSoilRC_mean": "mean soil radiocarbon residence metric",
    "GSoilRC_stdev": "soil radiocarbon residence metric standard deviation",
    "NPP_mean": "mean net primary productivity",
    "NPP_stdev": "net primary productivity standard deviation",
    "GPP_mean": "mean gross primary productivity",
    "GPP_stdev": "gross primary productivity standard deviation",
    "su": "GLiM unconsolidated sediment fraction",
    "ss": "GLiM siliciclastic sedimentary rock fraction",
    "sc": "GLiM carbonate sedimentary rock fraction",
    "sm": "GLiM mixed sedimentary rock fraction",
}

SAMPLE_DESCRIPTIONS = {
    "Discharge": "paired sample discharge (m³/s)",
    "Temp": "water temperature (°C)",
    "TSS": "total suspended sediment (mg/L)",
    "POC": "particulate organic carbon content (%)",
    "POC_pct": "particulate organic carbon content (%)",
    "log10_TSS": "log10 total suspended sediment",
    "log10_Discharge": "log10 paired sample discharge",
    "Qstar": "normalized discharge state from daily climatology",
    "dQdt": "daily discharge derivative (m³/s/day)",
    "rising_limb": "rising hydrograph limb indicator",
    "falling_limb": "falling hydrograph limb indicator",
    "thaw_potential_scaled": "river-scaled positive water temperature",
    "entrainment_potential": "river-scaled flow entrainment potential",
    "bank_access_gate": "joint thaw-entrainment bank access gate",
    "old_bank_access_index": "old-bank carbon access index",
    "surface_erosion_index": "surface erosion/runoff access index",
    "aquatic_biomass_score": "diagnostic aquatic biomass score (not measured endmember fraction)",
}


def ensure_dirs() -> None:
    for p in [RAW, PROCESSED, TABLES, FIGURES, REPORTS]:
        p.mkdir(parents=True, exist_ok=True)


def raw_path(name: str) -> Path:
    """Find a raw input in data/raw first, then repository root for backward compatibility."""
    p = RAW / name
    if p.exists():
        return p
    p = ROOT / name
    if p.exists():
        return p
    return RAW / name


def harmonize_river(value) -> str:
    if pd.isna(value):
        return value
    s = str(value).strip()
    key = re.sub(r"[^a-z]", "", s.lower())
    return RIVER_ALIASES.get(key, s.title())


def coerce_numeric_frame(
    df: pd.DataFrame, exclude: set[str] | None = None
) -> pd.DataFrame:
    exclude = exclude or set()
    out = df.copy()
    for col in out.columns:
        if col in exclude:
            continue
        if out[col].dtype == object:
            converted = pd.to_numeric(
                out[col].replace({"NA": np.nan, "na": np.nan, "": np.nan}),
                errors="coerce",
            )
            if (
                converted.notna().sum() > 0
                and converted.notna().sum() >= out[col].notna().sum() * 0.5
            ):
                out[col] = converted
    return out


def scale_0_1(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mn, mx = s.min(skipna=True), s.max(skipna=True)
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(np.where(s.notna(), 0.0, np.nan), index=s.index)
    return ((s - mn) / (mx - mn)).clip(0, 1)


def descriptive_name(col: str) -> str:
    if col in HYDRO_DESCRIPTIONS:
        return HYDRO_DESCRIPTIONS[col]
    if col in SAMPLE_DESCRIPTIONS:
        return SAMPLE_DESCRIPTIONS[col]
    if col.startswith("glc_pc_s"):
        return f"GLC2000 land-cover class {col[-2:]} extent (%)"
    if col.startswith("wet_pc_s"):
        return f"GLWD wetland class {col[-2:]} extent (%)"
    if col.startswith("migration_"):
        return col.replace("migration_", "REAL migration ").replace("_", " ")
    return col.replace("_", " ")
