#!/usr/bin/env python
"""Load raw ArcticGRO files, create inventory, and clean sample observations."""

from __future__ import annotations
import json
import pandas as pd
from common import (
    PROCESSED,
    TABLES,
    RIVERS,
    DISCHARGE_FILES,
    ensure_dirs,
    raw_path,
    harmonize_river,
    coerce_numeric_frame,
)

INPUTS = [
    "ArcticGRO-compilation.csv",
    "ArcticGRO-HydroAtlas.xlsx",
    "REAL_reach.csv",
    "POC_Summ.csv",
    *DISCHARGE_FILES.values(),
    "HydroATLAS_TechDoc_v10_1.pdf",
    "s41586-024-07978-w.pdf",
    "s41561-024-01476-4.pdf",
]


def inventory() -> pd.DataFrame:
    rows = []
    for name in INPUTS:
        p = raw_path(name)
        row = {
            "file": name,
            "path_found": str(p) if p.exists() else "MISSING",
            "exists": p.exists(),
            "rows": None,
            "columns": None,
            "headers": None,
        }
        if p.exists() and p.suffix.lower() in {".csv", ".xlsx", ".xls"}:
            try:
                df = (
                    pd.read_excel(p, nrows=5)
                    if p.suffix.lower().startswith(".xls")
                    else pd.read_csv(p, nrows=5)
                )
                if p.suffix.lower().startswith(".xls"):
                    full = pd.read_excel(p, usecols=[0])
                else:
                    full = pd.read_csv(p, usecols=[0])
                row.update(
                    {
                        "rows": len(full),
                        "columns": len(df.columns),
                        "headers": json.dumps(list(df.columns), ensure_ascii=False),
                    }
                )
            except Exception as exc:
                row["headers"] = f"ERROR: {exc}"
        rows.append(row)
    return pd.DataFrame(rows)


def clean_arcticgro() -> pd.DataFrame:
    df = pd.read_csv(raw_path("ArcticGRO-compilation.csv"))
    df.columns = [str(c).strip() for c in df.columns]
    df["River"] = df["River"].map(harmonize_river)
    df["Date_parsed"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = df["Date_parsed"].dt.year
    df["Julian_Day"] = df["Date_parsed"].dt.dayofyear
    if "Julian Day" in df.columns:
        df["Julian_Day_source"] = pd.to_numeric(df["Julian Day"], errors="coerce")
        df["Julian_Day"] = df["Julian_Day"].fillna(df["Julian_Day_source"])
    df = coerce_numeric_frame(
        df, exclude={"Phase", "River", "Date", "ID", "Date_parsed"}
    )
    rename = {"POC-14C": "Delta14C_POC", "POC-fm": "Fm_POC", "POC": "POC_pct"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "Fm_POC" in df and "Delta14C_POC" in df:
        df["Fm_POC"] = df["Fm_POC"].fillna(1 + df["Delta14C_POC"] / 1000.0)
    df = df[df["River"].isin(RIVERS)].copy()
    return df


def main() -> None:
    ensure_dirs()
    inv = inventory()
    inv.to_csv(TABLES / "data_inventory.csv", index=False)
    clean = clean_arcticgro()
    clean.to_csv(PROCESSED / "arcticgro_clean.csv", index=False)
    summary = (
        clean.groupby("River", dropna=False)
        .agg(
            samples=("River", "size"),
            target_available=("Delta14C_POC", lambda s: s.notna().sum()),
        )
        .reset_index()
    )
    summary.to_csv(TABLES / "arcticgro_clean_summary.csv", index=False)
    print(
        f"Wrote {PROCESSED/'arcticgro_clean.csv'} with {len(clean)} rows and {clean.shape[1]} columns"
    )


if __name__ == "__main__":
    main()
