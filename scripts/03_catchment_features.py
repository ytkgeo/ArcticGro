#!/usr/bin/env python
"""Merge HydroATLAS catchment features and REAL river-migration summaries."""

from __future__ import annotations
import pandas as pd
from common import (
    PROCESSED,
    TABLES,
    ensure_dirs,
    raw_path,
    harmonize_river,
    coerce_numeric_frame,
    descriptive_name,
)


def load_hydroatlas() -> pd.DataFrame:
    h = pd.read_excel(raw_path("ArcticGRO-HydroAtlas.xlsx"))
    h.columns = [str(c).strip() for c in h.columns]
    river_col = "Major_River" if "Major_River" in h.columns else "Short_Name"
    h["River"] = h[river_col].map(harmonize_river)
    h = coerce_numeric_frame(
        h,
        exclude={
            "River",
            "Major_River",
            "Basin",
            "Short_Name",
            "Type_River",
            "Continent",
        },
    )
    return h


def summarize_real(hydro: pd.DataFrame) -> pd.DataFrame:
    real = pd.read_csv(raw_path("REAL_reach.csv"))
    real.columns = [str(c).strip() for c in real.columns]
    for col in real.columns:
        if col != "basin_id":
            real[col] = pd.to_numeric(real[col], errors="coerce")
    frames = []
    ids = []
    for _, row in hydro.iterrows():
        river = row["River"]
        candidates = [row.get("HYBAS_ID"), row.get("PFAF_ID")]
        candidates = [int(x) for x in candidates if pd.notna(x)]
        sub = real[real["basin_id"].isin(candidates)] if candidates else real.iloc[0:0]
        if sub.empty and "PFAF_ID" in hydro.columns and pd.notna(row.get("PFAF_ID")):
            pf = str(int(row["PFAF_ID"]))
            sub = real[
                real["basin_id"]
                .astype("Int64")
                .astype(str)
                .str.startswith(pf[:3], na=False)
            ]
        if sub.empty:
            sub = (
                real.copy()
            )  # explicit fallback: dataset has no geometry; use global REAL summary with warning flag.
            matched = "global_fallback_no_spatial_clip"
        else:
            matched = "basin_id_or_pfaf_prefix"
        pos_erosion = sub["ERate"].where(sub["ERate"] > 0)
        summary = {
            "River": river,
            "migration_match_method": matched,
            "migration_reach_count": len(sub),
            "migration_positive_erosion_median": pos_erosion.median(),
            "migration_positive_erosion_p75": pos_erosion.quantile(0.75),
            "migration_erosion_max": sub.get("ERate_max", sub.get("ERate")).max(),
            "migration_accretion_median": sub.get(
                "ARate", pd.Series(dtype=float)
            ).median(),
            "migration_accretion_max": sub.get(
                "ARate_max", sub.get("ARate", pd.Series(dtype=float))
            ).max(),
            "migration_width_median": sub.get("width", pd.Series(dtype=float)).median(),
            "migration_mean_discharge_median": sub.get(
                "mean_Q", pd.Series(dtype=float)
            ).median(),
            "migration_mean_sediment_discharge_median": sub.get(
                "mean_Qs", pd.Series(dtype=float)
            ).median(),
            "migration_nodes_sum": sub.get("n_nodes", pd.Series(dtype=float)).sum(),
        }
        frames.append(summary)
        ids.append(
            {
                "River": river,
                "matched_basin_ids": ";".join(map(str, candidates)),
                "match_method": matched,
            }
        )
    pd.DataFrame(ids).to_csv(TABLES / "real_migration_match_audit.csv", index=False)
    return pd.DataFrame(frames)


def main() -> None:
    ensure_dirs()
    samples = pd.read_csv(
        PROCESSED / "arcticgro_with_stages.csv", parse_dates=["Date_parsed"]
    )
    hydro = load_hydroatlas()
    desc = pd.DataFrame(
        {
            "column": hydro.columns,
            "real_descriptive_name": [descriptive_name(c) for c in hydro.columns],
        }
    )
    desc.to_csv(TABLES / "hydroatlas_variable_dictionary.csv", index=False)
    migration = summarize_real(hydro)
    catchment = hydro.merge(migration, on="River", how="left")
    catchment.to_csv(PROCESSED / "catchment_features.csv", index=False)
    model = samples.merge(
        catchment, on="River", how="left", suffixes=("", "_catchment")
    )
    model.to_csv(PROCESSED / "modeling_table.csv", index=False)
    missing = (
        model[["River", "Delta14C_POC"]]
        .assign(target_available=model["Delta14C_POC"].notna())
        .groupby("River")
        .agg(rows=("River", "size"), target_available=("target_available", "sum"))
        .reset_index()
    )
    missing.to_csv(TABLES / "modeling_table_row_counts.csv", index=False)
    print(
        f"Wrote {PROCESSED/'modeling_table.csv'} with {len(model)} rows and {model.shape[1]} columns"
    )


if __name__ == "__main__":
    main()
