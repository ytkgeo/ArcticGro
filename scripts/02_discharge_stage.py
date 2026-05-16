#!/usr/bin/env python
"""Build daily discharge climatologies and assign six hydrograph stages to samples."""

from __future__ import annotations
import numpy as np
import pandas as pd
from common import (
    PROCESSED,
    TABLES,
    RIVERS,
    DISCHARGE_FILES,
    ensure_dirs,
    raw_path,
    harmonize_river,
)

STAGE_LABELS = {
    "S1": "S1 early-year frozen low-flow",
    "S2": "S2 melting surge",
    "S3": "S3 post-surge drop",
    "S4": "S4 precipitation/melt maintenance",
    "S5": "S5 pre-freeze drop",
    "S6": "S6 late-year frozen low-flow",
}


def circular_smooth(values: pd.Series, window: int = 15) -> pd.Series:
    v = values.reindex(range(1, 367)).interpolate(limit_direction="both")
    ext = pd.concat([v.tail(window), v, v.head(window)], ignore_index=True)
    sm = (
        ext.rolling(window * 2 + 1, center=True, min_periods=1)
        .median()
        .iloc[window : window + 366]
    )
    sm.index = range(1, 367)
    return sm


def stage_from_doy_q(row: pd.Series, peak_doy: int, low_threshold: float) -> str:
    doy = int(row["doy"])
    qstar = row.get("Qstar", np.nan)
    rising = row.get("rising_limb", 0) == 1
    if doy <= max(80, peak_doy - 45) and (pd.isna(qstar) or qstar <= low_threshold):
        return "S1"
    if doy <= peak_doy and (rising or qstar > low_threshold):
        return "S2"
    if doy <= min(peak_doy + 45, 230) and not rising:
        return "S3"
    if doy <= 250:
        return "S4"
    if doy <= 315 and (pd.isna(qstar) or qstar > low_threshold):
        return "S5"
    return "S6"


def process_river(river: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_excel(raw_path(DISCHARGE_FILES[river]))
    df.columns = [str(c).strip() for c in df.columns]
    df["river"] = (
        df.get("river", river).map(harmonize_river) if "river" in df else river
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["discharge"] = pd.to_numeric(df["discharge"], errors="coerce")
    df = df[df["date"].notna() & df["discharge"].gt(0)].sort_values("date")
    df["doy"] = df["date"].dt.dayofyear.clip(upper=366)
    med = df.groupby("doy")["discharge"].median()
    clim = circular_smooth(med)
    q_low = float(np.nanpercentile(clim, 5))
    q_peak = float(np.nanmax(clim))
    denom = q_peak - q_low if q_peak > q_low else np.nan
    cdf = pd.DataFrame(
        {"River": river, "doy": clim.index, "Q_clim_median": clim.values}
    )
    cdf["Q_low"] = q_low
    cdf["Q_peak"] = q_peak
    cdf["Qstar"] = (
        ((cdf["Q_clim_median"] - q_low) / denom).clip(0, 1)
        if denom == denom
        else np.nan
    )
    cdf["dQdt"] = (
        cdf["Q_clim_median"]
        .diff()
        .fillna(cdf["Q_clim_median"].iloc[0] - cdf["Q_clim_median"].iloc[-1])
    )
    cdf["rising_limb"] = (cdf["dQdt"] > 0).astype(int)
    cdf["falling_limb"] = (cdf["dQdt"] < 0).astype(int)
    peak_doy = int(cdf.loc[cdf["Q_clim_median"].idxmax(), "doy"])
    low_threshold = 0.15
    cdf["discharge_stage_code"] = cdf.apply(
        stage_from_doy_q, axis=1, peak_doy=peak_doy, low_threshold=low_threshold
    )
    cdf["discharge_stage"] = cdf["discharge_stage_code"].map(STAGE_LABELS)
    boundaries = (
        cdf.groupby("discharge_stage_code")
        .agg(
            start_doy=("doy", "min"),
            end_doy=("doy", "max"),
            qstar_min=("Qstar", "min"),
            qstar_max=("Qstar", "max"),
        )
        .reset_index()
    )
    boundaries.insert(0, "River", river)
    return cdf, boundaries


def main() -> None:
    ensure_dirs()
    climatologies, boundaries = zip(*(process_river(r) for r in RIVERS))
    clim = pd.concat(climatologies, ignore_index=True)
    bnd = pd.concat(boundaries, ignore_index=True)
    clim.to_csv(PROCESSED / "daily_discharge_climatology.csv", index=False)
    bnd.to_csv(TABLES / "stage_boundaries.csv", index=False)
    samples = pd.read_csv(
        PROCESSED / "arcticgro_clean.csv", parse_dates=["Date_parsed"]
    )
    samples = samples.merge(
        clim, left_on=["River", "Julian_Day"], right_on=["River", "doy"], how="left"
    )
    samples = samples.rename(columns={"Q_clim_median": "Q_daily_climatology"})
    samples.to_csv(PROCESSED / "arcticgro_with_stages.csv", index=False)
    print(f"Wrote {PROCESSED/'arcticgro_with_stages.csv'} with {len(samples)} rows")


if __name__ == "__main__":
    main()
