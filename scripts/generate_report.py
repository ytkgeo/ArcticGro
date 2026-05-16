#!/usr/bin/env python3
"""Generate a data audit report and SVG figures for the ArcticGro repository.

The script intentionally uses only the Python standard library so it can run in
restricted environments without downloading packages.
"""
from __future__ import annotations

import csv
import html
import math
import os
import statistics
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
REPORT_PATH = REPORT_DIR / "arcticgro_report.md"

NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}

RIVER_XLSX = [
    "Ob_20240322.xlsx",
    "Kolyma_20240322.xlsx",
    "Lena_20240322.xlsx",
    "Yenisey_20240322.xlsx",
    "Yukon_20240322.xlsx",
    "Mackenzie_20240322.xlsx",
]


def as_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or text.upper() in {"NA", "NAN", "NULL", "NONE"}:
        return None
    if text.startswith("<") or text.startswith(">"):
        text = text[1:].strip()
    try:
        return float(text)
    except ValueError:
        return None


def parse_date(value: str) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%y", "%m/%d/%Y"):
        try:
            return datetime.strptime(value.strip(), fmt)
        except ValueError:
            pass
    return None


def fmt_num(value: float | None, digits: int = 2) -> str:
    if value is None or math.isnan(value):
        return "NA"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:,.{digits}f}".rstrip("0").rstrip(".")


def get_shared_strings(zf: ZipFile) -> list[str]:
    try:
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    strings: list[str] = []
    for si in root.findall("a:si", NS):
        strings.append("".join(t.text or "" for t in si.findall(".//a:t", NS)))
    return strings


def workbook_sheets(zf: ZipFile) -> list[tuple[str, str]]:
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    relmap = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
    sheets: list[tuple[str, str]] = []
    for sheet in workbook.findall("a:sheets/a:sheet", NS):
        rid = sheet.attrib[f"{{{NS['r']}}}id"]
        target = relmap[rid]
        if not target.startswith("worksheets/"):
            target = "worksheets/" + target
        sheets.append((sheet.attrib["name"], "xl/" + target))
    return sheets


def column_index(ref: str) -> int:
    letters = "".join(ch for ch in ref if ch.isalpha())
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch.upper()) - 64
    return n - 1


def cell_value(cell: ET.Element, shared: list[str]) -> str:
    ctype = cell.attrib.get("t")
    if ctype == "inlineStr":
        return "".join(t.text or "" for t in cell.findall(".//a:t", NS))
    value_node = cell.find("a:v", NS)
    if value_node is None:
        return ""
    value = value_node.text or ""
    if ctype == "s":
        try:
            return shared[int(value)]
        except (ValueError, IndexError):
            return value
    return value


def read_xlsx_sheet(path: Path, sheet_name: str | None = None) -> list[dict[str, str]]:
    with ZipFile(path) as zf:
        shared = get_shared_strings(zf)
        sheets = workbook_sheets(zf)
        if sheet_name is None:
            sheet_path = sheets[0][1]
        else:
            sheet_path = next(p for n, p in sheets if n == sheet_name)
        root = ET.fromstring(zf.read(sheet_path))
        rows: list[list[str]] = []
        for row in root.findall("a:sheetData/a:row", NS):
            values: list[str] = []
            for cell in row.findall("a:c", NS):
                idx = column_index(cell.attrib.get("r", "A1"))
                while len(values) < idx:
                    values.append("")
                values.append(cell_value(cell, shared))
            rows.append(values)
        if not rows:
            return []
        headers = [h.strip() or f"column_{i}" for i, h in enumerate(rows[0])]
        out: list[dict[str, str]] = []
        for row in rows[1:]:
            if not any(str(v).strip() for v in row):
                continue
            out.append({headers[i]: row[i] if i < len(row) else "" for i in range(len(headers))})
        return out


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#263238} .axis{stroke:#455a64;stroke-width:1} .grid{stroke:#cfd8dc;stroke-width:.7} .label{font-size:12px} .title{font-size:18px;font-weight:bold} .small{font-size:10px}</style>',
        '<rect width="100%" height="100%" fill="white"/>',
    ]


def write_svg(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines + ["</svg>\n"]), encoding="utf-8")


def nice_max(value: float) -> float:
    if value <= 0:
        return 1
    exp = math.floor(math.log10(value))
    base = 10 ** exp
    for mult in (1, 2, 5, 10):
        if value <= mult * base:
            return mult * base
    return 10 * base


def bar_chart(path: Path, title: str, labels: list[str], values: list[float], ylabel: str, color: str = "#2f80ed") -> None:
    width, height = 920, 520
    left, right, top, bottom = 88, 30, 62, 110
    plot_w, plot_h = width - left - right, height - top - bottom
    ymax = nice_max(max(values) if values else 1)
    lines = svg_header(width, height)
    lines.append(f'<text x="{width/2}" y="30" text-anchor="middle" class="title">{html.escape(title)}</text>')
    for i in range(6):
        y = top + plot_h - (i / 5) * plot_h
        val = ymax * i / 5
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" class="small">{fmt_num(val, 0)}</text>')
    bar_gap = 8
    bar_w = max(8, (plot_w - bar_gap * (len(values) + 1)) / max(1, len(values)))
    for i, (label, val) in enumerate(zip(labels, values)):
        x = left + bar_gap + i * (bar_w + bar_gap)
        h = (val / ymax) * plot_h if ymax else 0
        y = top + plot_h - h
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}" rx="2"/>')
        lines.append(f'<text x="{x+bar_w/2:.1f}" y="{y-5:.1f}" text-anchor="middle" class="small">{fmt_num(val, 0)}</text>')
        lines.append(f'<text transform="translate({x+bar_w/2:.1f},{height-bottom+18}) rotate(-35)" text-anchor="end" class="small">{html.escape(label)}</text>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" class="axis"/>')
    lines.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" class="axis"/>')
    lines.append(f'<text transform="translate(22,{top+plot_h/2}) rotate(-90)" text-anchor="middle" class="label">{html.escape(ylabel)}</text>')
    write_svg(path, lines)


def line_chart(path: Path, title: str, series: dict[str, dict[int, float]], ylabel: str) -> None:
    width, height = 980, 560
    left, right, top, bottom = 82, 190, 58, 62
    plot_w, plot_h = width - left - right, height - top - bottom
    all_years = sorted({year for data in series.values() for year in data})
    all_values = [value for data in series.values() for value in data.values() if value is not None]
    xmin, xmax = min(all_years), max(all_years)
    ymin, ymax = 0, nice_max(max(all_values))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#17becf"]
    lines = svg_header(width, height)
    lines.append(f'<text x="{width/2}" y="30" text-anchor="middle" class="title">{html.escape(title)}</text>')
    for i in range(6):
        y = top + plot_h - (i / 5) * plot_h
        val = ymax * i / 5
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" class="small">{fmt_num(val, 0)}</text>')
    def xy(year: int, val: float) -> tuple[float, float]:
        x = left + ((year - xmin) / max(1, xmax - xmin)) * plot_w
        y = top + plot_h - ((val - ymin) / max(1, ymax - ymin)) * plot_h
        return x, y
    for idx, (name, data) in enumerate(sorted(series.items())):
        pts = [xy(y, data[y]) for y in sorted(data)]
        color = colors[idx % len(colors)]
        if len(pts) > 1:
            d = " ".join(f'{"M" if i == 0 else "L"}{x:.1f},{y:.1f}' for i, (x, y) in enumerate(pts))
            lines.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in pts[:: max(1, len(pts)//20)]:
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.3" fill="{color}"/>')
        ly = top + 20 + idx * 20
        lines.append(f'<rect x="{left+plot_w+22}" y="{ly-10}" width="12" height="12" fill="{color}"/>')
        lines.append(f'<text x="{left+plot_w+40}" y="{ly}" class="small">{html.escape(name)}</text>')
    for frac in (0, .25, .5, .75, 1):
        year = round(xmin + frac * (xmax - xmin))
        x = left + frac * plot_w
        lines.append(f'<text x="{x:.1f}" y="{top+plot_h+22}" text-anchor="middle" class="small">{year}</text>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" class="axis"/>')
    lines.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" class="axis"/>')
    lines.append(f'<text transform="translate(22,{top+plot_h/2}) rotate(-90)" text-anchor="middle" class="label">{html.escape(ylabel)}</text>')
    lines.append(f'<text x="{left+plot_w/2}" y="{height-20}" text-anchor="middle" class="label">Year</text>')
    write_svg(path, lines)


def scatter_plot(path: Path, title: str, points: list[tuple[float, float, str]], xlabel: str, ylabel: str, logx: bool = False, logy: bool = False) -> None:
    width, height = 900, 560
    left, right, top, bottom = 90, 170, 58, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    filtered = [(x, y, g) for x, y, g in points if x and y and x > 0 and y > 0]
    xs = [math.log10(x) if logx else x for x, _, _ in filtered]
    ys = [math.log10(y) if logy else y for _, y, _ in filtered]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmin == xmax: xmin -= 1; xmax += 1
    if ymin == ymax: ymin -= 1; ymax += 1
    pad_x = (xmax - xmin) * .06
    pad_y = (ymax - ymin) * .06
    xmin -= pad_x; xmax += pad_x; ymin -= pad_y; ymax += pad_y
    groups = sorted({g for _, _, g in filtered})
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#17becf", "#bcbd22"]
    color_map = {g: colors[i % len(colors)] for i, g in enumerate(groups)}
    lines = svg_header(width, height)
    lines.append(f'<text x="{width/2}" y="30" text-anchor="middle" class="title">{html.escape(title)}</text>')
    for i in range(6):
        x = left + (i/5)*plot_w
        y = top + plot_h - (i/5)*plot_h
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}" class="grid"/>')
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" class="grid"/>')
        raw_x = 10 ** (xmin + (i/5)*(xmax-xmin)) if logx else xmin + (i/5)*(xmax-xmin)
        raw_y = 10 ** (ymin + (i/5)*(ymax-ymin)) if logy else ymin + (i/5)*(ymax-ymin)
        lines.append(f'<text x="{x:.1f}" y="{top+plot_h+20}" text-anchor="middle" class="small">{fmt_num(raw_x, 1)}</text>')
        lines.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" class="small">{fmt_num(raw_y, 1)}</text>')
    for x, y, g in filtered:
        px = math.log10(x) if logx else x
        py = math.log10(y) if logy else y
        sx = left + ((px - xmin)/(xmax-xmin))*plot_w
        sy = top + plot_h - ((py - ymin)/(ymax-ymin))*plot_h
        lines.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="3" fill="{color_map[g]}" fill-opacity="0.62"/>')
    for i, g in enumerate(groups[:20]):
        ly = top + 18 + i * 18
        lines.append(f'<circle cx="{left+plot_w+24}" cy="{ly-4}" r="5" fill="{color_map[g]}"/>')
        lines.append(f'<text x="{left+plot_w+38}" y="{ly}" class="small">{html.escape(str(g))}</text>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" class="axis"/>')
    lines.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" class="axis"/>')
    lines.append(f'<text x="{left+plot_w/2}" y="{height-22}" text-anchor="middle" class="label">{html.escape(xlabel)}</text>')
    lines.append(f'<text transform="translate(24,{top+plot_h/2}) rotate(-90)" text-anchor="middle" class="label">{html.escape(ylabel)}</text>')
    write_svg(path, lines)


def histogram(path: Path, title: str, values: list[float], xlabel: str, bins: int = 20) -> None:
    if not values:
        return
    lo, hi = min(values), max(values)
    if lo == hi:
        lo -= 1; hi += 1
    step = (hi - lo) / bins
    counts = [0] * bins
    for value in values:
        idx = min(bins - 1, int((value - lo) / step))
        counts[idx] += 1
    labels = [fmt_num(lo + (i + .5) * step, 1) for i in range(bins)]
    bar_chart(path, title, labels, counts, "Rows", "#6c5ce7")


def summarize(values: Iterable[float]) -> dict[str, float | int | None]:
    vals = sorted(v for v in values if v is not None and not math.isnan(v))
    if not vals:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "min": vals[0],
        "max": vals[-1],
    }


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    river_rows: list[dict[str, str]] = []
    for file_name in RIVER_XLSX:
        river_rows.extend(read_xlsx_sheet(ROOT / file_name))

    by_river = defaultdict(list)
    annual_values = defaultdict(lambda: defaultdict(list))
    monthly_values = defaultdict(lambda: defaultdict(list))
    for row in river_rows:
        river = row.get("river", "Unknown") or "Unknown"
        discharge = as_float(row.get("discharge"))
        date = parse_date(row.get("date", ""))
        if discharge is not None:
            by_river[river].append(discharge)
            if date:
                annual_values[river][date.year].append(discharge)
                monthly_values[river][date.month].append(discharge)

    river_summary_rows = []
    for river in sorted(by_river):
        dates = [parse_date(r.get("date", "")) for r in river_rows if (r.get("river") or "Unknown") == river]
        dates = [d for d in dates if d]
        stats = summarize(by_river[river])
        river_summary_rows.append([
            river,
            f"{min(dates).date()} to {max(dates).date()}" if dates else "NA",
            fmt_num(stats["n"], 0),
            fmt_num(stats["mean"], 0),
            fmt_num(stats["median"], 0),
            fmt_num(stats["min"], 0),
            fmt_num(stats["max"], 0),
        ])

    bar_chart(FIG_DIR / "fig01_discharge_records_by_river.svg", "Daily discharge records by river", [r[0] for r in river_summary_rows], [float(str(r[2]).replace(',', '')) for r in river_summary_rows], "Valid discharge records")
    line_chart(FIG_DIR / "fig02_annual_mean_discharge.svg", "Annual mean discharge by river", {river: {year: statistics.fmean(vals) for year, vals in years.items() if vals} for river, years in annual_values.items()}, "Mean discharge (m³/s)")
    month_series = {river: {month: statistics.fmean(vals) for month, vals in months.items() if vals} for river, months in monthly_values.items()}
    line_chart(FIG_DIR / "fig03_monthly_discharge_climatology.svg", "Monthly discharge climatology by river", month_series, "Mean discharge (m³/s)")

    gro = read_csv(ROOT / "ArcticGRO-compilation.csv")
    gro_counts = Counter(row.get("River", "Unknown") or "Unknown" for row in gro)
    bar_chart(FIG_DIR / "fig04_arcticgro_samples_by_river.svg", "ArcticGRO chemistry samples by river", list(gro_counts.keys()), list(gro_counts.values()), "Samples", "#00a896")
    doc_points = []
    for row in gro:
        discharge = as_float(row.get("Discharge"))
        doc = as_float(row.get("DOC"))
        river = row.get("River", "Unknown") or "Unknown"
        if discharge and doc:
            doc_points.append((discharge, doc, river))
    scatter_plot(FIG_DIR / "fig05_doc_vs_discharge.svg", "DOC concentration vs discharge", doc_points, "Discharge (m³/s, log scale)", "DOC (mg/L, log scale)", logx=True, logy=True)

    hydro = read_xlsx_sheet(ROOT / "ArcticGRO-HydroAtlas.xlsx")
    hydro_area = [(row.get("Major_River", "Unknown"), as_float(row.get("UP_AREA"))) for row in hydro]
    hydro_area = [(name, val) for name, val in hydro_area if val is not None]
    bar_chart(FIG_DIR / "fig06_hydroatlas_upstream_area.svg", "HydroATLAS upstream area", [x[0] for x in hydro_area], [x[1] for x in hydro_area], "Upstream area (km²)", "#f39c12")

    morepoc = read_csv(ROOT / "MOREPOC_v1.1.csv")
    cont_counts = Counter(row.get("cont", "Unknown") or "Unknown" for row in morepoc)
    bar_chart(FIG_DIR / "fig07_morepoc_records_by_continent.svg", "MOREPOC records by continent", list(cont_counts.keys()), list(cont_counts.values()), "Records", "#e84393")
    poc_points = []
    for row in morepoc:
        spm = as_float(row.get("conc_spm"))
        poc = as_float(row.get("conc_poc"))
        cont = row.get("cont", "Unknown") or "Unknown"
        if spm and poc:
            poc_points.append((spm, poc, cont))
    scatter_plot(FIG_DIR / "fig08_morepoc_poc_vs_spm.svg", "MOREPOC POC vs suspended matter", poc_points, "SPM concentration (mg/L, log scale)", "POC concentration (mg/L, log scale)", logx=True, logy=True)

    real = read_csv(ROOT / "REAL_reach.csv")
    real_width = [as_float(row.get("width")) for row in real]
    real_width = [x for x in real_width if x is not None]
    histogram(FIG_DIR / "fig09_real_reach_width_histogram.svg", "REAL reach width distribution", real_width, "Width (m)", bins=24)
    erosion_points = []
    for row in real:
        width = as_float(row.get("width"))
        erate = as_float(row.get("ERate"))
        basin = row.get("basin_id", "Unknown") or "Unknown"
        if width and erate is not None and erate > 0:
            erosion_points.append((width, erate, basin[:5]))
    scatter_plot(FIG_DIR / "fig10_real_erosion_vs_width.svg", "REAL erosion rate vs reach width", erosion_points[:5000], "Width (m, log scale)", "Erosion rate (log scale)", logx=True, logy=True)

    bank_rows: list[dict[str, str]] = []
    with ZipFile(ROOT / "Geyman_bankfull_dataset (1).xlsx") as zf:
        for sheet_name, _ in workbook_sheets(zf):
            bank_rows.extend(read_xlsx_sheet(ROOT / "Geyman_bankfull_dataset (1).xlsx", sheet_name))
    bank_points = []
    for row in bank_rows:
        width = as_float(row.get("Width (m)"))
        discharge = as_float(row.get("Discharge (m3/s)"))
        comp = row.get("Compilation", "Unknown") or "Unknown"
        if width and discharge:
            bank_points.append((discharge, width, comp))
    scatter_plot(FIG_DIR / "fig11_bankfull_width_vs_discharge.svg", "Bankfull width vs discharge", bank_points, "Discharge (m³/s, log scale)", "Width (m, log scale)", logx=True, logy=True)

    data_inventory = []
    for path in sorted(ROOT.iterdir()):
        if path.suffix.lower() in {".csv", ".xlsx", ".pdf"}:
            if path.suffix.lower() == ".csv":
                rows = sum(1 for _ in path.open(encoding="utf-8-sig")) - 1
                detail = f"{rows:,} data rows"
            elif path.suffix.lower() == ".xlsx":
                with ZipFile(path) as zf:
                    names = [name for name, _ in workbook_sheets(zf)]
                detail = f"{len(names)} sheet(s): {', '.join(names[:5])}{'...' if len(names) > 5 else ''}"
            else:
                detail = "reference PDF"
            data_inventory.append([path.name, f"{path.stat().st_size/1024/1024:.2f} MB", detail])

    gro_doc = summarize(as_float(row.get("DOC")) for row in gro)
    gro_discharge = summarize(as_float(row.get("Discharge")) for row in gro)
    morepoc_poc = summarize(as_float(row.get("conc_poc")) for row in morepoc)
    morepoc_spm = summarize(as_float(row.get("conc_spm")) for row in morepoc)
    real_erate = summarize(as_float(row.get("ERate")) for row in real if (as_float(row.get("ERate")) or 0) > 0)
    bank_width = summarize(as_float(row.get("Width (m)")) for row in bank_rows)

    figures = [
        ("Figure 1", "Daily discharge record availability", "figures/fig01_discharge_records_by_river.svg"),
        ("Figure 2", "Annual mean discharge time series", "figures/fig02_annual_mean_discharge.svg"),
        ("Figure 3", "Mean monthly discharge climatology", "figures/fig03_monthly_discharge_climatology.svg"),
        ("Figure 4", "ArcticGRO sample coverage by river", "figures/fig04_arcticgro_samples_by_river.svg"),
        ("Figure 5", "DOC-discharge relationship", "figures/fig05_doc_vs_discharge.svg"),
        ("Figure 6", "HydroATLAS upstream basin area", "figures/fig06_hydroatlas_upstream_area.svg"),
        ("Figure 7", "MOREPOC geographic coverage", "figures/fig07_morepoc_records_by_continent.svg"),
        ("Figure 8", "MOREPOC POC versus SPM", "figures/fig08_morepoc_poc_vs_spm.svg"),
        ("Figure 9", "REAL reach-width distribution", "figures/fig09_real_reach_width_histogram.svg"),
        ("Figure 10", "REAL erosion rate versus width", "figures/fig10_real_erosion_vs_width.svg"),
        ("Figure 11", "Bankfull width versus discharge", "figures/fig11_bankfull_width_vs_discharge.svg"),
    ]

    report = [
        "# ArcticGro data run report",
        "",
        f"Generated with `python scripts/generate_report.py` on {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}.",
        "",
        "## Executive summary",
        "",
        f"- Processed {len(RIVER_XLSX)} daily discharge workbooks containing {sum(len(v) for v in by_river.values()):,} valid discharge observations across {len(by_river)} rivers.",
        f"- Processed {len(gro):,} ArcticGRO chemistry rows, {len(morepoc):,} MOREPOC rows, {len(real):,} REAL reach rows, {len(hydro):,} HydroATLAS rows, and {len(bank_rows):,} bankfull-geometry rows.",
        "- Generated all figures as dependency-free SVG files under `reports/figures/`.",
        "",
        "## Data inventory",
        "",
        markdown_table(["File", "Size", "Contents"], data_inventory),
        "",
        "## Daily discharge workbooks",
        "",
        markdown_table(["River", "Date range", "Valid Q rows", "Mean Q", "Median Q", "Min Q", "Max Q"], river_summary_rows),
        "",
        "## Cross-dataset numeric checks",
        "",
        markdown_table(
            ["Dataset / variable", "n", "mean", "median", "min", "max"],
            [
                ["ArcticGRO discharge", fmt_num(gro_discharge["n"], 0), fmt_num(gro_discharge["mean"], 2), fmt_num(gro_discharge["median"], 2), fmt_num(gro_discharge["min"], 2), fmt_num(gro_discharge["max"], 2)],
                ["ArcticGRO DOC", fmt_num(gro_doc["n"], 0), fmt_num(gro_doc["mean"], 2), fmt_num(gro_doc["median"], 2), fmt_num(gro_doc["min"], 2), fmt_num(gro_doc["max"], 2)],
                ["MOREPOC SPM", fmt_num(morepoc_spm["n"], 0), fmt_num(morepoc_spm["mean"], 2), fmt_num(morepoc_spm["median"], 2), fmt_num(morepoc_spm["min"], 2), fmt_num(morepoc_spm["max"], 2)],
                ["MOREPOC POC", fmt_num(morepoc_poc["n"], 0), fmt_num(morepoc_poc["mean"], 2), fmt_num(morepoc_poc["median"], 2), fmt_num(morepoc_poc["min"], 2), fmt_num(morepoc_poc["max"], 2)],
                ["REAL positive ERate", fmt_num(real_erate["n"], 0), fmt_num(real_erate["mean"], 4), fmt_num(real_erate["median"], 4), fmt_num(real_erate["min"], 4), fmt_num(real_erate["max"], 4)],
                ["Bankfull width", fmt_num(bank_width["n"], 0), fmt_num(bank_width["mean"], 2), fmt_num(bank_width["median"], 2), fmt_num(bank_width["min"], 2), fmt_num(bank_width["max"], 2)],
            ],
        ),
        "",
        "## Figures",
        "",
    ]
    for label, caption, rel_path in figures:
        report.extend([f"### {label}. {caption}", "", f"![{caption}]({rel_path})", ""])
    report.extend([
        "## Reproducibility notes",
        "",
        "- The workflow uses only the Python standard library and directly parses CSV/XLSX inputs.",
        "- SVG figures are written without external plotting libraries, which avoids package-download requirements in offline or restricted environments.",
        "- The HydroATLAS PDF is cataloged as a reference document and is not numerically parsed by this run.",
        "",
    ])
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)}")
    print(f"Wrote {len(figures)} figures to {FIG_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
