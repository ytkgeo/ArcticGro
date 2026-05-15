#!/usr/bin/env python
"""Run the full ArcticGRO Δ14C-POC workflow in order."""

from __future__ import annotations
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "01_load_and_clean.py",
    "02_discharge_stage.py",
    "03_catchment_features.py",
    "04_derive_source_access.py",
    "05_model_validate.py",
    "06_figures.py",
    "07_report.py",
]
ROOT = Path(__file__).resolve().parents[1]
for script in SCRIPTS:
    print(f"\n=== {script} ===")
    subprocess.run([sys.executable, str(ROOT / "scripts" / script)], check=True)
