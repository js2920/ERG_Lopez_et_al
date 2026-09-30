#!/usr/bin/env python3
"""
Step 8: Automated Statistical Parity & Model Validation Suite.

Executes unit-level numerical assertions testing paper parity:
  - 20 matched patients (176,397 malignant cells)
  - 15/20 patients expand ERG (75.0%)
  - Paired Wilcoxon signed-rank p = 0.00271
  - Cohort mean shift: 11.1% -> 19.6%
  - Normal holdout concordance >= 85.0%
  - Malignancy mixing <= 5.0%
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audit_parity import run_statistical_parity_check
from src.config import (
    JOINT_OUTPUT_H5AD,
    JOINT_VALIDATION_TSV,
    PATIENT_METRICS_AL_TSV,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("Step08_Audit")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 8: Automated Statistical Parity Assertions.")
    parser.add_argument("--metrics-tsv", type=Path, default=PATIENT_METRICS_AL_TSV, help="Path to patient metrics TSV.")
    parser.add_argument("--joint-h5ad", type=Path, default=JOINT_OUTPUT_H5AD, help="Path to joint AnnData.")
    parser.add_argument("--validation-tsv", type=Path, default=JOINT_VALIDATION_TSV, help="Path to validation metrics TSV.")
    parser.add_argument("--out-json", type=Path, default=Path("./audit_report.json"), help="Output path for JSON report.")
    args = parser.parse_args()

    report = run_statistical_parity_check(
        metrics_tsv=args.metrics_tsv,
        joint_h5ad=args.joint_h5ad,
        validation_tsv=args.validation_tsv,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2))
    logger.info("Saved audit report to %s", args.out_json)

    if not report["passed"]:
        logger.error("Statistical parity check FAILED!")
        sys.exit(1)
    else:
        logger.info("Step 8 Complete: 100% STATISTICAL PARITY VERIFIED.")


if __name__ == "__main__":
    main()
