#!/usr/bin/env python3
"""
Step 7: Render Publication Figure 11 (Patient-Level ERG Fraction Diagonal Shift).

Outputs diagonal shift plot comparing Diagnosis vs. Relapse ERG-positive blast
fraction across 20 matched cohorts with genomic subgroup coloring (PNG, PDF, SVG).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PATIENT_METRICS_AL_TSV
from src.io_utils import load_paired_patient_metrics
from src.render_figures import render_figure_11

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("Step07_Figure11")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 7: Render Publication Figure 11.")
    parser.add_argument("--metrics-tsv", type=Path, default=PATIENT_METRICS_AL_TSV, help="Path to paired patient metrics.")
    parser.add_argument("--output-dir", type=Path, default=Path("./figures"), help="Directory for rendered figures.")
    parser.add_argument("--prefix", type=str, default="11_all_malignant_ERG_fraction_DX_REL_mutations_manual_AL", help="File prefix.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = load_paired_patient_metrics(args.metrics_tsv)
    out_png, stats = render_figure_11(summary, output_dir=args.output_dir, file_prefix=args.prefix)
    logger.info(
        "Step 7 Complete: %d/%d (%.1f%%) patients expand ERG at relapse (Wilcoxon p=%.4g)",
        stats["n_expanded"], stats["n_patients"], stats["expansion_rate"] * 100.0, stats["wilcoxon_p"]
    )
    logger.info("Figure 11 rendered successfully at %s", out_png)


if __name__ == "__main__":
    main()
