#!/usr/bin/env python3
"""
Step 6: Render Publication Figure 4 (Progenitor UMAPs: ERG MAGIC & LSC17 UCell).

Outputs 4-panel grid in Joint-G scANVI latent space comparing MAGIC ERG and
LSC17-positive UCell activity at Diagnosis and Relapse (PNG 400 DPI, PDF, SVG).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    CELLS_PROGENITOR_TSV,
    MAGIC_SCORES_TSV,
    PAML_INPUT_H5AD,
)
from src.io_utils import load_progenitor_dataset
from src.render_figures import render_figure_04

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("Step06_Figure04")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 6: Render Publication Figure 4.")
    parser.add_argument("--cells-tsv", type=Path, default=CELLS_PROGENITOR_TSV, help="Path to progenitor cells table.")
    parser.add_argument("--magic-tsv", type=Path, default=MAGIC_SCORES_TSV, help="Path to MAGIC scores table.")
    parser.add_argument("--rna-h5ad", type=Path, default=PAML_INPUT_H5AD, help="Optional raw AnnData for cell types.")
    parser.add_argument("--output-dir", type=Path, default=Path("./figures"), help="Directory for rendered figures.")
    parser.add_argument("--prefix", type=str, default="04_AML_diagnosis_relapse_ERG_MAGIC_LSC17_G_A", help="File prefix.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cells = load_progenitor_dataset(
        cells_path=args.cells_tsv,
        magic_path=args.magic_tsv,
        rna_path=args.rna_h5ad if args.rna_h5ad.exists() else None,
    )
    out_png = render_figure_04(cells, output_dir=args.output_dir, file_prefix=args.prefix)
    logger.info("Step 6 Complete: Figure 4 rendered successfully at %s", out_png)


if __name__ == "__main__":
    main()
