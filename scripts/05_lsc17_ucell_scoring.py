#!/usr/bin/env python3
"""
Step 5: Single-Cell Direction-Aware LSC17 Stemness Scoring (UCell).

Computes single-cell Mann-Whitney rank scores for the 17-gene leukemic stem cell
signature (Ng et al., Nature 2016):
  LSC17_signed_diff = UCell(11 positive genes) - UCell(6 negative genes)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    LSC17_NEGATIVE_GENES,
    LSC17_POSITIVE_GENES,
    PAML_INPUT_H5AD,
    ROOT_DIR,
)
from src.magic_and_ucell import compute_direction_aware_ucell_lsc17

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("Step05_UCell")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 5: Direction-Aware LSC17 Stemness Scoring.")
    parser.add_argument("--paml-h5ad", type=Path, default=PAML_INPUT_H5AD, help="Path to pAML AnnData.")
    parser.add_argument("--out-dir", type=Path, default=ROOT_DIR / "processed" / "stemness_scores_A", help="Output directory.")
    args = parser.parse_args()

    out_file = args.out_dir / "cells_LSC17_signed_diff_A.tsv.gz"
    if out_file.exists():
        logger.info("LSC17 scores already computed and locked at %s", out_file)
        return

    if not args.paml_h5ad.exists():
        logger.warning("Raw AnnData not found at %s. Please use precomputed tables.", args.paml_h5ad)
        return

    import anndata as ad
    logger.info("Computing UCell LSC17 scores on raw counts...")
    adata = ad.read_h5ad(args.paml_h5ad)
    scored = compute_direction_aware_ucell_lsc17(
        adata,
        pos_genes=LSC17_POSITIVE_GENES,
        neg_genes=LSC17_NEGATIVE_GENES,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_file, sep="\t", index=False, compression="gzip")
    logger.info("Step 5 Complete: Saved LSC17 scores to %s", out_file)


if __name__ == "__main__":
    main()
