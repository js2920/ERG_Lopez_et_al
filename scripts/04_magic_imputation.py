#!/usr/bin/env python3
"""
Step 4: Within-Timepoint Data Diffusion (MAGIC Imputation).

Performs Markov affinity-based graph imputation of cells (MAGIC) on log1p(CP10k)
normalized expression strictly within Diagnosis and Relapse subsets. Prevents
across-condition information leakage while recovering continuous ERG expression.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MAGIC_SCORES_TSV, PAML_INPUT_H5AD
from src.magic_and_ucell import run_magic_smoothing_within_timepoints

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("Step04_MAGIC")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 4: Within-Timepoint MAGIC Imputation.")
    parser.add_argument("--paml-h5ad", type=Path, default=PAML_INPUT_H5AD, help="Path to pAML combined AnnData.")
    parser.add_argument("--out-tsv", type=Path, default=MAGIC_SCORES_TSV, help="Output path for MAGIC scores (TSV.GZ).")
    parser.add_argument("--genes", nargs="+", default=["ERG"], help="Genes to impute.")
    args = parser.parse_args()

    if args.out_tsv.exists():
        logger.info("MAGIC scores already computed and locked at %s", args.out_tsv)
        return

    if not args.paml_h5ad.exists():
        logger.warning("Raw AnnData not found at %s. Please download raw counts or use precomputed tables.", args.paml_h5ad)
        return

    import anndata as ad
    logger.info("Loading AnnData for within-timepoint MAGIC imputation...")
    adata = ad.read_h5ad(args.paml_h5ad)
    imputed = run_magic_smoothing_within_timepoints(adata, genes_to_impute=args.genes)
    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    imputed.to_csv(args.out_tsv, sep="\t", index=False, compression="gzip")
    logger.info("Step 4 Complete: Saved MAGIC imputed scores to %s", args.out_tsv)


if __name__ == "__main__":
    main()
