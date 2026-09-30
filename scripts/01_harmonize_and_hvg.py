#!/usr/bin/env python3
"""
Step 1: Multi-Cohort Preprocessing, Label Harmonization, and Ranked HVG Selection.

Harmonizes author-called cell states into coarse lineages, establishes hierarchical
batch keys, and computes rank-aggregated consensus HVGs across 4 biological compartments.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add package root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    GENE_SPACE_TXT,
    HEALTHY_INPUT_H5AD,
    N_HVG,
    PAML_INPUT_H5AD,
)
from src.harmonize_and_hvg import (
    compute_ranked_hvgs,
    harmonize_healthy_metadata,
    harmonize_paml_metadata,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("Step01_Harmonize")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1: Harmonization & Ranked HVG Selection.")
    parser.add_argument("--healthy-h5ad", type=Path, default=HEALTHY_INPUT_H5AD, help="Path to healthy reference h5ad.")
    parser.add_argument("--paml-h5ad", type=Path, default=PAML_INPUT_H5AD, help="Path to pAML combined h5ad.")
    parser.add_argument("--out-genes", type=Path, default=GENE_SPACE_TXT, help="Path to write ranked HVG gene list.")
    parser.add_argument("--n-top", type=int, default=N_HVG, help="Number of consensus HVGs.")
    args = parser.parse_args()

    if not args.healthy_h5ad.exists() or not args.paml_h5ad.exists():
        logger.warning("Input files not present at default paths. If testing with precomputed files, gene space is already locked.")
        if args.out_genes.exists():
            logger.info("Found pre-existing gene space at %s (%d lines).", args.out_genes, len(args.out_genes.read_text().splitlines()))
        return

    import anndata as ad
    logger.info("Loading input AnnData objects...")
    healthy = ad.read_h5ad(args.healthy_h5ad)
    paml = ad.read_h5ad(args.paml_h5ad)

    # Harmonize metadata
    healthy.obs = healthy.obs.join(harmonize_healthy_metadata(healthy.obs["cell_state"]))
    paml_meta = harmonize_paml_metadata(paml.obs["Classified_Celltype"])
    paml.obs = paml.obs.join(paml_meta)

    # Filter to progenitor tier
    paml_prog = paml[paml.obs["paml_tier"] == "Progenitor"].copy()

    # Define 4 biological compartments
    compartments = {
        "Shlush_adult_HSPC": healthy[healthy.obs["dataset"] == "Shlush_cHSPC"],
        "Suo_fetal_progenitor": healthy[healthy.obs["dataset"] == "Suo_fetal"],
        "pAML_normal_bridge": paml_prog[paml_prog.obs["Malignant"] == "Normal"],
        "pAML_malignant_blast": paml_prog[paml_prog.obs["Malignant"] == "Malignant"],
    }

    genes = compute_ranked_hvgs(compartments, n_top=args.n_top)
    args.out_genes.parent.mkdir(parents=True, exist_ok=True)
    args.out_genes.write_text("\n".join(genes) + "\n")
    logger.info("Step 1 Complete: Saved %d ranked genes to %s", len(genes), args.out_genes)


if __name__ == "__main__":
    main()
