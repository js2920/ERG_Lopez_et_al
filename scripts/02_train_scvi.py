#!/usr/bin/env python3
"""
Step 2: Unsupervised Generative Latent Modeling via scVI.

Fits scVI (2 hidden layers, 30 latent dimensions, ZINB likelihood) on the
weighted training dataset (3x healthy reference, 2x normal pAML bridge cells).
Batching is done on patient (aml_id) to strictly preserve temporal evolution.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    ACCELERATOR,
    BATCH_SIZE,
    GENE_SPACE_TXT,
    HEALTHY_INPUT_H5AD,
    N_LATENT,
    PAML_INPUT_H5AD,
    ROOT_DIR,
    SCVI_EPOCHS,
    SEEDS,
)
from src.harmonize_and_hvg import harmonize_healthy_metadata, harmonize_paml_metadata
from src.train_scvi_scanvi import assemble_weighted_training_anndata, partition_normal_holdout, assign_semi_supervised_roles

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("Step02_scVI")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2: Train Unsupervised scVI Model.")
    parser.add_argument("--seed", type=int, default=SEEDS[1], help="Random seed (default: 43).")
    parser.add_argument("--epochs", type=int, default=SCVI_EPOCHS, help="Maximum epochs.")
    parser.add_argument("--out-dir", type=Path, default=ROOT_DIR / "processed", help="Output model directory.")
    args = parser.parse_args()

    model_dir = args.out_dir / f"scvi_model_joint_G_seed{args.seed}"
    if model_dir.exists():
        logger.info("scVI model already exists at %s", model_dir)
        return

    import anndata as ad
    import scvi

    logger.info("Setting up AnnData for scVI pretraining with seed %d...", args.seed)
    # Check if raw files exist
    if not HEALTHY_INPUT_H5AD.exists() or not PAML_INPUT_H5AD.exists() or not GENE_SPACE_TXT.exists():
        logger.warning("Raw inputs not found at default paths. Precomputed joint models are stored in processed/.")
        return

    genes = GENE_SPACE_TXT.read_text().splitlines()
    healthy = ad.read_h5ad(HEALTHY_INPUT_H5AD)
    healthy.obs = healthy.obs.join(harmonize_healthy_metadata(healthy.obs["cell_state"]))
    paml = ad.read_h5ad(PAML_INPUT_H5AD)
    paml.obs = paml.obs.join(harmonize_paml_metadata(paml.obs["Classified_Celltype"]))
    paml_prog = paml[paml.obs["paml_tier"] == "Progenitor"].copy()

    holdout = partition_normal_holdout(paml_prog, seed=args.seed)
    assign_semi_supervised_roles(healthy, paml_prog, holdout)
    train = assemble_weighted_training_anndata(healthy, paml_prog, genes)

    scvi.settings.seed = args.seed
    scvi.model.SCVI.setup_anndata(train, batch_key="batch", layer="counts")
    model = scvi.model.SCVI(train, n_latent=N_LATENT, n_layers=2, gene_likelihood="zinb")
    model.train(
        max_epochs=args.epochs,
        accelerator=ACCELERATOR,
        devices="auto",
        batch_size=BATCH_SIZE,
        early_stopping=True,
        early_stopping_patience=30,
    )
    model.save(str(model_dir), overwrite=True, save_anndata=False)
    logger.info("Saved scVI model to %s", model_dir)


if __name__ == "__main__":
    main()
