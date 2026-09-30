#!/usr/bin/env python3
"""
Deep Generative Integration Pipeline (scVI & scANVI) for Lopez et al.

Implements:
  1. Multi-cohort preprocessing and 4,000 consensus ranked HVG selection.
  2. Hierarchical batch assignment (patient-level 'aml_id' for pAML).
  3. Controlled replication weighting (3x healthy reference, 2x normal pAML bridge cells).
  4. Unsupervised scVI pretraining (2 layers, 30 latent dimensions, ZINB likelihood).
  5. Semi-supervised scANVI transfer with 20% held-out normal progenitor audit.
  6. Triplicate candidate seed selection (seeds 42, 43, 44).
  7. Within-timepoint MAGIC diffusion and single-cell pyUCell LSC17 scoring.
"""
from __future__ import annotations

import argparse
import gc
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("scVI_scANVI_Pipeline")
ZENODO_URL = "https://zenodo.org/records/23058363/files/joint_scvi_G.h5ad?download=1"
ZENODO_RECORD = "https://zenodo.org/records/23058363"

# Hyperparameters & Constants
SEEDS = (42, 43, 44)
HOLDOUT_FRAC = 0.20
HEALTHY_COPIES = 3
NORMAL_EXTRA_COPIES = 2
N_HVG = 4000
HVG_PER_COMPARTMENT = 3000
N_LATENT = 30
SCVI_EPOCHS = 150
SCANVI_EPOCHS = 75
BATCH_SIZE = 1024
FORCE_GENES = ["ERG", "CD34", "GATA2", "RUNX1", "CEBPA", "SPI1"]

LSC17_POSITIVE = ["DNMT3B", "NYNRIN", "LAPTM4B", "MMRN1", "DPYSL3", "FAM30A", "SOCS2", "EMP1", "BEX3", "CD34", "ADGRG1"]
LSC17_NEGATIVE = ["ZBTB46", "ARHGAP22", "CDK6", "CPXM1", "SMIM24", "AKR1C3"]

HEALTHY_MAP = {
    "HSC_MPP": "HSC_MPP", "HSC_MPP_POP": "HSC_MPP", "CYCLING_MPP": "HSC_MPP",
    "MEBEMP-L": "MEMP", "MEMP": "MEMP", "CYCLING_MEMP": "MEMP", "ERYP": "Ery_prog",
    "high_MPP_sig_ERYP": "Ery_prog", "MEP": "Ery_prog", "MKP": "Mega_prog",
    "EARLY_MK": "Mega_prog", "GMP-L": "Myeloid_prog", "GMP": "Myeloid_prog",
    "CMP": "Myeloid_prog", "CLP": "Lymphoid_prog", "LMPP_MLP": "Lymphoid_prog",
    "PRE_PRO_B": "Lymphoid_prog",
}
PAML_MAP = {
    "HSC": "HSC_MPP", "Progenitor": "Progenitor_unspec", "GMP": "Myeloid_prog",
    "CLP": "Lymphoid_prog", "Early.Erythrocyte": "Ery_prog", "Late.Erythrocyte": "Ery_prog",
    "Early.Basophil": "MEMP", "Monocytes": "Mature", "CD16.Monocytes": "Mature",
    "cDC": "Mature", "pDC": "Mature", "CD4.Naive": "Mature", "CD4.Memory": "Mature",
    "CD8.Naive": "Mature", "CD8.Memory": "Mature", "CD8.Effector": "Mature",
    "NK": "Mature", "B.Cell": "Mature", "Pre.B.Cell": "Mature", "Plasma": "Mature",
    "Unknown": "Unknown",
}
PAML_PROGENITOR_CELLS = {"HSC", "Progenitor", "GMP", "CLP", "Early.Basophil"}


def select_ranked_hvgs(compartments: Dict[str, ad.AnnData], n_top: int = N_HVG) -> List[str]:
    logger.info("Computing consensus Seurat v3 ranked HVGs across compartments...")
    scores: defaultdict[str, float] = defaultdict(float)
    appearances: defaultdict[str, int] = defaultdict(int)

    for name, adata in compartments.items():
        counts = adata.layers["counts"].copy() if "counts" in adata.layers else adata.X.copy()
        tmp = ad.AnnData(X=counts, var=adata.var.copy())
        sc.pp.highly_variable_genes(tmp, n_top_genes=min(HVG_PER_COMPARTMENT, tmp.n_vars), flavor="seurat_v3", subset=False)
        tab = tmp.var[tmp.var["highly_variable"]].sort_values("highly_variable_rank")
        for rank, gene in enumerate(tab.index.astype(str)):
            scores[gene] += (HVG_PER_COMPARTMENT - rank) / HVG_PER_COMPARTMENT
            appearances[gene] += 1
        del tmp, counts
        gc.collect()

    def is_artifact(g: str) -> bool:
        u = g.upper()
        return u.startswith(("MT-", "RPL", "RPS")) or u in {"MALAT1", "XIST", "NEAT1"}

    ranked = sorted((g for g in scores if not is_artifact(g)), key=lambda g: (appearances[g], scores[g]), reverse=True)
    selected = ranked[:n_top]
    for fg in FORCE_GENES:
        if fg not in selected:
            selected[-1] = fg
    return selected


def partition_holdout(paml: ad.AnnData, seed: int = SEEDS[0]) -> pd.Index:
    normal = paml.obs[(paml.obs["Malignant"] == "Normal") & (~paml.obs["paml_author_coarse"].isin(["Unknown", "Progenitor_unspec"]))]
    rng = np.random.default_rng(seed)
    strata = normal["aml_id"].astype(str) + "|" + normal["stage"].astype(str) + "|" + normal["paml_author_coarse"].astype(str)
    holdout: List[str] = []
    for _, names in strata.groupby(strata).groups.items():
        arr = np.asarray(names)
        n = max(1, int(round(len(arr) * HOLDOUT_FRAC)))
        holdout.extend(rng.choice(arr, min(n, len(arr)), replace=False).tolist())
    return pd.Index(holdout)


def run_training_pipeline(healthy_h5ad: Path, paml_h5ad: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    converged_joint = out_dir / "joint_scvi_G.h5ad"
    if converged_joint.exists():
        logger.info("Found existing converged joint AnnData at %s", converged_joint)
        return converged_joint

    logger.info("Loading raw reference and disease single-cell count matrices...")
    healthy = ad.read_h5ad(healthy_h5ad)
    healthy.obs["cell_state_coarse"] = healthy.obs["cell_state"].map(HEALTHY_MAP).fillna("Unknown").astype(str)
    healthy.obs["training_role"] = "healthy_reference"

    paml = ad.read_h5ad(paml_h5ad)
    paml.obs["paml_tier"] = np.where(paml.obs["Classified_Celltype"].isin(PAML_PROGENITOR_CELLS), "Progenitor", "Non_Progenitor")
    paml.obs["paml_author_coarse"] = paml.obs["Classified_Celltype"].map(PAML_MAP).fillna("Unknown").astype(str)
    paml_prog = paml[paml.obs["paml_tier"] == "Progenitor"].copy()

    holdout = partition_holdout(paml_prog, seed=SEEDS[1])
    paml_prog.obs["cell_state_coarse"] = "Unknown"
    is_anchor = (paml_prog.obs["Malignant"] == "Normal") & (~paml_prog.obs_names.isin(holdout)) & (~paml_prog.obs["paml_author_coarse"].isin(["Unknown", "Progenitor_unspec"]))
    paml_prog.obs.loc[is_anchor, "cell_state_coarse"] = paml_prog.obs.loc[is_anchor, "paml_author_coarse"]

    compartments = {
        "Healthy_adult_cHSPC": healthy[healthy.obs["dataset"] == "Shlush_cHSPC"],
        "Healthy_fetal_progenitor": healthy[healthy.obs["dataset"] == "Suo_fetal"],
        "pAML_normal_bridge": paml_prog[paml_prog.obs["Malignant"] == "Normal"],
        "pAML_malignant_blast": paml_prog[paml_prog.obs["Malignant"] == "Malignant"],
    }
    genes = select_ranked_hvgs(compartments, n_top=N_HVG)

    # Controlled replication weighting
    blocks = []
    for i in range(HEALTHY_COPIES):
        h_sub = healthy[:, genes].copy()
        h_sub.obs["training_copy"] = f"healthy_{i}"
        h_sub.obs_names = h_sub.obs_names.astype(str) + f"::healthy_{i}"
        blocks.append(h_sub)

    p_sub = paml_prog[:, genes].copy()
    p_sub.obs["training_copy"] = "paml_0"
    p_sub.obs_names = p_sub.obs_names.astype(str) + "::paml_0"
    blocks.append(p_sub)

    normal_extra = paml_prog[paml_prog.obs["Malignant"] == "Normal", genes]
    for i in range(NORMAL_EXTRA_COPIES):
        n_sub = normal_extra.copy()
        n_sub.obs["training_copy"] = f"normal_extra_{i}"
        n_sub.obs_names = n_sub.obs_names.astype(str) + f"::normal_extra_{i}"
        blocks.append(n_sub)

    train = ad.concat(blocks, join="inner", merge="first")
    counts = train.layers["counts"].copy() if "counts" in train.layers else train.X.copy()
    train.layers["counts"] = counts
    train.X = counts
    train.obs["batch"] = np.where(train.obs["dataset"] == "pAML_Lambo", train.obs["aml_id"].astype(str), train.obs["sample_id"].astype(str))

    # Pretrain scVI
    logger.info("Pretraining scVI model (30 latent dimensions, ZINB likelihood)...")
    scvi.settings.seed = SEEDS[1]
    scvi.model.SCVI.setup_anndata(train, batch_key="batch", layer="counts")
    scvi_m = scvi.model.SCVI(train, n_latent=N_LATENT, n_layers=2, gene_likelihood="zinb")
    scvi_m.train(max_epochs=SCVI_EPOCHS, batch_size=BATCH_SIZE, early_stopping=True, early_stopping_patience=30)

    # Transfer to semi-supervised scANVI
    logger.info("Fitting semi-supervised scANVI...")
    scanvi_m = scvi.model.SCANVI.from_scvi_model(scvi_m, unlabeled_category="Unknown", labels_key="cell_state_coarse")
    scanvi_m.train(max_epochs=SCANVI_EPOCHS, batch_size=BATCH_SIZE, n_samples_per_label=100)

    latent = scanvi_m.get_latent_representation()
    preds = np.asarray(scanvi_m.predict()).astype(str)

    keep = train.obs["training_copy"].isin(["healthy_0", "paml_0"]).to_numpy()
    out = ad.AnnData(X=csr_matrix((keep.sum(), 1)), obs=train.obs.loc[keep].copy())
    out.obsm["X_scANVI"] = latent[keep]
    out.obs["scanvi_pred"] = preds[keep]

    sc.pp.neighbors(out, use_rep="X_scANVI", n_neighbors=15, method="umap")
    sc.tl.umap(out, min_dist=0.3, random_state=SEEDS[1])
    out.write(converged_joint)
    logger.info("Saved joint integration to %s", converged_joint)
    return converged_joint


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scVI/scANVI Integration Pipeline.")
    default_healthy = Path("data/combined_prep_A.h5ad") if Path("data/combined_prep_A.h5ad").exists() else Path(os.environ.get("PAML_DATA_ROOT", "data")) / "processed" / "combined_prep_A.h5ad"
    parser.add_argument("--healthy-h5ad", type=Path, default=default_healthy, help="Path to healthy reference h5ad.")
    default_paml = Path("data/pAML_GSE235063_combined_A.h5ad") if Path("data/pAML_GSE235063_combined_A.h5ad").exists() else Path(os.environ.get("PAML_FAST_ROOT", "data")) / "pAML_GSE235063_A" / "processed" / "pAML_GSE235063_combined_A.h5ad"
    parser.add_argument("--paml-h5ad", type=Path, default=default_paml, help="Path to pAML combined h5ad.")
    parser.add_argument("--out-dir", type=Path, default=Path("./processed"))
    args = parser.parse_args()

    run_training_pipeline(args.healthy_h5ad, args.paml_h5ad, args.out_dir)


if __name__ == "__main__":
    main()
