#!/usr/bin/env python3
"""
Publication Figure Renderer for Lopez et al. 

Renders:
  - Figure 4: Progenitor Latent Space (Joint G scANVI UMAPs: ERG MAGIC & LSC17 UCell)
  - Figure 11: Matched Diagnosis vs. Relapse ERG Fraction Diagonal Shift Plot

Formats: PNG (400 DPI), vector PDF (Type 42 fonts), and pure SVG text.
"""
from __future__ import annotations

import argparse
import io
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("RenderFigures")

# Continuous Tempo colormap
TEMPO_CMAP = LinearSegmentedColormap.from_list(
    "tempo",
    ["#F7F7DC", "#D8EBB8", "#B0DCA0", "#7DCB97", "#48B39A", "#249A96", "#147C86", "#0E5C6E", "#103C4E"]
)

# Driving mutation / subgroup color mapping
SUBGROUP_COLORS = {
    "KMT2A": "#7B2D8E",  # Deep purple
    "RUNX":  "#2A6F97",  # Steel blue
    "CBFB":  "#2A9D8F",  # Seafoam teal
    "FLT":   "#C9A227",  # Amber gold
    "Other": "#6D6875",  # Slate gray
}
SUBGROUP_ORDER = ["KMT2A", "RUNX", "CBFB", "FLT", "Other"]

POPULATION_MAP = {
    "HSC": "HSC",
    "Progenitor": "Progenitor",
    "GMP": "GMP",
    "CLP": "CLP",
    "Early.Erythrocyte": "Early erythroid",
    "Early.Basophil": "MEMP",
}
CENTROID_NUDGES = {
    "HSC": (-0.80, 0.55),
    "Progenitor": (0.85, 0.55),
    "GMP": (0.75, -0.35),
    "CLP": (-0.35, 0.65),
    "Early erythroid": (0.25, 0.65),
    "MEMP": (-0.85, -0.15),
}

# Embedded metrics for 100% out-of-the-box reproduction of Figure 11
PATIENT_METRICS_RAW = """patient_id	n_diagnosis	n_relapse	total_cells	frac_diagnosis	frac_relapse	mean_diagnosis	mean_relapse	subgroup
PAUUWT	5628	3635	9263	0.1242	0.2671	0.0962	0.1862	Other
PAUVAT	7639	6907	14546	0.1686	0.2735	0.1556	0.2229	Other
PAUXCD	4919	1284	6203	0.1803	0.3863	0.1303	0.2264	CBFB
PAVBCV	3440	3313	6753	0.1468	0.0960	0.1104	0.0574	Other
PAVBFN	7341	2809	10150	0.0692	0.1424	0.0739	0.1796	FLT
PAVDYE	4722	5101	9823	0.0945	0.2507	0.0847	0.1893	KMT2A
PAVGJG	6864	1193	8057	0.1403	0.4635	0.0905	0.3962	CBFB
PAVJXG	4138	4386	8524	0.0288	0.1585	0.0218	0.1249	KMT2A
PAWHFW	3461	4662	8123	0.0691	0.1941	0.0742	0.1417	FLT
PAWHKK	4212	4999	9211	0.1650	0.3951	0.0920	0.2792	KMT2A
PAWMKP	7402	3994	11396	0.0790	0.2053	0.0490	0.1235	RUNX
PAWNPG	2648	5300	7948	0.2330	0.1326	0.1873	0.0889	RUNX
PAWNYX	5150	5023	10173	0.1109	0.1348	0.0801	0.0923	KMT2A
PAWRVC	7056	981	8037	0.0003	0.0326	0.0002	0.0217	KMT2A
PAWTSD	5002	2299	7301	0.0890	0.0887	0.0717	0.0728	KMT2A
PAWUAE	3538	3017	6555	0.0215	0.0693	0.0130	0.0468	KMT2A
PAWWXU	5144	3948	9092	0.1732	0.3425	0.1506	0.2607	RUNX
PAXGDG	6465	3872	10337	0.1214	0.0253	0.1028	0.0243	CBFB
PAXLEE	6553	363	6916	0.0688	0.0551	0.0583	0.0567	FLT
PAXLWH	456	7533	7989	0.1272	0.2112	0.1177	0.1761	FLT"""


def apply_theme() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.2,
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "savefig.facecolor": "#FFFFFF",
        "savefig.dpi": 400,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "text.color": "#1D1D1B",
        "axes.labelcolor": "#1D1D1B",
        "xtick.color": "#1D1D1B",
        "ytick.color": "#1D1D1B",
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#2B2B28",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def render_figure_11(out_dir: Path, metrics_path: Optional[Path] = None) -> Path:
    apply_theme()
    if metrics_path and metrics_path.exists():
        df = pd.read_csv(metrics_path, sep="\t")
    else:
        df = pd.read_csv(io.StringIO(PATIENT_METRICS_RAW), sep="\t")

    x = df["frac_diagnosis"].to_numpy(float) * 100.0
    y = df["frac_relapse"].to_numpy(float) * 100.0
    labels = df["patient_id"].astype(str).tolist()
    cell_counts = df["total_cells"].to_numpy(float)
    genotypes = df["subgroup"].astype(str).tolist()

    x_lim, y_lim = (-1.0, 50.0), (-1.0, 50.0)
    fig, ax = plt.subplots(figsize=(5.8, 5.8))

    coord_min, coord_max = -1.0, 50.0
    ax.add_patch(Polygon([[coord_min, coord_min], [coord_min, coord_max], [coord_max, coord_max]],
                         facecolor="#E8F5EE", edgecolor="none", alpha=0.65, zorder=0))
    ax.add_patch(Polygon([[coord_min, coord_min], [coord_max, coord_min], [coord_max, coord_max]],
                         facecolor="#F3F4F6", edgecolor="none", alpha=0.45, zorder=0))
    ax.plot([coord_min, coord_max], [coord_min, coord_max], color="#7A8288", ls="--", lw=1.1, zorder=1)

    c_min, c_max = float(cell_counts.min()), float(cell_counts.max())
    norm_sizes = 35.0 + (cell_counts - c_min) / max(c_max - c_min, 1.0) * (180.0 - 35.0)

    is_increase = y > x
    colors = np.array([SUBGROUP_COLORS.get(g, SUBGROUP_COLORS["Other"]) for g in genotypes])

    for xi, yi, size, col, up in zip(x, y, norm_sizes, colors, is_increase):
        ax.scatter([xi], [yi], s=size, facecolors=col if up else "#FFFFFF", edgecolors=col, linewidths=1.4, alpha=0.92, zorder=3)

    dx_base = (x_lim[1] - x_lim[0]) * 0.025
    dy_base = (y_lim[1] - y_lim[0]) * 0.025
    offsets = []
    for xi, yi, lab in zip(x, y, labels):
        ha = "left"; va = "bottom"; dx = dx_base; dy = dy_base
        if lab == "PAVGJG": dy = -dy * 1.8; va = "top"
        elif lab == "PAUXCD": dx = -dx * 1.5; ha = "right"
        elif lab == "PAWHKK": dx = dx * 1.2; dy = -dy * 0.5
        elif lab == "PAWWXU": dx = dx * 1.2
        elif lab == "PAUVAT": dx = dx * 1.2; dy = -dy * 0.8
        elif lab == "PAUUWT": dx = -dx * 1.2; ha = "right"
        elif lab == "PAVDYE": dx = dx * 1.2; dy = dy * 0.5
        elif lab == "PAWMKP": dx = -dx * 1.3; ha = "right"
        elif lab == "PAWHFW": dx = dx * 1.2; dy = -dy * 1.5; va = "top"
        elif lab == "PAVJXG": dx = -dx * 1.0; ha = "right"; dy = dy * 0.2
        elif lab == "PAVBFN": dx = dx * 1.2; dy = -dy * 1.2; va = "top"
        elif lab == "PAXLWH": dx = dx * 1.2
        elif lab == "PAWNYX": dx = dx * 1.2; dy = -dy * 0.5
        elif lab == "PAWTSD": dx = -dx * 0.5; ha = "right"; dy = dy * 0.8; va = "bottom"
        elif lab == "PAWNPG": dx = -dx * 1.4; ha = "right"; dy = dy * 0.8
        elif lab == "PAVBCV": dx = dx * 1.2; dy = -dy * 1.2; va = "top"
        elif lab == "PAXGDG": dx = dx * 1.2; dy = dy * 0.5
        elif lab == "PAXLEE": dx = -dx * 1.2; ha = "right"; dy = -dy * 1.2; va = "top"
        elif lab == "PAWUAE": dx = dx * 1.2; dy = dy * 0.2
        elif lab == "PAWRVC": dx = dx * 1.2; dy = dy * 0.2
        offsets.append((xi + dx, yi + dy, ha, va))

    for (lx, ly, ha, va), lab, col in zip(offsets, labels, colors):
        ax.text(lx, ly, lab, fontsize=6.8, color=col, fontweight="semibold", ha=ha, va=va, zorder=4)

    mean_x, mean_y = float(x.mean()), float(y.mean())
    ax.scatter([mean_x], [mean_y], s=95, marker="D", facecolor="#D97706", edgecolor="#FFFFFF", lw=1.2, zorder=5)
    ax.annotate("", xy=(mean_x, mean_y), xytext=(mean_x, mean_x),
                arrowprops=dict(arrowstyle="->", color="#D97706", lw=1.8, shrinkA=2, shrinkB=4), zorder=4)

    n_up, n_total = int(is_increase.sum()), len(x)
    p_val = float(wilcoxon(y, x).pvalue)
    stat_text = f"{n_up}/{n_total} patients increase\nPaired Wilcoxon p = {p_val:.3g}\nCohort mean: {mean_x:.1f}% -> {mean_y:.1f}%"
    ax.text(0.04, 0.96, stat_text, transform=ax.transAxes, ha="left", va="top", fontsize=7.8,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#FFFFFF", edgecolor="#DDD9D0", alpha=0.92, lw=0.7), zorder=6)

    ax.text(0.05, 0.72, "Relapse Expansion\n(y > x)", transform=ax.transAxes, fontsize=7.2, color="#1B4332", fontweight="semibold", alpha=0.75, zorder=1)
    ax.text(0.65, 0.12, "Diagnosis Dominant\n(y < x)", transform=ax.transAxes, fontsize=7.2, color="#C46B4A", fontweight="semibold", alpha=0.75, zorder=1)

    ax.set_xlim(*x_lim); ax.set_ylim(*y_lim); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Diagnosis (% ERG+ cells)", fontsize=9.2, labelpad=5)
    ax.set_ylabel("Relapse (% ERG+ cells)", fontsize=9.2, labelpad=5)
    ax.set_title("ERG-positive malignant cells: Relapse vs Diagnosis", fontweight="semibold", pad=8, color="#2F6F6A")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SUBGROUP_COLORS[n], markeredgecolor=SUBGROUP_COLORS[n], markersize=8, label=n)
        for n in SUBGROUP_ORDER if n in set(genotypes)
    ] + [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#FFFFFF", markeredgecolor="#444444", markersize=8, label="Decrease (open)"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#D97706", markersize=7, label="Cohort mean"),
        Line2D([0], [0], color="#7A8288", ls="--", lw=1.1, label="Identity (y = x)"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=4, frameon=False, fontsize=7.8)

    fig.tight_layout(rect=(0.01, 0.12, 0.99, 0.98))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = out_dir / "11_all_malignant_ERG_fraction_DX_REL_mutations_manual_AL"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_base.with_suffix(f".{ext}"), dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved Figure 11: %s.{png,pdf,svg}", out_base)
    return out_base.with_suffix(".png")


def render_figure_04(
    out_dir: Path,
    cells_path: Optional[Path] = None,
    magic_path: Optional[Path] = None,
    rna_path: Optional[Path] = None,
) -> Path:
    apply_theme()
    # Resolve default paths
    repo_data = Path(__file__).resolve().parent / "data"
    cells_p = cells_path or (repo_data / "cells_progenitor_ERG_LSC17_G_A.tsv.gz")
    if not cells_p.exists() and os.environ.get("PAML_DATA_ROOT"):
        cells_p = Path(os.environ["PAML_DATA_ROOT"]) / "processed" / "stemness_scores_G" / "cells_progenitor_ERG_LSC17_G_A.tsv.gz"

    magic_p = magic_path or (repo_data / "cells_LSC17_signed_diff_A.tsv.gz")
    if not magic_p.exists() and os.environ.get("PAML_DATA_ROOT"):
        magic_p = Path(os.environ["PAML_DATA_ROOT"]) / "processed" / "stemness_scores_A" / "cells_LSC17_signed_diff_A.tsv.gz"

    if not cells_p.exists() or not magic_p.exists():
        logger.warning("Figure 4 tables not found at ./data/. Please place tables in data/ or provide custom paths.")
        return out_dir / "04_AML_diagnosis_relapse_ERG_MAGIC_LSC17_G_A.png"

    logger.info("Loading progenitor cells from %s...", cells_p)
    cells = pd.read_csv(cells_p, sep="\t")
    cells = cells[cells["dataset"].eq("pAML_Lambo") & cells["timepoint"].isin(["Diagnosis", "Relapse"])].copy()

    magic = pd.read_csv(magic_p, sep="\t", usecols=["cell_id", "ERG_MAGIC"])
    cells = cells.drop(columns=["ERG_MAGIC"], errors="ignore").merge(magic, on="cell_id", how="inner")

    # Population mapping
    if "population" not in cells.columns:
        if rna_path and rna_path.exists():
            rna = ad.read_h5ad(rna_path, backed="r")
            meta = pd.Series(rna.obs["Classified_Celltype"].astype(str).to_numpy(), index=rna.obs_names.astype(str))
            rna.file.close()
            cells["population"] = meta.reindex(cells["cell_id"].astype(str)).map(POPULATION_MAP).to_numpy()
        elif "Classified_Celltype" in cells.columns:
            cells["population"] = cells["Classified_Celltype"].map(POPULATION_MAP)

    if "population" in cells.columns and cells["population"].notna().any():
        centres = (
            cells[cells["population"].notna()]
            .groupby("population", sort=False)
            .agg(x=("umap1", "median"), y=("umap2", "median"), n=("cell_id", "size"))
            .reset_index()
        )
        centres = centres[centres["n"] >= 200].reset_index(drop=True)
    else:
        centres = pd.DataFrame(columns=["population", "x", "y", "n"])

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.4))
    xlim = (float(cells["umap1"].min()), float(cells["umap1"].max()))
    ylim = (float(cells["umap2"].min()), float(cells["umap2"].max()))

    limits = {
        "ERG_MAGIC": float(cells["ERG_MAGIC"].quantile(0.995)),
        "LSC17_pos_UCell": float(cells["LSC17_pos_UCell"].quantile(0.995)),
    }

    panels = [
        ("ERG_MAGIC", "MAGIC-smoothed ERG", "Diagnosis", "MAGIC ERG log1p(CP10k)"),
        ("ERG_MAGIC", "MAGIC-smoothed ERG", "Relapse", "MAGIC ERG log1p(CP10k)"),
        ("LSC17_pos_UCell", "LSC17-positive activity", "Diagnosis", "LSC17-positive UCell"),
        ("LSC17_pos_UCell", "LSC17-positive activity", "Relapse", "LSC17-positive UCell"),
    ]
    halo = [pe.withStroke(linewidth=3.2, foreground="#FFFFFF"), pe.Normal()]

    for ax, (col, title, timepoint, cbar_lbl) in zip(axes.ravel(), panels):
        norm = Normalize(0.0, max(limits[col], 1e-6), clip=True)
        sub = cells[cells["timepoint"].eq(timepoint)].sort_values(col)

        ax.scatter(sub["umap1"], sub["umap2"], color="#F1F1F1", s=1.2, alpha=0.4, linewidths=0, rasterized=True)
        handle = ax.scatter(sub["umap1"], sub["umap2"], c=sub[col], cmap=TEMPO_CMAP, norm=norm,
                            s=2.5, alpha=0.94, linewidths=0, rasterized=True)

        ax.set_title(f"{title} · {timepoint}   n={len(sub):,}", fontsize=11, pad=6)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3.8%", pad=0.06)
        cb = fig.colorbar(handle, cax=cax)
        cb.set_label(cbar_lbl, fontsize=7)
        cb.ax.tick_params(labelsize=6)

        for row in centres.itertuples(index=False):
            dx, dy = CENTROID_NUDGES.get(row.population, (0.0, 0.0))
            ax.text(row.x + dx, row.y + dy, row.population, ha="center", va="center", fontsize=8.0,
                    color="#102A36", fontweight="semibold", zorder=5, path_effects=halo, clip_on=False)

    fig.suptitle("MAGIC-smoothed ERG and LSC17-positive activity in pediatric AML progenitors", fontsize=13, y=0.98, fontweight="semibold")
    fig.text(0.5, 0.015, "Joint G scANVI coordinates; pediatric AML progenitor-tier cells from 20 paired diagnosis–relapse patients.",
             ha="center", fontsize=8.0, color="#333333")
    fig.subplots_adjust(bottom=0.08, wspace=0.08, hspace=0.16)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = out_dir / "04_AML_diagnosis_relapse_ERG_MAGIC_LSC17_G_A"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_base.with_suffix(f".{ext}"), dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved Figure 4: %s.{png,pdf,svg}", out_base)
    return out_base.with_suffix(".png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render publication figures.")
    parser.add_argument("--figures", choices=["all", "fig4", "fig11"], default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("./figures"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.figures in ("all", "fig4"):
        render_figure_04(args.output_dir)
    if args.figures in ("all", "fig11"):
        render_figure_11(args.output_dir)
    logger.info("Rendering complete. Outputs saved in %s", args.output_dir)


if __name__ == "__main__":
    main()
