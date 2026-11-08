#!/usr/bin/env python
import json
import sys
from pathlib import Path

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C                                      # noqa: E402

CROP = (919.0, 1152.0, 800.0)
MAX_Y = 4400.0
TISSUE_W_IN, CROP_W_IN = 11.0, 6.0
EDGE_TISSUE, EDGE_CROP = 0.06, 0.5
BAR_UM, DPI = 100.0, 400
OUT = C.OUT / "panelA"
OUT.mkdir(parents=True, exist_ok=True)

PAL = json.load(open(C.SDATA.parent / "celltype_colors.json"))["colors"]


def panel(g, colors, xlim, ylim, path, width_in, edge, boxes=(), bar_um=None):
    h = width_in * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
    fig = plt.figure(figsize=(width_in, h), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    g.plot(ax=ax, color=colors, edgecolor="white", linewidth=edge, aspect=None)
    for x, y, w, hh in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, hh, fill=False, ec="black", lw=1.2))
    if bar_um:
        x = xlim[1] - .04 * (xlim[1] - xlim[0]) - bar_um
        y = ylim[1] - .05 * (ylim[1] - ylim[0])
        ax.plot([x, x + bar_um], [y, y], color="black", lw=2)
    ax.set_xlim(*xlim)
    ax.set_ylim(ylim[1], ylim[0])
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
    print("->", path.name, flush=True)


ss = gpd.read_parquet(C.SDATA / f"shapes/{C.SHAPES_2D}/shapes.parquet").set_crs(None, allow_override=True)
import spatialdata as sd
cells = sd.read_zarr(C.SDATA, selection=("tables",)).tables[C.TABLE].obs
lab_s = pd.Series(cells[C.LABEL_COL].astype(str).values,
                  index=cells.cell_id.astype(str).values).reindex(ss.index.astype(str))
missing = sorted(set(lab_s.dropna()) - set(PAL))
assert not missing, f"labels with no colour in celltype_colors.json: {missing}"
col_s = [PAL[v] for v in lab_s]
print(f"SSAF-T cells {len(ss):,}, {lab_s.nunique()} labels", flush=True)


pv = pd.read_csv(C.BAYSOR_TX.parent / "polygon.csv")
pv[["x_location", "y_location"]] *= C.PX_UM
rings, nums = [], []
for n, g in pv.groupby("polygon_number"):
    q = g[["x_location", "y_location"]].to_numpy()
    if len(q) >= 3:
        rings.append(Polygon(q))
        nums.append(n)
pub = gpd.GeoDataFrame({"polygon_number": nums}, geometry=rings)
ref = ad.read_h5ad(C.REF_H5AD, backed="r")
cen = np.stack([pub.geometry.centroid.x.values, pub.geometry.centroid.y.values], 1)
d, k = cKDTree(np.stack([ref.obs.x_centroid.values, ref.obs.y_centroid.values], 1)).query(cen, k=1)
lab_p = pd.Series(ref.obs.celltype_plot.astype(str).values[k]).where(d < 5.0)
print(f"published cells {len(pub):,}, {int(lab_p.notna().sum()):,} matched to an annotated cell "
      f"within 5 µm", flush=True)


PAIR = {
    "proximal tubule cell S2": "PTS2", "proximal tubule cell S1": "PTS1",
    "thick ascending limb cell": "TAL", "proximal tubule cell S3": "PTS3",
    "glomerular endothelial cell": "Glom-EC", "endothelial cell": "EC",
    "macrophage": "Immune", "fibroblast": "Fib",
    "collecting duct principal cell": "PC", "distal convoluted tubule cell": "DCT",
    "connecting tubule cell": "CNT", "mural cell": "Per-SMC",
    "type B intercalated cell": "ICB", "descending thin limb cell": "DTL",
    "type A intercalated cell": "ICA", "podocyte": "Pod",
    "urothelial cell": "Uro", "parietal epithelial cell": "PEC",
}
REF_PAL = dict(zip(ref.obs.celltype_plot.cat.categories,
                   [str(c) for c in ref.uns["celltype_plot_colors"]]))
REF_PAL.update({pub_lab: PAL[ss_lab] for ss_lab, pub_lab in PAIR.items() if ss_lab in PAL})
unpaired = sorted(set(ref.obs.celltype_plot.astype(str).unique()) - set(PAIR.values()))
print(f"reference labels on the authors' own colour (no SSAF-T counterpart): {unpaired}",
      flush=True)
col_p = [REF_PAL.get(v, "#dddddd") if isinstance(v, str) else "#dddddd" for v in lab_p]


b = np.array([ss.total_bounds, pub.total_bounds])
xlim = (float(b[:, 0].min()), float(b[:, 2].max()))
ylim = (float(b[:, 1].min()), min(float(b[:, 3].max()), MAX_Y))
print(f"extent x {xlim[0]:.0f}..{xlim[1]:.0f}  y {ylim[0]:.0f}..{ylim[1]:.0f} µm", flush=True)

cx, cy, cs = CROP
for nm, g, cols in (("ssaft", ss, col_s), ("ref", pub, col_p)):
    panel(g, cols, xlim, ylim, OUT / f"{nm}_tissue.png", TISSUE_W_IN, EDGE_TISSUE)
    panel(g, cols, xlim, ylim, OUT / f"{nm}_tissue_box.png", TISSUE_W_IN, EDGE_TISSUE,
          boxes=[(cx, cy, cs, cs)], bar_um=500.0)
    m = g.cx[cx:cx + cs, cy:cy + cs]
    sub = [c for c, keep in zip(cols, g.index.isin(m.index)) if keep]
    panel(m, sub, (cx, cx + cs), (cy, cy + cs), OUT / f"{nm}_cropC.png", CROP_W_IN, EDGE_CROP,
          bar_um=BAR_UM if nm == "ssaft" else None)
