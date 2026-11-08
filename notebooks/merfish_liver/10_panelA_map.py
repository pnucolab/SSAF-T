#!/usr/bin/env python
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata as sd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C                                      # noqa: E402

TISSUE_W_IN, CROP_W_IN = 11.0, 6.0
EDGE_TISSUE, EDGE_CROP = 0.06, 0.5
BAR_UM, DPI = 100.0, 400
OUT = C.OUT / "panelA"
OUT.mkdir(parents=True, exist_ok=True)

PAL = json.load(open(C.SDATA.parent / "celltype_colors.json"))["colors"]
PAIR = {
    "periportal hepatocyte": "periportal hepatocyte",
    "pericentral hepatocyte": "pericentral hepatocyte",
    "pericentral LSEC": "pericentral endothelial cell",
    "periportal LSEC": "periportal endothelial cell",
    "Kupffer cell": "Kupffer cell",
    "hepatic stellate cell": "hepatic stellate cell",
    "cholangiocyte": "bile duct epithelial cell",
}


def panel(g, colors, xlim, ylim, path, width_in, edge, boxes=(), bar_um=None):
    h = width_in * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
    fig = plt.figure(figsize=(width_in, h), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    if len(g):
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
cells = sd.read_zarr(C.SDATA, selection=("tables",)).tables[C.TABLE].obs
lab_s = pd.Series(cells[C.LABEL_COL].astype(str).values,
                  index=cells.cell_id.astype(str).values).reindex(ss.index.astype(str))
missing = sorted(set(lab_s.dropna()) - set(PAL))
assert not missing, f"labels with no colour in celltype_colors.json: {missing}"
col_s = [PAL[v] for v in lab_s]
print(f"SSAF-T cells {len(ss):,}, {lab_s.nunique()} labels", flush=True)


pub = gpd.read_parquet(C.PUB_POLY)
if "geometry" not in pub.columns and "Geometry" in pub.columns:
    pub = pub.set_geometry("Geometry")
if "ZIndex" in pub:
    pub = pub[pub.ZIndex == 0]
pub = pub[~pub.geometry.is_empty].set_crs(None, allow_override=True).reset_index(drop=True)
meta = pd.read_csv(C.PUB_META)
lab_p = pub.EntityID.astype(str).map(
    pd.Series(meta.free_annotation.astype(str).values, index=meta.EntityID.astype(str).values))
REF_PAL = {pub_lab: PAL[ss_lab] for ss_lab, pub_lab in PAIR.items() if ss_lab in PAL}
unpaired = sorted(set(lab_p.dropna()) - set(REF_PAL))
print(f"published cells {len(pub):,}, {int(lab_p.notna().sum()):,} annotated; "
      f"no SSAF-T counterpart (grey): {unpaired}", flush=True)
col_p = [REF_PAL.get(v, "#dddddd") if isinstance(v, str) else "#dddddd" for v in lab_p]


b = np.array([ss.total_bounds, pub.total_bounds])
xlim = (float(b[:, 0].min()), float(b[:, 2].max()))
ylim = (float(b[:, 1].min()), float(b[:, 3].max()))
print(f"extent x {xlim[0]:.0f}..{xlim[1]:.0f}  y {ylim[0]:.0f}..{ylim[1]:.0f} µm", flush=True)

cx, cy, cs = C.CROP_A
for nm, g, cols in (("ssaft", ss, col_s), ("ref", pub, col_p)):
    panel(g, cols, xlim, ylim, OUT / f"{nm}_tissue.png", TISSUE_W_IN, EDGE_TISSUE)
    panel(g, cols, xlim, ylim, OUT / f"{nm}_tissue_box.png", TISSUE_W_IN, EDGE_TISSUE,
          boxes=[(cx, cy, cs, cs)], bar_um=500.0)
    m = g.cx[cx:cx + cs, cy:cy + cs]
    sub = [c for c, keep in zip(cols, g.index.isin(m.index)) if keep]
    panel(m, sub, (cx, cx + cs), (cy, cy + cs), OUT / f"{nm}_cropA.png", CROP_W_IN, EDGE_CROP,
          bar_um=BAR_UM if nm == "ssaft" else None)
