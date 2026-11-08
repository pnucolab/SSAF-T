#!/usr/bin/env python
import json

import numpy as np
import pandas as pd
import geopandas as gpd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from scipy import ndimage as ndi
from skimage.draw import polygon as dpoly
from skimage.segmentation import find_boundaries

from common import *

plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
pal = palette()
rgbf = lambda h: [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]

G = np.load(CACHE / "geometry.npz"); depth, tissue = G["depth"], G["tissue"]
V = np.load(CACHE / "cortex_valid.npz"); valid, cw = V["valid"], V["cw"]
S = json.load(open(CACHE / "stats_cortex.json"))
cm, names = load_map()
b = load_bin1()
LAYG = {"Cux2": "L2/3", "Rorb": "L4", "Fezf2": "L5", "Foxp2": "L6", "Ctgf": "L6b"}
GENE_LAYER = {"Cux2": ["L2/3 IT glutamatergic neuron"], "Rorb": ["L4 IT glutamatergic neuron"],
              "Fezf2": ["L5 ET glutamatergic neuron"],


              "Foxp2": ["L6 CT glutamatergic neuron"],
              "Ctgf": ["L6b glutamatergic neuron"]}
LAY = [l for l in LAYERS if l != "L6 IT glutamatergic neuron"]
gr = {g: raster(b, [g]) for g in LAYG}


wins = pd.DataFrame(S["columns"]); vw = wins[wins.valid].win.values
rorb = np.where(valid & (depth >= 0.37) & (depth < 0.57), gr["Rorb"], 0)
score = {w: rorb[cw == w].sum() / max(((cw == w) & (depth >= 0.37) & (depth < 0.57)).sum(), 1) for w in vw}
trip = [(w, w + 1, w + 2) for w in vw if w + 1 in vw and w + 2 in vw]
best = max(trip, key=lambda t: sum(score[x] for x in t))
colmask = np.isin(cw, best)
gx, gy = np.gradient(ndi.gaussian_filter(np.where(tissue, depth, 0), 3))
ang = np.degrees(np.arctan2(gy[colmask].mean(), gx[colmask].mean()))
xs, ys = np.nonzero(colmask); pad = 20
x0, x1, y0, y1 = xs.min() - pad, xs.max() + pad, ys.min() - pad, ys.max() + pad
print("column", best, "angle %.1f" % ang, flush=True)

crop1 = lambda a: a[x0 * PX:x1 * PX, y0 * PX:y1 * PX]
rot = lambda a, order=0: ndi.rotate(a.T, -(90 - ang), reshape=True, order=order, cval=0)

lab_r = rot(crop1(cm).astype(np.int32))
msk_r = rot(np.kron(colmask[x0:x1, y0:y1], np.ones((PX, PX), bool)).astype(np.uint8)) > 0
rr, cc = np.nonzero(msk_r); r0, r1, c0, c1 = rr.min(), rr.max(), cc.min(), cc.max()
SL = (slice(r0, r1), slice(c0, c1))
msk = msk_r[SL]; lab = lab_r[SL]


OUTZ = C.SDATA
gdf = gpd.read_parquet(OUTZ / "shapes/ssaft_cell_boundaries_2d/shapes.parquet")
lab_of = ad.read_zarr(OUTZ / "tables/ssaft_cells").obs["cellman_annotation"].astype(str).reindex(gdf.index)
bx0, bx1, by0, by1 = x0 * PX, x1 * PX, y0 * PX, y1 * PX
bnd = gdf.bounds
sel = ((bnd.maxx >= bx0) & (bnd.minx < bx1) & (bnd.maxy >= by0) & (bnd.miny < by1)).to_numpy()
gdf, lab_of = gdf[sel], lab_of[sel]
print("cells in the crop:", len(gdf), flush=True)
NAME_ID = {n: i + 1 for i, n in enumerate(names)}
H_, W_ = bx1 - bx0, by1 - by0
ras = np.zeros((H_, W_), np.int32)
ids = np.zeros((H_, W_), np.int32)
for j, (geom, l) in enumerate(zip(gdf.geometry, lab_of)):
    if geom is None or l not in NAME_ID:
        continue
    for gg in getattr(geom, "geoms", [geom]):
        xx, yy = np.asarray(gg.exterior.coords).T
        r_, c_ = dpoly(xx - bx0, yy - by0, (H_, W_))
        ras[r_, c_] = NAME_ID[l]; ids[r_, c_] = j + 1
ras_r = rot(ras)[SL]; ids_r = rot(ids)[SL]
edge = find_boundaries(ids_r, mode="inner")

PAL = np.array([[1.0, 1.0, 1.0]] + [rgbf(pal.get(n, "#cccccc")) for n in names])
GREY = "#3b3b3b"


def gene_img(g):
    d = ndi.gaussian_filter(gr[g].astype(float), 3)[x0:x1, y0:y1]
    d = rot(np.kron(d, np.ones((PX, PX))), order=1)[SL]
    return np.where(msk, d, 0)


fig = plt.figure(figsize=(9.2, 4.2))
gsp = fig.add_gridspec(2, 7, width_ratios=[1, 1, 1, 1, 1, 1, 1.35], wspace=.06, hspace=.08)
axs = np.array([[fig.add_subplot(gsp[r, c]) for c in range(6)] for r in range(2)])
alam = fig.add_subplot(gsp[:, 6])
for a in axs.ravel():
    a.axis("off")

im = PAL[np.clip(lab, 0, len(names))]; im[~msk] = im[~msk] * .35 + .65
axs[0, 0].imshow(im, interpolation="nearest"); axs[0, 0].set_title("SSAF-T cell-type map", fontsize=7)
h_, w_ = im.shape[:2]
axs[0, 0].add_patch(Rectangle((8, h_ - 30), 200, 9, color="k"))
axs[0, 0].text(108, h_ - 38, "200 µm", ha="center", fontsize=5.5)

ci = PAL[np.clip(ras_r, 0, len(names))]; ci[edge] = 1.0; ci[~msk] = ci[~msk] * .35 + .65
axs[1, 0].imshow(ci, interpolation="nearest"); axs[1, 0].set_title("SSAF-T cells", fontsize=7)

for k, g in enumerate(LAYG):
    d = gene_img(g); vmax = np.percentile(d[msk], 99)
    axs[0, k + 1].imshow(d, cmap="Greys", vmax=vmax, interpolation="bilinear")
    axs[0, k + 1].set_title(f"{g}\n({LAYG[g]})", fontsize=7)


    cells = np.ones(d.shape + (3,))
    any_ = np.zeros(d.shape, bool)
    for t in GENE_LAYER[g]:
        if t not in NAME_ID:
            continue
        m = (ras_r == NAME_ID[t]) & msk
        cells[m] = rgbf(pal.get(t, "#ff00ff")); any_ |= m
    cells[edge & any_] = 1.0
    axs[1, k + 1].imshow(cells, interpolation="nearest")
    axs[1, k + 1].set_title(" + ".join(LAYER_SHORT[t] for t in GENE_LAYER[g] if t in NAME_ID),
                            fontsize=6.5)


B50_C = "#7a7a7a"
NBs = 40
px_ = pd.read_parquet(CACHE / "cortex_ssaft_cells.parquet"); pc = px_[px_.win.isin(best)]
e = np.linspace(0, 1, NBs + 1)
lp = pc[pc.label.isin(LAY)]
tab = pd.crosstab(np.clip((lp.depth * NBs).astype(int), 0, NBs - 1), lp.label).reindex(range(NBs), fill_value=0)


MINB = 10
n_ = tab.sum(1)
frac = tab.div(n_.replace(0, 1), axis=0)
frac.loc[n_ < MINB] = 0.0
print(f"lamina: {int((n_ >= MINB).sum())}/{NBs} depth bins have >= {MINB} cells "
      f"(dropped {sorted(n_.index[(n_ > 0) & (n_ < MINB)])})")
left = np.zeros(NBs)
for l in LAY:
    if l in frac:
        alam.barh(e[:-1], frac[l].values, left=left, height=1 / NBs, align="edge", color=pal[l], lw=0)
        left += frac[l].values
alam.text(0.0, -0.155, f"bins with fewer than {MINB} cells left blank", fontsize=4.6,
          color="#888", va="top", ha="left", transform=alam.transAxes)
b50 = pd.read_parquet(CACHE / "bin50_anatomy.parquet"); b50 = b50[b50.win.isin(best)]
for l in ["Cortex_L1", "Cortex-L2/3", "Cortex_L4", "Cortex_L5", "Cortex_L6"]:
    v = b50.depth[b50.annotation == l]
    if len(v):
        alam.plot([1.12, 1.12], [v.quantile(.25), v.quantile(.75)], color=B50_C, lw=3, solid_capstyle="butt")
        alam.text(1.2, v.median(), l.replace("Cortex_", "").replace("Cortex-", ""), va="center",
                  fontsize=6, color="#333")
alam.set_ylim(1, 0); alam.set_xlim(0, 1.75); alam.set_xticks([0, 1])
alam.yaxis.tick_right(); alam.tick_params(labelsize=6)
alam.set_title("lamina\n(depth, pia 0 to WM 1)", fontsize=7)
alam.text(1.32, 1.045, "grey: published\nbin50 layers (IQR)", fontsize=5, ha="center", va="top", color=B50_C)
for s_ in ("top", "right"):
    alam.spines[s_].set_visible(False)

fig.text(.085, .70, "raw", fontsize=7, rotation=90, va="center", color="#444")
fig.text(.085, .30, "SSAF-T", fontsize=7, rotation=90, va="center", color="#444")
fig.legend(handles=[Patch(color=pal[l], label=LAYER_SHORT[l]) for l in LAY],
           loc="lower center", ncol=5, fontsize=6, frameon=False, bbox_to_anchor=(.42, .02))
fig.savefig(OUT / "panel_cortex_column_grid_cells.png", dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "panel_cortex_column_grid_cells.pdf", dpi=600, bbox_inches="tight", facecolor="white")
print("->", OUT / "panel_cortex_column_grid_cells.png", flush=True)
