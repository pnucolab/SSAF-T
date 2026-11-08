#!/usr/bin/env python
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.measure import find_contours
from scipy.spatial import cKDTree
from common import *

z = np.load(CACHE / "overview_rasters.npz")
tot, wmr = z["total"], z["WM"]


ssdna = np.load(CACHE / "ssdna_5um.npy")
s_ = ndi.gaussian_filter(ssdna, 3)
tissue = s_ > threshold_otsu(s_) / 2
tissue = ndi.binary_closing(tissue, iterations=6)
tissue = ndi.binary_fill_holes(tissue)
lab, n = ndi.label(tissue); sz = ndi.sum(tissue, lab, range(1, n + 1))
tissue = lab == (np.argmax(sz) + 1)
print("tissue (ssDNA) area mm2", tissue.sum() * PX * PX / 1e6)

lt = np.log1p(ndi.gaussian_filter(tot, 2))
rna = lt > threshold_otsu(lt[lt > 0]); rna = ndi.binary_opening(rna, iterations=2)
lab, n = ndi.label(rna); sz = ndi.sum(rna, lab, range(1, n + 1)); rna = lab == (np.argmax(sz) + 1)
halo = rna & ~tissue
dout = ndi.distance_transform_edt(~tissue) * PX
print("RNA-mask area mm2", rna.sum() * PX * PX / 1e6, " halo width um (median/p90 of the RNA mask's outer edge distance)",
      np.median(dout[rna & ~ndi.binary_erosion(rna)]), np.percentile(dout[rna & ~ndi.binary_erosion(rna)], 90))


frac = ndi.gaussian_filter(wmr, 5) / (ndi.gaussian_filter(tot, 5) + 1e-9)
lw = frac[tissue]
thr = threshold_otsu(lw)
wm = (frac > thr) & tissue
wm = ndi.binary_opening(wm, iterations=2)
lab, n = ndi.label(wm); sz = ndi.sum(wm, lab, range(1, n + 1))
wm = np.isin(lab, np.where(sz * PX * PX > 0.02e6)[0] + 1)
print("WM threshold (WM-gene UMI fraction)", thr, "WM area mm2", wm.sum() * PX * PX / 1e6)


tp = np.pad(tissue, 1)
d_pia, ip = ndi.distance_transform_edt(tp, return_indices=True)
d_pia = d_pia[1:-1, 1:-1]; ip = ip[:, 1:-1, 1:-1] - 1
frame = np.zeros_like(tissue); frame[[0, -1], :] = tissue[[0, -1], :]; frame[:, [0, -1]] |= tissue[:, [0, -1]]
d_wm, iw = ndi.distance_transform_edt(~wm, return_indices=True)
d_pia *= PX; d_wm *= PX
gx, gy = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
vp = np.stack([ip[0] - gx, ip[1] - gy]).astype(float)
vw = np.stack([iw[0] - gx, iw[1] - gy]).astype(float)
cos = (vp * vw).sum(0) / (np.linalg.norm(vp, axis=0) * np.linalg.norm(vw, axis=0) + 1e-9)
thick = d_pia + d_wm
depth = d_pia / np.maximum(thick, 1e-9)
cortex = tissue & ~wm & (cos < -0.7) & (thick > 500) & (thick < 1800)
lab, n = ndi.label(cortex); sz = ndi.sum(cortex, lab, range(1, n + 1))
cortex = np.isin(lab, np.where(sz * PX * PX > 0.3e6)[0] + 1)
print("cortex domain mm2", cortex.sum() * PX * PX / 1e6)


cont = max(find_contours(np.pad(tissue, 1).astype(float), 0.5), key=len)[:-1] - 1

cont = np.roll(cont, -int(np.argmin(cont[:, 0])), axis=0)
seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(cont, axis=0), axis=1))] * PX
tree = cKDTree(cont)
pia_pts = np.stack([ip[0][cortex], ip[1][cortex]], 1)
_, k = tree.query(pia_pts)
col = np.full(tissue.shape, np.nan); col[cortex] = seg[k]

onframe = np.zeros(tissue.shape, bool)
pxp = np.clip(ip[0], 0, NX - 1), np.clip(ip[1], 0, NY - 1)
onframe[cortex] = ((ip[0][cortex] <= 0) | (ip[0][cortex] >= NX - 1) | (ip[1][cortex] <= 0) | (ip[1][cortex] >= NY - 1))
print("cortex px whose nearest surface is the chip edge", int(onframe.sum()))

tan = np.gradient(cont, axis=0); ang = np.unwrap(np.arctan2(tan[:, 1], tan[:, 0]))

np.savez_compressed(CACHE / "geometry.npz", tissue=tissue, wm=wm, d_pia=d_pia, d_wm=d_wm,
                    depth=depth, cortex=cortex, col=col, thick=thick, cont=cont, seg=seg,
                    ang=ang, wm_thr=thr, onframe=onframe, rna_tissue=rna, halo=halo)

fig, ax = plt.subplots(1, 4, figsize=(24, 8))
ax[0].imshow(np.log1p(ndi.gaussian_filter(tot, 2)).T, cmap="gray")
ax[0].contour(tissue.T, [0.5], colors="c", linewidths=.6); ax[0].contour(wm.T, [0.5], colors="m", linewidths=.6)
ax[0].set_title("tissue (cyan) / WM (magenta)")
d = np.where(cortex, depth, np.nan); ax[1].imshow(d.T, cmap="viridis"); ax[1].set_title("normalised depth")
ax[2].imshow(np.where(cortex, thick, np.nan).T, cmap="plasma"); ax[2].set_title("thickness um")
ax[3].imshow(col.T, cmap="twilight"); ax[3].plot(cont[::200, 0], cont[::200, 1], "k.", ms=3)
for i in range(0, len(cont), 400): ax[3].text(cont[i, 0], cont[i, 1], f"{seg[i]:.0f}", fontsize=7)
ax[3].set_title("column (arc um)")
plt.tight_layout(); plt.savefig(CACHE / "geometry.png", dpi=60)
