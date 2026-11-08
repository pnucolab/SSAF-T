import json
import numpy as np
from scipy import ndimage as ndi
from skimage.graph import MCP_Geometric
from skimage import morphology
from common import *
MUSCLE_FRAC = 0.03
TISSUE_UMI = 10.0
FOLLICLE_FRAC = 0.0006
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

z = np.load(WORK / "rasters.npz")
tot, mus, pan = z["total"], z["muscle"], z["paneth_ho"]
imm = z["immune"]
H, W = tot.shape
g = lambda a, s: ndi.gaussian_filter(a.astype(np.float32), s)
tot_s = g(tot, 2.0)
tissue = tot_s > 6.0
tissue = morphology.remove_small_holes(tissue, 50)
tissue = morphology.remove_small_objects(tissue, 5000)


mus_f = g(mus, 3.0) / np.maximum(g(tot, 3.0), 1.0)
thr_m = MUSCLE_FRAC
mb = (mus_f > thr_m) & tissue
mb = morphology.binary_closing(mb, morphology.disk(2))
mb = morphology.remove_small_objects(mb, 1500)
print("muscle fraction threshold", thr_m, "px", mb.sum(), "pct of tissue", mb.sum() / tissue.sum())


tissue = (tot_s > TISSUE_UMI) | mb
tissue = morphology.remove_small_holes(tissue, 25)
tissue = morphology.remove_small_objects(tissue, 5000)
print("tissue px", tissue.sum(), "of in-tissue bins", z["tissue"].sum())


import scipy.sparse as sp
mm = np.load(WORK / "bins_meta.npz"); Xs = sp.load_npz(WORK / "bins_sel.npz").tocsc()
gl = list(mm["genes"]); b = np.zeros((H, W), np.float32)
b[mm["ri"], mm["ci"]] = Xs[:, gl.index("Ptprc")].toarray().ravel()
imm_s = g(b, 12.0) / np.maximum(g(tot, 12.0), 1e-3)
print("B fraction pct", np.percentile(imm_s[tissue], [50, 90, 99, 99.9]))
follicle = morphology.remove_small_objects((imm_s > FOLLICLE_FRAC) & tissue, 10000)
print("follicle px", follicle.sum())

def geodesic(seed):
    cost = np.where(tissue, 1.0, np.inf).astype(np.float64)
    m = MCP_Geometric(cost, fully_connected=True)
    d, _ = m.find_costs(np.argwhere(seed & tissue))
    d = d * PX; d[~tissue] = np.nan
    return d.astype(np.float32)

DA = geodesic(mb)
pan_s = g(pan, 2.0)
thr_p = np.percentile(pan_s[tissue], 97)
pb = morphology.remove_small_objects((pan_s > thr_p) & tissue, 30)
print("paneth threshold", thr_p, "px", pb.sum())
DB = geodesic(pb)

DAp = np.where(pb, DA, np.nan)


DA0 = DA
T = tissue & ~mb
def geodesic_in(seed, trav):
    cost = np.where(trav, 1.0, np.inf).astype(np.float64)
    d, _ = MCP_Geometric(cost, fully_connected=True).find_costs(np.argwhere(seed & trav))
    d = d * PX; d[~trav] = np.nan; return d.astype(np.float32)
face = ndi.binary_dilation(mb, iterations=1) & T
DPb = geodesic_in(pb, T)
seeds = face & (DPb < 60.0)
print("muscle face px", face.sum(), "mucosal-face seeds", seeds.sum())
DA = geodesic_in(seeds, T)
DA[mb] = 0.0
DA[np.isinf(DA)] = np.nan
np.savez_compressed(WORK / "axes.npz", DA=DA, DA0=DA0, DB=DB, tissue=tissue, muscle=mb, paneth=pb,
                    follicle=follicle)
print("DA percentiles", np.nanpercentile(DA, [5, 25, 50, 75, 95, 99]))
print("DA at Paneth band", np.nanpercentile(np.where(pb, DA, np.nan), [10, 50, 90]))

f = 4
def down(a, fn=np.nanmean):
    return fn(a[:H // f * f, :W // f * f].reshape(H // f, f, W // f, f), axis=(1, 3))
fig, axs = plt.subplots(1, 3, figsize=(24, 8))
axs[0].imshow(down(tissue.astype(float)) + 2 * down(mb.astype(float)) + 3 * down(pb.astype(float)) - 1.5 * down(follicle.astype(float)), cmap="viridis")
axs[1].imshow(down(DA), cmap="turbo", vmax=np.nanpercentile(DA, 99)); axs[1].set_title("A: height above muscle")
axs[2].imshow(down(DB), cmap="turbo", vmax=np.nanpercentile(DB, 99)); axs[2].set_title("B: from Paneth band")
for a in axs: a.axis("off")
plt.tight_layout(); plt.savefig(WORK / "axes_overview.png", dpi=70)
