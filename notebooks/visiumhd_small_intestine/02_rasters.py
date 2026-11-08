import json
import numpy as np, scipy.sparse as sp
from scipy import ndimage as ndi
from common import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

m = np.load(WORK / "bins_meta.npz"); Xs = sp.load_npz(WORK / "bins_sel.npz").tocsc()
ri, ci, tot = m["ri"], m["ci"], m["tot"]; H, W = m["shape"]; genes = list(m["genes"])
kg = set(json.load(open(WORK / "kg_markers.json"))["kg_markers"])
def img(v):
    a = np.zeros((H, W), np.float32); a[ri, ci] = v; return a
def gsum(gs):
    idx = [genes.index(g) for g in gs if g in genes]
    return np.asarray(Xs[:, idx].sum(1)).ravel()
hk = lambda g: g.startswith(HOUSEKEEPING_PREFIX)
groups = dict(
    total=tot, muscle=gsum(MUSCLE), paneth=gsum(PANETH_RAW), paneth_ho=gsum(PANETH_HELDOUT),
    stem=gsum(STEM), prolif=gsum(PROLIF), epi=gsum(EPI),
    moor_bottom_ho=gsum([g for g in MOOR_BOTTOM if g not in kg and not hk(g)]),
    moor_top_ho=gsum([g for g in MOOR_TOP if g not in kg and not hk(g)]),
    moor_bottom_all=gsum(MOOR_BOTTOM), moor_top_all=gsum(MOOR_TOP),
    immune=gsum(['Ptprc', 'Cd79a', 'Jchain']))
tissue = img(np.ones_like(tot))
np.savez_compressed(WORK / "rasters.npz", tissue=tissue.astype(bool),
                    **{k: img(v) for k, v in groups.items()})
print({k: float(v.sum()) for k, v in groups.items()})
print("held-out moor bottom:", [g for g in MOOR_BOTTOM if g not in kg and not hk(g) and g in genes])
print("held-out moor top:", [g for g in MOOR_TOP if g not in kg and not hk(g) and g in genes])


f = 4
def down(a): return a[:H // f * f, :W // f * f].reshape(H // f, f, W // f, f).sum((1, 3))
fig, axs = plt.subplots(2, 3, figsize=(18, 12))
for ax, k in zip(axs.ravel(), ["total", "muscle", "paneth", "prolif", "moor_bottom_ho", "moor_top_ho"]):
    a = ndi.gaussian_filter(down(img(groups[k])), 1)
    ax.imshow(a, cmap="magma", vmax=np.percentile(a[a > 0], 99)); ax.set_title(k); ax.axis("off")
plt.tight_layout(); plt.savefig(WORK / "overview_rasters.png", dpi=80)
