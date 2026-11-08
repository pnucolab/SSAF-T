#!/usr/bin/env python
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from common import *

b = load_bin1()
print("DNB rows", len(b["x"]), "x", b["x"].min(), b["x"].max(), "y", b["y"].min(), b["y"].max())
sets = {"total": None, "WM (Mbp,Plp1,Mobp)": ["Mbp", "Plp1", "Mobp"],
        "Cux2": ["Cux2"], "Rorb": ["Rorb"], "Fezf2": ["Fezf2"], "Foxp2": ["Foxp2"],
        "Ctgf": ["Ctgf"], "Prox1+C1ql2": ["Prox1", "C1ql2"], "Fibcd1+Wfs1": ["Fibcd1", "Wfs1"],
        "Bok": ["Bok"], "Tcf7l2": ["Tcf7l2"]}
ras = {k: raster(b, v) for k, v in sets.items()}
np.savez_compressed(CACHE / "overview_rasters.npz", **{k.split(" ")[0]: v for k, v in ras.items()})
cm, names = load_map()
lm = map_at(cm)
pal = palette()
rgb = np.ones(lm.shape + (3,))
for k, n in enumerate(names, 1):
    c = pal.get(n, "#cccccc"); rgb[lm == k] = [int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)]
rgb[lm == 0] = 0.85
fig, ax = plt.subplots(3, 5, figsize=(25, 22))
for a, (k, v) in zip(ax.flat, ras.items()):
    a.imshow(np.log1p(gaussian_filter(v, 2)).T, cmap="magma", origin="upper")
    a.set_title(k)
ax.flat[len(ras)].imshow(rgb.transpose(1, 0, 2)); ax.flat[len(ras)].set_title("SSAF-T map")
for a in ax.flat:
    a.set_xticks(np.arange(0, NX, 100)); a.set_yticks(np.arange(0, NY, 100))
    a.tick_params(labelsize=6); a.grid(alpha=.2)
plt.tight_layout(); plt.savefig(CACHE / "overview.png", dpi=60)
