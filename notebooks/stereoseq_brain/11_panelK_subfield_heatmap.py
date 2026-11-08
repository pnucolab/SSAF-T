#!/usr/bin/env python
import json
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
from common import CACHE, OUT, PX, STORE, load_bin1, load_map, ssaft_cells
import config as C

GENES = {"DG": ["Prox1", "C1ql2", "Dsp", "Trpc6"],
         "CA3": ["Bok", "Grik4", "Neto1", "Trps1", "Rerg"],
         "CA1": ["Spink8", "Fibcd1", "Wfs1", "Pou3f1", "Mpped1", "Itpka"]}
PAR = "hippocampal pyramidal neuron"
GROUP = {"DG": ["dentate gyrus granule cell"], "CA3": ["CA3 pyramidal neuron"],
         "CA1": ["CA1 pyramidal neuron"],
         "CA pooled": ["CA1 pyramidal neuron", "CA3 pyramidal neuron", PAR]}
COLS = ["DG", "CA3", "CA1", "CA pooled", "rest"]
H = np.load(CACHE / "hippo_masks.npz")
x0, x1, y0, y1 = [int(H[k]) for k in ("x0", "x1", "y0", "y1")]
X0, X1, Y0, Y1 = x0 * PX, x1 * PX, y0 * PX, y1 * PX

a = ssaft_cells()
xy = np.asarray(a.obsm["spatial"][:], float)
lab = a.obs["cellman_annotation"].astype(str).to_numpy()
inw = (xy[:, 0] >= X0) & (xy[:, 0] < X1) & (xy[:, 1] >= Y0) & (xy[:, 1] < Y1)
X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
genes = np.asarray(a.var_names, dtype=str); gidx = {g: i for i, g in enumerate(genes)}

named = {k: inw & np.isin(lab, v) for k, v in GROUP.items()}
named["rest"] = inw & ~(named["DG"] | named["CA1"] | named["CA3"])
print("cells in the window: " + ", ".join(f"{k} {int(named[k].sum()):,}" for k in COLS))

per, tot = {}, {}
for k in COLS:
    v = np.asarray(X[named[k]].sum(0)).ravel()
    per[k], tot[k] = v, float(v.sum())
rows, rlab, rsub = [], [], []
for sub, gs in GENES.items():
    for g in gs:
        if g not in gidx:
            print("missing", g); continue
        j = gidx[g]
        rows.append([per[k][j] / max(tot[k], 1) for k in COLS]); rlab.append(g); rsub.append(sub)
M = np.array(rows)
Zc = (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-12)


cm, names = load_map(); cm = cm[X0:X1, Y0:Y1]
names = ["none"] + [str(n) for n in names]; code = {n: i for i, n in enumerate(names)}
b = load_bin1()
ix = np.floor(b["x"]).astype(np.int32) - X0; iy = np.floor(b["y"]).astype(np.int32) - Y0
ins = (ix >= 0) & (ix < cm.shape[0]) & (iy >= 0) & (iy < cm.shape[1])
pl = cm[ix[ins], iy[ins]]; gi, um = b["gene"][ins], b["umi"][ins].astype(np.float64)
pnamed = {k: np.isin(pl, [code[n] for n in v if n in code]) for k, v in GROUP.items()}
pnamed["rest"] = (pl != 0) & ~(pnamed["DG"] | pnamed["CA1"] | pnamed["CA3"])
ng = len(b["genes"])
pper = {k: np.bincount(gi[pnamed[k]], weights=um[pnamed[k]], minlength=ng) for k in COLS}
ptot = {k: v.sum() for k, v in pper.items()}
Mp = np.array([[pper[k][b["gidx"][g]] / max(ptot[k], 1) for k in COLS] for g in rlab])
Zp = (Mp - Mp.mean(1, keepdims=True)) / (Mp.std(1, keepdims=True) + 1e-12)

hdr = "  ".join(f"{c:>9s}" for c in COLS)
print(f"\n{'gene':10s} {'sub':4s} | {'CELLS':^55s} | {'PIXELS':^55s}")
print(f"{'':10s} {'':4s} | {hdr} | {hdr}")
for i, (g, sv) in enumerate(zip(rlab, rsub)):
    print(f"{g:10s} {sv:4s} | " + "  ".join(f"{v:9.2f}" for v in Zc[i]) +
          " | " + "  ".join(f"{v:9.2f}" for v in Zp[i]))
argc = [COLS[i] for i in Zc.argmax(1)]; argp = [COLS[i] for i in Zp.argmax(1)]
print("\nhighest column per gene")
print(f"  cells : {dict(zip(rlab, argc))}")
print(f"  pixels: {dict(zip(rlab, argp))}")
print(f"  same in {sum(a_ == b_ for a_, b_ in zip(argc, argp))}/{len(rlab)} genes")
print(f"  z-score correlation between the two units: {np.corrcoef(Zc.ravel(), Zp.ravel())[0,1]:.3f}")
json.dump(dict(cols=COLS, genes=rlab, sub=rsub, z_cells=Zc.tolist(), z_pixels=Zp.tolist(),
               n_cells={k: int(named[k].sum()) for k in COLS}),
          open(OUT / "subfield_heatmap_cells.json", "w"), indent=1)
print("-> subfield_heatmap_cells.json")


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                     "pdf.fonttype": 42, "svg.fonttype": "none"})
COL = {"DG": "#e41a1c", "CA3": "#377eb8", "CA1": "#4daf4a"}
rcol = [COL[s_] for s_ in rsub]
fig, ax = plt.subplots(figsize=(2.9, 2.9))
im = ax.imshow(Zc, cmap="RdBu_r", vmin=-1.6, vmax=1.6, aspect="auto")
ax.set_xticks(range(len(COLS)))
ax.set_xticklabels(COLS, fontsize=6, rotation=40, ha="right")
ax.set_yticks(range(len(rlab))); ax.set_yticklabels(rlab, fontsize=5.6, style="italic")
for t, c in zip(ax.get_yticklabels(), rcol):
    t.set_color(c)
for i in np.cumsum([len(v) for v in GENES.values()])[:-1] - .5:
    ax.axhline(i, color="w", lw=1.2)
ax.axvline(2.5, color="k", lw=.7)
ax.set_title("Subfield markers", fontsize=8, pad=16)
ax.text(.30, 1.015, "SSAF-T subfields", transform=ax.transAxes, fontsize=5.2, ha="center",
        va="bottom", color="#2a5d9f")
ax.text(.80, 1.015, "reference's resolution", transform=ax.transAxes, fontsize=5.2, ha="center",
        va="bottom", color="#8a5a1a")
cb = fig.colorbar(im, ax=ax, fraction=.045, pad=.03)
cb.set_label("gene expression (z-score)", fontsize=5.4); cb.ax.tick_params(labelsize=5)
for ext in ("png", "pdf", "svg"):
    fig.savefig(OUT / f"panelK3_subfield_heatmap_cells.{ext}", dpi=600, bbox_inches="tight",
                facecolor="white")
print("->", OUT / "panelK3_subfield_heatmap_cells.pdf")
