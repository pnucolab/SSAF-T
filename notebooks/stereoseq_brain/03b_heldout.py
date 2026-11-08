#!/usr/bin/env python
import json
import numpy as np, pandas as pd, h5py
from scipy.stats import binomtest
from common import *

rng = np.random.default_rng(3)
ho = pd.read_csv(OUT / "cortex_heldout_genes.csv")
ho = ho[(ho.q < 1e-3) & (ho.peak_enr >= 1.5)]
BANDS = {"L2/3": (0.17, 0.37), "L4": (0.37, 0.57), "L5": (0.57, 0.78), "L6": (0.78, 0.95)}
gsets = {k: ho[(ho.peak_depth >= lo) & (ho.peak_depth < hi)].head(12).gene.tolist() for k, (lo, hi) in BANDS.items()}
print(json.dumps(gsets, indent=0))

b = load_bin1()
ras = {k: raster(b, v) for k, v in gsets.items()}
px = pd.read_parquet(CACHE / "cortex_pixels.parquet")
for k in gsets: px[k] = ras[k][px.ix.values, px.iy.values]
px["dbin"] = np.clip((px.depth * 40).astype(int), 0, 39)

PAIRS = [("L2/3 IT glutamatergic neuron", "L4 IT glutamatergic neuron", "L2/3", "L4"),
         ("L4 IT glutamatergic neuron", "L5 ET glutamatergic neuron", "L4", "L5"),
         ("L5 ET glutamatergic neuron", "L6 CT glutamatergic neuron", "L5", "L6")]

def stat(df, A, B, gA, gB, lab="label"):
    num, den = 0.0, 0.0
    for _, g in df.groupby("dbin"):
        a = g[g[lab] == A]; c = g[g[lab] == B]
        if len(a) < 20 or len(c) < 20: continue
        sa = np.log((a[gB].sum() + 0.5) / (a[gA].sum() + 0.5))
        sb = np.log((c[gB].sum() + 0.5) / (c[gA].sum() + 0.5))
        w = min(len(a), len(c)); num += w * (sb - sa); den += w
    return num / den if den else np.nan

def run(df, pairs, tag, lab="label", nperm=500, colkey="win"):
    out = []
    for A, B, ka, kb in pairs:
        d = df[df[lab].isin([A, B])].copy()
        obs = stat(d, A, B, ka, kb, lab)
        null = []
        for _ in range(nperm):
            q = d[lab].values.copy()
            for bi in np.unique(d.dbin):
                m = (d.dbin == bi).values; q[m] = rng.permutation(q[m])
            null.append(stat(d.assign(**{lab: q}), A, B, ka, kb, lab))
        null = np.array(null)
        percol = [stat(g, A, B, ka, kb, lab) for _, g in d.groupby(colkey)]
        percol = np.array([v for v in percol if np.isfinite(v)])
        st = binomtest(int((percol > 0).sum()), len(percol), 0.5) if len(percol) else None
        r = dict(A=A, B=B, genes_A=gsets[ka], genes_B=gsets[kb], stat=float(obs),
                 perm_p=float(((np.abs(null) >= abs(obs)).sum() + 1) / (nperm + 1)),
                 null_sd=float(np.nanstd(null)), n_columns=int(len(percol)),
                 columns_positive=int((percol > 0).sum()), sign_p=float(st.pvalue) if st else None)
        out.append(r)
        print(f"[{tag}] {A.split(' ')[0]} -> {B.split(' ')[0]}: stat {obs:+.3f} (null sd {np.nanstd(null):.3f}, p {r['perm_p']:.3f}); "
              f"columns positive {r['columns_positive']}/{r['n_columns']} sign p {r['sign_p']}")
    return out

res = dict(gene_sets=gsets)
res["ssaft"] = run(px, PAIRS, "SSAF-T 5um pixels")


cb = pd.read_parquet(CACHE / "cortex_cellbin.parquet")
with h5py.File(C.MOSTA / "Mouse_brain_cell_bin.h5ad", "r") as h:
    var = h["var/_index"][:].astype(str); names = h["obs/_index"][:].astype(str)
    ip = h["layers/counts/indptr"][:]; ind = h["layers/counts/indices"][:]; dat = h["layers/counts/data"][:]
import scipy.sparse as sp
X = sp.csr_matrix((dat, ind, ip), shape=(len(names), len(var)))
row = pd.Series(np.arange(len(names)), index=names)
Xc = X[row[cb.cell].values]
vi = pd.Series(np.arange(len(var)), index=var)
for k, gl in gsets.items():
    cols = [vi[g] for g in gl if g in vi.index]
    cb[k] = np.asarray(Xc[:, cols].sum(1)).ravel()
cb["dbin"] = np.clip((cb.depth * 40).astype(int), 0, 39)
cb = cb.drop(columns="label").rename(columns={"annotation": "label"})
PP = [("EX L2/3", "EX L4", "L2/3", "L4"), ("EX L4", "EX L5/6", "L4", "L5"), ("EX L5/6", "EX L6", "L5", "L6")]
def stat_cells(df, A, B, gA, gB, lab="label"):
    num, den = 0.0, 0.0
    for _, g in df.groupby("dbin"):
        a = g[g[lab] == A]; c = g[g[lab] == B]
        if len(a) < 5 or len(c) < 5: continue
        sa = np.log((a[gB].sum() + 0.5) / (a[gA].sum() + 0.5)); sb = np.log((c[gB].sum() + 0.5) / (c[gA].sum() + 0.5))
        w = min(len(a), len(c)); num += w * (sb - sa); den += w
    return num / den if den else np.nan
stat_px = stat; stat = stat_cells
res["cellbin"] = run(cb, PP, "CellBin cells")
stat = stat_px

json.dump(res, open(CACHE / "stats_heldout.json", "w"), indent=1, default=float)
print("done")
