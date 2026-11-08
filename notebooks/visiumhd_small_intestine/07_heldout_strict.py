#!/usr/bin/env python
import json

import anndata as ad
import numpy as np
import pandas as pd
import zarr

from common import OUT, WORK, MOOR_TOP, MOOR_BOTTOM, HOUSEKEEPING_PREFIX, ssaft_cells, signal_genes

N_PERM = 1000
rng = np.random.default_rng(0)

kg = set(json.load(open(WORK / "kg_markers.json"))["kg_markers"])
hk = lambda g: g.startswith(HOUSEKEEPING_PREFIX)
signal = set(signal_genes())

s = ssaft_cells(); s.raw = None
sg = pd.Index(s.var_names.astype(str))
ho = lambda lst: [g for g in lst if g not in kg and not hk(g) and g in sg]
SETS = {"as published (s05)": (ho(MOOR_TOP), ho(MOOR_BOTTOM)),
        "strict (also outside the 5,202 signal genes)":
            ([g for g in ho(MOOR_TOP) if g not in signal], [g for g in ho(MOOR_BOTTOM) if g not in signal])}

sc = pd.read_parquet(WORK / "ssaft_cells_axis.parquet")
sc = sc.set_index(sc.columns[0]) if sc.index.name is None and sc.columns[0] == "cell_id" else sc
lab = sc["label"].astype(str).values if "label" in sc else None
A, unit = sc["A"].values, sc["unit"].values

def cs(lst):
    return np.asarray(s[:, sg.get_indexer(lst)].X.sum(1)).ravel()

def cond_test(x, a, b):
    m = np.isin(lab, [a, b]) & np.isfinite(x) & np.isfinite(A) & (unit > 0)
    d = pd.DataFrame(dict(x=x[m], A=A[m], isb=(lab[m] == b)))
    d["dec"] = pd.qcut(d.A, 10, labels=False, duplicates="drop")
    y, st_ = d.x.values, d.dec.values

    def stat(fl):
        diffs, w = [], []
        for k in np.unique(st_):
            mm = st_ == k
            if fl[mm].sum() < 5 or (~fl[mm]).sum() < 5:
                continue
            diffs.append(y[mm][fl[mm]].mean() - y[mm][~fl[mm]].mean()); w.append(mm.sum())
        return np.average(diffs, weights=w)

    obs = stat(d.isb.values)
    nul = []
    for _ in range(N_PERM):
        f = d.isb.values.copy()
        for k in np.unique(st_):
            mm = st_ == k; f[mm] = rng.permutation(f[mm])
        nul.append(stat(f))
    nul = np.array(nul)
    return dict(n=int(m.sum()), diff=float(obs), null_p99=float(np.percentile(nul, 99)),
                p=float(((nul >= obs).sum() + 1) / (N_PERM + 1)))

out = {}
for name, (top, bot) in SETS.items():
    t, b = cs(top), cs(bot)
    x = np.where(t + b >= 3, t / np.maximum(t + b, 1), np.nan)
    r = cond_test(x, "villus-bottom enterocyte", "villus-tip enterocyte")
    r.update(n_top=len(top), n_bottom=len(bot), genes_top=top, genes_bottom=bot)
    out[name] = r
    print(f"{name}: top {len(top)} / bottom {len(bot)} genes, n={r['n']:,}, "
          f"Δ = {r['diff']:+.3f}, null p99 {r['null_p99']:.3f}, p = {r['p']:.4f}", flush=True)
(OUT / "heldout_strict.json").write_text(json.dumps(out, indent=2))
print("->", OUT / "heldout_strict.json", flush=True)
