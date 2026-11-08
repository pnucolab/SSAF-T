#!/usr/bin/env python
import json

import anndata as ad
import numpy as np
import pandas as pd
import zarr

from common import OUT, WORK, MOOR_TOP, MOOR_BOTTOM, HOUSEKEEPING_PREFIX, ssaft_cells, signal_genes
kg=set(json.load(open(WORK/"kg_markers.json"))["kg_markers"]); hk=lambda g: g.startswith(HOUSEKEEPING_PREFIX)

sig=set(signal_genes())
s=ssaft_cells(); s.raw=None; sg=pd.Index(s.var_names.astype(str))
ho=lambda L:[g for g in L if g not in kg and not hk(g) and g in sg and g not in sig]
TOP,BOT=ho(MOOR_TOP),ho(MOOR_BOTTOM)
X=np.asarray(s[:,sg.get_indexer(TOP+BOT)].X.todense())
sc=pd.read_parquet(WORK/"ssaft_cells_axis.parquet"); lab=sc["label"].astype(str).values; A=sc["A"].values; unit=sc["unit"].values
def delta(t,b):
    x=np.where(t+b>=3,t/np.maximum(t+b,1),np.nan)
    m=np.isin(lab,["villus-bottom enterocyte","villus-tip enterocyte"])&np.isfinite(x)&np.isfinite(A)&(unit>0)
    d=pd.DataFrame(dict(x=x[m],A=A[m],isb=(lab[m]=="villus-tip enterocyte")))
    d["dec"]=pd.qcut(d.A,10,labels=False,duplicates="drop")
    diffs,w=[],[]
    for k in d.dec.unique():
        mm=(d.dec==k).values
        if d.isb.values[mm].sum()<5 or (~d.isb.values[mm]).sum()<5: continue
        diffs.append(d.x.values[mm][d.isb.values[mm]].mean()-d.x.values[mm][~d.isb.values[mm]].mean()); w.append(mm.sum())
    return float(np.average(diffs,weights=w)), int(m.sum())
nt=len(TOP); T=X[:,:nt].sum(1); B=X[:,nt:].sum(1)
full,_=delta(T,B); print(f"full strict ({nt} top / {len(BOT)} bottom): {full:+.3f}")
rows=[]
for i,g in enumerate(TOP+BOT):
    if i<nt: d_,_=delta(T-X[:,i],B)
    else: d_,_=delta(T,B-X[:,i])
    rows.append((g,"top" if i<nt else "bottom", d_, float(X[:,i].sum())))
D=pd.DataFrame(rows,columns=["gene","side","delta_without","umi"]).sort_values("delta_without")
print("\nleave-one-out (Δ when that gene is dropped):"); print(D.to_string(index=False))
