import json
import numpy as np, pandas as pd, anndata as ad
from common import *

st = json.load(open(OUT / "stats.json"))
sc = pd.read_parquet(WORK / "ssaft_cells_axis_scored.parquet")
ec = pd.read_parquet(WORK / "enact_cells_axis_C.parquet")
CRYPT_TOP = 60.0
P = pd.read_csv(OUT / "raw_gradients_axisA.csv", index_col=0)
pr = P["proliferation, non-KG (Top2a, Cenpf, Birc5, Cdk1, Ccnb2, Stmn1)"]
from scipy import ndimage as ndi
prs = pd.Series(ndi.gaussian_filter1d(pr.values, 1.5), index=pr.index)
pk = prs.idxmax(); half = prs[prs.index > pk][prs[prs.index > pk] < prs.max() / 2].index.min()
print(f"raw proliferation peak {pk:.0f} um, half-max above it at {half:.0f} um")
res = dict(crypt_top_um_from_raw_proliferation_halfmax=float(half))
def conf(df, labs):
    d = df[(df.unit > 0) & np.isfinite(df.A)]
    return {L: dict(n=int((d.label == L).sum()), frac_in_crypt_zone=float((d.A[d.label == L] <= half).mean()))
            for L in labs}
res["ssaft"] = conf(sc, ["Paneth cell", "intestinal stem cell", "transit amplifying cell", "immature enterocyte",
                         "villus-bottom enterocyte", "villus-tip enterocyte", "goblet cell"])
res["enact"] = conf(ec, ["Paneth cells", "Crypt cells", "Enterocytes", "Goblet cells"])

e = enact_cells(); e.raw = None
eg = pd.Index(e.var_names.astype(str))
v = np.asarray(e.X[:, eg.get_indexer([g for g in PANETH_HELDOUT if g in eg])].sum(1)).ravel()
t = np.asarray(e.X.sum(1)).ravel()
f = v / np.maximum(t, 1e-9) * 1e3
lab = e.obs.cell_type.astype(str).values
res["enact_paneth_nonKG_per_1000umi"] = {L: float(np.mean(f[lab == L])) for L in ["Paneth cells", "Crypt cells", "Enterocytes", "Goblet cells"]}
res["enact_frac_paneth_cells_with_zero_paneth_nonKG_umi"] = float((v[lab == "Paneth cells"] == 0).mean())
s = ssaft_cells(); s.raw = None
sg = pd.Index(s.var_names.astype(str))
v2 = np.asarray(s.X[:, sg.get_indexer([g for g in PANETH_HELDOUT if g in sg])].sum(1)).ravel()
l2 = s.obs.cellman_annotation.astype(str).values
res["ssaft_frac_paneth_cells_with_zero_paneth_nonKG_umi"] = float((v2[l2 == "Paneth cell"] == 0).mean())
print(json.dumps(res, indent=1))
st["crypt_confinement"] = res
json.dump(st, open(OUT / "stats.json", "w"), indent=1, default=float)
