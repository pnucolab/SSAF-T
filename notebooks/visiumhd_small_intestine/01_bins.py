import json
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad, zarr
from common import *

import spatialdata as sd
o = sd.read_zarr(STORE)
t = o.tables["table"]
xy = np.asarray(t.obsm["spatial"], float)
c = (xy[:, 0] - X0) / PX; r = (xy[:, 1] - Y0) / PX
print("lattice residual", np.abs(c - np.round(c)).max(), np.abs(r - np.round(r)).max())
ci, ri = np.round(c).astype(int), np.round(r).astype(int)
cm = np.asarray(o.labels["ssaft_celltype_map"])
print("map", cm.shape, "bins", len(xy), "rows", ri.min(), ri.max(), "cols", ci.min(), ci.max())
X = sp.csr_matrix(t.X)
tot = np.asarray(X.sum(1)).ravel(); ng = np.diff(X.indptr)
genes = pd.Index(t.var_names.astype(str))
kg = kg_markers()
want = sorted(set(MOOR_BOTTOM + MOOR_TOP + MOOR_TEXT_BOTTOM + MOOR_TEXT_MID + MOOR_TEXT_TIP + MUSCLE +
                  PANETH_RAW + STEM + STEM_HELDOUT + PROLIF + EPI + ['Ptprc', 'Cd79a', 'Jchain', 'Dcn',
                  'Col1a1', 'Alpi', 'Fabp2', 'Muc2', 'Tff3', 'Chga', 'Fabp6', 'Apoa1', 'Mttp', 'Cd36',
                  'Neat1', 'Malat1', 'Fgfbp1', 'Saa1']))
have = [g for g in want if g in genes]
print("missing genes:", [g for g in want if g not in genes])
Xs = X[:, genes.get_indexer(have)].tocsc()
sp.save_npz(WORK / "bins_sel.npz", Xs)
np.savez_compressed(WORK / "bins_meta.npz", ri=ri, ci=ci, tot=tot.astype(np.float32), ng=ng,
                    shape=np.array(cm.shape), genes=np.array(have))
json.dump(dict(kg_markers=sorted(kg)), open(WORK / "kg_markers.json", "w"))


S = {}
S["n_bins_2um_in_tissue"] = int(len(tot))
S["umi_per_2um_bin"] = dict(median=float(np.median(tot)), mean=float(tot.mean()),
                            q25=float(np.percentile(tot, 25)), q75=float(np.percentile(tot, 75)),
                            frac_zero=float((tot == 0).mean()))
S["genes_per_2um_bin_median"] = float(np.median(ng))

k8 = (ri // 4) * 100000 + (ci // 4)
s8 = pd.Series(tot).groupby(k8).sum()
S["umi_per_8um_bin"] = dict(median=float(s8.median()), mean=float(s8.mean()),
                            q25=float(s8.quantile(.25)), q75=float(s8.quantile(.75)), n=int(len(s8)))
e = enact_cells()
et = np.asarray(e.X.sum(1)).ravel(); eg = np.asarray((e.X > 0).sum(1)).ravel()
S["umi_per_enact_cell"] = dict(median=float(np.median(et)), mean=float(et.mean()),
                               q25=float(np.percentile(et, 25)), q75=float(np.percentile(et, 75)),
                               n=int(len(et)), n_genes=int(e.n_vars),
                               note="ENACT's own area-weighted bin assignment (non-integer); "
                                    "over the genes ENACT itself reports, not the full bin panel")
S["genes_per_enact_cell_median"] = float(np.median(eg))
for f in (50, 100, 200, 500):
    S[f"frac_enact_cells_below_{f}_umi"] = float((et < f).mean())
for f in (100, 200):
    S[f"frac_enact_cells_below_{f}_genes"] = float((eg < f).mean())
s = ssaft_cells()
st = np.asarray(s.X.sum(1)).ravel(); sg = np.asarray((s.X > 0).sum(1)).ravel()
S["umi_per_ssaft_cell"] = dict(median=float(np.median(st)), q25=float(np.percentile(st, 25)),
                               q75=float(np.percentile(st, 75)), n=int(len(st)))
S["genes_per_ssaft_cell_median"] = float(np.median(sg))
for f in (50, 100, 200, 500):
    S[f"frac_ssaft_cells_below_{f}_umi"] = float((st < f).mean())


_nb = (e.obs.num_unique_bins.to_numpy(float) + e.obs.num_shared_bins.to_numpy(float))
S["enact_bins_per_cell_median"] = float(np.median(_nb))
S["enact_unique_bins_per_cell_median"] = float(np.median(e.obs.num_unique_bins))
S["enact_shared_bins_per_cell_median"] = float(np.median(e.obs.num_shared_bins))
S["enact_frac_cells_sharing_bins"] = float((e.obs.num_shared_bins.to_numpy(float) > 0).mean())
S["enact_area_um2_median_from_bins"] = float(np.median(_nb) * 4)
S["ssaft_area_um2_median"] = float(np.median(s.obs.area_um2))
print(json.dumps(S, indent=1))
json.dump(S, open(WORK / "sparsity.json", "w"), indent=1)
