#!/usr/bin/env python
import sys
from pathlib import Path

import numpy as np, pandas as pd, h5py

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C_                                      # noqa: E402

C = C_.CACHE
X0, Y0 = C_.X0, C_.Y0


class _M:
    MOSTA = C_.MOSTA


C.mkdir(parents=True, exist_ok=True)

g = pd.read_csv(C_.MOSTA / "Mouse_brain_Adult_GEM_bin1.tsv.gz", sep="\t",
                dtype={"geneID": "category", "x": np.int32, "y": np.int32, "MIDCounts": np.int32})
print("bin1 rows", len(g), flush=True)
genes = np.asarray(g.geneID.cat.categories, dtype=str)
np.savez(C / "bin1.npz", gene=g.geneID.cat.codes.values.astype(np.int16),
         x=g.x.values.astype(np.int16), y=g.y.values.astype(np.int16),
         umi=g.MIDCounts.values.astype(np.int16), genes=genes)
del g

c = pd.read_csv(C_.MOSTA / "Mouse_brain_Adult_GEM_CellBin.tsv.gz", sep="\t",
                usecols=["x", "y", "UMICount", "label"],
                dtype={"x": np.int32, "y": np.int32, "UMICount": np.int32, "label": np.float64})
print("cellbin rows", len(c), "unlabelled", int(c.label.isna().sum()), flush=True)
c = c.dropna(subset=["label"]); c["label"] = c.label.astype(np.int64)
agg = c.groupby("label").agg(umi=("UMICount", "sum"), x=("x", "mean"), y=("y", "mean"))
agg["n_dnb"] = c.drop_duplicates(["x", "y", "label"]).groupby("label").size()
agg["x_um"] = 0.5 * agg.x - X0; agg["y_um"] = 0.5 * agg.y - Y0

with h5py.File(C_.MOSTA / "Mouse_brain_cell_bin.h5ad", "r") as h:
    names = h["obs"][h["obs"].attrs["_index"]][:].astype(str)
    cats = h["obs/annotation/categories"][:].astype(str)
    codes = h["obs/annotation/codes"][:]
    ip = h["layers/counts/indptr"][:]; dat = h["layers/counts/data"]
    ngene = np.diff(ip)
    tot = np.add.reduceat(dat[:], ip[:-1]) if len(dat) else np.zeros(len(ip) - 1)
    sp = h["obsm/spatial"][:]
cb = pd.DataFrame({"cell": names, "annotation": cats[codes], "n_genes": ngene,
                   "total_counts": tot, "h5_x": sp[:, 0], "h5_y": sp[:, 1]})
cb["label"] = cb.cell.str.split("_").str[1].astype(np.int64)
cb = cb.merge(agg[["x_um", "y_um", "umi", "n_dnb"]], left_on="label", right_index=True, how="left")
print("matched", cb.x_um.notna().mean(), flush=True)
cb.to_parquet(C / "cellbin_cells.parquet")
print(cb.describe().T.to_string())


lab = np.zeros((C_.W, C_.H), np.int32)
ix = np.rint(0.5 * c.x.to_numpy(np.float64) - X0).astype(np.int64)
iy = np.rint(0.5 * c.y.to_numpy(np.float64) - Y0).astype(np.int64)
m = (ix >= 0) & (ix < C_.W) & (iy >= 0) & (iy < C_.H)
lab[ix[m], iy[m]] = c.label.to_numpy(np.int64)[m]
np.savez_compressed(C / "cellbin_label_raster.npz", label=lab)
ann = cb[["label", "annotation"]].dropna()
ann.to_parquet(C / "cellbin_annotation.parquet")
print(f"published CellBin raster: {int((lab > 0).sum()):,} labelled µm pixels, "
      f"{ann.label.nunique():,} cells", flush=True)


import json as _json
gem_cells = int(c.label.nunique())
kept = cb[["label", "n_genes", "total_counts"]]
dropped = gem_cells - len(cb)
qc = dict(gem_cells=gem_cells, published_cells=int(len(cb)), dropped=int(dropped),
          kept_min_genes=int(kept.n_genes.min()), kept_min_umi=int(kept.total_counts.min()))
_json.dump(qc, open(C / "stats_cellbin_qc.json", "w"), indent=1)
print("CellBin QC", qc, flush=True)
