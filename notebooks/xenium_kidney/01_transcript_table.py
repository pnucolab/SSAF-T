#!/usr/bin/env python
import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata as sd
from scipy.spatial import cKDTree
from spatialdata.transformations import get_transformation

import config as C

for p, what in ((C.BAYSOR_TX, "the authors' Baysor per-transcript assignment"),
                (C.BAYSOR_CELLS, "the authors' Baysor cell statistics"),
                (C.REF_H5AD, "the authors' annotated cells")):
    if not p.exists():
        raise SystemExit(f"missing input: {p}\n  this is {what}")

sdata = sd.read_zarr(C.SDATA, selection=("points", "shapes", "tables", "labels"))


aff = get_transformation(sdata.labels["ssaft_celltype_map"], "global").matrix
z_off = float(aff[0, -1])
print(f"lattice z offset {z_off:.4f} µm, voxel {C.VOXEL} µm", flush=True)

cols = ["transcript_id", "x", "y", "z", "feature_name", "cell_id", "qv",
        "nucleus_distance", "overlaps_nucleus"]
tx = sdata.points[C.POINTS][cols].compute()
print(f"all transcripts {len(tx):,}", flush=True)
tx["feature_name"] = tx.feature_name.astype(str)
ctrl = tx.feature_name.str.startswith(C.CONTROL_PREFIX)
tx = tx[~ctrl].reset_index(drop=True)
print(f"  control probes dropped {int(ctrl.sum()):,}  ->  {len(tx):,} gene molecules", flush=True)
tx["zlayer"] = np.floor((tx.z.values - z_off) / C.VOXEL).astype(int)
tx["cell_id"] = tx.cell_id.astype(str)

bs = pd.read_csv(C.BAYSOR_TX, usecols=["transcript_id", "cell"])
tx = tx.merge(bs.rename(columns={"cell": "baysor_cell"}), on="transcript_id", how="left")
tx["in_baysor_universe"] = tx.baysor_cell.notna()
tx["baysor_cell"] = tx.baysor_cell.fillna(0).astype(np.int64)
print(f"  in the Baysor universe (qv>=20): {100 * tx.in_baysor_universe.mean():.2f}%", flush=True)


g3 = sdata.shapes[C.SHAPES_3D]
layer_of = np.array([int(str(i).rsplit("_z", 1)[1]) for i in g3.index])
cell_of = np.array([str(i).rsplit("_z", 1)[0] for i in g3.index])
geom = np.asarray(g3.geometry.values, dtype=object)

assigned = np.full(len(tx), "", dtype=object)
for k in np.unique(tx.zlayer.values):
    m = tx.zlayer.values == k
    sel = layer_of == k
    if not sel.any():
        continue
    poly = gpd.GeoDataFrame({"ssaft_cell": cell_of[sel]}, geometry=list(geom[sel]),
                            index=np.arange(int(sel.sum())))
    idx = np.where(m)[0]
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(tx.x.values[idx], tx.y.values[idx]),
                           index=np.arange(len(idx)))
    j = gpd.sjoin(pts, poly, how="left", predicate="within")
    j = j[~j.index.duplicated()].reindex(np.arange(len(idx)))
    assigned[idx] = j.ssaft_cell.fillna("").values
    print(f"    z{k:>3}  {len(idx):>9,} molecules  {int(sel.sum()):>7,} polygons"
          f"  assigned {100 * (j.ssaft_cell.notna().mean()):5.1f}%", flush=True)
tx["ssaft_cell"] = assigned

cells = sdata.tables[C.TABLE].obs
tx["ssaft_label"] = tx.ssaft_cell.map(
    pd.Series(cells[C.LABEL_COL].astype(str).values,
              index=cells.cell_id.astype(str).values)).fillna("")


ref = ad.read_h5ad(C.REF_H5AD, backed="r")
stats = pd.read_csv(C.BAYSOR_CELLS)
S = np.stack([stats.x.values * C.PX_UM, stats.y.values * C.PX_UM], 1)
Rc = np.stack([ref.obs.x_centroid.values, ref.obs.y_centroid.values], 1)
d, k = cKDTree(S).query(Rc, k=1)
print(f"  ref cell -> baysor cell centroid match: <1e-3 µm {100 * (d < 1e-3).mean():.2f}%",
      flush=True)
tx["baysor_label"] = tx.baysor_cell.map(
    pd.Series(ref.obs.celltype_plot.astype(str).values, index=stats.cell.values[k])).fillna("")

tx["vendor_assigned"] = (tx.cell_id != "UNASSIGNED") & (tx.cell_id != "-1")
tx.drop(columns=["z", "x", "y"], inplace=True)
tx.attrs.clear()
out = C.TAB / "kidney_transcripts_3d.parquet"
tx.to_parquet(out)
print(f"-> {out}  {len(tx):,} rows", flush=True)

u = tx[tx.in_baysor_universe]
print(f"   qv>=20 universe {len(u):,}:  ssaft {100 * (u.ssaft_cell != '').mean():.2f}%"
      f"  baysor {100 * (u.baysor_cell > 0).mean():.2f}%"
      f"  vendor {100 * u.vendor_assigned.mean():.2f}%", flush=True)
