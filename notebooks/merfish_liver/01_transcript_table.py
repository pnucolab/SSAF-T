#!/usr/bin/env python
import argparse
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata as sd

import config as C

ap = argparse.ArgumentParser()
ap.add_argument("--chunk", type=int, default=1_000_000)
A = ap.parse_args()

for p, what in ((C.PUB_TX, "the authors' transcript table"),
                (C.PUB_POLY, "the authors' cell polygons"),
                (C.PUB_META, "the authors' cell metadata and annotation")):
    if not p.exists():
        raise SystemExit(f"missing input: {p}\n  this is {what}")

tx = pd.read_csv(C.PUB_TX, usecols=["global_x", "global_y", "global_z", "gene"])
tx = tx.rename(columns={"global_x": "x", "global_y": "y", "global_z": "z"})
tx["feature_name"] = tx.gene.astype(str)
ctrl = tx.feature_name.str.lower().str.startswith(C.CONTROL_PREFIX)
tx = tx[~ctrl].drop(columns=["gene"]).reset_index(drop=True)
tx["zlayer"] = tx.z.astype(int)
print(f"transcripts {len(tx):,}  planes {sorted(tx.zlayer.unique())}"
      f"  (blanks dropped {int(ctrl.sum()):,})", flush=True)
pts_all = gpd.GeoDataFrame(geometry=gpd.points_from_xy(tx.x.values, tx.y.values),
                           index=np.arange(len(tx)))


pub = gpd.read_parquet(C.PUB_POLY)
if "geometry" not in pub.columns and "Geometry" in pub.columns:
    pub = pub.set_geometry("Geometry")
pub = pub[pub.get("ZIndex", pd.Series(0, index=pub.index)) == 0] if "ZIndex" in pub else pub
ref_poly = gpd.GeoDataFrame({"ref_cell": pub["EntityID"].astype(str).values},
                            geometry=pub.geometry.values, index=range(len(pub)))
j = gpd.sjoin(pts_all, ref_poly, how="left", predicate="within")
j = j[~j.index.duplicated()].reindex(np.arange(len(tx)))
tx["ref_cell"] = j.ref_cell.fillna("").values
meta = pd.read_csv(C.PUB_META)
tx["ref_label"] = tx.ref_cell.map(
    pd.Series(meta.free_annotation.astype(str).values,
              index=meta.EntityID.astype(str).values)).fillna("")
print(f"  paper assigned {100 * (tx.ref_cell != '').mean():.2f}%", flush=True)


o = sd.read_zarr(C.SDATA, selection=("shapes", "tables"))
g3 = o.shapes[C.SHAPES_3D]
layer_of = np.array([int(str(i).rsplit("_z", 1)[1]) for i in g3.index])
cell_of = np.array([str(i).rsplit("_z", 1)[0] for i in g3.index])
geom = np.asarray(g3.geometry.values, dtype=object)
assigned = np.full(len(tx), "", dtype=object)
for k in np.unique(tx.zlayer.values):
    sel = layer_of == k
    if not sel.any():
        continue
    idx = np.where(tx.zlayer.values == k)[0]
    poly = gpd.GeoDataFrame({"ssaft_cell": cell_of[sel]}, geometry=list(geom[sel]),
                            index=np.arange(int(sel.sum())))
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(tx.x.values[idx], tx.y.values[idx]),
                           index=np.arange(len(idx)))
    jj = gpd.sjoin(pts, poly, how="left", predicate="within")
    jj = jj[~jj.index.duplicated()].reindex(np.arange(len(idx)))
    assigned[idx] = jj.ssaft_cell.fillna("").values
    print(f"    z{k}  {len(idx):>9,} molecules  {int(sel.sum()):>6,} polygons"
          f"  assigned {100 * jj.ssaft_cell.notna().mean():5.1f}%", flush=True)
tx["ssaft_cell"] = assigned
cells = o.tables[C.TABLE].obs
tx["ssaft_label"] = tx.ssaft_cell.map(
    pd.Series(cells[C.LABEL_COL].astype(str).values,
              index=cells.cell_id.astype(str).values)).fillna("")


nuc = gpd.GeoDataFrame(geometry=o.shapes[C.NUCLEI].geometry.values,
                       index=range(len(o.shapes[C.NUCLEI]))).set_crs(None, allow_override=True)
print(f"nuclei {len(nuc):,}", flush=True)
nd = np.full(len(tx), np.nan)
t0 = time.time()
for a in range(0, len(tx), A.chunk):
    b = min(a + A.chunk, len(tx))
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(tx.x.values[a:b], tx.y.values[a:b]),
                           index=np.arange(b - a)).set_crs(None, allow_override=True)
    nj = gpd.sjoin_nearest(pts, nuc, how="left", distance_col="d")
    nj = nj[~nj.index.duplicated()].reindex(np.arange(b - a))
    nd[a:b] = nj.d.values
    print(f"  {b:>10,} / {len(tx):,}   {time.time() - t0:6.0f}s", flush=True)
tx["nucleus_distance"] = nd
tx["overlaps_nucleus"] = (tx.nucleus_distance == 0).astype(float)

tx.drop(columns=["x", "y", "z"], inplace=True)
tx.attrs.clear()
out = C.TAB / "liver_transcripts_3d.parquet"
tx.to_parquet(out)
print(f"-> {out}  {len(tx):,} rows", flush=True)
print(f"   ssaft {100 * (tx.ssaft_cell != '').mean():.2f}%"
      f"   paper {100 * (tx.ref_cell != '').mean():.2f}%"
      f"   median nucleus distance {np.nanmedian(nd):.2f} µm", flush=True)
