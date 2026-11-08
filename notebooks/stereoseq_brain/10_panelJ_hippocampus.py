#!/usr/bin/env python
import json
import numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import box
from common import CACHE, OUT, PX, STORE
import config as C

rng = np.random.default_rng(1)
PAR = {"hippocampal pyramidal neuron"}
H = np.load(CACHE / "hippo_masks.npz"); ss, pub = H["ss"], H["pub"]
x0, y0, x1, y1 = (int(H[k]) for k in ("x0", "y0", "x1", "y1"))
tis = ss != "none"
band = {k: H[f"mask_{k}"] for k in ("DG", "CA1", "CA3")}
band["CA"] = band["CA1"] | band["CA3"]
SS_OK = {"DG": {"dentate gyrus granule cell"},
         "CA": {"CA1 pyramidal neuron", "CA3 pyramidal neuron"} | PAR}
PB_OK = {"DG": {"GN DG"}, "CA": {"EX CA"}}


import spatialdata as sd
sdata = sd.read_zarr(C.SDATA)
g = sdata.shapes["ssaft_cell_boundaries_2d"]
t = sdata.tables["ssaft_cells"]
lab_of = pd.Series(t.obs["cellman_annotation"].astype(str).values,
                   index=t.obs["cell_id"].astype(str).values if "cell_id" in t.obs else t.obs_names.astype(str))
g = g.copy()
g["label"] = [lab_of.get(str(i), "none") for i in g.index]
win = box((x0 * 5 + 2) - 3, (y0 * 5 + 2) - 3, ((x1 - 1) * 5 + 2) + 3, ((y1 - 1) * 5 + 2) + 3)
g = g[g.geometry.intersects(win)]
print(f"cells overlapping the window: {len(g):,}")

gi, gj = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1), indexing="ij")
pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy((gi * 5 + 2).ravel(), (gj * 5 + 2).ravel()),
                       crs=g.crs)
pts["k"] = np.arange(len(pts))
j = gpd.sjoin(pts, g[["geometry", "label"]], how="left", predicate="within")
j = j.drop_duplicates("k").sort_values("k")
sc = np.asarray(j.label.fillna("none").values, dtype=object).reshape(ss.shape).astype(str)
print(f"window pixels covered by a cell: {(sc != 'none').mean():.1%}   by the map: {tis.mean():.1%}")
np.savez_compressed(CACHE / "hippo_ss_cells.npz", sc=sc.astype(str))


def phi(a, b):
    return float(np.corrcoef(a[tis].astype(np.float32), b[tis].astype(np.float32))[0, 1])


def score(a, lab, keys, n=500):
    nul = []
    for _ in range(n):
        while True:
            dx, dy = rng.integers(0, lab.shape[0]), rng.integers(0, lab.shape[1])
            if np.hypot(min(dx, lab.shape[0] - dx) * PX, min(dy, lab.shape[1] - dy) * PX) >= 150:
                break
        nul.append(phi(a, np.isin(np.roll(np.roll(lab, dx, 0), dy, 1), list(keys))))
    return dict(r=phi(a, np.isin(lab, list(keys))), null=float(np.max(nul)),
                band_px=int(a[tis].sum()), label_px=int(np.isin(lab, list(keys))[tis].sum()))


res = {}
for k, keys in SS_OK.items():
    res[("SSAF-T cells", k)] = score(band[k], sc, keys)
    res[("SSAF-T map", k)] = score(band[k], ss, keys)
for k, keys in PB_OK.items():
    res[("Chen et al. cells", k)] = score(band[k], pub, keys)
json.dump({f"{w}|{k}": v for (w, k), v in res.items()}, open(OUT / "hippo_cells.json", "w"), indent=1)
print()
print(f"{'annotation':18s} {'field':4s} {'r':>6s} {'chance':>7s} {'band px':>9s} {'label px':>9s}")
for (w, k), v in res.items():
    print(f"{w:18s} {k:4s} {v['r']:6.3f} {v['null']:7.3f} {v['band_px']:9,d} {v['label_px']:9,d}")
print("-> hippo_cells.json")
