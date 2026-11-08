#!/usr/bin/env python
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import spatialdata as sd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C                                                          # noqa: E402
from _common.plot import BLUE, ORANGE, title, dapi_crop, dapi_axes, save    # noqa: E402

x0, y0, size = C.CROP_C
x1, y1 = x0 + size, y0 + size


pv = pd.read_csv(C.BAYSOR_TX.parent / "polygon.csv")
pv[["x_location", "y_location"]] *= C.PX_UM
keep = pv.groupby("polygon_number")[["x_location", "y_location"]].agg(["min", "max"])
hit = keep[(keep[("x_location", "max")] >= x0) & (keep[("x_location", "min")] <= x1) &
           (keep[("y_location", "max")] >= y0) & (keep[("y_location", "min")] <= y1)].index
pub = gpd.GeoDataFrame(geometry=[
    Polygon(g[["x_location", "y_location"]].to_numpy())
    for _, g in pv[pv.polygon_number.isin(hit)].groupby("polygon_number") if len(g) >= 3])
print(f"published outlines in the window: {len(pub):,}", flush=True)


ss = gpd.read_parquet(C.SDATA / f"shapes/{C.SHAPES_2D}/shapes.parquet")
ss = ss.set_crs(None, allow_override=True).cx[x0:x1, y0:y1]
print(f"SSAF-T outlines in the window:    {len(ss):,}", flush=True)

sdata = sd.read_zarr(C.SDATA, selection=("images",))
dp, ep = dapi_crop(sdata.images[C.DAPI_IMAGE], (x0, y0, x1, y1), 1 / C.PX_UM)

fig, axp = plt.subplots(1, 2, figsize=(2.95, 1.75), gridspec_kw=dict(wspace=.03))
dapi_axes(axp[0], dp, ep, ss, BLUE, "SSAF-T", bar_um=50)
dapi_axes(axp[1], dp, ep, pub, ORANGE, "Baysor (published)")
title(axp[0], "C", "Cell outlines (Xenium kidney)", pad=16)
axp[0].text(0, 1.015, "the same field twice: each method's cell outlines\non the nuclear stain (DAPI)",
            transform=axp[0].transAxes, fontsize=5, color="#555", va="bottom")
save(fig, C.OUT, "C", "kidney_dapi_outlines")
