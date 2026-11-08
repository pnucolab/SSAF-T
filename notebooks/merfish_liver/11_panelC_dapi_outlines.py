#!/usr/bin/env python
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import spatialdata as sd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C                                                          # noqa: E402
from _common.plot import BLUE, ORANGE, title, dapi_crop, dapi_axes, save    # noqa: E402

x0, y0, size = C.CROP_C
x1, y1 = x0 + size, y0 + size

pub = gpd.read_parquet(C.PUB_POLY)
if "geometry" not in pub.columns and "Geometry" in pub.columns:
    pub = pub.set_geometry("Geometry")
if "ZIndex" in pub:
    pub = pub[pub.ZIndex == 0]
pub = pub[~pub.geometry.is_empty].set_crs(None, allow_override=True).cx[x0:x1, y0:y1]
print(f"published outlines in the window: {len(pub):,}", flush=True)

ss = gpd.read_parquet(C.SDATA / f"shapes/{C.SHAPES_2D}/shapes.parquet")
ss = ss.set_crs(None, allow_override=True).cx[x0:x1, y0:y1]
print(f"SSAF-T outlines in the window:    {len(ss):,}", flush=True)

sdata = sd.read_zarr(C.SDATA, selection=("images",))


img = sdata.images[C.DAPI_IMAGE]
img = img["scale0"]["image"] if hasattr(img, "keys") else img
dp, ep = dapi_crop(img.isel(c=0, z=0), (x0, y0, x1, y1), C.PX_PER_UM)

fig, axp = plt.subplots(1, 2, figsize=(2.95, 1.75), gridspec_kw=dict(wspace=.03))
dapi_axes(axp[0], dp, ep, ss, BLUE, "SSAF-T", bar_um=50)
dapi_axes(axp[1], dp, ep, pub, ORANGE, "Cellpose (published)")
title(axp[0], "C", "Cell outlines (MERFISH liver)", pad=16)
axp[0].text(0, 1.015, "the same field twice: each method's cell outlines\non the nuclear stain (DAPI)",
            transform=axp[0].transAxes, fontsize=5, color="#555", va="bottom")
save(fig, C.OUT, "C", "liver_dapi_outlines")
