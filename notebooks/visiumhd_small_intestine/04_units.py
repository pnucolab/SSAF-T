import json
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage import morphology
from common import *

H_VILLUS = 80.0
MAX_A = 700.0

ax = np.load(WORK / "axes.npz")
DA, DB, tissue, follicle = ax["DA"], ax["DB"], ax["tissue"], ax["follicle"]
H, W = DA.shape
valid = tissue & ~follicle & np.isfinite(DA) & (DA <= MAX_A)


H_PROM = 30.0
Ds = ndi.gaussian_filter(np.nan_to_num(DA, nan=0.0), 2.0)
Ds[~valid] = 0
tips = morphology.h_maxima(Ds, H_PROM) & valid & (Ds > H_VILLUS)
lab, n = ndi.label(tips)
print("villus tips", n)
units = watershed(-Ds, markers=lab, mask=valid)
idx = np.arange(1, n + 1)
area = ndi.sum(np.ones_like(units), units, idx) * PX * PX
vil = valid & (DA > H_VILLUS)
vil_area = ndi.sum(vil.astype(np.int32), units, idx) * PX * PX
top = ndi.maximum(np.nan_to_num(DA, nan=0), units, idx)
width = vil_area / np.maximum(top - H_VILLUS, 1)
pan_px = ndi.sum(ax["paneth"].astype(np.int32), units, idx) * PX * PX
mus_px = ndi.sum(ax["muscle"].astype(np.int32), units, idx) * PX * PX
com = np.array(ndi.center_of_mass(vil, units, idx))
U = pd.DataFrame(dict(unit=idx, area_um2=area, villus_area_um2=vil_area, tip_um=top,
                      villus_width_um=width, paneth_band_um2=pan_px, muscle_um2=mus_px,
                      row=com[:, 0], col=com[:, 1]))
np.savez_compressed(WORK / "units.npz", units=units.astype(np.int32), tips=lab.astype(np.int32), valid=valid)
U.to_csv(WORK / "units.csv", index=False)
print(U.describe().T.to_string())

def sample(xy):
    c = np.round((xy[:, 0] - X0) / PX).astype(int); r = np.round((xy[:, 1] - Y0) / PX).astype(int)
    ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    r2, c2 = np.clip(r, 0, H - 1), np.clip(c, 0, W - 1)
    out = dict(r=r, c=c, A=np.where(ok, DA[r2, c2], np.nan), B=np.where(ok, DB[r2, c2], np.nan),
               unit=np.where(ok & valid[r2, c2], units[r2, c2], 0), follicle=ok & follicle[r2, c2])
    out["A"] = np.where(out["unit"] > 0, out["A"], np.nan)
    return pd.DataFrame(out)


s = ssaft_cells()
sc = sample(np.asarray(s.obsm["spatial"][:], float))
sc["label"] = s.obs.cellman_annotation.astype(str).values
sc["cell_id"] = s.obs_names.astype(str)
s.file.close()
sc.to_parquet(WORK / "ssaft_cells_axis.parquet")

e = enact_cells(backed="r")
ec = sample(np.asarray(e.obsm["spatial_um"][:], float))
ec["label"] = e.obs.cell_type.astype(str).values
e.file.close()
ec.to_parquet(WORK / "enact_cells_axis.parquet")

import spatialdata as sd
o = sd.read_zarr(STORE)
cm = np.asarray(o.labels["ssaft_celltype_map"])
names = list(o.tables["ssaft_cells"].uns["cellman_celltype_names"])
np.save(WORK / "celltype_map.npy", cm.astype(np.int16))
json.dump(names, open(WORK / "celltype_names.json", "w"))
m = (cm > 0) & valid
pix = pd.DataFrame(dict(r=np.nonzero(m)[0], c=np.nonzero(m)[1]))
pix["label"] = np.array(names, dtype=object)[cm[m] - 1]
pix["A"] = DA[m]; pix["B"] = DB[m]; pix["unit"] = units[m]
pix.to_parquet(WORK / "map_pixels_axis.parquet")
print("map pixels on axis", len(pix), " ssaft cells on axis", int((sc.unit > 0).sum()),
      " enact cells on axis", int((ec.unit > 0).sum()))
print(sc[sc.unit > 0].groupby("label").A.median().sort_values().to_string())
