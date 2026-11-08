import json
import numpy as np, pandas as pd
from scipy import ndimage as ndi
from common import *

U = pd.read_csv(WORK / "units.csv")
ax = np.load(WORK / "axes.npz"); un = np.load(WORK / "units.npz"); units = un["units"]
fol = ndi.binary_dilation(ax["follicle"], iterations=25)
touch = np.bincount(units[fol].ravel(), minlength=units.max() + 1)
U["touches_follicle"] = touch[U.unit.values] > 0
el = U[(U.villus_width_um.between(40, 80)) & (U.paneth_band_um2 >= 300) & (U.muscle_um2 >= 500)
       & (~U.touches_follicle) & (U.tip_um <= 450)].sort_values("tip_um", ascending=False)
print(f"eligible single-villus units: {len(el)} of {len(U)}")
pu = pd.read_csv(OUT / "per_unit_ssaft_A.csv").set_index("unit")
el = el.join(pu, on="unit")
print(el.head(12).to_string())
pick = el.head(2)
pctl = {int(u): float((pu.rho <= pu.rho.get(u, np.nan)).mean()) for u in pick.unit}
json.dump(dict(rule=__doc__.strip(), n_eligible=int(len(el)), n_units=int(len(U)),
               picked=pick.unit.astype(int).tolist(), supplementary=[466],
               picked_tip_um=pick.tip_um.round(1).tolist(),
               picked_rho=pick.rho.round(3).tolist(),
               picked_rho_percentile_among_units=pctl,
               eligible_rho_median=float(el.rho.median()),
               eligible_frac_rho_positive=float((el.rho > 0).mean())),
          open(WORK / "picked_units.json", "w"), indent=1)
el.to_csv(OUT / "eligible_units.csv", index=False)
print(json.load(open(WORK / "picked_units.json")))
