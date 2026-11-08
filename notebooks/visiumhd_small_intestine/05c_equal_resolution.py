import json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from common import *

st = json.load(open(OUT / "stats.json"))
sc = pd.read_parquet(WORK / "ssaft_cells_axis_scored.parquet")
ec = pd.read_parquet(WORK / "enact_cells_axis_C.parquet")
SS2 = {"Paneth cell": 0, "intestinal stem cell": 0, "transit amplifying cell": 0,
       "immature enterocyte": 1, "villus-bottom enterocyte": 1, "absorptive enterocyte": 1,
       "distal enterocyte": 1, "villus-tip enterocyte": 1}
EN2 = {"Paneth cells": 0, "Crypt cells": 0, "Enterocytes": 1}
def per_unit(df, mp):
    d = df[df.label.isin(mp) & np.isfinite(df.A) & (df.unit > 0)].copy(); d["lv"] = d.label.map(mp)
    r = d.groupby("unit").apply(lambda g: spearmanr(g.A, g.lv).statistic if len(g) >= 20 and g.lv.nunique() == 2 else np.nan).dropna()
    return d, r
ds, rs = per_unit(sc, SS2); de, re_ = per_unit(ec, EN2)
both = rs.index.intersection(re_.index)
res = dict(ssaft=dict(n_units=int(len(rs)), median_rho=float(rs.median()), frac_positive=float((rs > 0).mean()),
                      pooled_rho=float(spearmanr(ds.A, ds.lv).statistic),
                      crypt_median_um=float(ds.A[ds.lv == 0].median()), absorptive_median_um=float(ds.A[ds.lv == 1].median())),
           enact=dict(n_units=int(len(re_)), median_rho=float(re_.median()), frac_positive=float((re_ > 0).mean()),
                      pooled_rho=float(spearmanr(de.A, de.lv).statistic),
                      crypt_median_um=float(de.A[de.lv == 0].median()), absorptive_median_um=float(de.A[de.lv == 1].median())),
           paired_units=int(len(both)),
           frac_units_ssaft_higher=float((rs[both] > re_[both]).mean()),
           median_paired_difference=float((rs[both] - re_[both]).median()))
from scipy.stats import wilcoxon
res["wilcoxon_p"] = float(wilcoxon(rs[both], re_[both]).pvalue)
print(json.dumps(res, indent=1))
st["equal_resolution_crypt_vs_absorptive"] = res
json.dump(st, open(OUT / "stats.json", "w"), indent=1, default=float)
pd.DataFrame(dict(ssaft=rs, enact=re_)).to_csv(OUT / "per_unit_equal_resolution.csv")
