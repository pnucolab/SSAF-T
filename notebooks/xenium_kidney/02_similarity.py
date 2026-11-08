#!/usr/bin/env python
import json

import pandas as pd

import config as C

t = pd.read_parquet(C.TAB / "kidney_transcripts_3d.parquet",
                    columns=["baysor_cell", "ssaft_cell", "in_baysor_universe"])
t = t[t.in_baysor_universe]
t["ssaft"] = t.ssaft_cell.astype(str)
t.loc[t.ssaft == "nan", "ssaft"] = ""

nb = t[t.baysor_cell > 0].baysor_cell.value_counts()
ns = t[t.ssaft != ""].ssaft.value_counts()
both = t[(t.baysor_cell > 0) & (t.ssaft != "")]
pair = both.groupby(["baysor_cell", "ssaft"]).size().rename("n").reset_index()

fwd = pair.sort_values("n", ascending=False).drop_duplicates("baysor_cell").copy()
fwd["n_pub"] = fwd.baysor_cell.map(nb).to_numpy()
fwd["n_ssaft"] = fwd.ssaft.map(ns).to_numpy()
fwd["containment"] = fwd.n / fwd.n_pub
fwd["jaccard"] = fwd.n / (fwd.n_pub + fwd.n_ssaft - fwd.n)

missing = nb.index.difference(fwd.baysor_cell)
fwd = pd.concat([fwd, pd.DataFrame({"baysor_cell": missing, "containment": 0.0, "jaccard": 0.0})],
                ignore_index=True)
fwd[["baysor_cell", "containment", "jaccard", "n_pub"]].to_csv(
    C.TAB / "whole_tissue_similarity.csv", index=False)

rev = pair.sort_values("n", ascending=False).drop_duplicates("ssaft").copy()
rev["containment"] = rev.n / rev.ssaft.map(ns).to_numpy()
res = dict(n_published=int(len(nb)), n_ssaft=int(len(ns)),
           median_containment=float(fwd.containment.median()),
           median_jaccard=float(fwd.jaccard.median()),
           published_half_inside_one_ssaft=float((fwd.containment >= .5).mean()),
           ssaft_half_inside_one_published=float((rev.containment >= .5).mean()))
json.dump(res, open(C.TAB / "whole_tissue_similarity.json", "w"), indent=1)
for k, v in res.items():
    print(f"  {k:34} {v:,.3f}" if isinstance(v, float) else f"  {k:34} {v:,}")
print("->", C.TAB / "whole_tissue_similarity.csv")
