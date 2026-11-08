#!/usr/bin/env python
import json

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

import config as C

tx = pd.read_parquet(C.TAB / "liver_transcripts_3d.parquet", columns=["ref_cell", "ssaft_cell"])
a = tx.ref_cell.astype(str).to_numpy()
b = tx.ssaft_cell.astype(str).to_numpy()
a[a == "nan"] = ""
b[b == "nan"] = ""
both = (a != "") & (b != "")
print(f"molecules {len(a):,}; published {100 * (a != '').mean():.1f}%, "
      f"SSAF-T {100 * (b != '').mean():.1f}%, both {100 * both.mean():.1f}%", flush=True)

rng = np.random.default_rng(0)
idx = np.where(both)[0]
sub = rng.choice(idx, min(len(idx), 3_000_000), replace=False)
ari = float(adjusted_rand_score(a[sub], b[sub]))

pair = pd.DataFrame({"a": a[both], "b": b[both]}).value_counts().rename("n").reset_index()
best = pair.sort_values("n", ascending=False).drop_duplicates("a")
na = pd.Series(a[a != ""]).value_counts()
nb = pd.Series(b[b != ""]).value_counts()
best["jac"] = best.n / (best.a.map(na) + best.b.map(nb) - best.n)
best["cont"] = best.n / best.a.map(na)
jac = best.set_index("a").jac.reindex(na.index).fillna(0.0)
cont = best.set_index("a").cont.reindex(na.index).fillna(0.0)
pd.DataFrame({"jaccard": jac, "containment": cont}).to_csv(C.TAB / "per_cell_jaccard_liver.csv")

res = dict(n_molecules=int(len(a)), assigned_published=float((a != "").mean()),
           assigned_ssaft=float((b != "").mean()),
           n_cells_published=int(na.size), n_cells_ssaft=int(nb.size),
           ARI=ari, ari_molecules=int(len(sub)),
           matched=float((jac >= .5).mean()), median_jaccard=float(jac.median()),
           median_containment=float(cont.median()), contained_80=float((cont >= .8).mean()))
json.dump(res, open(C.TAB / "similarity_liver.json", "w"), indent=1)
print(json.dumps(res, indent=1))
