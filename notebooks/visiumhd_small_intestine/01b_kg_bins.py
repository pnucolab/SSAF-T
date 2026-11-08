#!/usr/bin/env python
import json
import urllib.request

import numpy as np
import scipy.sparse as sp
import spatialdata as sd

from common import WORK, STORE

SLIM_F = WORK / "slim_index.json"
if not SLIM_F.exists():
    SLIM_F.write_text(urllib.request.urlopen(
        "http://192.168.101.31:8088/module/mouse/small_intestine/slim-index", timeout=120).read().decode())
terms = json.loads(SLIM_F.read_text())["terms"]
want = sorted({str(g) for v in terms.values() for k in ("DF", "R", "S", "N") for g in (v.get(k) or [])})
print(f"slim-index gene pool: {len(want)}", flush=True)

o = sd.read_zarr(STORE)
t = o.tables["table"]
import pandas as pd
genes = pd.Index(t.var_names.astype(str))
have = [g for g in want if g in genes]
print(f"in the panel: {len(have)};  missing: {[g for g in want if g not in genes]}", flush=True)
X = sp.csr_matrix(t.X)[:, genes.get_indexer(have)].tocsc()
sp.save_npz(WORK / "bins_kg.npz", X)
np.savez_compressed(WORK / "bins_kg_meta.npz", genes=np.array(have))
print("->", WORK / "bins_kg.npz", X.shape, flush=True)
