#!/usr/bin/env python
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import spatialdata as sd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C                                      # noqa: E402

H5 = C.ENACT_OUT / "tmap/weighted_by_area|sargent_cells_adata.h5"
CSV = C.ENACT_OUT / "chunks/weighted_by_area/sargent_results/merged_results.csv"
for p in (H5, CSV):
    if not p.exists():
        raise SystemExit(
            f"missing input: {p}\n"
            f"  this is ENACT's own output. Run ENACT on {C.BINNED} with the\n"
            f"  weighted_by_area bin-to-cell assignment and the Sargent cell typer, then set\n"
            f"  ENACT_OUT to its output directory.")

a = ad.read_h5ad(H5)
m = pd.read_csv(CSV)
print(f"h5 {a.shape}, merged {m.shape}", flush=True)

ids = a.obs_names.astype(str).to_numpy()
assert (ids == m["id"].astype(str).to_numpy()).all(), "row order differs"
assert (a.obs["cell_type"].astype(str).str.lower().to_numpy()
        == m["cell_type"].astype(str).str.lower().to_numpy()).all(), "labels differ"

obs = pd.DataFrame({
    "cell_type": pd.Categorical(m["cell_type"].to_numpy()),
    "patch": pd.Categorical(m["chunk_name"].to_numpy()),
    "num_shared_bins": m["num_shared_bins"].to_numpy(np.float32),
    "num_unique_bins": m["num_unique_bins"].to_numpy(np.float32),
    "num_transcripts": m["num_transcripts"].to_numpy(np.float32),
}, index=pd.Index(ids, name=None))
enact = ad.AnnData(X=a.X, obs=obs, var=pd.DataFrame(index=a.var_names.copy()))

px = m[["cell_x", "cell_y"]].to_numpy(np.float64)
enact.obsm["spatial"] = px.astype(np.float32)

tab = sd.read_zarr(C.SDATA, selection=("tables",)).tables[C.BIN_TABLE]
A = np.asarray(tab.uns["cellman"]["affine_px_to_um"], dtype=np.float64)
um = px @ A[:, :2].T + A[:, 2]
enact.obsm["spatial_um"] = um.astype(np.float32)
enact.uns["source"] = "ENACT (StarDist nuclei + weighted_by_area bins + Sargent)"
enact.uns["affine_px_to_um"] = A
enact.uns["note"] = ("X is area-weighted bin assignment over 1,073 genes, not integer counts "
                     "(median ~4/cell); not depth-comparable with ssaft_cells")

out = C.TAB / "enact_cells.h5ad"
enact.write_h5ad(out)
ref = np.asarray(tab.obsm["spatial"], dtype=float)
print(f"  spatial   (WSI px) {px.min(0).round(1)} .. {px.max(0).round(1)}")
print(f"  spatial_um         {um.min(0).round(1)} .. {um.max(0).round(1)}")
print(f"  bin table µm range {ref.min(0).round(1)} .. {ref.max(0).round(1)}  <- must overlap")
print(f"  cells per type:\n{obs.cell_type.value_counts().to_string()}")
print(f"-> {out}")
