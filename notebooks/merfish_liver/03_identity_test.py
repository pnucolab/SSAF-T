#!/usr/bin/env python
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C                                      # noqa: E402
from _common.paths import DATA                          # noqa: E402

N_BOOT = N_NULL = 300
PSEUDO = 1e-4
rng = np.random.default_rng(0)

PAIR = {
    "periportal hepatocyte": "periportal hepatocyte",
    "pericentral hepatocyte": "pericentral hepatocyte",
    "pericentral LSEC": "pericentral endothelial cell",
    "periportal LSEC": "periportal endothelial cell",
    "Kupffer cell": "Kupffer cell",
    "hepatic stellate cell": "hepatic stellate cell",
    "cholangiocyte": "bile duct epithelial cell",
}
MARKERS = C.SDATA.parent / "celltype_markers.yaml"
if not MARKERS.exists():
    raise SystemExit(f"missing input: {MARKERS}\n"
                     f"  this is the marker tree the CeLLMan annotation wrote; it ships with the "
                     f"SSAF-T output")

tx = pd.read_parquet(C.TAB / "liver_transcripts_3d.parquet")
tx["gene"] = tx.feature_name.str.upper()
S = tx.ssaft_cell.values != ""
B = tx.ref_cell.values != ""
genes = pd.Index(sorted(tx.gene.unique()))
print(f"{len(tx):,} molecules; SSAF-T {100 * S.mean():.1f}%, published {100 * B.mean():.1f}%",
      flush=True)


ref = ad.read_h5ad(DATA / "merfish/MERFISH_liver_object.h5ad")
if "counts" not in ref.layers:
    raise SystemExit("MERFISH_liver_object.h5ad has no layers['counts'] — the identity test "
                     "needs raw counts, not the normalised .X")
refX = sp.csr_matrix(ref.layers["counts"])
refG = pd.Index([str(g).upper() for g in ref.var_names])
refL = ref.obs.free_annotation.astype(str).values
P = {}
for L in pd.unique(refL):
    v = np.asarray(refX[refL == L].sum(0)).ravel()
    if v.sum() > 0:
        P[L] = (pd.Series(v / v.sum(), index=refG).groupby(level=0).sum()
                .reindex(genes).fillna(0).values)
bgp = tx.gene.value_counts(normalize=True).reindex(genes).fillna(0).values
SM = {k: np.log((v + PSEUDO) / (v + PSEUDO).sum()) for k, v in P.items()}
SM["__bg__"] = np.log((bgp + PSEUDO) / (bgp + PSEUDO).sum())


MARK = {}


def walk(node, anc):
    for k, v in node.items():
        if k == "children" or not isinstance(v, dict):
            continue
        own = set(v.get("defining", [])) | set(v.get("required", [])) | set(v.get("supporting", []))
        MARK[k] = {g.upper() for g in anc | own}
        if "children" in v:
            walk(v["children"], anc | own)


walk(yaml.safe_load(open(MARKERS)), set())


def llk(counts, key, keep=None):
    if keep is None:
        return float(counts @ SM[key] / counts.sum())
    q = P[key][keep] + PSEUDO
    return float(counts[keep] @ np.log(q / q.sum()) / counts[keep].sum())


rec = tx[S & ~B]
keys = list(P)
M = np.stack([SM[k] for k in keys])
rows = []
for L, d in rec.groupby("ssaft_label"):
    own = PAIR.get(L)
    if own is None or own not in P or len(d) < 200:
        continue
    obs = d.gene.value_counts().reindex(genes).fillna(0).values
    n = int(obs.sum())
    io = keys.index(own)
    mask = np.ones(len(keys), bool)
    mask[io] = False
    bs = rng.multinomial(n, obs / n, size=N_BOOT).astype(float)
    sc = (bs @ M.T) / n
    bdelta = sc[:, io] - sc[:, mask].max(1)
    null = rng.multinomial(n, bgp, size=N_NULL).astype(float) @ M[io] / n
    keep = ~np.isin(genes, sorted(MARK.get(L, set())))
    top10 = ~np.isin(np.arange(len(genes)),
                     np.argsort((P[own] + 1e-6) / (bgp + 1e-6))[::-1][:10])
    rows.append(dict(
        label=L, n=n, own=own,
        r_own=llk(obs, own), r_best_other=max(llk(obs, k) for k in keys if k != own),
        delta=float(np.mean(bdelta)), d_lo=float(np.percentile(bdelta, 2.5)),
        d_hi=float(np.percentile(bdelta, 97.5)),
        delta_null=llk(obs, own) - float(null.mean()),
        z=float((llk(obs, own) - null.mean()) / max(null.std(), 1e-9)),
        n_markers=int((~keep).sum()), n_genes_kept=int(keep.sum()),
        delta_nomarker=llk(obs, own, keep) - max(llk(obs, k, keep) for k in keys if k != own),
        delta_notop10=llk(obs, own, top10) - max(llk(obs, k, top10) for k in keys if k != own)))

ident = pd.DataFrame(rows).sort_values("delta")
ident.to_csv(C.TAB / "identity_test.csv", index=False)
pd.set_option("display.width", 230)
print("\n[identity of the recovered molecules, nats per molecule]")
print(ident[["label", "n", "own", "delta", "d_lo", "d_hi", "delta_null", "z",
             "n_markers", "delta_nomarker"]]
      .to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
print(f"\n  beats every competing cell type: {int((ident.delta > 0).sum())}/{len(ident)}"
      f"   (CI excludes 0: {int((ident.d_lo > 0).sum())}/{len(ident)})"
      f"   without the annotation's own markers: "
      f"{int((ident.delta_nomarker > 0).sum())}/{len(ident)}"
      f"   without the reference's ten most enriched genes: "
      f"{int((ident.delta_notop10 > 0).sum())}/{len(ident)}")
print(f"-> {C.TAB / 'identity_test.csv'}")
