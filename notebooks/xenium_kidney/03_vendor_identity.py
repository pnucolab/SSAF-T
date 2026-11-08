#!/usr/bin/env python
import json

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

import config as C


PAIR = {
    "proximal tubule cell S2": "PTS2", "proximal tubule cell S1": "PTS1",
    "thick ascending limb cell": "TAL", "proximal tubule cell S3": "PTS3",
    "glomerular endothelial cell": "Glom-EC", "endothelial cell": "EC",
    "macrophage": "Immune", "fibroblast": "Fib",
    "collecting duct principal cell": "PC", "distal convoluted tubule cell": "DCT",
    "connecting tubule cell": "CNT", "mural cell": "Per-SMC",
    "type B intercalated cell": "ICB", "descending thin limb cell": "DTL",
    "type A intercalated cell": "ICA", "podocyte": "Pod",
    "urothelial cell": "Uro", "parietal epithelial cell": "PEC",
}
rng = np.random.default_rng(0)
N_BOOT = 200

tx = pd.read_parquet(C.TAB / "kidney_transcripts_3d.parquet",
                     columns=["feature_name", "cell_id", "baysor_cell", "baysor_label",
                              "ssaft_cell", "ssaft_label", "in_baysor_universe",
                              "vendor_assigned"])
tx = tx[tx.in_baysor_universe].copy()
tx["gene"] = tx.feature_name.str.upper()
print(f"{len(tx):,} molecules (qv >= 20)", flush=True)


shared = tx[tx.vendor_assigned & (tx.baysor_cell > 0) & (tx.baysor_label != "")]
vl = (shared.groupby(["cell_id", "baysor_label"]).size().rename("n").reset_index()
      .sort_values("n").drop_duplicates("cell_id", keep="last"))
tot = shared.groupby("cell_id").size()
vl["purity"] = vl.n.values / tot.reindex(vl.cell_id).values
vlab = pd.Series(vl.baysor_label.values, index=vl.cell_id.values)
print(f"vendor cells labelled through shared molecules: {len(vlab):,}; "
      f"median share of their Baysor-shared molecules carrying that label "
      f"{100 * np.median(vl.purity):.0f}%", flush=True)
tx["vendor_label"] = tx.cell_id.map(vlab).fillna("")


ref = ad.read_h5ad(C.REF_H5AD)
X = sp.csr_matrix(ref.raw.X if ref.raw is not None else ref.X)
G = pd.Index([str(g).upper() for g in (ref.raw.var_names if ref.raw is not None else ref.var_names)])
L = ref.obs.celltype_plot.astype(str).values
genes = pd.Index(sorted(tx.gene.unique()))
P = {}
for k in pd.unique(L):
    v = np.asarray(X[L == k].sum(0)).ravel()
    if v.sum():
        P[k] = (pd.Series(v / v.sum(), index=G).groupby(level=0).sum()
                .reindex(genes).fillna(0).values)
keys = list(P)
M = np.stack([np.log((P[k] + 1e-4) / (P[k] + 1e-4).sum()) for k in keys])


def margins(df, labcol):
    rows = []
    for lab, d in df.groupby(labcol):
        if lab == "" or lab not in P or len(d) < 200:
            continue
        c = d.gene.value_counts().reindex(genes).fillna(0).values
        n = c.sum()
        io = keys.index(lab)
        mask = np.ones(len(keys), bool)
        mask[io] = False
        sc = M @ c / n
        delta = sc[io] - sc[mask].max()
        bs = rng.multinomial(int(n), c / n, size=N_BOOT).astype(float) @ M.T / n
        bd = bs[:, io] - bs[:, mask].max(1)
        rows.append(dict(label=lab, n=int(n), delta=float(delta),
                         lo=float(np.percentile(bd, 2.5)), hi=float(np.percentile(bd, 97.5)),
                         best_other=keys[np.where(mask)[0][sc[mask].argmax()]]))
    return pd.DataFrame(rows).set_index("label")


disc = tx[tx.baysor_cell == 0]
S = disc.ssaft_cell != ""
V = disc.vendor_assigned
g1 = disc[S].assign(lab=disc[S].ssaft_label.map(PAIR).fillna(""))
g2 = disc[V].assign(lab=disc[V].vendor_label)
g3 = disc[V & ~S].assign(lab=disc[V & ~S].vendor_label)
print(f"\nBaysor-discarded molecules {len(disc):,}: SSAF-T takes {100 * S.mean():.1f}%, "
      f"vendor takes {100 * V.mean():.1f}%, vendor takes where SSAF-T refused "
      f"{100 * (V & ~S).mean():.1f}%", flush=True)

m1, m2, m3 = margins(g1, "lab"), margins(g2, "lab"), margins(g3, "lab")
tab = (m1[["n", "delta", "lo", "hi"]].add_prefix("ssaft_")
       .join(m2[["n", "delta", "lo", "hi"]].add_prefix("vendor_"), how="outer")
       .join(m3[["n", "delta", "lo", "hi"]].add_prefix("vendor_refused_"), how="outer"))
tab = tab.sort_values("ssaft_n", ascending=False)
tab.to_csv(C.TAB / "vendor_identity.csv")
pd.set_option("display.width", 220)
print("\n[identity margin of the EXTRA molecules, nats per molecule]")
print(tab[["ssaft_n", "ssaft_delta", "vendor_n", "vendor_delta",
           "vendor_refused_n", "vendor_refused_delta"]]
      .to_string(float_format=lambda v: f"{v:8.3f}"))

both = tab.dropna(subset=["ssaft_delta", "vendor_delta"])
summ = dict(
    n_types=len(both),
    ssaft_above_zero=int((m1.lo > 0).sum()), ssaft_types=len(m1),
    vendor_above_zero=int((m2.lo > 0).sum()), vendor_types=len(m2),
    refused_above_zero=int((m3.lo > 0).sum()), refused_types=len(m3),
    ssaft_higher=int((both.ssaft_delta > both.vendor_delta).sum()),
    median_ssaft=float(both.ssaft_delta.median()),
    median_vendor=float(both.vendor_delta.median()),
    median_refused=float(m3.delta.median()) if len(m3) else None,
    pct_disc_ssaft=float(S.mean()), pct_disc_vendor=float(V.mean()),
    pct_disc_vendor_refused=float((V & ~S).mean()))
print(f"\nabove zero (95% CI): SSAF-T {summ['ssaft_above_zero']}/{summ['ssaft_types']}  "
      f"vendor {summ['vendor_above_zero']}/{summ['vendor_types']}  "
      f"vendor-where-SSAF-T-refused {summ['refused_above_zero']}/{summ['refused_types']}")
print(f"same type, SSAF-T margin higher than vendor's: {summ['ssaft_higher']}/{summ['n_types']}; "
      f"medians {summ['median_ssaft']:.3f} vs {summ['median_vendor']:.3f}"
      f" (vendor where SSAF-T refused {summ['median_refused']})")
json.dump(summ, open(C.TAB / "vendor_identity.json", "w"), indent=1)
print(f"-> {C.TAB / 'vendor_identity.csv'}")
