#!/usr/bin/env python
import json
import numpy as np, pandas as pd, anndata as ad
from scipy.stats import spearmanr, kendalltau
from common import *

rng = np.random.default_rng(0)
G = np.load(CACHE / "geometry.npz"); depth = G["depth"]


px = pd.read_parquet(CACHE / "cortex_pixels.parquet")
valid = np.zeros((NX, NY), bool); valid[px.ix.values, px.iy.values] = True
cw = np.full((NX, NY), -1, np.int32); cw[px.ix.values, px.iy.values] = px.win.values

a = ssaft_cells()
xy = np.asarray(a.obsm["spatial"][:], float)
sc = pd.DataFrame(dict(label=a.obs["cellman_annotation"].astype(str).values,
                       x_um=xy[:, 0], y_um=xy[:, 1]))
a.file.close()
ix = np.clip((sc.x_um // PX).astype(int), 0, NX - 1); iy = np.clip((sc.y_um // PX).astype(int), 0, NY - 1)
sc["valid"] = valid[ix, iy]; sc["depth"] = depth[ix, iy]; sc["win"] = cw[ix, iy]
sc["blk"] = (ix // 10) * 10000 + (iy // 10)
scv = sc[sc.valid & np.isfinite(sc.depth)].copy()
print("SSAF-T cells in the valid cortex", len(scv))
print(scv.label.value_counts().head(12).to_string())


def order_stats(df, labels, rank, tag, unit, nboot=1000, nperm=1000):
    df = df[df.label.isin(labels)].copy()
    df["rank"] = df.label.map(rank)
    med = df.groupby("label").depth.agg(["size", "median", lambda s: s.quantile(.25), lambda s: s.quantile(.75)])
    med.columns = ["n", "median", "q25", "q75"]; med = med.loc[[l for l in labels if l in med.index]]
    rho = spearmanr(df["rank"], df.depth).statistic
    bm = df.groupby("blk").depth.mean(); bk = df.blk.values
    codes, ub = pd.factorize(bk); bmv = bm.reindex(ub).values
    rho_b = spearmanr(df["rank"], bmv[codes]).statistic
    null = np.array([spearmanr(df["rank"], rng.permutation(bmv)[codes]).statistic for _ in range(nperm)])
    groups = {l: df[df.label == l] for l in med.index}
    blist = {l: {k: v.depth.values for k, v in g.groupby("blk")} for l, g in groups.items()}
    allb = np.array(sorted(set(df.blk)))
    boots = {l: [] for l in med.index}
    for _ in range(nboot):
        s = rng.choice(allb, len(allb), replace=True)
        for l in med.index:
            arr = [blist[l][k] for k in s if k in blist[l]]
            boots[l].append(np.median(np.concatenate(arr)) if arr else np.nan)
    pairs = []
    lab_sorted = sorted(med.index, key=lambda l: rank[l])
    for aa, cc in zip(lab_sorted[:-1], lab_sorted[1:]):
        dlt = np.array(boots[cc]) - np.array(boots[aa])
        pairs.append(dict(shallow=aa, deep=cc, diff=float(med.loc[cc, "median"] - med.loc[aa, "median"]),
                          ci_lo=float(np.nanpercentile(dlt, 2.5)), ci_hi=float(np.nanpercentile(dlt, 97.5)),
                          frac_boot_wrong=float(np.nanmean(dlt <= 0))))
    per = []
    for w, g in df.groupby("win"):
        m = g.groupby("label").depth.median()
        m = m[[l for l in m.index if (g.label == l).sum() >= 5]]
        if len(m) >= 3:
            tau = kendalltau([rank[l] for l in m.index], m.values).statistic
            per.append(dict(win=int(w), n_types=len(m), tau=float(tau)))
    per = pd.DataFrame(per)
    order_obs = list(med.sort_values("median").index)
    res = dict(unit=unit, n=int(len(df)), medians=med.round(4).reset_index().to_dict("records"),
               observed_order=order_obs,
               order_matches_expected=bool(all(np.diff([rank[l] for l in order_obs]) >= 0)),
               spearman_rank_depth=float(rho), spearman_blockmean=float(rho_b),
               block_perm_null_max=float(np.abs(null).max()),
               block_perm_p=float((np.abs(null) >= abs(rho_b)).mean() + 1 / (nperm + 1)),
               adjacent_pairs=pairs,
               columns=dict(n=int(len(per)), tau_median=float(per.tau.median()) if len(per) else None,
                            frac_tau_pos=float((per.tau > 0).mean()) if len(per) else None,
                            frac_tau_ge_0p8=float((per.tau >= 0.8).mean()) if len(per) else None,
                            per=per.to_dict("records")))
    print(f"\n[{tag}]  unit={unit}  n={len(df)}")
    print("  order", [LAYER_SHORT.get(l, l) for l in order_obs], " matches expected:", res["order_matches_expected"])
    print(f"  rho={rho:.3f}  rho_block={rho_b:.3f}  null|max|={np.abs(null).max():.3f}  p={res['block_perm_p']:.4f}")
    print(med.round(3).to_string())
    print(pd.DataFrame(pairs).round(3).to_string())
    if len(per):
        print(f"  columns n={len(per)}  median tau {per.tau.median():.2f}  tau>0 {(per.tau>0).mean():.2f}  "
              f"tau>=0.8 {(per.tau>=0.8).mean():.2f}")
    return res


old = json.load(open(OUT / "stats.json"))["cortex"]

LAY = [m["label"] for m in old["ssaft"]["medians"]]
out = {"ssaft_cells": order_stats(scv, LAY, LAYER_RANK, "SSAF-T cells", "cell")}
out["ssaft_pixels_previous"] = {k: old["ssaft"][k] for k in
                                ("unit", "n", "medians", "observed_order", "order_matches_expected",
                                 "spearman_rank_depth", "spearman_blockmean", "block_perm_null_max", "columns")}
out["cellbin_previous"] = {k: old["cellbin"][k] for k in
                           ("unit", "n", "medians", "observed_order", "spearman_rank_depth", "columns")}
json.dump(out, open(OUT / "cortex_cells.json", "w"), indent=1)


scv.to_parquet(CACHE / "cortex_ssaft_cells.parquet")
NB = 20
d = scv[scv.label.isin(LAY)]
db = np.clip((d.depth * NB).astype(int), 0, NB - 1)
pd.crosstab(db, d.label).reindex(index=range(NB), columns=LAY, fill_value=0).to_csv(
    OUT / "cortex_ssaft_cells_depth_counts.csv")
json.dump(out["ssaft_cells"], open(CACHE / "stats_cortex_cells.json", "w"), indent=1, default=float)

print("\n================ pixel vs cell, SSAF-T ================")
pm = {m["label"]: m for m in old["ssaft"]["medians"]}
cm_ = {m["label"]: m for m in out["ssaft_cells"]["medians"]}
print(f"{'layer':12s} {'px n':>7s} {'px med':>7s} {'px IQR':>13s} | {'cell n':>6s} {'cell med':>8s} {'cell IQR':>13s}")
for l in LAY:
    p, c = pm.get(l), cm_.get(l)
    sh = LAYER_SHORT.get(l, l)
    if not c: print(f"{sh:12s} {p['n']:7d} {p['median']:7.3f} {p['q25']:6.3f}-{p['q75']:6.3f} |  absent"); continue
    print(f"{sh:12s} {p['n']:7d} {p['median']:7.3f} {p['q25']:6.3f}-{p['q75']:6.3f} | "
          f"{c['n']:6d} {c['median']:8.3f} {c['q25']:6.3f}-{c['q75']:6.3f}")
print(f"\nrho   pixels {old['ssaft']['spearman_rank_depth']:.3f}   cells {out['ssaft_cells']['spearman_rank_depth']:.3f}"
      f"   (CellBin cells {old['cellbin']['spearman_rank_depth']:.3f})")
print(f"tau   pixels {old['ssaft']['columns']['tau_median']:.2f} over {old['ssaft']['columns']['n']} columns"
      f"   cells {out['ssaft_cells']['columns']['tau_median']:.2f} over {out['ssaft_cells']['columns']['n']}"
      f"   (CellBin {old['cellbin']['columns']['tau_median']:.2f} over {old['cellbin']['columns']['n']})")
print("-> cortex_cells.json")
