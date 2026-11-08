#!/usr/bin/env python
import json, h5py, yaml
import numpy as np, pandas as pd
from scipy.stats import spearmanr, kendalltau, chi2_contingency
from sklearn.metrics import normalized_mutual_info_score
from scipy import ndimage as ndi
from common import *

rng = np.random.default_rng(0)
G = np.load(CACHE / "geometry.npz")
depth, cortex, col, thick = G["depth"], G["cortex"], G["col"], G["thick"]
cont, seg, ang = G["cont"], G["seg"], G["ang"]
NB = 20; edges = np.linspace(0, 1, NB + 1); mids = (edges[:-1] + edges[1:]) / 2
WIN = 250.0


cw = np.full(cortex.shape, -1); cw[cortex] = (col[cortex] // WIN).astype(int)
wins = []
for w in np.unique(cw[cw >= 0]):
    m = cw == w
    t = thick[m]
    s0, s1 = w * WIN, (w + 1) * WIN
    sel = (seg >= s0) & (seg < s1)
    curv = float(abs(ang[sel][-1] - ang[sel][0]) / max(np.ptp(seg[sel]), 1) * 1000) if sel.sum() > 2 else np.nan
    wins.append(dict(win=int(w), arc0=s0, n_px=int(m.sum()), thick_med=float(np.median(t)),
                     thick_cv=float(t.std() / t.mean()), curv_rad_per_mm=curv,
                     cx=float(np.argwhere(m)[:, 0].mean()), cy=float(np.argwhere(m)[:, 1].mean())))
wins = pd.DataFrame(wins)
wins["frame_frac"] = [float(G["onframe"][cw == w].mean()) for w in wins.win]


EDG = np.linspace(0.05, 1.0, 20)
cov = []
for w in wins.win:
    d = depth[(cw == w) & cortex]
    d = d[np.isfinite(d)]
    cov.append(float((np.histogram(d, bins=EDG)[0] >= 20).mean()) if len(d) >= 50 else 0.0)
wins["depth_cover"] = cov
wins["valid_geom"] = ((wins.thick_cv < 0.15) & wins.thick_med.between(700, 1500) & (wins.n_px > 2000)
                      & (wins.n_px < 3 * wins.n_px.median()) & (wins.frame_frac < 0.05))


_lb, _n = ndi.label(cortex)
_main = int(np.argmax(ndi.sum(cortex, _lb, range(1, _n + 1))) + 1)
wins["on_main"] = [float((_lb[(cw == w) & cortex] == _main).mean()) > .5 for w in wins.win]
wins["valid"] = wins.valid_geom & (wins.depth_cover >= 0.9) & wins.on_main
print(wins.to_string())
valid = np.isin(cw, wins.win[wins.valid].values)
print("valid units", int(wins.valid.sum()), "of", len(wins),
      f"(thickness alone kept {int(wins.valid_geom.sum())}; the traverse test drops "
      f"{sorted(wins.win[wins.valid_geom & (wins.depth_cover < .9)].tolist())}, "
      f"off-ribbon drops {sorted(wins.win[wins.valid_geom & (wins.depth_cover >= .9) & ~wins.on_main].tolist())})",
      "area mm2", valid.sum() * PX * PX / 1e6)


b = load_bin1()
ix = np.clip((b["x"] // PX).astype(np.int32), 0, NX - 1); iy = np.clip((b["y"] // PX).astype(np.int32), 0, NY - 1)
inv = valid[ix, iy]
dd = np.clip((depth[ix[inv], iy[inv]] * NB).astype(int), 0, NB - 1)
gg = b["gene"][inv].astype(np.int64); uu = b["umi"][inv].astype(np.float64)
ng = len(b["genes"])
prof = np.bincount(gg * NB + dd, weights=uu, minlength=ng * NB).reshape(ng, NB)
tot_d = prof.sum(0)
area_d = np.bincount(np.clip((depth[valid] * NB).astype(int), 0, NB - 1), minlength=NB) * PX * PX
print("valid-cortex UMIs", tot_d.sum(), " UMIs per depth bin", tot_d.astype(int))
enr = (prof / np.maximum(prof.sum(1, keepdims=True), 1)) / (tot_d / tot_d.sum())

mk = yaml.safe_load(open(C.SDATA.parent / "celltype_markers.yaml"))
used = set()
def walk(node):
    global used
    for k, v in node.items():
        if not isinstance(v, dict): continue
        for t in ("defining", "required", "supporting", "negative"): used |= set(v.get(t, []) or [])
        if "children" in v: walk(v["children"])
walk(mk)
import zarr
sgenes = set(signal_genes())
CANON = {"Cux2": "L2/3", "Lamp5": "L2/3", "Otof": "L2/3", "Rorb": "L4", "Fezf2": "L5",
         "Bcl11b": "L5", "Foxp2": "L6", "Syt6": "L6", "Ctgf": "L6b", "Nxph4": "L6b"}
canon = {}
for gname, lay in CANON.items():
    i = b["gidx"][gname]
    p = enr[i]
    canon[gname] = dict(layer=lay, umi=int(prof[i].sum()), peak_depth=float(mids[np.argmax(np.convolve(p, np.ones(3) / 3, "same"))]),
                        mean_depth=float((prof[i] * mids).sum() / prof[i].sum()),
                        annotation_marker=gname in used, in_ssaft_genes=gname in sgenes)
print(pd.DataFrame(canon).T.to_string())


cand = [i for i, gname in enumerate(b["genes"]) if gname not in sgenes and gname not in used
        and prof[i].sum() >= 150 and not gname.startswith(("mt-", "Rpl", "Rps", "Gm"))]
rows = []
for i in cand:
    obs = prof[i]; exp = tot_d / tot_d.sum() * obs.sum()
    chi = ((obs - exp) ** 2 / exp).sum()
    sm = np.convolve(enr[i], np.ones(3) / 3, "same")
    rows.append(dict(gene=b["genes"][i], umi=int(obs.sum()), chi2=chi, cramer=np.sqrt(chi / obs.sum() / (NB - 1)),
                     peak_depth=float(mids[np.argmax(sm)]), peak_enr=float(sm.max()),
                     mean_depth=float((obs * mids).sum() / obs.sum())))
ho = pd.DataFrame(rows)
from scipy.stats import chi2 as C2
ho["p"] = C2.sf(ho.chi2, NB - 1); ho["q"] = np.minimum(1, ho.p * len(ho))
ho = ho.sort_values("cramer", ascending=False)
laminar = ho[(ho.q < 1e-3) & (ho.peak_enr >= 2.0)]
print(f"strict held-out candidates {len(ho)}, laminar (Bonferroni q<1e-3, peak >=2x) {len(laminar)}")
print(laminar.head(40).to_string())
ho.to_csv(OUT / "cortex_heldout_genes.csv", index=False)


cm, names = load_map()
lm = map_at(cm)
del cm
nm = np.array(["none"] + names)
lab = np.where(lm > 0, lm, 0)
vx, vy = np.nonzero(valid)
L = nm[lab[vx, vy]]; D = depth[vx, vy]; Wc = cw[vx, vy]
blk = (vx // 10) * 10000 + (vy // 10)
px = pd.DataFrame(dict(label=L, depth=D, win=Wc, blk=blk, ix=vx, iy=vy))
px.to_parquet(CACHE / "cortex_pixels.parquet")
comp = px.label.value_counts(normalize=True)
print("\nlabel composition of valid cortex pixels (top 20)\n", comp.head(20).round(4).to_string())


cb = pd.read_parquet(CACHE / "cellbin_cells.parquet")
cix = np.clip((cb.x_um // PX).astype(int), 0, NX - 1); ciy = np.clip((cb.y_um // PX).astype(int), 0, NY - 1)
cb["valid"] = valid[cix, ciy]; cb["depth"] = depth[cix, ciy]; cb["win"] = cw[cix, ciy]
cb["blk"] = (cix // 10) * 10000 + (ciy // 10)
cbv = cb[cb.valid].copy()
print("\nCellBin cells in valid cortex", len(cbv), "\n", cbv.annotation.value_counts().head(15).to_string())
cbv.to_parquet(CACHE / "cortex_cellbin.parquet")

with h5py.File(C.MOSTA / "Mouse_brain.h5ad", "r") as h:
    nm50 = h["obs/cell_name"][:].astype(str)
    a50 = h["obs/__categories/annotation"][:].astype(str)[h["obs/annotation"][:]]
ij = np.array([list(map(int, s.split("_"))) for s in nm50])
b50 = pd.DataFrame(dict(annotation=a50, x_um=0.5 * (ij[:, 0] * 50 + 25) - X0, y_um=0.5 * (ij[:, 1] * 50 + 25) - Y0))
bix = np.clip((b50.x_um // PX).astype(int), 0, NX - 1); biy = np.clip((b50.y_um // PX).astype(int), 0, NY - 1)
b50["valid"] = valid[bix, biy]; b50["depth"] = depth[bix, biy]; b50["win"] = cw[bix, biy]
b50v = b50[b50.valid]
b50.to_parquet(CACHE / "bin50_anatomy.parquet")
print("\nbin50 in valid cortex\n", b50v.annotation.value_counts().to_string())


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
    for a, c in zip(lab_sorted[:-1], lab_sorted[1:]):
        dlt = np.array(boots[c]) - np.array(boots[a])
        pairs.append(dict(shallow=a, deep=c, diff=float(med.loc[c, "median"] - med.loc[a, "median"]),
                          ci_lo=float(np.nanpercentile(dlt, 2.5)), ci_hi=float(np.nanpercentile(dlt, 97.5)),
                          frac_boot_wrong=float(np.nanmean(dlt <= 0))))

    per = []
    for w, g in df.groupby("win"):
        m = g.groupby("label").depth.median()
        m = m[[l for l in m.index if (g.label == l).sum() >= 5]]
        if len(m) >= 3:
            tau = kendalltau([rank[l] for l in m.index], m.values).statistic
            per.append(dict(win=int(w), n_types=len(m), tau=float(tau),
                            perfect=bool(all(np.diff(m.loc[sorted(m.index, key=lambda l: rank[l])].values) > 0)
                                         if len(set(rank[l] for l in m.index)) == len(m) else np.nan)))
    per = pd.DataFrame(per)
    order_obs = list(med.sort_values("median").index)
    exp_ranks = [rank[l] for l in order_obs]
    res = dict(unit=unit, n=int(len(df)), medians=med.round(4).reset_index().to_dict("records"),
               observed_order=order_obs, order_matches_expected=bool(all(np.diff(exp_ranks) >= 0)),
               spearman_rank_depth=float(rho), spearman_blockmean=float(rho_b),
               block_perm_null_max=float(np.abs(null).max()), block_perm_p=float((np.abs(null) >= abs(rho_b)).mean() + 1 / (nperm + 1)),
               adjacent_pairs=pairs,
               columns=dict(n=int(len(per)), tau_median=float(per.tau.median()) if len(per) else None,
                            frac_tau_pos=float((per.tau > 0).mean()) if len(per) else None,
                            frac_tau_ge_0p8=float((per.tau >= 0.8).mean()) if len(per) else None,
                            per=per.to_dict("records")))
    print(f"\n[{tag}] n={len(df)} order={[LAYER_SHORT.get(l, l) for l in order_obs]} rho={rho:.3f} "
          f"rho_block={rho_b:.3f} null|max|={np.abs(null).max():.3f}")
    print(med.round(3).to_string()); print(pd.DataFrame(pairs).round(3).to_string())
    if len(per): print(f"  columns: n={len(per)} median tau {per.tau.median():.2f}  tau>0 {(per.tau>0).mean():.2f}")
    return res

stats = {}
lay_present = [l for l in LAYERS if (px.label == l).sum() >= 500]
stats["ssaft"] = order_stats(px, lay_present, LAYER_RANK, "SSAF-T map", "5um pixel")
cbv2 = cbv.drop(columns="label").rename(columns={"annotation": "label"})
stats["cellbin"] = order_stats(cbv2, PUB_LAYERS, PUB_RANK, "CellBin published", "cell")
B50L = ["Cortex_L1", "Cortex-L2/3", "Cortex_L4", "Cortex_L5", "Cortex_L6"]
b50r = {l: i for i, l in enumerate(B50L, 1)}
b50v2 = b50v.rename(columns={"annotation": "label"}).copy()
b50v2["blk"] = (np.clip((b50v2.x_um // PX).astype(int), 0, NX - 1) // 10) * 10000 + (np.clip((b50v2.y_um // PX).astype(int), 0, NY - 1) // 10)
stats["bin50"] = order_stats(b50v2, B50L, b50r, "published bin50 anatomy", "bin50 (25um)", nboot=300)


def comp_by_depth(df, labels):
    d = df[df.label.isin(labels)]
    db = np.clip((d.depth * NB).astype(int), 0, NB - 1)
    t = pd.crosstab(db, d.label).reindex(index=range(NB), columns=labels, fill_value=0)
    return t
cs = comp_by_depth(px, lay_present); cc = comp_by_depth(cbv2, PUB_LAYERS); c5 = comp_by_depth(b50v2, B50L)
allpx = pd.crosstab(np.clip((px.depth * NB).astype(int), 0, NB - 1), px.label)
cs.to_csv(OUT / "cortex_ssaft_depth_counts.csv"); cc.to_csv(OUT / "cortex_cellbin_depth_counts.csv")
c5.to_csv(OUT / "cortex_bin50_depth_counts.csv"); allpx.to_csv(CACHE / "cortex_allpx_depth_counts.csv")


for k, df, labs in (("ssaft", px, lay_present), ("cellbin", cbv2, PUB_LAYERS)):
    d = df[df.label.isin(labs)]
    stats[k]["nmi_depthbin_label"] = float(normalized_mutual_info_score(np.clip((d.depth * NB).astype(int), 0, NB - 1), d.label))

coll = {"L2/3 IT glutamatergic neuron": "EX L2/3", "L4 IT glutamatergic neuron": "EX L4",
        "L5 ET glutamatergic neuron": "EX L5/6", "L6 IT glutamatergic neuron": "EX L6",
        "L6 CT glutamatergic neuron": "EX L6", "L6b glutamatergic neuron": "EX L6"}
d = px[px.label.isin(lay_present)]
stats["ssaft"]["nmi_collapsed_to_published_vocab"] = float(normalized_mutual_info_score(
    np.clip((d.depth * NB).astype(int), 0, NB - 1), d.label.map(coll)))


def band(gname, frac=0.5):
    p = np.convolve(enr[b["gidx"][gname]], np.ones(3) / 3, "same")
    m = p >= p.max() * frac

    k = int(np.argmax(p)); lo = k; hi = k
    while lo > 0 and m[lo - 1]: lo -= 1
    while hi < NB - 1 and m[hi + 1]: hi += 1
    return float(edges[lo]), float(edges[hi + 1])
BANDS = {"L2/3": "Cux2", "L4": "Rorb", "L5": "Fezf2", "L6": "Foxp2", "L6b": "Ctgf"}
bands = {k: band(g) for k, g in BANDS.items()}
inband = {}
for lab_, key in (("L2/3 IT glutamatergic neuron", "L2/3"), ("L4 IT glutamatergic neuron", "L4"),
                  ("L5 ET glutamatergic neuron", "L5"), ("L6 CT glutamatergic neuron", "L6"),
                  ("L6b glutamatergic neuron", "L6b")):
    if lab_ in lay_present:
        v = px.depth[px.label == lab_]; lo, hi = bands[key]
        inband[lab_] = float(((v >= lo) & (v < hi)).mean())
for lab_, key in (("EX L2/3", "L2/3"), ("EX L4", "L4"), ("EX L5/6", "L5"), ("EX L6", "L6")):
    v = cbv2.depth[cbv2.label == lab_]; lo, hi = bands[key]
    inband[lab_] = float(((v >= lo) & (v < hi)).mean())
print("\nraw-gene half-max bands", bands); print("fraction inside own raw band", {k: round(v, 3) for k, v in inband.items()})


hog = laminar.gene.head(30).tolist()
lab_px = np.full((NX, NY), "", dtype=object)
ho_rows = []
if hog:
    sel = np.isin(b["gene"], [b["gidx"][g] for g in hog]) & inv
    gl = b["genes"][b["gene"][sel]]
    pl = nm[lab[ix[sel], iy[sel]]]
    t = pd.crosstab(gl, pl, values=b["umi"][sel], aggfunc="sum").fillna(0)
    t = t[[l for l in lay_present if l in t.columns]]
    area = px.label.value_counts()[t.columns]
    dens = t / area.values
    ref = t.sum(1) / area.sum()
    enrt = dens.div(ref, axis=0)
    for gname in hog:
        if gname not in enrt.index: continue
        r = enrt.loc[gname]
        best = r.idxmax()
        pk = float(laminar.set_index("gene").loc[gname, "peak_depth"])
        ho_rows.append(dict(gene=gname, peak_depth=pk, best_ssaft_type=LAYER_SHORT[best],
                            best_rank=LAYER_RANK[best],
                            weighted_rank=float((r * pd.Series(LAYER_RANK)[r.index]).sum() / r.sum()),
                            **{LAYER_SHORT[c]: float(r[c]) for c in r.index}))
hot = pd.DataFrame(ho_rows)
if len(hot):
    rr = spearmanr(hot.peak_depth, hot.weighted_rank).statistic
    null = [spearmanr(rng.permutation(hot.peak_depth.values), hot.weighted_rank).statistic for _ in range(5000)]
    stats["heldout_genes"] = dict(n_candidates=int(len(ho)), n_laminar=int(len(laminar)), tested=hog,
                                  spearman_peakdepth_vs_ssaft_weighted_rank=float(rr),
                                  perm_p=float((np.abs(null) >= abs(rr)).mean()),
                                  table=hot.round(3).to_dict("records"))
    print(hot.round(2).to_string()); print("held-out: spearman(peak depth, SSAF-T weighted rank)", rr)
    hot.to_csv(OUT / "cortex_heldout_in_ssaft_territories.csv", index=False)

stats["columns"] = wins.round(4).to_dict("records")
stats["canonical_genes"] = canon
stats["raw_gene_bands"] = bands
stats["fraction_in_own_raw_band"] = inband
stats["valid_area_mm2"] = float(valid.sum() * PX * PX / 1e6)
stats["label_composition_valid_cortex"] = comp.head(25).round(5).to_dict()
np.save(CACHE / "cortex_gene_enrichment.npy", enr); np.save(CACHE / "cortex_gene_profile.npy", prof)
json.dump(stats, open(CACHE / "stats_cortex.json", "w"), indent=1, default=float)
np.savez_compressed(CACHE / "cortex_valid.npz", valid=valid, cw=cw)
print("done")
