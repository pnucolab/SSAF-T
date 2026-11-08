#!/usr/bin/env python
import json, h5py
import numpy as np, pandas as pd, anndata as ad, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.stats import spearmanr, kendalltau
from common import *

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
rng = np.random.default_rng(0)
SS_C, CB_C, B50_C = "#2a5d9f", "#d9822b", "#7a7a7a"
pal = palette()
LAY = [m["label"] for m in json.load(open(OUT / "cortex_cells.json"))["ssaft_cells"]["medians"]]
LAYG = {"Cux2": "L2/3", "Rorb": "L4", "Fezf2": "L5", "Foxp2": "L6", "Ctgf": "L6b"}
NB = 20; mids = (np.arange(NB) + .5) / NB

G = np.load(CACHE / "geometry.npz"); depth, cortex, col = G["depth"], G["cortex"], G["col"]
cw = np.full(cortex.shape, -1, np.int32); cw[cortex] = (col[cortex] // 250.).astype(int)
_lb, _n = ndi.label(cortex)
_main = int(np.argmax(ndi.sum(cortex, _lb, range(1, _n + 1))) + 1)


def place(xy, lab):
    ix = np.clip((xy[:, 0] // PX).astype(int), 0, NX - 1)
    iy = np.clip((xy[:, 1] // PX).astype(int), 0, NY - 1)
    d = pd.DataFrame(dict(label=lab, depth=depth[ix, iy], win=cw[ix, iy], in_cortex=cortex[ix, iy],
                          blk=(ix // 10) * 10000 + (iy // 10)))
    return d[d.in_cortex & np.isfinite(d.depth)].copy()


a = ssaft_cells()
SS = place(np.asarray(a.obsm["spatial"][:], float), a.obs["cellman_annotation"].astype(str).to_numpy())
a.file.close()
cbr = pd.read_parquet(CACHE / "cellbin_cells.parquet")
CB = place(cbr[["x_um", "y_um"]].to_numpy(float), cbr.annotation.astype(str).to_numpy())
B50 = pd.read_parquet(CACHE / "bin50_anatomy.parquet")
B50L = ["Cortex_L1", "Cortex-L2/3", "Cortex_L4", "Cortex_L5", "Cortex_L6"]
B50R = {l: i for i, l in enumerate(B50L, 1)}


def n_types(d, labels):
    dl = d[d.label.isin(labels)]
    return {int(w): int((g.groupby("label").size() >= 5).sum()) for w, g in dl.groupby("win")}
ns, nc = n_types(SS, LAY), n_types(CB, PUB_LAYERS)
allw = sorted(np.unique(cw[cw >= 0]))
on_main = {int(w): float((_lb[(cw == w) & cortex] == _main).mean()) > .5 for w in allw}

NEO = {"Cortex_L1", "Cortex-L2/3", "Cortex_L4", "Cortex_L5", "Cortex_L6"}
_b = pd.read_parquet(CACHE / "bin50_anatomy.parquet")
_ix = np.clip((_b.x_um // PX).astype(int), 0, NX - 1); _iy = np.clip((_b.y_um // PX).astype(int), 0, NY - 1)
_b = _b.assign(inctx=cortex[_ix, _iy], win=cw[_ix, _iy]); _b = _b[_b.inctx]
neo = _b.groupby("win").annotation.apply(lambda s: float(s.isin(NEO).mean()))
SET = sorted(w for w in allw if ns.get(w, 0) >= 3 and nc.get(w, 0) >= 3
             and on_main[w] and float(neo.get(w, 0)) >= .5)
BLUE = sorted(w for w in allw if on_main[w] and float(neo.get(w, 0)) >= .5)
print(f"blue set (>=50% Cortex_L*, main sheet): {len(BLUE)} columns {BLUE}")
print(f"  of those, the statistic is defined in {len(SET)}: {SET}")
print(f"  dropped for too few cells: {sorted(set(BLUE)-set(SET))}")
old19 = sorted(np.unique(cw[np.load(CACHE / "cortex_valid.npz")["valid"]]).tolist())
print(f"columns where the statistic is defined for both AND on the neocortex: {len(SET)}")
print(f"  {SET}\n  the original 19 were {old19}\n  added: {sorted(set(SET)-set(old19))}")
inset = np.isin(cw, SET) & cortex
SS, CB = SS[SS.win.isin(SET)], CB[CB.win.isin(SET)]
B50 = B50[B50.win.isin(SET)].rename(columns={"annotation": "label"})
print(f"cells: SSAF-T {len(SS):,}  CellBin {len(CB):,}  bin50 {len(B50):,}")


b = load_bin1()
gx = np.clip((b["x"] // PX).astype(np.int32), 0, NX - 1); gy = np.clip((b["y"] // PX).astype(np.int32), 0, NY - 1)
sel = inset[gx, gy]
dd = np.clip((depth[gx[sel], gy[sel]] * NB).astype(int), 0, NB - 1)
prof = np.bincount(b["gene"][sel].astype(np.int64) * NB + dd, weights=b["umi"][sel].astype(float),
                   minlength=len(b["genes"]) * NB).reshape(len(b["genes"]), NB)
tot_d = prof.sum(0)
enr = (prof / np.maximum(prof.sum(1, keepdims=True), 1)) / (tot_d / tot_d.sum())
peak = {g: float(mids[np.argmax(np.convolve(enr[b["gidx"][g]], np.ones(3) / 3, "same"))]) for g in LAYG}
print("gene peaks:", {k: round(v, 3) for k, v in peak.items()})


def order_stats(df, labels, rank, unit, nperm=1000):
    d = df[df.label.isin(labels)].copy(); d["rank"] = d.label.map(rank)
    med = d.groupby("label").depth.agg(["size", "median", lambda s: s.quantile(.25), lambda s: s.quantile(.75)])
    med.columns = ["n", "median", "q25", "q75"]; med = med.loc[[l for l in labels if l in med.index]]
    rho = spearmanr(d["rank"], d.depth).statistic
    if "blk" in d:
        bm = d.groupby("blk").depth.mean(); codes, ub = pd.factorize(d.blk.values)
        bmv = bm.reindex(ub).values
        rho_b = spearmanr(d["rank"], bmv[codes]).statistic
        null = np.array([spearmanr(d["rank"], rng.permutation(bmv)[codes]).statistic for _ in range(nperm)])
        nullmax = float(np.abs(null).max())
    else:
        rho_b, nullmax = float("nan"), float("nan")
    per = []
    for w, g in d.groupby("win"):
        m = g.groupby("label").depth.median()
        m = m[[l for l in m.index if (g.label == l).sum() >= 5]]
        if len(m) >= 3:
            per.append(dict(win=int(w), tau=float(kendalltau([rank[l] for l in m.index], m.values).statistic)))
    per = pd.DataFrame(per)
    obs = list(med.sort_values("median").index)
    return dict(unit=unit, n=int(len(d)), medians=med.round(4).reset_index().to_dict("records"),
                observed_order=obs, order_matches_expected=bool(all(np.diff([rank[l] for l in obs]) >= 0)),
                spearman_rank_depth=float(rho), spearman_blockmean=float(rho_b),
                block_perm_null_max=nullmax,
                columns=dict(n=int(len(per)), tau_median=float(per.tau.median()) if len(per) else None,
                             frac_tau_pos=float((per.tau > 0).mean()) if len(per) else None,
                             frac_tau_perfect=float((per.tau >= .999).mean()) if len(per) else None,
                             per=per.to_dict("records")))


S = {"ssaft": order_stats(SS, LAY, LAYER_RANK, "cell"),
     "cellbin": order_stats(CB, PUB_LAYERS, PUB_RANK, "cell"),
     "bin50": order_stats(B50, B50L, B50R, "bin50 (25um)")}
S["columns_used"] = SET; S["columns_added_vs_19"] = sorted(set(SET) - set(old19))
S["canonical_gene_peaks"] = peak
json.dump(S, open(OUT / "cortex_neocortex_arc.json", "w"), indent=1, default=float)
for k in ("ssaft", "cellbin", "bin50"):
    c = S[k]["columns"]
    print(f"  {k:8s} n={S[k]['n']:6d}  order ok {S[k]['order_matches_expected']}  rho {S[k]['spearman_rank_depth']:.3f}"
          f"  columns {c['n']:2d}  median tau {c['tau_median']:.2f}  tau>0 {c['frac_tau_pos']:.0%}  tau=1 {c['frac_tau_perfect']:.0%}")


with h5py.File(C.MOSTA / "Mouse_brain_cell_bin.h5ad", "r") as h:
    PUBC = dict(zip(h["obs/annotation/categories"][:].astype(str), h["uns/annotation_colors"][:].astype(str)))


_extra = OUT.parent / "sdata_ssaft.zarr/tables/ref_cells"
if _extra.exists():
    _ref = ad.read_zarr(_extra)
    PUBC.update(dict(zip(_ref.obs["annotation"].cat.categories,
                         [str(c) for c in _ref.uns["annotation_colors"]])))

fig, axs = plt.subplots(1, 3, figsize=(7.8, 2.8), gridspec_kw=dict(width_ratios=[1.4, 1.4, 1.25], wspace=0.5))
a_ = axs[0]
def counts(d, labels):
    x = d[d.label.isin(labels)]
    t = pd.crosstab(np.clip((x.depth * NB).astype(int), 0, NB - 1), x.label).reindex(range(NB), columns=labels, fill_value=0)
    return t.div(t.sum(0).replace(0, 1), axis=1)
cs, cc = counts(SS, LAY), counts(CB, PUB_LAYERS)
for l in LAY:
    a_.plot(cs[l].values, mids, color=pal[l], lw=1.6)
    a_.text(cs[l].values.max() + .005, mids[cs[l].values.argmax()], LAYER_SHORT[l], fontsize=6, va="center")
for l in PUB_LAYERS: a_.plot(cc[l].values, mids, color=PUBC[l], lw=1, ls="--")
a_.set_ylim(1, 0); a_.set_ylabel("normalised depth (pia 0, WM 1)"); a_.set_xlabel("fraction of that type")
a_.set_title("Depth profiles", fontsize=8)
a_.plot([], [], color="#555", lw=1.6, label="SSAF-T (cells)")
a_.plot([], [], color="#555", lw=1, ls="--", label="CellBin (cells)")
a_.legend(fontsize=5.5, frameon=False, loc="lower right")

a_ = axs[1]
rows = [("SSAF-T", LAYER_SHORT[m["label"]], m["median"], m["q25"], m["q75"], pal[m["label"]]) for m in S["ssaft"]["medians"]]
rows += [("CellBin", m["label"], m["median"], m["q25"], m["q75"], PUBC[m["label"]]) for m in S["cellbin"]["medians"]]
rows += [("bin50 anatomy", m["label"].replace("Cortex_", "").replace("Cortex-", ""), m["median"], m["q25"], m["q75"], B50_C)
         for m in S["bin50"]["medians"]]
for gi, gname in enumerate(["bin50 anatomy", "SSAF-T", "CellBin"]):
    for r in [r for r in rows if r[0] == gname]:
        a_.plot([gi, gi], [r[3], r[4]], color=r[5], lw=4, alpha=.35, solid_capstyle="butt")
        a_.plot(gi, r[2], "o", ms=5, color=r[5], mec="white", mew=.8)
        a_.text(gi + .14, r[2], r[1], fontsize=5.5, va="center")
for g in LAYG:
    a_.axhline(peak[g], color="#bbbbbb", lw=.6, ls=":", zorder=0)
    a_.text(2.75, peak[g], g, fontsize=5, color="#888", va="center")
a_.set_xticks(range(3)); a_.set_xticklabels(["bin50\nanatomy", "SSAF-T", "CellBin"], fontsize=6.5)
a_.set_xlim(-.4, 3.2); a_.set_ylim(1, 0)
a_.set_ylabel("normalised depth (median, IQR)"); a_.set_xlabel("annotation")
a_.set_title("Layer depth", fontsize=8)
a_.text(.02, .02, "dotted: raw-gene peaks", transform=a_.transAxes, fontsize=5.2, color="#888", va="bottom")

a_ = axs[2]
t1 = pd.DataFrame(S["ssaft"]["columns"]["per"]).tau
SS3 = {"L2/3 IT glutamatergic neuron": 1, "L4 IT glutamatergic neuron": 2,
       "L5 ET glutamatergic neuron": 3, "L6 CT glutamatergic neuron": 3, "L6b glutamatergic neuron": 3}
CB3 = {"EX L2/3": 1, "EX L4": 2, "EX L5/6": 3, "EX L6": 3}
def per3(d, mp):
    x = d[d.label.isin(mp)].copy(); x["lv"] = x.label.map(mp); o = {}
    for w, g in x.groupby("win"):
        m = g.groupby("lv").depth.median()
        if len(m) >= 3 and (g.groupby("lv").size() >= 5).all():
            o[w] = kendalltau(m.index.values, m.values).statistic
    return pd.Series(o)
e_ss, e_cb = per3(SS, SS3), per3(CB, CB3)
bins = np.linspace(-1, 1, 11); med = lambda v: r" ($\tilde{\tau}$ = %.2f)" % v
a_.hist(t1, bins=bins, color="#8bbbe8", rwidth=.9, label=f"SSAF-T, 5 layers ({len(t1)} columns)" + med(t1.median()))
a_.hist(e_ss, bins=bins, histtype="step", color="#1b3f8f", lw=.9, label="SSAF-T, 3 layers" + med(e_ss.median()))
a_.hist(e_cb, bins=bins, histtype="step", color=CB_C, lw=1.1, label="Chen et al., 3 layers" + med(e_cb.median()))
_b = e_ss.index.intersection(e_cb.index)
a_.text(.02, .55, f"5 layers: the textbook order is perfect\nin {int((t1 >= .999).sum())} of {len(t1)} columns "
        f"(1 in 120 by chance)\n3 layers, paired: the two agree in\n{100 * (e_ss[_b] == e_cb[_b]).mean():.0f}% of {len(_b)} columns",
        transform=a_.transAxes, fontsize=4.8, va="top", color="#444")
a_.set_xlabel("Kendall τ, observed vs expected order"); a_.set_ylabel("number of columns")
a_.legend(fontsize=6, frameon=False, loc="upper left"); a_.set_title("Order per column", fontsize=8)
for e in ("png", "pdf"):
    fig.savefig(OUT / f"panel_cortex_depth_stats_neocortex.{e}", dpi=600, bbox_inches="tight")
plt.close(fig)
print(f"\n3-layer: SSAF-T tau=1 {100*(e_ss>=.999).mean():.0f}% of {len(e_ss)}, "
      f"Chen {100*(e_cb>=.999).mean():.0f}% of {len(e_cb)}, agree {100*(e_ss[_b]==e_cb[_b]).mean():.0f}% of {len(_b)}")
print("-> panel_cortex_depth_stats_neocortex.pdf")
