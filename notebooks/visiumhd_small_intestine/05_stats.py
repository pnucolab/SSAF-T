import json, itertools
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad
from scipy.stats import spearmanr, rankdata, kendalltau
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
from common import *

rng = np.random.default_rng(0)
N_PERM = 1000
S = json.load(open(WORK / "sparsity.json"))
kg = set(json.load(open(WORK / "kg_markers.json"))["kg_markers"])
hk = lambda g: g.startswith(HOUSEKEEPING_PREFIX)
ax = np.load(WORK / "axes.npz"); DA, DB = ax["DA"], ax["DB"]
un = np.load(WORK / "units.npz"); units, valid = un["units"], un["valid"]
U = pd.read_csv(WORK / "units.csv")
sc = pd.read_parquet(WORK / "ssaft_cells_axis.parquet")
ec = pd.read_parquet(WORK / "enact_cells_axis.parquet")
out = {"sparsity": S}


tree = cKDTree(np.c_[ec.r, ec.c][ec.label.values == "Paneth cells"])
DA0 = ax["DA0"]
sc["A0"] = DA0[np.clip(sc.r, 0, DA0.shape[0] - 1), np.clip(sc.c, 0, DA0.shape[1] - 1)]
sc["C"] = tree.query(np.c_[sc.r, sc.c])[0] * PX
ec["C"] = tree.query(np.c_[ec.r, ec.c])[0] * PX


m = np.load(WORK / "bins_meta.npz"); Xs = sp.load_npz(WORK / "bins_sel.npz").tocsc()
ri, ci, tot = m["ri"], m["ci"], m["tot"]; genes = list(m["genes"])
bu = units[ri, ci]; bA = np.where(bu > 0, DA[ri, ci], np.nan); bB = DB[ri, ci]
def gs(lst):
    idx = [genes.index(g) for g in lst if g in genes]
    return np.asarray(Xs[:, idx].sum(1)).ravel()
SETS = {
    "smooth muscle (axis anchor)": MUSCLE,
    "Paneth, non-KG (Defa21/26/28/35, Ang4, Itln1, Mptx2)": PANETH_HELDOUT,
    "stem, KG (Lgr5, Olfm4, Ascl2, Smoc2)": STEM,
    "stem, non-KG (Rgmb, Axin2, Slc12a2)": STEM_HELDOUT,
    "proliferation, non-KG (Top2a, Cenpf, Birc5, Cdk1, Ccnb2, Stmn1)": ['Top2a', 'Cenpf', 'Birc5', 'Cdk1', 'Ccnb2', 'Stmn1'],
    "Moor villus-bottom landmarks, non-KG": [g for g in MOOR_BOTTOM if g not in kg and not hk(g)],
    "Moor mid-villus transporters, non-KG (Slc2a2, Slc15a1, Npc1l1)": ['Slc2a2', 'Slc15a1', 'Npc1l1'],
    "Moor villus-top landmarks, non-KG": [g for g in MOOR_TOP if g not in kg and not hk(g)],
    "Moor tip program, non-KG (Enpp3, Egfr, Klf4)": ['Enpp3', 'Egfr', 'Klf4'],
}
EXP_SET_ORDER = list(range(len(SETS)))
edges = np.arange(0, 402, 6.0); mid = (edges[:-1] + edges[1:]) / 2
ok = np.isfinite(bA)
bin_i = np.digitize(bA, edges) - 1
tot_b = np.bincount(bin_i[ok & (bin_i < len(mid))], weights=tot[ok & (bin_i < len(mid))], minlength=len(mid))
prof = {}
for name, lst in SETS.items():
    v = gs(lst); sel = ok & (bin_i < len(mid))
    f = np.bincount(bin_i[sel], weights=v[sel], minlength=len(mid)) / np.maximum(tot_b, 1)
    prof[name] = f
P = pd.DataFrame(prof, index=mid); P.index.name = "height_um"
P.to_csv(OUT / "raw_gradients_axisA.csv")
Pn = P / P.max()
sm = Pn.apply(lambda c: pd.Series(ndi.gaussian_filter1d(c.values, 1.5), index=c.index))
peak = sm.idxmax(); half = {}
for k in sm:
    w = sm[k].values; half[k] = float((w * mid).sum() / w.sum())
rho_peak = spearmanr(EXP_SET_ORDER, [peak[k] for k in SETS]).statistic
out["raw_gradients"] = dict(genes={k: [g for g in v if g in genes] for k, v in SETS.items()},
                            peak_um={k: float(peak[k]) for k in SETS},
                            centre_of_mass_um=half,
                            spearman_expected_vs_peak=float(rho_peak),
                            n_bins=int(ok.sum()))
print("\n[raw landmark gradients along axis A]")
for k in SETS: print(f"  {k:70s} peak {peak[k]:6.1f}  centre {half[k]:6.1f}")
print("  Spearman(expected order, peak) =", round(rho_peak, 3))


top_ho, bot_ho = gs(SETS["Moor villus-top landmarks, non-KG"]), gs(SETS["Moor villus-bottom landmarks, non-KG"])
top_all, bot_all = gs(MOOR_TOP), gs(MOOR_BOTTOM)
k8 = (ri // 4) * 10000 + (ci // 4)
df8 = pd.DataFrame(dict(k=k8, t=top_ho, b=bot_ho, ta=top_all, ba=bot_all, A=bA, B=bB, u=bu,
                        tot=tot)).groupby("k").agg(dict(t="sum", b="sum", ta="sum", ba="sum",
                                                        A="mean", B="mean", u="max", tot="sum"))
df8 = df8[(df8.u > 0) & np.isfinite(df8.A) & (df8.t + df8.b >= 5)]
df8["x"] = df8.t / (df8.t + df8.b); df8["xa"] = df8.ta / (df8.ta + df8.ba)
vil8 = df8[df8.A > 60]
out["moor_x_vs_axes"] = dict(
    n_8um_bins=int(len(df8)),
    spearman_x_A_heldout=float(spearmanr(df8.x, df8.A).statistic),
    spearman_x_A_allgenes=float(spearmanr(df8.xa, df8.A).statistic),
    spearman_x_B_heldout=float(spearmanr(df8.x, df8.B).statistic),
    spearman_x_A_heldout_villus_only=float(spearmanr(vil8.x, vil8.A).statistic),
)
per_u = df8.groupby("u").apply(lambda d: spearmanr(d.x, d.A).statistic if len(d) >= 20 else np.nan).dropna()
out["moor_x_vs_axes"]["per_unit_rho_median"] = float(per_u.median())
out["moor_x_vs_axes"]["per_unit_frac_positive"] = float((per_u > 0).mean())
out["moor_x_vs_axes"]["n_units"] = int(len(per_u))
print("\n[Moor landmark coordinate vs axes]", json.dumps(out["moor_x_vs_axes"], indent=1))


epi = sc[(sc.unit > 0) & sc.label.isin(list(EXPECTED_FULL) + ["absorptive enterocyte", "distal enterocyte",
                                                              "goblet cell"])]
out["axis_agreement"] = dict(
    note="over SSAF-T epithelial cell positions (positions only; labels unused)",
    spearman_A_B=float(spearmanr(epi.A, epi.B, nan_policy="omit").statistic),
    spearman_A_C=float(spearmanr(epi.A, epi.C, nan_policy="omit").statistic),
    spearman_B_C=float(spearmanr(epi.B, epi.C, nan_policy="omit").statistic),
    spearman_A0naive_B=float(spearmanr(epi.A0, epi.B, nan_policy="omit").statistic),
    spearman_A_B_rawbins=float(spearmanr(df8.A, df8.B).statistic))
print("\n[axis agreement]", out["axis_agreement"])


def shuffle_within(groups, arr):
    o = np.lexsort((rng.random(len(groups)), groups))
    base = np.argsort(groups, kind="stable")
    res = arr.copy(); res[base] = arr[o]; return res

def order_test(df, expected, axis, name, block_um=(100, 50), per_unit=True, min_unit=20):
    d = df[df.label.isin(expected) & np.isfinite(df[axis]) & (df.unit > 0)].copy()
    lev = d.label.map(expected).values.astype(float)
    ra = rankdata(d[axis].values)
    rl = rankdata(lev)
    rho = float(np.corrcoef(ra, rl)[0, 1])
    res = dict(n=int(len(d)), spearman=rho)
    med = d.groupby("label")[axis].agg(["size", "median", lambda s: s.quantile(.25), lambda s: s.quantile(.75)])
    med.columns = ["n", "median", "q25", "q75"]; med["expected_rank"] = [expected[l] for l in med.index]
    med = med.sort_values("median")
    res["per_label"] = med.round(2).reset_index().to_dict(orient="records")

    for B in block_um:
        blk = (d.r.values * PX // B).astype(np.int64) * 100000 + (d.c.values * PX // B).astype(np.int64)
        nul = np.array([np.corrcoef(ra, shuffle_within(blk, rl))[0, 1] for _ in range(N_PERM)])
        res[f"null_block{B}um"] = dict(mean=float(nul.mean()), p99=float(np.percentile(nul, 99)),
                                       max=float(nul.max()), p=float(((nul >= rho).sum() + 1) / (N_PERM + 1)))

    lv = sorted(set(expected.values())); labs = sorted(expected)
    group_med = d.groupby("label")[axis].median()
    obs_tau = kendalltau([expected[l] for l in labs], [group_med[l] for l in labs]).statistic
    taus = []
    for p in itertools.permutations(lv):
        mp = dict(zip(lv, p))
        taus.append(kendalltau([mp[expected[l]] for l in labs], [group_med[l] for l in labs]).statistic)
    taus = np.array(taus)
    res["kendall_tau_medians"] = float(obs_tau)
    res["exact_level_permutation"] = dict(n_perm=int(len(taus)), rank_of_observed=int((taus >= obs_tau - 1e-12).sum()),
                                          p=float((taus >= obs_tau - 1e-12).mean()))

    if per_unit:
        rows = []
        for u, g in d.groupby("unit"):
            if len(g) < min_unit or g.label.map(expected).nunique() < 3 and len(set(expected.values())) > 2: continue
            if g.label.map(expected).nunique() < 2: continue
            r_ = spearmanr(g[axis], g.label.map(expected)).statistic
            rows.append(dict(unit=u, n=len(g), rho=r_, n_levels=int(g.label.map(expected).nunique())))
        pu = pd.DataFrame(rows)


        perms = list(itertools.permutations(lv))
        best, top5 = [], []
        for u in pu.unit:
            g = d[d.unit == u]
            ra_u = rankdata(g[axis].values)
            rr = []
            for p in perms:
                mp = dict(zip(lv, p)); x = g.label.map(lambda l: mp[expected[l]]).values
                rr.append(np.corrcoef(ra_u, rankdata(x))[0, 1] if len(set(x)) > 1 else np.nan)
            rr = np.array(rr); obs_u = rr[perms.index(tuple(lv))]
            best.append(bool(np.nanmax(rr) - obs_u < 1e-12))
            top5.append(float((rr >= obs_u - 1e-12).mean()))
        pu["textbook_is_best"] = best; pu["textbook_rank_frac"] = top5
        res["per_unit"] = dict(n_units=int(len(pu)), median_rho=float(pu.rho.median()),
                               frac_rho_positive=float((pu.rho > 0).mean()),
                               frac_rho_gt_0p3=float((pu.rho > 0.3).mean()),
                               n_level_perms=len(perms),
                               frac_units_textbook_best=float(np.mean(best)),
                               frac_units_textbook_top5pct=float((np.array(top5) <= 0.05).mean()),
                               chance_textbook_best=1.0 / len(perms))
        pu.to_csv(OUT / f"per_unit_{name}.csv", index=False)
    print(f"\n[{name}] n={res['n']} rho={rho:.3f}  tau(medians)={obs_tau:.3f}  "
          f"exact p={res['exact_level_permutation']['p']:.4f}")
    print(med.to_string(float_format=lambda v: f"{v:7.1f}"))
    for k, v in res.items():
        if k.startswith("null") or k == "per_unit": print("  ", k, v)
    return res

out["order"] = {}
out["order"]["ssaft_cells_axisA"] = order_test(sc, EXPECTED, "A", "ssaft_A")
out["order"]["ssaft_cells_axisB"] = order_test(sc, EXPECTED, "B", "ssaft_B", per_unit=False)
out["order"]["ssaft_cells_axisC_enactPaneth"] = order_test(sc, EXPECTED, "C", "ssaft_C", per_unit=False)
ENT = {"immature enterocyte": 0, "villus-bottom enterocyte": 1, "villus-tip enterocyte": 2}
out["order"]["ssaft_enterocyte_triad_axisA"] = order_test(sc, ENT, "A", "ssaft_triad_A")
out["order"]["enact_cells_axisA"] = order_test(ec, ENACT_EXPECTED, "A", "enact_A", min_unit=20)
out["order"]["enact_cells_axisC"] = order_test(ec, ENACT_EXPECTED, "C", "enact_C", per_unit=False)

pix = pd.read_parquet(WORK / "map_pixels_axis.parquet")
pix = pix.iloc[rng.choice(len(pix), 400_000, replace=False)]
out["order"]["ssaft_map_pixels_axisA"] = order_test(pix, EXPECTED, "A", "ssaft_map_A", per_unit=False)

allpos = lambda d, a: d[(d.unit > 0) & np.isfinite(d[a])].groupby("label")[a].agg(
    n="size", median="median", q25=lambda s: s.quantile(.25), q75=lambda s: s.quantile(.75)).sort_values("median")
allpos(sc, "A").to_csv(OUT / "ssaft_all_labels_axisA.csv")
allpos(ec, "A").to_csv(OUT / "enact_all_labels_axisA.csv")
print(allpos(ec, "A").to_string())


s = ssaft_cells()
s.raw = None
sg = pd.Index(s.var_names.astype(str))
def cs(lst):
    idx = sg.get_indexer([g for g in lst if g in sg]); return np.asarray(s.X[:, idx].sum(1)).ravel()
t_ho, b_ho = cs(SETS["Moor villus-top landmarks, non-KG"]), cs(SETS["Moor villus-bottom landmarks, non-KG"])
t_all, b_all = cs(MOOR_TOP), cs(MOOR_BOTTOM)
sc["x_ho"] = np.where(t_ho + b_ho >= 3, t_ho / np.maximum(t_ho + b_ho, 1), np.nan)
sc["x_all"] = np.where(t_all + b_all >= 3, t_all / np.maximum(t_all + b_all, 1), np.nan)
ENTALL = ["immature enterocyte", "villus-bottom enterocyte", "distal enterocyte",
          "absorptive enterocyte", "villus-tip enterocyte"]
hx = {}
for col in ("x_ho", "x_all"):
    d = sc[sc.label.isin(ENTALL) & np.isfinite(sc[col])]
    hx[col] = d.groupby("label")[col].agg(n="size", median="median", mean="mean").reindex(ENTALL).round(4).reset_index().to_dict(orient="records")

def cond_test(a, b, col="x_ho"):
    d = sc[sc.label.isin([a, b]) & np.isfinite(sc[col]) & np.isfinite(sc.A) & (sc.unit > 0)].copy()
    d["dec"] = pd.qcut(d.A, 10, labels=False, duplicates="drop")
    y = d[col].values; isb = (d.label == b).values; st = d.dec.values
    def stat(lbl):
        diffs, w = [], []
        for k in np.unique(st):
            mm = st == k
            if lbl[mm].sum() < 5 or (~lbl[mm]).sum() < 5: continue
            diffs.append(y[mm][lbl[mm]].mean() - y[mm][~lbl[mm]].mean()); w.append(mm.sum())
        return np.average(diffs, weights=w)
    obs = stat(isb)
    nul = np.array([stat(shuffle_within(st, isb)) for _ in range(N_PERM)])
    return dict(n=int(len(d)), diff_same_position=float(obs), null_mean=float(nul.mean()),
                null_p99=float(np.percentile(nul, 99)), p=float(((nul >= obs).sum() + 1) / (N_PERM + 1)))
out["heldout_landmarks_on_ssaft"] = dict(
    genes_top_heldout=[g for g in SETS["Moor villus-top landmarks, non-KG"] if g in sg],
    genes_bottom_heldout=[g for g in SETS["Moor villus-bottom landmarks, non-KG"] if g in sg],
    per_label=hx,
    tip_vs_bottom_same_position=cond_test("villus-bottom enterocyte", "villus-tip enterocyte"),
    bottom_vs_immature_same_position=cond_test("immature enterocyte", "villus-bottom enterocyte"),
    tip_vs_bottom_same_position_allgenes=cond_test("villus-bottom enterocyte", "villus-tip enterocyte", "x_all"))
print("\n[held-out landmarks on SSAF-T]", json.dumps(out["heldout_landmarks_on_ssaft"], indent=1)[:3000])

rows = []
for name in ["Paneth, non-KG (Defa21/26/28/35, Ang4, Itln1, Mptx2)",
             "proliferation, non-KG (Top2a, Cenpf, Birc5, Cdk1, Ccnb2, Stmn1)",
             "stem, non-KG (Rgmb, Axin2, Slc12a2)"]:
    v = cs(SETS[name]); tt = np.asarray(s.X.sum(1)).ravel()
    f = pd.Series(v / np.maximum(tt, 1) * 1e3, index=sc.index)
    for L in list(EXPECTED) + ["absorptive enterocyte", "distal enterocyte", "goblet cell"]:
        mm = (sc.label == L).values
        rows.append(dict(set=name, label=L, per_1000_umi=float(f[mm].mean()), n=int(mm.sum())))
out["heldout_crypt_genes_by_label"] = rows
print(pd.DataFrame(rows).pivot(index="label", columns="set", values="per_1000_umi").round(2).to_string())

sc.to_parquet(WORK / "ssaft_cells_axis_scored.parquet")
ec.to_parquet(WORK / "enact_cells_axis_C.parquet")
json.dump(out, open(OUT / "stats.json", "w"), indent=1, default=float)
print("-> stats.json")
