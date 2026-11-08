#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common      # noqa: E402
import vocab       # noqa: E402

SEED, N_DRAWS, N_PERM, N_BALANCE = 0, 5, 10, 500
MIN_SIDE = 15
SEP_MARGIN, SEP_Z = 0.10, 3.0


MIN_SIDE_FRAC = 0.15


def _groupings(obs):
    out = []
    n = len(obs)
    def _ok(m):
        s = min(int(m.sum()), int((~m).sum()))
        return s >= 2 * MIN_SIDE and s >= MIN_SIDE_FRAC * n
    if "method" in obs and obs["method"].nunique() == 2:
        lv = sorted(obs["method"].unique())
        m = (obs["method"] == lv[0]).to_numpy()
        if _ok(m):
            out.append((m, f"method:{lv[0]}|{lv[1]}"))
    if "donor_id" in obs:
        vc = obs["donor_id"].value_counts()
        if len(vc) >= 2:
            side, tot = set(), 0
            for d, c in vc.items():
                if tot < n / 2:
                    side.add(d); tot += c
            m = obs["donor_id"].isin(side).to_numpy()
            if _ok(m):
                out.append((m, f"donor({len(side)}|{len(vc)-len(side)})"))
    return out


def _recalls(X, y, tr):
    out = []
    for a, b in ((tr, ~tr), (~tr, tr)):
        P = []
        for i in (0, 1):
            m = a & (y == i)
            if m.sum() < 2:
                return None
            P.append(X[m].mean(axis=0))
        P = np.asarray(P)
        Pn = P - P.mean(axis=1, keepdims=True)
        Pn /= (np.linalg.norm(Pn, axis=1, keepdims=True) + 1e-12)
        Xt = X[b]
        Xn = Xt - Xt.mean(axis=1, keepdims=True)
        Xn /= (np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-12)
        pred = np.argmax(Xn @ Pn.T, axis=1)
        yb = y[b]
        out.append(np.mean([float((pred[yb == i] == i).mean()) for i in (0, 1) if (yb == i).sum()]))
    return float(np.mean(out)) if out else None


def _replicates(X, is_part, grp, rng):
    pos, neg = np.flatnonzero(is_part), np.flatnonzero(~is_part)
    m = min(len(pos), len(neg), N_BALANCE)
    obs, null = [], []
    for _ in range(N_DRAWS):
        sel = np.concatenate([rng.choice(pos, m, replace=False), rng.choice(neg, m, replace=False)])
        Xb, tb = X[sel], grp[sel]
        yb = np.r_[np.ones(m, int), np.zeros(m, int)]
        if min((tb & (yb == 0)).sum(), (tb & (yb == 1)).sum(),
               (~tb & (yb == 0)).sum(), (~tb & (yb == 1)).sum()) < MIN_SIDE:
            continue
        v = _recalls(Xb, yb, tb)
        if v is None:
            continue
        obs.append(v)
        for _ in range(N_PERM):
            p = _recalls(Xb, rng.permutation(yb), tb)
            if p is not None:
                null.append(p)
    if not obs or not null:
        return None
    o, nu, sd = float(np.mean(obs)), float(np.mean(null)), float(np.std(null))
    z = (o - nu) / max(sd, 1e-9)
    return dict(recall=round(o, 4), null=round(nu, 4), z=round(z, 2),
                replicates=bool(o >= nu + SEP_MARGIN and z >= SEP_Z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("organ")
    ap.add_argument("--min-block", type=int, default=50)
    a = ap.parse_args()
    organ = a.organ

    ad = common.load(organ)
    cm = ad.obs["cm"].astype(str).to_numpy()
    rf = ad.obs["ref"].astype(str).to_numpy()
    rel = ad.obs["relation"].astype(str).to_numpy()
    n = len(cm)
    R = {"organ": organ, "n_cells": int(n)}


    cnt = pd.Series(rel).value_counts()
    frac = {k: float(cnt.get(k, 0)) / n for k in
            ("same", "cellman_deeper", "ref_deeper", "conflict", "unknown")}
    R["A_preserve"] = {k: round(100 * v, 2) for k, v in frac.items()}
    R["A_preserve"]["ancestor_consistent"] = round(100 * (frac["same"] + frac["cellman_deeper"]), 2)
    print(f"\n{'='*78}\n  {organ}   {n:,} cells\n{'='*78}")
    print("\n[A] PRESERVE — how CeLLMan's label relates to the author's")
    for k, lbl in (("same", "identical"), ("cellman_deeper", "CeLLMan refined (descendant)"),
                   ("ref_deeper", "CeLLMan coarser (honest floor)"),
                   ("conflict", "contradicts"), ("unknown", "declined")):
        print(f"    {lbl:34s} {100*frac[k]:6.2f}%   {int(cnt.get(k,0)):>8,}")
    print(f"    {'-'*34} {'-'*7}")
    print(f"    {'ANCESTOR-CONSISTENT (same+refined)':34s} {R['A_preserve']['ancestor_consistent']:6.2f}%")


    refl = {x for x in rf if x not in ("Unknown", "Background")}
    cml = {x for x in cm if x not in ("Unknown", "Background")}
    new = cml - refl


    new_deep = int(((np.isin(cm, list(new))) & (rel == "cellman_deeper")).sum()) if new else 0
    new_all = int(np.isin(cm, list(new)).sum()) if new else 0
    R["B_extend"] = dict(n_ref_labels=len(refl), n_cm_labels=len(cml), n_new_labels=len(new),
                         cells_with_new_label=new_all, of_which_refinement=new_deep)
    print("\n[B] EXTEND — the label vocabulary")
    print(f"    author labels used here            {len(refl):>6}")
    print(f"    CeLLMan labels used here           {len(cml):>6}")
    print(f"    absent from the author's vocabulary{len(new):>6}")
    print(f"    cells carrying such a label        {new_all:>6,}  "
          f"({100*new_deep/max(new_all,1):.1f}% of them are a strict refinement, the rest are conflicts)")
    if new:
        top = pd.Series(cm[np.isin(cm, list(new))]).value_counts().head(8)
        for lab, c in top.items():
            dp = int(((cm == lab) & (rel == "cellman_deeper")).sum())
            print(f"        {lab:46s} {c:>7,}  refinement {100*dp/c:5.1f}%")


    R["D_abstain"] = dict(coarser_pct=round(100 * frac["ref_deeper"], 2),
                          declined_pct=round(100 * frac["unknown"], 2),
                          declined_cells=int(cnt.get("unknown", 0)))
    print("\n[D] ABSTAIN — where CeLLMan stopped")
    print(f"    floored above the author (coarser) {100*frac['ref_deeper']:6.2f}%   {int(cnt.get('ref_deeper',0)):>8,}")
    print(f"    declined to label (Unknown)        {100*frac['unknown']:6.2f}%   {int(cnt.get('unknown',0)):>8,}")
    if cnt.get("ref_deeper", 0):
        blk = (pd.DataFrame({"cm": cm, "ref": rf})[rel == "ref_deeper"]
               .groupby(["cm", "ref"]).size().nlargest(6))
        for (c, r), v in blk.items():
            print(f"        {v:>7,}  stopped at '{c}'  where the author said '{r}'")


    print("\n[C] GROUND — does the added resolution survive a held-out split?")
    kg = set()
    for mk in common.kg_markers(organ).values():
        for t in ("defining", "required", "supporting"):
            kg |= {str(g).upper() for g in mk.get(t, [])}
    import scanpy as sc
    hv = sc.pp.highly_variable_genes(ad, n_top_genes=2000, flavor="seurat", inplace=False)
    feats = [g for g in ad.var_names[np.asarray(hv["highly_variable"])] if str(g).upper() not in kg]
    print(f"    feature space: {len(feats)} marker-free HVGs "
          f"(this organ's HVG minus {len(kg)} KG marker symbols; the full pipeline uses a "
          f"cross-organ feature set)")
    sub = ad[:, feats]
    X = np.asarray(sub.X.todense() if hasattr(sub.X, "todense") else sub.X, dtype=np.float32)


    rng = np.random.default_rng(SEED)
    rows = []
    for kind, arena_src, part_src, relname in (("deeper", rf, cm, "cellman_deeper"),
                                               ("coarser", cm, rf, "ref_deeper")):
        for arena in sorted({r for r in arena_src[rel == relname]}):
            inside = arena_src == arena
            parts = pd.Series(part_src[inside]).value_counts()
            parts = parts[parts >= a.min_block]
            if len(parts) < 2:
                continue
            idx = np.flatnonzero(inside & np.isin(part_src, list(parts.index)))
            gopts = _groupings(ad.obs.iloc[idx])
            for lab in parts.index:
                r, gname = None, (gopts[0][1] if gopts else "not testable")
                for gm, gn in gopts:
                    r = _replicates(X[idx], (part_src[idx] == lab), gm, rng)
                    if r is not None:
                        gname = gn; break
                row = dict(kind=kind, arena=arena, part=lab, n=int(parts[lab]), grouping=gname)
                row.update(r or dict(recall=None, null=None, z=None, replicates=None))
                if row["replicates"] is None:      won = None
                elif kind == "deeper":             won = "cellman" if row["replicates"] else None
                else:                              won = "reference" if row["replicates"] else "cellman"
                row["favours"] = won


                if "donor_id" in ad.obs:
                    p = ad.obs["donor_id"].iloc[idx][part_src[idx] == lab].value_counts(normalize=True)
                    row["n_donor"] = int(len(p)); row["donor_hhi"] = round(float((p ** 2).sum()), 3)
                rows.append(row)
    C = pd.DataFrame(rows)
    if len(C):
        R["C_ground"] = {}
        for kind, title in (("deeper", "B) CeLLMan split it — is the refinement real?"),
                            ("coarser", "D) the AUTHOR split it — was CeLLMan's floor too high?")):
            K = C[C.kind == kind]
            if not len(K):
                continue
            dec = K[K.replicates.notna()]
            ok = int(dec.replicates.sum())
            R["C_ground"][kind] = dict(parts=len(K), tested=len(dec), replicated=ok,
                                       not_testable=int(K.replicates.isna().sum()))
            print(f"\n    {title}")
            print(f"      tested {len(dec)} of {len(K)} parts   replicated {ok}   "
                  f"not testable {int(K.replicates.isna().sum())}")
            show = K.sort_values("n", ascending=False).head(10)
            print(f"      {'part':40s}{'n':>7}{'recall':>8}{'null':>7}{'z':>7}  {'rep':4s}{'favours':>10}{'HHI':>7}")
            for _, r in show.iterrows():
                rp = "-" if r.replicates is None else ("YES" if r.replicates else "no")
                nn = lambda v: v if v == v and v is not None else float("nan")
                print(f"      {str(r.part)[:39]:40s}{r.n:>7,}{nn(r.recall):>8}{nn(r.null):>7}"
                      f"{nn(r.z):>7}  {rp:4s}{str(r.favours or '-'):>10}{r.get('donor_hhi',float('nan')):>7}")
        C.to_csv(os.path.join(common.TAB, f"replication_{organ}.csv"), index=False)
    else:
        R["C_ground"] = dict(parts=0, tested=0, replicated=0, pct=None, not_testable=0)
        print("    no reference label was split into >=2 parts above the size floor")

    json.dump(R, open(os.path.join(common.TAB, f"organ_report_{organ}.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"\n  -> tables/organ_report_{organ}.json\n")


if __name__ == "__main__":
    main()
