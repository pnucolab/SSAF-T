#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
import sys as _sys; _sys.path.insert(0, str(HERE))
import config as _C
DIRS = {"TS": (HERE / "validation_ts", "human"), "TM": (HERE / "validation_tm", "mouse")}
AUC_HIT, AUC_MISS = 0.65, 0.55
P_HIT = 0.60
MIN_GENES = 1
SHOW_GENES = 6
SEED = 0
RELS = ("cellman_deeper", "ref_deeper", "conflict")
CASE = {"cellman_deeper": "CeLLMan deeper", "ref_deeper": "CeLLMan coarser", "conflict": "conflict"}


def _load(name, path, extra):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(extra))
    try:
        spec.loader.exec_module(m)
    finally:
        sys.path.remove(str(extra))
    return m


def auc(a, b):
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    rk = pd.Series(np.concatenate([a, b])).rank().to_numpy()
    n1, n2 = len(a), len(b)
    return float((rk[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


class Organ:

    def __init__(self, common, organ, species, extra_children=None):
        import scanpy as sc
        self.sc = sc
        self.ad = common.load(organ)
        self.cm = self.ad.obs["cm"].astype(str).to_numpy()
        self.rf = self.ad.obs["ref"].astype(str).to_numpy()
        self.rel = self.ad.obs["relation"].astype(str).to_numpy()
        kg = common.kg_markers(organ, species=species)
        self.kg = {k.lower(): v for k, v in kg.items()}
        self.gene = {str(g).upper(): str(g) for g in self.ad.var_names}
        self.cache = {}


        self.children = extra_children or {}

    def markers(self, label, exclude=None):
        v = self.kg.get(str(label).lower())
        if v is None:
            ex = str(exclude).lower() if exclude is not None else None
            kids = [self.kg[c] for c in self.children.get(str(label).lower(), [])
                    if c in self.kg and c != ex]
            if not kids:
                return None
            srcs = kids
        else:
            srcs = [v]
        out = {}
        for src in srcs:
            for tier, key in ((0, "defining"), (1, "required")):
                for g in src.get(key, []):
                    gg = self.gene.get(str(g).upper())
                    if gg is not None and (gg not in out or tier < out[gg]):
                        out[gg] = tier
        return out

    def own(self, label, other):
        a, b = self.markers(label), self.markers(other, exclude=label)
        if a is None:
            return None
        if b is None:
            b = {}
        return {g: t for g, t in a.items() if g not in b}

    def score(self, genes):
        key = frozenset(genes)
        if key not in self.cache:
            self.sc.tl.score_genes(self.ad, sorted(genes), score_name="_s", random_state=SEED,
                                   use_raw=False)
            self.cache[key] = self.ad.obs["_s"].to_numpy().copy()
        return self.cache[key]


def judge(o, min_block):
    rows, plots = [], []
    df = pd.DataFrame({"cm": o.cm, "ref": o.rf, "rel": o.rel})
    blocks = df[df.rel.isin(RELS)].groupby(["rel", "cm", "ref"]).size()
    for (rel, x, y), n in blocks.items():
        row = dict(relation=rel, case=CASE[rel], cm=x, ref=y, n=int(n))
        blk = (o.rel == rel) & (o.cm == x) & (o.rf == y)
        if n < min_block:
            rows.append({**row, "verdict": "too_small"}); continue

        if rel in ("cellman_deeper", "ref_deeper"):

            if rel == "cellman_deeper":
                fine, parent = x, y
                cmp_ = (o.rf == y) & (o.cm != x)
            else:
                fine, parent = y, x
                cmp_ = (o.cm == x) & (o.rf != y)
            genes = o.own(fine, parent)
            if genes is None:
                rows.append({**row, "verdict": "untestable", "why": "label not in KG"}); continue
            if len(genes) < MIN_GENES:
                rows.append({**row, "verdict": "untestable", "why": "too few own markers"}); continue
            if int(cmp_.sum()) < min_block:
                rows.append({**row, "verdict": "untestable", "why": "no comparison group"}); continue
            s = o.score(genes)
            a = auc(s[blk], s[cmp_])
            hit = a >= AUC_HIT
            miss = a <= AUC_MISS
            if rel == "cellman_deeper":
                v = "cellman" if hit else "reference" if miss else "undecided"
            else:
                v = "reference" if hit else "cellman" if miss else "undecided"
            rows.append({**row, "verdict": v, "stat": round(a, 4), "method": "AUC",
                         "n_cmp": int(cmp_.sum()), "n_genes": len(genes),
                         "genes_fine": ";".join(sorted(genes))})
            plots.append(dict(row=rows[-1], groups=[("this block", blk), ("comparison", cmp_)],
                              genes=[(fine, genes)]))
            continue


        gx, gy = o.own(x, y), o.own(y, x)
        if gx is None or gy is None:
            rows.append({**row, "verdict": "untestable", "why": "label not in KG"}); continue
        if len(gx) < MIN_GENES or len(gy) < MIN_GENES:
            rows.append({**row, "verdict": "untestable", "why": "too few own markers"}); continue
        sx, sy = o.score(gx), o.score(gy)
        d = sx - sy
        ax_ = (o.cm == x) & (o.rf == x)
        ay_ = (o.cm == y) & (o.rf == y)
        method, stat = None, None
        if ax_.sum() >= min_block and ay_.sum() >= min_block:
            dx, dy = float(np.median(d[ax_])), float(np.median(d[ay_]))
            if dx > dy:
                stat = (float(np.median(d[blk])) - dy) / (dx - dy)
                method = "anchored position"
        if method is None:
            zx = (sx - sx.mean()) / (sx.std() or 1.0)
            zy = (sy - sy.mean()) / (sy.std() or 1.0)
            stat = float((zx[blk] > zy[blk]).mean())
            method = "z fraction"
        v = "cellman" if stat >= P_HIT else "reference" if stat <= 1 - P_HIT else "undecided"
        rows.append({**row, "verdict": v, "stat": round(stat, 4), "method": method,
                     "n_genes": min(len(gx), len(gy)),
                     "genes_cm": ";".join(sorted(gx)), "genes_ref": ";".join(sorted(gy))})
        groups = [("this block", blk)]
        if ax_.sum():
            groups.append((f"both: {x[:26]}", ax_))
        if ay_.sum():
            groups.append((f"both: {y[:26]}", ay_))
        plots.append(dict(row=rows[-1], groups=groups, genes=[(x, gx), (y, gy)]))
    return pd.DataFrame(rows), plots


def _pick(o, genes, masks):
    X = o.ad[:, list(genes)].X
    spread = {}
    for j, g in enumerate(genes):
        col = X[:, j]
        col = np.asarray(col.todense()).ravel() if hasattr(col, "todense") else np.asarray(col).ravel()
        means = [float(col[m].mean()) for m in masks]
        spread[g] = max(means) - min(means)
    return sorted(genes, key=lambda g: (genes[g], -spread[g]))[:SHOW_GENES]


def plot_organ(o, atlas, organ, plots, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if not plots:
        return
    order = {"CeLLMan deeper": 0, "CeLLMan coarser": 1, "conflict": 2}
    plots = sorted(plots, key=lambda p: (order[p["row"]["case"]], -p["row"]["n"]))
    VCOL = {"cellman": "#2f7fb5", "reference": "#6e6e6e", "undecided": "#b0b0b0"}
    VTXT = {"cellman": "supports CeLLMan", "reference": "supports the author", "undecided": "undecided"}

    ncol = 3
    nrow = math.ceil(len(plots) / ncol)
    fig = plt.figure(figsize=(7.2 * ncol, 3.25 * nrow + 1.2))
    gs = fig.add_gridspec(nrow, ncol, hspace=1.35, wspace=0.75, left=0.08, right=0.985,
                          top=1 - 0.9 / (3.25 * nrow + 1.2), bottom=0.45 / (3.25 * nrow + 1.2))
    for k, p in enumerate(plots):
        ax = fig.add_subplot(gs[k // ncol, k % ncol])
        r = p["row"]
        masks = [m for _, m in p["groups"]]
        genes, owner = [], []
        for lab, gset in p["genes"]:
            for g in _pick(o, gset, masks):
                genes.append(g); owner.append(lab)
        X = o.ad[:, genes].X
        X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
        M = np.array([[X[m, j].mean() for j in range(len(genes))] for m in masks])
        F = np.array([[(X[m, j] > 0).mean() for j in range(len(genes))] for m in masks])


        vmax = max(float(M.max()), 0.5)
        yy, xx = np.meshgrid(np.arange(len(masks)), np.arange(len(genes)), indexing="ij")
        sc_ = ax.scatter(xx.ravel(), yy.ravel(), s=12 + 150 * F.ravel(), c=M.ravel(), cmap="Reds",
                         vmin=0, vmax=vmax, edgecolor="#555555", lw=0.3, zorder=3)
        cb = fig.colorbar(sc_, ax=ax, fraction=0.025, pad=0.01)
        cb.ax.tick_params(labelsize=6.5); cb.set_label("mean log1p", fontsize=6.5)
        ax.set_xticks(range(len(genes)))
        ax.set_xticklabels(genes, rotation=60, ha="right", fontsize=7.5)
        if len(p["genes"]) == 2:
            for t, lab in zip(ax.get_xticklabels(), owner):
                t.set_color("#2f7fb5" if lab == r["cm"] else "#6e6e6e")
            cut = sum(1 for lab in owner if lab == p["genes"][0][0])
            ax.axvline(cut - 0.5, color="#999999", lw=0.8)
        ax.set_yticks(range(len(masks)))
        ax.set_yticklabels([f"{name}  (n={int(m.sum()):,})" for name, m in p["groups"]], fontsize=7.5)
        ax.set_xlim(-0.6, len(genes) - 0.4); ax.set_ylim(len(masks) - 0.5, -0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        stat = r.get("stat")
        head = f"{r['case']}:  CeLLMan `{r['cm']}`  |  author `{r['ref']}`"
        sub = (f"n={r['n']:,}   {r.get('method', '')} {stat:.2f}   ->  "
               if stat is not None and not pd.isna(stat) else f"n={r['n']:,}   ->  ")
        ax.set_title(head, fontsize=8.3, loc="left", pad=16)
        ax.text(0.0, 1.03, sub, transform=ax.transAxes, fontsize=7.8, color="#444444", va="bottom")
        ax.text(1.0, 1.03, VTXT[r["verdict"]], transform=ax.transAxes, fontsize=7.8,
                color=VCOL[r["verdict"]], va="bottom", ha="right", fontweight="bold")

    fig.suptitle(f"{atlas} {organ.replace('_', ' ')}  --  KG own-marker expression behind each verdict "
                 f"(CeLLMan deeper, CeLLMan coarser, conflict)", fontsize=12, x=0.07, ha="left")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", mfc="#dddddd", mec="#555555", ms=m, label=l)
                        for m, l in ((3.5, "10% of cells detect"), (7, "50%"), (9.5, "100%"))]
               + [Line2D([], [], ls="", label="colour: mean log1p (scale per panel)"),
                  Line2D([], [], ls="", label="gene names: blue CeLLMan's label, grey the author's")],
               loc="upper right", ncol=6, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.985, 1 - 0.25 / (3.25 * nrow + 1.2)))
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{atlas}_{organ}.png", dpi=110, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", choices=list(DIRS), default="TS")
    ap.add_argument("--organs", nargs="*", default=None)
    ap.add_argument("--min-block", type=int, default=20)
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()
    d, species = DIRS[a.atlas]
    common = _load(f"common_{a.atlas}", d / "common.py", d)
    vocab = _load(f"vocab_{a.atlas}", d / "vocab.py", d)
    extra_children = {}
    for child, parent in getattr(vocab, "EXTRA_EDGES", []):
        extra_children.setdefault(str(parent).lower(), []).append(str(child).lower())
    organs = a.organs or [l.strip() for l in (d / "organs.txt").read_text().split() if l.strip()]
    out_csv = _C.FIG / f"marker_score_{a.atlas}.csv"

    parts = []
    if a.organs and out_csv.exists():
        old = pd.read_csv(out_csv)
        parts.append(old[~old.organ.isin(organs)])
    for organ in organs:
        try:
            o = Organ(common, organ, species, extra_children)
            t, plots = judge(o, a.min_block)
        except Exception as e:                                       # noqa: BLE001
            print(f"  {organ:16s} skipped ({type(e).__name__}: {str(e)[:80]})", flush=True)
            continue
        t["atlas"], t["organ"] = a.atlas, organ
        parts.append(t)
        v = t.groupby(["case", "verdict"]).n.sum().unstack(fill_value=0)
        print(f"  {organ:16s} blocks {len(t):4d} tested {len(plots):4d}", flush=True)
        for case, r in v.iterrows():
            print(f"      {case:16s} " + "  ".join(f"{k} {int(x):,}" for k, x in r.items() if x),
                  flush=True)
        if not a.no_plot:
            plot_organ(o, a.atlas, organ, plots, _C.FIG / "marker_supp")
        del o
    D = pd.concat(parts, ignore_index=True)
    D.to_csv(out_csv, index=False)
    print(f"\n[{a.atlas}] cells by case and verdict")
    print(D.groupby(["case", "verdict"]).n.sum().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
