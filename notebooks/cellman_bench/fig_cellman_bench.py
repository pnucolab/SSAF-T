#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as _C


def _mod(name, path, extra=None):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    if extra:
        sys.path.insert(0, str(extra))
    try:
        spec.loader.exec_module(m)
    finally:
        if extra:
            sys.path.remove(str(extra))
    return m


fv = _mod("fig_validation", HERE / "fig_validation.py", HERE)

import matplotlib.pyplot as plt          # noqa: E402  (fv sets the Agg backend and the fonts)
from matplotlib.lines import Line2D      # noqa: E402
from matplotlib.patches import Patch     # noqa: E402

FS, CM, RF = fv.FS, fv.CM, fv.RF
AGREE, GROUND, ERROR, PALE, VOID = fv.AGREE, fv.GROUND, fv.ERROR, fv.PALE, fv.VOID
INK, INK2, INK3 = fv.INK, fv.INK2, fv.INK3

CASES = [("CeLLMan deeper", "cellman_deeper"), ("CeLLMan coarser", "ref_deeper"), ("conflict", "conflict")]
SIDES = [("cellman", "supports CeLLMan", CM), ("reference", "supports the author", GROUND),
         ("undecided", "undecided", PALE), ("untested", "not testable", VOID)]
AUC_HIT, AUC_MISS, P_HIT = 0.65, 0.55, 0.60


def verdicts():
    parts = [pd.read_csv(f) for f in sorted(_C.FIG.glob("marker_score_T?.csv"))]
    D = pd.concat(parts, ignore_index=True)
    D["side"] = D.verdict.where(D.verdict.isin(["cellman", "reference", "undecided"]), "untested")
    out = {}
    for (a, o, c, s), n in D.groupby(["atlas", "organ", "case", "side"]).n.sum().items():
        out.setdefault((a, o, c), {})[s] = int(n)
    return out, D


def filled(rel, V, atlas, organ, case, col):
    tot = float(rel[(rel.ds == atlas) & (rel.organ == organ)][col].sum())
    v = V.get((atlas, organ, case), {})
    got = [float(v.get(k, 0)) for k in ("cellman", "reference", "undecided")]
    return got + [max(tot - sum(got), 0.0)], tot


def _stack(ax, yy, vals, cols, height, total=None, edge="#ffffff", lw=0.5, label_min=None):
    tot = total if total is not None else sum(vals)
    if tot <= 0:
        return
    left = 0.0
    for v, c in zip(vals, cols):
        w = 100 * v / tot
        ax.barh(yy, w, left=left, height=height, color=c, edgecolor=edge, lw=lw, zorder=3)
        if label_min is not None and w >= label_min:
            ax.text(left + w / 2, yy, f"{w:.0f}", ha="center", va="center", fontsize=FS * 0.78,
                    color="#ffffff" if c in (CM, GROUND, ERROR) else INK, zorder=5)
        left += w


SIDECOL = [c for _, _, c in SIDES]


def pooled(rel, V, tag, case, col):
    vals, tot = np.zeros(4), 0.0
    for o in rel[rel.ds == tag].organ:
        v, t = filled(rel, V, tag, o, case, col)
        vals += np.array(v); tot += t
    return vals, tot


def verdict_bars(ax, rel, V, case, col, subtitle):
    for i, (_, tag, _) in enumerate(fv.DS):
        vals, tot = pooled(rel, V, tag, case, col)
        nall = float(rel[rel.ds == tag].n_cells.sum())
        _stack(ax, 1 - i, list(vals), SIDECOL, 0.62, edge=INK3, lw=0.5, label_min=8)
        ax.text(0, 1 - i + 0.43, f"{tag}   {int(tot):,} cells, {100*tot/nall:.1f}% of all",
                fontsize=FS * 0.80, color=INK2, va="bottom")
    ax.set_xlim(0, 100); ax.set_ylim(-0.5, 1.75)
    ax.set_yticks([]); ax.set_xlabel(subtitle, fontsize=FS * 0.86, color=INK2)
    fv.style(ax)
    ax.spines["left"].set_visible(False)


_ORGAN = {}


def organ(atlas, name):
    if (atlas, name) not in _ORGAN:
        d = HERE / ("validation_ts" if atlas == "TS" else "validation_tm")
        common = _mod(f"common_{atlas}", d / "common.py", d)
        _ORGAN[(atlas, name)] = common.load(name)
    return _ORGAN[(atlas, name)]


def dotplot(ax, a, groups, genes, owner_cols=None, show=5):
    masks = [m for _, m in groups]
    pick = []
    for gs in genes:
        gs = [g for g in gs if g in a.var_names]
        X = a[:, gs].X
        X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
        spread = [max(X[m, j].mean() for m in masks) - min(X[m, j].mean() for m in masks)
                  for j in range(len(gs))]
        pick.append([gs[j] for j in np.argsort(spread)[::-1][:show]])
    flat = [g for p in pick for g in p]
    X = a[:, flat].X
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    M = np.array([[X[m, j].mean() for j in range(len(flat))] for m in masks])
    F = np.array([[(X[m, j] > 0).mean() for j in range(len(flat))] for m in masks])
    yy, xx = np.meshgrid(np.arange(len(masks)), np.arange(len(flat)), indexing="ij")
    sc_ = ax.scatter(xx.ravel(), yy.ravel(), s=8 + 110 * F.ravel(), c=M.ravel(), cmap="Reds",
                     vmin=0, vmax=max(float(M.max()), 0.5), edgecolor="#666666", lw=0.3, zorder=3)
    ax.set_xticks(range(len(flat)))
    ax.set_xticklabels(flat, rotation=60, ha="right", fontsize=FS * 0.72)
    if owner_cols:
        k = 0
        for p, c in zip(pick, owner_cols):
            for _ in p:
                ax.get_xticklabels()[k].set_color(c); k += 1
        ax.axvline(len(pick[0]) - 0.5, color=INK3, lw=0.7)
    ax.set_yticks(range(len(masks)))
    ax.set_yticklabels([f"{n}  ({int(m.sum()):,})" for n, m in groups], fontsize=FS * 0.74)
    ax.set_xlim(-0.6, len(flat) - 0.4); ax.set_ylim(len(masks) - 0.5, -0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    cb = ax.figure.colorbar(sc_, ax=ax, fraction=0.03, pad=0.015)
    cb.ax.tick_params(labelsize=FS * 0.6); cb.set_label("mean log1p", fontsize=FS * 0.62)


def block_row(D, atlas, organ_, case, cm, ref):
    r = D[(D.atlas == atlas) & (D.organ == organ_) & (D.case == case) & (D.cm == cm) & (D.ref == ref)]
    return r.iloc[0]


def ex_deeper(ax, D, letter):
    a = organ("TS", "lymph_node")
    r = block_row(D, "TS", "lymph_node", "CeLLMan deeper", "naive B cell", "B cell")
    cm, rf = a.obs["cm"].astype(str).to_numpy(), a.obs["ref"].astype(str).to_numpy()
    dotplot(ax, a, [("CeLLMan: naive B cell", (cm == r.cm) & (rf == r.ref)),
                    ("rest of author's B cell", (rf == r.ref) & (cm != r.cm))],
            [r.genes_fine.split(";")], show=8)
    fv.title(ax, letter, "CeLLMan deeper, supported",
             f"TS lymph node: author `B cell`, CeLLMan `naive B cell` ({r.n:,} cells). The finer name's own "
             f"markers in the renamed cells against the author's other B cells: AUC {r.stat:.2f}", wrap=52)


def ex_conflict(ax, D, letter):
    a = organ("TS", "heart")
    r = block_row(D, "TS", "heart", "conflict", "atrial cardiomyocyte", "ventricular cardiomyocyte")
    cm, rf = a.obs["cm"].astype(str).to_numpy(), a.obs["ref"].astype(str).to_numpy()
    groups = [("disputed block", (cm == r.cm) & (rf == r.ref)),
              ("both: atrial", (cm == r.cm) & (rf == r.cm)),
              ("both: ventricular", (cm == r.ref) & (rf == r.ref))]
    dotplot(ax, a, [g for g in groups if g[1].sum() > 0],
            [r.genes_cm.split(";"), r.genes_ref.split(";")], owner_cols=[CM, GROUND], show=5)
    fv.title(ax, letter, "Conflict, markers side with CeLLMan",
             f"TS heart: CeLLMan `atrial cardiomyocyte`, author `ventricular cardiomyocyte` ({r.n:,} cells). "
             f"Blue genes CeLLMan's label, grey the author's; position {r.stat:.2f} between the two agreed groups",
             wrap=52)


def ex_floor(ax, D, letter):
    a = organ("TS", "spleen")
    r = block_row(D, "TS", "spleen", "CeLLMan coarser", "T cell", "CD8-positive, alpha-beta T cell")
    cm, rf = a.obs["cm"].astype(str).to_numpy(), a.obs["ref"].astype(str).to_numpy()
    dotplot(ax, a, [("CeLLMan T cell, author CD8", (cm == r.cm) & (rf == r.ref)),
                    ("both: CD8 T cell", (cm == r.ref) & (rf == r.ref))],
            [r.genes_fine.split(";"), ["CCL5", "GZMK", "NKG7"]], owner_cols=[INK, INK3], show=3)
    fv.title(ax, letter, "CeLLMan coarser, stopping justified",
             f"TS spleen: author `CD8 T cell`, CeLLMan `T cell` ({r.n:,} cells). Black: the scored CD8 markers "
             f"(AUC {r.stat:.2f}); grey: cytotoxic genes the KG lists as supporting, not scored", wrap=52)


def ex_prostate(ax, letter):
    d = pd.read_csv(_C.FIG / "case_prostate_seminal.csv")
    genes = list(dict.fromkeys(d.gene))
    groups = ["seminal (SEMG1-high)", "prostate luminal"]
    M = d.pivot(index="group", columns="gene", values="pct_detected").reindex(index=groups,
                                                                              columns=genes).to_numpy()
    n = {g: int(d[d.group == g].n.iloc[0]) for g in groups}
    ax.imshow(M, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=FS * 0.78,
                    color="#ffffff" if M[i, j] > 55 else INK)
    prog = dict(zip(d.gene, d.programme))
    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels(genes, fontsize=FS * 0.80, rotation=45, ha="right")
    for t, g in zip(ax.get_xticklabels(), genes):
        t.set_color(ERROR if prog[g] == "seminal vesicle" else INK)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([f"seminal vesicle  {n[groups[0]]:,} cells, 1 donor",
                        f"prostate luminal  {n[groups[1]]:,} cells"], fontsize=FS * 0.80)
    ax.axvline(sum(1 for g in genes if prog[g] == "seminal vesicle") - 0.5, color=INK, lw=1.2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    fv.title(ax, letter, "A reference error",
             "TS prostate: all 10,652 cells are `luminal cell of prostate epithelium` in the reference and in "
             "the paper's Table S2; % of cells detecting each gene (red: seminal-vesicle programme, black: "
             "prostate identity). CeLLMan stopped at `epithelial cell` for 7,852 of the seminal cells, and the "
             "prostate luminal markers are off in them (AUC 0.36). Also found: 5,235 TS thymus `thymocytes` "
             "without any thymocyte marker, 655 TS ovary granulosa cells filed as vascular smooth muscle",
             wrap=150)


def supp_figure(D, out):
    fig = plt.figure(figsize=(14.6, 12.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.62], hspace=0.95, wspace=0.95,
                          left=0.15, right=0.965, top=0.88, bottom=0.08)
    ex_deeper(fig.add_subplot(gs[0, 0]), D, "a")
    ex_conflict(fig.add_subplot(gs[0, 1]), D, "b")
    ex_floor(fig.add_subplot(gs[0, 2]), D, "c")
    ex_prostate(fig.add_subplot(gs[1, :]), "d")
    fig.text(0.15, 0.965, "Dot size: fraction of cells detecting the gene; colour: mean log1p expression "
                          "(one scale per panel). Every other block: figures/marker_supp/.",
             fontsize=FS * 0.86, color=INK2)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"cellman_bench_supp_examples.{ext}", dpi=200, facecolor="white")
    plt.close(fig)


LEDGER_W = [2.30, 0.66, 1.05, 1.05, 1.05, 1.25]


def _net(rel, V, atlas, organ_):
    n = float(rel[(rel.ds == atlas) & (rel.organ == organ_)].n_cells.iloc[0])
    pro = ag = 0.0
    for case, col in CASES:
        v, _ = filled(rel, V, atlas, organ_, case, col)
        pro += v[0]; ag += v[1]
    return 100 * pro / n, 100 * ag / n


def ledger(fig, rel, rpt, rows, V, *, top, bottom, left=0.128, right=0.985):
    n = len(rows)
    y = np.arange(n)[::-1]
    gs = fig.add_gridspec(2, 6, width_ratios=LEDGER_W, wspace=0.30, height_ratios=[3.4, n],
                          hspace=0.028, left=left, right=right, top=top, bottom=bottom)
    P = [fig.add_subplot(gs[0, j]) for j in range(6)]
    L = [fig.add_subplot(gs[1, j]) for j in range(6)]


    yp, tick = [1, 0], []
    net = {}
    for i, (_, tag, _) in enumerate(fv.DS):
        srel, q = rel[rel.ds == tag], rpt[rpt.ds == tag]
        nc = srel.n_cells.sum()
        anc = 100 * (srel["same"].sum() + srel["cellman_deeper"].sum()) / nc
        tick.append(f"{tag}   {int(nc):,} cells\ndoes not contradict the author {anc:.1f}%")
        _stack(P[0], yp[i], [srel[k].sum() for k, _, _ in fv.REL], [c for _, _, c in fv.REL],
               0.62, total=nc, lw=0.8, label_min=5.5)
        P[0].plot([anc, anc], [yp[i] - 0.36, yp[i] + 0.36], color=INK, lw=1.6, zorder=5)
        P[1].text(0.5, yp[i], f"{int(q.n_ref_labels.sum())} → {int(q.n_cm_labels.sum())}",
                  ha="center", va="center", fontsize=FS * 0.84, color=INK)
        pro = ag = 0.0
        for j, (case, col) in enumerate(CASES):
            vals, _ = pooled(rel, V, tag, case, col)
            _stack(P[2 + j], yp[i], list(vals), SIDECOL, 0.62, edge=INK3, lw=0.6, label_min=9)
            pro += vals[0]; ag += vals[1]
        net[tag] = (100 * pro / nc, 100 * ag / nc)
        P[5].barh(yp[i], net[tag][0], height=0.62, color=CM, zorder=3)
        P[5].barh(yp[i], -net[tag][1], height=0.62, color=GROUND, zorder=3)
        P[5].text(net[tag][0] + 1.0, yp[i], f"{net[tag][0]:.1f}", ha="left", va="center",
                  fontsize=FS * 0.80, color=INK)
        P[5].text(-net[tag][1] - 1.0, yp[i], f"{net[tag][1]:.1f}", ha="right", va="center",
                  fontsize=FS * 0.80, color=INK)

    lo = min(min(-_net(rel, V, t, r.organ)[1] for t, r in rows) * 1.15, -12)
    hi = max(max(_net(rel, V, t, r.organ)[0] for t, r in rows) * 1.15, 60)
    for j, ax in enumerate(P):
        ax.set_ylim(-0.85, 1.85); ax.set_yticks([]); ax.set_xticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlim(*((0, 1) if j == 1 else (lo, hi) if j == 5 else (0, 100)))
    P[0].set_yticks(yp); P[0].set_yticklabels(tick, fontsize=FS * 0.78, color=INK, linespacing=1.35)
    P[0].tick_params(length=0)
    heads = [("a  RELATION", ""), ("b  CELL TYPES", "author → CeLLMan"),
             ("c  DEEPER", "is CeLLMan's finer name marked?"),
             ("d  COARSER", "is the author's finer name marked?"),
             ("e  CONFLICT", "whose markers fit?"), ("f  NET", "cells the markers support")]
    for ax, (t, sub) in zip(P, heads):
        ax.set_title(t, fontsize=FS * 0.95, color=INK, loc="left", pad=6)
        if sub:
            ax.text(0.0, 1.74, sub, transform=ax.transAxes, fontsize=FS * 0.72, color=INK2,
                    ha="left", va="bottom")
    for j in range(6):
        P[j].axhline(-0.80, color=INK3, lw=0.9, zorder=6, clip_on=False)


    for i, (tag, r) in enumerate(rows):
        _stack(L[0], y[i], [r[k] for k, _, _ in fv.REL], [c for _, _, c in fv.REL], 0.72,
               total=r.n_cells)
        L[1].plot([r.n_ref_labels, r.n_cm_labels], [y[i]] * 2, color=INK3, lw=0.8, zorder=2)
        L[1].scatter([r.n_ref_labels], [y[i]], s=11, color=GROUND, zorder=3)
        L[1].scatter([r.n_cm_labels], [y[i]], s=11, color=CM, zorder=3)
        for j, (case, col) in enumerate(CASES):
            vals, tot = filled(rel, V, tag, r.organ, case, col)
            if tot > 0:
                _stack(L[2 + j], y[i], vals, SIDECOL, 0.72, edge=INK3, lw=0.35)
        a_, b_ = _net(rel, V, tag, r.organ)
        L[5].barh(y[i], a_, height=0.72, color=CM, zorder=3)
        L[5].barh(y[i], -b_, height=0.72, color=GROUND, zorder=3)
    for ax, xl in zip(L, ("% of cells", "cell types used", "% of deeper cells", "% of coarser cells",
                          "% of conflict cells", "← author   CeLLMan →   (% of organ)")):
        fv.style(ax)
        ax.set_ylim(-0.8, n - 0.2)
        ax.set_xlabel(xl, fontsize=FS * 0.86, color=INK2)
    for j in (0, 2, 3, 4):
        L[j].set_xlim(0, 100)
    L[5].set_xlim(lo, hi); L[5].axvline(0, color=INK3, lw=0.8, zorder=2)
    P[5].axvline(0, color=INK3, lw=0.8, zorder=2)
    L[0].set_yticks(y)
    L[0].set_yticklabels([f"{t}  {r.organ.replace('_', ' ')}" for t, r in rows], fontsize=FS * 0.80,
                         color=INK)
    for ax in L[1:]:
        plt.setp(ax.get_yticklabels(), visible=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_C.FIG))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    rel, _, rpt = fv.load()
    V, D = verdicts()

    rows = []
    for _, tag, _ in fv.DS:
        q = rpt[rpt.ds == tag].merge(rel[rel.ds == tag][["organ"] + [k for k, _, _ in fv.REL]],
                                     on="organ")
        rows += [(tag, r) for _, r in q.sort_values("A_ancestor_consistent",
                                                    ascending=False).iterrows()]
    n = len(rows)
    H = 1.3 + 0.213 * n + 1.1 + 2.6
    fig = plt.figure(figsize=(14.6, H))
    ledger(fig, rel, rpt, rows, V, top=1 - 1.3 / H, bottom=2.6 / H)
    fig.legend(handles=[Patch(facecolor=c, label=l) for _, l, c in fv.REL],
               loc="lower left", ncol=5, frameon=False, fontsize=FS * 0.80, labelcolor=INK2,
               title="a   relation of the two labels", title_fontsize=FS * 0.80, columnspacing=1.0,
               bbox_to_anchor=(0.128, 1.35 / H))
    fig.legend(handles=[Patch(facecolor=c, edgecolor=INK3, lw=0.5, label=l) for _, l, c in SIDES],
               loc="lower left", ncol=4, frameon=False, fontsize=FS * 0.80, labelcolor=INK2,
               title="c–f   which label the markers support", title_fontsize=FS * 0.80,
               columnspacing=1.0, bbox_to_anchor=(0.560, 1.35 / H))
    t = D[D.verdict.isin(["cellman", "reference", "undecided"])]
    foot = (f"{len(rpt)} organs, {int(rel.n_cells.sum()):,} cells "
            f"({', '.join(f'{x} {int(rel[rel.ds == x].n_cells.sum()):,}' for _, x, _ in fv.DS)}), each "
            "annotated by the group that generated it. Labels are related on the KG IS_A DAG, so a finer name under "
            "the author's is agreement, not error. Verdicts: scanpy score_genes on the KG's literature-grounded "
            "DEFINING and REQUIRED own markers of the two competing labels (genes both carry removed; supporting "
            "markers left out, since they include programmes a finer name does not own). c, d: AUC of the score "
            "between the block and the other cells of the same parent label (>= 0.65 / <= 0.55); e: position of "
            "the block between the cells both annotations give each label (>= 0.60 / <= 0.40). A block needs 20 "
            "cells, both labels in the KG and one own marker in the matrix, otherwise it is not testable; "
            f"{int((t.n_genes == 1).sum())} of {len(t):,} tested blocks rest on a single defining marker. Worked "
            "examples: cellman_bench_supp_examples; every block: marker_supp/.")
    fig.text(0.128, 0.15 / H, "\n".join(textwrap.wrap(foot, 175)), fontsize=FS * 0.78, color=INK2,
             ha="left", va="bottom", linespacing=1.5)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"cellman_bench_main.{ext}", dpi=200, facecolor="white")


    svg = out / "cellman_bench_main.svg"
    s = svg.read_text().replace("<svg ", '<svg xml:space="preserve" ', 1)
    s = s.replace("font-family: 'Liberation Sans', 'DejaVu Sans', sans-serif",
                  "font-family: 'Liberation Sans', Arial, Helvetica, sans-serif; white-space: pre")
    svg.write_text(s)
    plt.close(fig)
    print(f"wrote {out/'cellman_bench_main.png'}  (14.6 x {H:.1f} in)")


    try:
        supp_figure(D, out)
        print(f"wrote {out/'cellman_bench_supp_examples.png'}")
    except FileNotFoundError as e:
        print(f"supplement skipped: {e.filename} not found")


if __name__ == "__main__":
    main()
