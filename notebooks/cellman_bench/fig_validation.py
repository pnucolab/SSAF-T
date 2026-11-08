#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import json
import textwrap
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt       # noqa: E402
import numpy as np                    # noqa: E402
import pandas as pd                   # noqa: E402
from matplotlib.lines import Line2D   # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


for _f in fm.findSystemFonts(fontpaths=[str(Path.home() / ".fonts")]):
    fm.fontManager.addfont(_f)
_HAVE = {f.name for f in fm.fontManager.ttflist}
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [n for n in ("Arial", "Liberation Sans", "DejaVu Sans")
                                   if n in _HAVE] or ["DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
FS = 9.0

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as _C
DS = [("validation_ts", "TS", "Tabula Sapiens (human)"),
      ("validation_tm", "TM", "Tabula Muris Senis (mouse)")]


CM = "#2f7fb5"
RF = "#d9772e"
AGREE, GROUND = "#c6c6c6", "#8d8d8d"
ERROR, DECL, VOID = "#a8382b", "#5f9150", "#f2f2f2"
PALE = "#dcdcdc"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"
MK = {"TS": "^", "TM": "o"}
SEP_MARGIN, SEP_Z = 0.10, 3.0
REL = [("same", "identical", AGREE),
       ("cellman_deeper", "CeLLMan deeper", CM),
       ("ref_deeper", "CeLLMan coarser", GROUND),
       ("conflict", "conflict", ERROR),
       ("unknown", "declined", DECL)]


def load():
    rel, rep, rpt = [], [], []
    for d, tag, _ in DS:
        organs = [l.strip() for l in open(f"{HERE}/{d}/organs.txt") if l.strip()]
        r = pd.read_csv(f"{_C.tables(d)}/relation_summary.csv")
        r = r[r.organ.isin(organs)].copy(); r["ds"] = tag
        rel.append(r)
        for f in sorted(glob.glob(f"{_C.tables(d)}/replication_*.csv")):
            organ = os.path.basename(f)[len("replication_"):-4]
            if organ not in organs:
                continue
            t = pd.read_csv(f); t["ds"], t["organ"] = tag, organ
            rep.append(t)
        for f in sorted(glob.glob(f"{_C.tables(d)}/organ_report_*.json")):
            organ = os.path.basename(f)[len("organ_report_"):-5]
            if organ not in organs:
                continue
            j = json.load(open(f))
            row = dict(ds=tag, organ=organ, n_cells=j["n_cells"], **j["B_extend"],
                       **{f"A_{k}": v for k, v in j["A_preserve"].items()})
            for kind in ("deeper", "coarser"):
                g = (j.get("C_ground") or {}).get(kind) or {}
                for k in ("parts", "tested", "replicated", "not_testable"):
                    row[f"C_{kind}_{k}"] = g.get(k, 0)
            rpt.append(row)
    return (pd.concat(rel, ignore_index=True),
            pd.concat(rep, ignore_index=True) if rep else pd.DataFrame(),
            pd.DataFrame(rpt))


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(INK3); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=FS * 0.92, length=3)


def title(ax, letter, text, sub=None, wrap=78):
    if sub:
        sub = "\n".join(textwrap.wrap(sub, wrap))
    pad = 6 + 11 * (sub.count("\n") + 1 if sub else 0)
    ax.set_title(f"{letter}   {text}", fontsize=FS * 1.16, color=INK, loc="left", pad=pad)
    if sub:
        ax.text(0.0, 1.012, sub, transform=ax.transAxes, fontsize=FS * 0.85, color=INK2,
                ha="left", va="bottom", linespacing=1.45)


def panel_a(ax, rel):
    ys = []
    for i, (_, tag, _) in enumerate(DS):
        i = len(DS) - 1 - i
        s = rel[rel.ds == tag]; n = s.n_cells.sum(); left = 0.0; ys.append(i)
        for k, lab, c in REL:
            v = 100 * s[k].sum() / n
            ax.barh(i, v, left=left, height=0.5, color=c, edgecolor="#ffffff", lw=0.9,
                    label=lab if i == 0 else None, zorder=3)
            if v > 5.5:
                ax.text(left + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=FS * 0.88,
                        color="#ffffff" if c in (CM, GROUND, ERROR, DECL) else INK, zorder=5)
            left += v
        anc = 100 * (s["same"].sum() + s["cellman_deeper"].sum()) / n
        ax.plot([anc, anc], [i - 0.33, i + 0.33], color=INK, lw=1.6, zorder=6)
        ax.text(anc, i + 0.40, f"ancestor-consistent {anc:.1f}%", fontsize=FS * 0.88, color=INK,
                ha="center", va="bottom", zorder=6)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{t}\n{int(rel[rel.ds == t].n_cells.sum()):,} cells" for _, t, _ in DS],
                       fontsize=FS * 0.98, color=INK)
    ax.set_ylim(-0.6, len(DS) - 0.15)
    ax.set_xlim(0, 100); ax.set_xlabel("% of cells", fontsize=FS, color=INK2)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=5, frameon=False,
              fontsize=FS * 0.86, labelcolor=INK2, handlelength=1.1, columnspacing=1.0)
    title(ax, "a", "CeLLMan preserves the author's call",
          "grey is the author's call, blue is what CeLLMan added below it; `conflict` is the "
          "only class in which the two disagree")


def panel_b(ax, rel, letter="b"):
    for _, tag, _ in DS:
        s = rel[rel.ds == tag]
        x = 100 * s["same"] / s.n_cells
        y = 100 * (s["same"] + s["cellman_deeper"]) / s.n_cells
        ax.scatter(x, y, s=14 + 150 * np.sqrt(s.n_cells / rel.n_cells.max()), marker=MK[tag],
                   facecolor=CM if tag == "TM" else "none", edgecolor=CM, lw=1.2, alpha=0.8,
                   zorder=3, label=f"{tag}  ({len(s)} organs)")
    ax.plot([0, 100], [0, 100], color=INK3, lw=0.9, ls="--", zorder=2)
    ax.text(76, 68, "nothing refined", fontsize=FS * 0.82, color=INK2, rotation=40)
    x = 100 * rel["same"] / rel.n_cells
    y = 100 * (rel["same"] + rel["cellman_deeper"]) / rel.n_cells
    ax.text(0.985, 0.045, f"across the 47 organs\nidentical      {x.min():.0f} – {x.max():.0f}%"
                        f"\nancestor-consistent   {y.min():.0f} – {y.max():.0f}%",
            transform=ax.transAxes, fontsize=FS * 0.86, color=INK, va="bottom", ha="right",
            linespacing=1.5)
    ax.set_xlim(0, 100); ax.set_ylim(0, 102)
    ax.set_xlabel("% identical to the author's label", fontsize=FS, color=INK2)
    ax.set_ylabel("% ancestor-consistent", fontsize=FS, color=INK2)
    ax.legend(loc="lower left", fontsize=FS * 0.86, frameon=False, labelcolor=INK2)
    title(ax, letter, "Exact match measures the reference, not the annotation",
          "one point per organ, area proportional to cells; the vertical spread at any x is "
          "how much CeLLMan added below the author's stopping point")


def panel_c(ax, rpt):
    for _, tag, _ in DS:
        s = rpt[rpt.ds == tag]
        ax.scatter(s.n_ref_labels, s.n_cm_labels,
                   s=14 + 150 * np.sqrt(s.n_cells / rpt.n_cells.max()), marker=MK[tag],
                   facecolor=CM if tag == "TM" else "none", edgecolor=CM, lw=1.2, alpha=0.8,
                   zorder=3, label=tag)
    hi = max(rpt.n_ref_labels.max(), rpt.n_cm_labels.max()) * 1.06
    ax.plot([0, hi], [0, hi], color=INK3, lw=0.9, ls="--", zorder=2)
    ax.text(hi * 0.72, hi * 0.66, "same number", fontsize=FS * 0.82, color=INK2, rotation=40)
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_xlabel("cell types the author used", fontsize=FS, color=INK2)
    ax.set_ylabel("cell types CeLLMan used", fontsize=FS, color=INK2)
    ax.legend(loc="upper left", fontsize=FS * 0.86, frameon=False, labelcolor=INK2)


    tot = rpt.cells_with_new_label.sum(); ref = rpt.of_which_refinement.sum()
    title(ax, "c", "CeLLMan uses a finer vocabulary",
          f"of {tot:,} cells with a label the author never uses, {100*ref/max(tot,1):.0f}% sit in a "
          "refinement relation (the rest are conflicts and are not counted as extension)")


def panel_d(ax, rep, letter="d"):
    d = rep[rep.recall.notna()].copy()
    d["margin"] = d.recall - d["null"]


    cmap = plt.get_cmap("Greys")
    for kind, mk, c, lab in (("deeper", "o", CM, "CeLLMan split it"),
                             ("coarser", "s", RF, "the author split it")):
        k = d[d.kind == kind]
        ok = k[k.replicates == True]                                          # noqa: E712
        ax.scatter(ok.margin, ok.z, s=13, marker=mk, facecolor=c, edgecolor=c, lw=0.6,
                   alpha=0.75, zorder=3, label=f"{lab} — replicates")
    no = d[d.replicates == False]                                             # noqa: E712
    sc = None
    for kind, mk in (("deeper", "o"), ("coarser", "s")):
        m = no[no.kind == kind]
        if not len(m):
            continue
        sc = ax.scatter(m.margin, m.z, s=17, marker=mk,
                        c=m.donor_hhi.fillna(0.0), cmap=cmap, vmin=0.0, vmax=1.0,
                        edgecolor=INK2, lw=0.6, zorder=4)
    if sc is not None:
        cax = ax.inset_axes([0.055, 0.60, 0.017, 0.26])
        cb = ax.figure.colorbar(sc, cax=cax)
        cb.set_ticks([0.0, 1.0]); cb.set_ticklabels(["0", "1"])
        cb.ax.yaxis.set_ticks_position("left")
        cb.ax.tick_params(labelsize=FS * 0.72, colors=INK2, length=2)
        cb.outline.set_edgecolor(INK3); cb.outline.set_linewidth(0.6)
        ax.text(0.088, 0.885, "donor HHI\nof a failure", transform=ax.transAxes,
                fontsize=FS * 0.76, color=INK2, va="top", linespacing=1.4)
        hhi = no.donor_hhi.dropna()
        ax.text(0.985, 0.055, f"{len(no)} failures — donor HHI median {hhi.median():.2f}, "
                              f"{int((hhi > 0.5).sum())} above 0.5 (one donor)",
                transform=ax.transAxes, fontsize=FS * 0.80, color=INK2, ha="right")
    zmax = float(np.nanpercentile(d.z, 99.5)) * 1.05
    ax.axvline(SEP_MARGIN, color=INK, lw=1.0, zorder=2)
    ax.axhline(SEP_Z, color=INK, lw=1.0, zorder=2)
    ax.add_patch(plt.Rectangle((SEP_MARGIN, SEP_Z), 1.0, zmax * 2, facecolor=CM, alpha=0.045,
                               zorder=1))
    ax.text(0.995, 0.97, "replicates", transform=ax.transAxes, fontsize=FS * 0.86, color=INK, ha="right", va="top")
    ax.set_xlim(-0.06, float(d.margin.max()) * 1.05); ax.set_ylim(-1.5, zmax)
    ax.set_xlabel("balanced accuracy above the permuted-label null", fontsize=FS, color=INK2)
    ax.set_ylabel("z against that null", fontsize=FS, color=INK2)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc=CM, mec=CM, ms=5,
                              label="CeLLMan split it — replicates"),
                       Line2D([], [], marker="s", ls="", mfc=RF, mec=RF, ms=5,
                              label="the author split it — replicates"),
                       Line2D([], [], marker="o", ls="", mfc="#9a9a9a", mec=INK2, ms=5,
                              label="does not replicate, filled by donor HHI")],
              loc="upper left", fontsize=FS * 0.82, frameon=False, labelcolor=INK2,
              handletextpad=0.4, bbox_to_anchor=(0.31, 0.985))
    title(ax, letter, "Each split, tested on cells that did not define it",
          "train each sub-type's profile on one donor/assay side and classify the other; KG "
          "marker genes are removed from the feature space, so the engine's own decision rule "
          "cannot leak in")


def panel_e(ax, rep):
    rows, labels = [], []
    for _, tag, _ in DS:
        for kind, lab in (("deeper", "CeLLMan split it"), ("coarser", "the author split it")):
            k = rep[(rep.ds == tag) & (rep.kind == kind)]
            rows.append((int((k.replicates == True).sum()), int((k.replicates == False).sum()),
                         int(k.replicates.isna().sum())))
            labels.append(f"{tag}\n{lab}")
    y = np.arange(len(rows))[::-1]
    cols = [(CM, "replicates"), (PALE, "does not replicate"), ("#f2f2f2", "not testable")]
    for i, (r, yy) in enumerate(zip(rows, y)):
        left = 0
        for v, (c, lab) in zip(r, cols):
            ax.barh(yy, v, left=left, height=0.58, color=c, edgecolor=INK3, lw=0.6,
                    label=lab if i == 0 else None, zorder=3)
            if v:
                ax.text(left + v / 2, yy, str(v), ha="center", va="center", fontsize=FS * 0.86,
                        color="#ffffff" if c == CM else INK, zorder=4)
            left += v
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=FS * 0.90, color=INK)
    ax.set_xlabel("cell-type blocks", fontsize=FS, color=INK2)
    ax.legend(loc="lower right", fontsize=FS * 0.84, frameon=False, labelcolor=INK2)
    title(ax, "e", "Both directions of the same question",
          "'the author split it' replicating means CeLLMan floored too high there")


def panel_f(ax, rep, rpt):
    for _, tag, _ in DS:
        s = rpt[rpt.ds == tag]
        for _, r in s.iterrows():
            k = rep[(rep.ds == tag) & (rep.organ == r.organ) & (rep.kind == "coarser")]
            dec = k[k.replicates.notna()]
            if not len(dec):
                continue
            ax.scatter(r.A_ref_deeper + r.A_unknown, 100 * dec.replicates.mean(),
                       s=14 + 150 * np.sqrt(r.n_cells / rpt.n_cells.max()), marker=MK[tag],
                       facecolor=RF if tag == "TM" else "none", edgecolor=RF, lw=1.2, alpha=0.8,
                       zorder=3)
    pooled = {}
    for _, tag, _ in DS:
        k = rep[(rep.ds == tag) & (rep.kind == "coarser")]
        dec = k[k.replicates.notna()]
        pooled[tag] = (int(dec.replicates.sum()), len(dec), int(k.replicates.isna().sum()))
    ax.axhline(50, color=INK3, lw=0.9, ls="--", zorder=2)
    ax.set_ylim(-4, 104)
    ax.set_xlabel("% of the organ CeLLMan left coarser or declined", fontsize=FS, color=INK2)
    ax.set_ylabel("% of those splits that replicate\n(CeLLMan floored too high)", fontsize=FS,
                  color=INK2)
    ax.legend(handles=[Line2D([], [], marker=MK[t], ls="", mfc=RF if t == "TM" else "none",
                              mec=RF, ms=6,
                              label=f"{t}   {pooled[t][0]}/{pooled[t][1]} blocks, "
                                    f"{pooled[t][2]} not testable")
                       for _, t, _ in DS],
              loc="lower right", fontsize=FS * 0.84, frameon=False, labelcolor=INK2)
    title(ax, "f", "Where CeLLMan stopped, and whether it should have",
          "high on this axis is a MISS, not a virtue: the author's finer split survived a "
          "held-out test that CeLLMan's floor implicitly denied")


def ledger(rel, rep, rpt, out):
    rows = []
    for _, tag, _ in DS:
        s = rpt[rpt.ds == tag].merge(rel[rel.ds == tag][["organ"] + [k for k, _, _ in REL]],
                                     on="organ")
        s = s.sort_values("A_ancestor_consistent", ascending=False)
        rows += [(tag, r) for _, r in s.iterrows()]
    n = len(rows)
    H = 1.7 + 0.213 * n
    fig = plt.figure(figsize=(13.0, H))
    gs = fig.add_gridspec(1, 5, width_ratios=[2.45, 0.80, 0.72, 1.30, 0.80], wspace=0.34,
                          left=0.140, right=0.988, top=1 - 0.95 / H, bottom=1.05 / H)
    y = np.arange(n)[::-1]
    lab = [f"{t}  {r.organ.replace('_', ' ')}" for t, r in rows]

    axA = fig.add_subplot(gs[0])
    for i, (_, r) in enumerate(rows):
        left = 0.0
        for k, _l, c in REL:
            v = 100 * r[k] / r.n_cells
            axA.barh(y[i], v, left=left, height=0.72, color=c, edgecolor="#ffffff", lw=0.5,
                     zorder=3)
            left += v
    axA.set_yticks(y); axA.set_yticklabels(lab, fontsize=FS * 0.80, color=INK)
    axA.set_xlim(0, 100); axA.set_xlabel("% of cells", fontsize=FS * 0.9, color=INK2)
    axA.set_title("a  PRESERVE", fontsize=FS, color=INK, loc="left", pad=6)

    axB = fig.add_subplot(gs[1], sharey=axA)
    for i, (_, r) in enumerate(rows):
        axB.plot([r.n_ref_labels, r.n_cm_labels], [y[i]] * 2, color=INK3, lw=0.8, zorder=2)
        axB.scatter([r.n_ref_labels], [y[i]], s=11, color=GROUND, zorder=3)
        axB.scatter([r.n_cm_labels], [y[i]], s=11, color=CM, zorder=3)
    axB.set_xlabel("cell types used", fontsize=FS * 0.9, color=INK2)
    axB.set_title("b  EXTEND", fontsize=FS, color=INK, loc="left", pad=6)


    axR = fig.add_subplot(gs[2], sharey=axA)
    pooled = 100 * rpt.of_which_refinement.sum() / max(rpt.cells_with_new_label.sum(), 1)
    for i, (_, r) in enumerate(rows):
        if r.cells_with_new_label:
            axR.barh(y[i], 100 * r.of_which_refinement / r.cells_with_new_label, height=0.72,
                     color=CM, zorder=3)
    axR.axvline(pooled, color=INK, lw=1.0, ls=(0, (3, 2)), zorder=4)
    axR.text(pooled + 3, n * 0.62, f"{pooled:.0f}% pooled", fontsize=FS * 0.76, color=INK,
             ha="left", va="center", rotation=90)
    axR.set_xlim(0, 100)
    axR.set_xlabel("% of new-label\ncells refining", fontsize=FS * 0.9, color=INK2,
                   linespacing=1.4)
    axR.set_title("c  is it extension?", fontsize=FS, color=INK, loc="left", pad=6)


    axC = fig.add_subplot(gs[3], sharey=axA)
    for i, (_, r) in enumerate(rows):
        for dy, kind, c in ((0.19, "deeper", CM), (-0.19, "coarser", RF)):
            rep_, tes, nt = (r[f"C_{kind}_replicated"], r[f"C_{kind}_tested"],
                             r[f"C_{kind}_not_testable"])
            left = 0
            for v, col in ((rep_, c), (tes - rep_, PALE), (nt, VOID)):
                axC.barh(y[i] + dy, v, left=left, height=0.34, color=col, edgecolor=INK3, lw=0.35,
                         zorder=3)
                left += v
    axC.set_xlabel("blocks tested", fontsize=FS * 0.9, color=INK2)
    axC.set_title("d  GROUND — both directions", fontsize=FS, color=INK, loc="left", pad=6)

    axD = fig.add_subplot(gs[4], sharey=axA)
    for i, (_, r) in enumerate(rows):
        left = 0.0
        for v, c in ((r.A_ref_deeper, GROUND), (r.A_unknown, DECL)):
            axD.barh(y[i], v, left=left, height=0.72, color=c, edgecolor="#ffffff", lw=0.4,
                     zorder=3)
            left += v
    axD.set_xlabel("% coarser / declined", fontsize=FS * 0.9, color=INK2)
    axD.set_title("e  ABSTAIN", fontsize=FS, color=INK, loc="left", pad=6)

    for ax in (axA, axB, axR, axC, axD):
        style(ax)
        ax.set_ylim(-0.8, n - 0.2)
    for ax in (axB, axR, axC, axD):
        plt.setp(ax.get_yticklabels(), visible=False)

    fig.legend(handles=[Patch(facecolor=c, label=l) for _, l, c in REL],
               loc="lower left", ncol=5, frameon=False, fontsize=FS * 0.82, labelcolor=INK2,
               title="a, e   relation of the two labels", title_fontsize=FS * 0.82,
               columnspacing=1.0, bbox_to_anchor=(0.075, 0.004))
    fig.legend(handles=[Patch(facecolor=GROUND, label="the author"),
                        Patch(facecolor=CM, label="CeLLMan")],
               loc="lower left", ncol=2, frameon=False, fontsize=FS * 0.82, labelcolor=INK2,
               title="b   vocabulary", title_fontsize=FS * 0.82, bbox_to_anchor=(0.520, 0.004))
    fig.legend(handles=[Patch(facecolor=CM, label="CeLLMan's split replicates"),
                        Patch(facecolor=RF, label="the author's does"),
                        Patch(facecolor=PALE, label="does not"),
                        Patch(facecolor=VOID, edgecolor=INK3, label="not testable")],
               loc="lower left", ncol=4, frameon=False, fontsize=FS * 0.82, labelcolor=INK2,
               title="d   held-out verdict, upper bar = CeLLMan's direction",
               title_fontsize=FS * 0.82, columnspacing=1.0, bbox_to_anchor=(0.660, 0.004))
    fig.suptitle("Per-organ ledger — every organ of both atlases, sorted by ancestor-consistency",
                 fontsize=FS * 1.12, color=INK, x=0.140, ha="left", y=0.997)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}/validation_organs.{ext}", dpi=300, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)


def merged(rel, rep, rpt, out):
    rows = []
    for _, tag, _ in DS:
        s = rpt[rpt.ds == tag].merge(rel[rel.ds == tag][["organ"] + [k for k, _, _ in REL]],
                                     on="organ")
        rows += [(tag, r) for _, r in s.sort_values("A_ancestor_consistent",
                                                    ascending=False).iterrows()]
    n = len(rows)
    y = np.arange(n)[::-1]
    lab = [f"{t}  {r.organ.replace('_', ' ')}" for t, r in rows]
    pooled = 100 * rpt.of_which_refinement.sum() / max(rpt.cells_with_new_label.sum(), 1)

    H = 7.0 + 0.213 * n
    fig = plt.figure(figsize=(13.2, H))
    top = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.28], wspace=0.20,
                           left=0.140, right=0.988, top=1 - 1.05 / H, bottom=1 - 4.35 / H)
    axA = fig.add_subplot(top[0]); axB = fig.add_subplot(top[1])
    for ax in (axA, axB):
        style(ax)
    panel_b(axA, rel, letter="a")
    panel_d(axB, rep, letter="b")
    ledger_block(fig, rel, rep, rpt, rows, n, y, lab, pooled,
                 top=1 - 5.25 / H, bottom=1.90 / H, letters=("c", "d", "e", "f", "g", "g"))
    foot = (f"{len(rpt)} organs, {int(rel.n_cells.sum()):,} cells "
            f"({', '.join(f'{t} {int(rel[rel.ds == t].n_cells.sum()):,}' for _, t, _ in DS)}).  "
            "Method and the reasoning behind every choice: VALIDATION_METHOD.md.  Reference "
            "labels are the atlases' own `cell_type` after the two Table S2 corrections.  A "
            "failure to replicate withholds judgement rather than convicting (\u00a76) and is "
            "drawn with its donor concentration in b; the conflicts include known KG hierarchy "
            "defects (\u00a78), which are not yet decomposed.")
    fig.text(0.140, 0.16 / H, "\n".join(textwrap.wrap(foot, 150)), fontsize=FS * 0.80,
             color=INK2, ha="left", va="bottom", linespacing=1.5)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}/validation_main.{ext}", dpi=300, facecolor="white")
    plt.close(fig)


def ledger_block(fig, rel, rep, rpt, rows, n, y, lab, pooled, *, top, bottom,
                 letters=("c", "d", "e", "f", "g", "h"), conflict=None):
    H = fig.get_figheight()
    W = [2.45, 0.80, 0.72, 1.30] + ([1.05] if conflict is not None else []) + [0.80]
    NC = len(W)
    gs = fig.add_gridspec(2, NC, width_ratios=W, wspace=0.34, height_ratios=[3.4, n],
                          hspace=0.028, left=0.140, right=0.988,
                          top=top, bottom=bottom)
    P = [fig.add_subplot(gs[0, j]) for j in range(NC)]
    L = [fig.add_subplot(gs[1, j]) for j in range(NC)]


    yp = [1, 0]
    tick = []
    for i, (_, tag, _) in enumerate(DS):
        q = rpt[rpt.ds == tag]; nc = q.n_cells.sum(); k = rep[rep.ds == tag]
        srel = rel[rel.ds == tag]
        anc = 100 * (srel["same"].sum() + srel["cellman_deeper"].sum()) / nc
        tick.append(f"{tag}   {int(nc):,} cells\nancestor-consistent {anc:.1f}%")
        left = 0.0
        for kk, _l, c in REL:
            v = 100 * srel[kk].sum() / nc
            P[0].barh(yp[i], v, left=left, height=0.62, color=c, edgecolor="#ffffff", lw=0.8,
                      zorder=3)
            if v > 5.5:
                P[0].text(left + v / 2, yp[i], f"{v:.0f}", ha="center", va="center",
                          fontsize=FS * 0.82, zorder=4,
                          color="#ffffff" if c in (CM, GROUND, ERROR, DECL) else INK)
            left += v
        P[0].plot([anc, anc], [yp[i] - 0.36, yp[i] + 0.36], color=INK, lw=1.6, zorder=5)

        P[1].text(0.5, yp[i], f"{int(q.n_ref_labels.sum())} \u2192 {int(q.n_cm_labels.sum())}",
                  ha="center", va="center", fontsize=FS * 0.84, color=INK)

        fr = 100 * q.of_which_refinement.sum() / max(q.cells_with_new_label.sum(), 1)
        P[2].barh(yp[i], fr, height=0.62, color=CM, zorder=3)
        P[2].text(fr - 3, yp[i], f"{fr:.0f}", ha="right", va="center", fontsize=FS * 0.82,
                  color="#ffffff", zorder=4)

        for dy, kind, col in ((0.22, "deeper", CM), (-0.22, "coarser", RF)):
            kk = k[k.kind == kind]
            trio = (int((kk.replicates == True).sum()), int((kk.replicates == False).sum()),
                    int(kk.replicates.isna().sum()))                      # noqa: E712
            left = 0
            for v, c in zip(trio, (col, PALE, VOID)):
                P[3].barh(yp[i] + dy, v, left=left, height=0.38, color=c, edgecolor=INK3,
                          lw=0.4, zorder=3)
                if v > 28:
                    P[3].text(left + v / 2, yp[i] + dy, str(v), ha="center", va="center",
                              fontsize=FS * 0.74, zorder=4,
                              color="#ffffff" if c in (CM, RF) else INK)
                left += v

        if conflict is not None:
            tot = sum(sum(v.values()) for (t2, _o), v in conflict.items() if t2 == tag)
            left = 0.0
            for key, col in (("cellman", CM), ("reference", GROUND), ("undecided", PALE)):
                v = 100 * sum(x.get(key, 0) for (t2, _o), x in conflict.items()
                              if t2 == tag) / max(tot, 1)
                P[4].barh(yp[i], v, left=left, height=0.62, color=col, edgecolor="#ffffff",
                          lw=0.8, zorder=3)
                if v > 9:
                    P[4].text(left + v / 2, yp[i], f"{v:.0f}", ha="center", va="center",
                              fontsize=FS * 0.82, zorder=4,
                              color="#ffffff" if col in (CM, GROUND) else INK)
                left += v

        left = 0.0
        for v, c in ((100 * srel["ref_deeper"].sum() / nc, GROUND),
                     (100 * srel["unknown"].sum() / nc, DECL)):
            P[-1].barh(yp[i], v, left=left, height=0.62, color=c, edgecolor="#ffffff", lw=0.6,
                      zorder=3)
            left += v
        P[-1].text(left + 0.6, yp[i], f"{left:.1f}", ha="left", va="center", fontsize=FS * 0.82,
                  color=INK)

    for j, ax in enumerate(P):
        ax.set_ylim(-0.85, 1.85)
        ax.set_yticks([]); ax.set_xticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if j in (0, 2) or (conflict is not None and j == 4):
            ax.set_xlim(0, 100)
        elif j == 1:
            ax.set_xlim(0, 1)
        elif j == NC - 1:
            ax.set_xlim(0, 52)
    P[0].set_yticks(yp); P[0].set_yticklabels(tick, fontsize=FS * 0.78, color=INK,
                                              linespacing=1.35)
    P[0].tick_params(length=0)
    heads = [f"{letters[0]}  PRESERVE", f"{letters[1]}  EXTEND",
             f"{letters[2]}  is it extension?", f"{letters[3]}  GROUND \u2014 both directions"]
    if conflict is not None:
        heads.append(f"{letters[4]}  CONFLICT \u2014 who is right")
    heads.append(f"{letters[-1]}  ABSTAIN")
    for j, t in enumerate(heads):
        P[j].set_title(t, fontsize=FS, color=INK, loc="left", pad=6)
    P[1].text(0.5, 1.72, "cell types, summed over organs", ha="center", va="bottom",
              fontsize=FS * 0.74, color=INK2)
    P[3].text(0.0, 1.72, "upper bar: CeLLMan's direction", ha="left", va="bottom",
              fontsize=FS * 0.74, color=INK2)


    for i, (_, r) in enumerate(rows):
        left = 0.0
        for kk, _l, c in REL:
            v = 100 * r[kk] / r.n_cells
            L[0].barh(y[i], v, left=left, height=0.72, color=c, edgecolor="#ffffff", lw=0.5,
                      zorder=3)
            left += v
        L[1].plot([r.n_ref_labels, r.n_cm_labels], [y[i]] * 2, color=INK3, lw=0.8, zorder=2)
        L[1].scatter([r.n_ref_labels], [y[i]], s=11, color=GROUND, zorder=3)
        L[1].scatter([r.n_cm_labels], [y[i]], s=11, color=CM, zorder=3)
        if r.cells_with_new_label:
            L[2].barh(y[i], 100 * r.of_which_refinement / r.cells_with_new_label, height=0.72,
                      color=CM, zorder=3)
        for dy, kind, c in ((0.19, "deeper", CM), (-0.19, "coarser", RF)):
            rp, te, nt = (r[f"C_{kind}_replicated"], r[f"C_{kind}_tested"],
                          r[f"C_{kind}_not_testable"])
            left = 0
            for v, col in ((rp, c), (te - rp, PALE), (nt, VOID)):
                L[3].barh(y[i] + dy, v, left=left, height=0.34, color=col, edgecolor=INK3,
                          lw=0.35, zorder=3)
                left += v
        if conflict is not None:
            x = conflict.get((rows[i][0], r.organ), {})
            tot = sum(x.values())
            left = 0.0
            for key, col in (("cellman", CM), ("reference", GROUND), ("undecided", PALE)):
                if not tot:
                    break
                v = 100 * x.get(key, 0) / tot
                L[4].barh(y[i], v, left=left, height=0.72, color=col, edgecolor="#ffffff",
                          lw=0.4, zorder=3)
                left += v

        left = 0.0
        for v, c in ((r.A_ref_deeper, GROUND), (r.A_unknown, DECL)):
            L[-1].barh(y[i], v, left=left, height=0.72, color=c, edgecolor="#ffffff", lw=0.4,
                      zorder=3)
            left += v

    L[2].axvline(pooled, color=INK, lw=1.0, ls=(0, (3, 2)), zorder=4)
    L[2].text(pooled + 3, n * 0.62, f"{pooled:.0f}% pooled", fontsize=FS * 0.76, color=INK,
              ha="left", va="center", rotation=90)
    L[0].set_yticks(y); L[0].set_yticklabels(lab, fontsize=FS * 0.80, color=INK)
    L[0].set_xlim(0, 100); L[2].set_xlim(0, 100); L[-1].set_xlim(0, 52)
    if conflict is not None:
        L[4].set_xlim(0, 100)
    xls = ["% of cells", "cell types used", "% of new-label\ncells refining",
           "blocks tested"] + (["% of depth-matched\nconflict cells"] if conflict is not None
                               else []) + ["% coarser / declined"]
    for ax, xl in zip(L, xls):
        style(ax)
        ax.set_ylim(-0.8, n - 0.2)
        ax.set_xlabel(xl, fontsize=FS * 0.9, color=INK2, linespacing=1.4)
    for ax in L[1:]:
        plt.setp(ax.get_yticklabels(), visible=False)

    for j in range(NC):
        P[j].axhline(-0.80, color=INK3, lw=0.9, zorder=6, clip_on=False)


    fig.legend(handles=[Patch(facecolor=c, label=l) for _, l, c in REL],
               loc="lower left", ncol=5, frameon=False, fontsize=FS * 0.82, labelcolor=INK2,
               title="c, g   relation of the two labels", title_fontsize=FS * 0.82,
               columnspacing=1.0, bbox_to_anchor=(0.075, 0.72 / H))
    fig.legend(handles=[Patch(facecolor=GROUND, label="the author"),
                        Patch(facecolor=CM, label="CeLLMan")],
               loc="lower left", ncol=2, frameon=False, fontsize=FS * 0.82, labelcolor=INK2,
               title="d   vocabulary", title_fontsize=FS * 0.82,
               bbox_to_anchor=(0.520, 0.72 / H))
    fig.legend(handles=[Patch(facecolor=CM, label="CeLLMan's split replicates"),
                        Patch(facecolor=RF, label="the author's does"),
                        Patch(facecolor=PALE, label="does not"),
                        Patch(facecolor=VOID, edgecolor=INK3, label="not testable")],
               loc="lower left", ncol=4, frameon=False, fontsize=FS * 0.82, labelcolor=INK2,
               title="f   held-out verdict", title_fontsize=FS * 0.82, columnspacing=1.0,
               bbox_to_anchor=(0.660, 0.72 / H))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_C.FIG))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rel, rep, rpt = load()
    print(f"loaded  {len(rpt)} organ reports, {len(rep)} tested blocks, "
          f"{int(rel.n_cells.sum()):,} cells")

    merged(rel, rep, rpt, args.out)
    print(f"-> {args.out}/validation_main.png")

    ledger(rel, rep, rpt, args.out)
    print(f"-> {args.out}/validation_organs.png")

    head = []
    for _, tag, name in DS:
        r, k, q = rel[rel.ds == tag], rep[rep.ds == tag], rpt[rpt.ds == tag]
        n = r.n_cells.sum()
        d, c = k[k.kind == "deeper"], k[k.kind == "coarser"]
        head.append(dict(atlas=tag, dataset=name, organs=len(q), cells=int(n),
                         **{f"pct_{x}": round(100 * r[x].sum() / n, 2) for x, _, _ in REL},
                         pct_ancestor_consistent=round(
                             100 * (r["same"].sum() + r["cellman_deeper"].sum()) / n, 2),
                         ref_labels=int(q.n_ref_labels.sum()), cm_labels=int(q.n_cm_labels.sum()),
                         cells_new_label=int(q.cells_with_new_label.sum()),
                         of_which_refinement=int(q.of_which_refinement.sum()),
                         deeper_blocks=len(d), deeper_tested=int(d.replicates.notna().sum()),
                         deeper_replicated=int((d.replicates == True).sum()),
                         coarser_blocks=len(c), coarser_tested=int(c.replicates.notna().sum()),
                         coarser_replicated=int((c.replicates == True).sum())))
    pd.DataFrame(head).to_csv(f"{args.out}/validation_headline.csv", index=False)
    print(f"-> {args.out}/validation_headline.csv")
    print(pd.DataFrame(head).T.to_string())


if __name__ == "__main__":
    main()
