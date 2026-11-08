from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path
from matplotlib.patches import PathPatch


_PALETTE = [
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
    "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
    "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
    "#17becf", "#9edae5",
]

RELATION_COLORS = {
    "same":           "#9e9e9e",
    "cellman_deeper": "#1f77b4",
    "ref_deeper":     "#ff7f0e",
    "conflict":       "#d62728",
    "unknown":        "#2ca02c",
}


def _ribbon(ax, x0, x1, y0, y1, h0, h1, color, alpha):
    cx = (x0 + x1) / 2.0
    verts = [
        (x0, y0), (cx, y0), (cx, y1), (x1, y1),
        (x1, y1 + h1),
        (cx, y1 + h1), (cx, y0 + h0), (x0, y0 + h0),
        (x0, y0),
    ]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.LINETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor="none",
                           alpha=alpha, linewidth=0))


def sankey(flows: pd.DataFrame, *, title: str = "", left_title: str = "CeLLMan",
           right_title: str = "Tabula Muris Senis", color_by: str = "source",
           relation_col: str = "relation", min_frac: float = 0.0,
           gap_frac: float = 0.012, bar_w: float = 0.018, alpha: float = 0.55,
           label_fs: float = 7.0, figsize=None, ax=None):
    f = flows.loc[flows["value"] > 0].copy()
    total = f["value"].sum()
    if min_frac > 0:
        f = f.loc[f["value"] >= min_frac * total]

    lsz = f.groupby("source")["value"].sum().sort_values(ascending=False)
    rsz = f.groupby("target")["value"].sum()


    lpos = {s: i for i, s in enumerate(lsz.index)}
    bary = (f.assign(w=f["source"].map(lpos) * f["value"])
              .groupby("target")[["w", "value"]].sum())
    bary = (bary["w"] / bary["value"]).sort_values()
    rsz = rsz.loc[bary.index]

    n_gap = max(len(lsz), len(rsz))
    gap = gap_frac * total
    span_l = lsz.sum() + gap * (len(lsz) - 1)
    span_r = rsz.sum() + gap * (len(rsz) - 1)
    span = max(span_l, span_r)

    def _layout(sizes, span_used):
        y = (span - span_used) / 2.0
        out = {}
        for k, v in sizes.items():
            out[k] = y
            y += v + gap
        return out

    ltop, rtop = _layout(lsz, span_l), _layout(rsz, span_r)

    if ax is None:
        h = figsize[1] if figsize else max(4.5, 0.30 * n_gap + 1.2)
        w = figsize[0] if figsize else 11.0
        fig, ax = plt.subplots(figsize=(w, h))
    else:
        fig = ax.figure

    cmap = {s: _PALETTE[i % len(_PALETTE)] for i, s in enumerate(lsz.index)}
    x0, x1 = 0.0, 1.0


    lcur = dict(ltop)
    rcur = dict(rtop)
    order = f.sort_values(["source", "value"], ascending=[True, False])
    order["_l"] = order["source"].map(lpos)
    order = order.sort_values(["_l", "value"], ascending=[True, False])
    for _, r in order.iterrows():
        col = (cmap[r["source"]] if color_by == "source"
               else RELATION_COLORS.get(r.get(relation_col, "same"), "#9e9e9e"))
        _ribbon(ax, x0 + bar_w, x1, lcur[r["source"]], rcur[r["target"]],
                r["value"], r["value"], col, alpha)
        lcur[r["source"]] += r["value"]
        rcur[r["target"]] += r["value"]


    def _declutter(centres, min_sep):
        y = np.array(centres, dtype=float)
        for _ in range(60):
            moved = False
            for i in range(1, len(y)):
                d = min_sep - (y[i] - y[i - 1])
                if d > 1e-9:
                    y[i - 1] -= d / 2
                    y[i] += d / 2
                    moved = True
            if not moved:
                break
        return y

    min_sep = span * (label_fs + 2.0) / (fig.get_size_inches()[1] * 72.0)
    for side, sizes, tops in (("l", lsz, ltop), ("r", rsz, rtop)):
        keys = list(sizes.index)
        cen = [tops[k] + sizes[k] / 2 for k in keys]
        lab_y = _declutter(cen, min_sep)
        for k, c, ly in zip(keys, cen, lab_y):
            v = sizes[k]
            if side == "l":
                ax.add_patch(plt.Rectangle((x0, tops[k]), bar_w, v, color=cmap[k], lw=0))
                ax.text(x0 - 0.030, ly, f"{k}  ({v:,.0f})", ha="right", va="center",
                        fontsize=label_fs)
                if abs(ly - c) > min_sep * 0.25:
                    ax.plot([x0 - 0.026, x0 - 0.004], [ly, c], lw=0.4, color="#999999",
                            solid_capstyle="butt", zorder=5)
            else:
                ax.add_patch(plt.Rectangle((x1, tops[k]), bar_w, v, color="#666666", lw=0))
                ax.text(x1 + bar_w + 0.030, ly, f"{k}  ({v:,.0f})", ha="left", va="center",
                        fontsize=label_fs)
                if abs(ly - c) > min_sep * 0.25:
                    ax.plot([x1 + bar_w + 0.004, x1 + bar_w + 0.026], [c, ly], lw=0.4,
                            color="#999999", solid_capstyle="butt", zorder=5)

    ax.set_xlim(-0.62, 1.62)
    ax.set_ylim(span * 1.02, -span * 0.02)
    ax.axis("off")
    ax.text(x0 + bar_w / 2, -span * 0.015, left_title, ha="center", va="bottom",
            fontsize=label_fs + 2.5, fontweight="bold")
    ax.text(x1 + bar_w / 2, -span * 0.015, right_title, ha="center", va="bottom",
            fontsize=label_fs + 2.5, fontweight="bold")
    if title:
        ax.set_title(title, fontsize=label_fs + 3.5, pad=16)
    fig.tight_layout()
    return fig, ax
