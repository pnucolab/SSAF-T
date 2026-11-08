#!/usr/bin/env python
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import WORK, OUT, COLORS

CB_C = "#d9822b"

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"], "pdf.fonttype": 42})
HMAX, STEP, MIN_N = 320, 10, 40
SS = ["Paneth cell", "immature Paneth cell progenitor", "intestinal stem cell",
      "transit amplifying cell", "goblet-Paneth intermediate cell", "goblet cell",
      "immature enterocyte", "villus-bottom enterocyte", "distal enterocyte",
      "absorptive enterocyte", "villus-tip enterocyte"]
SHORT = {"Paneth cell": "Paneth", "immature Paneth cell progenitor": "imm. Paneth",
         "intestinal stem cell": "ISC", "transit amplifying cell": "TA",
         "goblet-Paneth intermediate cell": "Gob-Pan", "goblet cell": "goblet",
         "immature enterocyte": "iEnt", "villus-bottom enterocyte": "botEnt",
         "distal enterocyte": "distEnt", "absorptive enterocyte": "aEnt",
         "villus-tip enterocyte": "tipEnt"}
EE = ["Paneth cells", "Crypt cells", "Goblet cells", "Enterocytes", "Enteroendocrine cells"]


ENC = {"Paneth cells": "Paneth cell", "Goblet cells": "goblet cell",
       "Enteroendocrine cells": "enteroendocrine cell"}
ENC_FLAT = {"Crypt cells": "#8c8c8c", "Enterocytes": "#d0d0d0"}

s = pd.read_parquet(WORK / "ssaft_cells_axis_scored.parquet")
e = pd.read_parquet(WORK / "enact_cells_axis_C.parquet")
edges = np.arange(0, HMAX + STEP, STEP)
mid = edges[:-1] + STEP / 2


def stack(df, labs, colour_of):
    d = df[(df.unit > 0) & np.isfinite(df.A) & df.label.isin(labs)]
    b = np.digitize(d.A.values, edges) - 1
    ok = (b >= 0) & (b < len(mid))
    M = np.zeros((len(labs), len(mid)))
    for i, L in enumerate(labs):
        M[i] = np.bincount(b[ok & (d.label.values == L)], minlength=len(mid))[:len(mid)]
    n = M.sum(0)
    F = np.divide(M, np.where(n > 0, n, 1))
    F[:, n < MIN_N] = np.nan
    return F, n, [colour_of(L) for L in labs]


BSTEP, BMIN = 20, 5


def turnover(df, labs):
    d = df[(df.unit > 0) & np.isfinite(df.A) & df.label.isin(labs)]
    out = {}
    for u, g in d.groupby("unit"):
        bb = (g.A.values // BSTEP).astype(int)
        seq = [pd.Series(g.label.values[bb == k]).mode().iat[0]
               for k in np.unique(bb) if (bb == k).sum() >= BMIN]
        if len(seq) >= 4:
            out[u] = len(set(seq))
    return pd.Series(out)


TS, TE = turnover(s, SS), turnover(e, EE)
TJ = pd.concat([TS.rename("s"), TE.rename("e")], axis=1).dropna()
W = wilcoxon(TJ.s, TJ.e)
print(f"distinct dominant labels per unit: SSAF-T median {TS.median():.0f} "
      f"(IQR {TS.quantile(.25):.0f}-{TS.quantile(.75):.0f}, n={len(TS)}), "
      f"ENACT median {TE.median():.0f} (IQR {TE.quantile(.25):.0f}-{TE.quantile(.75):.0f}, n={len(TE)}); "
      f"SSAF-T higher in {100 * (TJ.s > TJ.e).mean():.0f}% of {len(TJ)} paired units, p = {W.pvalue:.1e}")

fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.2),
                        gridspec_kw=dict(width_ratios=[1, 1, .52], wspace=.34))
axs[1].sharey(axs[0])
for ax, (df, labs, col, ttl) in zip(axs[:2], (
        (s, SS, lambda L: COLORS.get(L, "#999"), "SSAF-T"),
        (e, EE, lambda L: ENC_FLAT.get(L) or COLORS.get(ENC.get(L, ""), "#999"), "Kamel et al."))):
    F, n, cols = stack(df, labs, col)
    keep = np.isfinite(F[0])
    ax.stackplot(mid[keep], np.nan_to_num(F[:, keep]), colors=cols, edgecolor="none")
    ax.set_xlim(0, HMAX); ax.set_ylim(0, 1)
    ax.set_xlabel("distance to muscularis (µm)", fontsize=6.5)
    ax.set_title(ttl, fontsize=7, pad=3)
    ax.tick_params(labelsize=5.5)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)

    for i, L in enumerate(labs):
        j = np.nanargmax(F[i]) if np.isfinite(F[i]).any() else None
        if j is None or F[i, j] < .12:
            continue
        y = np.nansum(F[:i, j]) + F[i, j] / 2
        nm = SHORT.get(L, L)
        x = mid[j]
        ha = "center"
        if x < 45:
            x, ha = 6, "left"
        ax.text(x, y, nm, fontsize=4.8, ha=ha, va="center", clip_on=True,
                color="white" if F[i, j] > .3 else "#222")
axs[1].tick_params(labelleft=False)
axs[0].set_ylabel("share of the epithelium in that bin", fontsize=6.5)
axs[0].text(0, -.42, f"{STEP} µm bins; bins with fewer than {MIN_N} epithelial cells are dropped",
            transform=axs[0].transAxes, fontsize=4.8, ha="left", va="top", color="#666")
ax = axs[2]
bp = ax.boxplot([TJ.s.values, TJ.e.values], positions=[0, 1], widths=.62, showfliers=False,
                patch_artist=True, medianprops=dict(color="black", lw=1),
                whiskerprops=dict(lw=.5), capprops=dict(lw=.5), boxprops=dict(lw=.5))
for b, c, h in zip(bp["boxes"], ("#2a5d9f", "#d9822b"), ("", "////")):
    b.set_facecolor(c); b.set_hatch(h); b.set_edgecolor("#333")
ax.set_xticks([0, 1]); ax.set_xticklabels(["SSAF-T", "ENACT"], fontsize=6.5)
ax.set_ylabel("distinct labels dominating\na height bin, per unit", fontsize=6)
ax.set_title("Turnover per unit", fontsize=7, pad=3); ax.tick_params(labelsize=5.5)
ax.set_ylim(0, max(TJ.s.max(), TJ.e.max()) + 1.6)
ax.text(.5, .98, f"higher in {100 * (TJ.s > TJ.e).mean():.0f}% of {len(TJ)} units\n"
        f"(Wilcoxon p < 1e-80)", transform=ax.transAxes, fontsize=5.0, ha="center", va="top",
        color="#444")
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
fig.suptitle("Epithelial composition", fontsize=8, y=1.03, x=.42)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"panel_epithelial_composition.{ext}", dpi=600, bbox_inches="tight", facecolor="white")
print("->", OUT / "panel_epithelial_composition.png")


fig = plt.figure(figsize=(3.2, 2.4))
gs = fig.add_gridspec(2, 1, height_ratios=[3.4, 1], hspace=.14)
ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
for ax, (df, labs, colf, tag) in zip((ax1, ax2), (
        (s, SS, lambda L: COLORS.get(L, "#999"), "SSAF-T"),
        (e, EE, lambda L: ENC_FLAT.get(L) or COLORS.get(ENC.get(L, ""), "#999"), "ENACT"))):
    F, n, _ = stack(df, labs, colf)
    keep = np.isfinite(F[0])
    ax.stackplot(mid[keep], np.nan_to_num(F[:, keep]), colors=[colf(L) for L in labs], edgecolor="none")
    ax.set_xlim(0, HMAX); ax.set_ylim(0, 1); ax.tick_params(labelsize=5.5)
    ax.set_yticks([0, .5, 1] if ax is ax1 else [0, 1])
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.set_ylabel(tag, fontsize=6.5, color="#2a5d9f" if ax is ax1 else CB_C, labelpad=2)
    for i, L in enumerate(labs):
        if not np.isfinite(F[i]).any():
            continue
        j2 = int(np.nanargmax(F[i]))
        if F[i, j2] < (.12 if ax is ax1 else .07):
            continue
        y = np.nansum(F[:i, j2]) + F[i, j2] / 2
        x, ha = (6, "left") if mid[j2] < 45 else (mid[j2], "center")
        ax.text(x, y, SHORT.get(L, L.replace(" cells", "")), fontsize=4.6, ha=ha, va="center",
                clip_on=True, color="white" if F[i, j2] > .3 else "#222")
ax1.tick_params(labelbottom=False)
fig.supylabel("share of the epithelium", fontsize=6, x=-.04)
ax2.set_xlabel("distance to muscularis (µm)", fontsize=6.5)
ax1.set_title("Epithelial composition", fontsize=8, pad=3)
fig.text(.5, -.12, f"labels dominating a height bin, per unit: SSAF-T {TS.median():.0f} vs ENACT "
         f"{TE.median():.0f} (SSAF-T higher in {100 * (TJ.s > TJ.e).mean():.0f}% of {len(TJ)} units)",
         fontsize=4.8, ha="center", va="top", color="#333")
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"panel_epithelial_composition_compact.{ext}", dpi=600, bbox_inches="tight",
                facecolor="white")
print("->", OUT / "panel_epithelial_composition_compact.png")
