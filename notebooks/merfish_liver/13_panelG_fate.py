#!/usr/bin/env python
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C                                                       # noqa: E402
from _common.plot import BLUE, ORANGE, GREY, title, clean, save          # noqa: E402


def fate_bar(ax, frac, cols):
    left = 0
    starts = np.cumsum([0.0] + frac[:-1])
    thin = [(v, st, c) for v, st, c in zip(frac, starts, cols) if v <= 6]
    for f_, c_ in zip(frac, cols):
        ax.barh(0, f_, left=left, color=c_, edgecolor="white", linewidth=.6, height=.6)
        if f_ > 6:
            ax.text(left + f_ / 2, 0, f"{f_:.1f}", ha="center", va="center", fontsize=5.5,
                    color="white" if c_ in (BLUE, ORANGE) else "#333")
        left += f_
    for k, (f_, l_, c_) in enumerate(sorted(thin, key=lambda t: t[1])):
        x = l_ + f_ / 2
        y = .55 + .30 * (k % 2)
        ax.plot([x, x], [.3, y - .05], color="#888", lw=.4, clip_on=False)
        ax.text(x, y, f"{f_:.1f}", ha="center", va="bottom", fontsize=5.2,
                color=c_ if c_ != "#efefef" else "#666", clip_on=False)


tx = pd.read_parquet(C.TAB / "liver_transcripts_3d.parquet",
                     columns=["ssaft_cell", "ref_cell", "nucleus_distance"])
S = tx.ssaft_cell.astype(str).values != ""
B = tx.ref_cell.astype(str).values != ""

fig = plt.figure(figsize=(1.75, 2.3))
gf = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=1.0)

ax = fig.add_subplot(gf[0])
frac = [100 * (S & B).mean(), 100 * (S & ~B).mean(), 100 * (~S & B).mean(), 100 * (~S & ~B).mean()]
cols = [GREY, BLUE, ORANGE, "#efefef"]
fate_bar(ax, frac, cols)
ax.set_ylim(-.45, 1.25)
ax.set_yticks([])
ax.set_xlim(0, 100)
ax.set_xticks([0, 50, 100])
for s_ in ("top", "right", "left"):
    ax.spines[s_].set_visible(False)
ax.set_xlabel("% of transcripts", labelpad=1)
ax.legend([plt.Rectangle((0, 0), 1, 1, fc=c) for c in cols],
          ["both", "SSAF-T only", "Liu et al. only", "neither"],
          fontsize=4.8, ncol=2, frameon=False, loc="lower left", bbox_to_anchor=(-.05, 1.0),
          handlelength=.7, columnspacing=.5, handletextpad=.3)
title(ax, "G", "Transcript assignment  (MERFISH liver)", pad=17)

ax = fig.add_subplot(gf[1])
for k, (nm, v, c_) in enumerate((("SSAF-T", 100 * S.mean(), BLUE),
                                 ("Liu et al.", 100 * B.mean(), ORANGE))):
    ax.barh(-k, v, color=c_, height=.6)
    ax.text(v + 2, -k, f"{v:.1f}%", va="center", fontsize=5.3)
ax.set_yticks([0, -1])
ax.set_yticklabels(["SSAF-T", "Liu et al."], fontsize=5.5)
ax.set_ylim(-1.9, .6)
ax.set_xlim(0, 120)
ax.set_xticks([0, 50, 100])
clean(ax)
ax.set_xlabel("% of transcripts assigned", labelpad=1)
print("liver fate both/ssaft-only/paper-only/neither", [round(f, 1) for f in frac],
      "assigned", round(100 * S.mean(), 1), round(100 * B.mean(), 1))
save(fig, C.OUT, "G", "liver_molecule_fate")
