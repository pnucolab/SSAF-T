#!/usr/bin/env python
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C                                                       # noqa: E402
from _common.plot import BLUE, ORANGE, GREY, DARK, title, clean, save    # noqa: E402


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


tx = pd.read_parquet(C.TAB / "kidney_transcripts_3d.parquet",
                     columns=["ssaft_cell", "baysor_cell", "in_baysor_universe",
                              "vendor_assigned", "nucleus_distance"])
tx = tx[tx.in_baysor_universe]
B = tx.baysor_cell.values > 0
S = tx.ssaft_cell.astype(str).values != ""

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
          ["both", "SSAF-T only", "Xuanyuan et al. only", "neither"],
          fontsize=4.8, ncol=2, frameon=False, loc="lower left", bbox_to_anchor=(-.05, 1.0),
          handlelength=.7, columnspacing=.5, handletextpad=.3)
title(ax, "G", "Transcript assignment  (Xenium kidney)", pad=17)

ax = fig.add_subplot(gf[1])
bars = [("SSAF-T", 100 * S.mean(), BLUE), ("Xuanyuan et al.", 100 * B.mean(), ORANGE),
        ("Xenium default", 100 * tx.vendor_assigned.mean(), DARK)]
for k, (nm, v, c_) in enumerate(bars):
    ax.barh(-k, v, color=c_, height=.6)
    ax.text(v + 2, -k, f"{v:.1f}%", va="center", fontsize=5.3)
ax.set_yticks([-k for k in range(len(bars))])
ax.set_yticklabels([b[0] for b in bars], fontsize=5.5)
ax.set_ylim(-2.5, .5)
ax.set_xlim(0, 120)
ax.set_xticks([0, 50, 100])
clean(ax)
ax.set_xlabel("% of transcripts assigned", labelpad=1)
print("kidney fate both/ssaft-only/ref-only/neither", [round(f, 1) for f in frac],
      "assigned", [(b[0], round(b[1], 1)) for b in bars])
save(fig, C.OUT, "G", "kidney_molecule_fate")
