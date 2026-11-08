#!/usr/bin/env python
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import config as C
from config import OUT, DATA, tag

plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "pdf.fonttype": 42})

V = pd.read_csv(DATA / f"villus_table{tag()}.csv")
E = V[V.volume_um3 >= C.MIN_VOLUME].copy()
E["ec_core"] = E.frac_endothelial / (1 - E.frac_STB)
E["lv"] = np.log10(E.volume_um3)


ITEMS = [("frac_STB", "STB fraction of villus (%)", "#FFD400"),
         ("frac_stroma", "Stroma fraction of villus (%)", "#00B050"),
         ("ec_core", "Endothelial fraction of villus (%)", "#FF2D2D")]
q = pd.qcut(E.lv, 5, labels=False)

fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3))
rho = []
for ax, (c, ylab, colr) in zip(axes, ITEMS):
    ax.hexbin(E.lv, E[c] * 100, gridsize=22, cmap="Greys", mincnt=1, linewidths=.2,
              edgecolors="white")
    mx = E.groupby(q).lv.median(); my = E.groupby(q)[c].median() * 100
    ax.scatter(mx, my, s=110, c=colr, edgecolor="k", lw=1.4, zorder=5)
    r = spearmanr(E.lv, E[c])[0]; rho.append(r)
    ax.set_xlabel("Villus, log10 (\u00b5m\u00b3)")
    ax.set_ylabel(ylab)
    ax.text(.97, .97, "\u03c1 = %.2f" % r, transform=ax.transAxes, ha="right", va="top",
            fontsize=13)
fig.tight_layout(w_pad=2.0)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"panelG_villus_size{tag()}.{ext}", dpi=300, bbox_inches="tight", pad_inches=.02)
print("villi", len(E), "| rho STB/stroma/EC_core vs size:", np.round(rho, 2))
