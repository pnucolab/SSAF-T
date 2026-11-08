#!/usr/bin/env python
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C                                              # noqa: E402
from _common.plot import BLUE, ORANGE, GREY, DARK               # noqa: E402

MAX_UM = 12.0
OTHER = "Liu et al."

tx = pd.read_parquet(C.TAB / "liver_transcripts_3d.parquet",
                     columns=["nucleus_distance", "ssaft_cell", "ref_cell"])
S = tx.ssaft_cell.astype(str).values != ""
B = tx.ref_cell.astype(str).values != ""
d = tx.nucleus_distance.values
ok = np.isfinite(d)
print(f"{len(tx):,} transcripts, {ok.sum():,} with a distance ({100 * ok.mean():.1f}%)", flush=True)

fig, ax = plt.subplots(figsize=(1.75, 2.3))
for nm, m, c_, ls in [("both", S & B, GREY, "-"),
                      ("SSAF-T only", S & ~B, BLUE, "-"),
                      (f"{OTHER} only", ~S & B, ORANGE, "-"),
                      ("neither", ~S & ~B, DARK, ":")]:
    v = np.sort(d[m & ok])
    med = np.median(v)
    v = v[:: max(1, len(v) // 20000)]
    ax.plot(v, 100 * np.arange(1, len(v) + 1) / len(v), color=c_, lw=1, ls=ls, label=nm)
    print(f"  {nm:22} n={int((m & ok).sum()):>10,}  median {med:5.2f} µm", flush=True)
ax.set_xlim(0, MAX_UM)
ax.set_ylim(0, 100)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
ax.set_xlabel("distance from the nearest\nnucleus (µm)", labelpad=1)
ax.set_ylabel("cumulative % of transcripts", labelpad=1)
ax.legend(fontsize=5, frameon=False, loc="lower right", labelspacing=.2, handlelength=1.2)
ax.set_title("I", loc="left", fontsize=7, fontweight="bold", pad=6, color="black")
for ext in ("png", "pdf"):
    fig.savefig(C.OUT / f"panelI_liver_recovered_where.{ext}", dpi=400,
                bbox_inches="tight", facecolor="white")
plt.close(fig)
print("-> panelI_liver_recovered_where.png", flush=True)
