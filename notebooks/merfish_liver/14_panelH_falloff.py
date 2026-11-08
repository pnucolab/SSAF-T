#!/usr/bin/env python
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C                                      # noqa: E402
from _common.plot import BLUE, ORANGE                   # noqa: E402

FIELD = "#d9d9d9"
MAX_UM, STEP = 12.0, 0.5
OTHER = "Liu et al."

tx = pd.read_parquet(C.TAB / "liver_transcripts_3d.parquet",
                     columns=["nucleus_distance", "ssaft_cell", "ref_cell"])
S = tx.ssaft_cell.astype(str).values != ""
B = tx.ref_cell.astype(str).values != ""
d = tx.nucleus_distance.values
ok = np.isfinite(d)
print(f"{len(tx):,} molecules, {ok.sum():,} with a nucleus distance; "
      f"SSAF-T {100 * S.mean():.1f}%, reference {100 * B.mean():.1f}%", flush=True)

edges = np.arange(0, MAX_UM + STEP, STEP)
b = np.digitize(d[ok], edges) - 1
keep = (b >= 0) & (b < len(edges) - 1)
mid = edges[:-1] + STEP / 2
n = np.bincount(b[keep], minlength=len(mid))
fs = np.bincount(b[keep], weights=S[ok][keep].astype(float), minlength=len(mid)) / np.maximum(n, 1)
fb = np.bincount(b[keep], weights=B[ok][keep].astype(float), minlength=len(mid)) / np.maximum(n, 1)
tab = pd.DataFrame(dict(distance_um=mid, n_molecules=n, ssaft=fs, reference=fb))
tab.to_csv(C.OUT / "panelH_liver_mask_falloff.csv", index=False)
print(tab.iloc[[0, 2, 8, 12, 16, 20]].round(3).to_string(index=False), flush=True)

fig, ax = plt.subplots(figsize=(1.95, 2.3))
ax2 = ax.twinx()
ax2.fill_between(mid, 100 * n / n.sum(), color=FIELD, lw=0, zorder=0)
ax2.set_ylim(0, 100 * n.max() / n.sum() * 3.2)
ax2.set_yticks([])
for s_ in ("top", "right", "left"):
    ax2.spines[s_].set_visible(False)
ax.plot(mid, 100 * fs, color=BLUE, lw=1.2, label="SSAF-T", zorder=3)
ax.plot(mid, 100 * fb, color=ORANGE, lw=1.2, label=OTHER, zorder=3)
ax.set_xlim(0, MAX_UM)
ax.set_ylim(0, 100)
ax.set_zorder(ax2.get_zorder() + 1)
ax.patch.set_visible(False)
ax.set_xlabel("distance to nearest nucleus (µm)", labelpad=1)
ax.set_ylabel("transcripts assigned (%)", labelpad=1)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
h, l = ax.get_legend_handles_labels()
h.append(Patch(facecolor=FIELD, edgecolor="none"))
l.append("transcripts at that distance")
ax.legend(h, l, fontsize=5, frameon=False, loc="lower left", handlelength=1.2, labelspacing=.25)
ax.set_title("H   Assignment by nucleus distance  (MERFISH liver)", loc="left",
             fontsize=7, fontweight="bold", pad=17, color="black")
out = C.OUT / "panelH_liver_mask_falloff.png"
fig.savefig(out, dpi=400, bbox_inches="tight", facecolor="white")
fig.savefig(out.with_suffix(".pdf"), dpi=400, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("->", out.name, flush=True)
