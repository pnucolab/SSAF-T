#!/usr/bin/env python
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C                                          # noqa: E402
from _common.plot import BLUE, title, clean, save           # noqa: E402

fig, ax = plt.subplots(figsize=(2.3, 2.3))
it = pd.read_csv(C.TAB / "identity_test.csv").sort_values("delta")
y = np.arange(len(it))
ax.hlines(y, it.d_lo, it.d_hi, color=BLUE, lw=.8)
ax.scatter(it.delta, y, s=7, color=BLUE, zorder=5, lw=0, label="all genes")
ax.scatter(it.delta_nomarker, y, s=9, facecolor="white", edgecolor="#17457c", lw=.6, zorder=6,
           label="marker genes removed")
ax.set_yticks(y)
ax.set_yticklabels([str(o) for o in it["own"]], fontsize=5.2)
ax.axvline(0, color="#444", lw=.6)
clean(ax)
ax.set_xlabel("identity advantage (nats per transcript)", labelpad=1)
ax.set_ylabel("cell type", labelpad=1)
ax.legend(fontsize=4.6, frameon=False, loc="lower right", bbox_to_anchor=(1.02, 0), ncol=1,
          handletextpad=.2, labelspacing=.2, borderaxespad=.2)
title(ax, "K", "Identity of the extra transcripts  (MERFISH liver)", pad=17)
ax.text(0, 1.015, f"molecules outside the masks, scored on the published cells' profiles;\n"
        f"{int((it.d_lo > 0).sum())}/{len(it)} above zero, "
        f"{int((it.delta_nomarker > 0).sum())}/{len(it)} without the annotation's own markers",
        transform=ax.transAxes, fontsize=4.9, color="#555", va="bottom")
save(fig, C.OUT, "K", "liver_identity")
