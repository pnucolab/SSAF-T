#!/usr/bin/env python
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C                                                  # noqa: E402
from _common.plot import BLUE, DARK, ORANGE, title, clean, save     # noqa: E402

fig = plt.figure(figsize=(2.3, 2.3))
ax = fig.add_subplot(111)
vi = (pd.read_csv(C.TAB / "vendor_identity.csv")
      .dropna(subset=["ssaft_delta"]).sort_values("ssaft_delta"))
vj = json.load(open(C.TAB / "vendor_identity.json"))
y = np.arange(len(vi))
for key, col, mk, lab, dy in [("ssaft", BLUE, "o", "taken by SSAF-T", .22),
                              ("vendor", DARK, "s", "taken by the vendor", 0),
                              ("vendor_refused", ORANGE, "D", "vendor, where SSAF-T refused", -.22)]:
    ax.hlines(y + dy, vi[f"{key}_lo"], vi[f"{key}_hi"], color=col, lw=.8)
    ax.scatter(vi[f"{key}_delta"], y + dy, s=7, marker=mk, color=col, zorder=5, lw=0, label=lab)
ax.set_yticks(y)
ax.set_yticklabels(vi.label, fontsize=5.2)
ax.axvline(0, color="#444", lw=.6)
clean(ax)
ax.set_xlabel("identity advantage (nats per transcript)", labelpad=1)
ax.set_ylabel("cell type", labelpad=1)
ax.legend(fontsize=4.6, frameon=False, loc="lower right", bbox_to_anchor=(1.02, 0), ncol=1,
          handletextpad=.2, labelspacing=.2, borderaxespad=.2)
title(ax, "J", "Identity of the extra transcripts  (Xenium kidney)", pad=17)
ax.text(0, 1.015,
        f"molecules Baysor discarded; SSAF-T > vendor in {vj['ssaft_higher']}/{vj['n_types']} "
        f"types;\nvendor where SSAF-T refused: median {vj['median_refused']:+.2f}",
        transform=ax.transAxes, fontsize=4.9, color="#555", va="bottom")
save(fig, C.OUT, "J", "kidney_identity")
