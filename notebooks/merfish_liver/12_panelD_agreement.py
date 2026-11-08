#!/usr/bin/env python
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C                                               # noqa: E402
from _common.plot import BLUE, title, clean, save                # noqa: E402

sim = json.load(open(C.TAB / "similarity_liver.json"))
per = pd.read_csv(C.TAB / "per_cell_jaccard_liver.csv", index_col=0)

fig, ax = plt.subplots(figsize=(1.9, 1.15))
bins = np.linspace(0, 1, 21)
ax.hist(per.jaccard.dropna(), bins=bins, histtype="stepfilled", color=BLUE, alpha=.45, lw=1,
        density=True, label=f"Jaccard: ARI {sim['ARI']:.2f}, median {sim['median_jaccard']:.2f}")
ax.hist(per.containment.dropna(), bins=bins, histtype="step", color="#0c3c78", lw=1.1,
        density=True, label=f"containment: median {sim['median_containment']:.2f}")
ax.axvline(.5, color="#999", lw=.6, ls=":")
ax.set_xlabel("overlap per published cell", labelpad=1)
ax.set_ylabel("published cells (density)", labelpad=1)
clean(ax)
ax.legend(fontsize=5, frameon=False, loc="upper left", handlelength=1.1)
title(ax, "D", "Overlap with the published cells", pad=16)
ax.set_title(f"MERFISH liver, whole section: SSAF-T assigns "
             f"{100 * sim['assigned_ssaft']:.1f}% of the molecules and\n"
             f"the published cells {100 * sim['assigned_published']:.1f}%, so Jaccard cannot "
             f"reach 1 even for a perfect match",
             loc="left", fontsize=5, color="#555", pad=4)
save(fig, C.OUT, "D", "liver_agreement")
