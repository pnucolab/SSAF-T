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

wt = pd.read_csv(C.TAB / "whole_tissue_similarity.csv")
wj = json.load(open(C.TAB / "whole_tissue_similarity.json"))

fig, ax = plt.subplots(figsize=(1.9, 1.15))
bins = np.linspace(0, 1, 21)
ax.hist(wt.jaccard, bins=bins, histtype="stepfilled", color=BLUE, alpha=.45, lw=1, density=True,
        label=f"Jaccard: median {wt.jaccard.median():.2f}")
ax.hist(wt.containment, bins=bins, histtype="step", color="#0c3c78", lw=1.1, density=True,
        label=f"containment: median {wt.containment.median():.2f}")
ax.axvline(.5, color="#999", lw=.6, ls=":")
ax.set_xlabel("overlap per published cell", labelpad=1)
ax.set_ylabel("published cells (density)", labelpad=1)
clean(ax)
ax.legend(fontsize=5, frameon=False, loc="upper left", handlelength=1.1, labelspacing=.25)
title(ax, "D", "Overlap with the published cells", pad=16)
ax.text(0, 1.015, f"Xenium kidney, whole section: {wj['n_published']:,} published cells, "
        f"{wj['n_ssaft']:,} SSAF-T cells;\n"
        f"{100 * wj['published_half_inside_one_ssaft']:.0f}% of published cells sit mostly inside "
        f"one SSAF-T cell, {100 * wj['ssaft_half_inside_one_published']:.0f}% the other way round",
        transform=ax.transAxes, fontsize=5, color="#555", va="bottom")
save(fig, C.OUT, "D", "kidney_agreement")
