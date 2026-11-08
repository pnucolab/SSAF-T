#!/usr/bin/env python
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
from matplotlib.colors import to_rgba                               # noqa: E402

import config as C                                                  # noqa: E402

matplotlib.rcParams["pdf.fonttype"] = 42
B = C.BLOCK_UM
IN_PER_MM = 0.55
FILL_STEPS = 2

NP = dict(np.load(C.TAB / f"wtniche{C.tag()}.npz"))
K = int(NP["K"])
who, lab = NP["who"], NP["lab"]
if K > len(C.NICHE_NAMES):
    raise SystemExit(f"k = {K} but only {len(C.NICHE_NAMES)} niche names and colours are defined")
comp = pd.read_csv(C.TAB / f"wtniche_comp{C.tag()}.csv")


LUT = np.array([to_rgba(c) for c in C.NICHE_COLORS[:K]] + [(0, 0, 0, 0)])


def build(section):
    d = np.load(C.grid_npz(section))
    counts = d["counts"].astype(np.float32)
    nz = d["z"][1] - d["z"][0]
    tissue = counts.sum(0) / (B * B * nz) >= C.TISSUE_FRAC
    H, W = tissue.shape

    ix, iy = NP[f"ix_{section}"], NP[f"iy_{section}"]
    L = lab[who == section]
    g = C.GRID_STEP
    nic = np.full(((H + g - 1) // g, (W + g - 1) // g), -1, int)
    nic[ix // g, iy // g] = L
    dist, (a, b) = ndi.distance_transform_edt(nic < 0, return_indices=True)
    f = nic[a, b]
    f[dist > FILL_STEPS] = -1
    f = np.repeat(np.repeat(f, g, 0), g, 1)[:H, :W]
    f[~tissue] = -1

    ys, xs = np.where(tissue.T)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return LUT[np.where(f.T < 0, K, f.T)][y0:y1, x0:x1]


SD = C.OUT / f"panelB{C.tag()}"
SD.mkdir(parents=True, exist_ok=True)
scale = lambda u: "1 mm" if u == 1000 else f"{u} um"

for g, s in C.SAMP:
    im = build(s)
    h, w = im.shape[:2]
    wmm, hmm = w * B / 1000, h * B / 1000
    fw, fh = wmm * IN_PER_MM, hmm * IN_PER_MM

    fig = plt.figure(figsize=(fw + 0.2, fh + 0.7), facecolor=C.BG)
    ax = fig.add_axes([0.1 / (fw + 0.2), 0.45 / (fh + 0.7), fw / (fw + 0.2), fh / (fh + 0.7)])
    ax.imshow(im, extent=[0, wmm, hmm, 0], interpolation="nearest")
    ax.set_xlim(0, wmm); ax.set_ylim(hmm, 0); ax.set_aspect("equal"); ax.axis("off")
    u = C.SCALEBAR_UM[s] / 1000
    ax.plot([0.1, 0.1 + u], [hmm + 0.15, hmm + 0.15], color=C.INK, lw=2.5, clip_on=False,
            solid_capstyle="butt")
    ax.text(0.1 + u / 2, hmm + 0.22, scale(C.SCALEBAR_UM[s]), ha="center", va="top", fontsize=9,
            color=C.INK, clip_on=False)
    fig.text(0.1 / (fw + 0.2), 1 - 0.12 / (fh + 0.7), s.replace("_v2", "").replace("_", " "),
             fontsize=12, fontweight="bold", color=C.GROUP_COLORS[g], va="top")
    fig.savefig(SD / f"niche_{s}.png", dpi=500, facecolor=C.BG)
    fig.savefig(SD / f"niche_{s}.pdf", dpi=500, facecolor=C.BG)
    plt.close(fig)

    fig = plt.figure(figsize=(fw, fh), facecolor=C.BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(im, extent=[0, wmm, hmm, 0], interpolation="nearest")
    ax.set_xlim(0, wmm); ax.set_ylim(hmm, 0); ax.set_aspect("auto"); ax.axis("off")
    fig.savefig(SD / f"niche_{s}_clean.png", dpi=500, facecolor=C.BG)
    plt.close(fig)
    print("saved", s, im.shape, flush=True)


fig = plt.figure(figsize=(4.4, 3.0), facecolor=C.BG)
fig.legend(handles=[plt.Line2D([], [], marker="o", ls="none", ms=9,
                               mfc=C.NICHE_COLORS[j], mec="none", label=C.NICHE_NAMES[j])
                    for j in comp.niche],
           loc="center", ncol=1, frameon=False, labelcolor=C.INK, fontsize=12,
           handletextpad=.6, labelspacing=.65)
fig.savefig(SD / "niche_legend.png", dpi=500, facecolor=C.BG)
fig.savefig(SD / "niche_legend.pdf", facecolor=C.BG)
print("->", SD)
