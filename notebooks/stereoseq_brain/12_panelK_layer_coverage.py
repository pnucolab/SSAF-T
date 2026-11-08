#!/usr/bin/env python
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from common import CACHE, OUT, PX

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                     "pdf.fonttype": 42, "svg.fonttype": "none"})
rng = np.random.default_rng(1)
SS_C, CB_C = "#2a5d9f", "#d9822b"
PAR = {"hippocampal pyramidal neuron"}
H = np.load(CACHE / "hippo_masks.npz"); ss, pub = H["ss"], H["pub"]
sc = np.load(CACHE / "hippo_ss_cells.npz")["sc"]
tis = ss != "none"
band = {k: H[f"mask_{k}"] for k in ("DG", "CA1", "CA3")}
band["CA"] = band["CA1"] | band["CA3"]
SS_OK = {"DG": {"dentate gyrus granule cell"},
         "CA": {"CA1 pyramidal neuron", "CA3 pyramidal neuron"} | PAR}
PB_OK = {"DG": {"GN DG"}, "CA": {"EX CA"}}


def phi(a, b):
    return float(np.corrcoef(a[tis].astype(np.float32), b[tis].astype(np.float32))[0, 1])


def score(a, lab, keys, n=500):
    nul = []
    for _ in range(n):
        while True:
            dx, dy = rng.integers(0, lab.shape[0]), rng.integers(0, lab.shape[1])
            if np.hypot(min(dx, lab.shape[0] - dx) * PX,
                        min(dy, lab.shape[1] - dy) * PX) >= 150:
                break
        nul.append(phi(a, np.isin(np.roll(np.roll(lab, dx, 0), dy, 1), list(keys))))
    return dict(r=phi(a, np.isin(lab, list(keys))), null=float(np.max(nul)),
                band_px=int(a[tis].sum()), label_px=int(np.isin(lab, list(keys))[tis].sum()))


res = {("SSAF-T", k): score(band[k], sc, g) for k, g in SS_OK.items()}
res.update({("Chen et al.", k): score(band[k], pub, g) for k, g in PB_OK.items()})
json.dump({f"{w}|{k}": v for (w, k), v in res.items()},
          open(OUT / "panelK_layer_coverage_cells.json", "w"), indent=1)
for (w, k), v in res.items():
    print(f"{w:<12}{k:<3} r {v['r']:.3f} (chance {v['null']:.3f})  "
          f"band {v['band_px']:,} px, label {v['label_px']:,} px", flush=True)

fig, a = plt.subplots(figsize=(2.5, 2.4))
xs = np.arange(2); KEYS = ("DG", "CA")
for j, (w, colr) in enumerate((("SSAF-T", SS_C), ("Chen et al.", CB_C))):
    for x, k in zip(xs, KEYS):
        v, n_ = res[(w, k)]["r"], res[(w, k)]["null"]
        xx = x + (j - .5) * .36
        a.bar(xx, v, .34, color=colr, label=w if k == "DG" else None,
              edgecolor="#333", linewidth=.4)
        a.plot([xx - .16, xx + .16], [n_, n_], color="k", lw=1.1)
        a.text(xx, v + .022, f"{v:.2f}", ha="center", fontsize=6)
        a.text(xx, n_ + .012, f"{n_:.2f}", ha="center", fontsize=4.8, color="#222")
a.set_xticks(xs); a.set_xticklabels(list(KEYS), fontsize=6.5)
a.set_ylim(0, 1.0)
a.set_ylabel("correlation with the layer band", fontsize=6.5)
a.set_xlabel("hippocampal subfield\nCA = CA1 + CA3, the reference's resolution", fontsize=6.5)
a.tick_params(labelsize=5.5)
h, l = a.get_legend_handles_labels()
h.append(Line2D([0], [0], color="k", lw=1.1)); l.append("chance level")
a.legend(h, l, fontsize=5.6, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(.5, 1.06),
         columnspacing=1.1, handlelength=1.4)
for s_ in ("top", "right"):
    a.spines[s_].set_visible(False)
fig.text(.5, -.16, "chance level = the highest correlation reached by 500 shifted copies of that map",
         fontsize=4.8, ha="center", va="top", color="#555")
for ext in ("png", "pdf", "svg"):
    fig.savefig(OUT / f"panelK_layer_coverage_cells.{ext}", dpi=600, bbox_inches="tight",
                facecolor="white")
print("->", OUT / "panelK_layer_coverage_cells.pdf")
