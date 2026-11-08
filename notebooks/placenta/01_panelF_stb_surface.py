#!/usr/bin/env python
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib as mpl
from scipy import ndimage as ndi
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import config as C
from config import OUT, DATA, SOURCE, palette, celltype_planes, tag
_names = C.names

Z0, Z1, ZEX = C.Z0, C.Z1, C.ZEX
X0, Y0, WIN = C.WIN_X, C.WIN_Y, C.WIN
names = _names(); NL = len(names) + 1
planes = list(celltype_planes(Z0, Z1 + 1)); Yn, Xn = planes[0].shape; central = planes[ZEX - Z0]
print("source", SOURCE, "planes", Z0, "-", Z1, flush=True)
rank = np.random.default_rng(0).permutation(np.arange(1, NL))
P = np.zeros((Yn, Xn), np.uint8); best = np.zeros((Yn, Xn), np.int32)
for l in range(1, NL):
    c = np.zeros((Yn, Xn), np.int16)
    for p in planes: c += (p == l)
    s = c.astype(np.int32) * 100 + (central == l) * 50 + int(rank[l - 1]); u = (c > 0) & (s > best); best[u] = s[u]; P[u] = l
del planes, best
print("projection: labelled fraction %.3f, STB fraction %.3f" % ((P > 0).mean(), (P == 1).mean()), flush=True)
I = {n: i + 1 for i, n in enumerate(names)}
pal = palette(); col = {n: pal[i] for i, n in enumerate(names)}
STB = I["syncytiotrophoblast"]
GROUPS = [("CTB", ["cytotrophoblast"], "#1565FF"), ("endothelium", ["endothelial cell"], "#FF2D2D"),
          ("villous stroma (PVF, Fib, Mes)", ["placental villous fibroblast / mesenchymal stromal cell", "fibroblast", "mesenchymal cell"], "#00E05A"),
          ("pericyte", ["pericyte"], "#E65100"), ("smooth muscle cell", ["smooth muscle cell"], "#6D4C41"), ("Hofbauer cell", ["Hofbauer cell"], "#B026FF"),
          ("EVT (incl. interstitial)", ["extravillous trophoblast", "interstitial extravillous trophoblast"], "#5D4037")]
NONE_C = "#1A1A1A"; OTHER_C = "#BBBBBB"; STB_C = "#FFD400"
gid = np.zeros(len(names) + 1, np.int8) - 1
for g, (_, mem, _) in enumerate(GROUPS):
    for n in mem: gid[I[n]] = g

cnt = np.zeros(len(names) + 1, np.int64)
for ax in (0, 1):
    a = [slice(None)] * 2; b = [slice(None)] * 2; a[ax] = slice(None, -1); b[ax] = slice(1, None); x, y = P[tuple(a)], P[tuple(b)]
    m1 = (x == STB) & (y != STB); m2 = (y == STB) & (x != STB); cnt += np.bincount(y[m1], minlength=len(cnt)) + np.bincount(x[m2], minlength=len(cnt))
tot = cnt.sum(); sh = {}
for g, (nm, mem, _) in enumerate(GROUPS): sh[nm] = 100 * sum(cnt[I[n]] for n in mem) / tot
sh["no labelled neighbour (intervillous space / no call)"] = 100 * cnt[0] / tot
sh["other labelled (immune, myofibroblast, ...)"] = 100 - sum(sh.values())
pd.Series(sh).round(2).rename("share_of_STB_surface_pct").to_csv(DATA / f"stb_surface_shares{tag()}.csv")
print(pd.Series(sh).round(1).to_string(), "| STB faces", int(tot), flush=True)

pl = P[Y0:Y0 + WIN, X0:X0 + WIN]; stb = pl == STB
nonstb4 = ndi.binary_dilation(~stb, structure=ndi.generate_binary_structure(2, 1))
surf = stb & nonstb4; best = np.zeros(pl.shape, np.float32); dom = np.full(pl.shape, -1, np.int8)
for g, (_, mem, _) in enumerate(GROUPS):
    c = ndi.uniform_filter(np.isin(pl, [I[n] for n in mem]).astype(np.float32), 3) * 9; u = c > best; best[u] = c[u]; dom[u] = g
oth = ndi.uniform_filter(((pl > 0) & (pl != STB) & (gid[pl] < 0)).astype(np.float32), 3) * 9; u = (oth > best); best[u] = oth[u]; dom[u] = len(GROUPS)
def rgb_(c): return [int(255 * v) for v in mpl.colors.to_rgb(c)]
img = np.full(pl.shape + (3,), 255, np.uint8); img[(pl > 0) & ~stb] = (222, 222, 222); img[stb] = rgb_(STB_C)
for g, (_, _, c) in enumerate(GROUPS): img[surf & (dom == g)] = rgb_(c)
img[surf & (dom == len(GROUPS))] = rgb_(OTHER_C); img[surf & (dom == -1)] = rgb_(NONE_C)
win_cnt = pd.Series({("none" if d == -1 else (GROUPS[d][0] if d < len(GROUPS) else "other")): int(((dom == d) & surf).sum()) for d in range(-1, len(GROUPS) + 1)})
print("r5 window STB surface pixels by dominant neighbour (%):", (100 * win_cnt / win_cnt.sum()).round(1).to_dict(), flush=True)


SHORT = {"villous stroma (PVF, Fib, Mes)": "Stroma", "smooth muscle cell": "SMC",
         "cytotrophoblast": "CTB", "other labelled (immune, myofibroblast, ...)": "Other",
         "syncytiotrophoblast": "STB", "pericyte": "Peri", "Hofbauer cell": "HBC",
         "endothelium": "Endo", "EVT (incl. interstitial)": "EVT",
         "no labelled neighbour (intervillous space / no call)": "Intervillous space"}
LEG = [("Stroma", "#00E05A"), ("SMC", "#6D4C41"), ("CTB", "#1565FF"), ("Other", OTHER_C),
       ("STB", STB_C), ("Peri", "#E65100"), ("HBC", "#B026FF"), ("Endo", "#FF2D2D"),
       ("EVT", "#5D4037"), ("Intervillous space", NONE_C)]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "pdf.fonttype": 42})


fig, ax = plt.subplots(figsize=(6.4, 6.4))
ax.imshow(img, interpolation="nearest")
ax.plot([WIN - 128, WIN - 78], [20, 20], color="k", lw=4, solid_capstyle="butt")
ax.text(WIN - 70, 20, "50\u00b5m", fontsize=13, va="center", ha="left")
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values(): sp.set_visible(False)
fig.tight_layout(pad=0.1)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"panelF_stb_surface_r5{tag()}.{ext}", dpi=300, bbox_inches="tight", pad_inches=.02)
plt.close(fig)


fig = plt.figure(figsize=(7.6, 6.4))
axs = fig.add_axes([.10, .20, .88, .78])
axs.set_xlim(-.12, 1.42); axs.set_ylim(-.06, 1.06); axs.axis("off")
cm = {nm: c for nm, _, c in GROUPS}
cm["no labelled neighbour (intervillous space / no call)"] = NONE_C
cm["other labelled (immune, myofibroblast, ...)"] = OTHER_C
order_ = sorted(sh, key=lambda k: -sh[k]); tot_ = sum(sh.values())
gap = .018; avail = 1 - gap * (len(order_) - 1)
xL0, xL1, xR0, xR1 = 0.0, .07, .93, 1.0
yl = yr = 1.0; LBL = []
for g in order_:
    h = sh[g] / tot_; hR = h * avail
    yl1, yl0 = yl, yl - h; yr1, yr0 = yr, yr - hR; xm = (xL1 + xR0) / 2
    verts = [(xL1, yl1), (xm, yl1), (xm, yr1), (xR0, yr1), (xR0, yr0), (xm, yr0), (xm, yl0),
             (xL1, yl0), (xL1, yl1)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CURVE4,
             Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
    axs.add_patch(PathPatch(Path(verts, codes), fc=cm[g], ec="white", lw=.8, alpha=.78))
    axs.add_patch(plt.Rectangle((xR0, yr0), xR1 - xR0, hR, fc=cm[g], ec="white", lw=.6))
    LBL.append(((yr0 + yr1) / 2, "%.1f%%" % sh[g]))
    yl -= h; yr -= hR + gap
ya = [l[0] for l in LBL]
for i in range(1, len(ya)):
    ya[i] = min(ya[i], ya[i - 1] - .045)
for (y0_, s_), y1_ in zip(LBL, ya):
    axs.text(xR1 + .05, y1_, s_, fontsize=13, va="center")
    if abs(y1_ - y0_) > .012:
        axs.plot([xR1 + .005, xR1 + .045], [y0_, y1_], color="#888", lw=.6)
axs.add_patch(plt.Rectangle((xL0, 0), xL1 - xL0, 1, fc=STB_C, ec="#888", lw=.8))
axs.text(xL0 - .03, .5, "STB surface", ha="center", va="center", fontsize=13, rotation=90)
axl = fig.add_axes([.10, .01, .88, .17]); axl.axis("off")


NCOL = 4
col_len = [len(LEG) // NCOL + (i < len(LEG) % NCOL) for i in range(NCOL)]
slot = [None] * len(LEG); at = 0
for c, n in enumerate(col_len):
    for r in range(n):
        slot[at] = r * NCOL + c; at += 1
ordered = [LEG[i] for i in sorted(range(len(LEG)), key=lambda i: slot.index(i))]
axl.legend(handles=[plt.Line2D([], [], marker="o", ls="none", ms=9, mfc=c, mec="none", label=n)
                    for n, c in ordered],
           loc="upper center", ncol=NCOL, frameon=False, fontsize=13, handletextpad=.3,
           columnspacing=1.4, labelspacing=.5)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"panelF_stb_surface_shares{tag()}.{ext}", dpi=300, bbox_inches="tight", pad_inches=.02)
print("saved")
