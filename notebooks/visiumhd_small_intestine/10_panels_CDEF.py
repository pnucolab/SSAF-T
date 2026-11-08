import json, os, sys
import numpy as np, pandas as pd, geopandas as gpd
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries
from matplotlib.colors import to_rgb, LinearSegmentedColormap
from matplotlib.patches import Polygon as MPoly, FancyBboxPatch, Ellipse, PathPatch
from matplotlib.path import Path as MPath
from matplotlib.collections import PatchCollection
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                     "font.size": 7, "axes.linewidth": .6, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "xtick.major.width": .6, "ytick.major.width": .6, "xtick.major.size": 2.5, "ytick.major.size": 2.5})
st = json.load(open(OUT / "stats.json"))
pk = json.load(open(WORK / "picked_units.json"))
names = json.load(open(WORK / "celltype_names.json"))
cm = np.load(WORK / "celltype_map.npy")
ax_ = np.load(WORK / "axes.npz"); DA = ax_["DA"]
units = np.load(WORK / "units.npz")["units"]
z = np.load(WORK / "rasters.npz")
sc = pd.read_parquet(WORK / "ssaft_cells_axis_scored.parquet")
ec = pd.read_parquet(WORK / "enact_cells_axis_C.parquet")
WHITE_BG = os.environ.get("FIELD_BG") == "white"


SHOW_AXIS_PANEL = os.environ.get("AXIS_PANEL") == "1"
C = lambda l: COLORS.get(l, "#bbbbbb")
SHORT = {"intestinal stem cell": "stem (ISC)", "Paneth cell": "Paneth", "transit amplifying cell": "TA",
         "immature enterocyte": "immature ent.", "villus-bottom enterocyte": "villus-bottom ent.",
         "absorptive enterocyte": "absorptive ent.", "distal enterocyte": "distal (ileal) ent.",
         "villus-tip enterocyte": "villus-tip ent.", "immature Paneth cell progenitor": "imm. Paneth prog.",
         "goblet cell": "goblet", "goblet-Paneth intermediate cell": "goblet-Paneth int.",
         "enteroendocrine cell": "EEC", "tuft cell": "tuft"}
def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=600, bbox_inches="tight", facecolor="white")


    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white", dpi=600)
    plt.close(fig)

if WHITE_BG:
    pass

ZONES = [("Paneth cell", -1, 1.0), ("intestinal stem cell", 1.0, 1.45), ("transit amplifying cell", 1.45, 3.0),
         ("immature enterocyte", 3.0, 4.3), ("villus-bottom enterocyte", 4.3, 6.2),
         ("absorptive enterocyte", 6.2, 8.4), ("villus-tip enterocyte", 8.4, 99)]
def zone_of(y):
    for L, y0, y1 in ZONES:
        if y0 <= y < y1: return L
def outline():
    pts = []
    rc, rv = .45, .9
    for t in np.linspace(np.pi * 1.5, np.pi, 8): pts.append((rc * np.cos(t), rc + rc * np.sin(t)))
    for y in np.linspace(rc, 3.0, 12): pts.append((-rc, y))
    for x in np.linspace(-rc - .1, -rv, 3): pts.append((x, 3.05))
    for y in np.linspace(3.2, 10 - rv, 26): pts.append((-rv, y))
    for t in np.linspace(np.pi, np.pi / 2, 8)[1:]: pts.append((rv * np.cos(t), 10 - rv + rv * np.sin(t)))
    return np.array(pts)
def schematic(ax):
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-1.3, 1.3); ax.set_ylim(-.75, 10.3)
    L_ = outline(); R_ = L_[::-1] * [-1, 1]
    full = np.vstack([L_, R_[1:]])
    ax.add_patch(MPoly(np.vstack([full, [[.45, .45], [-.45, .45]]]), closed=True, fc="#f4efe6", ec="none", zorder=0))

    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(full, axis=0), axis=1))]
    s_c = np.arange(0.0, seg[-1], .34)
    P_ = np.c_[np.interp(s_c, seg, full[:, 0]), np.interp(s_c, seg, full[:, 1])]
    T_ = np.gradient(P_, axis=0); T_ /= np.linalg.norm(T_, axis=1, keepdims=True)
    N_ = np.c_[T_[:, 1], -T_[:, 0]]
    th = .30
    mids = (P_[:-1] + P_[1:]) / 2
    special = {}
    for L, y, side in (("goblet cell", 2.4, 1), ("goblet cell", 5.0, -1), ("goblet cell", 7.0, -1),
                       ("goblet cell", 9.0, 1), ("enteroendocrine cell", 4.0, 1), ("enteroendocrine cell", 7.8, -1),
                       ("goblet cell", 6.0, 1)):
        cand = np.where(np.sign(mids[:, 0]) == side)[0]
        special[int(cand[np.argmin(np.abs(mids[cand, 1] - y))])] = L
    for i in range(len(P_) - 1):
        a0, a1 = P_[i], P_[i + 1]; n0, n1 = N_[i], N_[i + 1]
        quad = [a0, a1, a1 + th * n1, a0 + th * n0]
        y = (a0[1] + a1[1]) / 2
        L = special.get(i, zone_of(y))
        if y < 1.0: L = "Paneth cell" if i % 2 == 0 else "intestinal stem cell"
        ax.add_patch(MPoly(quad, closed=True, fc=C(L), ec="white", lw=.35))
    ax.add_patch(matplotlib.patches.Rectangle((-1.25, -.72), 2.5, .2, fc=C("smooth muscle cell"), ec="none"))
    ax.text(0, -.62, "muscularis", fontsize=5, ha="center", va="center", color="white")
def schematic_panel():
    fig, ax = plt.subplots(figsize=(1.7, 3.4))
    schematic(ax)
    labs = {"villus-tip enterocyte": 9.4, "absorptive enterocyte": 7.3, "villus-bottom enterocyte": 5.25,
            "immature enterocyte": 3.65, "transit amplifying cell": 2.3, "intestinal stem cell": 1.2, "Paneth cell": .45}
    for L, y in labs.items():
        x0 = .9 if y > 3 else .45
        ax.annotate(SHORT[L], xy=(x0 + .02, y), xytext=(1.35, y), fontsize=5.6, va="center", ha="left",
                    annotation_clip=False, arrowprops=dict(arrowstyle="-", lw=.4, color="#555"))
    for L, y in (("goblet cell", 7.0), ("enteroendocrine cell", 7.8)):
        ax.annotate(SHORT[L] + " (interspersed)", xy=(-.95, y), xytext=(-1.35, y), fontsize=5.2, va="center",
                    ha="right", color="#444", annotation_clip=False, arrowprops=dict(arrowstyle="-", lw=.4, color="#888"))


    for y, txt in ((-.62, "muscularis — height 0"), (3.05, "crypt top ≈ 63 µm"),
                   (9.9, "villus tip 250–300 µm")):
        ax.annotate(txt, xy=(-1.28, y), xytext=(-1.38, y), fontsize=5.0, va="center", ha="right",
                    color="#444", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", lw=.4, color="#888"))
    ax.set_title("textbook crypt-villus unit", fontsize=7, pad=4)
    save(fig, "panel_schematic")


if not WHITE_BG:
    schematic_panel()


import spatialdata as sd
o = sd.read_zarr(STORE)
poly = o.shapes["ssaft_cell_boundaries_2d"]
lab_of = pd.Series(o.tables["ssaft_cells"].obs.cellman_annotation.astype(str).values,
                   index=o.tables["ssaft_cells"].obs.cell_id.astype(str).values)
poly = gpd.GeoDataFrame({"label": [lab_of.get(str(i), "") for i in poly.index]}, geometry=poly.geometry.values)
ncol = len(names)
rgb = np.array([to_rgb(C(n)) for n in names])
m = np.load(WORK / "bins_meta.npz"); import scipy.sparse as sp
Xs = sp.load_npz(WORK / "bins_sel.npz").tocsc(); genes = list(m["genes"])


Xk = sp.load_npz(WORK / "bins_kg.npz").tocsc()
kgenes = list(np.load(WORK / "bins_kg_meta.npz")["genes"])
genes = sorted(set(genes) | set(kgenes))


def gcol(g):
    if g in kgenes:
        return np.asarray(Xk[:, kgenes.index(g)].todense()).ravel()
    return np.asarray(Xs[:, list(m["genes"]).index(g)].todense()).ravel()


H, W = cm.shape
def gimg(lst):
    a = np.zeros((H, W), np.float32)
    v = sum((gcol(g) for g in lst if g in genes), start=np.zeros(len(m["ri"]), np.float32))
    a[m["ri"], m["ci"]] = v
    return a
kg = set(json.load(open(WORK / "kg_markers.json"))["kg_markers"])
hk = lambda g: g.startswith(HOUSEKEEPING_PREFIX)


import colorsys, textwrap, urllib.request

SLIM_F = WORK / "slim_index.json"
if not SLIM_F.exists():
    SLIM_F.write_text(urllib.request.urlopen(
        "http://192.168.101.31:8088/module/mouse/small_intestine/slim-index", timeout=120).read().decode())
BY = {v["L"]: v for v in json.loads(SLIM_F.read_text())["terms"].values()}


LAYERS = [("muscularis (axis anchor)", ["smooth muscle cell"], ["Actg2", "Myh11", "Des"]),
          ("Paneth", ["Paneth cell"], None),
          ("Paneth progenitor", ["immature Paneth cell progenitor"], None),
          ("stem (ISC)", ["intestinal stem cell"], None),
          ("transit amplifying", ["transit amplifying cell"], None),
          ("goblet-Paneth int.", ["goblet-Paneth intermediate cell"], None),
          ("immature enterocyte", ["immature enterocyte"], None),
          ("villus-bottom ent.", ["villus-bottom enterocyte"], None),
          ("absorptive ent.", ["absorptive enterocyte"], None),
          ("villus-tip ent.", ["villus-tip enterocyte"], None)]


lab_at = cm[m["ri"], m["ci"]]
tis_at = lab_at > 0


def enrich(gs, t):
    if t not in names:
        return 0.0
    v = sum((gcol(g) for g in gs if g in genes), start=np.zeros(len(lab_at), np.float32))
    mm = lab_at == names.index(t) + 1
    out = v[tis_at & ~mm].sum() / max((tis_at & ~mm).sum(), 1)
    return float((v[mm].sum() / max(mm.sum(), 1)) / (out + 1e-9))


def kg_of(t, min_fold=2.0, min_umi=2000):
    v = BY.get(t, {})
    core = [str(x) for x in ((v.get("DF") or []) + (v.get("R") or [])) if str(x) in genes]
    core = list(dict.fromkeys(core))
    extra = []
    for g in [str(x) for x in (v.get("S") or []) if str(x) in genes and str(x) not in core]:
        col = gcol(g)
        if col.sum() >= min_umi and enrich([g], t) >= min_fold:
            extra.append(g)
    return core + extra


FIELD_TINT = {"absorptive enterocyte": "#ffb3c8", "villus-tip enterocyte": "#ff3d9a"}


def deepen(t):
    return to_rgb(FIELD_TINT.get(t, C(t)))


LAND = []
for title, types, fixed in LAYERS:
    gs = fixed if fixed else kg_of(types[0])
    gs = [g for g in gs if g in genes]
    if not gs:
        print("   no cached KG marker, skipped:", title, flush=True); continue
    LAND.append((title + "\n" + textwrap.fill(", ".join(gs), 17), gs, deepen(types[0])))
LIMG = [(t, gimg(g), col) for t, g, col in LAND]
print("fields:", [(t.split(chr(10))[0], g) for t, g, _ in LAND], flush=True)


def unit_crop(u, pad=25):
    rr, cc = np.nonzero(units == u)
    return slice(max(rr.min() - pad, 0), min(rr.max() + pad, H)), slice(max(cc.min() - pad, 0), min(cc.max() + pad, W))

def orient(u, sl):
    mm = units[sl] == u
    rr, cc = np.nonzero(mm); a = DA[sl][mm]
    base = np.array([rr[a <= 40].mean(), cc[a <= 40].mean()])
    tip = np.array([rr[a >= np.nanpercentile(a, 90)].mean(), cc[a >= np.nanpercentile(a, 90)].mean()])
    dr, dc = tip - base
    theta = np.degrees(np.arctan2(-dr, dc))
    return 90.0 - theta

from scipy.ndimage import rotate
def rot(img, ang, order=0, cval=0):
    return rotate(img, ang, reshape=True, order=order, cval=cval, mode="constant")


def unit_outline(ax, mask, colour, lw=.7, sigma=1.2):
    ax.contour(ndi.gaussian_filter(mask.astype(float), sigma), levels=[.5],
               colors=[colour], linewidths=lw, antialiased=True)

def exemplar(u, tag):
    sl = unit_crop(u)
    ang = orient(u, sl)
    mask_u = (units[sl] == u)
    cmc = cm[sl]
    im = np.ones(cmc.shape + (3,)); fg = cmc > 0
    im[fg] = rgb[cmc[fg] - 1]
    im[~mask_u] = 0.35 * im[~mask_u] + 0.65
    im = np.stack([rot(im[..., k], ang, 0, 1.0) for k in range(3)], -1)
    mr = rot(mask_u.astype(float), ang, 1, 0.0) > .5
    rr_m, cc_m = np.nonzero(mr); P_ = 12
    bb = (slice(max(rr_m.min() - P_, 0), rr_m.max() + P_), slice(max(cc_m.min() - P_, 0), cc_m.max() + P_))
    bb4 = (slice(bb[0].start * 4, bb[0].stop * 4), slice(bb[1].start * 4, bb[1].stop * 4))
    im = im[bb]
    nr = (3 if SHOW_AXIS_PANEL else 2) + len(LIMG)
    fig, axs = plt.subplots(1, nr, figsize=(0.95 * nr, 3.0), gridspec_kw=dict(wspace=.06))
    mrc = mr[bb]
    axs[0].imshow(np.clip(im, 0, 1), interpolation="nearest"); axs[0].set_title("SSAF-T map", fontsize=6.5, pad=2)
    unit_outline(axs[0], mrc, "black")

    x0, y0 = sl[1].start * PX + X0, sl[0].start * PX + Y0
    x1, y1 = sl[1].stop * PX + X0, sl[0].stop * PX + Y0
    sub = poly.cx[x0:x1, y0:y1]
    from skimage.draw import polygon as dpoly
    SC = 4
    hh, ww = (sl[0].stop - sl[0].start) * SC, (sl[1].stop - sl[1].start) * SC
    li = {n: i + 1 for i, n in enumerate(names)}
    ras = np.zeros((hh, ww), np.int32); ids = np.zeros((hh, ww), np.int32)
    for j, (g, l) in enumerate(zip(sub.geometry, sub.label)):
        if l not in li or g is None: continue
        geoms = getattr(g, "geoms", [g])
        for gg in geoms:
            xx, yy = np.asarray(gg.exterior.coords).T
            rr_, cc_ = dpoly((yy - (y0 - PX / 2)) / (PX / SC), (xx - (x0 - PX / 2)) / (PX / SC), (hh, ww))
            ras[rr_, cc_] = li[l]; ids[rr_, cc_] = j + 1
    edge = find_boundaries(ids, mode="inner")
    ci_ = np.ones((hh, ww, 3)); f2 = ras > 0; ci_[f2] = rgb[ras[f2] - 1]; ci_[edge > 0] = 1.0
    mu4 = np.kron(mask_u, np.ones((4, 4), bool))[:hh, :ww]
    ci_[~mu4] = 0.35 * ci_[~mu4] + 0.65
    ci_ = np.stack([rot(ci_[..., k], ang, 0, 1.0) for k in range(3)], -1)[bb4]
    mr4 = rot(mu4.astype(float), ang, 1, 0.0)[bb4] > .5
    axs[1].imshow(np.clip(ci_, 0, 1), interpolation="nearest"); axs[1].set_title("SSAF-T cells", fontsize=6.5, pad=2)
    unit_outline(axs[1], mr4, "black", lw=.7, sigma=4.8)
    f0 = 2
    if SHOW_AXIS_PANEL:
        av = np.where(mask_u, DA[sl], np.nan)
        avr = rot(np.nan_to_num(av, nan=-1), ang, 0, -1.0)[bb]
        axs[2].imshow(np.ma.masked_less(avr, 0), cmap="viridis", vmin=0, vmax=np.nanmax(av),
                      interpolation="nearest")
        axs[2].contour(avr, levels=[63], colors="white", linewidths=.6)
        unit_outline(axs[2], mrc, "black", lw=.6)
        axs[2].set_title(f"axis A: height above\nmuscularis, 0-{np.nanmax(av):.0f} µm\n"
                         f"(line: crypt top, 63 µm)", fontsize=5.2, pad=2)
        f0 = 3
    for k, (t, img, col) in enumerate(LIMG):
        a = ndi.gaussian_filter(img, 1.5)[sl]
        v = a / (np.percentile(a[mask_u], 99.5) + 1e-9)
        base = "#f7f7f7" if WHITE_BG else "#0a0a0a"
        cmap = LinearSegmentedColormap.from_list("c", [base, col])
        rgbimg = cmap(np.clip(v, 0, 1))[..., :3]
        pad = 1.0 if WHITE_BG else 0.0
        rgbimg = np.stack([rot(rgbimg[..., q], ang, 1, pad) for q in range(3)], -1)[bb]
        axs[f0 + k].imshow(np.clip(rgbimg, 0, 1), interpolation="nearest")
        unit_outline(axs[f0 + k], mrc, "#595959" if WHITE_BG else "#9a9a9a")
        axs[f0 + k].set_title(t, fontsize=4.8, pad=2, color="black", linespacing=1.15)
    for a in axs: a.axis("off")

    h_, w_ = im.shape[:2]
    axs[0].plot([w_ * .06, w_ * .06 + 50 / PX], [h_ * .97] * 2, color="black", lw=1.2)
    axs[0].text(w_ * .06 + 25 / PX, h_ * .955, "50 µm", ha="center", va="bottom", fontsize=5.5)
    save(fig, f"panel_exemplar_{tag}" + ("_w" if WHITE_BG else ""))
    return ang

for i, u in enumerate(pk["picked"] + pk.get("supplementary", [])):
    exemplar(u, f"unit{u}")
if WHITE_BG:
    sys.exit(0)


pu_ = pd.read_csv(OUT / "per_unit_ssaft_A.csv").set_index("unit")
rho_img = np.full(units.shape, np.nan, np.float32)
lut = np.full(units.max() + 1, np.nan, np.float32); lut[pu_.index.values] = pu_.rho.values
f = 3
uu = units[::f, ::f]; ri_ = lut[uu]; ri_[uu == 0] = np.nan
tis = ax_["tissue"][::f, ::f]
fig, ax = plt.subplots(figsize=(3.0, 3.0))
bg = np.where(tis, 0.88, 1.0)
ax.imshow(bg, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
im_ = ax.imshow(np.ma.masked_invalid(ri_), cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
for u in pk["picked"]:
    rr, cc = np.nonzero(units[::f, ::f] == u)
    ax.plot(cc.mean(), rr.mean(), marker="o", ms=7, mfc="none", mec="black", mew=.8)
    ax.text(cc.mean() + 25, rr.mean(), str(u), fontsize=5, va="center")
ax.axis("off")
cb = fig.colorbar(im_, ax=ax, fraction=.035, pad=.01); cb.set_label("per-unit Spearman ρ", fontsize=6)
cb.ax.tick_params(labelsize=5)
pp_ = st["order"]["ssaft_cells_axisA"]["per_unit"]
ax.set_title(f"{pp_['n_units']} crypt-villus units; ρ > 0 in {100*pp_['frac_rho_positive']:.0f}%", fontsize=6.5, pad=2)
ax.plot([30, 30 + 500 / (PX * f)], [uu.shape[0] - 30] * 2, color="black", lw=1.2)
ax.text(30 + 250 / (PX * f), uu.shape[0] - 45, "500 µm", ha="center", fontsize=5.5)
save(fig, "panel_unit_map")


ORDER_L = ["Paneth cell", "intestinal stem cell", "transit amplifying cell", "immature enterocyte",
           "villus-bottom enterocyte", "distal enterocyte", "absorptive enterocyte", "villus-tip enterocyte",
           "goblet cell", "enteroendocrine cell"]
d = sc[(sc.unit > 0) & np.isfinite(sc.A)]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(4.6, 2.5), gridspec_kw=dict(width_ratios=[3.2, 1.3], wspace=.08), sharey=True)
data = [d.A[d.label == L].values for L in ORDER_L]
vp = a1.violinplot(data, positions=range(len(ORDER_L)), widths=.8, showextrema=False, bw_method=.25)
for b, L in zip(vp["bodies"], ORDER_L):
    b.set_facecolor(C(L)); b.set_edgecolor("#333"); b.set_linewidth(.3); b.set_alpha(1)
for i, x in enumerate(data):
    q = np.percentile(x, [25, 50, 75])
    a1.plot([i, i], [q[0], q[2]], color="black", lw=1.2); a1.plot(i, q[1], "o", ms=2.5, mfc="white", mec="black", mew=.6)
a1.set_xticks(range(len(ORDER_L))); a1.set_xticklabels([SHORT.get(L, L) for L in ORDER_L], rotation=55, ha="right", fontsize=5.8)
a1.set_ylim(0, 400); a1.set_ylabel("distance from muscularis (µm)\naxis A: through tissue, raw smooth-muscle genes only")
for s_ in ("top", "right"): a1.spines[s_].set_visible(False)


a1.set_title(f"SSAF-T, {len(d):,} cells in {d.unit.nunique():,} crypt-villus units", fontsize=6.5, pad=3)
a1.axvline(7.5, color="#bbb", lw=.5, ls=":")
a1.text(8.5, 390, "interspersed", fontsize=5.4, ha="center", va="top", color="#666")
E_L = ["Paneth cells", "Crypt cells", "Enterocytes", "Goblet cells", "Enteroendocrine cells"]
de = ec[(ec.unit > 0) & np.isfinite(ec.A)]
ENC = {"Paneth cells": C("Paneth cell"), "Crypt cells": C("transit amplifying cell"), "Enterocytes": C("absorptive enterocyte"),
       "Goblet cells": C("goblet cell"), "Enteroendocrine cells": C("enteroendocrine cell")}
data = [de.A[de.label == L].values for L in E_L]
vp = a2.violinplot(data, positions=range(len(E_L)), widths=.8, showextrema=False, bw_method=.25)
for b, L in zip(vp["bodies"], E_L):
    b.set_facecolor(ENC[L]); b.set_edgecolor("#333"); b.set_linewidth(.3); b.set_alpha(1)
for i, x in enumerate(data):
    q = np.percentile(x, [25, 50, 75])
    a2.plot([i, i], [q[0], q[2]], color="black", lw=1.2); a2.plot(i, q[1], "o", ms=2.5, mfc="white", mec="black", mew=.6)
a2.set_xticks(range(len(E_L))); a2.set_xticklabels(E_L, rotation=55, ha="right", fontsize=5.8)
a2.set_title("published (ENACT)", fontsize=6.5, pad=3)
for s_ in ("top", "right", "left"): a2.spines[s_].set_visible(False)
a2.tick_params(left=False)
save(fig, "panel_order_axisA")


P = pd.read_csv(OUT / "raw_gradients_axisA.csv", index_col=0)
Pn = P.apply(lambda c: ndi.gaussian_filter1d(c.values, 1.5)); Pn.index = P.index
Pn = Pn / Pn.max()
cols = ["#61615a", "#008941", "#1b4400", "#4c7d2b", "#c8a600", "#ff8c00", "#8a6d3b", "#e0339b", "#8e0152"]
fig, ax = plt.subplots(figsize=(3.3, 2.3))
lbl = {"smooth muscle (axis anchor)": "smooth muscle (axis anchor)",
       "Paneth, non-KG (Defa21/26/28/35, Ang4, Itln1, Mptx2)": "Paneth (non-KG)",
       "stem, KG (Lgr5, Olfm4, Ascl2, Smoc2)": "stem Lgr5/Olfm4/Ascl2/Smoc2 (KG)",
       "stem, non-KG (Rgmb, Axin2, Slc12a2)": "stem Rgmb/Axin2/Slc12a2 (non-KG)",
       "proliferation, non-KG (Top2a, Cenpf, Birc5, Cdk1, Ccnb2, Stmn1)": "proliferation (non-KG)",
       "Moor villus-bottom landmarks, non-KG": "Moor villus-bottom (non-KG)",
       "Moor mid-villus transporters, non-KG (Slc2a2, Slc15a1, Npc1l1)": "mid transporters Slc2a2/Slc15a1/Npc1l1",
       "Moor villus-top landmarks, non-KG": "Moor villus-top (non-KG)",
       "Moor tip program, non-KG (Enpp3, Egfr, Klf4)": "tip program Enpp3/Egfr/Klf4"}
for c_, k in zip(cols, P.columns):
    ax.plot(Pn.index, Pn[k], color=c_, lw=1.1, label=lbl[k])
ax.set_xlim(0, 350); ax.set_ylim(0, 1.05)
ax.set_xlabel("height above muscularis (µm, axis A)"); ax.set_ylabel("share of raw UMI\n(scaled to max)")
ax.legend(fontsize=5, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.02))
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
ax.set_title(f"raw 2 µm bins, {st['raw_gradients']['n_bins']:,} bins", fontsize=6.5, pad=3)
save(fig, "panel_raw_gradients")


pu = pd.read_csv(OUT / "per_unit_ssaft_A.csv"); pe = pd.read_csv(OUT / "per_unit_enact_A.csv")
eq = pd.read_csv(OUT / "per_unit_equal_resolution.csv")
er = st["equal_resolution_crypt_vs_absorptive"]
fig, ax = plt.subplots(figsize=(2.4, 1.9))
bins = np.linspace(-1, 1, 41)


med = lambda v: r" ($\tilde{\rho}$ = %.2f)" % v
ax.hist(pu.rho, bins=bins, color="#8bbbe8", alpha=.9,
        label="SSAF-T, 5 layers: Paneth+ISC \u2192 TA \u2192 iEnt \u2192 botEnt \u2192 tipEnt"
              + med(pu.rho.median()))
ax.hist(eq.ssaft.dropna(), bins=bins, histtype="step", color="#1b3f8f", lw=.9,
        label="SSAF-T, 2 layers: crypt \u2192 villus" + med(eq.ssaft.median()))
ax.hist(eq.enact.dropna(), bins=bins, histtype="step", color="#eb6834", lw=1,
        label="Kamel et al., 2 layers: crypt \u2192 villus" + med(eq.enact.median()))
ax.axvline(0, color="#999", lw=.5)
pp = st["order"]["ssaft_cells_axisA"]["per_unit"]
ax.set_xlabel("per-unit Spearman \u03c1\n(expected rank vs distance from muscularis)")
ax.set_ylabel("units")
ax.legend(fontsize=4.6, frameon=False, loc="upper left", labelspacing=.45)
ax.text(.02, .55, f"\u03c1 > 0.5 in {100*(pu.rho > .5).mean():.0f}% of units\n"
        f"textbook order is the best of all 120\nrelabellings in {100*pp['frac_units_textbook_best']:.0f}% of units "
        f"(chance {100*pp['chance_textbook_best']:.1f}%)\n"
        f"2 layers, paired: SSAF-T higher in\n{100*er['frac_units_ssaft_higher']:.0f}% of {er['paired_units']} units",
        transform=ax.transAxes, fontsize=5.0, va="top")
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
save(fig, "panel_per_unit")


ENTALL = ["immature enterocyte", "villus-bottom enterocyte", "distal enterocyte", "absorptive enterocyte",
          "villus-tip enterocyte"]
fig, ax = plt.subplots(figsize=(2.2, 2.1))
dd = sc[sc.label.isin(ENTALL) & np.isfinite(sc.x_ho)]
data = [dd.x_ho[dd.label == L].values for L in ENTALL]
bp = ax.boxplot(data, positions=range(len(ENTALL)), widths=.6, showfliers=False, patch_artist=True,
                medianprops=dict(color="black", lw=.9), whiskerprops=dict(lw=.5), capprops=dict(lw=.5), boxprops=dict(lw=.5))
for b, L in zip(bp["boxes"], ENTALL): b.set_facecolor(C(L))
ax.set_xticks(range(len(ENTALL))); ax.set_xticklabels([SHORT[L] for L in ENTALL], rotation=55, ha="right", fontsize=5.8)
ax.set_ylabel("Moor landmark coordinate\ntop / (top + bottom), non-KG genes")
tb = st["heldout_landmarks_on_ssaft"]["tip_vs_bottom_same_position"]
ax.set_title(f"tip vs bottom at the same height:\nΔ = {tb['diff_same_position']:+.2f}, p < 0.001 (1,000 perm.)", fontsize=6, pad=3)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
save(fig, "panel_heldout_landmarks")
print("figures written")
