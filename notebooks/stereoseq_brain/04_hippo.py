#!/usr/bin/env python
import json
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize, disk
from common import *

rng = np.random.default_rng(1)
b = load_bin1()


SUB = {"DG": ["Prox1", "C1ql2"], "CA1": ["Spink8", "Fibcd1"], "CA3": ["Bok"]}
tot = raster(b)
graw = {k: raster(b, v) for k, v in SUB.items()}
G = np.load(CACHE / "geometry.npz"); tissue = G["tissue"]


xs_, ys_ = [], []
for k in SUB:
    hs = ndi.gaussian_filter(graw[k].astype(float), 6)
    m = hs > threshold_otsu(hs[tissue])
    lab, n = ndi.label(m); sz = ndi.sum(m, lab, range(1, n + 1))
    xs, ys = np.nonzero(lab == (np.argmax(sz) + 1)); xs_ += [xs.min(), xs.max()]; ys_ += [ys.min(), ys.max()]
    print(k, "largest component bbox um", xs.min() * PX, xs.max() * PX, ys.min() * PX, ys.max() * PX)
pad = 30
x0, x1 = max(min(xs_) - pad, 0), min(max(xs_) + pad, NX); y0, y1 = max(min(ys_) - pad, 0), min(max(ys_) + pad, NY)
print("hippocampal window (um)", x0 * PX, x1 * PX, y0 * PX, y1 * PX)
sl = (slice(x0, x1), slice(y0, y1))


t = tot[sl].astype(float)


import os


OTSU_MULT = float(os.environ.get("OTSU_MULT", "1.3"))
dens = ndi.gaussian_filter(t, 2)
band = (dens > threshold_otsu(dens[tissue[sl]]) * OTSU_MULT) & tissue[sl]
band = ndi.binary_opening(band, iterations=1)
band = ndi.binary_fill_holes(ndi.binary_closing(band, structure=disk(3))) & tissue[sl]

nd = {}
for k in SUB:
    v = ndi.gaussian_filter(graw[k][sl].astype(float), 3) / (ndi.gaussian_filter(t, 3) + 1e-9)
    thr = threshold_otsu(v[tissue[sl]])
    nd[k] = v / thr
stack = np.stack([nd[k] for k in SUB])
best = np.argmax(stack, 0); ok = stack.max(0) > 1


seed = np.zeros(band.shape, np.int16)
for i, k in enumerate(SUB):
    mk = band & ok & (best == i)
    mk = ndi.binary_closing(mk, iterations=2) & band
    lab, n = ndi.label(mk); szz = ndi.sum(mk, lab, range(1, n + 1))
    seed[np.isin(lab, np.where(szz * PX * PX >= 0.005e6)[0] + 1)] = i + 1


comp, ncomp = ndi.label(band, structure=np.ones((3, 3)))
filled = seed.copy()
for c in range(1, ncomp + 1):
    m = comp == c
    src = m & (seed > 0)
    if not src.any() or m.sum() * PX * PX < 0.005e6:
        continue
    dist, idx = ndi.distance_transform_edt(~src, return_distances=True, return_indices=True)
    near = m & (seed == 0) & (dist <= 10)
    filled[near] = seed[tuple(idx[:, near])]
sub_masks = {}
for i, k in enumerate(SUB):
    mk = filled == i + 1
    lab, n = ndi.label(mk, structure=np.ones((3, 3))); szz = ndi.sum(mk, lab, range(1, n + 1))
    sub_masks[k] = np.isin(lab, np.where(szz * PX * PX >= 0.005e6)[0] + 1)
    sk_ = skeletonize(sub_masks[k]); nseg = ndi.label(sk_, structure=np.ones((3, 3)))[1]
    th_ = np.median(2 * ndi.distance_transform_edt(sub_masks[k])[sk_]) * PX if sk_.any() else 0
    print(f"  {k}: area {sub_masks[k].sum() * PX * PX / 1e6:.3f} mm2, skeleton {sk_.sum() * PX / 1000:.2f} mm, "
          f"{nseg} pieces, thickness {th_:.0f} um")


cm, names = load_map()
lm = map_at(cm)[sl]; del cm
nm = np.array(["none"] + names)
ss = nm[np.where(lm > 0, lm, 0)]


lc = np.load(CACHE / "cellbin_label_raster.npz")["label"]
ann = pd.read_parquet(CACHE / "cellbin_annotation.parquet")
amap = dict(zip(ann.label.values, ann.annotation.fillna("unannotated").values))
cx = np.clip(np.arange(x0, x1) * PX + PX // 2, 0, lc.shape[0] - 1)
cy = np.clip(np.arange(y0, y1) * PX + PX // 2, 0, lc.shape[1] - 1)
ids = np.asarray(lc[np.ix_(cx, cy)])
print(f"published CellBin cover in the crop: {100 * (ids > 0).mean():.1f}% of 5 µm pixels",
      flush=True)
pub = np.vectorize(lambda i: amap.get(i, "unannotated") if i > 0 else "none", otypes=[object])(ids)

CORRECT_SS = {"DG": {"dentate gyrus granule cell"}, "CA1": {"CA1 pyramidal neuron"}, "CA3": {"CA3 pyramidal neuron"}}
PARENT_SS = {"DG": set(), "CA1": {"hippocampal pyramidal neuron"}, "CA3": {"hippocampal pyramidal neuron"}}
WRONG_SS = {"DG": {"CA1 pyramidal neuron", "CA3 pyramidal neuron", "hippocampal pyramidal neuron"},
            "CA1": {"CA3 pyramidal neuron", "dentate gyrus granule cell"},
            "CA3": {"CA1 pyramidal neuron", "dentate gyrus granule cell"}}
CORRECT_PUB = {"DG": {"GN DG"}, "CA1": {"EX CA"}, "CA3": {"EX CA"}}

def runs_gaps(cov_sk, sk):
    gl, gn = ndi.label(sk & ~cov_sk, structure=np.ones((3, 3)))
    gs = ndi.sum(np.ones_like(gl), gl, range(1, gn + 1)) * PX
    return int((gs >= 20).sum()), float(gs.max()) if gn else 0.0

def score(labels, k, correct, parent=set(), wrong=set()):
    sk = skel[k]
    L = labels[sk]
    c = np.isin(L, list(correct)); p = np.isin(L, list(parent)); w = np.isin(L, list(wrong))
    none = np.isin(L, ["none", "unannotated"])
    cov = np.zeros_like(sk); cov[sk] = c | p
    ngap, maxgap = runs_gaps(cov, sk)

    tm = np.isin(labels, list(correct | parent))
    th = 2 * ndi.distance_transform_edt(tm)[sk] * PX
    return dict(n_skel=int(sk.sum()), skel_len_mm=float(sk.sum() * PX / 1000),
                exact=float(c.mean()), exact_or_parent=float((c | p).mean()), wrong_subfield=float(w.mean()),
                unlabelled=float(none.mean()), other=float(1 - (c | p | w | none).mean()),
                gaps_ge20um_per_mm=float(ngap / (sk.sum() * PX / 1000)), max_gap_um=maxgap,
                thickness_um_median_where_covered=float(np.median(th[c | p])) if (c | p).any() else None,
                top_labels=pd.Series(L).value_counts(normalize=True).head(6).round(3).to_dict())

def null_cov(labels, k, correct, parent=set(), n=500):
    sk = skel[k]; out = []
    for _ in range(n):
        while True:
            dx, dy = rng.integers(0, labels.shape[0]), rng.integers(0, labels.shape[1])
            ddx = min(dx, labels.shape[0] - dx) * PX; ddy = min(dy, labels.shape[1] - dy) * PX
            if np.hypot(ddx, ddy) >= 150: break
        sh = np.roll(np.roll(labels, dx, 0), dy, 1)
        out.append(np.isin(sh[sk], list(correct | parent)).mean())
    return np.array(out)

skel, raw_th = {}, {}
res = dict(window_um=[int(x0 * PX), int(x1 * PX), int(y0 * PX), int(y1 * PX)])
for k in SUB:
    skel[k] = skeletonize(sub_masks[k])
    raw_th[k] = 2 * ndi.distance_transform_edt(sub_masks[k])[skel[k]] * PX
    s_ss = score(ss, k, CORRECT_SS[k], PARENT_SS[k], WRONG_SS[k])
    s_pb = score(pub, k, CORRECT_PUB[k])
    n_ss = null_cov(ss, k, CORRECT_SS[k], PARENT_SS[k]); n_pb = null_cov(pub, k, CORRECT_PUB[k])
    s_ss["null_mean"] = float(n_ss.mean()); s_ss["null_max"] = float(n_ss.max())
    s_ss["p"] = float(((n_ss >= s_ss["exact_or_parent"]).sum() + 1) / (len(n_ss) + 1))
    s_pb["null_mean"] = float(n_pb.mean()); s_pb["null_max"] = float(n_pb.max())
    s_pb["p"] = float(((n_pb >= s_pb["exact_or_parent"]).sum() + 1) / (len(n_pb) + 1))

    coarse = {"DG": {"dentate gyrus granule cell"},
              "CA1": {"CA1 pyramidal neuron", "CA3 pyramidal neuron", "hippocampal pyramidal neuron"},
              "CA3": {"CA1 pyramidal neuron", "CA3 pyramidal neuron", "hippocampal pyramidal neuron"}}[k]
    s_ss["coarse_CA_vs_DG"] = float(np.isin(ss[skel[k]], list(coarse)).mean())
    res[k] = dict(raw_band_area_mm2=float(sub_masks[k].sum() * PX * PX / 1e6),
                  raw_thickness_um_median=float(np.median(raw_th[k])),
                  raw_thickness_um_iqr=[float(np.percentile(raw_th[k], 25)), float(np.percentile(raw_th[k], 75))],
                  ssaft=s_ss, published_cellbin=s_pb)
    print(f"\n[{k}] skeleton {s_ss['skel_len_mm']:.2f} mm, raw thickness {np.median(raw_th[k]):.0f} um")
    print("  SSAF-T   ", {a: (round(v, 3) if isinstance(v, float) else v) for a, v in s_ss.items()})
    print("  CellBin  ", {a: (round(v, 3) if isinstance(v, float) else v) for a, v in s_pb.items()})


prec = {}
for k, lab_ in (("DG", "dentate gyrus granule cell"), ("CA1", "CA1 pyramidal neuron"), ("CA3", "CA3 pyramidal neuron")):
    mk = ss == lab_
    near = ndi.binary_dilation(sub_masks[k], iterations=4)
    anyb = ndi.binary_dilation(band, iterations=4)
    prec[lab_] = dict(n_px=int(mk.sum()), on_own_band_20um=float(near[mk].mean()), on_any_dense_band_20um=float(anyb[mk].mean()))
for lab_, ks in (("GN DG", ["DG"]), ("EX CA", ["CA1", "CA3"])):
    mk = pub == lab_
    near = ndi.binary_dilation(np.any([sub_masks[k] for k in ks], 0), iterations=4)
    prec[lab_] = dict(n_px=int(mk.sum()), on_own_band_20um=float(near[mk].mean()))
res["precision"] = prec


def own_width(mk, close=0):
    if close: mk = ndi.binary_closing(mk, iterations=close)
    lab, n = ndi.label(mk); szz = ndi.sum(mk, lab, range(1, n + 1))
    mk = np.isin(lab, np.where(szz * PX * PX >= 0.005e6)[0] + 1)
    if not mk.any(): return None
    sk = skeletonize(mk); w = 2 * ndi.distance_transform_edt(mk)[sk] * PX
    return dict(median=float(np.median(w)), iqr=[float(np.percentile(w, 25)), float(np.percentile(w, 75))], area_mm2=float(mk.sum() * PX * PX / 1e6))
res["width_own_skeleton"] = {
    "raw_DG": own_width(sub_masks["DG"]), "raw_CA1": own_width(sub_masks["CA1"]), "raw_CA3": own_width(sub_masks["CA3"]),
    "raw_dense_band_all": own_width(band),
    "ssaft_DG": own_width(ss == "dentate gyrus granule cell"), "ssaft_CA1": own_width(ss == "CA1 pyramidal neuron"),
    "ssaft_CA3": own_width(ss == "CA3 pyramidal neuron"),
    "ssaft_CA1_or_parent": own_width(np.isin(ss, ["CA1 pyramidal neuron", "hippocampal pyramidal neuron"])),
    "cellbin_GN_DG": own_width(pub == "GN DG", close=2), "cellbin_EX_CA": own_width(pub == "EX CA", close=2)}
print(json.dumps(res["width_own_skeleton"], indent=0))
print("\nprecision", json.dumps(prec, indent=0))
json.dump(res, open(CACHE / "stats_hippo.json", "w"), indent=1, default=float)
np.savez_compressed(CACHE / "hippo_masks.npz", band=band, x0=x0, y0=y0, x1=x1, y1=y1,
                    **{f"mask_{k}": v for k, v in sub_masks.items()}, **{f"skel_{k}": v for k, v in skel.items()},
                    ss=ss.astype(str), pub=pub.astype(str), dens=dens)

fig, ax = plt.subplots(1, 3, figsize=(21, 9))

ax[0].imshow(dens.T, cmap="gray", vmin=0, vmax=np.percentile(dens[dens > 0], 99))
ax[0].set_title("smoothed total UMI")
col = {"DG": "#e41a1c", "CA1": "#4daf4a", "CA3": "#377eb8"}
rgb = np.ones(band.shape + (3,)) * 0.9
for k, c in col.items(): rgb[sub_masks[k]] = [int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)]
for k in SUB: rgb[skel[k]] = 0
ax[1].imshow(rgb.transpose(1, 0, 2)); ax[1].set_title("raw-defined bands + skeleton")
pal = palette(); r2 = np.ones(band.shape + (3,))
for n_ in np.unique(ss):
    c = pal.get(n_, "#dddddd") if n_ != "none" else "#ffffff"; r2[ss == n_] = [int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)]
ax[2].imshow(r2.transpose(1, 0, 2)); ax[2].set_title("SSAF-T")
plt.tight_layout(); plt.savefig(CACHE / "hippo_diag.png", dpi=60)
print("done")
