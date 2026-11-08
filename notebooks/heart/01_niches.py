#!/usr/bin/env python
import warnings

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

import config as C

warnings.filterwarnings("ignore")
B = C.BLOCK_UM


def group_of(label):
    l = label.lower()
    if "cardiomyocyte" in l:
        return "CM"
    if "fibroblast" in l or "mesenchymal" in l:
        return "FB"
    if "endothelial" in l:
        return "EC"
    if "smooth muscle" in l:
        return "mural"
    if any(k in l for k in ("macrophage", "monocyte", "dendritic", "mast cell", "myeloid", "leukocyte")):
        return "myeloid"
    if any(k in l for k in ("t cell", "natural killer", "b cell", "plasma")):
        return "lymphoid"
    if "adipocyte" in l:
        return "adipocyte"
    return "other"


def disk(radius_um):
    r = radius_um // B
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return ((yy ** 2 + xx ** 2) <= r ** 2).astype(np.float32)


KERN = disk(C.NICHE_R)


def dsum(a, k=KERN):
    return np.maximum(fftconvolve(a.astype(np.float32), k, mode="same"), 0)


def hist2(x, y, H, W):
    return np.histogram2d(x, y, bins=[H, W], range=[[0, H * B], [0, W * B]])[0].astype(np.float32)


def prep(section):
    d = np.load(C.grid_npz(section))
    counts = d["counts"].astype(np.float32)
    nm = [str(x).lower() for x in d["names"]]
    off = d["offset"]
    H, W = d["bg"].shape
    nz = d["z"][1] - d["z"][0]

    cm = sum(counts[k] for k, n in enumerate(nm) if "cardiomyocyte" in n)
    labelled = sum(counts[k] for k, n in enumerate(nm) if "unknown" not in n)
    tissue = counts.sum(0) / (B * B * nz) >= C.TISSUE_FRAC

    a = C.read_cells(section)
    o = a.obs
    seg = o.segmented.astype(bool).values
    G = np.array([group_of(str(l)) for l in o.cellman_annotation.astype(str).values])
    xy = np.asarray(a.obsm["spatial"])[:, :2] - off[:2]
    keep = seg & np.isin(G, C.GRP)
    xy, G = xy[keep], G[keep]

    gi, gj = np.meshgrid(np.arange(0, H, C.GRID_STEP), np.arange(0, W, C.GRID_STEP), indexing="ij")
    gi, gj = gi.ravel(), gj.ravel()
    cnt = np.stack([dsum(hist2(xy[G == g, 0], xy[G == g, 1], H, W))[gi, gj] for g in C.GRP], 1)
    occN = (dsum(cm) / (dsum(labelled) + 1e-6))[gi, gj]
    ok = tissue[gi, gj]
    return dict(cnt=cnt[ok], occN=occN[ok], ix=gi[ok], iy=gj[ok], H=H, W=W,
                area=np.pi * C.NICHE_R ** 2, n_cells={g: int((G == g).sum()) for g in C.GRP})


def features(d):
    l = np.log(d["cnt"] + 1)
    return np.c_[d["occN"], l - l.mean(1, keepdims=True),
                 np.log1p(d["cnt"].sum(1) / d["area"] * 1e6)]


D = {}
for _, s in C.SAMP:
    D[s] = prep(s)
    print(f"prep {C.CELLS:9s} {s:12s} {len(D[s]['ix']):>7,} grid points  {D[s]['n_cells']}", flush=True)

F = {s: features(D[s]) for s in D}
allF = np.vstack([F[s] for _, s in C.SAMP])
allZ = (allF - allF.mean(0)) / (allF.std(0) + 1e-9)
who = np.concatenate([[s] * len(F[s]) for _, s in C.SAMP])

rng = np.random.default_rng(0)
sub = rng.choice(len(allZ), min(len(allZ), 150_000), replace=False)
sel, models = [], {}
for k in range(3, 10):
    lab = KMeans(k, n_init=10, random_state=0).fit(allZ[sub]).predict(allZ)
    ari = []
    for _, s in C.SAMP:
        held = who == s
        ms = np.where(~held)[0]
        ss = rng.choice(ms, min(len(ms), 120_000), replace=False)
        km2 = KMeans(k, n_init=3, random_state=0).fit(allZ[ss])
        ari.append(adjusted_rand_score(lab[held], km2.predict(allZ[held])))
    nsec = [sum((lab[who == s] == j).sum() >= 500 for _, s in C.SAMP) for j in range(k)]
    sel.append(dict(k=k, mean_LODO_ARI=np.mean(ari), min_sections_with_niche=min(nsec),
                    pass_a=min(nsec) >= 6, pass_b=np.mean(ari) >= 0.7))
    models[k] = lab
    print(sel[-1], flush=True)

SEL = pd.DataFrame(sel)
SEL.to_csv(C.TAB / f"wtniche_k_selection{C.tag()}.csv", index=False)
ok = SEL[SEL.pass_a & SEL.pass_b]
if not len(ok):
    raise SystemExit("no k passes both criteria -> no niches (the pre-registered rule)")
K = int(ok.k.max())
print("chosen k (rule)", K, flush=True)
if C.K_FORCE:
    K = C.K_FORCE
    print("K_FORCE: using k", K, SEL[SEL.k == K].to_dict("records"), flush=True)
lab = models[K]

raw = np.vstack([D[s]["cnt"] for _, s in C.SAMP]) + 1e-9
frac = raw / raw.sum(1, keepdims=True)
comp = pd.DataFrame([dict(niche=j, n_cells=int((lab == j).sum()),
                          CM_area_frac=allF[lab == j][:, 0].mean(),
                          **{"frac_" + g: frac[lab == j][:, i].mean() for i, g in enumerate(C.GRP)},
                          log_density=allF[lab == j][:, -1].mean()) for j in range(K)])
comp.to_csv(C.TAB / f"wtniche_comp{C.tag()}.csv", index=False)

ab = pd.DataFrame([dict(sample=s, group=g, niche=j,
                        frac=float((lab[who == s] == j).mean()),
                        n=int((lab[who == s] == j).sum()))
                   for g, s in C.SAMP for j in range(K)])
ab.to_csv(C.TAB / f"wtniche_abundance{C.tag()}.csv", index=False)

np.savez(C.TAB / f"wtniche{C.tag()}.npz", K=K, who=who, lab=lab,
         **{f"ix_{s}": D[s]["ix"] for _, s in C.SAMP},
         **{f"iy_{s}": D[s]["iy"] for _, s in C.SAMP})

pd.set_option("display.width", 250)
print(comp.round(3).to_string())
print(ab.pivot(index="sample", columns="niche", values="frac").round(2))
