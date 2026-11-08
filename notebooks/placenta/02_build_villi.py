#!/usr/bin/env python
import time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import ndimage as ndi
import multiprocessing as mp
import config as C
from config import DATA, SOURCE, celltype_planes, tag
SCR = C.SCRATCH
_names = C.names

ZLO, ZHI = C.ZLO, C.ZHI
HALO, MIN_CORE, ASSIGN_MAX, HOLE = C.HALO, C.MIN_CORE, C.ASSIGN_MAX, C.HOLE
NPROC = C.NPROC
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)

names = _names()
I = {n: i + 1 for i, n in enumerate(names)}
I_STB, I_CTB, I_END = I["syncytiotrophoblast"], I["cytotrophoblast"], I["endothelial cell"]
EXCL = [I["interstitial extravillous trophoblast"], I["extravillous trophoblast"], I["myofibroblast"]]

lab = celltype_planes(ZLO, ZHI).astype(np.int8)
NZ, NY, NX = lab.shape
log("source", SOURCE, "loaded", lab.shape)
S6 = ndi.generate_binary_structure(3, 1)

core = (lab > 0) & ~np.isin(lab, [I_STB, I_CTB] + EXCL)
dil = ndi.binary_dilation(core, S6, iterations=3)
clo = ndi.binary_erosion(dil, S6, iterations=3, border_value=1); del dil
for k in range(NZ):
    bg, n = ndi.label(~clo[k])
    if n:
        sz = np.bincount(bg.ravel()); sz[0] = 0
        edge = np.unique(np.r_[bg[0], bg[-1], bg[:, 0], bg[:, -1]])
        small = (sz <= HOLE); small[edge] = False
        clo[k] |= small[bg] & (bg > 0)
log("closed")
cid, n = ndi.label(clo, structure=S6); del clo
sz = np.bincount(cid.ravel()); sz[0] = 0
keep = np.where(sz >= MIN_CORE)[0]
remap = np.zeros(len(sz), np.int32); remap[keep] = np.arange(1, len(keep) + 1)
cid = remap[cid]; NV = len(keep)
log("components", n, "kept", NV, "core voxels", int((cid > 0).sum()))

VPATH = SCR / f"villus{tag()}.dat"
VIL = np.memmap(VPATH, np.int32, "w+", shape=lab.shape)


def work(span):
    x0, x1 = span
    xs, xe = max(0, x0 - HALO), min(NX, x1 + HALO)
    L = lab[:, :, xs:xe]; C = cid[:, :, xs:xe]
    corem = C > 0
    if corem.any():
        dist, idx = ndi.distance_transform_edt(~corem, return_indices=True)
        assigned = C[tuple(idx)]; del idx
        fld = (L == I_STB) | (L == I_CTB)
        V = np.where(corem, C, np.where(fld & (dist <= ASSIGN_MAX), assigned, 0)).astype(np.int32)
    else:
        V = np.zeros(C.shape, np.int32)
    c0, c1 = x0 - xs, x1 - xs
    VIL[:, :, x0:x1] = V[:, :, c0:c1]
    return x0


edges = np.linspace(0, NX, NPROC * 2 + 1).astype(int)
spans = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]
with mp.get_context("fork").Pool(NPROC) as p:
    for r in p.imap_unordered(work, spans):
        pass
VIL.flush(); log("villi assigned")


FIELDS = ["STB", "CTB", "endothelial", "stroma", "unlabelled"]
NF = len(FIELDS)


def field_of(l):
    f = np.full(l.shape, 3, np.int8)
    f[l == 0] = 4; f[l == I_STB] = 0; f[l == I_CTB] = 1; f[l == I_END] = 2
    return f


vf = np.zeros((NV + 1, NF), np.int64)
for k in range(NZ):
    v = np.asarray(VIL[k]); m = v > 0
    vf += np.bincount(v[m].astype(np.int64) * NF + field_of(lab[k][m]), minlength=(NV + 1) * NF).reshape(NV + 1, NF)
vol = vf.sum(1).astype(float)
df = pd.DataFrame(dict(villus=np.arange(NV + 1), volume_um3=vol))
for i, f in enumerate(FIELDS):
    df[f"frac_{f}"] = vf[:, i] / np.maximum(vol, 1)
V = df.iloc[1:].reset_index(drop=True)
V.to_csv(DATA / f"villus_table{tag()}.csv", index=False)
log("done; villi", NV, " median volume", int(np.median(vol[1:])), "voxels;  >= 3e4 um^3:", int((V.volume_um3 >= 3e4).sum()))
