#!/usr/bin/env python
import sys

import numpy as np
import zarr

import config as C

CHUNK = 200


def build(section):
    g = zarr.open(str(C.store(section)), mode="r")
    m = g["filtered_celltype_maps"]
    names = [str(x) for x in g["celltype_names"][:]]
    K, B = len(names), C.BLOCK_UM
    nx, ny = m.shape[0] // B, m.shape[1] // B
    counts = np.zeros((K, nx, ny), np.uint16)
    bg = np.zeros((nx, ny), np.uint16)
    for x0 in range(0, nx, CHUNK):
        x1 = min(nx, x0 + CHUNK)
        a = np.asarray(m[x0 * B:x1 * B, :ny * B, C.Z0:C.Z1]).reshape(x1 - x0, B, ny, B, C.Z1 - C.Z0)
        bg[x0:x1] = (a == -1).sum((1, 3, 4))
        for k in range(K):
            counts[k, x0:x1] = (a == k).sum((1, 3, 4))
    out = C.grid_npz(section)
    np.savez_compressed(out, counts=counts, bg=bg, names=np.array(names),
                        offset=g["coordinate_offset"][:], block=B, z=(C.Z0, C.Z1))
    nz = C.Z1 - C.Z0
    tissue = counts.sum(0) / (B * B * nz) >= C.TISSUE_FRAC
    cm = sum(counts[k] for k, n in enumerate(names) if "cardiomyocyte" in n.lower())
    print(f"{section:12s} {counts.shape}  tissue blocks {int(tissue.sum()):>7,}"
          f"  CM share of labelled {cm.sum() / max(counts.sum(), 1):.3f}  -> {out.name}", flush=True)


if __name__ == "__main__":
    for s in (sys.argv[1:] or C.SECTIONS):
        build(s)
