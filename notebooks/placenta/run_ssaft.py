#!/usr/bin/env python
from __future__ import annotations

import argparse
import os

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

import config as C                                      # noqa: E402
from _common.store import save_elements                 # noqa: E402


DIM = 3
ZSTACK_KEY = "morphology_zstack"
Z_STEP = 3.0
OTSU_MULT = 0.6
KERNEL, BANDWIDTH = "gaussian", 2.5
SD = 1.0
METHOD = "lda"
NORM_F = 0.3
NUCLEI = "nucleus_boundaries"
POINTS = "transcripts"
COLUMN = "cellman_annotation"
N_COMPS, N_NEIGHBORS, RESOLUTION = 50, 15, 1.0
BLOCK_UM = 10.0


N_SAMPLES = 5

DEVICE, NCORES = os.environ.get("SSAFT_DEVICE", "cuda:0"), 16


def store():
    import ssaft
    return ssaft.Analysis(ssaft.Dataset(str(C.STORE)), ncores=NCORES, verbose=True)


def step_down(args):
    import rapids_singlecell as rsc
    import spatialdata as sd
    an = store()
    adata = an.downsample_vector(
        sd.read_zarr(C.SDATA), z=DIM, image_key=ZSTACK_KEY, element_name=POINTS,
        n_samples=N_SAMPLES, block_um=BLOCK_UM, otsu_mult=OTSU_MULT,
        bandwidth=BANDWIDTH, sampling_distance=SD, kernel=KERNEL,
        revert_image=False, gpu_device=DEVICE)
    print(f"anchors {adata.n_obs:,} x {adata.n_vars} genes", flush=True)
    rsc.pp.pca(adata, n_comps=N_COMPS)
    rsc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS)
    rsc.tl.leiden(adata, resolution=RESOLUTION)
    adata.write_h5ad(args.cls)
    print(f"-> {args.cls}  ({adata.obs['leiden'].nunique()} clusters)", flush=True)


def step_map(args):
    import spatialdata as sd
    from pathlib import Path
    if not Path(args.annotated).exists():
        raise SystemExit(f"{args.annotated} not found - annotate {args.cls} with CeLLMan first")
    store().map_celltypes(sd.read_zarr(C.SDATA), method=METHOD, h5ad=str(args.annotated),
                          column=COLUMN, element_name=POINTS, kernel=KERNEL,
                          bandwidth=BANDWIDTH, sampling_distance=SD, gpu_device=DEVICE)


def step_filter(args):
    import spatialdata as sd
    sdata = store().filter_celltypemaps(sd.read_zarr(C.SDATA), norm_f=NORM_F, show_unknown=True)
    save_elements(sdata, ("ssaft_celltype_map", "ssaft_celltype_map_rgb"))


def step_segment(args):
    import spatialdata as sd
    sdata = store().segment_cells(sd.read_zarr(C.SDATA), nuclei=NUCLEI, transcripts=POINTS,
                                  seed_rule=args.seed_rule)
    save_elements(sdata, ("ssaft_cell_boundaries_2d", "ssaft_cell_boundaries_3d",
                          "ssaft_cell_labels_3d", "ssaft_cells"))
    print(f"cells {sdata.tables['ssaft_cells'].n_obs:,}", flush=True)


STEPS = {"down": step_down, "map": step_map, "filter": step_filter, "segment": step_segment}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", nargs="*", default=["all"], choices=["all", *STEPS])
    ap.add_argument("--cls", default=str(C.PLACENTA / "placenta_cls.h5ad"))
    ap.add_argument("--annotated", default=str(C.PLACENTA / "placenta_cls_annotated.h5ad"))
    ap.add_argument("--seed-rule", dest="seed_rule", default="column_all",
                    choices=["column_all", "nucleus"])
    a = ap.parse_args()
    for s in (list(STEPS) if a.step == ["all"] else a.step):
        print(f"== {s}", flush=True)
        STEPS[s](a)
    print("done", flush=True)
