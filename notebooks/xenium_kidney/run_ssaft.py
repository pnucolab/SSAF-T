#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

import config as C                                      # noqa: E402


DIM = 3
ZSTACK_KEY = "morphology_zstack"
Z_STEP = 3.0
N_SAMPLES, BLOCK_UM = 10, 10.0
OTSU_MULT = 0.6
KERNEL, BANDWIDTH, SD = "gaussian", 2.5, 1.0
N_COMPS, N_NEIGHBORS, RESOLUTION = 50, 15, 1.0
NORM_F = 0.3


NUCLEI = "nucleus_boundaries"
DEVICE, NCORES = os.environ.get("SSAFT_DEVICE", "cuda:0"), 16


def build_sdata():
    import spatialdata as sd
    import spatialdata_io
    import ssaft

    zarr = C.SDATA
    if zarr.exists():
        return sd.read_zarr(zarr)

    sdata = spatialdata_io.xenium(str(C.BUNDLE), n_jobs=NCORES)
    pts = sdata.points[C.POINTS]
    keep = ~pts["feature_name"].astype(str).str.startswith(C.CONTROL_PREFIX)
    sdata.points[C.POINTS] = pts[keep]

    sdata.images[ZSTACK_KEY] = ssaft.Analysis.zstack_image(
        str(C.BUNDLE / "morphology.ome.tif"), like=sdata.images[C.DAPI_IMAGE], z_step=Z_STEP)
    zarr.parent.mkdir(parents=True, exist_ok=True)
    sdata.write(zarr)
    return sdata


def store():
    import ssaft
    return ssaft.Analysis(ssaft.Dataset(str(C.SDATA.parent / "ssaft_store")),
                          ncores=NCORES, verbose=True)


def step_down(args):
    import rapids_singlecell as rsc
    sdata = build_sdata()
    an = store()
    adata = an.downsample_vector(
        sdata, z=DIM, image_key=ZSTACK_KEY, element_name=C.POINTS,
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
    if not Path(args.annotated).exists():
        raise SystemExit(f"{args.annotated} not found — annotate {args.cls} with CeLLMan first "
                         f"(annotate-mode writes obs['cellman_annotation'])")
    an = store()
    an.map_celltypes(sd.read_zarr(C.SDATA), method="lda", h5ad=str(args.annotated),
                     column=C.LABEL_COL, element_name=C.POINTS, kernel=KERNEL,
                     bandwidth=BANDWIDTH, sampling_distance=SD, gpu_device=DEVICE)


def step_filter(args):
    import spatialdata as sd
    an = store()
    sdata = an.filter_celltypemaps(sd.read_zarr(C.SDATA), norm_f=NORM_F, show_unknown=True)
    for k in ("ssaft_celltype_map", "ssaft_celltype_map_rgb"):
        sdata.write_element(k, overwrite=True)


def step_segment(args):
    import spatialdata as sd
    an = store()
    sdata = an.segment_cells(sd.read_zarr(C.SDATA), nuclei=NUCLEI, transcripts=C.POINTS)
    for k in (C.SHAPES_2D, C.SHAPES_3D, "ssaft_cell_labels_3d", C.TABLE):
        sdata.write_element(k, overwrite=True)
    print(f"cells {sdata.tables[C.TABLE].n_obs:,}", flush=True)


STEPS = {"down": step_down, "map": step_map, "filter": step_filter, "segment": step_segment}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", nargs="*", default=["all"], choices=["all", *STEPS])
    ap.add_argument("--cls", default=str(C.SDATA.parent / "kidney_cls.h5ad"))
    ap.add_argument("--annotated", default=str(C.SDATA.parent / "kidney_cls_annotated.h5ad"))
    a = ap.parse_args()
    for s in (list(STEPS) if a.step == ["all"] else a.step):
        print(f"== {s}", flush=True)
        STEPS[s](a)
    print("done", flush=True)
