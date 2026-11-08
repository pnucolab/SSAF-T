#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

import config as C                                      # noqa: E402


DIM = 3
ZSTACK_KEY = "dapi_zstack"
Z_STEP_IMG = 4.5
N_SAMPLES, BLOCK_UM = 10, 10.0
OTSU_MULT = 1.0
KERNEL, BANDWIDTH, SD = "gaussian", 2.5, 1.0
N_COMPS, N_NEIGHBORS, RESOLUTION = 50, 15, 1.0
NORM_F = 0.3


CELLPOSE_FLOW, CELLPOSE_CELLPROB, MIN_AREA_PX = 0.4, 0.0, 500
DEVICE, NCORES = os.environ.get("SSAFT_DEVICE", "cuda:0"), 30


def store():
    import ssaft
    return ssaft.Analysis(ssaft.Dataset(str(C.SDATA.parent / "ssaft_store")),
                          ncores=NCORES, verbose=True)


def step_prep(args):
    import pandas as pd
    import spatialdata as sd
    import ssaft
    from spatialdata.models import PointsModel

    if C.SDATA.exists():
        return
    tx = pd.read_csv(C.PUB_TX, usecols=["global_x", "global_y", "global_z", "gene"])
    tx = tx.rename(columns={"global_x": "x", "global_y": "y", "global_z": "z"})
    keep = ~tx.gene.astype(str).str.lower().str.startswith(C.CONTROL_PREFIX)
    pts = PointsModel.parse(tx[keep], coordinates={"x": "x", "y": "y", "z": "z"},
                            feature_key="gene")
    sdata = sd.SpatialData(points={C.POINTS: pts})
    sdata.images[ZSTACK_KEY] = ssaft.Analysis.zstack_image(
        sorted(C.PUB_IMAGES.glob("mosaic_DAPI_z*.tif")), z_step=Z_STEP_IMG)
    C.SDATA.parent.mkdir(parents=True, exist_ok=True)
    sdata.write(C.SDATA)


def step_nuclei(args):
    raise SystemExit(
        "nucleus detection is a separate step: run Cellpose ('nuclei' model) on the DAPI mosaic\n"
        f"  {C.PUB_IMAGES}/mosaic_DAPI_z3.tif\n"
        f"  diameter = median of Cellpose's SizeModel estimate over the tissue (~84 px at "
        f"{C.PX_PER_UM:.4f} px/µm), flow_threshold {CELLPOSE_FLOW}, "
        f"cellprob_threshold {CELLPOSE_CELLPROB}, minimum area {MIN_AREA_PX} px,\n"
        "  each nucleus emitted only by the tile whose core holds its centroid,\n"
        f"  then written into the store as shapes['{C.NUCLEI}'] in microns.")


def step_down(args):
    import rapids_singlecell as rsc
    import spatialdata as sd
    an = store()
    adata = an.downsample_vector(
        sd.read_zarr(C.SDATA), z=DIM, image_key=ZSTACK_KEY, element_name=C.POINTS,
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
        raise SystemExit(f"{args.annotated} not found — annotate {args.cls} with CeLLMan first")
    store().map_celltypes(sd.read_zarr(C.SDATA), method="lda", h5ad=str(args.annotated),
                          column=C.LABEL_COL, element_name=C.POINTS, kernel=KERNEL,
                          bandwidth=BANDWIDTH, sampling_distance=SD, gpu_device=DEVICE)


def step_filter(args):
    import spatialdata as sd
    sdata = store().filter_celltypemaps(sd.read_zarr(C.SDATA), norm_f=NORM_F, show_unknown=True)
    for k in ("ssaft_celltype_map", "ssaft_celltype_map_rgb"):
        sdata.write_element(k, overwrite=True)


def step_segment(args):
    import spatialdata as sd
    sdata = store().segment_cells(sd.read_zarr(C.SDATA), nuclei=C.NUCLEI, transcripts=C.POINTS)
    for k in (C.SHAPES_2D, C.SHAPES_3D, "ssaft_cell_labels_3d", C.TABLE):
        sdata.write_element(k, overwrite=True)
    print(f"cells {sdata.tables[C.TABLE].n_obs:,}", flush=True)


STEPS = {"prep": step_prep, "nuclei": step_nuclei, "down": step_down, "map": step_map,
         "filter": step_filter, "segment": step_segment}
ALL = ["prep", "down", "map", "filter", "segment"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", nargs="*", default=["all"], choices=["all", *STEPS])
    ap.add_argument("--cls", default=str(C.SDATA.parent / "liver_cls.h5ad"))
    ap.add_argument("--annotated", default=str(C.SDATA.parent / "liver_cls_annotated.h5ad"))
    a = ap.parse_args()
    for s in (ALL if a.step == ["all"] else a.step):
        print(f"== {s}", flush=True)
        STEPS[s](a)
    print("done", flush=True)
