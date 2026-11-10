#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

import config as C                                      # noqa: E402
from _common.store import save_elements                 # noqa: E402


DIM = 2
BIN_UM = 2.0
KERNEL = "gaussian"


BANDWIDTH = 2.5
SD = 2.0
PRUNE = 4.3
N_SAMPLES, BLOCK_UM = 5, 10.0
VF_NORM_PERC = 25
N_COMPS, N_NEIGHBORS, RESOLUTION = 50, 15, 1.0
NORM_F = 0.3


NUCLEI = "enact_nucleus_boundaries"
DEVICE, NCORES = os.environ.get("SSAFT_DEVICE", "cuda:0"), 16


def store():
    import ssaft
    return ssaft.Analysis(ssaft.Dataset(str(C.SDATA.parent / "ssaft_store")),
                          ncores=NCORES, verbose=True)


def signal_genes(adata):
    import scanpy as sc
    from common import kg_markers
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
    hvg = set(adata.var_names[adata.var.highly_variable])
    keep = sorted((hvg | set(kg_markers())) & set(map(str, adata.var_names)))
    print(f"signal genes {len(keep):,} = HVG {len(hvg):,} union KG markers", flush=True)
    return keep


def step_vf(args):
    import spatialdata as sd
    sdata = sd.read_zarr(C.SDATA)
    panel = signal_genes(sdata.tables[C.BIN_TABLE])
    store().compute_vf(sdata, bin_um=BIN_UM, panel=panel, sampling_distance=SD,
                       bandwidth_um=BANDWIDTH, kernel=KERNEL, prune_coefficient=PRUNE,
                       device=DEVICE)


def step_down(args):
    import rapids_singlecell as rsc
    import spatialdata as sd
    an = store()
    sdata = sd.read_zarr(C.SDATA)
    adata = an.downsample_vector(
        sdata, z=DIM, element_name=C.BIN_TABLE, n_samples=N_SAMPLES, block_um=BLOCK_UM,
        vf_norm_percentile=VF_NORM_PERC,
        bandwidth=BANDWIDTH, sampling_distance=SD, kernel=KERNEL, prune_coefficient=PRUNE,
        genes=signal_genes(sdata.tables[C.BIN_TABLE]), gpu_device=DEVICE)
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
                          column=C.LABEL_COL, kernel=KERNEL, bandwidth=BANDWIDTH,
                          sampling_distance=SD, gpu_device=DEVICE)


def step_filter(args):
    import spatialdata as sd
    sdata = store().filter_celltypemaps(sd.read_zarr(C.SDATA), norm_f=NORM_F, show_unknown=True)
    save_elements(sdata, ("ssaft_celltype_map", "ssaft_celltype_map_rgb"))


def step_segment(args):
    import spatialdata as sd
    sdata = store().segment_cells(sd.read_zarr(C.SDATA), nuclei=NUCLEI)
    save_elements(sdata, (C.SHAPES_2D, C.TABLE))
    print(f"cells {sdata.tables[C.TABLE].n_obs:,}", flush=True)


STEPS = {"vf": step_vf, "down": step_down, "map": step_map, "filter": step_filter,
         "segment": step_segment}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", nargs="*", default=["all"], choices=["all", *STEPS])
    ap.add_argument("--cls", default=str(C.SDATA.parent / "small_intestine_cls.h5ad"))
    ap.add_argument("--annotated",
                    default=str(C.SDATA.parent / "small_intestine_cls_annotated.h5ad"))
    a = ap.parse_args()
    for s in (list(STEPS) if a.step == ["all"] else a.step):
        print(f"== {s}", flush=True)
        STEPS[s](a)
    print("done", flush=True)
