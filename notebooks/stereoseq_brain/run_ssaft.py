#!/usr/bin/env python
from __future__ import annotations

import argparse
import os

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

import config as C                                      # noqa: E402


DIM = 2
BIN_UM = 0.5
BANDWIDTH_UM = 8.0
KERNEL = "gaussian"
SD = 1.0
HVG_N = 5000
N_SAMPLES, BLOCK_UM = 5, 10.0
VF_NORM_PERCENTILE = 25
N_COMPS, N_NEIGHBORS, RESOLUTION = 50, 15, 1.0
NORM_F = 0.3
METHOD = "lda"
NUCLEI = "nucleus_boundaries"
SPECIES, ORGAN = "mouse", "brain"
DEVICE, NCORES = os.environ.get("SSAFT_DEVICE", "cuda"), 30


def store():
    import ssaft
    return ssaft.Analysis(ssaft.Dataset(str(C.SDATA.parent / "ssaft_store")),
                          ncores=NCORES, verbose=True)


def step_prep(args):
    raise SystemExit(
        "prep: read the MOSTA bin1 GEM into a SpatialData on the lattice above. The reader is\n"
        "  SSAF-T/notebooks/stereoseq_mosta.py (load_mosta_stereoseq), which ships with the\n"
        "  tutorial because it is specific to this one release's file layout, not to a platform.\n"
        f"  Inputs: {C.BIN1} and the ssDNA image.")


def step_panel(args):
    import spatialdata as sd
    import ssaft
    sdata = sd.read_zarr(C.SDATA)
    panel, markers = ssaft.select_gene_panel(sdata.tables["table"], species=SPECIES, organ=ORGAN,
                                             hvg_n=HVG_N)
    print(f"panel {len(panel):,} genes = HVG union {len(markers):,} in-data KG markers", flush=True)
    return panel


def step_vf(args):
    import spatialdata as sd
    sdata = sd.read_zarr(C.SDATA)
    store().compute_vf(sdata, bin_um=BIN_UM, panel=step_panel(args), sampling_distance=SD,
                       bandwidth_um=BANDWIDTH_UM, kernel=KERNEL, device=DEVICE)


def step_down(args):
    import rapids_singlecell as rsc
    import spatialdata as sd
    an = store()
    adata = an.downsample_vector(sd.read_zarr(C.SDATA), z=DIM, n_samples_per_block=N_SAMPLES,
                                 block_um=BLOCK_UM, vf_norm_percentile=VF_NORM_PERCENTILE)
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
                          column="cellman_annotation", kernel=KERNEL,
                          marker_genes_path=str(args.markers), gpu_device=DEVICE)


def step_filter(args):
    import spatialdata as sd
    sdata = store().filter_celltypemaps(sd.read_zarr(C.SDATA), norm_f=NORM_F, show_unknown=True)
    for k in ("ssaft_celltype_map", "ssaft_celltype_map_rgb"):
        sdata.write_element(k, overwrite=True)


def step_segment(args):
    import spatialdata as sd
    sdata = store().segment_cells(sd.read_zarr(C.SDATA), nuclei=NUCLEI)
    for k in ("ssaft_cell_boundaries_2d", "ssaft_cells"):
        sdata.write_element(k, overwrite=True)
    print(f"cells {sdata.tables['ssaft_cells'].n_obs:,}", flush=True)


STEPS = {"prep": step_prep, "panel": step_panel, "vf": step_vf, "down": step_down,
         "map": step_map, "filter": step_filter, "segment": step_segment}
ALL = ["vf", "down", "map", "filter", "segment"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", nargs="*", default=["all"], choices=["all", *STEPS])
    ap.add_argument("--cls", default=str(C.SDATA.parent / "mouse_brain_cls.h5ad"))
    ap.add_argument("--annotated", default=str(C.SDATA.parent / "mouse_brain_cls_annotated.h5ad"))
    ap.add_argument("--markers", default=str(C.SDATA.parent / "celltype_markers.yaml"))
    a = ap.parse_args()
    for s in (ALL if a.step == ["all"] else a.step):
        print(f"== {s}", flush=True)
        STEPS[s](a)
    print("done", flush=True)
