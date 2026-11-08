#!/usr/bin/env python
from __future__ import annotations

import argparse
import os

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

import config as C                                      # noqa: E402


DIM = 3
UM_PX, Z_STEP = 0.2125, 3.0
ZSTACK_KEY = "ssaft_zstack"
IMAGE_KEY = "morphology_focus"
N_SAMPLES, BLOCK_UM = 3, 10.0
OTSU_MULT = 1.0


KERNEL, BANDWIDTH, SD = "gaussian", 2.5, 1.0
N_COMPS, N_NEIGHBORS, RESOLUTION = 50, 15, 1.0
PCA_FIT_N, PCA_CHUNK = 1_000_000, 500_000
NORM_F = 0.3
METHOD = "lda"
NUCLEI = "nuclei_cellpose"
EXCLUDE = ["cardiomyocyte"]

MERGE = {"ventricular cardiomyocyte": "cardiomyocyte", "atrial cardiomyocyte": "cardiomyocyte"}
COLUMN = "cellman_annotation"
DEVICE, NCORES = os.environ.get("SSAFT_DEVICE", "cuda:0"), 16


def paths(section):
    d = C.HEART / section / "ssaft"
    return dict(store=d / "ssaft_store", zarr=d / "spatialdata.zarr",
                anchors=d / "anchors.h5ad", cls=d / f"{section}_cls.h5ad",
                annotated=d / f"annotation_out/{section}_cls_annotated_cmmerged.h5ad",
                markers=d / "annotation_out/celltype_markers.yaml",
                cells=d / "ssaft_cells.h5ad")


def analysis(section):
    import ssaft
    return ssaft.Analysis(ssaft.Dataset(str(paths(section)["store"])), ncores=NCORES, verbose=True)


def step_census(section, p):
    raise SystemExit(
        "census: count the transcripts at each morphology plane of this section and keep the\n"
        "planes holding about 97% of them; record the range in config.py Z_RANGE. It is a\n"
        "property of the section, not of the platform.")


def step_zstack(section, p):
    raise SystemExit(
        "zstack: write planes Z_RANGE[0]..Z_RANGE[1] of morphology.ome.tif as '<name>_z<N>.tif'\n"
        f"and stack them with ssaft.Analysis.zstack_image(..., z_step={Z_STEP}), taking the x/y\n"
        f"placement ({UM_PX} um/px) from the 2D morphology element, then store the result as\n"
        f"images['{ZSTACK_KEY}'] beside points['transcripts'] in spatialdata.zarr.")


def step_nuclei(section, p):
    raise SystemExit(
        "nuclei: run Cellpose on the DAPI planes in use and write the result into the store as\n"
        f"shapes['{NUCLEI}'], whose 'z' column is the PLANE INDEX; segment_cells scales it by\n"
        f"z_step = {Z_STEP} to get microns.")


def step_down(section, p):
    import rapids_singlecell as rsc
    import spatialdata as sd
    an = analysis(section)
    adata = an.downsample_vector(
        sd.read_zarr(p["zarr"]), z=DIM, image_key=ZSTACK_KEY, element_name="transcripts",
        n_samples=N_SAMPLES, block_um=BLOCK_UM, otsu_mult=OTSU_MULT,
        bandwidth=BANDWIDTH, sampling_distance=SD, kernel=KERNEL,
        revert_image=False, gpu_device=DEVICE)
    if adata.n_obs == 0:
        raise SystemExit("no anchors: check the stain polarity (revert_image) and otsu_mult")
    print(f"anchors {adata.n_obs:,} x {adata.n_vars} genes", flush=True)
    adata.write_h5ad(p["anchors"])
    rsc.pp.pca(adata, n_comps=N_COMPS)
    rsc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS)
    rsc.tl.leiden(adata, resolution=RESOLUTION)
    adata.write_h5ad(p["cls"])
    print(f"-> {p['cls']}  ({adata.obs['leiden'].nunique()} clusters)", flush=True)


def step_map(section, p):
    import spatialdata as sd
    if not p["annotated"].exists():
        raise SystemExit(f"{p['annotated']} not found - annotate {p['cls']} with CeLLMan first, "
                         f"then merge {sorted(MERGE)} into 'cardiomyocyte'")
    analysis(section).map_celltypes(
        sd.read_zarr(p["zarr"]), method=METHOD, h5ad=str(p["annotated"]), column=COLUMN,
        element_name="transcripts", kernel=KERNEL, bandwidth=BANDWIDTH, sampling_distance=SD,
        marker_genes_path=str(p["markers"]), gpu_device=DEVICE)


def step_filter(section, p):
    import spatialdata as sd
    sdata = analysis(section).filter_celltypemaps(sd.read_zarr(p["zarr"]), norm_f=NORM_F,
                                                  show_unknown=True)
    for k in ("ssaft_celltype_map", "ssaft_celltype_map_rgb"):
        sdata.write_element(k, overwrite=True)


def step_segment(section, p):
    import spatialdata as sd
    an = analysis(section)
    sdata = sd.read_zarr(p["zarr"])
    nz = sdata.shapes[NUCLEI]["z"].to_numpy().astype(float) * Z_STEP
    sdata = an.segment_cells(sdata, nuclei=NUCLEI, nuclei_z=nz, nuclei_frame="intrinsic",
                             transcripts="transcripts", exclude_celltypes=EXCLUDE)
    for k in ("ssaft_cell_boundaries_2d", "ssaft_cell_boundaries_3d", "ssaft_cell_labels_3d",
              "ssaft_cells"):
        sdata.write_element(k, overwrite=True)
    cells = sdata.tables["ssaft_cells"]
    cells.write_h5ad(p["cells"])
    o = cells.obs
    print(f"cells {cells.n_obs:,} - segmented {int(o['segmented'].sum()):,}, "
          f"whole bodies {int((~o['segmented']).sum()):,}", flush=True)


STEPS = {"census": step_census, "zstack": step_zstack, "nuclei": step_nuclei, "down": step_down,
         "map": step_map, "filter": step_filter, "segment": step_segment}
ALL = ["down", "map", "filter", "segment"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--section", required=True, choices=C.SECTIONS)
    ap.add_argument("--step", nargs="*", default=["all"], choices=["all", *STEPS])
    a = ap.parse_args()
    pth = paths(a.section)
    for s in (ALL if a.step == ["all"] else a.step):
        print(f"== {a.section} {s}", flush=True)
        STEPS[s](a.section, pth)
    print("done", flush=True)
