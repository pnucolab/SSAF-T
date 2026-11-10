#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.paths import REPO                                      # noqa: E402
from _common.store import save_elements                              # noqa: E402


SKIN = Path(os.environ.get("SSAFT_SKIN", REPO / "_notebooks/fig5_skin"))
OUT = SKIN / "ssaft"
ZARR = OUT / "spatialdata.zarr"
STORE = OUT / "ssaft_store"
SPLIT_STORE = OUT / "ssaft_store_split"
BACKUP_STORE = OUT / "ssaft_store_pooled_backup"
ANNOTATION = OUT / "annotation_out"
ANNO = ANNOTATION / "wound3d_cls_annotated.h5ad"
MARKERS = ANNOTATION / "celltype_markers.yaml"
CLS = OUT / "wound3d_cls.h5ad"
MAP_INPUT = OUT / "wound3d_map_input_split.h5ad"


DIM = 3
ZSTACK_KEY = "morphology_zstack"
Z_STEP = 3.0
OTSU_MULT = 0.6
KERNEL, BANDWIDTH, SD = "gaussian", 2.5, 1.0
NORM_F = 0.3
METHOD = "lda"
NUCLEI = "nucleus_boundaries"
POINTS = "transcripts"
COLUMN = "cellman_free_annotation"
SPLITCOL = "map_class"
FLAT = "_one"
MIN_ANCHORS = 30
N_COMPS, N_NEIGHBORS, RESOLUTION = 50, 15, 1.0
BLOCK_UM = 10.0


N_SAMPLES = 2

DEVICE, NCORES = os.environ.get("SSAFT_DEVICE", "cuda:0"), 16


def log(*a):
    print(f"[{datetime.datetime.now():%H:%M:%S}]", *a, flush=True)


def analysis(store=STORE):
    import ssaft
    return ssaft.Analysis(ssaft.Dataset(str(store)), ncores=NCORES, verbose=True)


def step_down(args):
    import rapids_singlecell as rsc
    import spatialdata as sd
    adata = analysis().downsample_vector(
        sd.read_zarr(ZARR), z=DIM, image_key=ZSTACK_KEY, element_name=POINTS,
        n_samples=N_SAMPLES, block_um=BLOCK_UM, otsu_mult=OTSU_MULT,
        bandwidth=BANDWIDTH, sampling_distance=SD, kernel=KERNEL,
        revert_image=False, gpu_device=DEVICE)
    log(f"anchors {adata.n_obs:,} x {adata.n_vars} features "
        f"(controls included; see the module docstring)")
    rsc.pp.pca(adata, n_comps=N_COMPS)
    rsc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS)
    rsc.tl.leiden(adata, resolution=RESOLUTION)
    adata.write_h5ad(CLS)
    log(f"-> {CLS.name}  ({adata.obs['leiden'].nunique()} clusters)")


def _prepare_split():
    import anndata as ad
    import numpy as np
    import pandas as pd
    a = ad.read_h5ad(ANNO)
    if COLUMN not in a.obs:
        raise SystemExit(f"{COLUMN!r} not in obs: {list(a.obs.columns)}")
    lab = a.obs[COLUMN].astype(str).values
    leiden = a.obs["leiden"].astype(str).values
    key = np.array([f"{l}|{c}" for l, c in zip(lab, leiden)])
    small = {k for k, v in pd.Series(key).value_counts().items() if v < MIN_ANCHORS}
    cls = np.array([l if (l == "Unknown" or k in small) else k for l, k in zip(lab, key)])
    a.obs[SPLITCOL] = pd.Categorical(cls)
    a.obs[FLAT] = np.zeros(a.n_obs, dtype=np.int64)
    a.write_h5ad(MAP_INPUT)
    log(f"split dictionary: {len(set(lab))} labels -> {len(set(cls))} classes "
        f"({len(small)} groups under {MIN_ANCHORS} anchors left pooled) -> {MAP_INPUT.name}")


def step_map(args):
    import shutil
    import spatialdata as sd
    if not ANNO.exists():
        raise SystemExit(f"{ANNO} not found - annotate {CLS.name} with CeLLMan first")
    _prepare_split()


    skip = {"celltype_maps", "filtered_celltype_maps", "max_probabilities",
            "celltype_marker_score", "celltype_index.csv"}
    if SPLIT_STORE.exists():
        shutil.rmtree(SPLIT_STORE)
    SPLIT_STORE.mkdir(parents=True)
    for p in sorted(STORE.iterdir()):
        if p.name not in skip:
            (shutil.copytree if p.is_dir() else shutil.copy2)(p, SPLIT_STORE / p.name)

    kw = dict(method=METHOD, h5ad=str(MAP_INPUT), column=SPLITCOL, leiden_column=FLAT,
              element_name=POINTS, kernel=KERNEL, bandwidth=BANDWIDTH,
              sampling_distance=SD, gpu_device=DEVICE)
    if MARKERS.exists():
        kw["marker_genes_path"] = str(MARKERS)
    analysis(SPLIT_STORE).map_celltypes(sd.read_zarr(ZARR), **kw)

    _promote(backup=not args.no_backup)


def _promote(backup=True):
    import json
    import numpy as np
    import shutil
    import spatialdata as sd
    import ssaft
    import zarr

    XBLOCK = 1024
    REPLACED = ("celltype_maps", "filtered_celltype_maps", "max_probabilities",
                "celltype_marker_score", "centroids", "cluster_labels")
    zc = zarr.open_group(str(STORE), mode="a")
    zs = zarr.open_group(str(SPLIT_STORE), mode="r")
    names = [str(x) for x in zc["celltype_names"][:]]
    split = [str(x) for x in zs["celltype_names"][:]]
    pos = {n: i for i, n in enumerate(names)}
    assert sorted({x.split("|")[0] for x in split}) == sorted(names), \
        "a split class has no matching label; the two stores disagree"
    lut = np.array([pos[x.split("|")[0]] for x in split], dtype=np.int32)
    log(f"promote: {len(split)} dictionary classes -> {len(names)} labels "
        f"(ids and colours unchanged)")

    if backup:
        if BACKUP_STORE.exists():
            shutil.rmtree(BACKUP_STORE)
        BACKUP_STORE.mkdir(parents=True)
        for k in REPLACED + ("celltype_names",):
            if (STORE / k).exists():
                shutil.copytree(STORE / k, BACKUP_STORE / k)
        (BACKUP_STORE / "attrs.json").write_text(json.dumps(dict(zc.attrs), indent=1, default=str))
        log(f"promote: backed the replaced arrays up to {BACKUP_STORE.name}")


    for k in ("celltype_maps", "filtered_celltype_maps"):
        src, dst = zs[k], zc[k]
        assert src.shape == dst.shape, (k, src.shape, dst.shape)
        n = 0
        for x0 in range(0, src.shape[0], XBLOCK):
            x1 = min(x0 + XBLOCK, src.shape[0])
            b = np.asarray(src[x0:x1])
            out = np.where(b >= 0, lut[np.clip(b, 0, len(lut) - 1)], -1).astype(dst.dtype)
            dst[x0:x1] = out
            n += int((out >= 0).sum())
        log(f"promote: {k} rewritten, {n:,} labelled voxels")
    for k in ("max_probabilities", "celltype_marker_score"):
        if k in zs:
            src, dst = zs[k], zc[k]
            for x0 in range(0, src.shape[0], XBLOCK):
                x1 = min(x0 + XBLOCK, src.shape[0])
                dst[x0:x1] = np.asarray(src[x0:x1])
            log(f"promote: {k} copied from the split run")


    raw = np.asarray(zs["cluster_labels"][:])
    cl = np.where(raw < 0, -1, lut[raw.clip(0, len(lut) - 1)]).astype(np.int64)
    nv = zs["normalized_vectors"]
    cen = np.zeros((len(names), nv.shape[1]), dtype=np.float32)
    for i in range(len(names)):
        m = np.flatnonzero(cl == i)
        if m.size:
            cen[i] = np.asarray(nv[m, :], dtype=np.float64).mean(0)
    del zc["cluster_labels"]; zc["cluster_labels"] = cl
    del zc["centroids"]; zc["centroids"] = cen
    zc.attrs["annotation_h5ad"] = str(zs.attrs.get("annotation_h5ad", ""))
    zc.attrs["mapping_dictionary"] = "per (label, leiden), merged by label name"
    zc.attrs["mapping_dictionary_classes"] = len(split)
    log(f"promote: cluster_labels and centroids rebuilt on the {len(names)} merged classes")


    ds = ssaft.Dataset(str(STORE))
    for k in ("coordinate_offset", "coordinate_bounds", "original_coordinate_bounds"):
        if k in ds.zarr_group:
            setattr(ds, k, ds.zarr_group[k][:])
    ds.sampling_distance = float(ds.zarr_group.attrs.get("sampling_distance", SD))
    ds.filtered_celltype_maps = np.asarray(ds.zarr_group["filtered_celltype_maps"][:])
    sdata = ds.write_celltype_map(sd.read_zarr(ZARR))
    saved = save_elements(sdata, ("ssaft_celltype_map", "ssaft_celltype_map_rgb"))
    log(f"promote: persisted {saved}")


def step_filter(args):
    import numpy as np
    import spatialdata as sd
    an = analysis()
    sdata = an.filter_celltypemaps(sd.read_zarr(ZARR), norm_f=NORM_F, show_unknown=True)
    save_elements(sdata, ("ssaft_celltype_map", "ssaft_celltype_map_rgb"))
    fm = np.asarray(an.dataset.zarr_group["filtered_celltype_maps"][:])
    log(f"filter: {float((fm >= 0).mean()):.2%} of voxels kept")


def _nuclei_z(sdata):
    import numpy as np
    pts = sdata.points[POINTS]
    if "cell_id" not in pts.columns or "z" not in pts.columns:
        return None
    cols = ["z", "cell_id"] + (["overlaps_nucleus"] if "overlaps_nucleus" in pts.columns else [])
    df = pts[cols].compute()
    df["cell_id"] = df["cell_id"].astype(str)
    ids = sdata.shapes[NUCLEI].index.astype(str)
    z = df.groupby("cell_id", observed=True)["z"].median().reindex(ids).to_numpy(dtype=np.float64)
    if "overlaps_nucleus" in cols:
        inn = df[df["overlaps_nucleus"].astype(int) == 1]
        zn = (inn.groupby("cell_id", observed=True)["z"].median()
                 .reindex(ids).to_numpy(dtype=np.float64))
        z = np.where(np.isfinite(zn), zn, z)
    log(f"nucleus z from transcripts: {int(np.isfinite(z).sum()):,} of {len(ids):,}")
    return z if np.isfinite(z).any() else None


def step_segment(args):
    import spatialdata as sd
    an = analysis()
    sdata = sd.read_zarr(ZARR)
    kw = dict(transcripts=POINTS)
    z = _nuclei_z(sdata)
    if z is not None:
        kw["nuclei_z"] = z
    sdata = an.segment_cells(sdata, nuclei=NUCLEI, **kw)
    save_elements(sdata, ("ssaft_cell_boundaries_2d", "ssaft_cell_boundaries_3d",
                          "ssaft_cell_labels_3d", "ssaft_cells"))
    cells = sdata.tables["ssaft_cells"]
    cells.write_h5ad(OUT / "ssaft_cells.h5ad")
    log(f"segment: {cells.n_obs:,} cells")


STEPS = {"down": step_down, "map": step_map, "filter": step_filter, "segment": step_segment}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", nargs="*", default=["all"], choices=["all", *STEPS])
    ap.add_argument("--no-backup", action="store_true",
                    help="do not keep the pooled map in ssaft_store_pooled_backup")
    a = ap.parse_args()
    for s in (list(STEPS) if a.step == ["all"] else a.step):
        log(f"== {s}")
        STEPS[s](a)
    log("done")
