"""Read count platforms straight from the SpatialData ``spatialdata_io`` returns.

``spatialdata_io.visium_hd`` and ``spatialdata_io.stereoseq`` give the counts and the
staining image in their own pixel frames, with no µm contract: Visium HD bins sit at
full-resolution H&E pixels in a coordinate system named after the dataset, Stereo-seq
bins at DNB coordinates in ``'global'``. The count backend works on a µm lattice, so the
reader's output is turned into a :class:`~ssaft.GridDataset` view here, in memory, and the
input SpatialData is never modified.

The view carries a **frame** in ``meta['frame']``:

* ``coordinate_system`` -- the reader's coordinate system the bins and image live in;
* ``um_to_cs`` -- ``(3, 3)`` affine from the µm lattice to that coordinate system.

:meth:`ssaft.Analysis.compute_vf` persists it in the store, so the cell-type map, its RGB
rendering and the segmented cells are written back into that coordinate system and
overlay the reader's own elements, in any later process.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

COUNT_READERS = ("visium_hd", "stereoseq")


def reader_of(sdata) -> Optional[str]:
    """Name of the ``spatialdata_io`` reader that built ``sdata``, or ``None``."""
    attrs = getattr(sdata, "attrs", None) or {}
    return attrs.get("spatialdata_io_reader")


def is_count_reader_output(sdata) -> bool:
    """True for an unconverted ``spatialdata_io.visium_hd`` / ``stereoseq`` SpatialData."""
    if reader_of(sdata) not in COUNT_READERS:
        return False
    return not any("cellman" in getattr(t, "uns", {}) for t in sdata.tables.values())


def _image_array(el):
    """Full-resolution ``(c, y, x)`` array and the xarray node it came from."""
    from xarray import DataArray
    if not isinstance(el, DataArray):
        keys = list(el.keys())
        node = el["scale0"] if "scale0" in keys else el[keys[0]]
        names = list(node.data_vars) if hasattr(node, "data_vars") else list(node.keys())
        el = node["image"] if "image" in names else node[names[0]]
    return el


def _to_cs(el, cs):
    from spatialdata.transformations import get_transformation
    t = get_transformation(el, get_all=True).get(cs)
    if t is None:
        return None
    return np.asarray(t.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")), dtype=float)


def _grid(raw, image_el, cs, um_to_cs, *, platform, he_revert, table_key, image_key,
          species, organ, meta, load_image):
    """Assemble the GridDataset view; the image affine is image px -> cs -> µm."""
    from .._griddataset import GridDataset

    img_to_cs = _to_cs(image_el, cs)
    if img_to_cs is None:
        raise ValueError(f"images[{image_key!r}] has no transformation to {cs!r}")
    cs_to_um = np.linalg.inv(um_to_cs)
    px_to_um = (cs_to_um @ img_to_cs)[:2, :3]
    um_per_px = float(np.sqrt(abs(np.linalg.det(px_to_um[:, :2]))))

    arr = None
    if load_image:
        arr = np.asarray(_image_array(image_el).values)
        if arr.ndim == 3 and arr.shape[0] in (3, 4):
            arr = np.moveaxis(arr[:3], 0, -1)
        elif arr.ndim == 3:
            arr = arr[0]

    meta = dict(meta)
    meta["frame"] = {"coordinate_system": cs, "um_to_cs": um_to_cs.tolist(),
                     "reader": platform, "table_key": table_key, "image_key": image_key}
    return GridDataset(
        raw=raw, image=arr, um_per_px=um_per_px, species=species, organ=organ,
        platform=platform, origin=tuple(float(v) for v in np.asarray(raw.obsm["spatial"]).min(0)),
        affine_px_to_um=px_to_um, he_revert=he_revert, meta=meta)


def _visium_hd(sdata, *, bin_um, table_key, image_key, species, organ, load_image):
    import anndata as ad
    import pandas as pd
    from ._grid import fit_affine_lstsq

    tables = {k: t for k, t in sdata.tables.items() if "square_" in k}
    if table_key is None:
        want = f"square_{int(bin_um):03d}um" if bin_um is not None else None
        cands = sorted(k for k in tables if want is None or want in k)
        if not cands:
            raise ValueError(f"no Visium HD bin table{f' for {bin_um} µm' if bin_um else ''}; "
                             f"have {sorted(sdata.tables)}")
        table_key = cands[0]                                    # square_002um sorts first
    tab = sdata.tables[table_key]
    if bin_um is None:
        bin_um = int(table_key.split("square_")[1][:3])
    missing = [c for c in ("array_row", "array_col") if c not in tab.obs]
    if missing or "spatial" not in tab.obsm:
        raise ValueError(f"tables[{table_key!r}] lacks {missing or ['obsm[spatial]']}")

    region = tab.uns.get("spatialdata_attrs", {}).get("region")
    region = region[0] if isinstance(region, (list, tuple)) else region
    shapes = sdata.shapes.get(region) if region is not None else None
    cs = None
    if shapes is not None:
        from spatialdata.transformations import get_transformation
        cs = next((k for k, t in get_transformation(shapes, get_all=True).items()
                   if np.allclose(t.to_affine_matrix(("x", "y"), ("x", "y")), np.eye(3))), None)
    if image_key is None:
        image_key = next((k for k in sdata.images if k.endswith("_full_image")),
                         next((k for k in sdata.images if k.endswith("_hires_image")), None))
        if image_key is None:
            raise ValueError(f"no full-resolution or hires image; have {sorted(sdata.images)}")
    if cs is None:
        cs = next(iter(_to_cs_all(sdata.images[image_key])))

    keep = tab.obs["in_tissue"].astype(bool).to_numpy() if "in_tissue" in tab.obs else None
    if keep is not None and keep.all():
        keep = None
    X = tab.X if keep is None else tab.X[keep]
    obs = tab.obs if keep is None else tab.obs[keep]
    px = np.asarray(tab.obsm["spatial"], dtype=np.float64)
    px = px if keep is None else px[keep]
    col, row = obs["array_col"].to_numpy(), obs["array_row"].to_numpy()

    raw = ad.AnnData(X=X, obs=pd.DataFrame(index=obs.index.copy()), var=tab.var.copy())
    raw.var_names_make_unique()
    raw.obsm["spatial"] = np.stack([col, row], axis=1).astype(np.float64) * float(bin_um)

    # the bins' (col, row) against their cs pixels, fitted on the in-tissue bins;
    # µm = lattice * bin_um, so cs px -> µm is the inverse fit scaled by bin_um
    px_to_lattice = fit_affine_lstsq(col, row, px[:, 0], px[:, 1])
    um_to_cs = np.linalg.inv(np.vstack([px_to_lattice * float(bin_um), [0.0, 0.0, 1.0]]))
    return _grid(raw, sdata.images[image_key], cs, um_to_cs, platform="visiumhd",
                 he_revert=True, table_key=table_key, image_key=image_key,
                 species=species, organ=organ, meta={"bin_um": bin_um}, load_image=load_image)


def _to_cs_all(el):
    from spatialdata.transformations import get_transformation
    return get_transformation(el, get_all=True)


def _stereoseq(sdata, *, bin_um, table_key, image_key, species, organ, load_image, dnb_um=0.5):
    import anndata as ad
    import pandas as pd

    if table_key is None:
        table_key = "bin1_table"
    if table_key not in sdata.tables:
        raise ValueError(f"no {table_key!r}; read the SAW output with "
                         f"spatialdata_io.stereoseq(read_square_bin=True). Have {sorted(sdata.tables)}")
    tab = sdata.tables[table_key]
    attrs = tab.uns.get("spatialdata_attrs", {})
    region = attrs.get("region")
    region = region[0] if isinstance(region, (list, tuple)) else region
    ikey = attrs.get("instance_key", "instance_id")
    pts = sdata.points[region]
    xy = pts[["x", "y"]].compute().to_numpy(dtype=np.float64)
    inst = tab.obs[ikey].to_numpy().astype(np.int64)
    bin_size = int("".join(ch for ch in table_key.split("_")[0] if ch.isdigit()) or 1)

    raw = ad.AnnData(X=tab.X, obs=pd.DataFrame(index=tab.obs.index.copy()), var=tab.var.copy())
    raw.var_names_make_unique()
    # bin coordinates are DNB indices; a bin's centre is half a bin past its corner
    raw.obsm["spatial"] = (xy[inst] + (bin_size - 1) / 2.0) * dnb_um

    if image_key is None:
        regist = sorted(k for k in sdata.images if k.endswith("_regist"))
        image_key = next((k for k in regist if "HE_" not in k and not k.endswith("HE_regist")),
                         regist[0] if regist else None)
        if image_key is None:
            raise ValueError(f"no registered image (*_regist); have {sorted(sdata.images)}")
    he = "HE" in image_key.split("_")
    cs = "global"
    um_to_cs = np.diag([1.0 / dnb_um, 1.0 / dnb_um, 1.0])
    pts_to_cs = _to_cs(pts, cs)
    if pts_to_cs is not None:
        um_to_cs = pts_to_cs @ um_to_cs
    return _grid(raw, sdata.images[image_key], cs, um_to_cs, platform="stereoseq",
                 he_revert=he, table_key=table_key, image_key=image_key,
                 species=species, organ=organ, meta={"bin_um": bin_size * dnb_um},
                 load_image=load_image)


def grid_from_spatialdata_io(sdata, *, table_key=None, image_key=None, bin_um=None,
                             species="unknown", organ="unknown", load_image=True):
    """GridDataset view of an unconverted ``spatialdata_io`` count SpatialData.

    :param sdata: SpatialData from ``spatialdata_io.visium_hd`` or ``spatialdata_io.stereoseq``.
    :param table_key: Bin table; default the 2 µm bins (Visium HD) or ``'bin1_table'`` (Stereo-seq).
    :param image_key: Staining image; default the full-resolution H&E (Visium HD) or the
        registered ssDNA (Stereo-seq).
    :param bin_um: Visium HD bin size to pick when several were loaded.
    :param load_image: Read the image into memory (not needed to count bins per cell).
    """
    reader = reader_of(sdata)
    if reader == "visium_hd":
        return _visium_hd(sdata, bin_um=bin_um, table_key=table_key, image_key=image_key,
                          species=species, organ=organ, load_image=load_image)
    if reader == "stereoseq":
        return _stereoseq(sdata, bin_um=bin_um, table_key=table_key, image_key=image_key,
                          species=species, organ=organ, load_image=load_image)
    raise ValueError(f"not a spatialdata_io count reader output (reader={reader!r})")
