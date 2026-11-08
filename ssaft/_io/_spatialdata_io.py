"""Read count platforms directly from the SpatialData that ``spatialdata_io`` returns.

``spatialdata_io.visium_hd`` and ``spatialdata_io.stereoseq`` give the counts and the
staining image in their own pixel frames, with no micrometre contract. Visium HD bins
sit at full-resolution image pixels in a coordinate system named after the dataset,
and Stereo-seq bins sit at DNB coordinates in the coordinate system called ``global``.
The count backend works on a micrometre lattice, so the reader's output is turned into
a :class:`~ssaft.GridDataset` view here, in memory. The input SpatialData is never
modified.

The view carries a frame in ``meta['frame']`` with two parts. The entry
``coordinate_system`` names the reader coordinate system that the bins and the image
live in, and ``um_to_cs`` is the 3 by 3 affine from the micrometre lattice to that
coordinate system.

:meth:`ssaft.Analysis.compute_vf` persists that frame in the store. As a result the
cell-type map, its RGB rendering and the segmented cells are all written back into the
reader's coordinate system and overlay the reader's own elements, in any later process.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

COUNT_READERS = ("visium_hd", "stereoseq")


def reader_of(sdata) -> Optional[str]:
    """Return the name of the ``spatialdata_io`` reader that built ``sdata``.

    :param sdata: The SpatialData object to inspect.
    :type sdata: spatialdata.SpatialData
    :returns: The reader name recorded in ``sdata.attrs``, or ``None`` when the object
        carries no such record.
    :rtype: str or None
    """
    attrs = getattr(sdata, "attrs", None) or {}
    return attrs.get("spatialdata_io_reader")


def is_count_reader_output(sdata) -> bool:
    """Report whether ``sdata`` is an unconverted count-reader output.

    The object qualifies when it was built by ``spatialdata_io.visium_hd`` or
    ``spatialdata_io.stereoseq`` and none of its tables has yet been given the
    ``uns['cellman']`` metadata that conversion adds.

    :param sdata: The SpatialData object to inspect.
    :type sdata: spatialdata.SpatialData
    :returns: ``True`` when the object still needs conversion, ``False`` otherwise.
    :rtype: bool
    """
    if reader_of(sdata) not in COUNT_READERS:
        return False
    return not any("cellman" in getattr(t, "uns", {}) for t in sdata.tables.values())


def _image_array(el):
    """Return the full-resolution image node of a possibly multiscale element.

    A plain DataArray is returned unchanged. A multiscale element is resolved to its
    ``scale0`` node, or to its first node when no ``scale0`` exists, and then to the
    ``image`` variable of that node, or to the node's first variable.

    :param el: An image element, either a DataArray or a multiscale DataTree.
    :returns: The DataArray holding the image, with axes ordered as channel, y and x.
    :rtype: xarray.DataArray
    """
    from xarray import DataArray
    if not isinstance(el, DataArray):
        keys = list(el.keys())
        node = el["scale0"] if "scale0" in keys else el[keys[0]]
        names = list(node.data_vars) if hasattr(node, "data_vars") else list(node.keys())
        el = node["image"] if "image" in names else node[names[0]]
    return el


def _to_cs(el, cs):
    """Return the affine that maps an element into one coordinate system.

    :param el: A SpatialData element that carries transformations.
    :param cs: Name of the target coordinate system.
    :type cs: str
    :returns: The 3 by 3 affine matrix taking the element's own axes to ``cs``, or ``None``
        when the element has no transformation to that coordinate system.
    :rtype: numpy.ndarray or None
    """
    from spatialdata.transformations import get_transformation
    t = get_transformation(el, get_all=True).get(cs)
    if t is None:
        return None
    return np.asarray(t.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")), dtype=float)


def _grid(raw, image_el, cs, um_to_cs, *, platform, he_revert, table_key, image_key,
          species, organ, meta, load_image):
    """Assemble the GridDataset view, composing the image affine through the frame.

    The image affine is built as a chain: image pixels are mapped into the reader
    coordinate system, and that system is mapped into micrometres by the inverse of
    ``um_to_cs``. The micrometres per pixel are derived from the determinant of the
    resulting matrix, so no pixel size needs to be supplied.

    :param raw: Bin-level counts with ``obsm['spatial']`` already in micrometres.
    :type raw: anndata.AnnData
    :param image_el: The staining image element, as the reader stored it.
    :param cs: Name of the reader coordinate system that the bins and image share.
    :type cs: str
    :param um_to_cs: The 3 by 3 affine from the micrometre lattice to ``cs``.
    :type um_to_cs: numpy.ndarray
    :param platform: Platform name recorded on the view.
    :type platform: str
    :param he_revert: Whether the staining is bright-field, so that the tissue mask must
        invert it before thresholding.
    :type he_revert: bool
    :param table_key: Key of the table the counts came from, recorded in the frame.
    :type table_key: str
    :param image_key: Key of the image element, recorded in the frame.
    :type image_key: str
    :param species: Species name used for marker-panel lookup.
    :type species: str
    :param organ: Cell-Ontology module name for the gene panel.
    :type organ: str
    :param meta: Extra metadata to record on the view. The frame is added to a copy of it.
    :type meta: dict
    :param load_image: Whether to read the image pixels into memory. Counting bins per cell
        does not need them.
    :type load_image: bool
    :returns: The in-memory view of the reader's output.
    :rtype: ssaft.GridDataset
    :raises ValueError: If the image has no transformation to ``cs``.
    """
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
    """Build the GridDataset view of a ``spatialdata_io.visium_hd`` SpatialData.

    The bin table is chosen by bin size, the bins outside tissue are dropped, and each
    remaining bin is placed at its lattice index multiplied by the bin size in
    micrometres. The frame is fitted from the bins themselves: their lattice indices are
    regressed against the pixel positions the reader stored in ``obsm['spatial']``, and
    the inverse of that fit, scaled by the bin size, gives the affine from micrometres to
    the reader coordinate system.

    The coordinate system is taken to be the one in which the bins' own shapes element is
    the identity, which is the frame the reader places the counts in. When that element
    cannot be found, the first coordinate system of the chosen image is used instead.

    :param sdata: SpatialData returned by ``spatialdata_io.visium_hd``.
    :type sdata: spatialdata.SpatialData
    :param bin_um: Bin size in micrometres to select. When ``None`` the smallest available
        bin table is used and its size is read back from the table key.
    :type bin_um: int or None
    :param table_key: Explicit bin table key, which overrides the selection by bin size.
    :type table_key: str or None
    :param image_key: Explicit staining image key. Defaults to the full-resolution image,
        and failing that to the high-resolution image.
    :type image_key: str or None
    :param species: Species name used for marker-panel lookup.
    :type species: str
    :param organ: Cell-Ontology module name for the gene panel.
    :type organ: str
    :param load_image: Whether to read the image pixels into memory.
    :type load_image: bool
    :returns: The in-memory view of the reader's output.
    :rtype: ssaft.GridDataset
    :raises ValueError: If no bin table matches, or if the chosen table lacks the lattice
        indices or the pixel positions, or if no suitable image is present.
    """
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
    """Return every transformation an element carries, keyed by coordinate system.

    :param el: A SpatialData element that carries transformations.
    :returns: A mapping from coordinate-system name to transformation.
    :rtype: dict
    """
    from spatialdata.transformations import get_transformation
    return get_transformation(el, get_all=True)


def _stereoseq(sdata, *, bin_um, table_key, image_key, species, organ, load_image, dnb_um=0.5):
    """Build the GridDataset view of a ``spatialdata_io.stereoseq`` SpatialData.

    Bin positions come from the points element that the table annotates, indexed by the
    table's instance key. A bin's coordinates are DNB indices of its corner, so half a bin
    is added to reach its centre, and the result is scaled by the DNB pitch to give
    micrometres. The affine from micrometres to the reader's ``global`` coordinate system
    is therefore the inverse DNB pitch, composed with whatever transformation the points
    element itself carries.

    The staining image defaults to a registered image whose key ends in ``_regist``,
    preferring one that is not the bright-field image. When the bright-field image is the
    one selected, the tissue mask is told to invert it.

    :param sdata: SpatialData returned by ``spatialdata_io.stereoseq``.
    :type sdata: spatialdata.SpatialData
    :param bin_um: Accepted for signature compatibility with the Visium HD branch. The bin
        size is always read from the table key instead.
    :type bin_um: float or None
    :param table_key: Bin table key. Defaults to ``'bin1_table'``.
    :type table_key: str or None
    :param image_key: Explicit staining image key. Defaults as described above.
    :type image_key: str or None
    :param species: Species name used for marker-panel lookup.
    :type species: str
    :param organ: Cell-Ontology module name for the gene panel.
    :type organ: str
    :param load_image: Whether to read the image pixels into memory.
    :type load_image: bool
    :param dnb_um: Pitch of the DNB lattice in micrometres.
    :type dnb_um: float
    :returns: The in-memory view of the reader's output.
    :rtype: ssaft.GridDataset
    :raises ValueError: If the requested bin table is absent, or if no registered image is
        present.
    """
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
    """Return a GridDataset view of an unconverted ``spatialdata_io`` count SpatialData.

    The reader that built the object decides which branch runs. The input SpatialData is
    read but never modified.

    :param sdata: SpatialData from ``spatialdata_io.visium_hd`` or
        ``spatialdata_io.stereoseq``.
    :type sdata: spatialdata.SpatialData
    :param table_key: Bin table key. Defaults to the 2 micrometre bins for Visium HD and to
        ``'bin1_table'`` for Stereo-seq.
    :type table_key: str, optional
    :param image_key: Staining image key. Defaults to the full-resolution bright-field image
        for Visium HD and to the registered ssDNA image for Stereo-seq.
    :type image_key: str, optional
    :param bin_um: Visium HD bin size to pick when several bin tables were loaded.
    :type bin_um: int, optional
    :param species: Species name used for marker-panel lookup.
    :type species: str
    :param organ: Cell-Ontology module name for the gene panel.
    :type organ: str
    :param load_image: Whether to read the image into memory. Counting bins per cell does
        not need the pixels.
    :type load_image: bool
    :returns: The in-memory view of the reader's output.
    :rtype: ssaft.GridDataset
    :raises ValueError: If ``sdata`` was not produced by a supported count reader.
    """
    reader = reader_of(sdata)
    if reader == "visium_hd":
        return _visium_hd(sdata, bin_um=bin_um, table_key=table_key, image_key=image_key,
                          species=species, organ=organ, load_image=load_image)
    if reader == "stereoseq":
        return _stereoseq(sdata, bin_um=bin_um, table_key=table_key, image_key=image_key,
                          species=species, organ=organ, load_image=load_image)
    raise ValueError(f"not a spatialdata_io count reader output (reader={reader!r})")
