"""Platform loaders for count-based spatial data → :class:`spatialdata.SpatialData`.

Each loader reads a platform's raw output and returns a uniform SpatialData
(assembled by :meth:`GridDataset.to_spatialdata`): ``tables['table']`` = the count
matrix (``obsm['spatial']`` in µm, region-less, GridDataset metadata in
``uns['cellman']``) and ``images['staining']`` = the morphology image carrying an
``affine_px_to_um`` (image pixel → world µm) Affine transform — the single
cross-platform coordinate contract.

Only the raw read, the image source, the affine convention, and the species differ
between platforms; everything downstream (panel → compute_vf → downsample → cluster
→ map) is shared and consumes the lightweight :meth:`GridDataset.from_sdata` view.

Stereo-seq is intentionally absent: SAW deliveries (GEF / ``*.tissue.gef``) are read
by ``spatialdata_io``, and the one-off MOSTA bin1-GEM + ssDNA reader lives beside the
notebook that uses it (``notebooks/stereoseq_mosta.py``), not in this package.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile

from .._griddataset import GridDataset
from ._grid import fit_affine_lstsq


def load_visiumhd(path: str | Path, *, species: str = "mouse",
                  organ: str = "small_intestine", bin_um: int = 2,
                  fullres_image_file: Optional[str] = None):
    """Load a 10x Visium HD ``outs/`` directory at the ``bin_um`` µm bin.

    Convenience wrapper: reads the directory with ``spatialdata_io.visium_hd`` and
    converts the result with :func:`from_visiumhd`. If you already hold that
    SpatialData, call :func:`from_visiumhd` on it directly.

    :param path: Path to the Visium HD ``outs/`` directory.
    :type path: str or pathlib.Path
    :param species: Species name used for marker-panel lookup.
    :type species: str
    :param organ: Cell-Ontology module name for the gene panel.
    :type organ: str
    :param bin_um: Bin size in µm to load from the binned outputs.
    :type bin_um: int
    :param fullres_image_file: Optional path to an explicit fullres image file.
    :type fullres_image_file: str, optional
    :returns: The count SpatialData (see :func:`from_visiumhd`).
    :rtype: spatialdata.SpatialData
    """
    import spatialdata_io

    kw = {}
    if fullres_image_file is not None:
        kw["fullres_image_file"] = fullres_image_file
    sdata = spatialdata_io.visium_hd(str(path), bin_size=bin_um,
                                     load_all_images=True, **kw)
    um_per_px = None
    sf = Path(path) / "binned_outputs" / f"square_{bin_um:03d}um" / "spatial" / "scalefactors_json.json"
    if sf.exists():
        import json
        um_per_px = json.loads(sf.read_text()).get("microns_per_pixel")
    return from_visiumhd(sdata, species=species, organ=organ, bin_um=bin_um, um_per_px=um_per_px)


def from_visiumhd(sdata, *, species: str, organ: str, bin_um: int = 2,
                  table_key: Optional[str] = None, image_key: Optional[str] = None,
                  um_per_px: Optional[float] = None):
    """Convert a ``spatialdata_io.visium_hd`` SpatialData into the SSAF-T count layout.

    Takes the ``bin_um`` µm bin table, keeps the in-tissue bins, places each bin at
    ``(array_col, array_row) * bin_um`` µm, and fits the image-pixel -> µm affine from
    the bins' ``(array_col, array_row)`` against their full-resolution pixel positions
    (``obsm['spatial']`` as ``spatialdata_io`` stores it), on the same lattice the
    vector field uses.

    :param sdata: SpatialData returned by ``spatialdata_io.visium_hd`` (loaded with the
        ``bin_um`` bin size).
    :type sdata: spatialdata.SpatialData
    :param species: Species name used for marker-panel lookup.
    :type species: str
    :param organ: Cell-Ontology module name for the gene panel.
    :type organ: str
    :param bin_um: Bin size in µm; selects the ``square_{bin_um:03d}um`` table.
    :type bin_um: int
    :param table_key: Bin table key; defaults to the table whose key contains
        ``square_{bin_um:03d}um``.
    :type table_key: str, optional
    :param image_key: Staining image key; defaults to the full-resolution image (a key
        containing ``full``), else the first image.
    :type image_key: str, optional
    :param um_per_px: µm per full-resolution pixel. It sets the tissue-mask sampling density in
        ``downsample_vector``. Defaults to the scale of the fitted affine; Space Ranger's
        ``scalefactors_json.json`` ``microns_per_pixel`` differs from it in the fourth decimal,
        and :func:`load_visiumhd` passes that value so earlier runs reproduce exactly.
    :type um_per_px: float, optional
    :returns: SpatialData with ``tables['table']`` (bins x genes, ``obsm['spatial']`` in
        µm, metadata in ``uns['cellman']``) and ``images['staining']`` (the H&E with its
        pixel -> µm Affine).
    :rtype: spatialdata.SpatialData
    """
    suffix = f"square_{bin_um:03d}um"
    tkey = table_key or next((k for k in sdata.tables if suffix in k), None)
    if tkey is None:
        raise RuntimeError(f"no {bin_um}µm table; have {list(sdata.tables)}")
    adata = sdata.tables[tkey].copy()
    adata.var_names_make_unique()
    missing = [c for c in ("array_row", "array_col") if c not in adata.obs]
    if missing or "spatial" not in adata.obsm:
        raise RuntimeError(f"table {tkey!r} lacks {missing or ['obsm[spatial]']}; "
                           "pass the SpatialData returned by spatialdata_io.visium_hd")
    px = np.asarray(adata.obsm["spatial"], dtype=np.float64)       # (pxl_col, pxl_row)
    if "in_tissue" in adata.obs:
        keep = adata.obs["in_tissue"].astype(bool).to_numpy()
        adata, px = adata[keep].copy(), px[keep]
    array_col = adata.obs["array_col"].to_numpy()
    array_row = adata.obs["array_row"].to_numpy()

    adata.obsm["spatial"] = np.stack([array_col, array_row], axis=1).astype(np.float64) * bin_um
    for c in ("region", "location_id"):                              # TableModel annotation
        if c in adata.obs:
            del adata.obs[c]
    adata.uns.pop("spatialdata_attrs", None)

    ikey = image_key or next((k for k in sdata.images if "full" in k), next(iter(sdata.images)))
    img_el = sdata.images[ikey]
    from xarray import DataArray as _DataArray
    if not isinstance(img_el, _DataArray):
        _keys = list(img_el.keys())
        _node = img_el["scale0"] if "scale0" in _keys else img_el[_keys[0]]
        _vars = list(_node.data_vars) if hasattr(_node, "data_vars") else list(_node.keys())
        img_el = _node["image"] if "image" in _vars else _node[_vars[0]]
    arr = np.asarray(img_el.values)
    if arr.ndim == 3 and arr.shape[0] in (3, 4):
        arr = np.moveaxis(arr, 0, -1)

    affine_inv = fit_affine_lstsq(array_col, array_row, px[:, 0], px[:, 1])
    affine_px_to_um = affine_inv * float(bin_um)
    if um_per_px is None:
        um_per_px = float(np.sqrt(abs(np.linalg.det(affine_px_to_um[:, :2]))))

    return GridDataset(
        raw=adata, image=arr, um_per_px=um_per_px, species=species, organ=organ,
        platform="visiumhd", affine_px_to_um=affine_px_to_um, he_revert=True,
        meta={"bin_um": bin_um},
    ).to_spatialdata()


def load_strata(h5ad_path: str | Path, he_ome_tiff: str | Path, *,
                species: str = "human", organ: str = "breast",
                he_level: int = 2, um_per_px_full: float = 0.260) -> GridDataset:
    """Load a StrataMap NovaSeq-X sample: point-barcode ``.h5ad`` + registered H&E.

    Barcode coordinates are already in ``obsm['spatial']`` (µm) if the h5ad was
    prepared that way; otherwise pass an h5ad whose ``obs_names`` encode
    ``…:X:Y:…`` in nm. Gene symbols are promoted to ``var_names``. The H&E axes
    are transposed relative to barcode µm (image row is world x, image col is
    world y).

    :param h5ad_path: Path to the point-barcode ``.h5ad`` file.
    :type h5ad_path: str or pathlib.Path
    :param he_ome_tiff: Path to the registered H&E OME-TIFF image.
    :type he_ome_tiff: str or pathlib.Path
    :param species: Species name used for marker-panel lookup.
    :type species: str
    :param organ: Cell-Ontology module name for the gene panel.
    :type organ: str
    :param he_level: Pyramid level of the H&E image to read.
    :type he_level: int
    :param um_per_px_full: Fallback µm-per-pixel at full resolution when the OME
        metadata lacks ``PhysicalSizeX``.
    :type um_per_px_full: float
    """
    import scanpy as sc

    adata = sc.read_h5ad(str(h5ad_path))
    if "gene_symbols" in adata.var:
        adata.var["ensembl"] = adata.var_names.to_numpy()
        adata.var_names = adata.var["gene_symbols"].astype(str).to_numpy()
        adata.var_names_make_unique()
    if "spatial" not in adata.obsm:
        xy = np.array([b.split(":")[1:3] for b in adata.obs_names],
                      dtype=np.float64) / 1000.0
        adata.obsm["spatial"] = xy

    with tifffile.TiffFile(str(he_ome_tiff)) as tif:
        base = tif.series[0]
        full_h = base.levels[0].shape[0]
        arr = base.levels[he_level].asarray()
        m = re.search(r'PhysicalSizeX="([\d.]+)"', tif.ome_metadata or "")
    full_um_per_px = float(m.group(1)) / 1000.0 if m else um_per_px_full
    if arr.ndim == 3 and arr.shape[0] in (3, 4):
        arr = np.moveaxis(arr, 0, -1)
    arr = arr[..., :3] if arr.ndim == 3 else arr
    he_um_per_px = full_um_per_px * (full_h / arr.shape[0])

    affine_px_to_um = np.array([[0., he_um_per_px, 0.],
                                [he_um_per_px, 0., 0.]])

    return GridDataset(
        raw=adata, image=arr, um_per_px=he_um_per_px, species=species, organ=organ,
        platform="strata", affine_px_to_um=affine_px_to_um, he_revert=True,
        meta={"he_level": he_level},
    ).to_spatialdata()
