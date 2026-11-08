"""The canonical intermediate for count-matrix spatial platforms.

Transcript platforms such as Xenium, MERSCOPE and CosMx flow through SpatialData
points and the kernel density pipeline unchanged. Count platforms such as Stereo-seq,
Visium HD and NovaSeq X StrataMap do not provide molecule-resolved transcripts at all.
They give a matrix of cells or bins by genes on a spatial lattice.

Unrolling those counts into synthetic transcripts would blow up memory on dense data,
since a Stereo-seq bin1 sample carries on the order of a hundred million unique
molecules. The count path therefore rasterizes the counts onto a grid and blurs them
on the GPU into a field equivalent to the kernel density one. See
:meth:`Analysis.compute_vf`.

``GridDataset`` is the lightweight hand-off between a platform loader, in
:mod:`ssaft._io`, and that backend. It deliberately does not wrap a SpatialData
object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from anndata import AnnData


class _RawView:
    """A minimal stand-in for AnnData over a count ``GridDataset.raw``.

    The class exposes only the surface that ``aggregate_to_grid``, ``compute_vf`` and
    ``GridDataset`` actually touch, namely ``X``, ``obsm['spatial']``, ``var``,
    ``var_names``, ``n_obs``, ``n_vars``, ``shape`` and slicing by column through
    ``[:, genes]``.

    Its purpose is to avoid constructing a full AnnData over a count table of a hundred
    million rows. Materializing the string index of ``obs_names`` alone costs roughly 144
    seconds on such a table, and that is the real bottleneck when ``spatialdata`` or
    ``anndata`` read a huge count store. The view is read-only and cannot be written back
    to SpatialData.
    """
    def __init__(self, X, spatial, var):
        """Wrap a count matrix, its coordinates and its gene table.

        :param X: The count matrix, of shape ``(n_obs, n_vars)``.
        :param spatial: The coordinates in micrometres, of shape ``(n_obs, 2)``.
        :param var: The gene table, whose index holds the gene symbols.
        """
        self.X = X
        self.obsm = {"spatial": spatial}
        self.var = var

    @property
    def var_names(self):
        """The gene symbols, taken from the index of ``var``."""
        return self.var.index

    @property
    def n_obs(self):
        """The number of rows, meaning points or bins."""
        return self.X.shape[0]

    @property
    def n_vars(self):
        """The number of columns, meaning genes."""
        return self.X.shape[1]

    @property
    def shape(self):
        """The shape of the count matrix, as rows by columns."""
        return self.X.shape

    def copy(self):
        """Return the view itself, since it is read-only and needs no copy."""
        return self

    def __getitem__(self, key):
        """Slice the view by column, which is the only slicing the backend needs.

        A row slice, or a column slice that is a plain ``slice`` object, returns the view
        unchanged. A list of gene symbols returns a new view holding only those genes that
        exist, in the order given.

        :param key: Either a column selector, or a tuple whose second element is one.
        :returns: This view, or a new view restricted to the named genes.
        :rtype: _RawView
        """
        cols = key[1] if isinstance(key, tuple) else slice(None)
        if isinstance(cols, slice):
            return self
        idx = self.var.index.get_indexer(list(cols))
        idx = idx[idx >= 0]
        return _RawView(self.X[:, idx].tocsr(), self.obsm["spatial"], self.var.iloc[idx])



def _jsonify(obj):
    """Coerce a metadata value into something the AnnData zarr writer accepts in ``uns``.

    Tuples become lists and numpy scalars become plain Python scalars, recursively. The
    count loaders store values such as ``meta['flip'] = (0, 1)``, and anndata cannot
    encode a tuple.

    :param obj: The value to coerce.
    :returns: The coerced value.
    """
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


@dataclass
class GridDataset:
    """A platform-agnostic container for one count-based spatial sample.

    Attributes
    ----------
    raw : AnnData
        Point-level or bin-level counts. ``obsm['spatial']`` holds x and y in
        micrometres, and ``var_names`` holds gene symbols. This is the unit that
        selection of highly variable genes and rasterization operate on, through
        :func:`ssaft._io.aggregate_to_grid`.
    image : numpy.ndarray
        The morphology or staining image used for sampling voxels inside the tissue
        mask, shaped either as channel by y by x, or as y by x. Examples are ssDNA for
        Stereo-seq and bright-field staining for Visium HD and StrataMap.
    um_per_px : float
        The physical scale of the image, in micrometres per pixel, at the resolution
        stored.
    species : str
        Species name used for marker-panel lookup.
    organ : str
        Cell-Ontology module name, for example ``'whole_brain'`` or ``'breast'``, used
        to fetch the marker genes of the gene panel.
    platform : str
        One of ``'stereoseq'``, ``'visiumhd'`` or ``'strata'``. It is informational.
    origin : tuple(float, float)
        The shared global origin in micrometres, as ``(x0, y0)``, which is the minimum
        of ``raw.obsm['spatial']`` along each axis. Every lattice must use this same
        origin so that they align: the coarse grid for highly variable genes, the fine
        grid of the vector field, and the voxel mapping.
    affine_px_to_um : numpy.ndarray, optional
        A 2 by 3 affine mapping an image pixel, given as column, row and one, to world
        coordinates in micrometres. This is the single cross-platform coordinate
        contract. Each loader folds its own convention into this matrix, whether that
        is the StrataMap axis transposition, the Stereo-seq mirror flip or the Visium
        HD least-squares lattice fit, so that
        :meth:`Analysis.downsample_vector` samples voxels identically for all of them.
        World micrometres are converted to grid bins at compute time using ``origin``
        and the bin size. A value of ``None`` falls back to a plain ``um_per_px``
        scale.
    he_revert : bool
        The polarity the tissue mask assumes when thresholding by Otsu's method.
        ``True`` means a bright background with dark tissue, as in bright-field
        staining, which is inverted before thresholding. ``False`` means a dark
        background with bright tissue, as in fluorescence or ssDNA, which is
        thresholded directly.
    meta : dict
        Any further per-platform metadata, carried through unchanged.
    """

    raw: AnnData
    image: np.ndarray
    um_per_px: float
    species: str
    organ: str
    platform: str = "unknown"
    origin: Tuple[float, float] = (0.0, 0.0)
    affine_px_to_um: Optional[np.ndarray] = None
    he_revert: bool = True
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate the counts and fill in the origin when it was left at the default.

        :raises ValueError: If ``raw`` carries no ``obsm['spatial']``.
        """
        if "spatial" not in self.raw.obsm:
            raise ValueError("GridDataset.raw needs obsm['spatial'] (x,y in µm).")
        sp = np.asarray(self.raw.obsm["spatial"])
        if self.origin == (0.0, 0.0):
            self.origin = (float(sp[:, 0].min()), float(sp[:, 1].min()))

    @property
    def n_genes(self) -> int:
        """The number of genes in the count matrix."""
        return int(self.raw.n_vars)

    def subset_genes(self, genes) -> "GridDataset":
        """Return a copy of this dataset restricted to the named genes.

        All metadata is preserved.

        :param genes: Gene symbols to keep. Any symbol absent from ``raw.var_names`` is
            ignored.
        :type genes: list of str
        :returns: The restricted copy.
        :rtype: GridDataset
        """
        keep = [g for g in genes if g in set(self.raw.var_names)]
        sub = self.raw[:, keep].copy()
        return GridDataset(
            raw=sub, image=self.image, um_per_px=self.um_per_px,
            species=self.species, organ=self.organ, platform=self.platform,
            origin=self.origin, affine_px_to_um=self.affine_px_to_um,
            he_revert=self.he_revert, meta=dict(self.meta),
        )

    def to_spatialdata(self, *, table_key: str = "table",
                       image_key: str = "staining", copy: bool = True):
        """Assemble a :class:`spatialdata.SpatialData` from this count sample.

        The table is the count AnnData, carrying no region annotation, with all GridDataset
        metadata stored in ``uns['cellman']`` so that it survives a round trip. The image is
        the staining image, shaped channel by y by x, carrying an Affine transform from pixels
        to world micrometres built from ``affine_px_to_um``, or an isotropic Scale built from
        ``um_per_px`` when no affine is set. That transform is the single cross-platform
        coordinate contract, here expressed natively.

        The count pipeline, meaning :meth:`Analysis.compute_vf` and
        :meth:`~Analysis.downsample_vector`, threads this object through, and
        :meth:`from_sdata` reconstructs the lightweight view the backend needs.

        :param table_key: Key under which the count table is stored in the resulting
            SpatialData.
        :type table_key: str
        :param image_key: Key under which the staining image is stored in the resulting
            SpatialData.
        :type image_key: str
        :param copy: Whether to duplicate ``raw`` before adding metadata to it. Pass ``False``
            to modify ``raw`` in place and skip the copy.
        :type copy: bool
        :returns: The assembled SpatialData.
        :rtype: spatialdata.SpatialData
        """
        import spatialdata as sd
        from spatialdata.models import Image2DModel, TableModel
        from spatialdata.transformations import Affine, Scale

        img = np.asarray(self.image)
        orig_ndim = int(img.ndim)
        if img.ndim == 2:
            cyx, nchan = img[None, :, :], 1
        elif img.shape[-1] in (3, 4):
            cyx = np.moveaxis(img[..., :3], -1, 0); nchan = cyx.shape[0]
        elif img.shape[0] in (3, 4):
            cyx = img[:3]; nchan = cyx.shape[0]
        else:
            cyx = img.reshape(-1, *img.shape[-2:])[:1]; nchan = 1

        if self.affine_px_to_um is not None:
            M = np.asarray(self.affine_px_to_um, float)
            mat = np.array([[M[0, 0], M[0, 1], M[0, 2]],
                            [M[1, 0], M[1, 1], M[1, 2]],
                            [0.0, 0.0, 1.0]])
            tf = Affine(mat, input_axes=("x", "y"), output_axes=("x", "y"))
        else:
            tf = Scale([self.um_per_px, self.um_per_px], axes=("x", "y"))
        image = Image2DModel.parse(cyx, dims=("c", "y", "x"),
                                   transformations={"global": tf})

        tab = self.raw.copy() if copy else self.raw
        tab.uns["cellman"] = _jsonify({
            "species": self.species, "organ": self.organ, "platform": self.platform,
            "um_per_px": float(self.um_per_px),
            "affine_px_to_um": (None if self.affine_px_to_um is None
                                else np.asarray(self.affine_px_to_um, float).tolist()),
            "origin": [float(self.origin[0]), float(self.origin[1])],
            "he_revert": bool(self.he_revert),
            "image_ndim": orig_ndim, "image_nchan": int(nchan),
            "image_key": image_key, "meta": dict(self.meta),
        })
        table = TableModel.parse(tab)
        return sd.SpatialData(tables={table_key: table}, images={image_key: image})

    @classmethod
    def from_sdata(cls, sdata, *, table_key=None, image_key=None) -> "GridDataset":
        """Reconstruct a GridDataset view from a count SpatialData.

        The object is expected to be one built by :meth:`to_spatialdata`, and the count table,
        its ``uns['cellman']`` metadata and the staining image are read back out of it. The
        heavy backend consumes this view directly, which is what lets the blur and lookup
        internals stay array-based and unchanged.

        A SpatialData that came straight from ``spatialdata_io.visium_hd`` or
        ``spatialdata_io.stereoseq``, and therefore carries no ``uns['cellman']``, is
        converted on the fly by
        :func:`ssaft._io._spatialdata_io.grid_from_spatialdata_io`. The input is not modified
        in either case.

        :param sdata: The count SpatialData to read.
        :type sdata: spatialdata.SpatialData
        :param table_key: Count-table key. Defaults to the table carrying ``uns['cellman']``.
        :type table_key: str, optional
        :param image_key: Staining-image key. Defaults to the key recorded in the metadata.
        :type image_key: str, optional
        :returns: The reconstructed view.
        :rtype: GridDataset
        """
        from ._io._spatialdata_io import grid_from_spatialdata_io, is_count_reader_output
        if is_count_reader_output(sdata):
            return grid_from_spatialdata_io(sdata, table_key=table_key, image_key=image_key)
        if table_key is None:
            table_key = next((k for k, t in sdata.tables.items()
                              if "cellman" in t.uns), next(iter(sdata.tables)))
        tab = sdata.tables[table_key]
        meta = dict(tab.uns.get("cellman", {}))
        ik = image_key or meta.get("image_key", "staining")
        if ik not in sdata.images:
            ik = next(iter(sdata.images))
        arr = np.asarray(sdata.images[ik].data)
        nchan = int(meta.get("image_nchan", arr.shape[0] if arr.ndim == 3 else 1))
        if arr.ndim == 3 and nchan == 1:
            image = arr[0]
        elif arr.ndim == 3:
            image = np.moveaxis(arr[:3], 0, -1)
        else:
            image = arr
        aff = meta.get("affine_px_to_um")
        return cls(
            raw=tab, image=image, um_per_px=float(meta.get("um_per_px", 1.0)),
            species=meta.get("species", "unknown"), organ=meta.get("organ", "unknown"),
            platform=meta.get("platform", "unknown"),
            origin=tuple(meta.get("origin", (0.0, 0.0))),
            affine_px_to_um=(None if aff is None else np.asarray(aff, float)),
            he_revert=bool(meta.get("he_revert", True)),
            meta=dict(meta.get("meta", {})),
        )

    @classmethod
    def from_sdata_zarr(cls, sdata_path, *, table_key="table",
                        image_key=None) -> "GridDataset":
        """Read a count SpatialData store directly from zarr, quickly and read-only.

        The matrix is read as compressed sparse rows, ``obsm/spatial`` is read through zarr,
        and ``var`` and ``uns`` are read with ``anndata.read_elem``. The counts are then
        wrapped in :class:`_RawView`. This bypasses ``spatialdata.read_zarr`` and
        ``anndata.read_zarr``, whose cost of roughly 144 seconds on a count table of a hundred
        million rows is dominated by materializing that many string values for ``obs_names``,
        which the count backend never uses. The result is about 25 to 30 times faster, and the
        GridDataset it returns is consumed identically by ``aggregate_to_grid`` and
        ``compute_vf``.

        Use this in place of ``sd.read_zarr(path)`` followed by ``from_sdata(sdata)`` on the
        heavy count path. Do not call ``to_spatialdata`` on the result.

        :param sdata_path: Filesystem path to the SpatialData zarr store.
        :type sdata_path: str or pathlib.Path
        :param table_key: Count-table key within the store.
        :type table_key: str
        :param image_key: Staining-image key. Defaults to the key recorded in the metadata.
        :type image_key: str, optional
        :returns: The reconstructed view.
        :rtype: GridDataset
        """
        import zarr
        from scipy.sparse import csr_matrix
        try:
            from anndata.io import read_elem
        except Exception:  # noqa: BLE001
            from anndata.experimental import read_elem
        tpath = f"{sdata_path}/tables/{table_key}"
        tg = zarr.open_group(tpath, mode="r")
        Xg = zarr.open_group(f"{tpath}/X", mode="r")
        indptr = Xg["indptr"][:]; indices = Xg["indices"][:]; data = Xg["data"][:]
        var = read_elem(tg["var"])
        X = csr_matrix((data, indices, indptr),
                       shape=(len(indptr) - 1, var.shape[0]))
        spatial = np.asarray(zarr.open(f"{tpath}/obsm/spatial", mode="r")[:])
        meta = dict(read_elem(tg["uns"]).get("cellman", {}))
        raw = _RawView(X, spatial, var)
        ik = image_key or meta.get("image_key", "staining")
        ipath = f"{sdata_path}/images/{ik}"
        ig = zarr.open_group(ipath, mode="r")

        def _find(g, pre=""):
            """Walk a zarr group depth first and return the path of its first array.

            Image elements may be stored plainly or as a multiscale pyramid, so the array is found
            by descent rather than by assuming a fixed layout.

            :param g: The zarr group to search.
            :param pre: Path prefix accumulated so far during the descent.
            :type pre: str
            :returns: The full path of the first array found, or ``None`` when the group holds
                none.
            :rtype: str or None
            """
            for k in g.keys():
                full = f"{pre}/{k}" if pre else k
                node = g[k]
                if hasattr(node, "shape"):
                    return f"{ipath}/{full}"
                s = _find(node, full)
                if s:
                    return s
            return None
        arr = np.asarray(zarr.open(_find(ig), mode="r")[:])
        nchan = int(meta.get("image_nchan", arr.shape[0] if arr.ndim == 3 else 1))
        if arr.ndim == 3 and nchan == 1:
            image = arr[0]
        elif arr.ndim == 3:
            image = np.moveaxis(arr[:3], 0, -1)
        else:
            image = arr
        aff = meta.get("affine_px_to_um")
        return cls(
            raw=raw, image=image, um_per_px=float(meta.get("um_per_px", 1.0)),
            species=meta.get("species", "unknown"),
            organ=meta.get("organ", "unknown"),
            platform=meta.get("platform", "unknown"),
            origin=tuple(meta.get("origin", (0.0, 0.0))),
            affine_px_to_um=(None if aff is None else np.asarray(aff, float)),
            he_revert=bool(meta.get("he_revert", True)),
            meta=dict(meta.get("meta", {})),
        )
