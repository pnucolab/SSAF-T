"""Loaders and rasterization helpers for count-based spatial platforms.

A loader reads a platform's raw output and returns a :class:`spatialdata.SpatialData`
with two elements. ``tables['table']`` holds the count matrix, which carries no region
annotation and stores the GridDataset metadata in ``uns['cellman']``.
``images['staining']`` holds the morphology image together with an Affine transform
that maps image pixels to world micrometres. The count backend, for example
:meth:`Analysis.compute_vf`, consumes that object through the lightweight
:meth:`ssaft.GridDataset.from_sdata` view.

Stereo-seq is deliberately not covered here. SAW output is read by ``spatialdata_io``,
and the MOSTA bin1 GEM reader ships beside the notebook that uses it, in
``notebooks/stereoseq_mosta.py``.

Transcript platforms such as Xenium, MERSCOPE and CosMx flow through their own
SpatialData of points and images instead. In both cases the finished cell-type map is
written back as ``labels['ssaft_celltype_map']``, ``images['ssaft_celltype_map_rgb']``
and ``tables['ssaft_downsampled_vectors']``.
"""

from ._grid import aggregate_to_grid, fit_affine_lstsq
from ._loaders import from_visiumhd, load_strata, load_visiumhd

LOADERS = {
    "visiumhd": load_visiumhd,
    "visium_hd": load_visiumhd,
    "strata": load_strata,
    "stratamap": load_strata,
    "novaseq": load_strata,
}


def load(platform: str, *args, **kwargs):
    """Dispatch to a count-platform loader by name, ignoring case.

    :param platform: Platform name, which must be a key of ``LOADERS``. The name is
        matched case-insensitively and any hyphen is treated as an underscore.
    :type platform: str
    :param args: Positional arguments forwarded to the selected loader.
    :param kwargs: Keyword arguments forwarded to the selected loader.
    :returns: Whatever the selected loader returns, which is a count SpatialData.
    :rtype: spatialdata.SpatialData
    :raises ValueError: If no loader is registered under that name.
    """
    key = platform.lower().replace("-", "_")
    if key not in LOADERS:
        raise ValueError(f"unknown count platform {platform!r}; "
                         f"supported: {sorted(set(LOADERS))}")
    return LOADERS[key](*args, **kwargs)


__all__ = [
    "aggregate_to_grid", "fit_affine_lstsq",
    "load_visiumhd", "from_visiumhd", "load_strata", "load", "LOADERS",
]
