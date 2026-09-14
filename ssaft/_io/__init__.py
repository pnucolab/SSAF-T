"""Count-based spatial loaders and rasterization for ssaft.

Loaders return a :class:`spatialdata.SpatialData` with ``tables['table']`` (the
count matrix, region-less, + ``uns['cellman']`` metadata) and ``images['staining']``
(the morphology image with an ``px → µm`` Affine transform). The count backend
(:meth:`Analysis.compute_vf` etc.) adapts it via
:meth:`ssaft.GridDataset.from_sdata`. Stereo-seq is not covered here: SAW output is read by ``spatialdata_io``, and the
MOSTA bin1-GEM reader ships with its notebook (``notebooks/stereoseq_mosta.py``).
Transcript platforms
(Xenium/MERSCOPE/CosMx) flow through their own SpatialData (points + images); the
final cell-type map is written back into either as ``labels['ssaft_celltype_map']``
+ ``images['ssaft_celltype_map_rgb']`` + ``tables['ssaft_downsampled_vectors']``.
"""

from ._grid import aggregate_to_grid, fit_affine_lstsq
from ._loaders import load_strata, load_visiumhd

LOADERS = {
    "visiumhd": load_visiumhd,
    "visium_hd": load_visiumhd,
    "strata": load_strata,
    "stratamap": load_strata,
    "novaseq": load_strata,
}


def load(platform: str, *args, **kwargs):
    """Dispatch to a count-platform loader by name (case-insensitive).

    :param platform: Platform name (a key of ``LOADERS``); matched
        case-insensitively with ``-`` normalized to ``_``.
    :type platform: str
    :param args: Positional arguments forwarded to the selected loader.
    :param kwargs: Keyword arguments forwarded to the selected loader.
    """
    key = platform.lower().replace("-", "_")
    if key not in LOADERS:
        raise ValueError(f"unknown count platform {platform!r}; "
                         f"supported: {sorted(set(LOADERS))}")
    return LOADERS[key](*args, **kwargs)


__all__ = [
    "aggregate_to_grid", "fit_affine_lstsq",
    "load_visiumhd", "load_strata", "load", "LOADERS",
]
