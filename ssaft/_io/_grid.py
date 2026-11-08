"""Rasterization helpers shared by the count-based loaders."""

from __future__ import annotations

from typing import Optional, Tuple

import anndata as ad
import numpy as np
from scipy.sparse import csr_matrix, issparse


def aggregate_to_grid(adata: ad.AnnData, bin_um: float,
                      origin: Optional[Tuple[float, float]] = None) -> ad.AnnData:
    """Sum point or bin counts onto a regular lattice of ``bin_um`` micrometre squares.

    Every count row is assigned to the square bin that contains its ``obsm['spatial']``
    coordinate in micrometres. A shared global origin is used so that lattices built at
    different bin sizes, such as a coarse lattice for highly variable gene selection and a
    fine one for the vector field, stay aligned with each other.

    :param adata: Point-level or bin-level counts whose ``obsm['spatial']`` is in micrometres.
    :type adata: anndata.AnnData
    :param bin_um: Lattice bin size in micrometres.
    :type bin_um: float
    :param origin: Shared origin as ``(x0, y0)`` in micrometres. Defaults to the minimum
        of ``obsm['spatial']`` along each axis.
    :type origin: tuple of float, optional
    :returns: Bin-level counts carrying ``obs['array_row']`` and ``obs['array_col']``, with
        ``obsm['spatial']`` set to ``(col, row)`` and the bin size and origin recorded in
        ``uns``.
    :rtype: anndata.AnnData
    """
    sp = np.asarray(adata.obsm["spatial"])
    if origin is None:
        x0, y0 = float(sp[:, 0].min()), float(sp[:, 1].min())
    else:
        x0, y0 = float(origin[0]), float(origin[1])

    col = np.floor((sp[:, 0] - x0) / bin_um).astype(np.int64)
    row = np.floor((sp[:, 1] - y0) / bin_um).astype(np.int64)
    n_col = int(col.max()) + 1
    bid = row * n_col + col
    uniq, inv = np.unique(bid, return_inverse=True)

    S = csr_matrix((np.ones(len(inv), np.float32), (inv, np.arange(len(inv)))),
                   shape=(len(uniq), adata.n_obs))
    X = adata.X.tocsr() if issparse(adata.X) else adata.X
    binX = (S @ X).tocsr() if issparse(X) else csr_matrix(S @ X)

    out = ad.AnnData(X=binX, var=adata.var.copy())
    out.obs["array_row"] = (uniq // n_col).astype(np.int64)
    out.obs["array_col"] = (uniq % n_col).astype(np.int64)
    out.obsm["spatial"] = np.stack(
        [out.obs["array_col"].to_numpy(), out.obs["array_row"].to_numpy()],
        axis=1).astype(np.float32)
    out.uns["bin_size_um"] = float(bin_um)
    out.uns["grid_origin_um"] = (x0, y0)
    return out


def fit_affine_lstsq(array_col, array_row, px_col, px_row) -> np.ndarray:
    """Fit a least-squares affine from lattice indices to pixels and return its inverse.

    The forward fit maps ``(col, row, 1)`` to ``(px_x, px_y)``. The returned 2 by 3 matrix
    is the inverse of that fit, so it maps pixels back to lattice indices.

    This is used by platforms that publish a per-bin correspondence table between the
    lattice and the image, such as the Visium HD ``tissue_positions`` file, which pairs
    ``array_col`` and ``array_row`` with ``pxl_col`` and ``pxl_row``. Fit the
    correspondence on the same lattice that the vector field uses.

    :param array_col: Lattice column indices of the correspondence points.
    :type array_col: array-like
    :param array_row: Lattice row indices of the correspondence points.
    :type array_row: array-like
    :param px_col: Pixel x coordinates, that is column positions, of the correspondence points.
    :type px_col: array-like
    :param px_row: Pixel y coordinates, that is row positions, of the correspondence points.
    :type px_row: array-like
    :returns: The inverse affine as a 2 by 3 matrix.
    :rtype: numpy.ndarray
    """
    A = np.c_[np.asarray(array_col), np.asarray(array_row), np.ones(len(array_col))]
    cx, *_ = np.linalg.lstsq(A, np.asarray(px_col), rcond=None)
    cy, *_ = np.linalg.lstsq(A, np.asarray(px_row), rcond=None)
    fwd = np.array([cx, cy])
    inv = np.linalg.inv(np.vstack([fwd, [0, 0, 1]]))[:2, :3]
    return inv
