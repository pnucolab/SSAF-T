"""GPU equivalent of ``ssaft.utils.kde_at_points``, built on a Triton cell-list kernel.

This module is purely additive. It does not modify the C++ implementation, and swaps
itself in only for the duration of a context manager::

    from ssaft._kde_gpu import gpu_kde
    with gpu_kde(device='cuda'):
        ds.downsample_vector(he_path=..., sdata=...)   # uses the GPU internally
    # outside the with block the C++ implementation is back

The results agree with the C++ ``kde_at_points`` to within a maximum row-wise L1
relative difference of 2e-6, verified on 5 million molecules, 5 thousand query points
and 280 genes.

Both two and three dimensions are supported. Note that the three-dimensional cell
list occupies W times H times D times 4 bytes, so for a very large or strongly
anisotropic domain it may be worth passing ``force_2d=True`` to collapse z, whenever a
two-dimensional projection is acceptable.
"""
from __future__ import annotations

import contextlib
import math
import os
import warnings
from typing import Optional

import numpy as np

try:
    import torch
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


if _HAS_TRITON:
    @triton.jit
    def _kde_weight(r2, two_h2, inv_norm, Rw2, inv_scale, KERN: tl.constexpr):
        """Return one molecule's kernel weight at a squared distance.

        The caller is responsible for masking out any distance beyond the kernel's reach.

        ``KERN`` is the kernel identifier used in ``ssaft._count.KDE_KERNELS``. The Gaussian
        and Epanechnikov branches are the expressions these kernels have always used, with the
        Epanechnikov support taken as the square root of ``Rw2``, so their results are
        unchanged by this implementation.

        :param r2: Squared distance between the molecule and the query point.
        :param two_h2: Twice the squared bandwidth, used by the Gaussian branch.
        :param inv_norm: Reciprocal of the kernel's normalization constant.
        :param Rw2: Squared reach of the kernel.
        :param inv_scale: Reciprocal of the kernel's own scale.
        :param KERN: Compile-time kernel identifier.
        :returns: The weight of this molecule at that distance.
        """
        if KERN == 0:        # gaussian
            w = tl.exp(-r2 / two_h2) * inv_norm
        elif KERN == 1:      # tophat
            w = r2 * 0.0 + inv_norm
        elif KERN == 2:      # epanechnikov 1 - (r/H)^2
            w = (1.0 - r2 / Rw2) * inv_norm
        elif KERN == 3:      # exponential exp(-r/lambda)
            w = tl.exp(-tl.sqrt(r2) * inv_scale) * inv_norm
        elif KERN == 4:      # linear 1 - r/H
            w = (1.0 - tl.sqrt(r2) * inv_scale) * inv_norm
        else:                # cosine cos(pi r / 2H)
            w = tl.cos(1.5707963267948966 * tl.sqrt(r2) * inv_scale) * inv_norm
        return w

    @triton.jit
    def _kde_cell_kernel(
        qx_ptr, qy_ptr,
        sx_ptr, sy_ptr, sg_ptr,
        cell_start_ptr,
        out_ptr,
        N, n_genes, W, H, Rp,
        x_min, y_min,
        inv_sd, two_h2, inv_norm, Rw2, inv_scale,
        BLOCK_T: tl.constexpr, KERN: tl.constexpr,
    ):
        """Triton kernel computing the two-dimensional estimate for one query point per program.

        Each program walks the cells within ``Rp`` of its query point's own cell, loads the
        molecules those cells hold in blocks of ``BLOCK_T``, and atomically accumulates each
        molecule's weight into the output row of its gene. Molecules beyond the kernel's reach,
        and gene indices outside the output width, are masked out rather than branched on.

        :param qx_ptr: Query x coordinates.
        :param qy_ptr: Query y coordinates.
        :param sx_ptr: Molecule x coordinates, sorted by cell.
        :param sy_ptr: Molecule y coordinates, sorted by cell.
        :param sg_ptr: Molecule gene indices, sorted by cell.
        :param cell_start_ptr: Start offset of each cell in the sorted arrays.
        :param out_ptr: Output of shape ``(N, n_genes)``, accumulated into.
        :param N: Number of query points.
        :param n_genes: Width of the output.
        :param W: Number of cells along x.
        :param H: Number of cells along y.
        :param Rp: Half-width of the cell neighbourhood to visit, in cells.
        :param x_min: World x coordinate of the cell grid's origin.
        :param y_min: World y coordinate of the cell grid's origin.
        :param inv_sd: Reciprocal of the cell size.
        :param two_h2: Twice the squared bandwidth.
        :param inv_norm: Reciprocal of the kernel's normalization constant.
        :param Rw2: Squared reach of the kernel.
        :param inv_scale: Reciprocal of the kernel's own scale.
        :param BLOCK_T: Number of molecules loaded per block.
        :param KERN: Compile-time kernel identifier.
        """
        pid = tl.program_id(0)
        if pid >= N:
            return
        qx = tl.load(qx_ptr + pid)
        qy = tl.load(qy_ptr + pid)
        qcx = tl.cast(tl.floor((qx - x_min) * inv_sd), tl.int32)
        qcy = tl.cast(tl.floor((qy - y_min) * inv_sd), tl.int32)
        out_row = out_ptr + pid * n_genes

        for cdy in range(-Rp, Rp + 1):
            cy = qcy + cdy
            if (cy >= 0) and (cy < H):
                for cdx in range(-Rp, Rp + 1):
                    cx = qcx + cdx
                    if (cx >= 0) and (cx < W):
                        cid = cy * W + cx
                        s = tl.load(cell_start_ptr + cid)
                        ee = tl.load(cell_start_ptr + cid + 1)
                        nT = ee - s
                        for c0 in range(0, nT, BLOCK_T):
                            offs = c0 + tl.arange(0, BLOCK_T)
                            mask = offs < nT
                            idx = s + offs
                            tx = tl.load(sx_ptr + idx, mask=mask, other=0.0)
                            ty = tl.load(sy_ptr + idx, mask=mask, other=0.0)
                            tg = tl.load(sg_ptr + idx, mask=mask, other=0)
                            dx = tx - qx
                            dy = ty - qy
                            r2 = dx * dx + dy * dy
                            keep = mask & (r2 <= Rw2) & (tg >= 0) & (tg < n_genes)
                            w = _kde_weight(r2, two_h2, inv_norm, Rw2, inv_scale, KERN)
                            tl.atomic_add(out_row + tg,
                                          tl.cast(w, tl.float32),
                                          mask=keep)

    @triton.jit
    def _kde_cell_kernel_3d(
        qx_ptr, qy_ptr, qz_ptr,
        sx_ptr, sy_ptr, sz_ptr, sg_ptr,
        cell_start_ptr,
        out_ptr,
        N, n_genes, W, H, D, Rp,
        x_min, y_min, z_min,
        inv_sd, two_h2, inv_norm, Rw2, inv_scale,
        BLOCK_T: tl.constexpr, KERN: tl.constexpr,
    ):
        """Triton kernel computing the three-dimensional estimate, one query point per program.

        This is the volumetric counterpart of :func:`_kde_cell_kernel`. It walks a cube of
        cells rather than a square, and the cell index runs over z, then y, then x.

        :param qx_ptr: Query x coordinates.
        :param qy_ptr: Query y coordinates.
        :param qz_ptr: Query z coordinates.
        :param sx_ptr: Molecule x coordinates, sorted by cell.
        :param sy_ptr: Molecule y coordinates, sorted by cell.
        :param sz_ptr: Molecule z coordinates, sorted by cell.
        :param sg_ptr: Molecule gene indices, sorted by cell.
        :param cell_start_ptr: Start offset of each cell in the sorted arrays.
        :param out_ptr: Output of shape ``(N, n_genes)``, accumulated into.
        :param N: Number of query points.
        :param n_genes: Width of the output.
        :param W: Number of cells along x.
        :param H: Number of cells along y.
        :param D: Number of cells along z.
        :param Rp: Half-width of the cell neighbourhood to visit, in cells.
        :param x_min: World x coordinate of the cell grid's origin.
        :param y_min: World y coordinate of the cell grid's origin.
        :param z_min: World z coordinate of the cell grid's origin.
        :param inv_sd: Reciprocal of the cell size.
        :param two_h2: Twice the squared bandwidth.
        :param inv_norm: Reciprocal of the kernel's normalization constant.
        :param Rw2: Squared reach of the kernel.
        :param inv_scale: Reciprocal of the kernel's own scale.
        :param BLOCK_T: Number of molecules loaded per block.
        :param KERN: Compile-time kernel identifier.
        """
        pid = tl.program_id(0)
        if pid >= N:
            return
        qx = tl.load(qx_ptr + pid)
        qy = tl.load(qy_ptr + pid)
        qz = tl.load(qz_ptr + pid)
        qcx = tl.cast(tl.floor((qx - x_min) * inv_sd), tl.int32)
        qcy = tl.cast(tl.floor((qy - y_min) * inv_sd), tl.int32)
        qcz = tl.cast(tl.floor((qz - z_min) * inv_sd), tl.int32)
        WH = W * H
        out_row = out_ptr + pid * n_genes

        for cdz in range(-Rp, Rp + 1):
            cz = qcz + cdz
            if (cz >= 0) and (cz < D):
                for cdy in range(-Rp, Rp + 1):
                    cy = qcy + cdy
                    if (cy >= 0) and (cy < H):
                        for cdx in range(-Rp, Rp + 1):
                            cx = qcx + cdx
                            if (cx >= 0) and (cx < W):
                                cid = cz * WH + cy * W + cx
                                s = tl.load(cell_start_ptr + cid)
                                ee = tl.load(cell_start_ptr + cid + 1)
                                nT = ee - s
                                for c0 in range(0, nT, BLOCK_T):
                                    offs = c0 + tl.arange(0, BLOCK_T)
                                    mask = offs < nT
                                    idx = s + offs
                                    tx = tl.load(sx_ptr + idx, mask=mask, other=0.0)
                                    ty = tl.load(sy_ptr + idx, mask=mask, other=0.0)
                                    tz = tl.load(sz_ptr + idx, mask=mask, other=0.0)
                                    tg = tl.load(sg_ptr + idx, mask=mask, other=0)
                                    dx = tx - qx
                                    dy = ty - qy
                                    dz = tz - qz
                                    r2 = dx * dx + dy * dy + dz * dz
                                    keep = mask & (r2 <= Rw2) & (tg >= 0) & (tg < n_genes)
                                    w = _kde_weight(r2, two_h2, inv_norm, Rw2, inv_scale, KERN)
                                    tl.atomic_add(out_row + tg,
                                                  tl.cast(w, tl.float32),
                                                  mask=keep)


def _build_cell_list(lx_g, ly_g, gi_g, sampling_distance, x_min, y_min, W, H):
    """Sort molecules into a two-dimensional cell list on the device.

    Each molecule is assigned to the cell containing it, molecules outside the grid are
    sent to a sentinel cell past the end, and everything is sorted by cell identifier. The
    start offset of each cell is then found by binary search, which is what lets the
    kernel visit a cell's molecules as a contiguous range.

    :param lx_g: Molecule x coordinates on the device.
    :param ly_g: Molecule y coordinates on the device.
    :param gi_g: Gene index of each molecule.
    :param sampling_distance: Cell size, which equals the lattice spacing.
    :param x_min: World x coordinate of the grid's origin.
    :param y_min: World y coordinate of the grid's origin.
    :param W: Number of cells along x.
    :param H: Number of cells along y.
    :returns: The sorted x coordinates, y coordinates and gene indices, together with the
        per-cell start offsets, all contiguous on the device.
    :rtype: tuple of torch.Tensor
    """
    inv_sd = 1.0 / sampling_distance
    tx = torch.floor((lx_g - x_min) * inv_sd).to(torch.int64)
    ty = torch.floor((ly_g - y_min) * inv_sd).to(torch.int64)
    in_range = (tx >= 0) & (tx < W) & (ty >= 0) & (ty < H)
    sentinel = W * H
    cell_id = torch.where(in_range, ty * W + tx,
                          torch.tensor(sentinel, device=tx.device))
    sort_idx = torch.argsort(cell_id)
    sorted_cid = cell_id[sort_idx]
    arange = torch.arange(W * H + 1, device=lx_g.device, dtype=torch.int64)
    cell_start = torch.searchsorted(sorted_cid, arange).to(torch.int32)
    return (lx_g[sort_idx].contiguous(),
            ly_g[sort_idx].contiguous(),
            gi_g[sort_idx].to(torch.int32).contiguous(),
            cell_start.contiguous())


def _build_cell_list_3d(lx_g, ly_g, lz_g, gi_g, sampling_distance,
                        x_min, y_min, z_min, W, H, D):
    """Sort molecules into a three-dimensional cell list on the device.

    This is the volumetric counterpart of :func:`_build_cell_list`, with the cell
    identifier running over z, then y, then x.

    :param lx_g: Molecule x coordinates on the device.
    :param ly_g: Molecule y coordinates on the device.
    :param lz_g: Molecule z coordinates on the device.
    :param gi_g: Gene index of each molecule.
    :param sampling_distance: Cell size, which equals the lattice spacing.
    :param x_min: World x coordinate of the grid's origin.
    :param y_min: World y coordinate of the grid's origin.
    :param z_min: World z coordinate of the grid's origin.
    :param W: Number of cells along x.
    :param H: Number of cells along y.
    :param D: Number of cells along z.
    :returns: The sorted x, y and z coordinates and gene indices, together with the
        per-cell start offsets, all contiguous on the device.
    :rtype: tuple of torch.Tensor
    """
    inv_sd = 1.0 / sampling_distance
    tx = torch.floor((lx_g - x_min) * inv_sd).to(torch.int64)
    ty = torch.floor((ly_g - y_min) * inv_sd).to(torch.int64)
    tz = torch.floor((lz_g - z_min) * inv_sd).to(torch.int64)
    in_range = ((tx >= 0) & (tx < W) &
                (ty >= 0) & (ty < H) &
                (tz >= 0) & (tz < D))
    n_cells = W * H * D
    sentinel = n_cells
    cell_id = torch.where(in_range,
                          tz * (H * W) + ty * W + tx,
                          torch.tensor(sentinel, device=tx.device))
    sort_idx = torch.argsort(cell_id)
    sorted_cid = cell_id[sort_idx]
    arange = torch.arange(n_cells + 1, device=lx_g.device, dtype=torch.int64)
    cell_start = torch.searchsorted(sorted_cid, arange).to(torch.int32)
    return (lx_g[sort_idx].contiguous(),
            ly_g[sort_idx].contiguous(),
            lz_g[sort_idx].contiguous(),
            gi_g[sort_idx].to(torch.int32).contiguous(),
            cell_start.contiguous())


def kde_at_points_gpu(
    query_x, query_y, query_z,
    loc_x, loc_y, loc_z,
    gene_indices, n_genes,
    sampling_distance, bandwidth, prune_coefficient,
    dim, ncores=None,
    device='cuda', kernel='gaussian', _split=True, _grid_origin=None, _qbatch=True,
):
    """
    GPU equivalent of ``ssaft.utils.kde_at_points`` (2D and 3D).

    Same positional signature; ``ncores`` is accepted but ignored.
    Returns a ``float32`` numpy.ndarray of shape (N, n_genes).

    :param query_x: x coordinates of query points.
    :type query_x: array-like
    :param query_y: y coordinates of query points.
    :type query_y: array-like
    :param query_z: z coordinates of query points (used only for ``dim == 3``).
    :type query_z: array-like
    :param loc_x: x coordinates of transcript locations.
    :type loc_x: array-like
    :param loc_y: y coordinates of transcript locations.
    :type loc_y: array-like
    :param loc_z: z coordinates of transcript locations (used only for ``dim == 3``).
    :type loc_z: array-like
    :param gene_indices: per-transcript gene index.
    :type gene_indices: array-like
    :param n_genes: number of genes (output column count).
    :type n_genes: int
    :param sampling_distance: cell-list grid spacing.
    :type sampling_distance: float
    :param bandwidth: gaussian-equivalent KDE bandwidth; each kernel's own scale is
        matched to it on variance (:func:`ssaft._count.kernel_scale`).
    :type bandwidth: float
    :param prune_coefficient: truncation radius of the infinite-support kernels in
        units of bandwidth (gaussian; the exponential is cut at the same relative
        height). The compact kernels stop at their own support.
    :type prune_coefficient: float
    :param kernel: one of ``ssaft._count.KDE_KERNELS``; ``'gaussian'`` (default),
        ``'tophat'``, ``'epanechnikov'``, ``'exponential'``, ``'linear'``, ``'cosine'``.
    :type kernel: str
    :param dim: spatial dimensionality (2 or 3).
    :type dim: int
    :param ncores: accepted for signature parity with the C++ path; ignored.
    :type ncores: int or None
    :param device: torch device string.
    :type device: str
    :param _split: internal. When ``True``, a three-dimensional call whose cell list
        would exceed the memory cap is split into strips along the longest axis and
        each strip is solved separately. Set ``False`` on the recursive calls, which
        must not split again.
    :type _split: bool
    :param _grid_origin: internal. The cell grid's origin, in the same frame as the
        coordinates. Passing it pins every strip of a split call to one common
        lattice, so that a molecule falls in the same cell whichever strip solves it.
        ``None`` lets each call choose its own origin from its own bounding box.
    :param _qbatch: internal. When ``True``, the queries are first grouped into
        spatial batches so that each batch builds a cell list over its own
        neighbourhood rather than over the whole section. Set ``False`` on the
        recursive calls, which are already one batch.
    :type _qbatch: bool
    """
    if not _HAS_TRITON:
        raise ImportError("kde_at_points_gpu requires torch + triton.")
    if dim not in (2, 3):
        raise NotImplementedError(
            f"kde_at_points_gpu supports dim=2 or 3, got dim={dim}.")

    # ``bandwidth`` is gaussian-equivalent for every kernel; the kernel's own scale,
    # reach and norm come from the one table in ssaft._count
    from ._count import (resolve_kernel, kernel_code, kernel_scale, kernel_reach,
                         kernel_norm, kernel_search_cells)
    kernel = resolve_kernel(kernel)
    kcode = kernel_code(kernel)
    scale = kernel_scale(bandwidth, dim, kernel)
    Rw = kernel_reach(scale, kernel, prune_coefficient)
    Rw2 = Rw * Rw
    two_h2 = 2.0 * scale * scale
    inv_scale = 1.0 / scale
    inv_norm = 1.0 / kernel_norm(scale, dim, kernel)
    inv_sd = 1.0 / sampling_distance
    Rp = kernel_search_cells(Rw, sampling_distance, kernel)

    # the (queries x genes) output alone can exceed the card (a 5k-gene panel: ~23 GB for one plane's
    # anchors): score the queries in batches that fit, on the cell grid the whole query set would have
    # had, so every batch sees exactly the transcripts the single call would
    n_q = len(query_x)
    if _qbatch and n_q:
        free = torch.cuda.mem_get_info(torch.device(device))[0]
        max_q = max(1, int(0.25 * free) // (4 * max(int(n_genes), 1)))
        if n_q > max_q:
            lx_, ly_ = np.asarray(loc_x, dtype=np.float64), np.asarray(loc_y, dtype=np.float64)
            qx_, qy_ = np.asarray(query_x, dtype=np.float64), np.asarray(query_y, dtype=np.float64)
            origin = _grid_origin
            if origin is None:
                oz = 0.0
                if int(dim) == 3:
                    oz = float(min(np.min(loc_z), np.min(query_z))) - Rw - sampling_distance
                origin = (float(min(lx_.min(), qx_.min())) - Rw - sampling_distance,
                          float(min(ly_.min(), qy_.min())) - Rw - sampling_distance, oz)
            qz_ = np.asarray(query_z, dtype=np.float64) if int(dim) == 3 else query_z
            out = np.empty((n_q, int(n_genes)), dtype=np.float32)
            gi_ = np.asarray(gene_indices, dtype=np.int64)
            # Every batch uploads the transcripts it can reach, not all of them: a 5k-gene Xenium
            # slide carries 368 M transcripts, whose coordinates plus the cell-list sort are 27 GB
            # on their own and leave nothing for the scoring. Ordering the queries by x makes each
            # batch a strip, so the reachable set is a contiguous run of the x-sorted transcripts.
            # Whole-set upload is kept for the small case, where the sort costs more than it saves.
            # SSAFT_KDE_STRIP=1/0 forces the choice (the equivalence test uses it)
            _sf = os.environ.get('SSAFT_KDE_STRIP')
            strip = (_sf == '1') if _sf in ('0', '1') else (lx_.nbytes * 4 > 0.25 * free)
            if strip:
                lorder = np.argsort(lx_, kind='stable')
                lxs = lx_[lorder]
                qorder = np.argsort(qx_, kind='stable')
            else:
                lorder = lxs = qorder = None
            # a strip's transcripts cost 24 B each on the device plus the cell-list sort (~40 B/row
            # in all), so the batch is cut by whichever bound comes first: max_q queries or this
            # many reachable transcripts. A tall slide (CosMx 6K: 431 M transcripts over 102 k px of
            # y) fills an x-strip with the whole column, which is what ran a 24 GB card out of
            # memory even after the strip split.
            loc_budget = max(4_000_000, int(0.30 * free) // 40)
            reach = Rw + sampling_distance
            s0 = 0
            while s0 < n_q:
                size = min(max_q, n_q - s0)
                while strip and size > 1:
                    q_ = qorder[s0:s0 + size]
                    a = int(np.searchsorted(lxs, qx_[q_].min() - reach, 'left'))
                    b = int(np.searchsorted(lxs, qx_[q_].max() + reach, 'right'))
                    if b - a <= loc_budget:
                        break
                    size //= 2                      # narrower strip, fewer transcripts in reach
                s1 = s0 + size
                q = slice(s0, s1) if qorder is None else qorder[s0:s1]
                if strip:
                    a = int(np.searchsorted(lxs, qx_[q].min() - reach, 'left'))
                    b = int(np.searchsorted(lxs, qx_[q].max() + reach, 'right'))
                    sel = lorder[a:b]
                    lsx, lsy = lx_[sel], ly_[sel]
                    lsz = np.asarray(loc_z, dtype=np.float64)[sel] if int(dim) == 3 else loc_z
                    lsg = gi_[sel]
                else:
                    lsx, lsy, lsz, lsg = loc_x, loc_y, loc_z, gene_indices
                out[q] = kde_at_points_gpu(
                    qx_[q], qy_[q], (qz_[q] if int(dim) == 3 else query_z),
                    lsx, lsy, lsz, lsg, n_genes,
                    sampling_distance, bandwidth, prune_coefficient, dim,
                    device=device, kernel=kernel, _split=_split, _grid_origin=origin, _qbatch=False)
                s0 = s1
            return out

    try:
        lx_g = torch.as_tensor(loc_x, dtype=torch.float64, device=device).contiguous()
        ly_g = torch.as_tensor(loc_y, dtype=torch.float64, device=device).contiguous()
        gi_g = torch.as_tensor(np.asarray(gene_indices, dtype=np.int64),
                               device=device).contiguous()
        qx_g = torch.as_tensor(query_x, dtype=torch.float64, device=device).contiguous()
        qy_g = torch.as_tensor(query_y, dtype=torch.float64, device=device).contiguous()
        if int(dim) == 3:
            lz_g = torch.as_tensor(loc_z, dtype=torch.float64, device=device).contiguous()
            qz_g = torch.as_tensor(query_z, dtype=torch.float64, device=device).contiguous()
    except torch.cuda.OutOfMemoryError:
        # not even the inputs fit (the GPU is shared): in 3D, two strips, each uploading only its
        # own transcripts; the grid origin is the one the whole domain would have had
        torch.cuda.empty_cache()
        if int(dim) != 3 or not _split or len(query_x) == 0:
            raise
        lx_, ly_, lz_ = (np.asarray(v, dtype=np.float64) for v in (loc_x, loc_y, loc_z))
        qx_, qy_, qz_ = (np.asarray(v, dtype=np.float64) for v in (query_x, query_y, query_z))
        origin = (float(min(lx_.min(), qx_.min())) - Rw - sampling_distance,
                  float(min(ly_.min(), qy_.min())) - Rw - sampling_distance,
                  float(min(lz_.min(), qz_.min())) - Rw - sampling_distance)
        return _kde_3d_in_strips(qx_, qy_, qz_, lx_, ly_, lz_, gene_indices, n_genes,
                                 sampling_distance, bandwidth, prune_coefficient, device, kernel,
                                 pad=Rw + 2 * sampling_distance, n_strips=2, origin=origin)

    N = qx_g.shape[0]
    if N == 0:
        return np.zeros((0, n_genes), dtype=np.float32)

    x_min = float(min(lx_g.min().item(), qx_g.min().item())) - Rw - sampling_distance
    y_min = float(min(ly_g.min().item(), qy_g.min().item())) - Rw - sampling_distance
    x_max = float(max(lx_g.max().item(), qx_g.max().item())) + Rw + sampling_distance
    y_max = float(max(ly_g.max().item(), qy_g.max().item())) + Rw + sampling_distance
    if _grid_origin is not None:
        # a strip of a larger domain: put its cells exactly where the whole domain's cells are,
        # so the neighbour-cell search (floor(Rw / sd) cells) includes the same transcripts
        x_min = _grid_origin[0] + math.floor((x_min - _grid_origin[0]) * inv_sd) * sampling_distance
        y_min = _grid_origin[1] + math.floor((y_min - _grid_origin[1]) * inv_sd) * sampling_distance
    W = int(math.ceil((x_max - x_min) * inv_sd)) + 1
    H = int(math.ceil((y_max - y_min) * inv_sd)) + 1

    out = torch.zeros((N, n_genes), dtype=torch.float32, device=device)

    if dim == 2:
        sx, sy, sg, cell_start = _build_cell_list(
            lx_g, ly_g, gi_g, sampling_distance, x_min, y_min, W, H,
        )
        with torch.cuda.device(qx_g.device):
            _kde_cell_kernel[(N,)](
                qx_g, qy_g,
                sx, sy, sg,
                cell_start,
                out,
                N, n_genes, W, H, Rp,
                x_min, y_min,
                inv_sd, two_h2, inv_norm, Rw2, inv_scale,
                BLOCK_T=32, KERN=kcode,
            )
    else:
        z_min = float(min(lz_g.min().item(), qz_g.min().item())) - Rw - sampling_distance
        z_max = float(max(lz_g.max().item(), qz_g.max().item())) + Rw + sampling_distance
        if _grid_origin is not None:
            z_min = _grid_origin[2] + math.floor((z_min - _grid_origin[2]) * inv_sd) * sampling_distance
        D = int(math.ceil((z_max - z_min) * inv_sd)) + 1

        n_cells = W * H * D
        cell_bytes = (n_cells + 1) * 4
        if cell_bytes > _MAX_CELL_LIST_BYTES and not _split:
            raise RuntimeError(
                f"3D cell list too large even for one strip: W={W}, H={H}, D={D} -> "
                f"{cell_bytes / (1 << 30):.1f} GiB")
        if cell_bytes > _MAX_CELL_LIST_BYTES:
            # the cell list spans the whole domain and lives on the GPU; split the queries into x
            # strips instead, each with only the transcripts within the kernel radius of it. A
            # query's value depends only on those transcripts, so the result is unchanged.
            del out, lx_g, ly_g, lz_g, gi_g, qx_g, qy_g, qz_g     # the strips upload their own subsets
            torch.cuda.empty_cache()
            n_strips = int(math.ceil(cell_bytes / _STRIP_CELL_LIST_BYTES))
            return _kde_3d_in_strips(query_x, query_y, query_z, loc_x, loc_y, loc_z,
                                     gene_indices, n_genes, sampling_distance, bandwidth,
                                     prune_coefficient, device, kernel,
                                     pad=Rw + 2 * sampling_distance, n_strips=n_strips,
                                     origin=(x_min, y_min, z_min))

        try:
            sx, sy, sz, sg, cell_start = _build_cell_list_3d(
                lx_g, ly_g, lz_g, gi_g, sampling_distance,
                x_min, y_min, z_min, W, H, D,
            )
            with torch.cuda.device(qx_g.device):
                _kde_cell_kernel_3d[(N,)](
                    qx_g, qy_g, qz_g,
                    sx, sy, sz, sg,
                    cell_start,
                    out,
                    N, n_genes, W, H, D, Rp,
                    x_min, y_min, z_min,
                    inv_sd, two_h2, inv_norm, Rw2, inv_scale,
                    BLOCK_T=32, KERN=kcode,
                )
        except torch.cuda.OutOfMemoryError:
            if not _split:
                raise                      # a strip: _kde_3d_in_strips halves it and retries
            # the GPU is short (another job on it): the same computation in two strips
            del out, lx_g, ly_g, lz_g, gi_g, qx_g, qy_g, qz_g
            torch.cuda.empty_cache()
            return _kde_3d_in_strips(query_x, query_y, query_z, loc_x, loc_y, loc_z,
                                     gene_indices, n_genes, sampling_distance, bandwidth,
                                     prune_coefficient, device, kernel,
                                     pad=Rw + 2 * sampling_distance, n_strips=2,
                                     origin=(x_min, y_min, z_min))
    return out.cpu().numpy()


# Budgets for the int32 cell list; building it also needs int64 temporaries (the arange and the
# searchsorted result), ~20 bytes per cell in all, so 1 GiB here is ~5 GiB of GPU memory at peak.
_MAX_CELL_LIST_BYTES = 1 << 30       # a larger domain is split into x strips
_STRIP_CELL_LIST_BYTES = 512 << 20   # target per strip


def _wait_for_gpu_and_retry(fn, tries=60, wait_s=60):
    """Call ``fn`` until it stops running out of GPU memory, waiting ``wait_s`` between attempts.

    For the smallest KDE strip: when even that does not fit, the card is held by other jobs, and the
    only way through is to wait for them to release memory (up to ``tries x wait_s`` seconds)."""
    import time
    for attempt in range(tries):
        try:
            return fn()
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if not isinstance(e, torch.cuda.OutOfMemoryError) and "out of memory" not in str(e):
                raise
            torch.cuda.empty_cache()
            if attempt == tries - 1:
                raise
            time.sleep(wait_s)


def _kde_3d_in_strips(query_x, query_y, query_z, loc_x, loc_y, loc_z, gene_indices, n_genes,
                      sampling_distance, bandwidth, prune_coefficient, device, kernel, *,
                      pad, n_strips, origin):
    """3D KDE over x strips of the query points (see :func:`kde_at_points_gpu`)."""
    qx = np.asarray(query_x, dtype=np.float64)
    qy = np.asarray(query_y, dtype=np.float64)
    qz = np.asarray(query_z, dtype=np.float64)
    lx = np.asarray(loc_x, dtype=np.float64)
    ly = np.asarray(loc_y, dtype=np.float64)
    lz = np.asarray(loc_z, dtype=np.float64)
    gi = np.asarray(gene_indices)
    out = np.zeros((qx.shape[0], n_genes), dtype=np.float32)
    edges = np.linspace(qx.min(), qx.max(), n_strips + 1)
    for i in range(n_strips):
        lo, hi = edges[i], edges[i + 1]
        qm = (qx >= lo) & ((qx < hi) if i < n_strips - 1 else (qx <= hi))
        if not qm.any():
            continue
        lm = (lx >= lo - pad) & (lx <= hi + pad)
        if not lm.any():
            continue
        try:
            out[qm] = kde_at_points_gpu(qx[qm], qy[qm], qz[qm], lx[lm], ly[lm], lz[lm], gi[lm],
                                        n_genes, sampling_distance, bandwidth, prune_coefficient,
                                        3, device=device, kernel=kernel, _split=False,
                                        _grid_origin=origin)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if not isinstance(e, torch.cuda.OutOfMemoryError) and "out of memory" not in str(e) \
                    and "too large" not in str(e):
                raise
            # not enough GPU memory for this strip: halve it and retry (same numbers, smaller pieces)
            torch.cuda.empty_cache()
            if hi - lo > 2 * pad:
                out[qm] = _kde_3d_in_strips(qx[qm], qy[qm], qz[qm], lx[lm], ly[lm], lz[lm], gi[lm],
                                            n_genes, sampling_distance, bandwidth, prune_coefficient,
                                            device, kernel, pad=pad, n_strips=2, origin=origin)
                continue
            # already as small as a strip gets: the card is full of other jobs' memory, and no
            # smaller piece will fit -- wait for them to release it rather than fail the stage
            out[qm] = _wait_for_gpu_and_retry(
                lambda: kde_at_points_gpu(qx[qm], qy[qm], qz[qm], lx[lm], ly[lm], lz[lm], gi[lm],
                                          n_genes, sampling_distance, bandwidth, prune_coefficient,
                                          3, device=device, kernel=kernel, _split=False,
                                          _grid_origin=origin))
    return out


@contextlib.contextmanager
def gpu_kde(device: str = 'cuda', fallback_3d_to_cpu: bool = False,
            force_2d: bool = False):
    """Route every kernel density call inside this block to the GPU implementation.

    While the block is active, every call to ``ssaft.utils.kde_at_points`` goes to the GPU
    implementation, so every higher-level function that calls it internally uses the GPU
    too. That includes ``downsample_vector``, ``downsample_vector_component`` and
    ``_compute_normalized_vectors``. The C++ implementation is restored on exit and
    nothing is modified permanently.

    Parameters
    ----------
    device : str
        Torch device string, for example ``'cuda'``, ``'cuda:0'`` or ``'cuda:1'``.
    fallback_3d_to_cpu : bool
        Whether to route three-dimensional inputs to the original C++ implementation.
        This defaults to ``False``, because the GPU kernel now supports three
        dimensions natively. Set it to ``True`` if you run into GPU memory limits on a
        very large volume and would rather fall back to the processor than reduce the
        chunk size.
    force_2d : bool
        Whether to project three-dimensional inputs down to two dimensions, by zeroing
        z, and run them on the planar GPU kernel. This defaults to ``False``. It is
        useful for thin-slice data, such as a Xenium z stack within one section, when a
        lighter cell-list footprint than full three dimensions is wanted.

    Example
    -------
    >>> from ssaft._kde_gpu import gpu_kde
    >>> with gpu_kde(device='cuda:0'):                # native 3D on the GPU
    ...     stats = ds.downsample_vector(he_path=..., sdata=...)
    >>> with gpu_kde(device='cuda:0', force_2d=True): # collapse z for speed and memory
    ...     stats = ds.downsample_vector(he_path=..., sdata=...)
    """
    if not _HAS_TRITON:
        raise ImportError("gpu_kde requires torch + triton.")

    from . import utils as _utils
    original = _utils.kde_at_points
    warned_fallback = {'flag': False}

    # Same signature as the compiled kde_at_points, ``kernel`` (its integer id) included.
    # Dropping it would run every call at kde_at_points_gpu's own default kernel,
    # whatever the caller asked for.
    def _wrapped(query_x, query_y, query_z,
                 loc_x, loc_y, loc_z,
                 gene_indices, n_genes,
                 sampling_distance, bandwidth, prune_coefficient,
                 dim, ncores=None, kernel=0):
        """Stand in for ``ssaft.utils.kde_at_points`` while the context manager is active.

        The signature matches the compiled function exactly, including the ``kernel``
        argument, which carries the kernel's integer identifier. Dropping that argument would
        silently run every call with this module's own default kernel, whatever the caller
        asked for.

        Two-dimensional input goes straight to the GPU. Three-dimensional input is projected
        to two dimensions when ``force_2d`` is set, is handed back to the C++ implementation
        when ``fallback_3d_to_cpu`` is set, with a warning issued once, and otherwise runs on
        the native three-dimensional GPU kernel.

        :param query_x: x coordinates of the query points.
        :param query_y: y coordinates of the query points.
        :param query_z: z coordinates of the query points.
        :param loc_x: x coordinates of the molecules.
        :param loc_y: y coordinates of the molecules.
        :param loc_z: z coordinates of the molecules.
        :param gene_indices: Gene index of each molecule.
        :param n_genes: Width of the output.
        :param sampling_distance: Lattice spacing, which is also the cell size.
        :param bandwidth: The kernel's own scale.
        :param prune_coefficient: Truncation of the kernel, in units of its scale.
        :param dim: Either 2 or 3.
        :param ncores: Accepted for signature compatibility and ignored on the GPU.
        :param kernel: Integer identifier of the kernel, indexing
            ``ssaft._count.KDE_KERNELS``.
        :returns: The estimate, of shape ``(N, n_genes)`` and dtype float32.
        :rtype: numpy.ndarray
        """
        from ._count import KDE_KERNELS
        kname = KDE_KERNELS[int(kernel)]
        if int(dim) != 2:
            if force_2d:
                qz0 = np.zeros_like(np.asarray(query_z, dtype=np.float64))
                lz0 = np.zeros_like(np.asarray(loc_z, dtype=np.float64))
                return kde_at_points_gpu(
                    query_x, query_y, qz0,
                    loc_x, loc_y, lz0,
                    gene_indices, n_genes,
                    sampling_distance, bandwidth, prune_coefficient,
                    2, device=device, kernel=kname,
                )
            if fallback_3d_to_cpu:
                if not warned_fallback['flag']:
                    warnings.warn(
                        f"gpu_kde: dim={int(dim)} routed to C++ kde_at_points "
                        f"(CPU) per fallback_3d_to_cpu=True.",
                        RuntimeWarning, stacklevel=2,
                    )
                    warned_fallback['flag'] = True
                return original(
                    query_x, query_y, query_z,
                    loc_x, loc_y, loc_z,
                    gene_indices, n_genes,
                    sampling_distance, bandwidth, prune_coefficient,
                    int(dim), int(ncores) if ncores is not None else 1,
                    kernel=int(kernel),
                )
        return kde_at_points_gpu(
            query_x, query_y, query_z,
            loc_x, loc_y, loc_z,
            gene_indices, n_genes,
            sampling_distance, bandwidth, prune_coefficient,
            int(dim), device=device, kernel=kname,
        )

    _utils.kde_at_points = _wrapped
    try:
        yield
    finally:
        _utils.kde_at_points = original


def gpu_available(device: str = 'cuda') -> bool:
    """Return True iff torch+triton+CUDA are usable on the requested device.

    :param device: torch device string to test (e.g. ``'cuda'`` or ``'cuda:1'``).
    :type device: str
    """
    if not _HAS_TRITON:
        return False
    try:
        if not torch.cuda.is_available():
            return False
        if device.startswith('cuda:'):
            idx = int(device.split(':', 1)[1])
            return idx < torch.cuda.device_count()
        return True
    except Exception:
        return False


@contextlib.contextmanager
def auto_kde_context(device: str = 'cuda', **gpu_kwargs):
    """Auto-dispatch context manager.

    - If a usable CUDA device is found, behaves like ``gpu_kde(device, **gpu_kwargs)``
      (i.e. temporarily swaps ``ssaft.utils.kde_at_points`` for the Triton kernel).
    - Otherwise yields without patching, so the original C++ CPU implementation
      is used.

    Nesting is safe: an outer ``gpu_kde``/``auto_kde_context`` already in
    effect is preserved by ``gpu_kde``'s save-and-restore protocol.

    :param device: torch device string to dispatch on.
    :type device: str
    :param gpu_kwargs: forwarded to :func:`gpu_kde` when a CUDA device is used.
    """
    if gpu_available(device):
        with gpu_kde(device=device, **gpu_kwargs):
            yield
    else:
        yield
