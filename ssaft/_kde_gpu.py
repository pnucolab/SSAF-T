"""
GPU equivalent of ``ssaft.utils.kde_at_points`` using a Triton
cell-list kernel.

This module is fully additive — it does not modify the C++ implementation;
it swaps it in only for the duration of a context manager:

    from ssaft._kde_gpu import gpu_kde
    with gpu_kde(device='cuda'):
        ds.downsample_vector(he_path=..., sdata=...)   # uses GPU internally
    # back to C++ outside the with-block

Numerical equivalence vs C++ ``kde_at_points``:
    max row-L1 relative diff < 2e-6 (verified on M=5M, N=5k, n_genes=280).

Supports both dim==2 and dim==3 (3D KDE). Note that the 3D cell-list array
size is W*H*D*4 bytes; for very large/anisotropic domains consider using
``force_2d=True`` to collapse z when 2D projection is acceptable.
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
        """One molecule's weight at squared distance ``r2`` (the caller masks r2 > Rw2).

        ``KERN`` is the id in ``ssaft._count.KDE_KERNELS``. The gaussian and
        Epanechnikov lines are the expressions these kernels have always used (the
        Epanechnikov's support H is ``sqrt(Rw2)``), so their results are unchanged.
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
    """
    Inside this with-block, every call to ``ssaft.utils.kde_at_points``
    is routed to the GPU implementation. All higher-level APIs that internally
    call ``kde_at_points`` (downsample_vector, downsample_vector_component,
    _compute_normalized_vectors, etc.) automatically use GPU.

    The C++ implementation is restored on exit; nothing is modified permanently.

    Parameters
    ----------
    device : str
        Torch device string (e.g. ``'cuda'``, ``'cuda:0'``, ``'cuda:1'``).
    fallback_3d_to_cpu : bool
        If True, 3D inputs are routed to the original C++ implementation.
        Default False — the GPU kernel now supports 3D natively. Set True
        if you hit GPU memory limits on very large 3D domains and prefer
        CPU over reducing the chunk size.
    force_2d : bool
        If True, 3D inputs are projected to 2D (z is zeroed) and run on
        the 2D GPU kernel. Default False. Useful for thin-slice data
        (e.g. Xenium z-stack within a single section) when you want a
        lighter cell-list footprint than full 3D.

    Example
    -------
    >>> from ssaft._kde_gpu import gpu_kde
    >>> with gpu_kde(device='cuda:0'):                # native 3D on GPU
    ...     stats = ds.downsample_vector(he_path=..., sdata=...)
    >>> with gpu_kde(device='cuda:0', force_2d=True): # collapse z for speed/memory
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
