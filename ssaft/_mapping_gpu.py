"""GPU helpers for ``Analysis.map_celltypes_gpu``.

Reuses the Triton KDE kernel + cell-list builders from
``ssaft._kde_gpu`` and adds:

* ``_kde_setup_celllist``  — build the cell-list once per outer chunk.
* ``_kde_run_subchunk``    — run KDE for one sub-chunk of query points
                             using a pre-built cell-list (no rebuild).
* ``_pearson_corr_gpu``    — (k, g) × (n, g) → (k, n) Pearson correlation
                             entirely on GPU via torch.matmul.

These primitives let ``map_celltypes_gpu`` stream chunks like
``map_celltypes_he`` while staying on GPU through the correlation step.

Numerical equivalence vs the C++ ``map_celltypes_he`` path:
- KDE row-L1 relative diff <2e-6 (Triton kernel inherits the same
  bound documented in ``_kde_gpu``).
- Pearson correlation absolute diff <1e-5 (cuBLAS GEMM vs AVX512 FMA).
- Final ``celltype_maps`` may flip labels at decision boundaries
  (≪0.1% of voxels in well-separated cluster cases).
"""
from __future__ import annotations

import math

import numpy as np

try:
    import torch
    from ._kde_gpu import (
        _kde_cell_kernel,
        _kde_cell_kernel_3d,
        _build_cell_list,
        _build_cell_list_3d,
    )
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


def _kde_setup_celllist(loc_x_t, loc_y_t, loc_z_t, gi_t,
                        chunk_q_xmin, chunk_q_ymin, chunk_q_zmin,
                        chunk_q_xmax, chunk_q_ymax, chunk_q_zmax,
                        sampling_distance, bandwidth, prune_coefficient,
                        dim, kernel='gaussian'):
    """Build a cell-list spanning the chunk's query bbox + transcripts.

    Returns a state dict consumed by :func:`_kde_run_subchunk`, or
    ``None`` when there are no transcripts (caller treats the result
    as all-zero KDE).

    :param kernel: one of ``ssaft._count.KDE_KERNELS`` (default ``'gaussian'``).
        ``bandwidth`` here is the kernel's OWN scale, already converted from the
        gaussian-equivalent one (:func:`ssaft._count.kernel_scale`); the reach is
        :func:`ssaft._count.kernel_reach` of it, so ``prune_coefficient`` matters
        only for the gaussian and the exponential.
    """
    if not _HAS_TRITON:
        raise ImportError("_kde_setup_celllist requires torch + triton.")
    if loc_x_t.shape[0] == 0:
        return None

    from ._count import resolve_kernel, kernel_reach
    kernel = resolve_kernel(kernel)
    Rw = kernel_reach(bandwidth, kernel, prune_coefficient)
    inv_sd = 1.0 / sampling_distance

    x_min = float(min(loc_x_t.min().item(), chunk_q_xmin)) - Rw - sampling_distance
    y_min = float(min(loc_y_t.min().item(), chunk_q_ymin)) - Rw - sampling_distance
    x_max = float(max(loc_x_t.max().item(), chunk_q_xmax)) + Rw + sampling_distance
    y_max = float(max(loc_y_t.max().item(), chunk_q_ymax)) + Rw + sampling_distance
    W = int(math.ceil((x_max - x_min) * inv_sd)) + 1
    H = int(math.ceil((y_max - y_min) * inv_sd)) + 1

    state = {
        'dim': int(dim),
        'sampling_distance': float(sampling_distance),
        'bandwidth': float(bandwidth),
        'prune_coefficient': float(prune_coefficient),
        'kernel': kernel,
        'x_min': x_min, 'y_min': y_min, 'W': W, 'H': H,
    }

    if dim == 2:
        sx, sy, sg, cell_start = _build_cell_list(
            loc_x_t, loc_y_t, gi_t, sampling_distance, x_min, y_min, W, H,
        )
        state.update(sx=sx, sy=sy, sg=sg, cell_start=cell_start)
    else:
        z_min = float(min(loc_z_t.min().item(), chunk_q_zmin)) - Rw - sampling_distance
        z_max = float(max(loc_z_t.max().item(), chunk_q_zmax)) + Rw + sampling_distance
        D = int(math.ceil((z_max - z_min) * inv_sd)) + 1
        sx, sy, sz, sg, cell_start = _build_cell_list_3d(
            loc_x_t, loc_y_t, loc_z_t, gi_t, sampling_distance,
            x_min, y_min, z_min, W, H, D,
        )
        state.update(sx=sx, sy=sy, sz=sz, sg=sg, cell_start=cell_start,
                     z_min=z_min, D=D)
    return state


def _kde_nonzero_queries(loc_x, loc_y, loc_z, x0, nx, ny, nz, sampling_distance, reach,
                         kernel='gaussian'):
    """Which voxels of a query slab can have a nonzero KDE: flat (C-order ``(nx, ny, nz)``) bool.

    ``_kde_run_subchunk`` sums only transcripts in the cells within
    :func:`ssaft._count.kernel_search_cells` of the query's cell, so a voxel with no transcript
    in a cube of half-width that + 2 voxels around it (the +2 absorbs the cell grid's own
    origin) is exactly zero; skipping it changes nothing. On a 3D section most of the bounding box is such voxels (XEN010: ~70%).
    The occupancy grid is dilated with three separable 1D max-pools on the device.
    Query voxel ``(i, j, k)`` of the slab sits at ``((x0 + i) sd, j sd, k sd)`` in the frame of
    ``loc_*`` (the frame the queries are built in).
    """
    import torch.nn.functional as F
    from ._count import kernel_search_cells
    r = kernel_search_cells(reach, sampling_distance, kernel) + 2
    dev = loc_x.device
    ix = torch.floor(loc_x / sampling_distance).long() - (x0 - r)
    iy = torch.floor(loc_y / sampling_distance).long() + r
    iz = (torch.floor(loc_z / sampling_distance).long() + r) if loc_z is not None else \
        torch.full_like(ix, r)
    px, py, pz = nx + 2 * r, ny + 2 * r, nz + 2 * r
    ok = (ix >= 0) & (ix < px) & (iy >= 0) & (iy < py) & (iz >= 0) & (iz < pz)
    occ = torch.zeros((px, py, pz), dtype=torch.float16, device=dev)
    occ[ix[ok], iy[ok], iz[ok]] = 1
    del ix, iy, iz, ok
    k = 2 * r + 1
    g = occ[None, None]
    g = F.max_pool3d(g, (k, 1, 1), stride=1)
    g = F.max_pool3d(g, (1, k, 1), stride=1)
    g = F.max_pool3d(g, (1, 1, k), stride=1)
    return (g[0, 0] > 0).reshape(-1).cpu().numpy()


def _kde_run_subchunk(state, qx, qy, qz, n_genes, device):
    """Run KDE for a sub-chunk of query points using ``state``.

    Returns a torch.float32 tensor of shape ``(N, n_genes)`` on
    ``device``.
    """
    if not _HAS_TRITON:
        raise ImportError("_kde_run_subchunk requires torch + triton.")

    qx_g = torch.as_tensor(qx, dtype=torch.float64, device=device).contiguous()
    N = int(qx_g.shape[0])
    if N == 0 or state is None:
        return torch.zeros((N, n_genes), dtype=torch.float32, device=device)

    qy_g = torch.as_tensor(qy, dtype=torch.float64, device=device).contiguous()

    from ._count import kernel_code, kernel_reach, kernel_norm, kernel_search_cells
    bw = state['bandwidth']              # the kernel's own scale
    pc = state['prune_coefficient']
    sd = state['sampling_distance']
    kernel = state.get('kernel', 'gaussian')
    kcode = kernel_code(kernel)
    dim = state['dim']
    Rw = kernel_reach(bw, kernel, pc)
    Rw2 = Rw * Rw
    two_h2 = 2.0 * bw * bw
    inv_scale = 1.0 / bw
    inv_norm = 1.0 / kernel_norm(bw, dim, kernel)
    inv_sd = 1.0 / sd
    Rp = kernel_search_cells(Rw, sd, kernel)

    out = torch.zeros((N, n_genes), dtype=torch.float32, device=device)

    if state['dim'] == 2:
        with torch.cuda.device(qx_g.device):
            _kde_cell_kernel[(N,)](
                qx_g, qy_g,
                state['sx'], state['sy'], state['sg'], state['cell_start'],
                out,
                N, n_genes, state['W'], state['H'], Rp,
                state['x_min'], state['y_min'],
                inv_sd, two_h2, inv_norm, Rw2, inv_scale,
                BLOCK_T=32, KERN=kcode,
            )
    else:
        qz_g = torch.as_tensor(qz, dtype=torch.float64, device=device).contiguous()
        with torch.cuda.device(qx_g.device):
            _kde_cell_kernel_3d[(N,)](
                qx_g, qy_g, qz_g,
                state['sx'], state['sy'], state['sz'], state['sg'], state['cell_start'],
                out,
                N, n_genes, state['W'], state['H'], state['D'], Rp,
                state['x_min'], state['y_min'], state['z_min'],
                inv_sd, two_h2, inv_norm, Rw2, inv_scale,
                BLOCK_T=32, KERN=kcode,
            )
    return out


def _pearson_corr_gpu(centroids_t, data_t):
    """Pearson correlation between (k, g) centroids and (n, g) rows.

    Returns a torch tensor of shape ``(k, n)`` on the same device.
    Both inputs are assumed float32; the result is float32.
    """
    if not _HAS_TRITON:
        raise ImportError("_pearson_corr_gpu requires torch.")
    c = centroids_t - centroids_t.mean(dim=1, keepdim=True)
    d = data_t - data_t.mean(dim=1, keepdim=True)
    cn = torch.clamp(torch.linalg.norm(c, dim=1), min=1e-12)
    dn = torch.clamp(torch.linalg.norm(d, dim=1), min=1e-12)
    return (c @ d.T) / (cn[:, None] * dn[None, :])


def _pearson_max_gpu(centroids_t, data_t):
    """Shared mapping kernel: best-centroid label + Pearson score per row.

    ``centroids_t`` (k, g) and ``data_t`` (n, g) → ``(best_idx (n,), best_corr
    (n,))`` on the same device. This is the single consumer used by BOTH the
    count path (field read from the stored ``vf_normalized``) and the transcript
    path (KDE field streamed chunk-by-chunk) — identical Pearson-max so results
    do not depend on how the field was produced. Returns int32 idx, float32 corr.
    """
    corr = _pearson_corr_gpu(centroids_t, data_t)
    best_corr, best_idx = corr.max(dim=0)
    return best_idx.to(torch.int32), best_corr.to(torch.float32)
