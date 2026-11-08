"""GPU helpers used by ``Analysis.map_celltypes_gpu``.

This module reuses the Triton kernel-density kernel and the cell-list builders from
``ssaft._kde_gpu`` and adds three primitives. ``_kde_setup_celllist`` builds the cell
list once per outer chunk. ``_kde_run_subchunk`` runs the kernel density estimate for
one sub-chunk of query points against a cell list that already exists, so nothing is
rebuilt. ``_pearson_corr_gpu`` correlates a set of centroids of shape k by g against a
set of rows of shape n by g, returning a k by n matrix, entirely on the GPU through
``torch.matmul``.

Together these let ``map_celltypes_gpu`` stream chunks the way ``map_celltypes_he``
does while staying on the GPU all the way through the correlation step.

The results agree with the C++ ``map_celltypes_he`` path to the following tolerances.
The relative difference in the row-wise L1 norm of the kernel density estimate is below
2e-6, a bound the Triton kernel inherits from the one documented in ``_kde_gpu``. The
absolute difference in Pearson correlation is below 1e-5, which is the gap between a
cuBLAS matrix multiply and AVX512 fused multiply-add. The final ``celltype_maps`` may
still flip a label at a decision boundary, which in well separated cases affects far
less than 0.1 percent of voxels.
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
    """Build a cell list spanning the chunk's query bounding box and its transcripts.

    :param loc_x_t: Transcript x coordinates on the device.
    :param loc_y_t: Transcript y coordinates on the device.
    :param loc_z_t: Transcript z coordinates on the device, used only when ``dim`` is 3.
    :param gi_t: Gene index of each transcript.
    :param chunk_q_xmin: Lowest query x coordinate in this chunk.
    :param chunk_q_ymin: Lowest query y coordinate in this chunk.
    :param chunk_q_zmin: Lowest query z coordinate in this chunk.
    :param chunk_q_xmax: Highest query x coordinate in this chunk.
    :param chunk_q_ymax: Highest query y coordinate in this chunk.
    :param chunk_q_zmax: Highest query z coordinate in this chunk.
    :param sampling_distance: Spacing of the lattice, which is also the cell size.
    :param bandwidth: The kernel's own scale, already converted from the Gaussian
        equivalent scale by :func:`ssaft._count.kernel_scale`. The reach follows from it
        through :func:`ssaft._count.kernel_reach`, so ``prune_coefficient`` matters only
        for the Gaussian and the exponential kernels.
    :param prune_coefficient: Truncation of the kernel, measured in units of its scale.
    :param dim: Either 2 or 3, selecting the planar or volumetric cell list.
    :type dim: int
    :param kernel: One of ``ssaft._count.KDE_KERNELS``.
    :type kernel: str
    :returns: A state dictionary consumed by :func:`_kde_run_subchunk`, or ``None`` when
        there are no transcripts, in which case the caller treats the estimate as zero
        everywhere.
    :rtype: dict or None
    :raises ImportError: If torch and Triton are not both available.
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
    """Mark which voxels of a query slab can possibly have a nonzero density.

    :func:`_kde_run_subchunk` sums only the transcripts lying in the cells within
    :func:`ssaft._count.kernel_search_cells` of the query's own cell. A voxel with no
    transcript inside a cube of that half-width plus two voxels around it is therefore
    exactly zero, and skipping it changes nothing. The two extra voxels absorb the offset
    between the cell grid's origin and the query lattice. On a three-dimensional section
    most of the bounding box consists of such voxels, around 70 percent on the XEN010
    sample, so the saving is large.

    The occupancy grid is dilated on the device with three separable one-dimensional max
    pooling passes. Query voxel ``(i, j, k)`` of the slab sits at ``((x0 + i) * sd, j *
    sd, k * sd)`` in the frame of the ``loc_`` arrays, which is the frame the queries
    are built in.

    :param loc_x: Transcript x coordinates on the device.
    :param loc_y: Transcript y coordinates on the device.
    :param loc_z: Transcript z coordinates on the device, or ``None`` in two dimensions.
    :param x0: Index of the slab's first voxel along x.
    :type x0: int
    :param nx: Number of voxels of the slab along x.
    :type nx: int
    :param ny: Number of voxels of the slab along y.
    :type ny: int
    :param nz: Number of voxels of the slab along z.
    :type nz: int
    :param sampling_distance: Spacing of the lattice.
    :param reach: Radius beyond which the kernel is treated as zero.
    :param kernel: One of ``ssaft._count.KDE_KERNELS``.
    :type kernel: str
    :returns: A flat boolean array in C order over ``(nx, ny, nz)``, true where the
        density can be nonzero.
    :rtype: numpy.ndarray
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
    """Run the kernel density estimate for a sub-chunk of query points.

    The cell list in ``state`` is reused as it stands, so nothing is rebuilt between
    sub-chunks.

    :param state: The state dictionary returned by :func:`_kde_setup_celllist`, or
        ``None`` when the chunk held no transcripts.
    :type state: dict or None
    :param qx: Query x coordinates.
    :param qy: Query y coordinates.
    :param qz: Query z coordinates, used only for a three-dimensional state.
    :param n_genes: Width of the output, meaning the number of genes.
    :type n_genes: int
    :param device: The torch device to allocate the output on.
    :returns: A float32 tensor of shape ``(N, n_genes)`` on ``device``, which is all
        zeros when there are no queries or no transcripts.
    :rtype: torch.Tensor
    :raises ImportError: If torch and Triton are not both available.
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
    """Correlate every centroid against every row, on the GPU.

    :param centroids_t: Centroids of shape ``(k, g)``, assumed float32.
    :type centroids_t: torch.Tensor
    :param data_t: Row vectors of shape ``(n, g)``, assumed float32.
    :type data_t: torch.Tensor
    :returns: Pearson correlations of shape ``(k, n)``, float32, on the same device.
    :rtype: torch.Tensor
    :raises ImportError: If torch is not available.
    """
    if not _HAS_TRITON:
        raise ImportError("_pearson_corr_gpu requires torch.")
    c = centroids_t - centroids_t.mean(dim=1, keepdim=True)
    d = data_t - data_t.mean(dim=1, keepdim=True)
    cn = torch.clamp(torch.linalg.norm(c, dim=1), min=1e-12)
    dn = torch.clamp(torch.linalg.norm(d, dim=1), min=1e-12)
    return (c @ d.T) / (cn[:, None] * dn[None, :])


def _pearson_max_gpu(centroids_t, data_t):
    """Return the best-matching centroid and its Pearson score for every row.

    This is the single mapping kernel shared by both paths. The count path reads the field
    from the stored ``vf_normalized``, and the transcript path streams a kernel density
    field chunk by chunk, but both finish here, so a result never depends on how the field
    was produced.

    :param centroids_t: Centroids of shape ``(k, g)``.
    :type centroids_t: torch.Tensor
    :param data_t: Row vectors of shape ``(n, g)``.
    :type data_t: torch.Tensor
    :returns: A pair of tensors on the same device. The first holds the index of the
        winning centroid for each row as int32, and the second holds that row's
        correlation as float32.
    :rtype: tuple of torch.Tensor
    """
    corr = _pearson_corr_gpu(centroids_t, data_t)
    best_corr, best_idx = corr.max(dim=0)
    return best_idx.to(torch.int32), best_corr.to(torch.float32)
