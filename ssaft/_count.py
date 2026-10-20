"""Count-data vector-field backend for ssaft.

For count platforms the KDE-equivalent vector field is built by **rasterizing bin
counts onto a grid and GPU Gaussian-blurring** it (:func:`compute_vf_blur`),
persisting only ``vf_normalized`` + ``vf_norm`` to the zarr store. Voxels are
then sampled from the tissue image and *looked up* in that field
(:func:`select_vectors_from_image`) — the count analogue of the transcript path's
KDE-at-sampled-points. Cluster centroids are mapped over the field by correlation
(:func:`map_corr`).

All functions take the ``Analysis`` instance so they can read/write
``analysis.dataset.zarr_group`` and honour ``analysis.verbose``.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from scipy.sparse import issparse

try:
    from tqdm.auto import tqdm
except Exception:  # noqa: BLE001
    def tqdm(x, **k):
        return x


def _gaussian_kernel_1d(sigma_bin, device, prune_coefficient=4.3):
    import torch
    radius = int(np.ceil(prune_coefficient * sigma_bin))
    k = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    g = torch.exp(-k ** 2 / (2 * sigma_bin ** 2)); g /= g.sum()
    return g.view(1, 1, 1, -1), g.view(1, 1, -1, 1), radius


def _radial_kernel_2d(sigma_bin, kernel, device, prune_coefficient=4.3):
    """Dense radial ``kernel`` on the bin raster, sum-normalized.

    ``sigma_bin`` is the gaussian-equivalent sigma in bins; the kernel's own scale is
    derived from it by 2D variance matching (:func:`kernel_scale`), e.g. the
    Epanechnikov support ``H = sigma_bin * sqrt(6)``. Returned as a single
    ``(1, 1, 2r+1, 2r+1)`` kernel because only the gaussian is separable -- the other
    radial kernels need the full square of taps, one cuDNN launch. At the default
    sigma_bin=1 the compact kernels span 5x5 to 7x7 taps against the gaussian's
    2x11=22; the exponential's long tail takes 13x13.
    """
    import torch
    scale = kernel_scale(sigma_bin, 2, kernel)
    reach = kernel_reach(scale, kernel, prune_coefficient)
    radius = int(np.ceil(reach))
    k = np.arange(-radius, radius + 1, dtype=np.float64)
    r = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    w = np.where(r <= reach, kernel_profile(r / scale, kernel), 0.0)
    w /= w.sum()
    w = torch.as_tensor(w, dtype=torch.float32, device=device)
    return w.view(1, 1, 2 * radius + 1, 2 * radius + 1), radius


def _iter_spatial_chunks(n_R, chunk_h, radius):
    for r0 in range(0, n_R, chunk_h):
        r1 = min(r0 + chunk_h, n_R)
        yield r0, r1, max(0, r0 - radius), min(n_R, r1 + radius)


def _rasterize_blur_tile_gpu(Xt, Rt, Ct, rp0, rp1, cp0, cp1, n_G,
                             kx_g, ky_g, radius, device, dtype):
    """Scatter one (row band × col band) TILE of pre-sliced bins onto a GPU tensor
    and separable-Gaussian-blur it.

    ``Xt``/``Rt``/``Ct`` are ALREADY the bins of this tile (raster row in
    ``[rp0, rp1)`` AND raster col in ``[cp0, cp1)`` — the caller slices the
    column-sorted band with ``searchsorted``, so no boolean re-scan happens here).
    Only the nonzero coo inside the tile is transferred (no dense CPU array), and
    the conv runs in ``dtype`` (fp32). ``kx_g``/``ky_g`` are the per-gene expanded
    kernels, built ONCE by the caller (constant across tiles).

    The tile carries a ``±radius`` halo on every side (the caller sets ``rp0/rp1``
    and ``cp0/cp1`` accordingly), so the truncated separable Gaussian is EXACT for
    the interior output rows/cols the caller keeps — the halo rows/cols absorb the
    zero-padded edge effect and are discarded. Bounds GPU memory to
    ``n_G × (band_h + 2r) × (band_w + 2r)`` regardless of the full raster size.

    With ``ky_g=None`` the kernel is treated as a RADIAL 2D one (every kernel but the
    gaussian) and applied in a single dense conv; the halo logic is unchanged."""
    import torch
    import torch.nn.functional as F
    h = rp1 - rp0
    w = cp1 - cp0
    raster = torch.zeros((n_G, h, w), device=device, dtype=dtype)
    if Xt.shape[0] > 0:
        sub = Xt.tocoo()
        sub_R = (Rt - rp0)[sub.row]
        sub_C = (Ct - cp0)[sub.row]
        g = torch.as_tensor(sub.col, device=device, dtype=torch.long)
        r = torch.as_tensor(sub_R, device=device, dtype=torch.long)
        c = torch.as_tensor(sub_C, device=device, dtype=torch.long)
        v = torch.as_tensor(sub.data, device=device, dtype=dtype)
        raster.index_put_((g, r, c), v, accumulate=True)
    t = raster.unsqueeze(0)
    if ky_g is None:            # radial, not separable: one dense pass
        t = F.conv2d(t, kx_g, padding=(radius, radius), groups=n_G)
    else:
        t = F.conv2d(t, ky_g, padding=(radius, 0), groups=n_G)
        t = F.conv2d(t, kx_g, padding=(0, radius), groups=n_G)
    return t.squeeze(0)


def _l1_normalize_log1p(blurred_gpu, scale=10.0, eps=1e-12):
    import torch
    vf_norm = blurred_gpu.sum(dim=0)
    blurred_gpu.div_(torch.clamp(vf_norm, min=eps).unsqueeze(0)).mul_(scale)
    torch.log1p_(blurred_gpu)
    return blurred_gpu, vf_norm


def _tensorstore_array(group_dir, name, shape, chunks):
    """Open a store array as a tensorstore zarr-v3 array whose metadata MIRRORS
    the zarr-python writer EXACTLY (float32, bytes+zstd level 0, ``default``
    '/'-separated chunk keys, fill 0.0) so the on-disk store is read-back-identical
    and any zarr-v3 reader (zarr-python, the downstream sparse gather) sees no
    difference. Works for the 3D count field ``(n_G, R, C)`` and the 4D transcript
    field ``(n_G, vx, vy, vz)`` alike.

    Routing the large ``vf_normalized`` write through tensorstore's C++
    multithreaded async pipeline is ~4.7× faster than the single-event-loop
    zarr-python writer (~2.6× end-to-end since async writes overlap the next tile's
    GPU work) — the write is the dominant cost. Values are bit-identical (same zstd
    codec; lossless). Returns ``None`` if tensorstore is not installed, the store is
    not a local directory, or ``CELLMAN_NO_TENSORSTORE`` is set — the caller then
    falls back to the zarr-python writer.
    """
    if group_dir is None:
        return None
    if os.environ.get("CELLMAN_NO_TENSORSTORE"):
        return None
    try:
        import tensorstore as ts
    except Exception:  # noqa: BLE001
        return None
    spec = {
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": os.path.join(group_dir, name)},
        "metadata": {
            "shape": [int(s) for s in shape],
            "data_type": "float32",
            "chunk_grid": {"name": "regular",
                           "configuration": {"chunk_shape": [int(c) for c in chunks]}},
            "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
            "fill_value": 0.0,
            "codecs": [{"name": "bytes", "configuration": {"endian": "little"}},
                       {"name": "zstd", "configuration": {"level": 0, "checksum": False}}],
        },
    }
    return ts.open(spec, create=True, delete_existing=True).result()


class _SlabWriter:
    """Write x-slabs ``arr[x0:x1, ...]`` of existing store arrays through tensorstore, asynchronously.

    The per-voxel map outputs (vf_norm, celltype_maps, the score and marker volumes) are written
    one x-slab at a time; through zarr-python that is one event-loop thread (~35 MB/s on this
    storage), which made writing a 3D map take longer than computing it. The arrays are opened
    as they are on disk (zarr.json read, not re-declared), so dtype, fill value and codecs are
    the store's own and the bytes are identical. Up to ``max_inflight`` slab writes overlap the
    next slab's GPU work; ``flush()`` waits for all of them. Falls back to plain zarr-python
    assignment when tensorstore is unavailable, the store is not a local directory, or
    ``CELLMAN_NO_TENSORSTORE`` is set.
    """

    def __init__(self, zg, max_inflight=8):
        self.zg, self.max_inflight = zg, int(max_inflight)
        self.gd = _store_group_dir(zg)
        self.ts = None
        if self.gd is not None and not os.environ.get("CELLMAN_NO_TENSORSTORE"):
            try:
                import tensorstore as ts
                self.ts = ts
            except Exception:  # noqa: BLE001
                self.ts = None
        self.arrays, self.inflight = {}, []

    def __call__(self, name, x0, x1, arr):
        if self.ts is None:
            self.zg[name][x0:x1, ...] = arr
            return
        a = self.arrays.get(name)
        if a is None:
            a = self.ts.open({"driver": "zarr3",
                              "kvstore": {"driver": "file", "path": os.path.join(self.gd, name)}},
                             open=True).result()
            self.arrays[name] = a
        f = a[x0:x1].write(arr)
        f.copy.result()          # the source is copied: the caller may reuse its buffer; the commit stays async
        self.inflight.append(f.commit)
        while len(self.inflight) > self.max_inflight:
            self.inflight.pop(0).result()

    def flush(self):
        for f in self.inflight:
            f.result()
        self.inflight.clear()


def remove_array(zg, name):
    """Delete ``zg[name]`` even when a killed writer left debris in it.

    tensorstore leaves ``*.__lock`` files behind when its process dies mid-write, and on NFS a file
    still open on a lost host lingers as ``.nfs*``; zarr's own delete then stops on the first key it
    cannot remove. Fall back to removing the directory, and if even that leaves something (a
    silly-renamed file), move the array aside so the name is free for the new one.
    """
    import shutil
    import time
    if name not in zg:
        return
    try:
        del zg[name]
        return
    except OSError:
        pass
    gd = _store_group_dir(zg)
    if gd is None:
        raise
    path = os.path.join(gd, name)
    shutil.rmtree(path, ignore_errors=True)
    if os.path.exists(path):
        os.rename(path, f"{path}.stale.{int(time.time())}")


def _store_group_dir(zg):
    """Local directory holding ``zg``'s arrays, or None for a non-directory store."""
    root = getattr(getattr(zg, "store", None), "root", None)
    if root is None:
        return None
    return os.path.join(str(root), zg.path) if zg.path else str(root)


def _tensorstore_open_read(group_dir, name):
    """Open an existing array for READING through tensorstore, or None.

    The writer already routes through tensorstore; the readers did not, and on a big
    count field that is where the whole mapping run goes. zarr-python decodes chunk by
    chunk on one thread (~0.3-0.8 GiB/s on this store); tensorstore's C++ pipeline
    decodes them in parallel (~1.7 GiB/s) and hands back an identical array. Returns
    None -- caller falls back to zarr-python -- when tensorstore is missing, the store
    is not a local directory, or ``CELLMAN_NO_TENSORSTORE`` is set.
    """
    if group_dir is None or os.environ.get("CELLMAN_NO_TENSORSTORE"):
        return None
    try:
        import tensorstore as ts
    except Exception:  # noqa: BLE001
        return None
    try:
        return ts.open({"driver": "zarr3",
                        "kvstore": {"driver": "file",
                                    "path": os.path.join(group_dir, name)}},
                       open=True, create=False).result()
    except Exception:  # noqa: BLE001
        return None


def iter_field_bands(zg, row_bands, *, name="vf_normalized", prefetch=True):
    """Yield ``(r0, r1, block)`` for each requested row band of a 3D count field.

    ``row_bands`` is the list of ``(r0, r1)`` the caller actually wants, so bands with
    no signal are never touched. The NEXT band's read is started before the current one
    is handed over, which hides the read behind the caller's GPU work -- the two are on
    different engines, and the read is otherwise ~60% of a mapping run.

    ``block`` is ``(n_G, r1 - r0, n_C)`` float32, exactly what ``zg[name][:, r0:r1, :]``
    returns; only how it got there changes.
    """
    A = _tensorstore_open_read(_store_group_dir(zg), name)
    z = zg[name]
    if A is not None:
        def start(b):
            r0, r1 = b
            return A[:, r0:r1, :].read()          # future; the C++ pool does the work
        def finish(f):
            return np.asarray(f.result())
    else:
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(1)              # zarr-python releases the GIL in zstd
        def start(b):
            r0, r1 = b
            return pool.submit(lambda: np.asarray(z[:, r0:r1, :]))
        def finish(f):
            return f.result()
    bands = list(row_bands)
    if not bands:
        return
    try:
        pending = start(bands[0])
        for i, (r0, r1) in enumerate(bands):
            nxt = start(bands[i + 1]) if (prefetch and i + 1 < len(bands)) else None
            block = finish(pending)
            pending = nxt
            yield r0, r1, block
            del block
    finally:
        if A is None:
            pool.shutdown(wait=False)


def band_rows_for_budget(n_G, n_C, chunk_h, *, budget_bytes, align=32):
    """Rows per band that keep ONE band under ``budget_bytes`` on the device.

    A band is uploaded whole, so its size -- not the number of signal voxels -- is what
    decides whether the run fits in VRAM. Rounded down to a multiple of the store's row
    chunk so a band still reads whole chunks, and never below one.
    """
    per_row = int(n_G) * int(n_C) * 4
    fit = max(1, int(budget_bytes) // max(per_row, 1))
    fit = min(int(chunk_h), fit)
    return max(align, (fit // align) * align) if fit >= align else fit


def _tensorstore_vf(group_dir, n_G, out_R, out_C, sh, sw):
    """Thin wrapper naming the 3D count ``vf_normalized`` shape and chunking."""
    return _tensorstore_array(group_dir, "vf_normalized",
                              (n_G, out_R, out_C), (n_G, sh, sw))


# ----------------------------------------------------------------------------
# KDE kernels
#
# Every kernel is radial, K(r) = f(r / s) / Z, with its OWN scale ``s``: sigma for
# the gaussian, the support radius H for the compact ones, the decay length for
# the exponential. The user-facing ``bandwidth`` is gaussian-equivalent for all
# of them: ``s`` is chosen so the kernel has the gaussian's per-axis VARIANCE
# (h^2) in the grid's dimensionality, so one bandwidth means one spatial width
# whatever the kernel. The C extension (utils.cpp) and the Triton kernels
# (_kde_gpu.py) index the same table by position in KDE_KERNELS and hold their
# own copy of the scale / norm constants; tests assert they agree with this one.
#
#   kernel        f(u)            scale s / h (dim d)          reach
#   gaussian      exp(-u^2/2)     1                           prune_coefficient * s
#   tophat        1               sqrt(d + 2)                 s
#   epanechnikov  1 - u^2         sqrt(d + 4)                 s
#   exponential   exp(-u)         1 / sqrt(d + 1)             prune_coefficient^2 / 2 * s
#   linear        1 - u           sqrt((d+2)(d+3) / (d+1))    s
#   cosine        cos(pi u / 2)   1 / sqrt(E[u^2] / d)        s
#
# The exponential is cut where it falls to the gaussian's own relative value at
# ``prune_coefficient`` sigma (exp(-4.3^2/2) at the default), so both
# infinite-support kernels drop the same share of their peak.
# ----------------------------------------------------------------------------

KDE_KERNELS = ('gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine')
_COMPACT_KERNELS = frozenset({'tophat', 'epanechnikov', 'linear', 'cosine'})

# cosine: per-axis variance of f(u) = cos(pi u / 2) on the unit ball, E[u^2] / d
_A = np.pi / 2
_COSINE_VAR = {
    1: (1.0 / _A - 2.0 / _A ** 3) / (1.0 / _A),
    2: (1.0 / _A - 6.0 / _A ** 3 + 6.0 / _A ** 4) / (1.0 / _A - 1.0 / _A ** 2) / 2.0,
    3: (1.0 / _A - 12.0 / _A ** 3 + 24.0 / _A ** 5)
       / (1.0 / _A - 2.0 / _A ** 3) / 3.0,
}


def resolve_kernel(kernel):
    """Canonical kernel name from ``kernel`` (case-insensitive).

    Accepts the six names in :data:`KDE_KERNELS` and, for stores and callers that
    predate them, any unambiguous prefix (``'gauss'``, ``'epa'``). Raises
    ``ValueError`` for anything else, so a typo cannot fall through to a default.
    """
    k = str(kernel).strip().lower()
    if k in KDE_KERNELS:
        return k
    hits = [name for name in KDE_KERNELS if name.startswith(k)] if len(k) >= 3 else []
    if len(hits) == 1:
        return hits[0]
    raise ValueError(f"unknown kernel {kernel!r}; use one of {', '.join(KDE_KERNELS)}")


def kernel_code(kernel):
    """Integer id of ``kernel`` for the compiled and Triton KDEs (its index in KDE_KERNELS)."""
    return KDE_KERNELS.index(resolve_kernel(kernel))


def kernel_scale(bandwidth, ndim, kernel='gaussian'):
    """Gaussian-equivalent ``bandwidth`` -> the kernel's own scale, matched on per-axis variance.

    ``ndim`` is the grid's dimensionality (2 or 3; 1 is accepted for completeness):
    the same width needs a different scale in 3D, since a radial kernel's variance
    per axis depends on how much of its mass sits at large r.
    """
    k = resolve_kernel(kernel); d = int(ndim); h = float(bandwidth)
    if k == 'gaussian':
        return h
    if k == 'tophat':
        return h * float(np.sqrt(d + 2))
    if k == 'epanechnikov':
        return h * float(np.sqrt(d + 4))
    if k == 'exponential':
        return h / float(np.sqrt(d + 1))
    if k == 'linear':
        return h * float(np.sqrt((d + 2) * (d + 3) / (d + 1)))
    return h / float(np.sqrt(_COSINE_VAR[d]))


def kernel_reach(scale, kernel='gaussian', prune_coefficient=4.3):
    """Truncation radius of ``kernel`` at its own ``scale``, in the same units.

    The compact kernels reach zero on their own at ``scale``, so
    ``prune_coefficient`` applies only to the gaussian and the exponential.
    """
    k = resolve_kernel(kernel)
    if k in _COMPACT_KERNELS:
        return float(scale)
    if k == 'exponential':
        return float(scale) * float(prune_coefficient) ** 2 / 2.0
    return float(scale) * float(prune_coefficient)


def kernel_search_cells(reach, sampling_distance, kernel='gaussian'):
    """Neighbour-search radius, in grid cells, of the cell-list KDEs (C and Triton alike).

    Query and molecule are each binned by ``floor``, so a molecule within ``reach``
    can sit ``floor(reach / sd) + 1`` cells away. The gaussian and exponential keep
    the historical ``floor(reach / sd)``: what that misses is their tail below ~1e-4
    of the peak, and the gaussian's fields stay what they have always been. A
    compact kernel still carries real weight at its edge (all of it for the
    tophat), so it gets the extra cell and its whole support.
    """
    extra = 1 if resolve_kernel(kernel) in _COMPACT_KERNELS else 0
    return int(np.floor(float(reach) / float(sampling_distance) + 1e-12)) + extra


def kernel_norm(scale, ndim, kernel='gaussian'):
    """Integral of the unnormalized profile ``f(r / scale)`` over R^ndim (so ``K = f / norm``).

    Closed forms; the gaussian and exponential integrals are the untruncated ones,
    as the compiled KDE has always used for the gaussian.
    """
    k = resolve_kernel(kernel); d = int(ndim); s = float(scale)
    unit_ball = {1: 2.0, 2: np.pi, 3: 4.0 * np.pi / 3.0}[d]
    if k == 'gaussian':
        return float((np.sqrt(2.0 * np.pi) * s) ** d)
    if k == 'tophat':
        return float(unit_ball * s ** d)
    if k == 'epanechnikov':
        return float(unit_ball * s ** d * 2.0 / (d + 2))
    if k == 'linear':
        return float(unit_ball * s ** d / (d + 1))
    if k == 'exponential':
        return float({1: 2.0, 2: 2.0 * np.pi, 3: 8.0 * np.pi}[d] * s ** d)
    return float({1: 4.0 / np.pi,
                  2: 4.0 - 8.0 / np.pi,
                  3: 8.0 - 64.0 / np.pi ** 2}[d] * s ** d)


def kernel_profile(u, kernel='gaussian'):
    """Unnormalized radial profile ``f(u)`` at ``u = r / scale`` (numpy; zero outside the support)."""
    k = resolve_kernel(kernel)
    u = np.asarray(u, dtype=np.float64)
    if k == 'gaussian':
        return np.exp(-0.5 * u * u)
    if k == 'exponential':
        return np.exp(-u)
    inside = u <= 1.0
    if k == 'tophat':
        f = np.ones_like(u)
    elif k == 'epanechnikov':
        f = 1.0 - u * u
    elif k == 'linear':
        f = 1.0 - u
    else:
        f = np.cos(_A * u)
    return np.where(inside, f, 0.0)


def epanechnikov_support(bandwidth, ndim):
    """Gaussian bandwidth -> Epanechnikov support radius, matched on VARIANCE.

    The spherical Epanechnikov ``K(r) ∝ 1 - (r/H)^2`` on the ball of radius ``H`` has
    per-axis variance ``H^2 / (ndim + 4)``; a gaussian of bandwidth ``h`` has ``h^2``.
    Equating them gives

        H = h * sqrt(ndim + 4)          (2.236 h in 1D, 2.449 h in 2D, 2.646 h in 3D)

    Kept for callers that predate :func:`kernel_scale`, which generalizes it.
    """
    return kernel_scale(bandwidth, ndim, 'epanechnikov')


def kernel_peak(scale, ndim, kernel='gaussian'):
    """Peak of ONE molecule's kernel = ``1 / V_eff``, the pipeline's ``expression_threshold``.

    ``scale`` is the kernel's OWN scale in the units ``vf_norm`` is measured in
    (gaussian sigma, the support radius of a compact kernel, the exponential's decay
    length -- convert with :func:`kernel_scale`, never pass the gaussian bandwidth here).
    Every profile is 1 at the origin, so the peak is ``1 / kernel_norm``:

        gaussian       1 / (sqrt(2*pi) * sigma) ** ndim
        epanechnikov   3/(4H) [1D]   2/(pi H^2) [2D]   15/(8 pi H^3) [3D]

    Everything downstream is anchored on this one number: ``norm_threshold`` is
    2x it (a voxel holding less than two molecules' worth of evidence), and the
    LDA count reconstruction multiplies the density back up by ``V_eff = 1/peak``.
    Both come out wrong -- silently, by a factor of ~2 at the default bandwidth --
    if the gaussian formula is applied to another kernel's field.
    """
    return float(1.0 / kernel_norm(scale, ndim, kernel))


def compute_vf_blur(analysis, bin_adata, *, sigma_bin: float = 1.0,
                    chunk_h: int = 64, chunk_w: int = 512, bin_size_um: float = 2.0,
                    sampling_distance: float = None, prune_coefficient: float = 4.3,
                    kernel: str = "gaussian",
                    size_after_normalization: float = 10.0,
                    store_chunk=32,
                    device: str = "cuda") -> None:
    """Rasterize ``bin_adata`` counts and GPU-blur into ``vf_normalized`` (zarr).

    Two independent resolutions:
      * ``bin_size_um`` — the **blur raster** grid the Gaussian is computed on
        (finer = more accurate blur; σ in bins = ``sigma_bin``, bandwidth =
        ``sigma_bin * bin_size_um`` µm).
      * ``sampling_distance`` — the **output vector-field** grid spacing. The
        blurred field is point-sampled (decimated) by ``stride =
        round(sampling_distance / bin_size_um)``, giving a coarser vf than the raster
        (fewer voxels on disk) while the blur itself stays raster-accurate.
        Defaults to ``bin_size_um`` (stride 1 = output == raster).

    Writes ``vf_normalized`` ``(n_G, out_R, out_C)`` and ``vf_norm``
    ``(out_R, out_C)`` at ``sampling_distance`` spacing (+ metadata,
    ``backend='blur'``). Only the normalized field is persisted (never the raw 4D vf).

    ``store_chunk`` decouples the on-disk chunk shape of ``vf_normalized`` from the
    GPU compute tile (``chunk_h`` × ``chunk_w``). The compute tile stays large for
    GPU efficiency; the store is written in small SQUARE spatial tiles (gene axis
    always full) so a sparse voxel gather (``select_vectors_from_image`` /
    ``downsample_vector``) decodes only the tiles that actually contain a sampled
    voxel instead of whole full-width row bands. Give an int (square side) or a
    ``(h, w)`` tuple; ``None`` makes the on-disk chunk match the compute tile. For
    alignment (no read-modify-write on interior writes) ``chunk_h`` and ``chunk_w``
    should be integer multiples of the store tile (the defaults 64/512 vs 32 are).

    :param analysis: the ``Analysis`` instance (provides ``dataset`` and ``verbose``).
    :type analysis: Analysis
    :param bin_adata: bin-count AnnData with ``obs['array_row']`` / ``obs['array_col']``.
    :type bin_adata: AnnData
    :param sigma_bin: Gaussian sigma in raster bins.
    :type sigma_bin: float
    :param chunk_h: GPU compute tile height in output rows.
    :type chunk_h: int
    :param chunk_w: GPU compute tile width in output columns.
    :type chunk_w: int
    :param bin_size_um: blur raster grid spacing in microns.
    :type bin_size_um: float
    :param sampling_distance: output vector-field grid spacing in microns; defaults to ``bin_size_um``.
    :type sampling_distance: float
    :param prune_coefficient: truncation radius of the infinite-support kernels in
        units of sigma (gaussian; the exponential is cut at the same relative height).
    :type prune_coefficient: float
    :param kernel: one of :data:`KDE_KERNELS` -- ``'gaussian'`` (default),
        ``'tophat'``, ``'epanechnikov'``, ``'exponential'``, ``'linear'``,
        ``'cosine'``. ``sigma_bin`` keeps the same meaning for all of them: each
        kernel's own scale is derived from it by variance matching
        (:func:`kernel_scale`, e.g. the Epanechnikov support ``H = sigma_bin *
        sqrt(6)`` bins), so they smooth over the same width. Only the gaussian is
        separable; the others run as one dense conv, whose cost grows with the
        SQUARE of the radius while the gaussian's separable pair grows linearly.
        Below ``sigma_bin`` about 6 a compact kernel's smaller radius still wins
        (~0.8x at 1-4 for the Epanechnikov); above it the gaussian pulls ahead.
    :type kernel: str
    :param size_after_normalization: L1 scale applied before ``log1p``.
    :type size_after_normalization: float
    :param store_chunk: on-disk spatial chunk (int square side, ``(h, w)`` tuple, or ``None`` to match the compute tile).
    :type store_chunk: int or tuple or None
    :param device: torch device string.
    :type device: str
    """
    import torch
    zg = analysis.dataset.zarr_group
    ds = analysis.dataset

    R = bin_adata.obs["array_row"].to_numpy()
    C = bin_adata.obs["array_col"].to_numpy()
    n_R, n_C = int(R.max() + 1), int(C.max() + 1)
    n_G = bin_adata.n_vars
    X = bin_adata.X.tocsr() if issparse(bin_adata.X) else bin_adata.X

    if sampling_distance is None:
        sampling_distance = bin_size_um
    stride = max(1, int(round(float(sampling_distance) / float(bin_size_um))))
    sd_um = float(bin_size_um) * stride
    bw_um = float(sigma_bin * bin_size_um)
    out_R = (n_R + stride - 1) // stride
    out_C = (n_C + stride - 1) // stride
    kernel = resolve_kernel(kernel)
    if kernel == 'gaussian':
        kx, ky, radius = _gaussian_kernel_1d(sigma_bin, device, prune_coefficient)
    else:
        kx, radius = _radial_kernel_2d(sigma_bin, kernel, device, prune_coefficient)
        ky = None
    # the field's own smoothing scale, in the units vf_norm lives in (bins -> µm)
    kde_scale_bin = kernel_scale(sigma_bin, 2, kernel)
    blur_dtype = torch.float32
    chunk_w = max(1, int(chunk_w))
    row_order = np.argsort(R, kind="stable")
    R_s = R[row_order]; C_s = C[row_order]

    for k in ("vf_normalized", "vf_norm", "genes", "kde_computed", "vf_params",
              "coordinate_offset", "coordinate_bounds", "original_coordinate_bounds"):
        if k in zg:
            del zg[k]
    if store_chunk is None:
        sh_store, sw_store = chunk_h, min(chunk_w, out_C)
    elif isinstance(store_chunk, (tuple, list)):
        sh_store, sw_store = int(store_chunk[0]), int(store_chunk[1])
    else:
        sh_store = sw_store = int(store_chunk)
    sh_store = max(1, min(sh_store, out_R))
    sw_store = max(1, min(sw_store, out_C))
    _store_root = getattr(zg.store, "root", None)
    _group_dir = (os.path.join(str(_store_root), zg.path) if zg.path else str(_store_root)) \
        if _store_root is not None else None
    vf_ts = _tensorstore_vf(_group_dir, n_G, out_R, out_C, sh_store, sw_store) \
        if _group_dir is not None else None
    vf_z = None
    if vf_ts is None:
        vf_z = zg.create_array("vf_normalized", shape=(n_G, out_R, out_C),
                               chunks=(n_G, sh_store, sw_store), dtype="f4", fill_value=0.0)
    _vf_inflight = []
    col_chunk = min(chunk_w, out_C)
    norm_z = zg.create_array("vf_norm", shape=(out_R, out_C),
                             chunks=(chunk_h, col_chunk), dtype="f4", fill_value=0.0)
    coord_bnd = np.array([0.0, 0.0, 0.0, out_C * sd_um, out_R * sd_um, 0.0], dtype="f8")
    zg["genes"] = np.asarray(list(bin_adata.var_names))
    zg["kde_computed"] = np.zeros(n_G, dtype=bool)
    zg["vf_params"] = np.array([sd_um, kde_scale_bin * float(bin_size_um)], dtype="f8")
    zg["coordinate_offset"] = np.zeros(3, "f8")
    zg["coordinate_bounds"] = coord_bnd
    zg["original_coordinate_bounds"] = coord_bnd.copy()
    zg.attrs["sigma_bin"] = float(sigma_bin)
    zg.attrs["bin_size_um"] = float(bin_size_um)
    zg.attrs["sampling_distance"] = sd_um
    zg.attrs["stride"] = int(stride)
    zg.attrs["prune_coefficient"] = float(prune_coefficient)
    zg.attrs["kde_kernel"] = kernel
    zg.attrs["size_after_normalization"] = float(size_after_normalization)
    zg.attrs["backend"] = "blur"

    ds.genes = list(bin_adata.var_names)
    ds.coordinate_offset = [0.0, 0.0, 0.0]
    ds.coordinate_bounds = coord_bnd.tolist()
    ds.original_coordinate_bounds = coord_bnd.tolist()
    ds.sampling_distance = sd_um
    ds.shape = (out_C, out_R, 1)

    # Same SSAM quantity the transcript backend sets, in THIS backend's units.
    # The blur is a discrete convolution on the bin raster with a kernel that sums
    # to 1 over bins, so vf_norm is molecules per bin^2, not per µm^2 (verified:
    # summing vf_norm over the field reproduces the panel's total counts to 4
    # significant figures). One molecule's peak is therefore
    # 1 / (sqrt(2*pi) * sigma_bin)^2 with sigma in BINS -- a factor
    # bin_size_um^2 above the µm-space value the transcript formula gives (4x at
    # 2 µm bins). Setting it here keeps ``ds.norm_threshold`` meaning the same
    # thing on both backends; the mappers derive it independently via
    # :func:`analytic_vf_norm_threshold`, so the two always agree.
    ds.ndim = 2
    ds.bandwidth = kde_scale_bin * float(bin_size_um)
    ds.expression_threshold = kernel_peak(kde_scale_bin, 2, kernel)
    ds.norm_threshold = ds.expression_threshold * 2

    kx_g = kx.to(blur_dtype).expand(n_G, 1, *kx.shape[2:]).contiguous()
    ky_g = None if ky is None else ky.to(blur_dtype).expand(n_G, 1, -1, 1).contiguous()
    _cuda = str(device).startswith("cuda")

    o_starts = list(range(0, out_R, chunk_h))
    if analysis.verbose:
        o_starts = tqdm(o_starts, desc="compute_vf (blur, tiled)")
    for o0 in o_starts:
        o1 = min(out_R, o0 + chunk_h)
        frows = np.arange(o0, o1) * stride
        frows = frows[frows < n_R]
        if len(frows) == 0:
            continue
        rp0 = max(0, int(frows[0]) - radius)
        rp1 = min(n_R, int(frows[-1]) + 1 + radius)
        lo = int(np.searchsorted(R_s, rp0, side="left"))
        hi = int(np.searchsorted(R_s, rp1, side="left"))
        if hi <= lo:
            continue
        pos = row_order[lo:hi]
        Rb = R_s[lo:hi]; Cb = C_s[lo:hi]; Xb = X[pos]
        col_order = np.argsort(Cb, kind="stable")
        Cb_c = Cb[col_order]; Rb_c = Rb[col_order]; Xb_c = Xb[col_order]
        lr = torch.as_tensor(frows - rp0, device=device, dtype=torch.long)
        for oc0 in range(0, out_C, chunk_w):
            oc1 = min(out_C, oc0 + chunk_w)
            ocols = np.arange(oc0, oc1) * stride
            ocols = ocols[ocols < n_C]
            if len(ocols) == 0:
                continue
            cp0 = max(0, int(ocols[0]) - radius)
            cp1 = min(n_C, int(ocols[-1]) + 1 + radius)
            clo = int(np.searchsorted(Cb_c, cp0, side="left"))
            chi = int(np.searchsorted(Cb_c, cp1, side="left"))
            if chi <= clo:
                continue
            Xt = Xb_c[clo:chi]; Rt = Rb_c[clo:chi]; Ct = Cb_c[clo:chi]
            blurred = _rasterize_blur_tile_gpu(Xt, Rt, Ct, rp0, rp1, cp0, cp1, n_G,
                                               kx_g, ky_g, radius, device, blur_dtype)
            lc = torch.as_tensor(ocols - cp0, device=device, dtype=torch.long)
            chunk = blurred.index_select(1, lr).index_select(2, lc).float().contiguous()
            normalized, vfn = _l1_normalize_log1p(chunk, size_after_normalization)
            nn = normalized.cpu().numpy()
            if vf_ts is not None:
                _vf_inflight.append(
                    vf_ts[:, o0:o0 + len(frows), oc0:oc0 + len(ocols)].write(nn))
                while len(_vf_inflight) > 8:
                    _vf_inflight.pop(0).result()
            else:
                vf_z[:, o0:o0 + len(frows), oc0:oc0 + len(ocols)] = nn
            norm_z[o0:o0 + len(frows), oc0:oc0 + len(ocols)] = vfn.cpu().numpy()
            del blurred, chunk, normalized
    for _f in _vf_inflight:
        _f.result()
    if _cuda:
        torch.cuda.empty_cache()
    zg["kde_computed"][:] = True
    if analysis.verbose:
        print(f"[compute_vf] blur raster {(n_R, n_C)}@{bin_size_um}µm → vf {(n_G, out_R, out_C)}@{sd_um}µm "
              f"(stride {stride}, compute tile {chunk_h}×{chunk_w}, store chunk {sh_store}×{sw_store}, "
              f"radius {radius}, {kernel} scale {kde_scale_bin * bin_size_um:.4g}µm "
              f"(= bandwidth {bw_um}µm), fp32)")


def _image_gray(image, revert):
    arr = np.asarray(image)
    if arr.ndim == 3:
        if arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
            arr = np.moveaxis(arr, 0, -1)
        arr = arr[..., :3]
        gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    else:
        gray = arr.astype(np.float64)
    return (255.0 - gray) if revert else gray.astype(np.float64)


def select_vectors_from_image(analysis, grid, *, bin_size_um: float,
                              n_samples_per_block: int = 3, block_um: float = 10.0,
                              otsu_threshold: Optional[float] = None,
                              otsu_mult: Optional[float] = None,
                              norm_threshold_mult: float = 2.0,
                              vf_norm_percentile: Optional[float] = 25.0,
                              lookup_chunk_h: int = 64, seed: int = 0,
                              lookup_n_workers: Optional[int] = None,
                              lookup_ram_gb: float = 32.0) -> np.ndarray:
    """Sample voxels from the tissue image (Otsu mask) and look them up in the
    precomputed ``vf_normalized`` field into ``normalized_vectors`` (count downsample).

    Pixels are mapped to world µm via ``grid.affine_px_to_um`` (folds each
    platform's transpose/flip), then to grid bins via ``grid.origin`` + ``bin_size_um``.

    :param analysis: the ``Analysis`` instance (provides ``dataset`` and ``verbose``).
    :type analysis: Analysis
    :param grid: raster/image grid with ``image``, ``origin``, ``um_per_px``, ``affine_px_to_um``, ``he_revert``.
    :type grid: object
    :param bin_size_um: grid bin spacing in microns.
    :type bin_size_um: float
    :param n_samples_per_block: pixels sampled per ``block_um`` square.
    :type n_samples_per_block: int
    :param block_um: sampling block size in microns.
    :type block_um: float
    :param otsu_threshold: fixed tissue-mask threshold; computed by Otsu when ``None``.
    :type otsu_threshold: float or None
    :param otsu_mult: scale the Otsu value ssaft computes by this factor (``0.6`` lowers the
        threshold and widens the mask). Only with ``otsu_threshold=None`` -- a threshold that
        was given is used as given.
    :type otsu_mult: float or None
    :param norm_threshold_mult: multiplier for the analytic ``vf_norm`` threshold when no percentile is used.
    :type norm_threshold_mult: float
    :param vf_norm_percentile: percentile of nonzero ``vf_norm`` used as the keep threshold; ``None`` uses the analytic form.
    :type vf_norm_percentile: float or None
    :param lookup_chunk_h: retained for API compatibility; no longer sizes the read.
    :type lookup_chunk_h: int
    :param seed: RNG seed for pixel sampling.
    :type seed: int
    :param lookup_n_workers: worker count for the parallel tile lookup; defaults to ``analysis.ncores``.
    :type lookup_n_workers: int or None
    :param lookup_ram_gb: RAM budget bounding concurrent tile decodes.
    :type lookup_ram_gb: float
    """
    from skimage.filters import threshold_otsu
    zg = analysis.dataset.zarr_group
    ds = analysis.dataset
    vf_z, norm_z = zg["vf_normalized"], zg["vf_norm"]
    n_G, n_R, n_C = vf_z.shape
    sigma_bin = float(zg.attrs.get("sigma_bin", 1.0))
    X0, Y0 = grid.origin

    he = _image_gray(grid.image, grid.he_revert)
    upp = grid.um_per_px
    if otsu_threshold is None:
        otsu_threshold = float(threshold_otsu(he))
        if otsu_mult is not None:
            otsu_threshold *= float(otsu_mult)
    elif otsu_mult is not None:
        raise ValueError('otsu_mult scales the Otsu threshold ssaft computes; with '
                         'otsu_threshold given there is none to scale — pass one or the other')
    mask = he > otsu_threshold
    ys, xs = np.where(mask)
    n_mask = len(ys)
    block_px = (block_um / upp) ** 2
    n_sel = max(int(n_mask / block_px * n_samples_per_block), 1)
    rng = np.random.default_rng(seed)
    pick = rng.choice(n_mask, min(n_sel, n_mask), replace=False)
    px_x = xs[pick].astype(np.float64); px_y = ys[pick].astype(np.float64)

    M = grid.affine_px_to_um
    if M is not None:
        M = np.asarray(M)
        x_um = M[0, 0] * px_x + M[0, 1] * px_y + M[0, 2]
        y_um = M[1, 0] * px_x + M[1, 1] * px_y + M[1, 2]
    else:
        x_um = px_x * upp; y_um = px_y * upp
    C_i = np.floor((x_um - X0) / bin_size_um).astype(np.int64)
    R_i = np.floor((y_um - Y0) / bin_size_um).astype(np.int64)

    norm_full = norm_z[:]
    if vf_norm_percentile is not None:
        nz = norm_full[norm_full > 0]
        thr = float(np.percentile(nz, vf_norm_percentile)) if nz.size else 0.0
    else:
        thr = norm_threshold_mult / (np.sqrt(2 * np.pi) * sigma_bin) ** 2
    in_b = (R_i >= 0) & (R_i < n_R) & (C_i >= 0) & (C_i < n_C)
    keep = in_b.copy()
    keep[in_b] &= norm_full[R_i[in_b], C_i[in_b]] > thr
    R_sel, C_sel = R_i[keep], C_i[keep]
    order = np.argsort(R_sel)
    R_sel, C_sel = R_sel[order], C_sel[order]

    from concurrent.futures import ThreadPoolExecutor
    nv = np.zeros((len(R_sel), n_G), dtype=np.float32)
    tile_h = int(vf_z.chunks[1]); tile_w = int(vf_z.chunks[2])
    n_col_tiles = (n_C + tile_w - 1) // tile_w
    tr = R_sel // tile_h
    tc = C_sel // tile_w
    tile_key = tr.astype(np.int64) * n_col_tiles + tc
    torder = np.argsort(tile_key, kind="stable")
    key_o = tile_key[torder]
    uniq = np.unique(key_o)
    starts = np.searchsorted(key_o, uniq)
    ends = np.r_[starts[1:], len(key_o)]

    tile_bytes = n_G * tile_h * tile_w * 4
    ram_cap = max(1, int(float(lookup_ram_gb) * (1024 ** 3) / max(1, tile_bytes)))
    n_workers = int(lookup_n_workers if lookup_n_workers is not None
                    else getattr(analysis, "ncores", 1) or 1)
    n_workers = max(1, min(n_workers, len(uniq), ram_cap))

    def _lookup_tile(i):
        s, e = int(starts[i]), int(ends[i])
        key = int(uniq[i])
        rt, ct = key // n_col_tiles, key % n_col_tiles
        r0, r1 = rt * tile_h, min(rt * tile_h + tile_h, n_R)
        c0, c1 = ct * tile_w, min(ct * tile_w + tile_w, n_C)
        block = np.asarray(vf_z[:, r0:r1, c0:c1])
        idx = torder[s:e]
        nv[idx] = block[:, R_sel[idx] - r0, C_sel[idx] - c0].T
        return e - s

    try:
        import numcodecs
        _prev_blosc = numcodecs.blosc.get_nthreads()
        numcodecs.blosc.set_nthreads(1)
    except Exception:  # noqa: BLE001
        numcodecs, _prev_blosc = None, None
    try:
        if n_workers > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                it = ex.map(_lookup_tile, range(len(uniq)))
                if analysis.verbose:
                    it = tqdm(it, total=len(uniq),
                              desc=f"downsample_vector (lookup, {len(uniq)} tiles, {n_workers}w)")
                for _ in it:
                    pass
        else:
            rng_it = range(len(uniq))
            it = tqdm(rng_it, desc="downsample_vector (lookup)") if analysis.verbose else rng_it
            for i in it:
                _lookup_tile(i)
    finally:
        if numcodecs is not None and _prev_blosc is not None:
            numcodecs.blosc.set_nthreads(_prev_blosc)

    gx = C_sel.astype(np.int64)
    gy = R_sel.astype(np.int64)
    gz = np.zeros(len(C_sel), np.int64)
    dvn = norm_full[R_sel, C_sel].astype(np.float32)
    gpx = (X0 + (C_sel + 0.5) * bin_size_um).astype(np.float64)
    gpy = (Y0 + (R_sel + 0.5) * bin_size_um).astype(np.float64)
    for k in ("normalized_vectors", "downsampled_vectors",
              "downsampled_vf_norm", "downsampled_global_xy"):
        if k in zg:
            del zg[k]
    zg["normalized_vectors"] = nv
    zg["downsampled_vectors"] = np.stack([gx, gy, gz], axis=0)
    zg["downsampled_vf_norm"] = dvn
    zg["downsampled_global_xy"] = np.stack([gpx, gpy], axis=0)
    ds.normalized_vectors = nv
    ds.downsampled_vectors = (gx, gy, gz)
    ds.downsampled_vf_norm = dvn
    ds.downsampled_global_xy = np.stack([gpx, gpy], axis=0)
    zg.attrs["select_n_kept"] = int(len(R_sel))
    zg.attrs["select_otsu_threshold"] = float(otsu_threshold)
    zg.attrs["select_he_um_per_px"] = float(upp)
    zg.attrs["select_vf_norm_threshold"] = float(thr)
    if analysis.verbose:
        print(f"[downsample] otsu={otsu_threshold:.4g} mask_px={n_mask} "
              f"sampled={n_sel} kept(vf_norm>{thr:.4g})={len(R_sel)}")
    return nv


#: mapping method -> the zarr/dataset name its per-voxel score is stored under.
#: ``corr`` yields a Pearson correlation (filtered by ``min_r``), ``gnb`` a
#: top-class posterior probability (filtered by ``min_p``).
SCORE_KEYS = {"corr": "max_correlations", "gnb": "max_probabilities",
              "lda": "max_probabilities"}


def score_key_for(method):
    """``'corr'|'gnb'|'lda'`` -> the score volume name that method writes.

    ``gnb`` and ``lda`` share ``max_probabilities``: both produce a posterior, so both
    are thresholded with ``min_p``. Which of the two wrote the live map is recorded
    separately in ``zg.attrs['mapping_method']``.
    """
    try:
        return SCORE_KEYS[method]
    except KeyError:
        raise ValueError(f"unknown mapping method {method!r} "
                         "(expected 'corr', 'gnb' or 'lda')")


def _other_score_key(score_key):
    """The score volume name NOT written by the method that writes ``score_key``."""
    return ("max_probabilities" if score_key == "max_correlations"
            else "max_correlations")


def _write_maps(zg, ds, M, Rc, sd, *, overwrite=True, set_ds_attrs=True,
                score_key="max_correlations", method=None):
    """Write the label raster + its per-voxel score under ``score_key``.

    ``score_key`` names the metric actually produced: ``max_correlations`` for the
    centroid correlation mapper, ``max_probabilities`` for the GaussianNB posterior.
    Both volumes may coexist — re-running one method no longer throws away the other
    method's result. ``zg.attrs['mapping_method']`` records which one is live, and
    ``filter_celltypemaps`` reads it to pick its threshold (``min_p`` vs ``min_r``).
    """
    cm_3d = M.T[:, :, None].astype(np.int32)
    mc_3d = Rc.T[:, :, None].astype(np.float32)
    vx, vy, vz = cm_3d.shape
    if overwrite:
        for nm in ("celltype_maps", score_key):
            if nm in zg:
                del zg[nm]
    # 'gnb' and 'lda' share the posterior volume, so score_key alone cannot name the
    # method; a caller that is neither corr nor gnb must say so explicitly.
    zg.attrs['mapping_method'] = method or (
        'gnb' if score_key == 'max_probabilities' else 'corr')
    zg["celltype_maps"] = cm_3d
    zg[score_key] = mc_3d
    if set_ds_attrs:
        ds.shape = (vx, vy, vz)
        ds.x_scale = 1.0
        ds.y_scale = 1.0
        ds.original_coordinate_bounds = [[0.0, vx * sd], [0.0, vy * sd], [0.0, 0.0]]
        ds.celltype_maps = cm_3d
        setattr(ds, score_key, mc_3d)
        # the other method's volume stays on disk; reload it so both are addressable
        _other = _other_score_key(score_key)
        setattr(ds, _other, zg[_other][:] if _other in zg else None)
    return cm_3d, mc_3d


def analytic_vf_norm_threshold(zg, mult: float = 2.0) -> float:
    """SSAM's ``vf_norm`` floor: ``mult`` x the KDE peak of a single molecule.

    This is the threshold every other part of ssaft uses
    (``Analysis.dataset.norm_threshold = expression_threshold * 2``); the mappers
    must use it too. It is an absolute quantity — it says "this voxel holds less
    than two molecules' worth of evidence" — so it means the same thing on every
    dataset, unlike a percentile, which always discards a fixed *share* of the
    tissue no matter how good the signal is.

    Units follow the field the backend actually stored, so sigma must be given in
    the same units as ``vf_norm``'s support:

    * count backend (Visium HD / Stereo-seq): ``vf_norm`` is per bin^2 and the
      blur is ``sigma_bin`` **bins** -> ``mult / (sqrt(2*pi) * sigma_bin)^2``
    * transcript backend: ``vf_norm`` is per µm^ndim and the blur is the
      bandwidth in **µm** -> ``mult / (sqrt(2*pi) * bandwidth)^ndim``

    Any other ``attrs['kde_kernel']`` swaps in that kernel's peak
    (:func:`kernel_peak`) instead; the stored scale is already the kernel's own.

    Mixing the two is a factor of ``bin_size_um^ndim`` (4x at 2 µm bins in 2D).

    Returns ``0.0`` when the store records neither, i.e. keep every voxel the
    field reached — never a percentile, which would silently cut real signal.
    """
    a = dict(zg.attrs)
    kern = a.get('kde_kernel', 'gaussian')
    sigma_bin = a.get('sigma_bin')
    if sigma_bin:
        return float(mult * kernel_peak(kernel_scale(sigma_bin, 2, kern), 2, kern))
    bandwidth = a.get('bandwidth', a.get('bandwidth_um'))
    if bandwidth is None and 'vf_params' in zg:
        vp = np.asarray(zg['vf_params'][:]).ravel()
        bandwidth = float(vp[1]) if vp.size > 1 else None
    if bandwidth:
        # vf_params[1] / attrs['bandwidth'] store the kernel's OWN scale (H for a
        # compact kernel), so no conversion here -- only the peak formula changes.
        ndim = int(a.get('ndim', 2))
        return float(mult * kernel_peak(bandwidth, ndim, kern))
    return 0.0


def resolve_vf_norm_threshold(zg, vf_norm_threshold=None, mult: float = 2.0) -> float:
    """Threshold a mapper should use: explicit value, else the analytic one.

    Deliberately does NOT fall back to ``attrs['select_vf_norm_threshold']``.
    That key records the cut ``downsample_vector`` used to *pick training
    voxels* — with ``vf_norm_percentile`` it is a percentile of the field, which
    is the right call for sampling (take the denser part of the tissue to learn
    from) and the wrong one for mapping (label every voxel that carries signal).
    Reusing it made the map inherit a 25% cut of the tissue that no caller asked
    for.
    """
    if vf_norm_threshold is not None:
        return float(vf_norm_threshold)
    return analytic_vf_norm_threshold(zg, mult=mult)


def map_corr(analysis, centroids=None, *, vf_norm_threshold=None, chunk_h=128,
             device="cuda", overwrite=True, gene_indices=None, marker_index_sets=None,
             gpu_target_bytes=int(4e9), sub_batch=None):
    """Map the blur field to celltypes by centroid Pearson correlation (no KDE).

    Reads ``vf_normalized`` chunk-by-chunk, mean-centers + L2-normalizes each
    voxel and each centroid, assigns the max-correlation cluster. Output layout
    matches the transcript KDE path so downstream filter/plot work unchanged.

    :param analysis: the ``Analysis`` instance (provides ``dataset`` and ``verbose``).
    :type analysis: Analysis
    :param centroids: cluster centroids ``(k, n_G)``; read from the zarr group when ``None``.
    :type centroids: numpy.ndarray or None
    :param vf_norm_threshold: minimum ``vf_norm`` for a voxel to be mapped; auto-derived when ``None``.
    :type vf_norm_threshold: float or None
    :param chunk_h: number of raster rows processed per GPU chunk.
    :type chunk_h: int
    :param device: torch device string.
    :type device: str
    :param overwrite: replace any existing ``celltype_maps`` / ``max_correlations``.
    :type overwrite: bool
    :param gene_indices: positions into the store's gene axis to keep; ``None`` = all.
        Set by ``map_celltypes(marker_genes_path=...)`` to the union of the marker
        panel, so the correlation stops spending its variance on genes no celltype
        was called on.
    :type gene_indices: list[int] or None
    :param marker_index_sets: per-celltype gene positions *within* ``gene_indices``
        (see :func:`ssaft._marker_corr.marker_index_sets`). When given, each celltype
        is correlated only on its own markers instead of the shared axis.
    :type marker_index_sets: list or None
    """
    import torch
    zg = analysis.dataset.zarr_group
    ds = analysis.dataset
    vf_z = zg["vf_normalized"]
    n_G, n_R, n_C = vf_z.shape
    norm = np.asarray(zg["vf_norm"][:])
    sd = float(getattr(ds, "sampling_distance", zg.attrs.get("bin_size_um", 1.0)))

    if centroids is None:
        if "centroids" not in zg:
            raise ValueError("centroids missing — pass h5ad=<annotated .h5ad> to map_celltypes to build them.")
        centroids = zg["centroids"][:]
    centroids = np.asarray(centroids, dtype=np.float32)
    if centroids.shape[1] != n_G:
        raise ValueError(f"centroids gene dim {centroids.shape[1]} != vf {n_G}")
    if gene_indices is not None:
        gene_indices = np.asarray(sorted(gene_indices), dtype=np.int64)
        centroids = centroids[:, gene_indices]

    vf_norm_threshold = resolve_vf_norm_threshold(zg, vf_norm_threshold)

    c = torch.as_tensor(centroids, dtype=torch.float32, device=device)
    if marker_index_sets is None:      # one shared axis -> pre-normalize once
        c = c - c.mean(dim=1, keepdim=True)
        c = c / torch.clamp(torch.linalg.norm(c, dim=1, keepdim=True), min=1e-12)

    M = np.full((n_R, n_C), -1, dtype=np.int32)
    Rc = np.full((n_R, n_C), -1.0, dtype=np.float32)

    # Same shape of fix as map_lda: the band is read through tensorstore with the next
    # one prefetched, uploaded exactly as it sits on disk (gene-major), and the
    # transpose + signal gather happen on the device. The host used to do both, and on
    # a wide gene axis that cost more than everything else put together.
    chunk_h = band_rows_for_budget(n_G, n_C, chunk_h, budget_bytes=gpu_target_bytes)
    if sub_batch is None:
        sub_batch = max(4096, int(gpu_target_bytes // 16) // max(n_G, 1))
    gi_t = (torch.as_tensor(np.asarray(gene_indices), device=device)
            if gene_indices is not None else None)
    bands = [(a, min(a + chunk_h, n_R)) for a in range(0, n_R, chunk_h)]
    bands = [(a, b) for a, b in bands if (norm[a:b] > vf_norm_threshold).any()]
    it = iter_field_bands(zg, bands)
    if analysis.verbose:
        it = tqdm(it, total=len(bands), desc="map_celltypes (corr)")
    for r0, r1, raw in it:
        h = r1 - r0
        sig_flat = (norm[r0:r1] > vf_norm_threshold).reshape(-1)
        pos = np.where(sig_flat)[0]
        Bt = torch.as_tensor(raw.reshape(n_G, h * n_C), device=device)
        pos_t = torch.as_tensor(pos, device=device)
        bi = np.empty(pos.size, np.int32); bc = np.empty(pos.size, np.float32)
        for a in range(0, pos.size, sub_batch):
            b = min(pos.size, a + sub_batch)
            W = Bt.index_select(1, pos_t[a:b])
            if gi_t is not None:
                W = W.index_select(0, gi_t)
            d = W.T.contiguous()
            del W
            if marker_index_sets is not None:
                from ._marker_corr import masked_pearson_max_torch
                best_idx, best_corr = masked_pearson_max_torch(d, c, marker_index_sets)
            else:
                d = d - d.mean(dim=1, keepdim=True)
                d = d / torch.clamp(torch.linalg.norm(d, dim=1, keepdim=True), min=1e-12)
                best_corr, best_idx = (d @ c.T).max(dim=1)
            bi[a:b] = best_idx.cpu().numpy().astype(np.int32)
            bc[a:b] = best_corr.cpu().numpy().astype(np.float32)
            del d, best_corr, best_idx
        m_flat = M[r0:r1].reshape(-1); rc_flat = Rc[r0:r1].reshape(-1)
        m_flat[pos] = bi
        rc_flat[pos] = bc
        M[r0:r1] = m_flat.reshape(h, n_C); Rc[r0:r1] = rc_flat.reshape(h, n_C)
        del Bt, pos_t, raw
        torch.cuda.empty_cache()
    if analysis.verbose:
        _sig = int((norm > vf_norm_threshold).sum()); _tot = int((norm > 0).sum())
        print(f"[map_corr] assigned {int((M >= 0).sum())} voxels "
              f"(vf_norm>{vf_norm_threshold:.4g} keeps {_sig}/{_tot} = {_sig/max(_tot,1):.1%} of the field)")
    return _write_maps(zg, ds, M, Rc, sd, overwrite=overwrite)
