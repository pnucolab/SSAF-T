"""The vector-field backend for count data.

For a count platform the field equivalent to a kernel density estimate is built by
rasterizing the bin counts onto a grid and blurring that raster on the GPU, which
:func:`compute_vf_blur` does. Only ``vf_normalized`` and ``vf_norm`` are persisted to
the zarr store. Voxels are then sampled from the tissue image and looked up in that
field by :func:`select_vectors_from_image`, which is the count analogue of estimating
the density at sampled points on the transcript path. Cluster centroids are finally
mapped over the field by correlation, in :func:`map_corr`.

Every function here takes the ``Analysis`` instance, so that it can read and write
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
        """Stand in for ``tqdm.auto.tqdm`` when the package is not installed.

        The argument is returned unchanged, so a wrapped iterable still iterates and simply
        shows no progress bar.

        :param x: The iterable that would have been wrapped.
        :param k: Keyword arguments, accepted and ignored.
        :returns: The iterable, unchanged.
        """
        return x


def _gaussian_kernel_1d(sigma_bin, device, prune_coefficient=4.3):
    """Build the separable one-dimensional Gaussian taps for a blur on the GPU.

    The kernel is truncated at ``prune_coefficient`` standard deviations and normalized to
    sum to one, so that the blur preserves total mass.

    :param sigma_bin: Standard deviation of the Gaussian, in raster bins.
    :param device: The torch device to build the taps on.
    :param prune_coefficient: Truncation radius, in units of the standard deviation.
    :returns: A triple holding the taps shaped for a convolution along x, the taps shaped
        for a convolution along y, and the truncation radius in bins.
    :rtype: tuple
    """
    import torch
    radius = int(np.ceil(prune_coefficient * sigma_bin))
    k = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    g = torch.exp(-k ** 2 / (2 * sigma_bin ** 2)); g /= g.sum()
    return g.view(1, 1, 1, -1), g.view(1, 1, -1, 1), radius


def _radial_kernel_2d(sigma_bin, kernel, device, prune_coefficient=4.3):
    """Build a dense radial kernel on the bin raster, normalized to sum to one.

    ``sigma_bin`` is the Gaussian-equivalent standard deviation in bins, and the kernel's
    own scale is derived from it by matching the variance in two dimensions, through
    :func:`kernel_scale`. For the Epanechnikov kernel, for instance, that gives a support
    radius of ``sigma_bin`` times the square root of six.

    The result is returned as a single square kernel rather than a separable pair,
    because only the Gaussian is separable. The other radial kernels need the full square
    of taps, which is one cuDNN launch. At the default standard deviation of one bin the
    compact kernels span five by five to seven by seven taps against the Gaussian's two
    passes of eleven, while the exponential's long tail takes thirteen by thirteen.

    :param sigma_bin: Gaussian-equivalent standard deviation, in raster bins.
    :param kernel: Name of the kernel to build.
    :type kernel: str
    :param device: The torch device to build the kernel on.
    :param prune_coefficient: Truncation radius of the infinite-support kernels, in units
        of their own scale.
    :returns: A pair holding the kernel shaped for a two-dimensional convolution and its
        radius in bins.
    :rtype: tuple
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
    """Yield row bands together with the padded range each one needs.

    The padded range extends the band by the kernel radius on both sides, clipped to the
    raster, so that a truncated convolution is exact over the interior rows the caller
    keeps.

    :param n_R: Number of raster rows.
    :type n_R: int
    :param chunk_h: Height of each band, in rows.
    :type chunk_h: int
    :param radius: Kernel radius in rows.
    :type radius: int
    :returns: An iterator of ``(r0, r1, padded_r0, padded_r1)`` tuples.
    """
    for r0 in range(0, n_R, chunk_h):
        r1 = min(r0 + chunk_h, n_R)
        yield r0, r1, max(0, r0 - radius), min(n_R, r1 + radius)


def _rasterize_blur_tile_gpu(Xt, Rt, Ct, rp0, rp1, cp0, cp1, n_G,
                             kx_g, ky_g, radius, device, dtype):
    """Scatter one tile of pre-sliced bins onto the GPU and blur it.

    The count arrays passed in are already the bins of this tile, meaning those whose
    raster row lies in the padded row range and whose raster column lies in the padded
    column range. The caller slices the column-sorted band with a binary search, so no
    boolean rescan happens here. Only the nonzero entries inside the tile are
    transferred, so no dense array is built on the host, and the convolution runs in the
    requested precision. The expanded per-gene kernels are built once by the caller,
    because they are constant across tiles.

    The tile carries a halo of one kernel radius on every side, which is why the caller
    sets the padded ranges. The truncated separable Gaussian is therefore exact for the
    interior rows and columns the caller keeps, while the halo absorbs the effect of the
    zero padding at the edge and is then discarded. GPU memory is bounded by the number
    of genes times the padded tile area, whatever the size of the full raster.

    Passing no kernel for the y axis means the kernel is a radial two-dimensional one,
    which is every kernel but the Gaussian, and it is applied in a single dense
    convolution. The halo logic is unchanged.

    :param Xt: Counts of this tile's bins, as a sparse matrix of bins by genes.
    :param Rt: Raster row of each of this tile's bins.
    :param Ct: Raster column of each of this tile's bins.
    :param rp0: First padded row of the tile.
    :param rp1: One past the last padded row of the tile.
    :param cp0: First padded column of the tile.
    :param cp1: One past the last padded column of the tile.
    :param n_G: Number of genes.
    :type n_G: int
    :param kx_g: Per-gene kernel for the convolution along x, or the full radial kernel.
    :param ky_g: Per-gene kernel for the convolution along y, or ``None`` for a radial one.
    :param radius: Kernel radius in bins.
    :type radius: int
    :param device: The torch device to compute on.
    :param dtype: Precision of the convolution.
    :returns: The blurred tile, of shape genes by padded rows by padded columns.
    :rtype: torch.Tensor
    """
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
    """Normalize each voxel to a fixed total, then apply ``log1p``, in place.

    :param blurred_gpu: The blurred field, of shape genes by rows by columns. It is
        modified in place.
    :param scale: The total each voxel is scaled to before the logarithm.
    :param eps: Floor applied to the total, so that an empty voxel does not divide by zero.
    :returns: A pair holding the transformed field and the per-voxel total before scaling,
        which is the density ``vf_norm``.
    :rtype: tuple
    """
    import torch
    vf_norm = blurred_gpu.sum(dim=0)
    blurred_gpu.div_(torch.clamp(vf_norm, min=eps).unsqueeze(0)).mul_(scale)
    torch.log1p_(blurred_gpu)
    return blurred_gpu, vf_norm


def _tensorstore_array(group_dir, name, shape, chunks):
    """Open a store array through tensorstore, mirroring the zarr writer's metadata.

    The metadata is written to match the zarr-python writer exactly: float32 elements, a
    bytes codec followed by zstd at level zero, default chunk keys separated by slashes,
    and a fill value of zero. The store on disk is therefore identical on read-back, and
    any zarr version 3 reader, whether zarr-python or the downstream sparse gather, sees
    no difference. The function works for the three-dimensional count field and the
    four-dimensional transcript field alike.

    Routing the large write of ``vf_normalized`` through the multithreaded asynchronous
    pipeline of tensorstore is about 4.7 times faster than the single-event-loop
    zarr-python writer, and about 2.6 times faster end to end, because the asynchronous
    writes overlap the next tile's GPU work. The write is the dominant cost. The values
    are bit-identical, since the codec is the same and is lossless.

    :param group_dir: Directory of the zarr group, or ``None`` when the store is not a
        local directory.
    :param name: Name of the array within the group.
    :type name: str
    :param shape: Shape of the array.
    :param chunks: Chunk shape of the array.
    :returns: The opened tensorstore array, or ``None`` when tensorstore is not installed,
        the store is not a local directory, or ``CELLMAN_NO_TENSORSTORE`` is set, in which
        case the caller falls back to the zarr-python writer.
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
    """Write slabs of existing store arrays through tensorstore, asynchronously.

    The per-voxel outputs of mapping, namely the density, the cell-type map, the score
    and the marker volumes, are written one slab at a time along x. Through zarr-python
    that is a single event-loop thread, which reaches about 35 megabytes per second on
    this storage and made writing a three-dimensional map take longer than computing it.

    The arrays are opened as they already are on disk, by reading their metadata rather
    than redeclaring it, so the element type, the fill value and the codecs are the
    store's own and the bytes written are identical. Up to ``max_inflight`` slab writes
    overlap the next slab's GPU work, and :meth:`flush` waits for all of them.

    The class falls back to plain zarr-python assignment when tensorstore is unavailable,
    when the store is not a local directory, or when ``CELLMAN_NO_TENSORSTORE`` is set.
    """

    def __init__(self, zg, max_inflight=8):
        """Prepare the writer, deciding whether tensorstore can be used.

        :param zg: The zarr group holding the arrays that will be written.
        :param max_inflight: Largest number of slab writes allowed to be outstanding at once.
        :type max_inflight: int
        """
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
        """Write one slab of an array, asynchronously where possible.

        The source buffer is copied before the call returns, so the caller may reuse it
        immediately, while the commit itself stays asynchronous. Once more than
        ``max_inflight`` commits are outstanding, the oldest is waited on.

        :param name: Name of the array in the group.
        :type name: str
        :param x0: First index of the slab along x.
        :type x0: int
        :param x1: One past the last index of the slab along x.
        :type x1: int
        :param arr: The data to write.
        """
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
        """Wait for every outstanding write to commit."""
        for f in self.inflight:
            f.result()
        self.inflight.clear()


def remove_array(zg, name):
    """Delete an array from a zarr group, even when a killed writer left debris.

    tensorstore leaves lock files behind when its process dies in the middle of a write,
    and on a network file system a file still open on a lost host lingers under a hidden
    name. The delete offered by zarr then stops on the first key it cannot remove. This
    function falls back to removing the directory, and if even that leaves something
    behind, such as a silly-renamed file, it moves the array aside so that the name is
    free for the new one.

    :param zg: The zarr group holding the array.
    :param name: Name of the array to remove.
    :type name: str
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
    """Return the on-disk directory of a zarr group, if it has one.

    :param zg: The zarr group.
    :returns: The directory path, or ``None`` when the group is not backed by a local
        directory.
    :rtype: str or None
    """
    root = getattr(getattr(zg, "store", None), "root", None)
    if root is None:
        return None
    return os.path.join(str(root), zg.path) if zg.path else str(root)


def _tensorstore_open_read(group_dir, name):
    """Open an existing store array through tensorstore for reading.

    :param group_dir: Directory of the zarr group, or ``None``.
    :param name: Name of the array within the group.
    :type name: str
    :returns: The opened array, or ``None`` when tensorstore cannot or should not be used.
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
    """Yield each requested row band of a three-dimensional count field.

    ``row_bands`` is the list of bands the caller actually wants, so bands with no signal
    are never touched. The next band's read is started before the current one is handed
    over, which hides the read behind the caller's GPU work, since the two run on
    different engines and the read is otherwise about 60 percent of a mapping run.

    The block handed over is exactly what slicing the array by that band returns, of
    shape genes by band rows by columns and of dtype float32. Only how it got there
    changes.

    :param zg: The zarr group holding the field.
    :param row_bands: The bands to read, as ``(r0, r1)`` pairs.
    :param name: Name of the field array in the group.
    :type name: str
    :param prefetch: Whether to start the next band's read before handing over the
        current one.
    :type prefetch: bool
    :returns: An iterator of ``(r0, r1, block)`` tuples.
    """
    A = _tensorstore_open_read(_store_group_dir(zg), name)
    z = zg[name]
    if A is not None:
        def start(b):
            """Start the read of one band through tensorstore and return its future.

            :param b: The band, as an ``(r0, r1)`` pair.
            :returns: A future that will hold the band.
            """
            r0, r1 = b
            return A[:, r0:r1, :].read()          # future; the C++ pool does the work
        def finish(f):
            """Wait for a tensorstore read and return the band as an array.

            :param f: The future returned by the matching start function.
            :returns: The band, of shape genes by band rows by columns.
            :rtype: numpy.ndarray
            """
            return np.asarray(f.result())
    else:
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(1)              # zarr-python releases the GIL in zstd
        def start(b):
            """Start the read of one band on a worker thread and return its future.

            This is the fallback used when tensorstore is unavailable. zarr-python releases the
            interpreter lock inside zstd, so one worker thread is enough to overlap the read.

            :param b: The band, as an ``(r0, r1)`` pair.
            :returns: A future that will hold the band.
            """
            r0, r1 = b
            return pool.submit(lambda: np.asarray(z[:, r0:r1, :]))
        def finish(f):
            """Wait for a worker-thread read and return the band.

            :param f: The future returned by the matching start function.
            :returns: The band, of shape genes by band rows by columns.
            :rtype: numpy.ndarray
            """
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
    """Choose how many raster rows fit one band within a memory budget.

    :param n_G: Number of genes.
    :type n_G: int
    :param n_C: Number of raster columns.
    :type n_C: int
    :param chunk_h: The requested number of rows, which is only ever reduced.
    :type chunk_h: int
    :param budget_bytes: Memory budget for one band, in bytes.
    :type budget_bytes: int
    :param align: The result is rounded down to a multiple of this, so that a band stays
        aligned to the store's chunk grid and no interior write becomes a
        read-modify-write. A budget too small for even one aligned band is honoured
        instead of the alignment.
    :type align: int
    :returns: The number of rows per band, never below one.
    :rtype: int
    """
    per_row = int(n_G) * int(n_C) * 4
    fit = max(1, int(budget_bytes) // max(per_row, 1))
    fit = min(int(chunk_h), fit)
    return max(align, (fit // align) * align) if fit >= align else fit


def _tensorstore_vf(group_dir, n_G, out_R, out_C, sh, sw):
    """Open the vector-field array for writing through tensorstore.

    :param group_dir: Directory of the zarr group, or ``None`` when the store is not a
        local directory.
    :param n_G: Number of genes, which is the first axis of the field.
    :type n_G: int
    :param out_R: Number of output rows.
    :type out_R: int
    :param out_C: Number of output columns.
    :type out_C: int
    :param sh: Chunk height on disk, in rows.
    :type sh: int
    :param sw: Chunk width on disk, in columns.
    :type sw: int
    :returns: The opened array, or ``None`` when tensorstore cannot or should not be used.
    """
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
    """Normalize a kernel name and check that it is one this package implements.

    :param kernel: The kernel name, in any case.
    :type kernel: str
    :returns: The canonical name, as it appears in :data:`KDE_KERNELS`.
    :rtype: str
    :raises ValueError: If the name is not one of the supported kernels.
    """
    k = str(kernel).strip().lower()
    if k in KDE_KERNELS:
        return k
    hits = [name for name in KDE_KERNELS if name.startswith(k)] if len(k) >= 3 else []
    if len(hits) == 1:
        return hits[0]
    raise ValueError(f"unknown kernel {kernel!r}; use one of {', '.join(KDE_KERNELS)}")


def kernel_code(kernel):
    """Return the integer identifier of a kernel, as the compiled code expects it.

    :param kernel: The kernel name.
    :type kernel: str
    :returns: The index of that kernel in :data:`KDE_KERNELS`.
    :rtype: int
    """
    return KDE_KERNELS.index(resolve_kernel(kernel))


def kernel_scale(bandwidth, ndim, kernel='gaussian'):
    """Convert a Gaussian bandwidth to another kernel's own scale, matching variance.

    Each kernel is parameterized by its own scale: the standard deviation for the
    Gaussian, the support radius for a compact kernel, and the decay length for the
    exponential. Matching the per-axis variance of the two profiles in the given number
    of dimensions gives a fixed multiple of the bandwidth for each kernel, so that every
    kernel smooths over the same width.

    :param bandwidth: The Gaussian bandwidth, in whatever unit the field is measured in.
    :param ndim: Number of dimensions the kernel acts over.
    :type ndim: int
    :param kernel: The kernel whose scale is wanted.
    :type kernel: str
    :returns: That kernel's own scale, in the same unit as the bandwidth.
    :rtype: float
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
    """Return the radius beyond which a kernel contributes nothing.

    A compact kernel reaches exactly its own scale. The Gaussian and the exponential are
    infinite, so they are truncated at ``prune_coefficient`` times their scale, the
    exponential being cut at the same relative height as the Gaussian.

    :param scale: The kernel's own scale.
    :param kernel: The kernel name.
    :type kernel: str
    :param prune_coefficient: Truncation radius of the infinite-support kernels, in units
        of their scale.
    :returns: The reach, in the same unit as the scale.
    :rtype: float
    """
    k = resolve_kernel(kernel)
    if k in _COMPACT_KERNELS:
        return float(scale)
    if k == 'exponential':
        return float(scale) * float(prune_coefficient) ** 2 / 2.0
    return float(scale) * float(prune_coefficient)


def kernel_search_cells(reach, sampling_distance, kernel='gaussian'):
    """Return how many cells of the lattice a kernel of this reach can span.

    This is the half-width of the neighbourhood a cell-list search must visit.

    :param reach: The kernel's reach.
    :param sampling_distance: The lattice spacing, which is also the cell size.
    :param kernel: The kernel name.
    :type kernel: str
    :returns: The half-width, in cells.
    :rtype: int
    """
    extra = 1 if resolve_kernel(kernel) in _COMPACT_KERNELS else 0
    return int(np.floor(float(reach) / float(sampling_distance) + 1e-12)) + extra


def kernel_norm(scale, ndim, kernel='gaussian'):
    """Return the normalization constant that makes a kernel integrate to one.

    :param scale: The kernel's own scale.
    :param ndim: Number of dimensions the kernel acts over.
    :type ndim: int
    :param kernel: The kernel name.
    :type kernel: str
    :returns: The constant the unnormalized profile must be divided by.
    :rtype: float
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
    """Evaluate a kernel's unnormalized profile at a scaled radius.

    Every profile equals one at the origin, so the peak of the normalized kernel is the
    reciprocal of :func:`kernel_norm`.

    :param u: Radius divided by the kernel's own scale.
    :param kernel: The kernel name.
    :type kernel: str
    :returns: The profile value, which is zero beyond the kernel's support.
    """
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
    """Convert a Gaussian bandwidth to an Epanechnikov support radius, by variance.

    The spherical Epanechnikov kernel, which is proportional to one minus the squared
    scaled radius on the ball of radius H, has a per-axis variance of H squared divided by
    the number of dimensions plus four. A Gaussian of bandwidth h has a per-axis variance
    of h squared. Equating the two gives H as h times the square root of the number of
    dimensions plus four, which is 2.236 times h in one dimension, 2.449 in two and 2.646
    in three.

    This function is kept for callers that predate :func:`kernel_scale`, which generalizes
    it to every kernel.

    :param bandwidth: The Gaussian bandwidth.
    :param ndim: Number of dimensions the kernel acts over.
    :type ndim: int
    :returns: The Epanechnikov support radius.
    :rtype: float
    """
    return kernel_scale(bandwidth, ndim, 'epanechnikov')


def kernel_peak(scale, ndim, kernel='gaussian'):
    """Return the peak of one molecule's kernel, which is the expression threshold.

    The peak is the reciprocal of the kernel's effective volume, and it is the quantity
    the pipeline calls ``expression_threshold``.

    The scale must be the kernel's own scale, in the units ``vf_norm`` is measured in:
    the standard deviation for a Gaussian, the support radius for a compact kernel, or the
    decay length for the exponential. Convert with :func:`kernel_scale`, and never pass a
    Gaussian bandwidth here. Every profile equals one at the origin, so the peak is simply
    the reciprocal of :func:`kernel_norm`. For a Gaussian that is one over the
    ``ndim`` power of the standard deviation times the square root of two pi, and for the
    Epanechnikov it is three over four H in one dimension, two over pi H squared in two,
    and fifteen over eight pi H cubed in three.

    Everything downstream is anchored on this one number. The density floor
    ``norm_threshold`` is twice it, meaning a voxel holding less than two molecules' worth
    of evidence, and the count reconstruction of the LDA mapper multiplies the density
    back up by its reciprocal. Both come out wrong, silently and by about a factor of two
    at the default bandwidth, if the Gaussian formula is applied to another kernel's
    field.

    :param scale: The kernel's own scale, in the units of the field.
    :param ndim: Number of dimensions the kernel acts over.
    :type ndim: int
    :param kernel: The kernel name.
    :type kernel: str
    :returns: The peak density of a single molecule.
    :rtype: float
    """
    return float(1.0 / kernel_norm(scale, ndim, kernel))


def compute_vf_blur(analysis, bin_adata, *, sigma_bin: float = 1.0,
                    chunk_h: int = 64, chunk_w: int = 512, bin_size_um: float = 2.0,
                    sampling_distance: float = None, prune_coefficient: float = 4.3,
                    kernel: str = "gaussian",
                    size_after_normalization: float = 10.0,
                    store_chunk=32,
                    device: str = "cuda") -> None:
    """Rasterize bin counts and blur them on the GPU into the normalized field.

    Two independent resolutions are involved. The bin size is the spacing of the raster
    the blur is computed on, where finer means a more accurate blur; the standard
    deviation is given in bins, so the bandwidth is that standard deviation times the bin
    size. The sampling distance is the spacing of the output field. The blurred field is
    point-sampled, that is decimated, by the ratio of the two rounded to an integer,
    giving a coarser field than the raster and so fewer voxels on disk, while the blur
    itself stays raster-accurate. The sampling distance defaults to the bin size, which
    makes the output the raster.

    The function writes ``vf_normalized``, of shape genes by output rows by output
    columns, and ``vf_norm``, of shape output rows by output columns, both at the
    sampling distance, together with metadata recording that the backend was the blur one.
    Only the normalized field is persisted; the raw field never is.

    The on-disk chunk shape is decoupled from the GPU compute tile. The compute tile stays
    large for GPU efficiency, while the store is written in small square spatial tiles,
    with the gene axis always whole, so that a sparse voxel gather decodes only the tiles
    that actually contain a sampled voxel rather than whole full-width row bands. For the
    writes to stay aligned, and so avoid a read-modify-write on interior writes, the
    compute tile dimensions should be integer multiples of the store tile, as the defaults
    are.

    :param analysis: The ``Analysis`` instance, which provides the dataset and the
        verbosity setting.
    :type analysis: Analysis
    :param bin_adata: Bin-level counts carrying ``obs['array_row']`` and
        ``obs['array_col']``.
    :type bin_adata: AnnData
    :param sigma_bin: Gaussian standard deviation, in raster bins.
    :type sigma_bin: float
    :param chunk_h: Height of the GPU compute tile, in output rows.
    :type chunk_h: int
    :param chunk_w: Width of the GPU compute tile, in output columns.
    :type chunk_w: int
    :param bin_size_um: Spacing of the blur raster, in micrometres.
    :type bin_size_um: float
    :param sampling_distance: Spacing of the output field, in micrometres. It defaults to
        the bin size.
    :type sampling_distance: float
    :param prune_coefficient: Truncation radius of the infinite-support kernels, in units
        of the standard deviation. The exponential is cut at the same relative height as
        the Gaussian.
    :type prune_coefficient: float
    :param kernel: One of :data:`KDE_KERNELS`, namely the Gaussian, tophat, Epanechnikov,
        exponential, linear or cosine kernel. The standard deviation keeps the same
        meaning for all of them, because each kernel's own scale is derived from it by
        variance matching in :func:`kernel_scale`, so that they smooth over the same
        width. Only the Gaussian is separable; the others run as one dense convolution,
        whose cost grows with the square of the radius while the Gaussian's separable pair
        grows linearly. Below a standard deviation of about six bins, a compact kernel's
        smaller radius still wins, taking about 0.8 times as long for the Epanechnikov
        between one and four bins, and above it the Gaussian pulls ahead.
    :type kernel: str
    :param size_after_normalization: The total each voxel is scaled to before ``log1p``.
    :type size_after_normalization: float
    :param store_chunk: On-disk spatial chunk, given as an integer square side, a pair of
        height and width, or ``None`` to match the compute tile.
    :type store_chunk: int or tuple or None
    :param device: Torch device string.
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
    """Reduce a staining image to a single grey channel, inverting it when asked.

    A colour image is converted with the usual luminance weights, after being moved to
    channel-last order and cut down to three channels. Inverting is what a bright-field
    image needs before the tissue mask is thresholded.

    :param image: The staining image.
    :param revert: Whether the image has a bright background and dark tissue, so that it
        must be inverted.
    :type revert: bool
    :returns: The grey image, as float64.
    :rtype: numpy.ndarray
    """
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
    """Sample voxels inside the tissue mask and look them up in the stored field.

    The tissue mask comes from thresholding the staining image by Otsu's method, voxels
    are sampled within it, and their vectors are read out of the precomputed
    ``vf_normalized`` field into ``normalized_vectors``. This is the count platform's
    counterpart of the transcript path's downsampling.

    Pixels are mapped to world micrometres through the grid's own affine, which folds in
    each platform's transposition and flipping, and then to lattice bins through the
    grid's origin and bin size.

    :param analysis: The ``Analysis`` instance, which provides the dataset and the
        verbosity setting.
    :type analysis: Analysis
    :param grid: The raster and image grid, carrying the image, the origin, the pixel
        size, the affine from pixels to micrometres and the mask polarity.
    :type grid: object
    :param bin_size_um: Spacing of the lattice, in micrometres.
    :type bin_size_um: float
    :param n_samples_per_block: Number of pixels sampled per square block.
    :type n_samples_per_block: int
    :param block_um: Side of the sampling block, in micrometres.
    :type block_um: float
    :param otsu_threshold: A fixed threshold for the tissue mask. It is computed by Otsu's
        method when ``None``.
    :type otsu_threshold: float or None
    :param otsu_mult: Factor applied to the threshold this package computes, where 0.6
        lowers it and so widens the mask. It applies only when no explicit threshold was
        given, because a threshold that was given is used as given.
    :type otsu_mult: float or None
    :param norm_threshold_mult: Multiplier for the analytic density threshold, used when
        no percentile is given.
    :type norm_threshold_mult: float
    :param vf_norm_percentile: Percentile of the nonzero densities to use as the keep
        threshold. Passing ``None`` uses the analytic form instead.
    :type vf_norm_percentile: float or None
    :param lookup_chunk_h: Retained for compatibility and no longer used to size the read.
    :type lookup_chunk_h: int
    :param seed: Seed of the random number generator used for pixel sampling.
    :type seed: int
    :param lookup_n_workers: Number of workers for the parallel tile lookup. It defaults
        to the analysis's core count.
    :type lookup_n_workers: int or None
    :param lookup_ram_gb: Memory budget bounding how many tiles are decoded at once.
    :type lookup_ram_gb: float
    :returns: The sampled voxel positions, as lattice indices.
    :rtype: numpy.ndarray
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
        """Read one raster tile and gather every sampled voxel that falls inside it.

        The sampled voxels were sorted by tile beforehand, so this reads the tile once and
        fills the rows of ``normalized_vectors`` belonging to it.

        :param i: Index of the tile within the list of occupied tiles.
        :type i: int
        :returns: The number of voxels filled from this tile.
        :rtype: int
        """
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
    """Return the name of the score volume a given mapping method writes.

    :param method: The mapping method, such as the correlation, the naive Bayes or the
        LDA one.
    :type method: str
    :returns: Either ``'max_correlations'`` or ``'max_probabilities'``.
    :rtype: str
    """
    try:
        return SCORE_KEYS[method]
    except KeyError:
        raise ValueError(f"unknown mapping method {method!r} "
                         "(expected 'corr', 'gnb' or 'lda')")


def _other_score_key(score_key):
    """Return the score volume the other family of mapping methods writes.

    :param score_key: One of the two score volume names.
    :type score_key: str
    :returns: The other one.
    :rtype: str
    """
    return ("max_probabilities" if score_key == "max_correlations"
            else "max_correlations")


def _write_maps(zg, ds, M, Rc, sd, *, overwrite=True, set_ds_attrs=True,
                score_key="max_correlations", method=None):
    """Write a cell-type map and its score volume into the store.

    The score key names the metric actually produced: correlations for the centroid
    correlation mapper, and probabilities for the naive Bayes and LDA posteriors. Both
    volumes may coexist, so rerunning one method no longer throws away the other method's
    result. The attribute ``mapping_method`` records which one is live, and
    ``filter_celltypemaps`` reads it to pick the matching threshold.

    :param zg: The zarr group to write into.
    :param ds: The dataset whose attributes are updated.
    :param M: The cell-type map, of shape rows by columns.
    :param Rc: The score of each voxel, on the same grid.
    :param sd: Lattice spacing, in micrometres.
    :param overwrite: Whether to delete any existing arrays of the same name first.
    :type overwrite: bool
    :param score_key: Name of the score volume to write.
    :type score_key: str
    :param method: Name of the mapping method, needed when the score key alone does not
        identify it, because the naive Bayes and LDA mappers share the posterior volume.
    :type method: str or None
    :param set_ds_attrs: Whether to update the dataset's own attributes from what was
        written.
    :type set_ds_attrs: bool
    :returns: A pair holding the written cell-type map and score volume, each with a
        trailing singleton depth axis.
    :rtype: tuple
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
    """Return the density floor, as a multiple of a single molecule's kernel peak.

    This is the threshold every other part of the package uses, and the mappers must use
    it too. It is an absolute quantity, saying that a voxel holds less than so many
    molecules' worth of evidence, so it means the same thing on every dataset. A
    percentile would not, because it always discards a fixed share of the tissue however
    good the signal is.

    The units follow the field the backend actually stored, so the scale must be given in
    the same units as the support of ``vf_norm``. On the count backend, as used for Visium
    HD and Stereo-seq, the density is per squared bin and the blur is a standard deviation
    in bins. On the transcript backend the density is per unit volume in micrometres and
    the blur is a bandwidth in micrometres. Whatever kernel the store records is used, with
    that kernel's own peak from :func:`kernel_peak`, and the recorded scale is already
    that kernel's own.

    Mixing the two conventions is wrong by the bin size raised to the number of
    dimensions, which is a factor of four at two micrometre bins in two dimensions.

    :param zg: The zarr group whose recorded blur settings are read.
    :param mult: Number of molecules' worth of evidence a voxel must hold.
    :type mult: float
    :returns: The threshold, or zero when the store records neither convention, which
        keeps every voxel.
    :rtype: float
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
    """Resolve the density threshold to use, from an explicit value or the store.

    :param zg: The zarr group whose recorded blur settings are read.
    :param vf_norm_threshold: An explicit threshold, or ``None`` to derive it from the
        store through :func:`analytic_vf_norm_threshold`.
    :param mult: Number of molecules' worth of evidence a voxel must hold, used only when
        the threshold is derived.
    :type mult: float
    :returns: The threshold to apply.
    :rtype: float
    """
    if vf_norm_threshold is not None:
        return float(vf_norm_threshold)
    return analytic_vf_norm_threshold(zg, mult=mult)


def map_corr(analysis, centroids=None, *, vf_norm_threshold=None, chunk_h=128,
             device="cuda", overwrite=True, gene_indices=None, marker_index_sets=None,
             gpu_target_bytes=int(4e9), sub_batch=None):
    """Map the count and blur field to cell types by centroid correlation.

    Every voxel whose density exceeds the threshold is assigned the centroid it correlates
    best with, and the correlation is written alongside the label as the score volume.

    :param analysis: The ``Analysis`` instance, which provides the dataset and the
        verbosity setting.
    :type analysis: Analysis
    :param centroids: The cluster centroids, of shape clusters by genes. They are read
        from the store when ``None``.
    :param vf_norm_threshold: Density below which a voxel is left unmapped. It is resolved
        from the store when ``None``.
    :param chunk_h: Number of raster rows per band, before any reduction to fit the memory
        budget.
    :type chunk_h: int
    :param device: Torch device for the correlation.
    :type device: str
    :param overwrite: Whether to replace existing map arrays in the store.
    :type overwrite: bool
    :param gene_indices: Positions into the store's gene axis to keep, where ``None``
        keeps all of them.
    :param marker_index_sets: Per-celltype marker gene positions, from
        :func:`ssaft._marker_corr.marker_index_sets`. When given, each celltype is scored
        only on its own markers. When ``None``, every celltype shares one gene axis, which
        lets the centroids be normalized once up front.
    :param gpu_target_bytes: Device memory budget for one band of the field. The number of
        rows per band is reduced to fit it, so that a wide gene axis cannot run the card
        out of memory the way a fixed row count can.
    :type gpu_target_bytes: int
    :param sub_batch: Voxels scored per device batch inside a band. It is sized from the
        same budget when ``None``.
    :returns: Whatever :func:`_write_maps` returns after writing the label raster and the
        correlations into the store.
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
