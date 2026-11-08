"""Pearson correlation restricted, per celltype, to that celltype's marker genes.

The correlation itself is the usual one. Only the gene axis changes from celltype to
celltype. The default scores every celltype on the same axis, correlating a voxel's
whole vector against a centroid's whole vector. That axis is dominated by genes which
are flat across celltypes, so a rare celltype defined by three genes is judged mostly
on the sixty or so genes it shares with everything else.

When a marker panel is supplied, through ``map_celltypes(marker_genes_path=...)``,
each celltype is correlated only on its own markers. Those are the markers on its own
node of the ``celltype_markers.yaml`` tree together with the markers of every
ancestor, because the Cell-Ontology is normalized to record only the delta at each
level. See :meth:`~ssaft.Analysis.read_marker_genes`. A glomerular endothelial cell is
therefore scored on ``Ehd3``, ``Podxl`` and ``Icam2`` as well as on the endothelial
cell markers above it, while a proximal tubule cell is scored on its own markers.

The centroids themselves are still built over the union of all marker genes. One
shared gene axis is what makes the celltypes comparable at all, and the marker set
only picks which of its columns a given celltype is judged on.

Marker sets that are too small
------------------------------
A Pearson correlation over n points needs n to be at least 3 to mean anything,
because over 2 points it is always exactly 1 or minus 1 whatever the values. A
celltype with two on-panel markers therefore scores 1.0 on every voxel where its two
markers happen to be ordered the way its centroid's are. It wins those voxels
outright and, worse, it is immune to ``filter_celltypemaps(min_r=...)``, because no
threshold below 1 can remove it.

Measured on the mouse kidney Xenium panel, where the proximal tubule cell had only
``Epcam`` and ``Slc5a12`` on panel, 23.5 percent of its cells scored exactly 1.0 and
29.4 percent of all cells ended up saturated above 0.999 in absolute value.

A celltype with fewer than :data:`MIN_MARKER_GENES` markers on the panel therefore
falls back to the full axis and is named in a warning. That keeps the score
filterable, but it is a guard rather than a fix. The fix is for the annotation to
emit more on-panel markers for that celltype in ``celltype_markers.yaml``.

The fallback belongs to the correlation and nowhere else. :func:`marker_score` is a
mean over the marker genes, and a mean over one gene is neither degenerate nor
unfilterable. Routing it through the fallback would score that celltype 0, which
:func:`marker_score_cutoffs` then skips, leaving it with no cut at all. The LDA path
therefore builds its score on the fallback-free sets, by passing ``min_genes=0``.

Why a loop and not one masked matrix multiply
---------------------------------------------
A masked correlation can be batched into three matrix multiplies of shape n by g
against g by k, using the computational formula for covariance. It is not worth it
here. Marker sets are small, a handful of genes out of hundreds, so gathering the
columns of one celltype costs n times the sum of the set sizes, which is well under
the n times g times k of the batched form. The two-pass centering used below is also
exact, whereas the computational formula loses about three digits in float32 on
marker sets that are nearly constant.
"""
import numpy as np

__all__ = ['marker_index_sets', 'masked_pearson_max', 'masked_pearson_max_torch',
           'marker_score', 'half_sample_mode', 'marker_score_cutoffs',
           'density_cutoffs', 'MIN_MARKER_GENES']

#: Fewest on-panel markers a celltype needs before it is scored on its own axis.
#: Below this the correlation is degenerate (n=2 is always ±1) and unfilterable.
MIN_MARKER_GENES = 3


def marker_score(density, labels, marker_idx):
    """Return each voxel's mean density over the marker genes of its own celltype.

    This is the quantity that says whether a celltype is present in sufficient quantity to
    be called, as opposed to ``vf_norm``, which sums all genes and so measures how
    transcript-rich the neighbourhood is regardless of which celltype won. Expression
    level differs by an order of magnitude between celltypes, so the comparison means
    something only within a celltype. Hence one score per voxel, taken over the winning
    celltype's markers, with a threshold set per celltype further downstream.

    :param density: Absolute densities of shape ``(n, G)``, in molecules per cubic or
        square micrometre according to the dimensionality, on the same gene axis that
        ``marker_idx`` indexes.
    :param labels: Celltype identifier of each voxel, of shape ``(n,)``. A value of minus
        one scores zero.
    :param marker_idx: The list of index arrays returned by :func:`marker_index_sets`. An
        entry of ``None`` means the celltype has no marker set and is scored zero, so that
        the filter leaves it alone.
    :returns: The per-voxel score, of shape ``(n,)`` and dtype float32.
    :rtype: numpy.ndarray
    """
    out = np.zeros(len(labels), dtype=np.float32)
    for k in np.unique(labels):
        if k < 0 or k >= len(marker_idx):
            continue
        idx = marker_idx[k]
        if idx is None or len(idx) == 0:
            continue
        # Rows AND columns in one gather. Selecting either one first materialises a
        # block far larger than what is kept: ``density[m][:, idx]`` copies the rows at
        # full width (the whole chunk, once per celltype, on a 5k-gene panel) and
        # ``density[:, idx][m]`` copies every row's markers before dropping most rows.
        # ``np.ix_`` takes only the n_k x |idx| values the mean actually needs.
        rows = np.flatnonzero(labels == k)
        out[rows] = np.asarray(density[np.ix_(rows, idx)], dtype=np.float64).mean(1)
    return out


def marker_index_sets(celltype_names, marker_genes, gene_names, *,
                      min_genes=MIN_MARKER_GENES, warn_few=True,
                      verbose=False, log=print):
    """Return the gene positions that each celltype should be correlated on.

    Each returned index array indexes into ``gene_names``, that is into the gene axis
    actually being mapped after any removal requested through ``exclude_gene_indices``, so
    a caller can apply it to its data chunk directly.

    An entry of ``None`` means the celltype is scored on the whole axis. A celltype gets
    that in two cases. The first is that it has no entry in ``marker_genes``, which is the
    honest fallback for the annotation's ``unknown1`` to ``unknownN`` classes, since those
    were never defined by markers. The second is that fewer than ``min_genes`` of its
    markers are on the panel; see the module docstring on why a two-gene correlation
    cannot be thresholded.

    :param celltype_names: The celltype names, in the order of the centroid rows.
    :param marker_genes: Mapping from celltype name to its marker gene symbols. Lookup is
        case-insensitive.
    :type marker_genes: dict
    :param gene_names: The gene axis being mapped.
    :param min_genes: Fewest on-panel markers a celltype needs before it is scored on its
        own axis rather than the full one.
    :type min_genes: int
    :param warn_few: Whether to name the celltypes that fell back for having too few
        markers.
    :type warn_few: bool
    :param verbose: Whether to log a summary at all.
    :type verbose: bool
    :param log: The callable used for logging.
    :returns: One entry per celltype, each either an int64 array of gene positions or
        ``None``.
    :rtype: list
    """
    pos = {g: i for i, g in enumerate(gene_names)}
    lower = {str(g).lower(): i for i, g in enumerate(gene_names)}
    mk_lower = {str(k).lower(): v for k, v in marker_genes.items()}

    out, n_named, n_unnamed, too_few = [], 0, 0, []
    for name in celltype_names:
        genes = marker_genes.get(name, mk_lower.get(str(name).strip().lower()))
        if genes is None:
            out.append(None)
            n_unnamed += 1
            continue
        idx = sorted({pos[g] if g in pos else lower[str(g).lower()]
                      for g in genes if g in pos or str(g).lower() in lower})
        if len(idx) < int(min_genes):
            out.append(None)
            too_few.append(f'{name} ({len(idx)})')
            continue
        out.append(np.asarray(idx, dtype=np.int64))
        n_named += 1

    if verbose:
        sizes = [len(i) for i in out if i is not None]
        log(f'map_celltypes: marker sets — {n_named} celltypes '
            f'scored on their own markers (median {int(np.median(sizes)) if sizes else 0} '
            f'genes, range {min(sizes) if sizes else 0}-{max(sizes) if sizes else 0} '
            f'of {len(gene_names)}), {n_unnamed} with no marker set on the full axis')
        if too_few and warn_few:
            log(f'map_celltypes: WARNING {len(too_few)} celltypes have <{min_genes} '
                f'markers on the panel, where Pearson is degenerate (n=2 is always '
                f'±1 and min_r cannot filter it); they fall back to the full axis: '
                f'{too_few[:8]}. Add on-panel markers for them in celltype_markers.yaml.')
    return out


def _pearson_against(sub, cen):
    """Correlate every row of a matrix against one vector.

    :param sub: The rows, of shape ``(n, g)``. The array is centered in place.
    :param cen: The vector to correlate against, of shape ``(g,)``.
    :returns: The correlations, of shape ``(n,)``.
    :rtype: numpy.ndarray
    """
    sub = sub - sub.mean(axis=1, keepdims=True)
    sub /= np.maximum(np.linalg.norm(sub, axis=1, keepdims=True), 1e-12)
    cen = cen - cen.mean()
    cen = cen / max(float(np.linalg.norm(cen)), 1e-12)
    return sub @ cen


def masked_pearson_max(data, centroids, index_sets, *, bar=None):
    """Pick the best celltype per row, each scored on its own marker genes, in NumPy.

    :param data: Voxel vectors of shape ``(n_voxels, n_genes)``.
    :param centroids: Centroids of shape ``(k, n_genes)``, on the same gene axis.
    :param index_sets: The index arrays from :func:`marker_index_sets`, where ``None``
        means the full axis. Pass ``None`` for the whole argument to score every celltype
        on the full axis.
    :param bar: Optional progress bar, advanced once per celltype.
    :returns: A pair. The first element holds the winning celltype index per row as int32,
        and is minus one where no celltype scored. The second holds that row's correlation
        as float32.
    :rtype: tuple of numpy.ndarray
    """
    n = data.shape[0]
    best_corr = np.full(n, -1.0, dtype=np.float32)
    best_idx = np.full(n, -1, dtype=np.int32)
    for k, cen in enumerate(centroids):
        idx = index_sets[k] if index_sets is not None else None
        sub = data if idx is None else data[:, idx]
        c = cen if idx is None else cen[idx]
        r = _pearson_against(np.asarray(sub, dtype=np.float32), np.asarray(c, np.float32))
        r = np.nan_to_num(r, nan=-1.0).astype(np.float32, copy=False)
        improved = r > best_corr
        best_corr[improved] = r[improved]
        best_idx[improved] = k
        if bar is not None:
            bar.update(1)
    return best_idx, best_corr


def masked_pearson_max_torch(data_t, centroids_t, index_sets, *, bar=None):
    """Pick the best celltype per row, in torch, keeping everything on the device.

    This is the counterpart of :func:`masked_pearson_max` and returns the same quantities.

    :param data_t: Voxel vectors of shape ``(n, g)``.
    :type data_t: torch.Tensor
    :param centroids_t: Centroids of shape ``(k, g)``.
    :type centroids_t: torch.Tensor
    :param index_sets: The index arrays from :func:`marker_index_sets`, where ``None``
        means the full axis.
    :param bar: Optional progress bar, advanced once per celltype.
    :returns: A pair of tensors on the same device, holding the winning celltype index per
        row as int32 and that row's correlation as float32.
    :rtype: tuple of torch.Tensor
    """
    import torch
    n = data_t.shape[0]
    best_corr = torch.full((n,), -1.0, dtype=torch.float32, device=data_t.device)
    best_idx = torch.full((n,), -1, dtype=torch.int32, device=data_t.device)
    for k in range(centroids_t.shape[0]):
        idx = index_sets[k] if index_sets is not None else None
        if idx is None:
            sub, c = data_t, centroids_t[k]
        else:
            ii = torch.as_tensor(np.asarray(idx), device=data_t.device, dtype=torch.long)
            sub, c = data_t.index_select(1, ii), centroids_t[k].index_select(0, ii)
        sub = sub - sub.mean(dim=1, keepdim=True)
        sub = sub / torch.clamp(torch.linalg.norm(sub, dim=1, keepdim=True), min=1e-12)
        c = c - c.mean()
        c = c / torch.clamp(torch.linalg.norm(c), min=1e-12)
        r = torch.nan_to_num(sub @ c, nan=-1.0).to(torch.float32)
        improved = r > best_corr
        best_corr = torch.where(improved, r, best_corr)
        best_idx = torch.where(improved, torch.tensor(k, dtype=torch.int32,
                                                      device=data_t.device), best_idx)
        if bar is not None:
            bar.update(1)
    return best_idx, best_corr


#: Scores at or below this are treated as "nothing measured" when locating the peak.
MARKER_SCORE_FLOOR = 1e-4


def half_sample_mode(x):
    """Locate the peak of a sample by the half-sample mode of Robertson and Cryer, 1974.

    The method repeatedly keeps the narrowest half of the sorted values, so it converges
    on the densest region and ignores both tails. It is used here instead of a mean, a
    kernel density argmax or a two-component mixture, because the marker-score
    distributions fail all three. They are skewed, so a mean lands to the left of the
    peak. They carry a spike of exact zeros at the clip floor, which a kernel density
    argmax mistakes for the mode whenever a celltype has a single sparse marker. And they
    are typically three or four modal rather than two, so a two-component fit has no
    principled component to threshold on.

    The function reports a value for any input. That is the point, and also the caveat:
    the value is meaningful only when the majority of the sample is the population you
    meant. See :func:`marker_score_cutoffs`, which flags the two ways that assumption
    fails.

    :param x: The sample. It is flattened before use.
    :returns: The located peak, or not-a-number for an empty sample.
    :rtype: float
    """
    x = np.sort(np.asarray(x, dtype=np.float64).ravel())
    if x.size == 0:
        return float('nan')
    while x.size > 3:
        n = x.size
        h = (n + 1) // 2
        w = x[h - 1:] - x[:n - h + 1]
        i = int(np.argmin(w))
        x = x[i:i + h]
    if x.size == 3:
        d1, d2 = x[1] - x[0], x[2] - x[1]
        if d1 == d2:
            return float(x[1])
        return float(x[:2].mean() if d1 < d2 else x[1:].mean())
    return float(np.mean(x))


def marker_score_cutoffs(celltype_maps, marker_score, frac, *, min_voxels=300,
                         floor=MARKER_SCORE_FLOOR):
    """Return a marker-score threshold per celltype, set at a fraction of its own peak.

    Expression level differs by an order of magnitude between celltypes. Measured on the
    mouse kidney Xenium panel the peak was 0.156 for the cortical thick ascending limb
    against 0.014 for the parietal epithelium, a spread of eleven times, so one global cut
    is meaningless. Each celltype's own peak is located with :func:`half_sample_mode` on
    the base ten logarithm of the score, and everything below ``frac`` times that peak is
    dropped.

    The accompanying notes name the celltypes whose peak should not be trusted. The first
    case is more than half the voxels sitting on the clip floor, where the peak is the
    floor and the cutoff degenerates to zero, meaning no filtering at all. The second is
    the peak landing in the bottom 15 percent or the top 85 percent of its own
    distribution, which is what a peak sitting on the spike of zeros, or on a lone cluster
    in the tail, looks like. Both are reported rather than silently corrected, because a
    celltype that trips them has a marker problem, and the honest response is to fix the
    markers rather than the cutoff.

    :param celltype_maps: Celltype identifiers on the mapping grid.
    :param marker_score: The per-voxel score, on the same grid.
    :param frac: Fraction of the peak below which a voxel is cut.
    :type frac: float
    :param min_voxels: A celltype with fewer voxels than this is given no cut.
    :type min_voxels: int
    :param floor: Scores at or below this are treated as nothing measured when the peak is
        located.
    :type floor: float
    :returns: A pair. The first element maps celltype identifier to its cut. The second is
        a list of ``(celltype id, voxel count, message)`` triples naming the untrustworthy
        peaks.
    :rtype: tuple
    """
    cm = np.asarray(celltype_maps)
    ms = np.asarray(marker_score, dtype=np.float64)
    cutoffs, notes = {}, []
    for k in np.unique(cm):
        if k < 0:
            continue
        m = cm == k
        n = int(m.sum())
        if n < int(min_voxels):
            continue
        v = ms[m]
        if not np.isfinite(v).any() or np.nanmax(v) <= 0:
            continue
        x = np.log10(np.clip(v, floor, None))
        peak = half_sample_mode(x)
        if not np.isfinite(peak):
            continue
        cutoffs[int(k)] = float(10.0 ** peak * float(frac))
        on_floor = float((v <= floor).mean())
        q = float((x < peak).mean())
        if on_floor >= 0.5:
            notes.append((int(k), n, f'{on_floor:.0%} of voxels on the score floor '
                                     '- peak is the floor, cutoff does not filter'))
        elif q < 0.15 or q > 0.85:
            notes.append((int(k), n, f'peak sits at quantile {q:.2f} of its own '
                                     'distribution - not a central peak'))
    return cutoffs, notes


def density_cutoffs(celltype_maps, vf_norm, frac, *, mask=None, depth_normalize=True,
                    min_voxels=300):
    """Return a density threshold per celltype, set at a fraction of its own peak.

    This applies the rule of :func:`marker_score_cutoffs` to ``vf_norm`` instead of the
    marker score, which is what lets it reach every celltype. A marker score exists only
    for the celltypes the annotation gave a marker set to, which was 10 of 28 on the mouse
    kidney Xenium panel and 8 percent of the mapped volume, whereas a density exists for
    all of them. Densities differ between celltypes as well, with peaks spanning a factor
    of 3.4 on that panel, from 0.058 for the parietal epithelium to 0.197 for the S3
    segment, so the cut is made per celltype. It is the half-sample-mode peak of the base
    ten logarithm of ``vf_norm`` over that celltype's voxels, multiplied by ``frac``. What
    falls below it is the kernel density halo around a cell rather than the cell itself.

    Depth normalization matters on a three-dimensional map. Density falls off with depth:
    on that section the plane median ran from 1.00 in the middle of the tissue to between
    0.05 and 0.08 at its top and bottom. One number per celltype therefore removes the
    shallow and deep planes wholesale, keeping 2 percent of the mapped voxels at plane 7
    against 81 percent at plane 13. Each plane's density is instead divided first by that
    plane's median over the selected voxels, taken relative to the densest plane, and both
    the peak and the cut are computed on the normalized density. The fraction kept per
    plane then stays flat, between 79 and 82 percent across the tissue. A single-plane map
    is unaffected.

    :param celltype_maps: Celltype identifiers of shape ``(x, y)`` or ``(x, y, z)``, where
        minus one means unmapped.
    :param vf_norm: Density on the same grid.
    :param frac: Fraction of the peak below which a voxel is cut.
    :type frac: float
    :param mask: Optional boolean array. Only the voxels it selects enter the peaks and
        the plane medians.
    :param depth_normalize: Whether to divide by the plane median first. This has an
        effect on three-dimensional maps only.
    :type depth_normalize: bool
    :param min_voxels: A celltype with fewer selected voxels than this is given no cut.
    :type min_voxels: int
    :returns: A pair. The first element maps celltype identifier to its cut on the
        normalized density. The second is the per-plane factor, of shape ``(nz,)``. A voxel
        of celltype k in plane z is below the cut when ``vf_norm / plane_factor[z]`` is
        less than ``cutoffs[k]``. The factor is all ones when not normalizing by depth, and
        not-a-number for a plane holding no selected voxel.
    :rtype: tuple
    :raises ValueError: If ``vf_norm`` and ``celltype_maps`` are not on the same grid.
    """
    cm = np.asarray(celltype_maps)
    vf = np.asarray(vf_norm, dtype=np.float32)
    if vf.shape != cm.shape:
        raise ValueError(f'vf_norm {vf.shape} and celltype_maps {cm.shape} are not on the '
                         f'same grid')
    sel = (cm >= 0) & (vf > 0)
    if mask is not None:
        sel &= np.asarray(mask, dtype=bool)
    nz = cm.shape[2] if cm.ndim == 3 else 1
    factor = np.ones(nz, dtype=np.float64)
    vfn = vf
    if depth_normalize and cm.ndim == 3 and nz > 1:
        med = np.full(nz, np.nan)
        for z in range(nz):
            s = sel[..., z]
            if s.any():
                med[z] = float(np.median(vf[..., z][s]))
        if np.isfinite(med).any() and np.nanmax(med) > 0:
            factor = med / np.nanmax(med)
            vfn = vf / np.nan_to_num(factor, nan=1.0).astype(np.float32)[None, None, :]
    labels = cm[sel]
    values = vfn[sel]
    del sel
    counts = np.bincount(labels.astype(np.int64)) if labels.size else np.zeros(0, np.int64)
    cutoffs = {}
    for k in np.nonzero(counts >= int(min_voxels))[0]:
        v = values[labels == k]
        peak = half_sample_mode(np.log10(v))
        if np.isfinite(peak):
            cutoffs[int(k)] = float(10.0 ** peak * float(frac))
    return cutoffs, factor
