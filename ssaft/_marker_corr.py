"""Per-celltype marker-restricted Pearson correlation.

Same correlation as always — only the gene axis changes per celltype. The default
scores every celltype on the SAME axis: a voxel's whole vector against a centroid's
whole vector. That axis is dominated by genes which are flat across celltypes, so a
rare celltype defined by three genes is judged mostly on the ~60 genes it shares
with everything else.

With a marker panel (``map_celltypes(marker_genes_path=...)``) each celltype is
correlated ONLY on its own markers — its node's markers in the
``celltype_markers.yaml`` tree plus every ancestor's, since the Cell-Ontology is
normalized delta-only (see :meth:`~ssaft.Analysis.read_marker_genes`). So
``glomerular endothelial cell`` is scored on ``Ehd3, Podxl, Icam2`` *and* the
``endothelial cell`` markers above it, while ``proximal tubule cell`` is scored on
its own.

The centroids themselves are still built over the union of all marker genes: one
shared gene axis is what makes the celltypes comparable at all, and the marker set
only picks which of its columns a given celltype is judged on.

Marker sets that are too small
------------------------------
Pearson over ``n`` points needs ``n >= 3`` to mean anything: **over 2 points it is
always exactly ±1**, whatever the values. A celltype with 2 on-panel markers
therefore scores ``r = 1.0`` on every voxel where its two markers happen to be
ordered like its centroid's, wins those voxels outright, and — worse — is immune to
``filter_celltypemaps(min_r=...)``, since no threshold below 1 can remove it.

Measured on the mouse-kidney Xenium panel, where ``proximal tubule cell`` had only
``Epcam, Slc5a12`` on panel: 23.5% of its cells scored exactly 1.0, and 29.4% of all
cells ended up saturated at ``|r| > 0.999``.

So a celltype with fewer than :data:`MIN_MARKER_GENES` markers on the panel falls
back to the full axis and is named in a warning. That keeps the score filterable,
but it is a guard, not a fix — the fix is for the annotation to emit more on-panel
markers for that celltype in ``celltype_markers.yaml``.

The fallback belongs to the correlation and nowhere else. :func:`marker_score` is a
mean over the marker genes, and a mean over one gene is neither degenerate nor
unfilterable — routing it through the fallback would score that celltype 0, which
:func:`marker_score_cutoffs` then skips, leaving it with no cut at all. So the lda
path builds its score on the fallback-free OWN sets (``min_genes=0``) instead.

Why a loop and not one masked GEMM
----------------------------------
A masked correlation can be batched into three ``(N,G)x(G,K)`` matmuls via the
computational formula for covariance. It is not worth it here: marker sets are
small (a handful of genes out of hundreds), so gathering ``D[:, idx_k]`` costs
``N x sum(n_k)`` — well under the ``N x G x K`` of the batched form — and the
two-pass centering below is exact, where the computational formula loses three
digits in float32 on nearly-constant marker sets.
"""
import numpy as np

__all__ = ['marker_index_sets', 'masked_pearson_max', 'masked_pearson_max_torch',
           'marker_score', 'half_sample_mode', 'marker_score_cutoffs',
           'density_cutoffs', 'MIN_MARKER_GENES']

#: Fewest on-panel markers a celltype needs before it is scored on its own axis.
#: Below this the correlation is degenerate (n=2 is always ±1) and unfilterable.
MIN_MARKER_GENES = 3


def marker_score(density, labels, marker_idx):
    """``(n,)`` — each voxel's mean density over ITS OWN celltype's marker genes.

    The quantity a celltype is present *enough* to be called, as opposed to
    ``vf_norm``, which sums all genes and so measures how transcript-rich the
    neighbourhood is regardless of which celltype won. Expression level differs by
    an order of magnitude between celltypes, so the comparison only means anything
    within a celltype — hence one score per voxel, taken over the winner's markers,
    and a threshold set per celltype downstream.

    :param density: ``(n, G)`` absolute densities (molecules per µm^ndim), on the
        same gene axis ``marker_idx`` indexes.
    :param labels: ``(n,)`` celltype id per voxel; ``-1`` scores 0.
    :param marker_idx: ``[idx_0 ... idx_{K-1}]`` from :func:`marker_index_sets`;
        ``None`` for a celltype means "no marker set", scored 0 so the filter
        leaves it alone.
    :returns: ``(n,)`` float32.
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
    """``[idx_0, ..., idx_{K-1}]`` — gene positions each celltype is correlated on.

    ``idx_k`` indexes into ``gene_names``, i.e. the gene axis actually being
    mapped (after any ``exclude_gene_indices`` removal), so callers can apply it
    to their data chunk directly.

    ``None`` means "score on the whole axis". A celltype gets it when

    * it has no entry in ``marker_genes`` — the honest fallback for the
      annotation's ``unknown1..N`` classes, which were never defined by markers;
    * fewer than ``min_genes`` of its markers are on the panel — see the module
      docstring on why a 2-gene correlation cannot be thresholded.
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
    """Pearson of every row of ``sub`` (n, g) against the vector ``cen`` (g,)."""
    sub = sub - sub.mean(axis=1, keepdims=True)
    sub /= np.maximum(np.linalg.norm(sub, axis=1, keepdims=True), 1e-12)
    cen = cen - cen.mean()
    cen = cen / max(float(np.linalg.norm(cen)), 1e-12)
    return sub @ cen


def masked_pearson_max(data, centroids, index_sets, *, bar=None):
    """Best celltype per row, each correlated on its own marker genes (NumPy).

    :param data: ``(n_voxels, n_genes)`` voxel vectors.
    :param centroids: ``(k, n_genes)`` centroids on the same axis.
    :param index_sets: from :func:`marker_index_sets`; ``None`` = full axis.
    :returns: ``(best_idx int32 (n,), best_corr float32 (n,))``; ``-1`` where no
        celltype scored (empty input).
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
    """torch counterpart of :func:`masked_pearson_max`; keeps everything on device.

    ``data_t`` is ``(n, g)``, ``centroids_t`` is ``(k, g)``.
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
    """Peak of ``x`` by the half-sample mode (Robertson & Cryer 1974).

    Repeatedly keeps the NARROWEST half of the sorted values, so it converges on the
    densest region and ignores both tails. Used instead of a mean, a KDE argmax or a
    two-component mixture because the marker-score distributions here fail all three:
    they are skewed (a mean lands left of the peak), they carry a spike of exact
    zeros at the clip floor that a KDE argmax mistakes for the mode when a celltype
    has a single sparse marker, and they are typically 3-4 modal rather than 2, so a
    two-component fit has no principled component to threshold on.

    It reports a value for any input — that is the point, and also the caveat: the
    value is only meaningful when the majority of the sample is the population you
    meant. See :func:`marker_score_cutoffs`, which flags the two ways that fails.
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
    """``(cutoffs, notes)`` — a marker-score threshold per celltype, at ``frac`` x its peak.

    Expression level differs by an order of magnitude between celltypes (measured on
    the mouse-kidney Xenium panel: peak 0.156 for cortical TAL vs 0.014 for the
    parietal epithelium, an 11x spread), so one global cut is meaningless. Each
    celltype's own peak is located with :func:`half_sample_mode` on ``log10`` of the
    score, and everything below ``frac`` x that peak is dropped.

    ``notes`` names the celltypes whose peak should not be trusted:

    * more than half the voxels sitting on the clip floor — the peak IS the floor and
      the cutoff degenerates to zero, i.e. no filtering;
    * the peak landing in the bottom 15% or top 85% of its own distribution, which is
      what a peak sitting on the zero spike (or on a lone tail cluster) looks like.

    Both are reported rather than silently corrected: a celltype that trips them has
    a marker problem, and the honest response is to fix the markers, not the cutoff.
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
    """``(cutoffs, plane_factor)`` — a density threshold per celltype, at ``frac`` x its peak.

    The rule of :func:`marker_score_cutoffs`, applied to ``vf_norm`` instead of the marker
    score, which is what lets it reach every celltype: a marker score exists only for the
    celltypes the annotation gave a marker set (10 of 28 on the mouse-kidney Xenium
    panel, 8% of the mapped volume), a density exists for all of them. Densities differ
    between celltypes as well (peaks span 3.4x on that panel, 0.058 for the parietal
    epithelium to 0.197 for S3), so the cut is per celltype: the half-sample-mode peak of
    ``log10(vf_norm)`` over that celltype's voxels, times ``frac``. What falls below it
    is the KDE halo around a cell rather than the cell.

    ``depth_normalize`` matters on a 3D map. Density falls off with depth — on that
    section the plane median runs from 1.00 in the middle of the tissue to 0.05-0.08 at
    its top and bottom — so one number per celltype removes the shallow and deep planes
    wholesale (2% of the mapped voxels kept at z=7 against 81% at z=13). Each plane's
    density is therefore divided first by that plane's median over the selected voxels,
    relative to the densest plane, and the peak and the cut are taken on the normalised
    density; the fraction kept per plane then stays flat (79-82% across the tissue). A
    single-plane map is unchanged by it.

    :param celltype_maps: ``(x, y[, z])`` celltype ids, ``-1`` = unmapped.
    :param vf_norm: density on the same grid.
    :param frac: fraction of the peak below which a voxel is cut.
    :param mask: optional boolean array; only these voxels enter the peaks and the plane
        medians.
    :param depth_normalize: divide by the plane median first (3D maps only).
    :param min_voxels: a celltype with fewer selected voxels gets no cut.
    :returns: ``({celltype id: cut on the normalised density}, plane_factor (nz,))``. A
        voxel of celltype ``k`` in plane ``z`` is below the cut when
        ``vf_norm / plane_factor[z] < cutoffs[k]``. ``plane_factor`` is all ones when
        not depth-normalising, and ``nan`` for a plane with no selected voxel.
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
