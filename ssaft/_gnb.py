"""Gaussian naive Bayes celltype mapping for ssaft.

An alternative to the centroid-correlation mapper (:func:`ssaft._count.map_corr`).
A :class:`~sklearn.naive_bayes.GaussianNB` (cuML on GPU, scikit-learn on CPU) is fitted on
the labeled ``normalized_vectors`` — ALL of them, streamed via ``partial_fit`` so the full
matrix is never resident — and every field voxel is classified by its posterior. Being
GENERATIVE (one Gaussian per class), each class keeps a non-empty decision region, so GNB
preserves rare cell types that the discriminative correlation argmax can squeeze out.

Both regimes are supported, mirroring the correlation mapper:

  * **count / blur** — :func:`map_gnb` reads the stored 3D ``vf_normalized`` and classifies
    each voxel (the array-based counterpart of :func:`map_corr`).
  * **transcript / imaging** — ``_map_celltypes_cpu`` / ``_map_celltypes_gpu`` fit a model
    here (on the shared-gene subset) and score each KDE chunk on the fly (the KDE
    counterpart, like the correlation transcript path).

Vectors are row-normalized (mean-center + L2) before fit and predict — the same Pearson
space the correlation mapper uses — so the two are directly comparable.
"""
import os

import numpy as np


def _row_normalize(A, xp):
    """Per-row mean-center + L2 (the corr Pearson space); ``A`` on ``xp`` (numpy/cupy)."""
    A = A - A.mean(axis=1, keepdims=True)
    n = xp.linalg.norm(A, axis=1, keepdims=True)
    return A / xp.where(n == 0, xp.asarray(1.0, dtype=A.dtype), n)


def _resolve_backend(gpu_device, force_cpu):
    """``(GaussianNB, xp, use_gpu, device_int)`` — cuML+cupy on GPU, else scikit-learn+numpy."""
    if not force_cpu:
        try:
            import cupy as cp
            from cuml.naive_bayes import GaussianNB as _GNB
            if cp.cuda.runtime.getDeviceCount() > 0:
                dev = gpu_device
                if isinstance(dev, str):
                    dev = int(dev.split(':')[1]) if ':' in dev else 0
                return _GNB, cp, True, int(dev)
        except Exception:
            pass
    from sklearn.naive_bayes import GaussianNB as _GNB
    return _GNB, np, False, None


def fit_gnb(zg, *, gpu_device='cuda', force_cpu=False, row_normalize=True,
            fit_chunk=100_000, var_smoothing=1e-9, gene_indices=None):
    """Fit a GaussianNB on ALL labeled ``normalized_vectors`` (streamed; no subsample).

    :param zg: zarr group holding ``normalized_vectors`` (L, g) + ``cluster_labels`` (L,).
    :param gene_indices: optional column subset of the vectors — used by the transcript
        path so the classifier's genes match the KDE field's shared-gene axis.
    :returns: ``(clf, classes, backend)`` where ``classes`` are the sorted celltype ids and
        ``backend = (xp, use_gpu, device)`` for :func:`predict_gnb`.
    """
    GNB, xp, use_gpu, dev = _resolve_backend(gpu_device, force_cpu)
    nv = zg['normalized_vectors']
    y_all = np.asarray(zg['cluster_labels'][:])
    classes = np.unique(y_all[y_all >= 0])
    L = nv.shape[0]
    gi = None if gene_indices is None else np.asarray(gene_indices)

    def _run():
        clf = GNB(var_smoothing=var_smoothing)
        cls_dev = xp.asarray(classes)
        n_used = 0
        for s in range(0, L, fit_chunk):
            e = min(s + fit_chunk, L)
            yc = y_all[s:e]
            m = yc >= 0
            if not m.any():
                continue
            Xc = np.asarray(nv[s:e])[m]
            if gi is not None:
                Xc = Xc[:, gi]
            Xc = xp.asarray(Xc, dtype=xp.float32)
            if row_normalize:
                Xc = _row_normalize(Xc, xp)
            clf.partial_fit(Xc, xp.asarray(yc[m]), classes=cls_dev)
            n_used += int(m.sum())
        return clf, n_used

    if use_gpu:
        import cupy as cp
        with cp.cuda.Device(dev):
            clf, n_used = _run()
    else:
        clf, n_used = _run()
    print(f"[gnb] fit GaussianNB on {n_used} labeled vectors, {len(classes)} classes "
          f"({'GPU cuML' if use_gpu else 'CPU sklearn'}, streamed, no subsample)")
    return clf, classes, (xp, use_gpu, dev)


def fit_gnb_h5ad(h5ad_path, *, gene_names, column='cellman_annotation',
                 leiden_column='leiden', row_normalize=True, fit_chunk=100_000,
                 var_smoothing=1e-9, verbose=True):
    """Fit :class:`sklearn.naive_bayes.GaussianNB` on the annotation h5ad itself.

    The ground truth is ``obs[column]`` — the labels CeLLMan finalized — not the
    store's ``cluster_labels``, and the features are ``gene_names``, i.e. the marker
    panel from ``celltype_markers.yaml`` rather than the whole matrix. Restricting the
    features is the point: GaussianNB weights every gene equally, so hundreds of genes
    that are flat across celltypes dilute the handful that actually separate them.

    Labels are resolved to celltype ids with the SAME rule the centroid builder uses
    (:func:`ssaft._mapping_inputs._build_units`): one class per named label,
    ``Unknown`` split per source cluster into ``unknown1..N``, Background dropped.
    Ids therefore line up with ``celltype_names`` / ``centroids`` / the colour palette,
    which is what keeps the written ``celltype_maps`` readable by everything else.

    Always scikit-learn, never cuML: this fit runs once on the annotated cells (not per
    voxel), so it is not the bottleneck, and the reference implementation is worth more
    here than the GPU. Prediction still honours whatever backend the caller asks for.

    :param h5ad_path: annotated h5ad written by CeLLMan.
    :param gene_names: feature genes, in the exact order the voxel vectors carry them.
    :returns: ``(clf, classes, backend)`` — same triple as :func:`fit_gnb`.
    """
    import anndata as ad
    from sklearn.naive_bayes import GaussianNB
    from ._mapping_inputs import _build_units

    adata = ad.read_h5ad(h5ad_path, backed='r')
    try:
        if column not in adata.obs.columns:
            raise ValueError(f'column {column!r} not in obs '
                             f'(have: {list(adata.obs.columns)})')
        var = {str(g): i for i, g in enumerate(adata.var_names)}
        missing = [g for g in gene_names if g not in var]
        if missing:
            raise ValueError(f'{len(missing)} marker genes are not in {h5ad_path} '
                             f'(e.g. {missing[:5]}) — the h5ad and the store must share '
                             'a gene naming')
        cols = np.asarray([var[g] for g in gene_names], dtype=np.int64)

        celltype = adata.obs[column].astype(str).values
        if leiden_column in adata.obs.columns:
            import pandas as pd
            leiden = (pd.to_numeric(adata.obs[leiden_column].astype(str), errors='coerce')
                        .fillna(-1).astype(np.int64).values)
        else:
            leiden = np.zeros(adata.n_obs, dtype=np.int64)
        units = _build_units(celltype, leiden)
        if not units:
            raise ValueError(f'no callable celltype in obs[{column!r}]')
        y = np.full(adata.n_obs, -1, dtype=np.int64)
        for new_id, u in enumerate(units):
            y[u['mask']] = new_id
        classes = np.arange(len(units), dtype=np.int64)

        clf = GaussianNB(var_smoothing=var_smoothing)
        n_used = 0
        X = adata.X
        for s in range(0, adata.n_obs, fit_chunk):
            e = min(s + fit_chunk, adata.n_obs)
            m = y[s:e] >= 0
            if not m.any():
                continue
            Xc = X[s:e]
            Xc = Xc.toarray() if hasattr(Xc, 'toarray') else np.asarray(Xc)
            Xc = np.ascontiguousarray(Xc[m][:, cols], dtype=np.float32)
            if row_normalize:
                Xc = _row_normalize(Xc, np)
            clf.partial_fit(Xc, y[s:e][m], classes=classes)
            n_used += int(m.sum())
    finally:
        if getattr(adata, 'file', None) is not None:
            adata.file.close()

    if verbose:
        print(f"[gnb] fit sklearn GaussianNB on {n_used} cells from "
              f"{os.path.basename(h5ad_path)}: y = obs[{column!r}], "
              f"{len(classes)} classes, {len(gene_names)} marker genes")
    return clf, classes, (np, False, None)


def predict_gnb(clf, classes, V, backend, *, row_normalize=True, batch=20_000):
    """Classify ``(n, g)`` vectors ``V`` (numpy) -> ``(labels int32 celltype ids, conf float32)``.

    ``conf`` is the top-class posterior probability. ``classes`` maps the argmax index back
    to the celltype id (both cuML and scikit-learn keep ``classes_`` sorted, matching).

    Predicted in ``batch``-row sub-chunks: ``GaussianNB.predict_proba`` builds an
    ``(n, n_genes)`` array per class, so scoring a whole KDE/field chunk (hundreds of
    thousands of voxels x thousands of genes) at once OOMs the GPU. ``batch`` bounds the
    transient; lower it if a card is small.
    """
    xp, use_gpu, dev = backend
    n = int(V.shape[0])
    if n == 0:
        return np.empty(0, np.int32), np.empty(0, np.float32)

    def _one(Vb):
        Vx = xp.asarray(np.ascontiguousarray(Vb), dtype=xp.float32)
        if row_normalize:
            Vx = _row_normalize(Vx, xp)
        proba = clf.predict_proba(Vx)
        best = proba.argmax(axis=1)
        conf = proba.max(axis=1)
        if use_gpu:
            best, conf = best.get(), conf.get()
            del Vx, proba
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
        return np.asarray(best), np.asarray(conf, dtype=np.float32)

    def _loop():
        bs, cs = [], []
        for s in range(0, n, batch):
            b, c = _one(V[s:s + batch])
            bs.append(b); cs.append(c)
        return np.concatenate(bs), np.concatenate(cs)

    if use_gpu:
        import cupy as cp
        with cp.cuda.Device(dev):
            best, conf = _loop()
    else:
        best, conf = _loop()
    return classes[best].astype(np.int32), conf


def map_gnb(analysis, *, centroids=None, gpu_device='cuda', force_cpu=False,
            chunk_h=128, vf_norm_threshold=None, row_normalize=True,
            var_smoothing=1e-9, fit_chunk=100_000, overwrite=True,
            gene_indices=None, h5ad=None, column='cellman_annotation',
            gene_names=None):
    """Map the count/blur field to celltypes with GaussianNB (the GNB counterpart of
    :func:`map_corr`).

    Fits on the labeled ``normalized_vectors`` then classifies every ``vf_normalized`` voxel
    with ``vf_norm > threshold``. The label raster has the same layout as the correlation
    mapper (``celltype_maps``), but the score is a posterior probability, so it is written to
    ``max_probabilities`` (not ``max_correlations``) and ``filter_celltypemaps`` thresholds it
    with ``min_p``. Any ``max_correlations`` left by a previous ``corr`` run is dropped.

    :param centroids: accepted for signature parity with :func:`map_corr`; unused (GNB fits
        its own per-class Gaussians from the labeled vectors).
    :param gene_indices: positions into the store's gene axis to keep; ``None`` = all.
        Set by ``map_celltypes(marker_genes_path=...)`` to the marker panel, so the
        classifier is not diluted by genes no celltype was called on.
    :param h5ad: annotated h5ad to fit on (:func:`fit_gnb_h5ad`, scikit-learn, ground
        truth ``obs[column]``). ``None`` falls back to the store's labeled
        ``normalized_vectors``, which encode the same labels.
    :param gene_names: names of the kept genes, in ``gene_indices`` order; required
        with ``h5ad`` so the h5ad columns can be matched to the field's gene axis.
    """
    from tqdm.auto import tqdm
    from ._count import _write_maps

    zg = analysis.dataset.zarr_group
    ds = analysis.dataset
    vf = zg['vf_normalized']
    n_G, n_R, n_C = vf.shape
    norm = np.asarray(zg['vf_norm'][:])
    from ._count import resolve_vf_norm_threshold
    vf_norm_threshold = resolve_vf_norm_threshold(zg, vf_norm_threshold)
    sd = float(getattr(ds, 'sampling_distance', None)
               or zg.attrs.get('sampling_distance', zg.attrs.get('bin_size_um', 1.0)))

    if h5ad is not None:
        if gene_names is None:
            raise ValueError('map_gnb(h5ad=...) needs gene_names for the kept gene axis')
        clf, classes, backend = fit_gnb_h5ad(
            h5ad, gene_names=list(gene_names), column=column,
            row_normalize=row_normalize, fit_chunk=fit_chunk,
            var_smoothing=var_smoothing, verbose=analysis.verbose)
    else:
        clf, classes, backend = fit_gnb(zg, gpu_device=gpu_device, force_cpu=force_cpu,
                                        row_normalize=row_normalize, fit_chunk=fit_chunk,
                                        var_smoothing=var_smoothing,
                                        gene_indices=gene_indices)
    k = int(zg['centroids'].shape[0]) if 'centroids' in zg else len(classes)

    M = np.full((n_R, n_C), -1, dtype=np.int32)
    P = np.full((n_R, n_C), -1.0, dtype=np.float32)
    it = range(0, n_R, chunk_h)
    if analysis.verbose:
        it = tqdm(list(it), desc='map_celltypes (gnb)')
    for r0 in it:
        r1 = min(r0 + chunk_h, n_R)
        sig = norm[r0:r1] > vf_norm_threshold
        if not sig.any():
            continue
        h = r1 - r0
        block = np.asarray(vf[:, r0:r1, :]).reshape(n_G, h * n_C).T
        if gene_indices is not None:
            block = block[:, np.asarray(gene_indices, dtype=np.int64)]
        sig_flat = sig.reshape(-1)
        labels, conf = predict_gnb(clf, classes, block[sig_flat], backend,
                                   row_normalize=row_normalize)
        pos = np.where(sig_flat)[0]
        m_flat = M[r0:r1].reshape(-1); p_flat = P[r0:r1].reshape(-1)
        m_flat[pos] = labels
        p_flat[pos] = conf
        M[r0:r1] = m_flat.reshape(h, n_C)
        P[r0:r1] = p_flat.reshape(h, n_C)

    if analysis.verbose:
        _sig = int((norm > vf_norm_threshold).sum()); _tot = int((norm > 0).sum())
        print(f"[gnb] assigned {int((M >= 0).sum())} voxels (k={k}, "
              f"vf_norm>{vf_norm_threshold:.4g} keeps {_sig}/{_tot} = {_sig/max(_tot,1):.1%} of the field)")
    return _write_maps(zg, ds, M, P, sd, overwrite=overwrite,
                       score_key='max_probabilities')
