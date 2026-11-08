"""Gaussian naive Bayes celltype mapping.

This is an alternative to the centroid-correlation mapper
(:func:`ssaft._count.map_corr`). A Gaussian naive Bayes classifier, from cuML on the
GPU and from scikit-learn on the CPU, is fitted on the labeled
``normalized_vectors``. All of them are used, streamed through ``partial_fit`` so
that the full matrix is never resident, and every voxel of the field is then
classified by its posterior. Because the model is generative, with one Gaussian per
class, each class keeps a non-empty decision region. That is what preserves the rare
cell types which the discriminative correlation argmax can squeeze out.

Both regimes are supported, mirroring the correlation mapper. On the count and blur
path, :func:`map_gnb` reads the stored three-dimensional ``vf_normalized`` and
classifies each voxel, which is the array-based counterpart of :func:`map_corr`. On
the transcript and imaging path, ``_map_celltypes_cpu`` and ``_map_celltypes_gpu``
fit a model here, on the subset of genes the two sides share, and score each kernel
density chunk as it is produced.

Vectors are row-normalized, by mean-centering and then scaling to unit L2 norm,
before both fit and predict. That is the same Pearson space the correlation mapper
works in, so the two mappers are directly comparable.
"""
import os

import numpy as np


def _row_normalize(A, xp):
    """Mean-center each row and scale it to unit L2 norm.

    This puts the rows in the same space the correlation mapper uses, so that a dot
    product between two of them is a Pearson correlation. A row of zero norm is left
    unscaled rather than divided by zero.

    :param A: The matrix to normalize, with one observation per row.
    :param xp: The array module that ``A`` belongs to, either numpy or cupy.
    :returns: The normalized matrix, in the same array module.
    """
    A = A - A.mean(axis=1, keepdims=True)
    n = xp.linalg.norm(A, axis=1, keepdims=True)
    return A / xp.where(n == 0, xp.asarray(1.0, dtype=A.dtype), n)


def _resolve_backend(gpu_device, force_cpu):
    """Choose the Gaussian naive Bayes implementation and array module to use.

    The GPU pair of cuML and cupy is preferred, and the CPU pair of scikit-learn and
    numpy is the fallback. The GPU is skipped when it is turned off by the caller, when
    either GPU package is missing, or when no CUDA device is visible.

    :param gpu_device: The device to use, either an integer or a string such as
        ``'cuda:1'``. A string without an index means device zero.
    :param force_cpu: Whether to skip the GPU even when it is available.
    :type force_cpu: bool
    :returns: A tuple of the classifier class, the array module, a flag saying whether
        the GPU was chosen, and the integer device index, which is ``None`` on the CPU.
    :rtype: tuple
    """
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
    """Fit a Gaussian naive Bayes model on every labeled vector in the store.

    The vectors are streamed in chunks and no subsample is taken.

    :param zg: The zarr group holding ``normalized_vectors``, of shape ``(L, g)``, and
        ``cluster_labels``, of shape ``(L,)``.
    :param gpu_device: The device to fit on, passed to :func:`_resolve_backend`.
    :param force_cpu: Whether to force the scikit-learn backend.
    :type force_cpu: bool
    :param row_normalize: Whether to put each vector in Pearson space before fitting.
    :type row_normalize: bool
    :param fit_chunk: Number of rows read per streaming step.
    :type fit_chunk: int
    :param var_smoothing: Variance smoothing passed to the classifier.
    :type var_smoothing: float
    :param gene_indices: An optional column subset of the vectors. The transcript path
        sets it so that the classifier's genes match the shared gene axis of the kernel
        density field.
    :returns: A triple of the fitted classifier, the sorted celltype identifiers, and the
        backend tuple ``(xp, use_gpu, device)`` that :func:`predict_gnb` expects.
    :rtype: tuple
    """
    GNB, xp, use_gpu, dev = _resolve_backend(gpu_device, force_cpu)
    nv = zg['normalized_vectors']
    y_all = np.asarray(zg['cluster_labels'][:])
    classes = np.unique(y_all[y_all >= 0])
    L = nv.shape[0]
    gi = None if gene_indices is None else np.asarray(gene_indices)

    def _run():
        """Stream the labeled vectors through ``partial_fit`` and return the model.

        :returns: The fitted classifier and the number of labeled rows it saw.
        :rtype: tuple
        """
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
    """Fit a Gaussian naive Bayes model on the annotated h5ad itself.

    The ground truth is ``obs[column]``, meaning the labels CeLLMan finalized, rather
    than the store's ``cluster_labels``. The features are ``gene_names``, meaning the
    marker panel from ``celltype_markers.yaml`` rather than the whole matrix. Restricting
    the features is the point: Gaussian naive Bayes weights every gene equally, so
    hundreds of genes that are flat across celltypes would dilute the handful that
    actually separate them.

    Labels are resolved to celltype identifiers by the same rule the centroid builder
    uses, in :func:`ssaft._mapping_inputs._build_units`. There is one class per named
    label, ``Unknown`` is split per source cluster into ``unknown1`` to ``unknownN``, and
    Background is dropped. The identifiers therefore line up with ``celltype_names``,
    with ``centroids`` and with the colour palette, which is what keeps the
    ``celltype_maps`` that get written readable by everything else.

    This fit always uses scikit-learn and never cuML. It runs once on the annotated
    cells rather than once per voxel, so it is not the bottleneck, and the reference
    implementation is worth more here than the GPU. Prediction still honours whatever
    backend the caller asks for.

    :param h5ad_path: The annotated h5ad written by CeLLMan.
    :param gene_names: The feature genes, in the exact order the voxel vectors carry them.
    :param column: The ``obs`` column holding the finalized labels.
    :type column: str
    :param leiden_column: The ``obs`` column holding the source cluster, used to split
        ``Unknown``. A missing column is treated as a single cluster.
    :type leiden_column: str
    :param row_normalize: Whether to put each cell in Pearson space before fitting.
    :type row_normalize: bool
    :param fit_chunk: Number of cells read per streaming step.
    :type fit_chunk: int
    :param var_smoothing: Variance smoothing passed to the classifier.
    :type var_smoothing: float
    :param verbose: Whether to print a summary of the fit.
    :type verbose: bool
    :returns: The same triple as :func:`fit_gnb`.
    :rtype: tuple
    :raises ValueError: If the label column is absent, if any marker gene is missing from
        the h5ad, or if no callable celltype is present.
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
    """Classify a matrix of vectors and return celltype identifiers and confidences.

    The confidence is the posterior probability of the top class. The ``classes`` array
    maps the index of the argmax back to the celltype identifier; both cuML and
    scikit-learn keep ``classes_`` sorted, so the two agree.

    Prediction proceeds in sub-chunks of ``batch`` rows. ``GaussianNB.predict_proba``
    builds an array of shape ``(n, n_genes)`` for every class, so scoring a whole field
    chunk at once, which can be hundreds of thousands of voxels by thousands of genes,
    exhausts GPU memory. The batch size bounds that transient, and can be lowered on a
    small card.

    :param clf: The fitted classifier.
    :param classes: The sorted celltype identifiers the classifier was fitted on.
    :param V: The vectors to classify, of shape ``(n, g)``, as a numpy array.
    :param backend: The backend triple returned by the fit.
    :param row_normalize: Whether to put each vector in Pearson space before predicting.
        This must match what was done during the fit.
    :type row_normalize: bool
    :param batch: Number of rows scored per sub-chunk.
    :type batch: int
    :returns: A pair of arrays, the celltype identifiers as int32 and the top posterior
        probabilities as float32.
    :rtype: tuple of numpy.ndarray
    """
    xp, use_gpu, dev = backend
    n = int(V.shape[0])
    if n == 0:
        return np.empty(0, np.int32), np.empty(0, np.float32)

    def _one(Vb):
        """Score one sub-chunk of rows and bring the result back to the host.

        :param Vb: The rows of this sub-chunk.
        :returns: The argmax class index and the top posterior probability per row.
        :rtype: tuple of numpy.ndarray
        """
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
        """Score every sub-chunk in turn and concatenate the results.

        :returns: The argmax class index and the top posterior probability for all rows.
        :rtype: tuple of numpy.ndarray
        """
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
    """Map the count and blur field to celltypes with Gaussian naive Bayes.

    This is the counterpart of :func:`map_corr`. The model is fitted on the labeled
    ``normalized_vectors`` and then every ``vf_normalized`` voxel whose ``vf_norm``
    exceeds the threshold is classified.

    The label raster has the same layout as the one the correlation mapper writes, under
    ``celltype_maps``, but the score is a posterior probability. It is therefore written
    to ``max_probabilities`` rather than ``max_correlations``, and ``filter_celltypemaps``
    thresholds it with ``min_p``. Any ``max_correlations`` left behind by an earlier
    correlation run is dropped.

    :param analysis: The analysis whose dataset and store are read and written.
    :param centroids: Accepted for signature parity with :func:`map_corr` and otherwise
        unused, because the model fits its own per-class Gaussians from the labeled
        vectors.
    :param gpu_device: The device to use for the fit and the prediction.
    :param force_cpu: Whether to force the scikit-learn backend.
    :type force_cpu: bool
    :param chunk_h: Number of raster rows processed per step.
    :type chunk_h: int
    :param vf_norm_threshold: Density below which a voxel is left unmapped. Resolved from
        the store when ``None``.
    :param row_normalize: Whether to put the vectors in Pearson space.
    :type row_normalize: bool
    :param var_smoothing: Variance smoothing passed to the classifier.
    :type var_smoothing: float
    :param fit_chunk: Number of rows read per streaming step of the fit.
    :type fit_chunk: int
    :param overwrite: Whether to replace existing map arrays in the store.
    :type overwrite: bool
    :param gene_indices: Positions into the store's gene axis to keep, where ``None``
        keeps all of them. ``map_celltypes(marker_genes_path=...)`` sets this to the
        marker panel so that the classifier is not diluted by genes no celltype was
        called on.
    :param h5ad: An annotated h5ad to fit on instead, through :func:`fit_gnb_h5ad`, which
        takes ``obs[column]`` as ground truth and always uses scikit-learn. Passing
        ``None`` falls back to the store's labeled ``normalized_vectors``, which encode
        the same labels.
    :param column: The ``obs`` column holding the labels, used with ``h5ad``.
    :type column: str
    :param gene_names: Names of the kept genes, in the order of ``gene_indices``. This is
        required with ``h5ad`` so that the h5ad columns can be matched to the field's gene
        axis.
    :returns: Whatever ``_write_maps`` returns after writing the label raster and the
        posterior scores into the store.
    :raises ValueError: If ``h5ad`` is given without ``gene_names``.
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
