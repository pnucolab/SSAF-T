"""Build the inputs of ``map_celltypes`` from an annotated h5ad.

The inputs in question are ``cluster_labels`` and ``centroids``.
``map_celltypes`` calls :func:`build_centroids_from_h5ad` itself when the store holds
neither of them, and also whenever an h5ad is passed explicitly. The annotated h5ad,
which must be row-aligned with the store's ``normalized_vectors``, is grouped into
celltypes purely by the label string in ``obs[column]``.

A named celltype becomes one class per distinct label, so a different label always
means a different celltype and names carrying a state suffix stay separate. The label
``Unknown`` is instead grouped by the original cluster in ``obs[leiden_column]``, so
that each source cluster holding unknown vectors becomes its own class, named
``unknown1``, ``unknown2`` and so on. Unresolved populations are therefore kept
separable and are genuinely mapped, being real classes for both the correlation and
the naive Bayes mapper, rather than dropped. The labels ``Background`` and the
never-finalized subclustering labels drop to minus one, because they are not cells.

Each centroid is the mean of the member vectors after each has been scaled to unit L2
norm, which places them in Pearson correlation space. Only ``cluster_labels`` and
``centroids`` are written. The store's ``normalized_vectors`` is never modified, so
the operation is non-destructive and idempotent.
"""
import os

import anndata as ad
import numpy as np
import pandas as pd

DROP_LABELS = {'Background', 'needs_subclustering', 'unresolved_subclustering'}
UNKNOWN_LABEL = 'Unknown'


def _qualitative_pool(n):
    """Return a pool of at least ``n`` maximally distinct hex colours, as best it can.

    Celltypes are a nominal variable, so the palette must be qualitative. Sampling a
    continuous map such as ``jet`` hands neighbouring identifiers nearly the same colour,
    and with tens of classes the map becomes unreadable.

    Four sources are used in order of size, each already optimized for categorical
    separation. Up to 20 colours come from the matplotlib ``tab20`` map, de-paired so that
    consecutive identifiers get the saturated variants first, because ``tab20`` is stored
    as dark and light pairs of one hue and would otherwise give adjacent identifiers
    near-identical colours. Up to 28 come from scanpy's ``default_28``, the Zeileis
    palette, which is the same palette scanpy uses for a categorical ``obs`` column, so
    that the celltype map and any ``sc.pl.spatial`` of the annotated h5ad agree. Up to 102
    come from scanpy's ``default_102``, the Godsnot palette. Beyond that, ``default_102``
    is cycled with a lightness shift applied per cycle, so colours repeat in family but
    never exactly.

    :param n: Number of colours needed.
    :type n: int
    :returns: Hex colour strings, at least ``n`` of them.
    :rtype: list of str
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex, to_rgb

    if n <= 20:
        tab = plt.get_cmap("tab20").colors
        return [to_hex(tab[i]) for i in list(range(0, 20, 2)) + list(range(1, 20, 2))]
    try:
        from scanpy.plotting import palettes as _pal
        pool = list(_pal.default_28) if n <= 28 else list(_pal.default_102)
    except Exception:  # noqa: BLE001 — scanpy optional at this layer
        pool = [to_hex(c) for name in ("tab20", "tab20b", "tab20c")
                for c in plt.get_cmap(name).colors]
    if n <= len(pool):
        return pool
    out = []
    for cycle in range((n + len(pool) - 1) // len(pool)):
        f = 1.0 - 0.25 * (cycle % 3)
        out += [to_hex(tuple(min(1.0, max(0.0, ch * f)) for ch in to_rgb(c)))
                for c in pool]
    return out


def _uns_color_map(adata, column):
    """Read the annotation's own palette out of ``uns`` and key it by label.

    scanpy stores that list index-aligned with ``obs[column].cat.categories``, so the
    mapping is by name. Reading it positionally is the bug this function exists to
    prevent: the store orders celltypes by identifier, which is grouping order, while the
    categorical orders them alphabetically, so the two lists hold the same colours against
    different celltypes.

    :param adata: The annotated data, still open.
    :type adata: anndata.AnnData
    :param column: The ``obs`` column whose colours are wanted.
    :type column: str
    :returns: A mapping from label to hex colour, or ``None`` when the annotation carries
        no palette or when its length does not match the categories.
    :rtype: dict or None
    """
    key = f'{column}_colors'
    cols = adata.uns.get(key, None)
    if cols is None:
        return None
    col = adata.obs[column]
    cats = list(col.cat.categories) if hasattr(col, 'cat') else None
    if cats is None or len(cats) != len(cols):
        print(f'[map_celltypes] uns[{key!r}] has {len(cols)} colours for '
              f'{len(cats) if cats else "?"} categories — ignoring it')
        return None
    from matplotlib.colors import to_hex, to_rgba
    return {str(c): to_hex(to_rgba(x)) for c, x in zip(cats, cols)}


def _colors_from_names(n, name_colors, names, fresh_for=None):
    """Reuse the annotation's colour for each celltype name and mint new ones for the rest.

    An entry of ``fresh_for`` forces a new colour even when the name matches. That is what
    ``Unknown`` needs: the annotation calls every unassignable cell ``Unknown`` and gives
    it a single colour, but the store splits it per source cluster into ``unknown1`` and
    its successors. Handing all of them that one colour would make them indistinguishable
    on the map, so they are given fresh and mutually distinct colours instead.

    New colours are drawn from the qualitative pool, skipping anything the annotation has
    already used, so a minted colour never collides with a real celltype's.

    :param n: Number of celltypes.
    :type n: int
    :param name_colors: Mapping from label to hex colour, as carried by the annotation.
    :type name_colors: dict
    :param names: Celltype names in identifier order.
    :param fresh_for: Per-celltype flags forcing a newly minted colour.
    :returns: One hex colour per celltype, in identifier order.
    :rtype: list of str
    :raises ValueError: If the number of names does not match ``n``.
    """
    names = list(names or [])
    if len(names) != n:
        raise ValueError(f'{len(names)} names for {n} celltypes')
    fresh_for = list(fresh_for or [False] * n)
    taken = {c.lower() for c in name_colors.values()}
    spare = [c for c in _qualitative_pool(n + len(taken) + 8)
             if c.lower() not in taken]
    out, minted, unmatched = [], 0, []
    for i, nm in enumerate(names):
        if not fresh_for[i] and nm in name_colors:
            out.append(name_colors[nm])
            continue
        if not fresh_for[i]:
            unmatched.append(nm)
        c = spare[minted] if minted < len(spare) else _qualitative_pool(minted + 1)[minted]
        taken.add(c.lower())
        out.append(c)
        minted += 1
    print(f'[map_celltypes] palette from the h5ad: {n - minted}/{n} celltypes matched '
          f'by name, {minted} given new colours'
          + (f' (unmatched: {unmatched[:5]})' if unmatched else ''))
    return out


def celltype_colors(n_celltypes, cmap=None, *, name_colors=None, names=None,
                    fresh_for=None):
    """Return one hex colour per celltype, index-aligned with the celltype identifiers.

    The palette is fixed here, once, against the number of classes the mapping actually
    uses, which is ``centroids.shape[0]``. It is deliberately not fixed against
    ``celltype_maps.max()``, which counts only the classes that won at least one voxel.
    Everything downstream reads this one list, including the label raster, its RGB
    rendering, the annotated table's ``uns['cellman_annotation_colors']`` and any legend,
    so the same celltype cannot come out in different colours in different views.

    :param n_celltypes: Number of celltype classes, which is ``centroids.shape[0]``.
    :type n_celltypes: int
    :param cmap: Leaving this at ``None`` picks a qualitative palette sized to the class
        count, as described under :func:`_qualitative_pool`. Passing the name of a
        matplotlib colormap samples it continuously instead, for example ``'jet'``, which
        is legible only for a handful of classes. An explicit colormap overrides
        ``name_colors``.
    :type cmap: str or matplotlib.colors.Colormap or None
    :param name_colors: The mapping from label to hex colour carried by the annotation, in
        its ``uns[f'{column}_colors']``. When it is given and ``cmap`` is ``None``, every
        celltype whose name appears there keeps the colour it had in the annotation, so
        that the celltype map and an ``sc.pl.spatial`` of the h5ad agree cell type by cell
        type rather than merely drawing from the same list.
    :type name_colors: dict or None
    :param names: Celltype names in identifier order. This is required with
        ``name_colors``.
    :type names: list[str] or None
    :param fresh_for: Per-celltype flags forcing a newly minted colour even when the name
        matches, used for the split of a single ``Unknown`` into ``unknown1`` and its
        successors.
    :type fresh_for: list[bool] or None
    :returns: One hex colour per celltype.
    :rtype: list of str
    """
    n = int(n_celltypes)
    if cmap is None and name_colors:
        return _colors_from_names(n, name_colors, names, fresh_for)
    if cmap is None:
        return _qualitative_pool(n)[:n]
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex
    cm = plt.get_cmap(cmap)
    return [to_hex(cm(i / max(n - 1, 1))) for i in range(n)]


def compute_centroids(nv, labels, n_clusters, chunk=20000):
    """Compute each cluster's mean over its member vectors, scaled to unit L2 norm.

    :param nv: The store's ``normalized_vectors``, of shape ``(L, g)``. It is read in
        chunks.
    :type nv: array-like
    :param labels: Cluster index of each vector, of shape ``(L,)``, where minus one means
        the vector is dropped.
    :type labels: numpy.ndarray
    :param n_clusters: Number of clusters, which is the number of rows of the result.
    :type n_clusters: int
    :param chunk: Number of rows accumulated per step. Accumulation is in float64, so the
        result does not depend on this value.
    :type chunk: int
    :returns: A pair holding the centroids, of shape ``(k, g)`` and dtype float32, and the
        member counts, of shape ``(k,)`` and dtype int64.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    :raises ValueError: If any cluster ended up with no members.
    """
    n_genes = nv.shape[1]
    acc = np.zeros((n_clusters, n_genes), dtype=np.float64)
    counts = np.zeros(n_clusters, dtype=np.int64)
    for start in range(0, nv.shape[0], chunk):
        stop = min(start + chunk, nv.shape[0])
        lab = labels[start:stop]
        sel = lab >= 0
        if not sel.any():
            continue
        block = np.asarray(nv[start:stop])[sel].astype(np.float64, copy=False)
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        block = block / np.where(norms == 0, 1.0, norms)
        np.add.at(acc, lab[sel], block)
        np.add.at(counts, lab[sel], 1)
    if (counts == 0).any():
        raise ValueError(f'empty cluster(s): {np.flatnonzero(counts == 0).tolist()}')
    return (acc / counts[:, None]).astype(np.float32), counts


def _build_units(celltype, leiden):
    """Group vectors into an ordered list of celltype units, by label string.

    Each named label becomes one unit. The label ``Unknown`` becomes one unit per source
    cluster, named ``unknown1`` upwards in ascending order of cluster. The labels in
    ``DROP_LABELS`` are excluded, and their vectors stay at minus one.

    :param celltype: Label string of each vector, of shape ``(L,)``.
    :type celltype: numpy.ndarray
    :param leiden: Original cluster identifier of each vector, of shape ``(L,)``. It is
        needed only to split ``Unknown``.
    :type leiden: numpy.ndarray
    :returns: The units, each a dictionary holding the label, the member mask, whether it
        came from ``Unknown`` and its source cluster, ordered by original cluster and then
        by label.
    :rtype: list(dict)
    """
    units = []
    named = sorted(lab for lab in set(celltype.tolist())
                   if lab not in DROP_LABELS and lab != UNKNOWN_LABEL)
    for lab in named:
        mask = celltype == lab
        units.append(dict(label=lab, mask=mask, is_unknown=False,
                          source_cluster=-1, _sort=int(leiden[mask].min())))

    umask = celltype == UNKNOWN_LABEL
    if umask.any():
        for j, lc in enumerate(sorted({int(v) for v in leiden[umask]}), 1):
            units.append(dict(label=f'unknown{j}', mask=umask & (leiden == lc),
                              is_unknown=True, source_cluster=int(lc), _sort=int(lc)))

    units.sort(key=lambda u: (u['_sort'], u['label']))
    return units


def build_centroids_from_h5ad(z, h5ad_path, *, column='cellman_annotation',
                              leiden_column='leiden', chunk=20000,
                              store_dir=None, out_csv_name='celltype_index.csv',
                              cmap=None):
    """Group an annotated h5ad into celltypes and write the mapping inputs into a store.

    What is written into the open zarr group is ``cluster_labels``, ``centroids``,
    ``celltype_names`` and the colour palette. Grouping is by the label string in
    ``obs[column]``, as described in the module docstring: named labels become one class
    each, ``Unknown`` is split by original cluster into ``unknown1`` and its successors,
    and background or unfinalized labels drop to minus one. This is the function
    :meth:`Analysis.map_celltypes` invokes when the store lacks its inputs.

    :param z: An open zarr group in mode ``r+``. It must already hold
        ``normalized_vectors``, and it is where ``cluster_labels`` and ``centroids`` are
        written, overwriting any that exist.
    :type z: zarr.hierarchy.Group
    :param h5ad_path: The annotated h5ad, row-aligned with ``normalized_vectors``.
    :type h5ad_path: str
    :param column: The ``obs`` column holding the final celltype label.
    :type column: str
    :param leiden_column: The ``obs`` column holding the original cluster identifier, used
        to split ``Unknown``. It is required only when unknown vectors are present.
    :type leiden_column: str
    :param chunk: Number of rows accumulated per step when computing centroids.
    :type chunk: int
    :param store_dir: When given, the celltype index is also written here as a CSV file.
    :type store_dir: str or None
    :param out_csv_name: File name of that celltype index.
    :type out_csv_name: str
    :param cmap: Palette used to fix the per-celltype colours, as described under
        :func:`celltype_colors`. Leaving it at ``None`` gives a qualitative palette sized
        to the class count. The result is stored in
        ``z.attrs['cellman_celltype_colors']``.
    :type cmap: str or matplotlib.colors.Colormap or None
    :returns: The celltype index, mapping identifier to label, whether the class came from
        ``Unknown``, its source cluster and its number of vectors.
    :rtype: pandas.DataFrame
    :raises ValueError: If the store holds no ``normalized_vectors``, if the h5ad is not
        row-aligned with them, if the label column is absent, if unknown vectors are
        present without the cluster column, or if no callable celltype remains.
    :raises FileNotFoundError: If the annotated h5ad does not exist.
    """
    if 'normalized_vectors' not in z:
        raise ValueError('store has no normalized_vectors; run downsample_vector() first')
    nv = z['normalized_vectors']
    n_total = nv.shape[0]

    if not os.path.isfile(h5ad_path):
        raise FileNotFoundError(f'annotated h5ad not found: {h5ad_path}')
    adata = ad.read_h5ad(h5ad_path, backed='r')
    if adata.n_obs != n_total:
        raise ValueError(f'h5ad rows ({adata.n_obs}) != normalized_vectors rows ({n_total}); '
                         'the annotated h5ad must be row-aligned with the store')
    obs = adata.obs.copy()
    # the annotation's own palette, if it carries one. Read BEFORE closing the file.
    h5ad_colors = _uns_color_map(adata, column)
    adata.file.close()
    if column not in obs.columns:
        raise ValueError(f'column {column!r} not in obs (have: {list(obs.columns)})')

    celltype = obs[column].astype(str).values
    has_unknown = bool((celltype == UNKNOWN_LABEL).any())
    if leiden_column in obs.columns:
        leiden = pd.to_numeric(obs[leiden_column].astype(str), errors='coerce') \
                   .fillna(-1).astype(np.int64).values
    elif has_unknown:
        raise ValueError(
            f'obs has no {leiden_column!r} column but contains Unknown vectors; Unknown '
            'is split into unknown1..N by the original cluster, so that column is required.')
    else:
        leiden = np.zeros(n_total, dtype=np.int64)

    units = _build_units(celltype, leiden)
    if not units:
        raise ValueError(f'no callable celltypes in obs[{column!r}] '
                         '(only Background/unfinalized labels?)')

    labels = np.full(n_total, -1, dtype=np.int64)
    print(f'[map_celltypes] building centroids from {os.path.basename(h5ad_path)}: '
          f'{len(units)} celltypes, {n_total} vectors')
    for new_id, u in enumerate(units):
        labels[u['mask']] = new_id
        tag = 'unknown' if u['is_unknown'] else 'celltype'
        print(f'    id={new_id:3d}  {u["label"]:<32}  [{tag}]  ({int(u["mask"].sum())} vectors)')
    n_labeled = int((labels >= 0).sum())
    print(f'[map_celltypes] labeled {n_labeled}/{n_total} (dropped -1: {n_total - n_labeled})')

    centroids, counts = compute_centroids(nv, labels, len(units), chunk)

    def overwrite(name, arr):
        """Write an array into the zarr group, replacing any array already under that name.

        :param name: Name of the array in the group.
        :type name: str
        :param arr: The array to store.
        """
        if name in z:
            del z[name]
        z[name] = arr

    names = [u['label'] for u in units]
    colors = celltype_colors(len(names), cmap=cmap, name_colors=h5ad_colors,
                             names=names, fresh_for=[u['is_unknown'] for u in units])
    overwrite('cluster_labels', labels)
    overwrite('centroids', centroids)
    overwrite('celltype_names', np.asarray(names))
    z.attrs['cellman_celltype_colors'] = colors
    print(f'[map_celltypes] wrote cluster_labels {labels.shape} + centroids {centroids.shape} '
          f'+ celltype_names ({len(names)}) + colors (normalized_vectors untouched)')

    df = pd.DataFrame([
        dict(cluster=i, celltype=u['label'], is_unknown=u['is_unknown'],
             source_cluster=u['source_cluster'], n_vectors=int(counts[i]))
        for i, u in enumerate(units)])
    if store_dir:
        out_csv = os.path.join(store_dir, out_csv_name)
        df.to_csv(out_csv, index=False)
        print(f'[map_celltypes] wrote {out_csv}')
    return df
