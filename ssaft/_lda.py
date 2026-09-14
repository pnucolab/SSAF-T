"""FICTURE-style pixel-level factor inference for ssaft.

A third mapper alongside the centroid correlation (:func:`ssaft._count.map_corr`) and
the GaussianNB posterior (:func:`ssaft._gnb.map_gnb`). Where those score a voxel
against a class summary, this one models the voxel's molecule counts generatively and
lets nearby *anchors* — the already-annotated ``downsampled_vectors`` — say which
celltypes are even possible there.

The model is FICTURE's (Si et al., 2023), with the one simplification its Methods
explicitly allow: the factors come from an external annotated reference instead of an
unsupervised hexagon LDA, and the anchors' factor distributions are fixed rather than
learned.

    factor k              a CeLLMan celltype; beta_k is its gene distribution
    anchor j              a downsampled_vector position; theta_j = its label, one-hot
    pixel i               a voxel; its counts come from ONE factor, not a mixture

    log psi_ik  =  sum_g c_ig * log beta_kg   +   log pi_ik
                   \\_ multinomial likelihood _/    \\_ anchor prior _/

    pi_ik = sum_{j : label_j = k} p_ij ,   p_ij = 1 - (||x_i - y_j|| / d_max)^nu

    label(i) = argmax_k psi_ik ,   score(i) = max_k psi_ik   (a posterior -> min_p)

Three things follow from the model rather than being bolted on:

* **The pixel is not an admixture.** LDA's proportion vector theta lives on the
  *anchor*; step 3c of FICTURE's generative process draws a pixel's counts from a
  single ``beta_{z_i}``. A 1 µm voxel sits inside one cell, so this is both faithful
  and cheap — the likelihood is one GEMM, not a per-pixel variational loop.
* **The anchor prior is a hard support constraint.** ``p_ij`` is a compact-support
  kernel: it is exactly 0 at ``d_max``. A celltype with no anchor within ``d_max``
  gets ``pi = 0``, so ``log pi = -inf`` and no likelihood can resurrect it. Measured
  on mouse-kidney Xenium, that leaves a median of 2-3 candidate celltypes out of 27.
* **Pixels with no anchor in range fall back to the likelihood alone.** The anchors
  here are a *sampled* lattice (``downsample_vector``), not FICTURE's regular hexagonal
  one, so coverage is uneven and some voxels genuinely have no neighbour.

Counts, not normalized vectors
------------------------------
A multinomial needs counts, but the pipeline carries L1-normalized (+log1p) vectors.
Both sides are converted back the same way::

    counts = expm1(vec) / size_after_normalization * vf_norm * V_eff

``vf_norm`` is a density (molecules per µm^d), verified mass-preserving: summing it
over a 500x500 µm Xenium crop gives 274,042 against 276,641 transcripts actually
present. Turning a density into "how many molecules of evidence is this" needs the
kernel's effective volume, and ssaft's ``expression_threshold`` is exactly one
transcript's KDE peak, so its reciprocal is that volume::

    V_eff = 1 / expression_threshold = (sqrt(2*pi) * bandwidth)^ndim   (246 at bw 2.5, 3D)

Skip it and every voxel carries ~0.06 molecules of evidence, far too little to move a
posterior off its prior. The count backend stores real binned counts, so ``V_eff = 1``
there.

Anchors inform, they do not decide
----------------------------------
An anchor's only channel into the result is ``pi``, and it WEIGHTS rather than gates.
Earlier versions multiplied the likelihood by a prior that is exactly 0 outside
``d_max``, which made anchor absence a veto no expression could undo — and anchor
absence is three different things at once (the celltype is not here, the downsample
lattice missed it, the clustering merged it into a neighbour). Measured on mouse-kidney
Xenium: that veto left one single candidate at 18.8% of nuclei, and 52% of the nuclei a
reference segmentation calls fibroblast had no fibroblast anchor within ``d_max`` at
all. The prior is also not the more accurate of the two signals — binned by the
dictionary's own top1-top2 gap, it is no better than the dictionary in any bin.

So the decision is now: the DICTIONARY proposes (every class within ``cand_nats`` of the
best log-likelihood is a candidate), the PRIOR ranks what it proposed (added as
``prior_weight * log max(pi, prior_floor)``, which cannot reach zero), and the
annotation's ``unknown*`` classes never win. Even the voxel an anchor sits on is
labelled by ``argmax psi`` like every other voxel, so the likelihood can and does
disagree with the annotation there — that disagreement is the signal worth seeing.
"""
import numpy as np

__all__ = ['build_factors', 'anchor_positions', 'AnchorPrior', 'LDAScorer',
           'GriddedAnchorPrior',
           'map_lda']


def _pseudo_counts(vec, norm, san, veff, dtype=np.float64):
    """L1-normalized (log1p'd) vectors -> molecule-count scale. See module docstring.

    ``dtype`` picks the working precision. ``build_factors`` keeps float64 because it
    accumulates hundreds of thousands of anchors into one sum; the mapper passes
    float32, which is what :class:`LDAScorer` casts to anyway before the matmul, so
    the extra precision was being computed and then thrown away — at 5k genes that
    doubles both the memory traffic and the cost of ``expm1``.

    The scaling is applied in place and in the original order, so the result is the
    value the chained expression produced, minus three full-size temporaries.
    """
    dt = np.dtype(dtype)
    out = np.expm1(np.asarray(vec, dtype=dt))
    out /= dt.type(san)
    out *= np.asarray(norm, dtype=dt)[:, None]
    out *= dt.type(veff)
    return out


def _real_class_mask(zg, K):
    """``(K,) bool`` — the classes that are real celltypes, not ``unknown*``/``background``."""
    import re
    names = [str(s) for s in zg['celltype_names'][:]][:K] if 'celltype_names' in zg else []
    names = names + [f'class{i}' for i in range(len(names), K)]
    return np.array([not re.fullmatch(r'unknown\d*|background', n.strip(), re.I)
                     for n in names], dtype=bool)


def _core_weights(pos, labels, *, radius=8.0):
    """``(n,)`` — each anchor's share of same-label anchors within ``radius`` µm.

    An anchor is a KDE sample, not a cell: one sitting where two celltypes meet carries
    both their genes, and summing it into its own class is how a neighbour's markers get
    into a dictionary row. Its neighbourhood says which kind it is — an anchor deep
    inside a tubule is surrounded by its own label, one on the interstitial boundary is
    not — so the boundary anchors are down-weighted rather than dropped (dropping them
    loses the rare classes entirely, which live almost only at boundaries).
    """
    from scipy.spatial import cKDTree
    pos = np.atleast_2d(np.asarray(pos, dtype=np.float64))
    lab = np.asarray(labels)
    w = np.ones(len(lab), dtype=np.float64)
    for i, js in enumerate(cKDTree(pos).query_ball_point(pos, r=float(radius), workers=-1)):
        if len(js) > 1:
            j = np.asarray(js)
            w[i] = (int((lab[j] == lab[i]).sum()) - 1) / (len(j) - 1)
    return w


def _owned_gene_sets(lam, real, frac):
    """Per-class positions of the genes that class's OWN data owns.

    The marker-aware rule needs to know which genes speak for a celltype. Taking that
    from the annotation's YAML leaves the 8 of 18 mouse-kidney classes that carry no
    yaml marker with no owned gene at all, so every one of their genes is replaced by
    the tissue background and the row stops discriminating. The anchors already answer
    the question: a gene belongs to the classes whose rate reaches ``frac`` x the
    highest rate any real class shows for it.
    """
    K = lam.shape[0]
    p = lam / np.maximum(lam.sum(1, keepdims=True), 1e-12)
    if not np.asarray(real).any():
        return [None] * K
    own = (p >= float(frac) * p[real].max(0)[None, :]) & np.asarray(real)[:, None] & (p > 0)
    return [np.nonzero(own[k])[0] if (real[k] and own[k].any()) else None for k in range(K)]


def build_factors(zg, *, gene_indices=None, size_after_normalization=10.0,
                  veff=1.0, log_transformed=True, sampling_distance=1.0,
                  purity_radius=8.0, min_effective=300.0, own_frac=0.5,
                  verbose=False):
    """``beta (K, G)`` — each celltype's gene distribution, from the annotation.

    Built by summing the annotated ``normalized_vectors`` per class in
    ``cluster_labels`` (which encodes ``obs['cellman_annotation']``) after converting
    them back to counts, then normalizing each row. This is FICTURE's "external
    reference used directly as input factors", with two corrections the anchors
    themselves pay for:

    * **core weighting** (:func:`_core_weights`) — an anchor counts for its class in
      proportion to how much of its own ``purity_radius`` neighbourhood carries the same
      label. A class whose total weight falls below ``min_effective`` is softened to
      ``0.5 (w + 1)`` instead, so a rare celltype that only ever appears at boundaries
      keeps its evidence;
    * **data-driven ownership** (:func:`_owned_gene_sets`) — which genes a class owns is
      read off the anchors rather than the marker YAML, then handed to
      :func:`_marker_aware_beta` exactly as before.

    :param zg: zarr group holding ``normalized_vectors``, ``downsampled_vf_norm``,
        ``downsampled_vectors`` and ``cluster_labels``.
    :param gene_indices: optional column subset, in the order the field carries them.
    :param veff: kernel effective volume; see the module docstring. ``1.0`` for the
        count backend, ``1/expression_threshold`` for the transcript backend. Only the
        *relative* scale matters here (rows are normalized), so this is for symmetry.
    :param log_transformed: whether ``normalized_vectors`` had ``log1p`` applied.
    :param sampling_distance: µm per voxel index, to put the anchors in µm for the
        neighbourhood radius.
    :param purity_radius: neighbourhood radius, µm, for the core weight.
    :param min_effective: below this total weight a class is softened rather than
        core-weighted.
    :param own_frac: a gene belongs to every real class whose rate reaches this
        fraction of the highest real-class rate for that gene.

    The rule :func:`_marker_aware_beta` then applies, now on the owned sets:

        * a gene that is the celltype's own marker, or nobody's marker, keeps the
          celltype's measured rate;
        * a gene that is ANOTHER celltype's marker, or was never detected in this
          celltype, is set to the gene's tissue-wide background rate.

        The second case is the point. A KDE voxel mixes neighbouring cells, so a
        podocyte anchor carries some of the glomerular endothelium's ``Ehd3``; left
        in, that borrowed signal lets podocyte compete on a marker it does not own
        (measured: ``Ehd3`` 3.11% in podocyte vs 3.61% in glomerular EC, so the real
        marker contributed almost nothing to the decision). Zeroing it instead would
        make the entry a veto whose strength is set by ``smooth`` rather than by the
        data; the background rate says the honest thing — "this celltype's data tells
        us nothing about this gene, so assume the tissue average".
    :returns: ``(beta, classes)`` with ``beta`` row-normalized.
    """
    if 'normalized_vectors' not in zg or 'cluster_labels' not in zg:
        raise ValueError('LDA factors need normalized_vectors + cluster_labels; '
                         'pass h5ad=<annotated .h5ad> to map_celltypes first')
    V = np.asarray(zg['normalized_vectors'][:], dtype=np.float64)
    if not log_transformed:
        V = np.log1p(V)
    y = np.asarray(zg['cluster_labels'][:])
    if 'downsampled_vf_norm' in zg:
        vn = np.asarray(zg['downsampled_vf_norm'][:], dtype=np.float64).ravel()
    else:                       # no per-anchor density: treat every vector equally
        vn = np.ones(V.shape[0], dtype=np.float64)
    K = int(zg['centroids'].shape[0]) if 'centroids' in zg else int(y.max()) + 1
    cnt = _pseudo_counts(V, vn, size_after_normalization, veff)
    del V
    if gene_indices is not None:
        cnt = cnt[:, np.asarray(gene_indices, dtype=np.int64)]
    pos, _ = anchor_positions(zg, sampling_distance=sampling_distance)
    w = _core_weights(pos, y, radius=purity_radius)
    lam = np.zeros((K, cnt.shape[1]), dtype=np.float64)
    n_soft = 0
    for k in range(K):
        m = y == k
        if not m.any():
            continue
        wk = w[m]
        if wk.sum() < float(min_effective):
            wk = 0.5 * (wk + 1.0)
            n_soft += 1
        lam[k] = (cnt[m] * wk[:, None]).sum(0)
    real = _real_class_mask(zg, K)
    sets = _owned_gene_sets(lam, real, own_frac)
    beta = _marker_aware_beta(lam, sets)
    if verbose:
        n_lab = int((y >= 0).sum())
        n_own = sum(0 if s is None else len(s) for s in sets)
        print(f'[lda] factors {beta.shape} from {n_lab:,} annotated vectors, {K} celltypes '
              f'({int(real.sum())} real); core-weighted r={purity_radius:g} µm '
              f'(median w={np.median(w):.2f}, {n_soft} classes softened), '
              f'{n_own:,} owned gene slots at own_frac={own_frac:g}')
    return beta, np.arange(K)


def _marker_aware_beta(lam, marker_sets):
    """``beta`` where cells the celltype's own data cannot speak to fall back to background.

    ``lam`` is the raw per-celltype pseudobulk (K, G). Two kinds of entry carry no
    information about the celltype: a gene that belongs to a DIFFERENT celltype's
    marker set (whatever is there came from a neighbouring cell through the kernel)
    and a gene never detected in this celltype at all. Both are replaced by the
    gene's tissue-wide rate, which is what "no information" should look like.
    """
    lam = np.asarray(lam, dtype=np.float64)
    K, G = lam.shape
    own = np.zeros((K, G), dtype=bool)
    for k, idx in enumerate(marker_sets or []):
        if k >= K or idx is None:
            continue
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size:
            own[k, idx] = True
    any_marker = own.any(0)
    total = lam.sum()
    if total <= 0:
        return np.full((K, G), 1.0 / max(G, 1))
    p = lam / np.maximum(lam.sum(1, keepdims=True), 1e-12)
    bg = lam.sum(0) / total
    uninformative = (any_marker[None, :] & ~own) | (lam <= 0)
    l = np.where(uninformative, bg[None, :], p)
    return l / np.maximum(l.sum(1, keepdims=True), 1e-12)


def anchor_positions(zg, *, sampling_distance=1.0, coordinate_offset=None):
    """``(xyz (n, 3) in µm, labels (n,))`` for the annotated downsampled vectors.

    ``downsampled_vectors`` holds voxel indices relative to the grid that produced
    them, so they become absolute µm through the SAME offset that grid used. When the
    store is later re-gridded (mapping a crop, say) that offset no longer matches and
    the caller must pass ``coordinate_offset`` explicitly — the mismatch is silent
    otherwise, and an anchor set displaced by even a few hundred µm makes every prior
    uniform, which looks exactly like "the anchors did nothing".

    The z column is returned even when every anchor shares one plane, which is what
    ``downsample_vector`` writes today for a field it treated as 2D. The prior then
    computes the same distances it always did (a constant column cancels), so a store
    written before anchors carried depth maps exactly as before; a store whose anchors
    DO span z gets a prior that can tell two celltypes stacked in depth apart.
    """
    dv = np.asarray(zg['downsampled_vectors'][:])
    if dv.shape[0] not in (2, 3):
        dv = dv.T
    off = (np.asarray(zg['coordinate_offset'][:], dtype=np.float64)
           if coordinate_offset is None else np.asarray(coordinate_offset, np.float64))
    z = (dv[2] * sampling_distance + off[2] if dv.shape[0] > 2
         else np.zeros(dv.shape[1], dtype=np.float64))
    xyz = np.stack([dv[0] * sampling_distance + off[0],
                    dv[1] * sampling_distance + off[1],
                    np.asarray(z, dtype=np.float64)], 1).astype(np.float64)
    return xyz, np.asarray(zg['cluster_labels'][:])


class _LatticeKNNPrior:
    """:class:`AnchorPrior`'s nearest-``k_max`` prior, computed on the GPU for lattice anchors.

    Anchors and voxels share one lattice (``downsampled_vectors`` are voxel indices), so the
    ``k_max`` nearest anchors within ``d_max`` are the anchors met first when the lattice
    offsets inside the ``d_max`` ball are walked in increasing distance. The anchor voxels are
    kept as a sorted key list with a per-voxel class histogram (several anchors may share a
    voxel); each query batch looks every candidate offset up with ``searchsorted``, caps the
    running anchor count at ``k_max`` in distance order and adds the weighted histograms.
    Offsets whose z holds no anchor plane are skipped for each query depth.

    Memory is bounded per batch: roughly 40 bytes per (query, offset) pair, with the batch
    sized to a quarter of the device's free memory (at most ``max_bytes``) and halved on an
    out-of-memory error. Ties at the ``k_max``-th distance are broken by offset order.
    """

    def __init__(self, pos, labels, n_classes, *, d_max, nu, k_max, sampling_distance,
                 device, mem_frac=0.25, max_bytes=4 << 30):
        import torch
        pos = np.atleast_2d(np.asarray(pos, dtype=np.float64))
        if pos.shape[1] == 2:
            pos = np.column_stack([pos, np.zeros(len(pos))])
        lab = np.asarray(labels).astype(np.int64)
        self.K, self.d_max, self.nu, self.k_max = int(n_classes), float(d_max), float(nu), int(k_max)
        self.sd, self.dev = float(sampling_distance), torch.device(device)
        self.mem_frac, self.max_bytes = float(mem_frac), int(max_bytes)
        self.r = r = int(np.ceil(self.d_max / self.sd))
        self.o = pos.min(0)
        idx = np.rint((pos - self.o) / self.sd).astype(np.int64)
        if len(pos) == 0 or np.abs(idx * self.sd + self.o - pos).max() > 1e-3 * self.sd:
            raise ValueError('anchors are not on a lattice of the sampling distance')
        # 2r of padding: a query kept for lookup (index in [r, N - r)) never wraps a row
        self.lo = idx.min(0) - 2 * r
        idx -= self.lo
        self.N = idx.max(0) + 2 * r + 1
        NY, NZ = int(self.N[1]), int(self.N[2])
        key = (idx[:, 0] * NY + idx[:, 1]) * NZ + idx[:, 2]
        u, inv = np.unique(key, return_inverse=True)
        H = np.zeros((len(u), self.K), dtype=np.float32)
        np.add.at(H, (inv, lab), 1.0)
        self.u = torch.as_tensor(u, device=self.dev)
        self.H = torch.as_tensor(H, device=self.dev)
        self.C = self.H.sum(1)
        self.zplanes = np.unique(idx[:, 2])
        g = np.arange(-r, r + 1)
        dx, dy, dz = (a.ravel() for a in np.meshgrid(g, g, g, indexing='ij'))
        d = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2) * self.sd
        keep = d < self.d_max                       # the KD-tree's own test: dist < d_max
        dx, dy, dz, d = dx[keep], dy[keep], dz[keep], d[keep]
        order = np.lexsort((dx, dy, dz, d))         # increasing distance
        self.dx, self.dy, self.dz = dx[order], dy[order], dz[order]
        self.w = (1.0 - (np.minimum(d[order], self.d_max) / self.d_max) ** self.nu
                  ).astype(np.float32)
        self._zcache = {}
        self.last_empty = 0

    def _offsets(self, iz):
        import torch
        if iz not in self._zcache:
            NY, NZ = int(self.N[1]), int(self.N[2])
            m = np.isin(iz + self.dz, self.zplanes)
            lin = (self.dx[m] * NY + self.dy[m]) * NZ + self.dz[m]
            self._zcache[iz] = (torch.as_tensor(lin, device=self.dev),
                                torch.as_tensor(self.w[m], device=self.dev))
        return self._zcache[iz]

    def _batch(self, n_off):
        import torch
        free, _ = torch.cuda.mem_get_info(self.dev)
        budget = min(free * self.mem_frac, self.max_bytes)
        return int(max(256, budget // (max(n_off, 1) * 40)))

    def _accumulate(self, out, rows_t, key_t, off, w):
        import torch
        cand = key_t[:, None] + off[None, :]
        j = torch.searchsorted(self.u, cand).clamp_max_(self.u.numel() - 1)
        c = torch.where(self.u[j] == cand, self.C[j], torch.zeros((), device=self.dev))
        del cand
        before = c.cumsum(1) - c
        take = torch.minimum((self.k_max - before).clamp_min_(0), c)
        del before
        bi, oi = torch.nonzero(take > 0, as_tuple=True)
        jj = j[bi, oi]
        coef = w[oi] * take[bi, oi] / self.C[jj]
        out.index_add_(0, rows_t[bi], coef[:, None] * self.H[jj])

    def __call__(self, q):
        """``(n, K)`` row-normalized prior on the device; ValueError for off-lattice queries."""
        import torch
        q = np.atleast_2d(np.asarray(q, dtype=np.float64))
        if q.shape[1] == 2:
            q = np.column_stack([q, np.zeros(len(q))])
        iq = np.rint((q - self.o) / self.sd).astype(np.int64)
        if len(q) and np.abs(iq * self.sd + self.o - q).max() > 1e-3 * self.sd:
            raise ValueError("queries are not on the anchors' lattice")
        iq -= self.lo
        NY, NZ = int(self.N[1]), int(self.N[2])
        out = torch.zeros((len(q), self.K), dtype=torch.float32, device=self.dev)
        # a query farther than r lattice steps outside the anchors' box has no anchor in range
        inside = np.all((iq >= self.r) & (iq < self.N - self.r), axis=1)
        key = (iq[:, 0] * NY + iq[:, 1]) * NZ + iq[:, 2]
        for iz in np.unique(iq[inside, 2]):
            sel = np.flatnonzero(inside & (iq[:, 2] == iz))
            off, w = self._offsets(int(iz))
            if off.numel() == 0:
                continue
            B, s = self._batch(off.numel()), 0
            while s < len(sel):
                rows = sel[s:s + B]
                try:
                    self._accumulate(out, torch.as_tensor(rows, device=self.dev),
                                     torch.as_tensor(key[rows], device=self.dev), off, w)
                    s += len(rows)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if B <= 256:
                        raise
                    B //= 2
        tot = out.sum(1, keepdim=True)
        empty = tot[:, 0] <= 0
        self.last_empty = int(empty.sum())
        out = out / tot.clamp_min(1e-30)
        out[empty] = 1.0 / self.K                   # no anchor in range -> uniform
        return out


class AnchorPrior:
    """Compact-support kernel over nearby anchors -> per-pixel factor prior.

    ``p_ij = 1 - (d_ij / d_max)^nu`` for anchors inside ``d_max``, exactly 0 outside.
    FICTURE picks ``d_max`` larger than the anchor spacing and smaller than a cell
    diameter, so the prior refines boundaries without smoothing across cells.
    """

    def __init__(self, xy, labels, n_classes, *, d_max=12.0, nu=2.0, k_max=48,
                 sampling_distance=None):
        from scipy.spatial import cKDTree
        pos = np.atleast_2d(np.asarray(xy, dtype=np.float64))
        labels = np.asarray(labels)
        # Background anchors carry label -1 (no cell type): they add to no class's prior, as in
        # GriddedAnchorPrior, which counts lab == k per class and so never sees them.
        keep = labels >= 0
        pos, labels = pos[keep], labels[keep]
        self.ndim = int(pos.shape[1])
        # A z column that never varies is kept: the KD-tree distance is unchanged by a
        # constant coordinate, so single-plane anchors behave exactly as they did in 2D.
        self.zfill = float(np.median(pos[:, 2])) if self.ndim > 2 else 0.0
        self.tree = cKDTree(pos)
        self.labels = np.asarray(labels)
        self.K = int(n_classes)
        self.d_max = float(d_max)
        self.nu = float(nu)
        self.k_max = int(k_max)
        # Anchors on a voxel lattice of this spacing let torch_lookup answer on the GPU
        # (:class:`_LatticeKNNPrior`); None keeps the KD-tree for every lookup.
        self.sampling_distance = None if sampling_distance is None else float(sampling_distance)
        self._pos = pos
        self._gpu = None
        self._gpu_off = False

    def _query_points(self, pix):
        """Queries in the anchors' own dimensionality: pad a 2D query, drop a 3D one."""
        q = np.atleast_2d(np.asarray(pix, dtype=np.float64))
        if q.shape[1] == self.ndim:
            return q
        if q.shape[1] < self.ndim:
            return np.column_stack([q, np.full((len(q), self.ndim - q.shape[1]),
                                               self.zfill)])
        return np.ascontiguousarray(q[:, :self.ndim])

    def to_torch(self, device):
        """Build the GPU lookup now, if the anchors allow it (see :meth:`torch_lookup`)."""
        self._lattice(device)
        return self

    def _lattice(self, device):
        """The :class:`_LatticeKNNPrior` for ``device``, or None when the KD-tree must answer."""
        import torch
        dev = torch.device(device)
        if self._gpu_off or self.sampling_distance is None or dev.type != 'cuda':
            return None
        if self._gpu is None or self._gpu.dev != dev:
            try:
                self._gpu = _LatticeKNNPrior(
                    self._pos, self.labels, self.K, d_max=self.d_max, nu=self.nu,
                    k_max=self.k_max, sampling_distance=self.sampling_distance, device=dev)
            except ValueError:              # anchors off the lattice
                self._gpu_off = True
                return None
        return self._gpu

    def torch_lookup(self, pix_xy, device):
        """The same rows as :meth:`__call__`, on ``device``.

        With the anchors on a voxel lattice and a CUDA device, the nearest-``k_max`` search
        runs on the GPU (:class:`_LatticeKNNPrior`; 7.5x the KD-tree on the mouse-kidney
        section, where the 3D prior had been the CPU stall inside the GPU KDE loop).
        Otherwise the KD-tree answers on the host and the rows are uploaded.

        The two agree except where several anchors share the ``k_max``-th distance. Which of
        them take the last slots is arbitrary in both, and they break the tie differently:
        7.7% of voxels on that section, with the prior's argmax unchanged on 99.87%.
        """
        import torch
        gpu = self._lattice(device)
        if gpu is not None:
            try:
                P = gpu(self._query_points(pix_xy))
                self.last_empty = gpu.last_empty
                return P
            except ValueError:              # queries off the anchors' lattice
                pass
        return torch.as_tensor(self(pix_xy), device=device)

    def __call__(self, pix_xy, *, batch=2_000_000):
        """``(n, K)`` row-normalized prior; rows with no anchor in range are uniform."""
        pix_xy = self._query_points(pix_xy)
        out = np.empty((len(pix_xy), self.K), dtype=np.float32)
        n_empty = 0
        for s in range(0, len(pix_xy), batch):
            e = min(s + batch, len(pix_xy))
            dist, idx = self.tree.query(pix_xy[s:e], k=self.k_max,
                                        distance_upper_bound=self.d_max, workers=-1)
            if dist.ndim == 1:
                dist, idx = dist[:, None], idx[:, None]
            ok = np.isfinite(dist) & (dist < self.d_max)
            w = np.where(ok, 1.0 - (np.minimum(dist, self.d_max) / self.d_max) ** self.nu,
                         0.0).astype(np.float32)
            lab = self.labels[np.where(ok, np.minimum(idx, len(self.labels) - 1), 0)]
            rows = np.repeat(np.arange(e - s), lab.shape[1])
            P = np.bincount((rows * self.K + lab.ravel()), weights=w.ravel(),
                            minlength=(e - s) * self.K).reshape(e - s, self.K)
            empty = P.sum(1) <= 0
            n_empty += int(empty.sum())
            P[empty] = 1.0
            out[s:e] = (P / P.sum(1, keepdims=True)).astype(np.float32)
        self.last_empty = n_empty
        return out


class GriddedAnchorPrior:
    """The same prior as :class:`AnchorPrior`, evaluated once as a convolution.

    ``AnchorPrior`` asks a KD-tree for the 48 nearest anchors of every voxel. That is
    36 s per chunk on a mouse-kidney store - two thirds of the whole mapping - and it
    repeats work: the prior depends on POSITION only, never on the counts.

    The anchors sit on the same 1 µm lattice as the voxels (``downsample_vectors`` are
    voxel indices), so ``pi_k`` is exactly a convolution: paint each celltype's anchors
    into an image and convolve with the compact kernel ``1 - (d/d_max)^nu``. Computed
    once for the tissue, a voxel's prior is then an integer lookup.

    A convolution has no notion of "the nearest k", so on its own it would quietly drop
    ``k_max`` truncation — and that is not a free improvement: the classes with the most
    anchors fill those k slots, so removing the cut moves 0.08% of voxels, nearly all of
    them toward the proximal tubule. To stay bit-for-bit with the KD-tree, the number of
    anchors within ``d_max`` is itself convolved (a sum of ones, so exact), and the few
    nodes above ``k_max`` are computed the old way. ``exact_k_max=False`` keeps the
    untruncated prior, which is what the model actually specifies.
    """

    def __init__(self, xy, labels, n_classes, *, d_max=12.0, nu=2.0, k_max=48,
                 sampling_distance=1.0, origin=None, exact_k_max=True, verbose=False):
        xy = np.atleast_2d(np.asarray(xy, dtype=np.float64))
        # This prior is a 2D convolution over the voxel lattice. Anchors that span depth
        # need a 3D kernel, and the (nx, ny, nz, K) field that would take is tens of GB
        # on a whole section, so the caller falls back to the KD-tree AnchorPrior.
        if xy.shape[1] > 2 and float(np.ptp(xy[:, 2])) > 0:
            raise ValueError('GriddedAnchorPrior needs anchors on a single z plane; '
                             'these span depth — use AnchorPrior')
        xy = np.ascontiguousarray(xy[:, :2])
        lab = np.asarray(labels)
        self.K = int(n_classes)
        self.d_max = float(d_max)
        self.nu = float(nu)
        self.sd = float(sampling_distance)
        ox = float(origin[0]) if origin is not None else float(xy[:, 0].min())
        oy = float(origin[1]) if origin is not None else float(xy[:, 1].min())
        r = int(np.ceil(self.d_max / self.sd))
        ix = np.rint((xy[:, 0] - ox) / self.sd).astype(np.int64)
        iy = np.rint((xy[:, 1] - oy) / self.sd).astype(np.int64)
        x0, y0 = int(ix.min()) - r, int(iy.min()) - r
        nx = int(ix.max()) - x0 + r + 1
        ny = int(iy.max()) - y0 + r + 1
        self.ox = ox + x0 * self.sd
        self.oy = oy + y0 * self.sd
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        d = np.hypot(xx, yy) * self.sd
        W = np.where(d < self.d_max,
                     1.0 - (np.minimum(d, self.d_max) / self.d_max) ** self.nu,
                     0.0).astype(np.float32)
        from scipy.signal import oaconvolve
        field = np.zeros((nx, ny, self.K), dtype=np.float32)
        for k in range(self.K):
            m = lab == k
            if not m.any():
                continue
            img = np.zeros((nx, ny), dtype=np.float32)
            np.add.at(img, (ix[m] - x0, iy[m] - y0), 1.0)
            field[..., k] = oaconvolve(img, W, mode='same')
        # An overlap-add FFT leaves ~1e-7 of the signal's scale everywhere, including
        # where no anchor is in range. Row-normalising turns that noise into a full
        # prior, so it is cleared first: the smallest weight a real anchor can carry is
        # 1 - ((d_max - sd) / d_max)^nu, and the cut sits an order of magnitude below it.
        w_min = 1.0 - ((self.d_max - self.sd) / self.d_max) ** self.nu
        field[field < np.float32(w_min * 0.1)] = 0.0
        # Where more anchors than k_max are in range the KD-tree truncates; those
        # nodes are marked here (the count is a convolution of ones, hence exact) and
        # answered by the KD-tree at lookup time.
        self.k_max = int(k_max)
        self.exact_k_max = bool(exact_k_max)
        self._tree = None
        self._lab = lab
        self.trunc = None
        if self.exact_k_max:
            disc = (d < self.d_max).astype(np.float32)
            img = np.zeros((nx, ny), dtype=np.float32)
            np.add.at(img, (ix - x0, iy - y0), 1.0)
            cnt = np.rint(oaconvolve(img, disc, mode='same'))
            self.trunc = cnt > self.k_max
            if self.trunc.any():
                from scipy.spatial import cKDTree
                self._tree = cKDTree(np.asarray(xy, dtype=np.float64))
                if verbose:
                    print(f'[lda] {int(self.trunc.sum()):,} lattice nodes hold more '
                          f'than k_max={self.k_max} anchors; those keep the KD-tree '
                          f'answer')
            else:
                self.trunc = None
        tot = field.sum(2, keepdims=True)
        self.empty = (tot[..., 0] <= 0)
        np.divide(field, np.maximum(tot, 1e-30), out=field)
        field[self.empty] = 1.0 / self.K            # no anchor in range -> uniform
        self.nx, self.ny = nx, ny
        self._flat = np.ascontiguousarray(field.reshape(-1, self.K))
        self.last_empty = 0
        self._t = None
        if verbose:
            print(f'[lda] gridded anchor prior {nx}x{ny}x{self.K} '
                  f'({self._flat.nbytes / 1e9:.2f} GB), kernel radius {r} px')

    def to_torch(self, device):
        import torch
        self._t = torch.as_tensor(self._flat, device=device)
        return self

    def _index(self, pix_xy):
        xy = np.asarray(pix_xy)
        ix = np.clip(np.rint((xy[:, 0] - self.ox) / self.sd), 0,
                     self.nx - 1).astype(np.int64)
        iy = np.clip(np.rint((xy[:, 1] - self.oy) / self.sd), 0,
                     self.ny - 1).astype(np.int64)
        return ix * self.ny + iy

    def _exact_rows(self, pix_xy, sel):
        """The KD-tree prior for the handful of positions k_max would truncate."""
        from scipy.spatial import cKDTree
        q = np.asarray(pix_xy)[sel]
        dist, idx = self._tree.query(q, k=self.k_max,
                                     distance_upper_bound=self.d_max, workers=-1)
        if dist.ndim == 1:
            dist, idx = dist[:, None], idx[:, None]
        ok = np.isfinite(dist) & (dist < self.d_max)
        w = np.where(ok, 1.0 - (np.minimum(dist, self.d_max) / self.d_max) ** self.nu,
                     0.0).astype(np.float32)
        lab = self._lab[np.where(ok, np.minimum(idx, len(self._lab) - 1), 0)]
        rows = np.repeat(np.arange(len(q)), lab.shape[1])
        P = np.bincount((rows * self.K + lab.ravel()), weights=w.ravel(),
                        minlength=len(q) * self.K).reshape(len(q), self.K)
        empty = P.sum(1) <= 0
        P[empty] = 1.0
        return (P / P.sum(1, keepdims=True)).astype(np.float32)

    def __call__(self, pix_xy, *, batch=None):
        flat = self._index(pix_xy)
        self.last_empty = int(self.empty.reshape(-1)[flat].sum())
        out = self._flat.take(flat, axis=0)
        if self.trunc is not None:
            sel = np.flatnonzero(self.trunc.reshape(-1)[flat])
            if sel.size:
                out = out.copy()
                out[sel] = self._exact_rows(pix_xy, sel)
        return out

    def torch_lookup(self, pix_xy, device):
        import torch
        idx = self._index(pix_xy)
        if self._t is None:
            self.to_torch(device)
        out = self._t.index_select(0, torch.as_tensor(idx, device=device))
        if self.trunc is not None:
            sel = np.flatnonzero(self.trunc.reshape(-1)[idx])
            if sel.size:
                out[torch.as_tensor(sel, device=device)] = torch.as_tensor(
                    self._exact_rows(pix_xy, sel), device=device)
        return out


def _with_z(pix_xy, pix_z=None):
    """``(n, 2)`` positions plus their z, when the caller has it."""
    if pix_z is None:
        return pix_xy
    xy = np.atleast_2d(np.asarray(pix_xy))
    if xy.shape[1] > 2:
        return xy
    return np.column_stack([xy, np.asarray(pix_z, dtype=xy.dtype).reshape(-1)])


class LDAScorer:
    """Turns a chunk of voxel counts + positions into ``(label, posterior)``.

    The dictionary proposes and the prior ranks (see the module docstring):

    1. ``unknown*`` classes are taken out of the running. The annotation abstained
       there, so they are a label for a voxel nobody claims — not a candidate that
       should beat a real celltype because it happens to have an anchor nearby;
    2. the candidates are every class within ``cand_nats`` of the best log-likelihood.
       One nat is a factor of e: anything further behind is not a close call;
    3. among those, ``prior_weight * log max(pi, prior_floor)`` is added. The floor is
       what makes the prior a weight rather than a veto — a class with no anchor in
       range pays a constant penalty and still wins where the likelihood is far enough
       ahead.

    Measured on mouse-kidney Xenium against a reference segmentation labelled by an
    independent snRNA-seq judge: voxel accuracy 78.3 -> 82.7%, macro F1 62.9 -> 73.2%,
    fibroblast nucleus voxels 43.4 -> 78.7%.
    """

    def __init__(self, beta, prior=None, unknown=None, *, cand_nats=1.0,
                 prior_weight=0.5, prior_floor=0.02):
        beta = np.asarray(beta, dtype=np.float64)
        beta = beta / beta.sum(1, keepdims=True)
        # A gene the panel never detected anywhere is 0 in EVERY factor (its
        # tissue-wide background rate is 0, so _marker_aware_beta hands the whole
        # column back as 0). log would make that column -inf, and the matmul turns
        # the 0 * -inf of every voxel that does not carry the gene into NaN — ONE
        # such gene NaNs the whole score matrix, after which argmax lands on class 0
        # everywhere and the posterior is NaN, so filter_celltypemaps drops the
        # entire map. The floor sits far below any real rate and the entry is only
        # ever multiplied by a zero count, so it changes no score that matters.
        np.maximum(beta, 1e-12, out=beta)
        self.log_beta = np.log(beta).astype(np.float32)
        self.prior = prior
        self.K = beta.shape[0]
        self.unknown = (np.zeros(self.K, dtype=bool) if unknown is None
                        else np.asarray(unknown, dtype=bool).reshape(self.K))
        self.cand_nats = float(cand_nats)
        self.prior_weight = float(prior_weight)
        self.prior_floor = float(prior_floor)

    def to_torch(self, device):
        """Move the dictionary (and whatever the prior holds) to the GPU.

        The producer already has the field on the device; scoring there as well removes
        a (voxels x genes) host copy per chunk — 1.7 GB on this store — and the numpy
        work behind it. Results differ only by float32 summation order.
        """
        import torch
        self._dev = device
        self._tlb = torch.as_tensor(self.log_beta, device=device)
        self._tunk = torch.as_tensor(self.unknown, device=device)
        if self.prior is not None and hasattr(self.prior, 'to_torch'):
            self.prior.to_torch(device)
        return self

    def torch_call(self, counts_t, pix_xy=None, pix_z=None):
        """Same as ``__call__`` with a torch tensor in, and no host round-trip."""
        return self.torch_call_ll(counts_t @ self._tlb.T, pix_xy, pix_z)

    def torch_call_ll(self, Z, pix_xy=None, pix_z=None):
        """Score from the log-likelihood ``Z (n, K)`` that is already on the device.

        The likelihood is the ONLY thing the counts are needed for, and it can be formed
        without ever building the ``(voxels x genes)`` count matrix -- see
        :func:`map_lda`, where the field block is contracted gene-first instead.
        Splitting it out here lets the caller skip that matrix, which on a whole-brain
        count field is several GB per band.
        """
        import torch
        NEG = float(np.finfo(np.float32).min / 4)
        Z = Z.masked_fill(self._tunk.unsqueeze(0), NEG)
        cand = Z >= (Z.max(1, keepdim=True).values - self.cand_nats)
        if self.prior is not None and pix_xy is not None:
            P = self.prior.torch_lookup(_with_z(pix_xy, pix_z), Z.device)
            Z = Z + self.prior_weight * P.clamp_min(self.prior_floor).log()
        Z = torch.softmax(Z.masked_fill(~cand, NEG), dim=1)
        best = Z.max(1)
        return (best.indices.to(torch.int32).cpu().numpy(),
                best.values.to(torch.float32).cpu().numpy())

    def __call__(self, counts, pix_xy=None, pix_z=None):
        """``counts (n, G)`` -> ``(best_idx int32, best_prob float32)``.

        :param pix_xy: voxel centres in µm, for the anchor prior.
        :param pix_z: voxel depth in µm. Used when the anchors carry depth themselves;
            with single-plane anchors it changes no distance.
        """
        NEG = np.float32(-1e30)
        Z = np.asarray(counts, dtype=np.float32) @ self.log_beta.T
        Z[:, self.unknown] = NEG
        cand = Z >= Z.max(1, keepdims=True) - np.float32(self.cand_nats)
        if self.prior is not None and pix_xy is not None:
            P = self.prior(_with_z(pix_xy, pix_z))
            Z = Z + np.float32(self.prior_weight) * np.log(
                np.maximum(P, np.float32(self.prior_floor))).astype(np.float32)
        Z = np.where(cand, Z, NEG).astype(np.float32)
        Z -= Z.max(1, keepdims=True)
        np.exp(Z, out=Z)
        Z /= Z.sum(1, keepdims=True)
        return Z.argmax(1).astype(np.int32), Z.max(1).astype(np.float32)


def _marker_mean_matrix(index_sets, n_genes, n_classes):
    """``(G, K)`` so that ``counts @ M`` is each celltype's own-marker MEAN.

    The same quantity :func:`ssaft._marker_corr.marker_score` computes by indexing,
    written as a contraction so the device can do it in one matmul.
    """
    M = np.zeros((int(n_genes), int(n_classes)), dtype=np.float32)
    for k, idx in enumerate(index_sets or []):
        if k >= M.shape[1] or idx is None or not len(idx):
            continue
        idx = np.asarray(idx, dtype=np.int64)
        M[idx, k] = 1.0 / len(idx)
    return M


def map_lda(analysis, *, centroids=None, vf_norm_threshold=None, chunk_h=128,
            overwrite=True, gene_indices=None, size_after_normalization=10.0,
            veff=1.0, d_max=12.0, nu=2.0, k_max=48, use_anchor_prior=True,
            anchor_xy=None, anchor_labels=None, beta=None, marker_idx=None,
            marker_sets=None, device=None, gridded_prior=True,
            cand_nats=1.0, prior_weight=0.5, prior_floor=0.02,
            gpu_target_bytes=int(4e9), sub_batch=None):
    """Map the count/blur field with the FICTURE-style scorer.

    The array-backend counterpart of :func:`ssaft._count.map_corr` /
    :func:`ssaft._gnb.map_gnb`. Writes ``celltype_maps`` plus ``max_probabilities``
    (the score is a posterior, so ``filter_celltypemaps`` thresholds it with ``min_p``).

    :param centroids: accepted for signature parity; unused — the factors come from
        the annotated vectors, not the centroids.
    :param veff: molecule-count scale for the density; ``1.0`` is right for the count
        backend, whose ``vf_norm`` already counts molecules per bin.
    :param d_max: anchor influence radius in µm. Larger than the anchor spacing and
        smaller than a cell diameter.
    :param use_anchor_prior: set False to score on the likelihood alone.
    :param marker_idx: legacy per-celltype marker gene positions, used for the marker
        score only when ``marker_sets`` is absent (see
        :func:`ssaft._marker_corr.marker_index_sets`) on the mapped gene axis. When
        given, ``celltype_marker_score`` is written alongside the map so
        ``filter_celltypemaps`` can threshold each celltype on its own markers.
    :param cand_nats: a class is a candidate when its log-likelihood is within this much
        of the best one (see :class:`LDAScorer`). The anchor prior only ranks what the
        dictionary proposed, so it can no longer veto a celltype outright.
    :param prior_weight: weight on the log anchor prior.
    :param prior_floor: floor under the prior before its log is taken — this is what
        turns "no anchor in range" from a veto into a constant penalty.
    :param device: CUDA device for the scoring, e.g. ``'cuda'`` / ``'cuda:1'``.
        ``None`` (default) keeps everything on the CPU. The field still comes off disk
        on the host — that part is I/O — but the likelihood is a
        ``(voxels x genes) @ (genes x celltypes)`` GEMM, which is what the card is for.
        Falls back to the CPU, with a message, when torch or CUDA is unavailable.
    :param gridded_prior: build the anchor prior as a precomputed convolution when the
        anchors sit on the voxel lattice (see :class:`GriddedAnchorPrior`), which also
        keeps the prior lookup on the device. ``False`` forces the KD-tree version.
    :param marker_sets: per-celltype OWN-marker positions used to build the
        dictionary; see :func:`build_factors`. Unlike ``marker_idx`` this one has no
        minimum-marker fallback — a celltype with a single marker still needs that
        marker recognised as its own, or the dictionary would send it to background.
    :param gpu_target_bytes: VRAM budget for ONE field band. ``chunk_h`` is reduced to
        fit it (see :func:`ssaft._count.band_rows_for_budget`), so a wide gene axis
        cannot run the card out of memory the way a fixed ``chunk_h`` can.
    :param sub_batch: voxels scored per device batch inside a band; sized from the same
        budget when None. Bounds the one temporary that scales with the gene axis.

    On a CUDA device the field is never transposed or gathered on the host. The band
    is uploaded as it sits on disk, ``(genes, rows, cols)``, and the likelihood is
    contracted gene-first,

        ``Z[m, k] = s[m] * sum_g expm1(v[g, m]) * log_beta[k, g]``

    with ``s = veff * vf_norm / size_after_normalization`` a per-voxel scalar. That is
    the same number the host path computes as ``_pseudo_counts(...) @ log_beta.T``, but
    it needs neither the transpose (the field is gene-major on disk and voxel-major in
    the old expression) nor the ``(voxels x genes)`` count matrix. On a 3.8k-gene
    whole-brain field those two copies were ~2.8 s of the ~5.9 s each band cost, against
    0.03 s for the matmul itself.
    """
    from tqdm.auto import tqdm
    from ._count import _write_maps

    zg = analysis.dataset.zarr_group
    ds = analysis.dataset
    vf = zg['vf_normalized']
    n_G, n_R, n_C = vf.shape
    norm = np.asarray(zg['vf_norm'][:])
    sd = float(getattr(ds, 'sampling_distance', None)
               or zg.attrs.get('sampling_distance', zg.attrs.get('bin_size_um', 1.0)))
    from ._count import resolve_vf_norm_threshold
    vf_norm_threshold = resolve_vf_norm_threshold(zg, vf_norm_threshold)

    gi = None if gene_indices is None else np.asarray(sorted(gene_indices), np.int64)
    if beta is None:
        beta, _ = build_factors(zg, gene_indices=gi, veff=veff,
                                size_after_normalization=size_after_normalization,
                                sampling_distance=sd, verbose=analysis.verbose)
    prior = None
    if use_anchor_prior:
        if anchor_xy is None:
            anchor_xy, anchor_labels = anchor_positions(zg, sampling_distance=sd)
        prior = analysis._make_anchor_prior(
            (anchor_xy, anchor_labels), beta.shape[0], sd,
            dict(d_max=d_max, nu=nu, k_max=k_max, gridded_prior=gridded_prior))

    off = np.asarray(zg['coordinate_offset'][:], dtype=np.float64) \
        if 'coordinate_offset' in zg else np.zeros(3)
    _unk = np.zeros(beta.shape[0], dtype=bool)
    _unk[[i for i in analysis._unknown_class_ids() if i < beta.shape[0]]] = True
    scorer = LDAScorer(beta, prior=prior, unknown=_unk, cand_nats=cand_nats,
                       prior_weight=prior_weight, prior_floor=prior_floor)
    _dev = None
    if device is not None:
        try:
            import torch
            if torch.cuda.is_available():
                _dev = torch.device(device)
                scorer.to_torch(_dev)
            else:
                analysis._m('map_celltypes (lda): CUDA not available, scoring on CPU')
        except ImportError:
            analysis._m('map_celltypes (lda): torch not installed, scoring on CPU')
    M = np.full((n_R, n_C), -1, dtype=np.int32)
    P = np.full((n_R, n_C), -1.0, dtype=np.float32)
    # The per-gene field only exists inside this loop, so the marker level a voxel
    # would be filtered on has to be taken here (see ssaft._marker_corr.marker_score).
    # The score's axis is each celltype's OWN markers with no MIN_MARKER_GENES
    # fallback: that fallback exists so a 2-point Pearson stays filterable, and a mean
    # over one gene has no such degeneracy. Falling back to it here would leave a
    # 1-2 marker celltype scored 0 -> no cutoff -> no filtering at all.
    score_idx = marker_sets if marker_sets is not None else marker_idx
    MS = np.zeros((n_R, n_C), dtype=np.float32) if score_idx is not None else None
    _Mfull_t = None

    if _dev is not None:
        from ._count import band_rows_for_budget
        chunk_h = band_rows_for_budget(n_G, n_C, chunk_h,
                                       budget_bytes=gpu_target_bytes)
        if sub_batch is None:
            # the one temporary that scales with the gene axis is (n_G x sub_batch);
            # a quarter of the budget, so it sits alongside the band rather than
            # competing with it
            sub_batch = max(4096, int(gpu_target_bytes // 16) // max(n_G, 1))
        # The dictionary is expanded back onto the FULL gene axis, zeros at the
        # excluded genes. Their coefficient is then 0, so the contraction is the same
        # number as subsetting the block would give -- and the block, which is the big
        # array, never has to be gathered along its gene axis at all.
        _tlb_full = scorer._tlb
        if gi is not None:
            import torch
            _tlb_full = torch.zeros((scorer.K, n_G), device=_dev,
                                    dtype=scorer._tlb.dtype)
            _tlb_full[:, torch.as_tensor(gi, device=_dev)] = scorer._tlb
        if analysis.verbose:
            analysis._m(f'map_celltypes (lda): {chunk_h} rows/band '
                        f'({n_G * chunk_h * n_C * 4 / 2**30:.2f} GiB), '
                        f'{sub_batch} voxels/batch')

    bands = [(r0, min(r0 + chunk_h, n_R)) for r0 in range(0, n_R, chunk_h)]
    bands = [(a, b) for a, b in bands if (norm[a:b] > vf_norm_threshold).any()]
    from ._count import iter_field_bands
    it = iter_field_bands(zg, bands)
    if analysis.verbose:
        it = tqdm(it, total=len(bands), desc='map_celltypes (lda)')
    for r0, r1, raw in it:
        h = r1 - r0
        flat = (norm[r0:r1] > vf_norm_threshold).reshape(-1)
        pos = np.where(flat)[0]
        xy = None
        if prior is not None:
            rr, cc = np.divmod(pos, n_C)        # grid indices: column = x, row = y
            xy = np.stack([cc * sd + off[0], (rr + r0) * sd + off[1]], 1)
        nf = norm[r0:r1].reshape(-1)[pos]
        if _dev is None:
            block = raw.reshape(n_G, h * n_C).T
            if gi is not None:
                block = block[:, gi]
            X = _pseudo_counts(block[pos], nf, size_after_normalization, veff,
                               dtype=np.float32)
            lab, sc = scorer(X, xy)
            _msv = None
            if MS is not None:
                from ._marker_corr import marker_score
                _msv = marker_score(X / max(veff, 1e-12), lab, score_idx)
            del block, X
        else:
            import torch
            Bt = torch.as_tensor(raw.reshape(n_G, h * n_C), device=_dev)
            if MS is not None and _Mfull_t is None:
                _Mm = _marker_mean_matrix(score_idx,
                                          len(gi) if gi is not None else n_G,
                                          beta.shape[0])
                _Mfull_t = torch.zeros((n_G, beta.shape[0]), device=_dev)
                if gi is not None:
                    _Mfull_t[torch.as_tensor(gi, device=_dev)] = \
                        torch.as_tensor(_Mm, device=_dev)
                else:
                    _Mfull_t = torch.as_tensor(_Mm, device=_dev)
            pos_t = torch.as_tensor(pos, device=_dev)
            s_t = (torch.as_tensor(nf, device=_dev, dtype=torch.float32)
                   * (float(veff) / float(size_after_normalization)))
            lab = np.empty(pos.size, np.int32); sc = np.empty(pos.size, np.float32)
            _msv = np.empty(pos.size, np.float32) if MS is not None else None
            for a in range(0, pos.size, sub_batch):
                b = min(pos.size, a + sub_batch)
                # gene-major slice of THIS batch's voxels; no host copy, no transpose
                W = torch.expm1_(Bt.index_select(1, pos_t[a:b]))
                sb = s_t[a:b].unsqueeze(1)
                Z = (_tlb_full @ W).T * sb
                lab[a:b], sc[a:b] = scorer.torch_call_ll(
                    Z, None if xy is None else xy[a:b])
                if MS is not None:
                    ms = (_Mfull_t.T @ W).T * sb
                    _msv[a:b] = (ms.gather(
                        1, torch.as_tensor(lab[a:b], device=_dev).long().unsqueeze(1))
                        .squeeze(1) / max(veff, 1e-12)).cpu().numpy()
                    del ms
                del W, Z
            del Bt, pos_t, s_t
            torch.cuda.empty_cache()
        mf = M[r0:r1].reshape(-1); pf = P[r0:r1].reshape(-1)
        mf[pos] = lab
        pf[pos] = sc
        M[r0:r1] = mf.reshape(h, n_C); P[r0:r1] = pf.reshape(h, n_C)
        if MS is not None:
            sf = MS[r0:r1].reshape(-1)
            sf[pos] = _msv
            MS[r0:r1] = sf.reshape(h, n_C)
        del raw
    if analysis.verbose:
        _sig = int((norm > vf_norm_threshold).sum()); _tot = int((norm > 0).sum())
        print(f'[lda] assigned {int((M >= 0).sum())} voxels '
              f'(vf_norm>{vf_norm_threshold:.4g} keeps {_sig}/{_tot} = {_sig/max(_tot,1):.1%} of the field)')
    out = _write_maps(zg, ds, M, P, sd, overwrite=overwrite,
                      score_key='max_probabilities', method='lda')
    if MS is not None:
        if 'celltype_marker_score' in zg:
            del zg['celltype_marker_score']
        zg['celltype_marker_score'] = MS.T[:, :, None].astype(np.float32)
        ds.celltype_marker_score = np.asarray(zg['celltype_marker_score'][:])
    return out
