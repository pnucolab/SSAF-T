import numcodecs
from multiprocessing.pool import ThreadPool
import pickle
import dask
import dask.array as da
import numpy as np
import pandas as pd
import zarr

import multiprocessing
import sys, os
import warnings
import gc

from sklearn import preprocessing
import scipy
from scipy import ndimage
from sklearn.decomposition import PCA
from tempfile import TemporaryDirectory
from sklearn.neighbors import kneighbors_graph
import igraph as ig
from sklearn.cluster import DBSCAN, OPTICS
from skimage import filters
from skimage.morphology import disk
from skimage import measure
import subprocess
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering

from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import psutil
import math

try:
    import scanpy as sc
    import anndata as ad
    _SCANPY_AVAILABLE = True
except ImportError:
    _SCANPY_AVAILABLE = False
from sklearn.neighbors import LocalOutlierFactor
from tqdm import tqdm


from scipy.ndimage import map_coordinates

from packaging import version

try:
    from .utils import check_avx512f, calc_corrmap, calc_kde, accumulate_gene, accumulate_gene_vector, normalize_vf
    _UTILS_AVAILABLE = True
except ImportError:
    _UTILS_AVAILABLE = False
    def check_avx512f():
        return False
    
    def calc_corrmap(*args, **kwargs):
        raise NotImplementedError("C++ extension not available. Please build cellman properly.")
    
    def calc_kde(*args, **kwargs):
        raise NotImplementedError("C++ extension not available. Please build cellman properly.")

try:
    import spatialdata as sd
    from spatialdata import SpatialData
    _SPATIALDATA_AVAILABLE = True
except ImportError:
    _SPATIALDATA_AVAILABLE = False
    SpatialData = None


def _nest_rings(g):
    """Rebuild the shell/hole nesting of a polygon that came out of
    :func:`spatialdata.to_polygons`.

    ``to_polygons`` turns EVERY contour of a label into its own polygon and
    packs them into a ``MultiPolygon``, so a cell with an interior hole comes
    back as nested shells: an invalid geometry (OGC forbids overlapping parts)
    whose area ADDS the hole instead of subtracting it. A label can also be
    diagonally pinched into two pieces that genuinely are separate parts, so
    "everything but the largest is a hole" is wrong too.

    Classify by nesting parity instead — a part inside an odd number of others
    is a hole, and belongs to the smallest part that contains it.
    """
    from shapely.geometry import MultiPolygon, Polygon
    if g is None or g.is_empty or g.geom_type != "MultiPolygon":
        return g
    parts = sorted(g.geoms, key=lambda p: p.area, reverse=True)
    n = len(parts)
    # Which part contains which, by spatial index rather than by comparing every pair. A cell has
    # one or two parts and the pairs are free; a cell type EXCLUDED from the watershed does not --
    # it comes back as one body covering most of the section, perforated by every vessel and every
    # other cell inside it, so one z layer can hand back tens of thousands of rings and the pairwise
    # pass becomes ~1e8 containment tests on many-vertex polygons (measured: hours per layer on a
    # heart section). Containment implies envelope containment, so the tree returns a superset and
    # the predicate still decides -- same answer, without the square.
    if n > 8:
        from shapely import STRtree
        tree = STRtree(parts)
        # 'within': the predicate runs as predicate(input, tree_item), so this asks which tree
        # parts the input sits inside -- the parts that CONTAIN it
        depths = [len(set(int(j) for j in tree.query(p, predicate="within")) - {i})
                  for i, p in enumerate(parts)]
    else:
        depths = [sum(1 for q in parts if q is not p and q.contains(p)) for p in parts]
    shells = [p for p, d in zip(parts, depths) if not d % 2]
    holes = [p for p, d in zip(parts, depths) if d % 2]
    inner = [[] for _ in shells]
    if shells and holes:
        if len(shells) > 8:
            from shapely import STRtree
            stree = STRtree(shells)
            owners = [[int(j) for j in stree.query(h, predicate="within")] for h in holes]
        else:
            owners = [[j for j, s in enumerate(shells) if s.contains(h)] for h in holes]
        for h, own in zip(holes, owners):
            if not own:
                continue
            # a hole belongs to the SMALLEST shell containing it (ties keep both, as before)
            lo = min(shells[j].area for j in own)
            for j in own:
                if shells[j].area == lo:
                    inner[j].append(h)
    out = [Polygon(s.exterior, [h.exterior for h in hs]) for s, hs in zip(shells, inner)]
    if not out:
        return g
    return out[0] if len(out) == 1 else MultiPolygon(out)


def _watershed3d_tile(slab, seeds_xyz, seeds_ct, seeds_id, x0, x1, y0, y1, pad, exclude=()):
    """One padded xy tile of the 3D celltype-constrained watershed; returns ``(core, x0, y0)``.

    ``slab`` is the ``(x, y, z)`` cell-type map. Each seed is one voxel ``(x, y, z)`` of its
    celltype; for each celltype the flood is the watershed of the Euclidean distance to that
    celltype's seeds, confined to that celltype's voxels. A tile keeps only its core; ``pad``
    exceeds a cell, so a voxel is decided by the same seeds in whichever tile it is computed.
    """
    from skimage.segmentation import watershed
    X, Y, _ = slab.shape
    px0, px1, py0, py1 = max(0, x0 - pad), min(X, x1 + pad), max(0, y0 - pad), min(Y, y1 + pad)
    sub = np.asarray(slab[px0:px1, py0:py1, :])
    lab = np.zeros(sub.shape, np.int32)
    sel = ((seeds_xyz[:, 0] >= px0) & (seeds_xyz[:, 0] < px1)
           & (seeds_xyz[:, 1] >= py0) & (seeds_xyz[:, 1] < py1))
    if sel.any():
        sx, sy, sz = seeds_xyz[sel, 0] - px0, seeds_xyz[sel, 1] - py0, seeds_xyz[sel, 2]
        sct, sid = seeds_ct[sel], seeds_id[sel]
        for c in np.unique(sct):
            if int(c) in exclude:          # left whole, labelled by connected component instead
                continue
            m = sct == c
            mask = sub == c
            if not mask.any():
                continue
            # the celltype's bounding box: every seed of c sits on a c voxel, so the distance to the
            # nearest seed and the masked flood inside it are the same as on the whole tile (and the
            # raster order the watershed breaks ties by is preserved); most of a tile, and most of
            # its z range on a thin section, is not c
            b = [np.flatnonzero(mask.any(axis=ax)) for ax in ((1, 2), (0, 2), (0, 1))]
            (a0, a1), (b0, b1), (c0, c1) = ((v[0], v[-1] + 1) for v in b)
            mk = np.zeros((a1 - a0, b1 - b0, c1 - c0), np.int32)
            mk[sx[m] - a0, sy[m] - b0, sz[m] - c0] = sid[m]
            box = (slice(a0, a1), slice(b0, b1), slice(c0, c1))
            lab_c = watershed(ndimage.distance_transform_edt(mk == 0), mk, mask=mask[box])
            hit = lab_c > 0
            lab[box][hit] = lab_c[hit]
    cx0, cy0 = x0 - px0, y0 - py0
    return lab[cx0:cx0 + (x1 - x0), cy0:cy0 + (y1 - y0), :], x0, y0


def _blob_labels(mask, first_id, out):
    """Label ``mask``'s connected components ``first_id, first_id+1, ...`` into ``out``; returns how many.

    This is what a cell type that is NOT segmented gets: one object per connected piece of it, so a
    syncytium (or any tissue the nucleus-per-cell model does not fit) stays the one body the cell-type
    map found instead of being cut into one fragment per nucleus.
    """
    cc, n = ndimage.label(mask)
    if n:
        hit = cc > 0
        out[hit] = cc[hit].astype(out.dtype) + np.int32(first_id - 1)
    return int(n)


def _cap_label_volume(labels, seed_xyz, voxel_vol_um3, max_volume_um3, counts=None):
    """Trim every cell to the voxels nearest its own seed, up to ``max_volume_um3``.

    The 3D counterpart of :func:`_cap_label_area`, and the right bound for a 3D run: a cap on each
    LAYER's cross-section leaves the total free to grow with the number of layers a cell spans
    (200 um^2 over 30 layers is 6,000 um^3), whereas a cell's size in 3D is its volume. Distance is
    measured in 3D to the seed voxel, which sits at distance 0 and so is never what gets trimmed.

    ``seed_xyz`` is indexed by label id. Only labels already over the cap are touched, and their
    voxels are gathered a layer at a time, so the cost is set by the oversized cells rather than by
    the volume. ``labels`` is modified in place; returns the ids that were trimmed.
    """
    if not max_volume_um3:
        return np.empty(0, dtype=np.int64)
    cap = max(1, int(float(max_volume_um3) // float(voxel_vol_um3)))
    n_lab = labels.max() + 1 if counts is None else len(counts)
    if counts is None:
        counts = np.zeros(int(n_lab), np.int64)
        for k in range(labels.shape[2]):
            counts += np.bincount(labels[:, :, k].ravel(), minlength=int(n_lab))[:int(n_lab)]
    over = np.flatnonzero(counts > cap)
    over = over[over > 0]
    if over.size == 0:
        return np.empty(0, dtype=np.int64)
    sel = np.zeros(int(n_lab), bool)
    sel[over] = True
    xs, ys, zs, labs = [], [], [], []
    for k in range(labels.shape[2]):
        plane = labels[:, :, k]
        hit = sel[plane]
        if not hit.any():
            continue
        a, b = np.nonzero(hit)
        xs.append(a); ys.append(b)
        zs.append(np.full(a.size, k, dtype=np.int64))
        labs.append(plane[a, b])
    x = np.concatenate(xs); y = np.concatenate(ys); z = np.concatenate(zs)
    lab = np.concatenate(labs).astype(np.int64)
    d2 = ((x - seed_xyz[lab, 0]) ** 2 + (y - seed_xyz[lab, 1]) ** 2
          + (z - seed_xyz[lab, 2]) ** 2)
    order = np.lexsort((d2, lab))
    lab_s = lab[order]
    uniq, first, cnt = np.unique(lab_s, return_index=True, return_counts=True)
    rank = np.arange(order.size, dtype=np.int64) - np.repeat(first, cnt)
    drop = order[rank >= cap]
    if drop.size:
        labels[x[drop], y[drop], z[drop]] = 0
    return uniq[cnt > cap]


def _cap_label_area(lab2d, seed_a, seed_b, voxel_area_um2, max_area_um2):
    """Trim every label on one plane to the voxels nearest its own seed, up to ``max_area_um2``.

    The flood is poured on the nucleus, so the nucleus is what a cap must never take: the seed
    voxel sits at distance 0 and is kept first, and the cells that ran away lose their outside,
    inwards, until they fit. Without this a nucleus sitting alone in a large region of its own
    celltype floods that whole region -- the watershed has nothing else to stop it, and on a
    tissue edge or a sparse celltype that produces single "cells" of hundreds of µm².

    ``seed_a`` / ``seed_b`` give each label's seed position along axis 0 / axis 1 of ``lab2d``,
    indexed by label id. Voxels equally far from the seed are kept in index order. ``lab2d`` is
    modified in place; returns the ids of the labels that were trimmed.
    """
    if not max_area_um2:
        return np.empty(0, dtype=np.int64)
    cap = max(1, int(float(max_area_um2) // float(voxel_area_um2)))
    aa, bb = np.nonzero(lab2d)                   # works on a strided view (labels[:, :, k])
    if aa.size == 0:
        return np.empty(0, dtype=np.int64)
    lab = lab2d[aa, bb]
    d2 = ((aa - seed_a[lab]) ** 2 + (bb - seed_b[lab]) ** 2)
    order = np.lexsort((d2, lab))
    lab_sorted = lab[order]
    uniq, first, counts = np.unique(lab_sorted, return_index=True, return_counts=True)
    over = (np.arange(order.size, dtype=np.int64) - np.repeat(first, counts)) >= cap
    if over.any():
        drop = order[over]
        lab2d[aa[drop], bb[drop]] = 0
    return uniq[counts > cap]


def _plane_polygons(lab_xy, sd_um, ox, oy, simplify=None):
    """``[(label, polygon)]`` of one ``(x, y)`` label plane, in the map's µm frame.

    Traced along voxel boundaries by :func:`spatialdata.to_polygons` (one chunk, so no seams),
    shell/hole nesting restored by :func:`_nest_rings`.
    """
    import geopandas as gpd
    from spatialdata import to_polygons
    from spatialdata.models import Labels2DModel
    lab = np.ascontiguousarray(np.asarray(lab_xy).T)
    if not lab.any():
        return []
    poly = to_polygons(Labels2DModel.parse(lab, chunks=lab.shape))
    gs = gpd.GeoSeries([_nest_rings(g) for g in poly.geometry]).affine_transform(
        [sd_um, 0, 0, sd_um, ox, oy])
    if simplify:
        gs = gs.simplify(simplify)
    return [(int(lb), g) for lb, g in zip(poly.index.to_numpy(), gs.to_numpy())
            if g is not None and not g.is_empty]


def _binary_fill_holes_fast(mask):
    """``scipy.ndimage.binary_fill_holes`` with an optional CuPy backend.

    The GPU path (``cupyx.scipy.ndimage.binary_fill_holes``) uses the same
    default connectivity-1 structuring element as SciPy and is a purely
    boolean morphological operation, so the result is bit-identical to the
    CPU path. Any failure (CuPy missing, no CUDA device, out of memory)
    falls back transparently to SciPy. Set ``CELLMAN_DISABLE_GPU=1`` to
    force the CPU path.
    """
    if os.environ.get("CELLMAN_DISABLE_GPU"):
        return ndimage.binary_fill_holes(mask)
    try:
        import cupy as cp
        from cupyx.scipy.ndimage import binary_fill_holes as _gpu_fill
        d_mask = cp.asarray(mask)
        d_out = _gpu_fill(d_mask)
        out = cp.asnumpy(d_out)
        del d_mask, d_out
        cp.get_default_memory_pool().free_all_blocks()
        return out
    except Exception:
        return ndimage.binary_fill_holes(mask)

if check_avx512f():
    from .utils import corr, calc_ctmap
else:
    corr = lambda a, b: np.corrcoef(a, b)[0, 1]
    def calc_ctmap(x, Y, _):
        mean_x = np.mean(x)
        mean_Y = np.mean(Y, axis=1)
        x_diff = x - mean_x
        Y_diff = Y - mean_Y[:, None]
        numerator = np.einsum('i,ji->j', x_diff, Y_diff)
        denominator = np.sqrt(np.einsum('i,i->', x_diff, x_diff) * np.einsum('ji,ji->j', Y_diff, Y_diff))
        correlation = np.where(denominator == 0, np.nan, numerator / denominator)
        return correlation


#: Sentinel separating "the caller said nothing" from "the caller passed the default".
#: What it buys is ``hsm_f``: a deliberate ``hsm_f=`` fails loudly on a store that
#: cannot support the cut, while the default quietly stands down.
_UNSET = object()


from ._count import remove_array as _remove_array

class Analysis(object):
    """
    A class to run a ``ssaft`` analysis.
    
    Supports both pandas.DataFrame and spatialdata.SpatialData as input data formats.
    
    For DataFrame input, required columns:
    - 'gene': gene names (can be index or column)
    - 'x', 'y': spatial coordinates in micrometers
    - 'z': depth coordinate (optional, for 3D data)
    
    For SpatialData input:
    - Points element with spatial coordinates and gene information
    - Automatically detects coordinate systems and gene annotation fields
    - Supports both 2D and 3D spatial data

    :param dataset: A Dataset object.
    :type dataset: Dataset
    :param ncores: Number of cores for parallel computation. If a negative value is given,
        ((# of all available cores on system) - abs(ncores)) cores will be used.
    :type ncores: int
    :param verbose: If True, then it prints out messages during the analysis.
    :type verbose: bool
    """
    def __init__(self, dataset, ncores=1, verbose=False):
        self.dataset = dataset
        if not ncores > 0:
            ncores += multiprocessing.cpu_count()
        if ncores > multiprocessing.cpu_count():
            ncores = multiprocessing.cpu_count()
        if not ncores > 0:
            raise ValueError("Invalid number of cores.")
        os.environ["OMP_NUM_THREADS"] = str(ncores)
        os.environ["OPENBLAS_NUM_THREADS"] = str(ncores)
        os.environ["MKL_NUM_THREADS"] = str(ncores)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncores)
        os.environ["NUMEXPR_NUM_THREADS"] = str(ncores)
        dask.config.set(pool=ThreadPool(ncores))
        self.ncores = ncores
        self.verbose = verbose

        

        try:
            zk = set(self.dataset.zarr_group.array_keys())
            zk |= set(self.dataset.zarr_group.group_keys())
        except Exception:
            try:
                zk = set(self.dataset.zarr_group.keys())
            except Exception:
                zk = set()

        if self.verbose:
            self._m(f"zarr keys: {sorted(zk) if zk else '(empty)'}")

        if 'kde_computed' in zk and all(self.dataset.zarr_group['kde_computed']):
            self._m("Loaded existing KDE results.")
            self._load_kde()

        # The dimensionality downsample_vector decided (2 or 3). Mapping reads it so the
        # dictionary and the field it is scored against are built in the same dimension.
        self.n_dims = None
        try:
            _nd = self.dataset.zarr_group.attrs.get('n_dims')
            self.n_dims = int(_nd) if _nd is not None else None
        except Exception:
            self.n_dims = None

        if 'coordinate_offset' in zk:
            self.dataset.coordinate_offset = self.dataset.zarr_group['coordinate_offset'][:]
        if 'coordinate_bounds' in zk:
            self.dataset.coordinate_bounds = self.dataset.zarr_group['coordinate_bounds'][:]
        if 'original_coordinate_bounds' in zk:
            self.dataset.original_coordinate_bounds = self.dataset.zarr_group['original_coordinate_bounds'][:]

        if 'downsampled_vectors' in zk:
            self._m("Loaded existing local maxima.")
            self.dataset.downsampled_vectors = tuple(self.dataset.zarr_group['downsampled_vectors'][:])

        if 'normalized_vectors' in zk:
            self._m("Loaded a precomputed normalized vector.")
            self.dataset.normalized_vectors = self.dataset.zarr_group['normalized_vectors'][:]

        if 'genes' in zk:
            self._m("Loaded existing gene list.")
            self.dataset.genes = list(self.dataset.zarr_group['genes'][:])

        if 'vf_normalized' in zk:
            try :
                self._m("Loaded a precomputed normalized vector field.")
                self.dataset.vf_normalized = da.from_zarr(self.dataset.zarr_group['vf_normalized'])
            except:
                self._m("Failed to load precomputed normalized vector field. Perhaps the zarr data is in chunks.")

        if 'vf_scaled' in zk and 'scaled_vectors' in zk:
            self.dataset.vf_scaled = da.from_zarr(self.dataset.zarr_group['vf_scaled'])
            self.dataset.scaled_vectors = self.dataset.zarr_group['scaled_vectors'][:]

        if 'cluster_labels' in zk:
            self._m("Loaded existing cluster labels.")
            self.dataset.cluster_labels = self.dataset.zarr_group['cluster_labels'][:]
        if 'centroids' in zk:
            self.dataset.centroids = self.dataset.zarr_group['centroids'][:]

        if 'tsne' in zk:
            self._m("Loaded an existing t-SNE result.")
            self.dataset.tsne = self.dataset.zarr_group['tsne'][:]
        
        if 'umap' in zk:
            self._m("Loaded an existing UMAP result.")
            self.dataset.umap = self.dataset.zarr_group['umap'][:]
        
        if 'celltype_maps' in zk:
            self._m("Loaded existing cell type maps.")
            self.dataset.celltype_maps = self.dataset.zarr_group['celltype_maps'][:]
            if 'max_correlations' in zk:
                self.dataset.max_correlations = self.dataset.zarr_group['max_correlations'][:]
            else:
                self.dataset.max_correlations = None
            if 'max_probabilities' in zk:
                self.dataset.max_probabilities = self.dataset.zarr_group['max_probabilities'][:]
            else:
                self.dataset.max_probabilities = None

        if 'filtered_celltype_maps' in zk:
            self._m("Loaded existing filtered cell type maps.")
            self.dataset.filtered_celltype_maps = self.dataset.zarr_group['filtered_celltype_maps'][:]

        if 'celltype_binned_centers' in zk and 'celltype_binned_counts' in zk:
            self._m("Loaded existing cell type binned centers and counts.")
            self.dataset.celltype_binned_centers = self.dataset.zarr_group['celltype_binned_centers'][:]
            self.dataset.celltype_binned_counts = self.dataset.zarr_group['celltype_binned_counts'][:]
        
        if 'inferred_domains' in zk and 'inferred_domains_cells' in zk:
            self._m("Loaded existing inferred domains.")
            self.dataset.inferred_domains = self.dataset.zarr_group['inferred_domains'][:]
            self.dataset.inferred_domains_cells = self.dataset.zarr_group['inferred_domains_cells'][:]

        if 'inferred_domains_compositions' in self.dataset.zarr_group:
            self._m("Loaded existing inferred domains compositions.")
            self.dataset.inferred_domains_compositions = self.dataset.zarr_group['inferred_domains_compositions'][:]
        
        if 'watershed_segments' in self.dataset.zarr_group and 'watershed_celltype_maps' in self.dataset.zarr_group:
            self._m("Loaded existing watershed segmentations.")
            self.dataset.watershed_segments = self.dataset.zarr_group['watershed_segments'][:]
            self.dataset.watershed_celltype_maps = self.dataset.zarr_group['watershed_celltype_maps'][:]
        
        if 'transferred_cluster_labels' in self.dataset.zarr_group and 'transferred_cluster_correlations' in self.dataset.zarr_group:
            self._m("Loaded existing transferred cluster labels.")
            self.dataset.transferred_cluster_labels = self.dataset.zarr_group['transferred_cluster_labels'][:]
            self.dataset.transferred_cluster_correlations = self.dataset.zarr_group['transferred_cluster_correlations'][:]

    def _m(self, message):
        if self.verbose:
            print(message, flush=True)
    
    def _load_kde(self):
        assert 'kde_computed' in self.dataset.zarr_group, "KDE has not been computed!"
        assert all(self.dataset.zarr_group['kde_computed']), "KDE data is incomplete!"
        self.dataset.genes = list(self.dataset.zarr_group['genes'][:])
        if 'vf' in self.dataset.zarr_group:
            self.dataset._vf = da.from_zarr(self.dataset.zarr_group['vf'])
        self.dataset._vf_norm = da.from_zarr(self.dataset.zarr_group['vf_norm'])
        self.dataset.sampling_distance = self.dataset.zarr_group['vf_params'][0]
        self.dataset.bandwidth = self.dataset.zarr_group['vf_params'][1]
        self.dataset.shape = self.dataset.vf_norm.shape
        self.dataset.ndim = 2 if self.dataset.vf_norm.shape[-1] == 1 else 3
        # vf_params[1] is the kernel's OWN scale (H for a compact kernel's field), so
        # only the peak formula depends on which kernel wrote it.
        from ._count import kernel_peak
        self.dataset.kde_kernel = str(self.dataset.zarr_group.attrs.get('kde_kernel', 'gaussian'))
        self.dataset.expression_threshold = kernel_peak(
            self.dataset.bandwidth, self.dataset.ndim, self.dataset.kde_kernel)
        self.dataset.norm_threshold = self.dataset.expression_threshold * 2
        
    #: Elements ssaft itself writes into the input SpatialData. They are never
    #: candidates when an input element has to be chosen — otherwise a second run
    #: could silently read its own output back in (``ssaft_cells`` even has the
    #: shape of a count table, so nothing would raise).
    _OWN_ELEMENTS = frozenset({
        'ssaft_cells', 'ssaft_downsampled_vectors', 'ssaft_cell_boundaries',
        'ssaft_cell_boundaries_2d', 'ssaft_cell_boundaries_3d', 'ssaft_cell_labels_3d',
        'ssaft_celltype_map', 'ssaft_celltype_map_rgb',
    })

    def _pick_element(self, container, explicit, *, kind, param, required=True):
        """Resolve one input element, refusing to guess when it is ambiguous.

        An explicit key is validated and returned. Otherwise the choice is made
        only when it is unambiguous — exactly one candidate once ssaft's own
        outputs are excluded. With several candidates the caller must say which,
        because picking "the first" silently changes the result depending on the
        order elements happen to sit in.

        :returns: the chosen key, or None when nothing is available and
            ``required`` is False.
        """
        container = container or {}
        if explicit is not None:
            if explicit not in container:
                raise ValueError(f"{kind} element '{explicit}' not found. "
                                 f"Available: {sorted(container)}")
            return explicit
        candidates = [k for k in container if k not in self._OWN_ELEMENTS]
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            if required:
                raise ValueError(f"no {kind} element in the SpatialData "
                                 f"(found: {sorted(container)})")
            return None
        raise ValueError(
            f"{len(candidates)} {kind} elements — pass {param}= to say which one: "
            f"{sorted(candidates)}")

    def _spatialdata_to_dataframe(self, sdata, element_name=None, coordinate_system=None):
        """
        Convert SpatialData object to the DataFrame format used by this package.
        
        :param sdata: SpatialData object containing points data
        :type sdata: spatialdata.SpatialData
        :param element_name: Name of the points element to extract. If None, uses the first available.
        :type element_name: str
        :param coordinate_system: Coordinate system to use. If None, uses the first available.
        :type coordinate_system: str
        :returns: DataFrame with 'gene', 'x', 'y' columns (and 'z' if 3D) where gene is a column, not index
        :rtype: pandas.DataFrame
        """
        if not _SPATIALDATA_AVAILABLE:
            raise ImportError("SpatialData is not available. Install with: pip install spatialdata")
        
        if not isinstance(sdata, SpatialData):
            raise TypeError(f"Expected SpatialData object, got {type(sdata)}")
        
        element_name = self._pick_element(getattr(sdata, 'points', {}), element_name,
                                          kind='points', param='element_name')
        points_df = sdata.points[element_name]
        
        if hasattr(points_df, 'compute'):
            points_df = points_df.compute()
        
        if coordinate_system is None:
            if hasattr(points_df, 'attrs') and 'transform' in points_df.attrs:
                coord_systems = list(points_df.attrs['transform'].keys()) if points_df.attrs['transform'] else []
                if coord_systems:
                    coordinate_system = coord_systems[0]
        
        spatial_coords = ['x', 'y']
        if 'z' in points_df.columns:
            spatial_coords.append('z')
        
        if not all(coord in points_df.columns for coord in spatial_coords[:2]):
            raise ValueError(f"Required spatial coordinates 'x', 'y' not found in points data. Available columns: {points_df.columns.tolist()}")
        
        gene_col = None
        possible_gene_cols = ['gene', 'gene_symbol', 'feature_name', 'target']
        for col in possible_gene_cols:
            if col in points_df.columns:
                gene_col = col
                break
        
        if gene_col is None:
            raise ValueError(f"No gene information found. Expected one of: {possible_gene_cols}. Available columns: {points_df.columns.tolist()}")
        
        if 'is_gene' in points_df.columns:
            points_df = points_df[points_df['is_gene'] == True]

        result_cols = [gene_col] + spatial_coords
        result_df = points_df[result_cols].copy()
        
        if pd.api.types.is_categorical_dtype(result_df[gene_col]):
            result_df[gene_col] = result_df[gene_col].astype(str)
        
        for coord in spatial_coords:
            if not pd.api.types.is_numeric_dtype(result_df[coord]):
                result_df[coord] = pd.to_numeric(result_df[coord], errors='coerce')
        
        if gene_col != 'gene':
            result_df = result_df.rename(columns={gene_col: 'gene'})
        
        return result_df

    def _validate_input_data(self, data, element_name=None, width=None, height=None, depth=None):
        """
        Validate and normalize input data for analysis.
        
        :param data: Input data - either DataFrame or SpatialData object
        :param element_name: Name of the points element to extract from SpatialData
        :param width: Image width in um (optional, auto-detected if not provided)
        :param height: Image height in um (optional, auto-detected if not provided)  
        :param depth: Image depth in um (optional, auto-detected if not provided)
        :returns: Tuple of (DataFrame, width, height, depth, is_3d, coordinate_metadata)
        """
        if _SPATIALDATA_AVAILABLE and isinstance(data, SpatialData):
            df = self._spatialdata_to_dataframe(data, element_name)
            
            x_min, x_max = df['x'].min(), df['x'].max()
            y_min, y_max = df['y'].min(), df['y'].max()
            z_min = z_max = 0
            if 'z' in df.columns:
                z_min, z_max = df['z'].min(), df['z'].max()
            
            actual_width = float(x_max - x_min)
            actual_height = float(y_max - y_min)
            actual_depth = float(z_max - z_min) if 'z' in df.columns else 1
            
            z_threshold = 1.0
            is_3d = ('z' in df.columns and 
                    actual_depth > z_threshold and 
                    df['z'].nunique() > 1)
            
            if width is None:
                width = actual_width
            if height is None:
                height = actual_height
            if depth is None and is_3d:
                depth = actual_depth
            elif depth is None:
                depth = 1
                
            df_normalized = df.copy()
            df_normalized['x'] = df_normalized['x'] - x_min
            df_normalized['y'] = df_normalized['y'] - y_min
            if 'z' in df_normalized.columns:
                df_normalized['z'] = df_normalized['z'] - z_min
            
            coordinate_metadata = {
                'offset': [float(x_min), float(y_min), float(z_min)],
                'bounds': [width, height, depth],
                'original_bounds': [[float(x_min), float(x_max)], 
                                [float(y_min), float(y_max)], 
                                [float(z_min), float(z_max)]]
            }
                
            return df_normalized, width, height, depth, is_3d, coordinate_metadata
        
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
            
            if df.index.name != 'gene':
                if 'gene' in df.columns:
                    df = df.set_index('gene')
                else:
                    raise ValueError("DataFrame must have 'gene' column or 'gene' as index")
            
            required_cols = ['x', 'y']
            if depth is not None and depth > 1:
                required_cols.append('z')
                
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            x_min, x_max = df['x'].min(), df['x'].max()
            y_min, y_max = df['y'].min(), df['y'].max()
            z_min = z_max = 0
            if 'z' in df.columns:
                z_min, z_max = df['z'].min(), df['z'].max()
            
            actual_width = float(x_max - x_min)
            actual_height = float(y_max - y_min)
            actual_depth = float(z_max - z_min) if 'z' in df.columns else 1
            
            z_threshold = 1.0
            is_3d = ('z' in df.columns and 
                    actual_depth > z_threshold and 
                    df['z'].nunique() > 1)
            
            if width is None:
                width = actual_width
            if height is None:
                height = actual_height
            if depth is None and is_3d:
                depth = actual_depth
            elif depth is None:
                depth = 1
                
            df_normalized = df.copy()
            df_normalized['x'] = df_normalized['x'] - x_min
            df_normalized['y'] = df_normalized['y'] - y_min
            if 'z' in df_normalized.columns:
                df_normalized['z'] = df_normalized['z'] - z_min
            
            coordinate_metadata = {
                'offset': [float(x_min), float(y_min), float(z_min)],
                'bounds': [width, height, depth],
                'original_bounds': [[float(x_min), float(x_max)], 
                                [float(y_min), float(y_max)], 
                                [float(z_min), float(z_max)]]
            }
                
            return df_normalized, width, height, depth, is_3d, coordinate_metadata
        
        else:
            raise TypeError(f"Input data must be pandas.DataFrame or spatialdata.SpatialData, got {type(data)}")
        
    def calc_chunk(self, vx, vy, vz, g, mode="kde", rho=1.0, avail_frac=0.1, g2=None):
        """Compute the X-axis chunk width and chunk count that fit in available RAM.

        :param vx: Number of voxels along the X axis.
        :type vx: int
        :param vy: Number of voxels along the Y axis.
        :type vy: int
        :param vz: Number of voxels along the Z axis.
        :type vz: int
        :param g: Number of genes used to size the per-voxel byte budget.
        :type g: int
        :param mode: Memory model to use; ``"map_celltypes_cpu"`` accounts for the
            mapping buffers, any other value uses the plain KDE estimate.
        :type mode: str
        :param rho: Fraction of kept (above-threshold) voxels assumed when sizing
            the mapping buffers.
        :type rho: float
        :param avail_frac: Fraction of currently available system memory the chunk
            is allowed to occupy.
        :type avail_frac: float
        :param g2: Secondary gene count used by the mapping buffers; defaults to
            ``g`` when None.
        :type g2: int or None
        :returns: Tuple ``(cx, n_chunk)`` of chunk width in voxels and chunk count.
        :rtype: tuple(int, int)
        """
        avail = psutil.virtual_memory().available
        if mode == "map_celltypes_cpu":
            if g2 is None:
                g2 = g
            bytes_per_voxel = 4*g + 12 + rho*(4*g + 12*g2 + 40)
        else:
            bytes_per_voxel = 4*g

        bytes_per_xslice = vy * vz * bytes_per_voxel
        cx = max(1, min(vx, int(avail * avail_frac / bytes_per_xslice)))
        n_chunk = math.ceil(vx / cx)
        return cx, n_chunk


    def _downsampled_adata(self):
        """Build the AnnData that :meth:`downsample_vector` returns: ``X`` =
        normalized_vectors, ``obsm['spatial']`` = downsampled positions,
        ``obs['vf_norm']`` = per-voxel local density, ``var`` = genes. Feed it to
        ``sc.pp.pca`` / ``sc.pp.neighbors`` / ``sc.tl.leiden``."""
        ds = self.dataset
        nv = np.asarray(ds.normalized_vectors, dtype=np.float32)
        genes = getattr(ds, 'genes', None)
        if genes is None and getattr(ds, 'zarr_store', None) is not None:
            store_path = ds._zarr_store_path(ds.zarr_store)
            genes = zarr.open(os.path.join(store_path, 'genes'), mode='r')[:]
            ds.genes = genes
        if genes is None:
            genes = [f'g{i}' for i in range(nv.shape[1])]
        genes = [g.decode() if isinstance(g, bytes) else str(g) for g in genes]
        adata = ad.AnnData(
            X=nv,
            obs=pd.DataFrame(index=[f'v5_{i}' for i in range(nv.shape[0])]),
            var=pd.DataFrame(index=pd.Index(genes, name='gene')),
        )
        dv = getattr(ds, 'downsampled_vectors', None)
        if dv is not None:
            adata.obsm['spatial'] = np.stack(
                [np.asarray(dv[0]), np.asarray(dv[1])], axis=1).astype(np.float32)
        vfn = getattr(ds, 'downsampled_vf_norm', None)
        if vfn is not None and len(vfn) == adata.n_obs:
            adata.obs['vf_norm'] = np.asarray(vfn, dtype=np.float32)
        if dv is not None and len(dv) > 2 and getattr(self, 'n_dims', None) == 3:
            adata.obs['z_index'] = np.asarray(dv[2], dtype=np.int32)
        return adata

    def _downsampled_world_xy(self):
        """World-µm ``(N, 2)`` positions of the downsampled vectors.

        The count backend records them directly (``downsampled_global_xy``); the
        transcript backend stores grid indices, so they are converted with the
        vector field's own ``coordinate_offset`` + ``sampling_distance``. Either
        way the result lives in the SAME global µm frame as the staining image
        and the celltype-map labels, which is what makes the table overlay them.
        """
        zg = self.dataset.zarr_group
        if 'downsampled_global_xy' in zg:
            return np.asarray(zg['downsampled_global_xy'][:]).T.astype(np.float64)
        dv = np.asarray(zg['downsampled_vectors'][:])
        sd_um = float(zg.attrs.get(
            'sampling_distance',
            zg['vf_params'][0] if 'vf_params' in zg else 1.0))
        off = (np.asarray(zg['coordinate_offset'][:]) if 'coordinate_offset' in zg
               else np.zeros(3))
        return np.stack([off[0] + dv[0] * sd_um, off[1] + dv[1] * sd_um],
                        axis=1).astype(np.float64)

    def _celltype_names_colors(self, cmap=None):
        """``(names, colors)`` for the celltype ids 0..K-1, both length K.

        Written by :func:`~ssaft._mapping_inputs.build_centroids_from_h5ad`.
        K is ``centroids.shape[0]`` — every class the mapping can emit, including
        the ones that never win a voxel — so the palette never depends on which
        classes happen to survive filtering. Stores built before these keys
        existed fall back to the celltype-index CSV, then to positional names.
        """
        from ._mapping_inputs import celltype_colors
        zg = self.dataset.zarr_group
        K = int(zg['centroids'].shape[0]) if 'centroids' in zg else 0
        if 'celltype_names' in zg:
            names = [str(x) for x in zg['celltype_names'][:]]
        else:
            names = None
            csv = self.dataset._zarr_store_path(self.dataset.zarr_store)
            csv = os.path.join(csv, 'celltype_index.csv') if csv else None
            if csv and os.path.isfile(csv):
                idx = pd.read_csv(csv).sort_values('cluster')
                if len(idx) == K:
                    names = idx['celltype'].astype(str).tolist()
            if names is None:
                names = [f'celltype_{i}' for i in range(K)]
        colors = list(zg.attrs.get('cellman_celltype_colors', []))
        if len(colors) != len(names):
            colors = celltype_colors(len(names), cmap=cmap)
        return names, colors

    def set_celltype_palette(self, cmap=None, sdata=None, *, colors=None,
                             table_key='ssaft_downsampled_vectors',
                             labels_key='ssaft_celltype_map',
                             image_key='ssaft_celltype_map_rgb', z=None):
        """Re-fix the per-celltype palette WITHOUT re-running the mapping.

        Colour is a presentation choice, but it is stored once so every view
        agrees — so changing it means updating that one place and re-rendering
        what was derived from it. This rewrites
        ``attrs['cellman_celltype_colors']``, the annotated table's
        ``uns['cellman_annotation_colors']``, and (if a filtered map exists) the
        RGB image. The labels raster and the celltype ids are untouched.

        :param cmap: ``None`` (default) = qualitative palette sized to the class
            count; a colormap name samples it continuously instead.
        :type cmap: str or matplotlib.colors.Colormap or None
        :param sdata: SpatialData to update; defaults to the remembered one.
        :type sdata: spatialdata.SpatialData or None
        :param colors: explicit per-id colours, overriding ``cmap``.
        :type colors: list or None
        :param table_key: key of the annotated downsampled table to update.
        :type table_key: str
        :param labels_key: key of the labels element (re-parsed unchanged).
        :type labels_key: str
        :param image_key: key of the RGB image element to re-render.
        :type image_key: str
        :param z: Z index used when re-rendering a 3D map.
        :type z: int or None
        :returns: The updated SpatialData, or ``None`` if none was available.
        :rtype: spatialdata.SpatialData or None
        """
        from matplotlib.colors import to_hex, to_rgba
        from ._mapping_inputs import celltype_colors
        zg = self.dataset.zarr_group
        names, _ = self._celltype_names_colors()
        if colors is None:
            colors = celltype_colors(len(names), cmap=cmap)
        colors = [to_hex(to_rgba(c)) for c in colors]
        if len(colors) != len(names):
            raise ValueError(f'{len(colors)} colours for {len(names)} celltypes')
        zg.attrs['cellman_celltype_colors'] = colors
        self._m(f'set_celltype_palette: {len(colors)} colours '
                f"(cmap={cmap!r})" if cmap else
                f'set_celltype_palette: {len(colors)} qualitative colours')

        sdata = sdata if sdata is not None else getattr(
            self.dataset, '_original_input_data', None)
        if sdata is None:
            return None
        if table_key in getattr(sdata, 'tables', {}):
            sdata.tables[table_key].uns['cellman_annotation_colors'] = colors
        if getattr(self.dataset, 'filtered_celltype_maps', None) is not None:
            self.dataset.write_celltype_map(
                sdata, colors=colors, z=z,
                labels_key=labels_key, image_key=image_key)
        return sdata

    def _finish_map(self, sdata, table_key):
        """Attach the annotated downsampled table to ``sdata`` and return it."""
        if sdata is None:
            self._m('map_celltypes: no SpatialData given and none remembered; '
                    'store written, table skipped')
            return None
        return self._write_downsampled_table(sdata, table_key=table_key)

    def _write_downsampled_table(self, sdata, *, table_key='ssaft_downsampled_vectors'):
        """Write the annotated downsampled vectors into ``sdata.tables[table_key]``.

        ``obs['cellman_annotation']`` is REBUILT from ``cluster_labels`` — not
        copied from the annotated h5ad. The two differ: the h5ad's ``Unknown`` is
        split per source cluster into ``unknown1..N``, so the h5ad may carry (say)
        27 distinct labels where the mapping uses 36 classes. Rebuilding keeps the
        categorical's categories index-aligned with the celltype ids, the rows of
        ``centroids``, the values of ``celltype_maps``, and
        ``uns['cellman_annotation_colors']`` — one shared axis, nothing to drift.

        Vectors dropped from the mapping (``cluster_labels == -1``: Background /
        never-finalized labels) are EXCLUDED; they are background, not a class.
        """
        import pandas as pd
        from spatialdata.models import TableModel

        zg = self.dataset.zarr_group
        names, colors = self._celltype_names_colors()
        labels = np.asarray(zg['cluster_labels'][:])
        keep = labels >= 0

        adata = self._downsampled_adata()
        xy = self._downsampled_world_xy()
        if not keep.all():
            adata = adata[keep].copy()
        adata.obsm['spatial'] = xy[keep]
        adata.obs['cellman_annotation'] = pd.Categorical.from_codes(
            labels[keep], categories=names)
        adata.uns['cellman_annotation_colors'] = colors
        adata.uns['cellman_celltype_names'] = names

        if table_key in sdata.tables:
            del sdata.tables[table_key]
        sdata.tables[table_key] = TableModel.parse(adata)
        n_drop = int((~keep).sum())
        print(f"[map_celltypes] tables['{table_key}'] {adata.shape} "
              f"{len(names)} celltypes"
              + (f" (excluded {n_drop} background vectors)" if n_drop else ""))
        return sdata

    def _prepare_grid(self, sdata, element_name=None, bandwidth=2.5, sampling_distance=1.0,
                      prune_coefficient=4.3, size_after_normalization=10, n_dims=None):
        """
        Parse SpatialData, compute grid dimensions (vx, vy, vz), and store them in
        ``self.dataset.shape``.  Also persists coordinate metadata to zarr.

        ``n_dims`` (default: ``self.n_dims``, set by :meth:`downsample_vector`) decides the
        grid's depth instead of the transcripts: ``2`` gives a single-plane grid even when
        the transcripts carry z, ``3`` requires them to. ``None`` — a store downsampled
        before the dimensionality was recorded — keeps inferring it from the z range.

        Returns
        -------
        vx, vy, vz : int
        """
        locations, width, height, depth, is_3d, coord_metadata = self._validate_input_data(
            sdata, element_name, None, None, None
        )
        nd = n_dims if n_dims is not None else getattr(self, 'n_dims', None)
        if nd == 2:
            is_3d, depth = False, 1
            coord_metadata['bounds'] = list(coord_metadata['bounds'][:2]) + [1]
        elif nd == 3 and not is_3d:
            raise ValueError('n_dims=3 but the transcripts carry no usable z (no z column, '
                             'or a z range under 1 unit) — downsample with z=2')

        self.dataset.coordinate_offset = coord_metadata['offset']
        self.dataset.coordinate_bounds = coord_metadata['bounds']
        self.dataset.original_coordinate_bounds = coord_metadata['original_bounds']
        self.dataset.sampling_distance = sampling_distance
        self.bandwidth = bandwidth
        self.prune_coefficient = prune_coefficient
        self.size_after_normalization = size_after_normalization

        if locations.index.name != 'gene':
            locations = locations.set_index('gene')
        if is_3d and depth > 1:
            self.locations = locations.reindex(['x', 'y', 'z'], axis=1)
        else:
            self.locations = locations.reindex(['x', 'y'], axis=1)

        grid_xyz = np.ceil(np.array([width, height, depth]) / sampling_distance).astype(int)
        vx, vy, vz = int(grid_xyz[0]), int(grid_xyz[1]), int(grid_xyz[2])
        self.dataset.shape = (vx, vy, vz)

        for name, value in [
            ('coordinate_offset', np.array(self.dataset.coordinate_offset)),
            ('coordinate_bounds', np.array(self.dataset.coordinate_bounds)),
            ('original_coordinate_bounds', np.array(self.dataset.original_coordinate_bounds)),
        ]:
            if name in self.dataset.zarr_group:
                _remove_array(self.dataset.zarr_group, name)
            self.dataset.zarr_group[name] = value

        return vx, vy, vz

    def _parse_transcripts(self, sdata, sampling_distance=1.0, filter_is_gene=True, n_dims=None):
        """
        Extract transcript coordinates from SpatialData and compute grid indices.

        Returns
        -------
        meta : dict
            Keys: gene_names, coordinate_offset, tx, ty, tz,
                  xc_g, yc_g, zc_g, is_3d, dim
        """
        pts = sdata.points[list(sdata.points.keys())[0]]
        if hasattr(pts, 'compute'):
            pts = pts.compute()
        if filter_is_gene and 'is_gene' in pts.columns:
            pts = pts[pts['is_gene'] == True]

        gene_col = next(
            (c for c in ['gene', 'gene_symbol', 'feature_name', 'target'] if c in pts.columns),
            None,
        )
        if gene_col is None:
            raise ValueError(f'gene column not found. columns: {pts.columns.tolist()}')

        gene_names = pts[gene_col].astype(str).values
        tx_raw = pts['x'].values.astype(np.float64)
        ty_raw = pts['y'].values.astype(np.float64)
        has_z = 'z' in pts.columns
        tz_raw = pts['z'].values.astype(np.float64) if has_z else np.zeros(len(tx_raw), dtype=np.float64)

        x_min, y_min, z_min = tx_raw.min(), ty_raw.min(), tz_raw.min()
        tx = tx_raw - x_min
        ty = ty_raw - y_min
        tz = tz_raw - z_min

        z_threshold = 1.0
        is_3d = has_z and (tz_raw.max() - tz_raw.min()) > z_threshold and len(set(tz_raw[:10000])) > 1
        # the recorded dimensionality wins over the inference (see _prepare_grid)
        nd = n_dims if n_dims is not None else getattr(self, 'n_dims', None)
        if nd == 2:
            is_3d = False
        elif nd == 3 and not is_3d:
            raise ValueError('n_dims=3 but the transcripts carry no usable z — '
                             'downsample with z=2')
        dim = 3 if is_3d else 2

        xc_g = np.floor(tx / sampling_distance).astype(np.int64)
        yc_g = np.floor(ty / sampling_distance).astype(np.int64)
        zc_g = np.floor(tz / sampling_distance).astype(np.int64) if is_3d else np.zeros(len(tx), dtype=np.int64)

        return {
            'gene_names': gene_names,
            'coordinate_offset': [float(x_min), float(y_min), float(z_min)],
            'tx': tx, 'ty': ty, 'tz': tz,
            'xc_g': xc_g, 'yc_g': yc_g, 'zc_g': zc_g,
            'is_3d': is_3d, 'dim': dim,
        }


    def _compute_normalized_vectors(self, query_voxels, meta, genes=None,
                                     bandwidth=2.5, sampling_distance=1.0,
                                     prune_coefficient=4.3, size_after_normalization=10,
                                     log_transform=True, kernel='gaussian',
                                     gpu_device='cuda'):
        """
        Compute L1-normalised (+ optional log1p) KDE vectors at arbitrary voxel positions.

        Uses ``ssaft.utils.kde_at_points`` (grid-cell hash map + OpenMP), which
        produces results identical to ``accumulate_gene_vector`` at the same
        coordinates; inside :func:`ssaft._kde_gpu.gpu_kde` (``downsample_vector``'s
        ``auto_gpu``) that name is the Triton cell-list kernel instead. Both take
        every kernel in ``ssaft._count.KDE_KERNELS``.

        These vectors ARE the dictionary the mapper later scores against (they are
        clustered, annotated and summed into ``beta``), so ``kernel`` must be the
        one ``map_celltypes`` uses: building them with another would match, say, a
        gaussian dictionary against an Epanechnikov field. On this panel that moves
        2 of every 10 top genes per anchor, so it is a real mismatch, not a
        rounding difference.

        Parameters
        ----------
        query_voxels : ndarray, shape (N, 2|3)  — grid-index coordinates
        meta : dict returned by ``_parse_transcripts``

        Returns
        -------
        normalized : ndarray (N, n_genes)
        vf_norm    : ndarray (N,)
        genes      : list[str]
        coordinate_offset : list[float]
        """
        from .utils import kde_at_points

        gene_names = meta['gene_names']
        tx, ty, tz = meta['tx'], meta['ty'], meta['tz']
        is_3d = meta['is_3d']
        dim = meta['dim']
        coordinate_offset = meta['coordinate_offset']

        if genes is None:
            genes = sorted(np.unique(gene_names))
        gene_to_idx = {g: i for i, g in enumerate(genes)}
        n_genes = len(genes)

        gidx_arr = np.array([gene_to_idx.get(g, -1) for g in gene_names], dtype=np.int32)
        valid = gidx_arr >= 0
        tx, ty, tz = tx[valid], ty[valid], tz[valid]
        gidx_arr = gidx_arr[valid]

        query_voxels = np.asarray(query_voxels, dtype=np.float64)
        qx = (query_voxels[:, 0] * sampling_distance).astype(np.float64)
        qy = (query_voxels[:, 1] * sampling_distance).astype(np.float64)
        qz = ((query_voxels[:, 2] * sampling_distance).astype(np.float64)
              if (is_3d and query_voxels.shape[1] > 2)
              else np.zeros(len(qx), dtype=np.float64))

        from ._count import kernel_code
        kde_values = kde_at_points(
            qx, qy, qz,
            tx.astype(np.float64), ty.astype(np.float64), tz.astype(np.float64),
            gidx_arr, n_genes,
            float(sampling_distance), float(bandwidth), float(prune_coefficient),
            int(dim), int(self.ncores), kernel=kernel_code(kernel),
        )

        vf_norm = kde_values.sum(axis=1)
        normalized = np.zeros_like(kde_values)
        nonzero = vf_norm > 0
        normalized[nonzero] = (kde_values[nonzero] / vf_norm[nonzero, np.newaxis]
                               * size_after_normalization)
        if log_transform:
            np.log1p(normalized, out=normalized)

        return normalized, vf_norm, genes, coordinate_offset


    def _he_array_from_sdata(self, sdata, image_key='he'):
        """Extract an (H, W, 3) uint8 RGB ndarray from ``sdata.images[image_key]``.

        Handles both single-scale (``DataArray``) and multiscale
        (``DataTree``) representations as written by spatialdata-io / NGFF.
        For multiscale, the highest-resolution level is returned.

        RGB(A) images stored channel-first ``(c, y, x)`` are transposed to
        ``(y, x, c)``. Single-channel images — e.g. a fluorescence morphology
        MIP stored as ``(1, y, x)`` or ``(y, x, 1)`` — are squeezed to 2-D and
        replicated across 3 channels. Higher-bit-depth integer images (e.g.
        uint16 fluorescence) are contrast-stretched per image to 0-255 (99.9th
        percentile → 255) rather than clipped, so tissue spans the dynamic
        range instead of saturating; uint8 RGB (H&E) is returned unchanged.

        The caller is responsible for ensuring that the H&E element in
        ``sdata`` already spatially corresponds to the morphology image
        (i.e., cropped to the sequenced region and rasterized onto an
        axis-aligned grid). This function does not apply any
        coordinate-system transform.
        """
        if image_key not in sdata.images:
            raise KeyError(
                f"sdata.images does not contain key '{image_key}'. "
                f"Available: {list(sdata.images.keys())}"
            )
        node = sdata.images[image_key]

        if not hasattr(node, 'values') or hasattr(node, 'children'):
            scale_keys = list(node.keys())
            if not scale_keys:
                raise RuntimeError(
                    f"H&E multiscale image at images['{image_key}'] has no scales."
                )
            leaf = node[scale_keys[0]]
            if hasattr(leaf, 'image'):
                arr = leaf['image'].values
            else:
                data_vars = list(leaf.data_vars)
                if not data_vars:
                    raise RuntimeError(
                        f"H&E multiscale leaf '{scale_keys[0]}' has no data_vars."
                    )
                arr = leaf[data_vars[0]].values
        else:
            arr = node.values

        if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
            arr = np.moveaxis(arr, 0, -1)
        elif arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.ndim == 3 and arr.shape[0] < arr.shape[-1] and arr.shape[0] not in (3, 4):
            arr = arr.max(axis=0)
        elif arr.ndim == 3 and arr.shape[-1] not in (3, 4) and arr.shape[-1] < arr.shape[0]:
            arr = arr.max(axis=-1)

        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0:
                arr = (arr.astype(np.float32) * 255.0)
            else:
                sub = arr[::8, ::8]
                hi = float(np.percentile(sub, 99.9))
                if hi <= 0.0:
                    hi = float(arr.max())
                scale = (255.0 / hi) if hi > 255.0 else 1.0
                arr = arr.astype(np.float32) * scale
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim != 3 or arr.shape[-1] < 3:
            raise RuntimeError(
                f"image at images['{image_key}'] has unsupported shape {arr.shape}."
            )
        arr = arr[:, :, :3]
        return arr

    def _he_array_axis_aligned(self, sdata, image_key='he',
                               element_name=None, sampling_distance=1.0,
                               target_um_per_px=None, pad=True,
                               revert_image=True, arr=None, gray_u8=False):
        """Return an H&E ndarray axis-aligned in the global coordinate
        system **and cropped/padded so its pixel grid spans exactly the
        transcripts global bounding box**.

        ``_select_vectors`` zooms this array to the transcripts-derived
        VF grid and converts the resulting voxel indices to global µm
        via ``coordinate_offset + idx * sampling_distance``. That
        conversion is only correct if array index (row=0, col=0) maps
        to global (ty_min, tx_min) and (row=H, col=W) maps to
        (ty_max, tx_max). For 90°-multiple rotations the array is
        de-rotated losslessly with ``np.rot90`` and then cropped/padded
        to the transcripts bbox; for arbitrary rotations it falls back
        to ``spatialdata.rasterize`` at native pixel resolution onto
        the transcripts bbox (with the unavoidable interpolation blur).

        Parameters
        ----------
        target_um_per_px : float, optional
            Only used by the rasterize fallback (non-90° rotation case).
            If ``None``, the native pixel size is auto-detected.
        revert_image : bool, default True
            Background polarity of the staining image. ``True`` for
            bright-background / dark-tissue images (e.g. H&E); the bbox
            padding is filled with white (255) so it reads as background.
            ``False`` for dark-background / bright-tissue images (e.g.
            polyT); padding is filled with black (0) so it reads as
            background. Must match the value passed to the otsu / mask
            steps so padding is never mistaken for tissue.
        arr : ndarray, optional
            Pixels to use instead of ``sdata.images[image_key]`` — one plane of
            a z-stack. The plane is placed with ``images[image_key]``'s own
            transform, so it has to share that image's pixel grid; the shape
            is checked.
        """
        from spatialdata.transformations import get_transformation as _get_tf
        from spatialdata import get_extent as _get_extent

        if image_key not in sdata.images:
            raise KeyError(
                f"sdata.images does not contain key '{image_key}'. "
                f"Available: {list(sdata.images.keys())}"
            )

        if arr is None:
            arr = self._he_array_from_sdata(sdata, image_key=image_key)
        else:
            want = self._sdata_image_hw(sdata, image_key)
            if not gray_u8:
                arr = self._image_to_rgb_uint8(arr)
            elif arr.ndim != 3 or arr.shape[-1] != 1 or arr.dtype != np.uint8:
                raise ValueError(f"gray_u8=True expects an (H, W, 1) uint8 plane, got {arr.shape} {arr.dtype}")
            if tuple(arr.shape[:2]) != tuple(want):
                raise ValueError(
                    f"image plane is {tuple(arr.shape[:2])} px but images[{image_key!r}] is "
                    f"{tuple(want)}: a plane is placed with that image's transform, so it has "
                    f"to share its pixel grid (e.g. Xenium morphology.ome.tif with "
                    f"morphology_focus)")
        H, W = arr.shape[:2]

        try:
            tf = _get_tf(sdata.images[image_key], to_coordinate_system='global')
            M = np.asarray(tf.to_affine_matrix(
                input_axes=('x', 'y'), output_axes=('x', 'y')
            ))
        except Exception:
            return arr

        A = M[:2, :2]
        t = M[:2, 2]

        points_key = element_name
        if points_key is None:
            if 'transcripts' in sdata.points:
                points_key = 'transcripts'
            elif sdata.points:
                points_key = next(iter(sdata.points.keys()))

        tx_min = tx_max = ty_min = ty_max = None
        if points_key is not None:
            try:
                # the extent is a full pass over the points; a 3D run asks for it twice per
                # plane of every tile, so it is computed once per points element
                # keyed on the SpatialData object itself (weakly: a tile's entry goes with it);
                # sdata.points[key] hands back a new object on every access
                import weakref
                cache = self.__dict__.setdefault('_points_extent_cache', weakref.WeakKeyDictionary())
                per = cache.setdefault(sdata, {})
                if points_key not in per:
                    per[points_key] = self._points_global_extent(sdata, points_key)
                ext = per[points_key]
                tx_min, tx_max = ext['x']
                ty_min, ty_max = ext['y']
            except Exception:
                pass

        angle_deg = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))

        nearest_k = int(round(angle_deg / 90.0)) % 4
        residual_deg = abs(angle_deg - nearest_k * 90.0)
        if residual_deg > 45:
            residual_deg = 90 - residual_deg

        if residual_deg < 5.0:
            k_rot = (-nearest_k) % 4
            arr_rot = np.ascontiguousarray(np.rot90(arr, k=k_rot))

            if tx_min is None:
                return arr_rot

            corners_px = np.array(
                [[0.0, 0.0], [W, 0.0], [0.0, H], [W, H]], dtype=np.float64
            )
            corners_global = (A @ corners_px.T).T + t
            ix0 = float(corners_global[:, 0].min())
            iy0 = float(corners_global[:, 1].min())

            s = float(np.sqrt(abs(np.linalg.det(A))))
            if not np.isfinite(s) or s <= 0:
                return arr_rot

            Hr, Wr = arr_rot.shape[:2]

            col_start = int(np.floor((tx_min - ix0) / s))
            col_end   = int(np.ceil ((tx_max - ix0) / s))
            row_start = int(np.floor((ty_min - iy0) / s))
            row_end   = int(np.ceil ((ty_max - iy0) / s))

            if not pad:
                vr0, vr1 = max(0, row_start), min(arr_rot.shape[0], row_end)
                vc0, vc1 = max(0, col_start), min(arr_rot.shape[1], col_end)
                return np.ascontiguousarray(arr_rot[vr0:vr1, vc0:vc1, :])

            pad_left   = max(0, -col_start)
            pad_top    = max(0, -row_start)
            pad_right  = max(0, col_end - Wr)
            pad_bottom = max(0, row_end - Hr)
            if pad_left or pad_top or pad_right or pad_bottom:
                pad_value = 255 if revert_image else 0
                arr_rot = np.pad(
                    arr_rot,
                    ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                    mode='constant', constant_values=pad_value,
                )
                col_start += pad_left
                col_end   += pad_left
                row_start += pad_top
                row_end   += pad_top

            cropped = arr_rot[row_start:row_end, col_start:col_end, :]
            return np.ascontiguousarray(cropped)

        from spatialdata import rasterize as _rasterize

        if tx_min is None:
            raise RuntimeError(
                "sdata has no points element; cannot derive H&E bbox for "
                "non-90° rotated affine."
            )

        if target_um_per_px is None:
            target_um_per_px = float(np.sqrt(abs(np.linalg.det(A))))
            if not np.isfinite(target_um_per_px) or target_um_per_px <= 0:
                target_um_per_px = float(sampling_distance)
        target_unit_to_pixels = 1.0 / float(target_um_per_px)

        raster = _rasterize(
            sdata.images[image_key],
            axes=('x', 'y'),
            min_coordinate=[tx_min, ty_min],
            max_coordinate=[tx_max, ty_max],
            target_coordinate_system='global',
            target_unit_to_pixels=target_unit_to_pixels,
        )
        arr = raster.values if hasattr(raster, 'values') else np.asarray(raster)
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (3, 4):
            arr = np.moveaxis(arr, 0, -1)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    @staticmethod
    def _image_to_rgb_uint8(arr):
        """One image plane -> ``(H, W, 3)`` uint8, the way :meth:`_he_array_from_sdata` does it.

        A single-channel plane is replicated across 3 channels; a higher-bit-depth plane
        is stretched so its own 99.9th percentile maps to 255. Each plane is stretched on
        its own, so an out-of-focus plane is not judged against the brightest one.
        """
        arr = np.asarray(arr)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
            arr = np.moveaxis(arr, 0, -1)                # (c, y, x) as a SpatialData image stores it
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0:
                arr = arr.astype(np.float32) * 255.0
            else:
                hi = float(np.percentile(arr[::8, ::8], 99.9))
                if hi <= 0.0:
                    hi = float(arr.max())
                arr = arr.astype(np.float32) * ((255.0 / hi) if hi > 255.0 else 1.0)
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim != 3 or arr.shape[-1] < 3:
            raise RuntimeError(f'unsupported image plane shape {arr.shape}')
        return arr[:, :, :3]

    @staticmethod
    def _points_global_extent(sdata, points_key):
        """``{'x': (min, max), 'y': (min, max)}`` of a points element in ``'global'``.

        spatialdata's ``get_extent`` transforms every point, which fails on a whole-section
        element whose dask partitions repeat index labels ("cannot reindex on an axis with
        duplicate labels") and is a full pass anyway. When the transformation is axis-aligned
        (scale / translation / 90-degree rotation) the extent is the transformed bounding box
        of the intrinsic min/max -- the same numbers from two column reductions.
        """
        from spatialdata import get_extent as _get_extent
        from spatialdata.transformations import get_transformation as _get_tf
        el = sdata.points[points_key]
        M = np.asarray(_get_tf(el, to_coordinate_system='global').to_affine_matrix(
            input_axes=('x', 'y'), output_axes=('x', 'y')), dtype=np.float64)
        A = M[:2, :2]
        if not (np.allclose(A[0, 1], 0) and np.allclose(A[1, 0], 0)) and \
                not (np.allclose(A[0, 0], 0) and np.allclose(A[1, 1], 0)):
            return _get_extent(el, coordinate_system='global')
        import dask
        lo_x, hi_x, lo_y, hi_y = dask.compute(el['x'].min(), el['x'].max(), el['y'].min(), el['y'].max())
        c = np.array([[lo_x, lo_y], [hi_x, lo_y], [lo_x, hi_y], [hi_x, hi_y]], dtype=np.float64)
        g = c @ A.T + M[:2, 2]
        return {'x': (float(g[:, 0].min()), float(g[:, 0].max())),
                'y': (float(g[:, 1].min()), float(g[:, 1].max()))}

    @staticmethod
    def _image_to_u8_plane(arr):
        """A single-channel plane -> ``(H, W)`` uint8 exactly as :meth:`_image_to_rgb_uint8` makes
        each of its three identical channels; ``None`` when the plane has several channels."""
        arr = np.asarray(arr)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if arr.ndim != 2:
            return None
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0:
                arr = arr.astype(np.float32) * 255.0
            else:
                hi = float(np.percentile(arr[::8, ::8], 99.9))
                if hi <= 0.0:
                    hi = float(arr.max())
                arr = arr.astype(np.float32) * ((255.0 / hi) if hi > 255.0 else 1.0)
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    def _plane_luts(self, revert_image, intensity):
        """Per-value tables for a uint8 plane replicated to RGB: the float intensity the full path
        computes (inverted as the mask test inverts it) and the uint8 image Otsu sees."""
        v = np.arange(256, dtype=np.uint8)
        rgb = np.stack([v, v, v], axis=-1).reshape(1, 256, 3)
        gray = self._rgb_to_intensity(rgb, intensity=intensity)[0]          # same float64 ops
        inv = (255.0 - gray) if revert_image else gray
        return inv, inv.astype(np.uint8)

    @staticmethod
    def _sdata_image_hw(sdata, image_key):
        """``(H, W)`` of ``sdata.images[image_key]`` at full resolution, without loading it."""
        if image_key not in sdata.images:
            raise KeyError(f"sdata.images does not contain key '{image_key}'. "
                           f"Available: {list(sdata.images.keys())}")
        return Analysis._element_hw(sdata.images[image_key])

    @staticmethod
    def _element_hw(node):
        """``(H, W)`` of one image element (2D or 3D, single- or multiscale) at full resolution."""
        # isinstance, not hasattr(node, 'values'): hasattr evaluates the property, and a
        # DataArray's .values loads the whole image just to read its shape
        from xarray import DataArray as _DataArray
        if not isinstance(node, _DataArray):
            leaf = node[list(node.keys())[0]]
            var = leaf['image'] if hasattr(leaf, 'image') else leaf[list(leaf.data_vars)[0]]
        else:
            var = node
        dims = list(var.dims)
        return int(var.shape[dims.index('y')]), int(var.shape[dims.index('x')])

    @staticmethod
    def _points_px_per_um(sdata):
        """x/y scale of the points element's ``'global'`` transform, read as
        :meth:`_select_vectors` reads it (the mask is counted in those pixels)."""
        from spatialdata.transformations import get_transformation as _get_tf
        for _etype, _ename, _elem in sdata.gen_elements():
            if _etype != 'points':
                continue
            _tf = _get_tf(_elem, get_all=True).get('global', None)
            if _tf is not None and hasattr(_tf, 'scale'):
                return float(_tf.scale[0]), float(_tf.scale[1])
            try:
                _aff = _tf.to_affine_matrix(input_axes=('x', 'y'), output_axes=('x', 'y'))
                return (abs(float(_aff[0, 0])) if _aff.shape[0] > 0 else 1.0,
                        abs(float(_aff[1, 1])) if _aff.shape[0] > 1 else 1.0)
            except Exception:
                return 1.0, 1.0
        raise RuntimeError('No points element found in sdata.')

    @staticmethod
    def _image_planes(images, z_step=None, z_origin=0.0):
        """Resolve z-stack files into imaging planes, for :meth:`zstack_image`.

        Accepts

        * **one z-stack file** — a ``ZYX`` TIFF / OME-TIFF such as Xenium
          ``morphology.ome.tif``. Planes are read one at a time; ``z_step`` defaults to
          the OME ``PhysicalSizeZ``;
        * **a folder of per-plane files** — e.g. a MERSCOPE ``images/`` folder with
          ``mosaic_DAPI_z0.tif ... mosaic_PolyT_z6.tif``. The plane number comes from
          ``_z<N>`` in the name. When the folder holds several stains, pick one with a glob;
        * **a glob pattern** (``'.../mosaic_PolyT_z*.tif'``) or **a list of files**.

        Per-plane files record no spacing, so ``z_step`` is required for them. ``z`` of
        plane ``N`` is ``z_origin + N * z_step``, in the units of the transcripts' z
        column (µm for Xenium; MERSCOPE ``global_z`` is a plane index, so its z_step is 1
        unless the loader converted it).

        :returns: ``(planes, z_step)`` with ``planes`` a list of
            ``{'index', 'z', 'read', 'source'}`` in plane order; ``read()`` returns the
            plane's pixels.
        """
        import glob as _glob
        import re
        import tifffile
        from pathlib import Path as _P

        pat = re.compile(r'_z(\d+)(?:\.ome)?\.tiff?$', re.I)

        def _from_files(files, how):
            if not files:
                raise FileNotFoundError(f'images={images!r}: no image files found')
            hits = [pat.search(_P(f).name) for f in files]
            if all(h is not None for h in hits):
                order = sorted(zip((int(h.group(1)) for h in hits), files))
            else:
                order = list(enumerate(sorted(files)))
            if len({i for i, _ in order}) != len(order):
                raise ValueError(f'images={images!r}: two files claim the same plane number '
                                 f'({[_P(f).name for _, f in order]})')
            if z_step is None:
                # Per-plane files record no spacing. Default to 1, i.e. the plane's own number
                # is its z -- which is exactly the unit MERSCOPE's `global_z` is in, so the
                # planes and the molecules line up with nothing to look up. The price is
                # geometry: x/y stay µm while z counts planes, so the KDE kernel and every
                # volume are stretched in z by the real spacing (1.5 µm on MERSCOPE, from
                # dataorganization.csv `zPos`). Pass z_step=<µm> when that matters.
                warnings.warn(
                    f'images={images!r} is {how} and carries no z spacing: z_step defaults to '
                    '1, so a plane sits at its own number (the unit MERSCOPE global_z uses). '
                    'x/y are µm, so z distances and volumes are off by the true spacing; pass '
                    'z_step=<µm> for true geometry.', UserWarning, stacklevel=2)
            step = 1.0 if z_step is None else float(z_step)

            def _win(path, r0, r1, c0, c1):
                store = tifffile.imread(path, aszarr=True)
                try:
                    a = zarr.open(store, mode='r')
                    if not hasattr(a, 'shape'):          # a pyramid: its full-resolution level
                        a = a['0']
                    return np.asarray(a[r0:r1, c0:c1])
                finally:
                    store.close()

            return [dict(index=int(i), z=float(z_origin) + int(i) * step,
                         read=(lambda f=f: tifffile.imread(f)),
                         read_window=(lambda r0, r1, c0, c1, f=f: _win(f, r0, r1, c0, c1)),
                         source=str(f))
                    for i, f in order], step

        if isinstance(images, (list, tuple)):
            return _from_files([str(f) for f in images], 'a list of files')
        s = str(images)
        if any(c in s for c in '*?['):
            return _from_files(sorted(_glob.glob(s)), 'a glob pattern')
        p = _P(s)
        if p.is_dir():
            groups = {}
            for f in sorted(p.iterdir()):
                if pat.search(f.name):
                    groups.setdefault(pat.sub('', f.name), []).append(str(f))
            if not groups:
                raise FileNotFoundError(f'images={s!r}: no per-plane *_z<N>.tif files in this folder')
            if len(groups) > 1:
                ex = sorted(groups)[0]
                raise ValueError(
                    f'images={s!r} holds {len(groups)} plane series ({sorted(groups)}); pick one '
                    f"with a glob, e.g. images='{p / (ex + '_z*.tif')}'")
            return _from_files(next(iter(groups.values())), 'a folder')
        if not p.is_file():
            raise FileNotFoundError(f'images={s!r} does not exist')

        with tifffile.TiffFile(s) as tf:
            ser = tf.series[0]
            axes = ser.axes.upper()
            shape = tuple(ser.shape)
            md = tf.ome_metadata or ''
        if 'Z' not in axes:
            raise ValueError(f'{s} has no Z axis (axes {axes}); pass a z-stack, or a folder / '
                             f'glob of per-plane images')
        extra = [a for a in axes if a not in 'ZYX']
        if extra:
            raise ValueError(f'{s} has axes {axes}: only a single-stain ZYX stack is supported; '
                             f'save the stain to threshold as its own file')
        step = z_step
        if step is None:
            m = re.search(r'PhysicalSizeZ="([0-9.eE+-]+)"', md)
            if not m:
                raise ValueError(f'{s} records no PhysicalSizeZ; pass z_step=')
            step = float(m.group(1))
        zax = axes.index('Z')

        yax, xax = axes.index('Y'), axes.index('X')

        def _read(k, window=None, path=s):
            store = tifffile.imread(path, aszarr=True, level=0)
            try:
                a = zarr.open(store, mode='r')
                sl = [slice(None)] * a.ndim
                sl[zax] = k
                if window is not None:              # read only this window of the plane
                    r0, r1, c0, c1 = window
                    sl[yax], sl[xax] = slice(r0, r1), slice(c0, c1)
                return np.asarray(a[tuple(sl)])
            finally:
                store.close()

        return [dict(index=k, z=float(z_origin) + k * float(step),
                     read=(lambda k=k: _read(k)),
                     read_window=(lambda r0, r1, c0, c1, k=k: _read(k, (r0, r1, c0, c1))),
                     source=f'{s}[Z={k}]')
                for k in range(shape[zax])], float(step)

    @staticmethod
    def _select_planes(planes, step, z, z_origin):
        """Keep the planes ``z`` asks for and give back the (step, origin) that place them.

        ``planes`` carry their ORIGINAL number in ``index`` (the page for a z-stack file, the
        ``_z<N>`` in the name for per-plane files), and that number is what ``z`` selects on,
        so a subset keeps the z it had in the full stack.

        The element's z axis is one scale + one translation, so the kept planes have to be
        evenly spaced: plane ``j`` of the subset ends up at ``origin + j * step'``. A stride is
        fine (every second plane doubles the step); an irregular pick is refused rather than
        silently placed on a lattice it does not sit on.

        :returns: ``(planes, step, z_origin)`` for the subset.
        """
        if z is None:
            return planes, step, z_origin
        have = [int(pl['index']) for pl in planes]
        if isinstance(z, slice):
            want = have[z]
        else:
            want = [int(v) for v in (z if hasattr(z, '__iter__') else [z])]
            missing = [v for v in want if v not in have]
            if missing:
                raise ValueError(f'z={z!r}: no plane numbered {missing} (have {have})')
            want = sorted(set(want))
        if not want:
            raise ValueError(f'z={z!r} selects no plane (have {have})')
        if len(want) > 1:
            strides = {b - a for a, b in zip(want, want[1:])}
            if len(strides) != 1:
                raise ValueError(
                    f'z={z!r} picks planes {want}, which are not evenly spaced: the element '
                    'places plane j at origin + j * step, so an irregular subset cannot be '
                    'written as one z transformation. Pick a constant stride, or build one '
                    'element per run of planes.')
            stride = strides.pop()
        else:
            stride = 1
        keep = [pl for pl in planes if int(pl['index']) in set(want)]
        return keep, float(step) * stride, float(z_origin) + want[0] * float(step)

    @staticmethod
    def zstack_planes(source, sdata=None, *, z=None, z_step=None, z_origin=0.0,
                      element_name=None, transcripts_z=None, z_unit=None, slab=None,
                      verbose=True):
        """What each imaging plane is worth: where it sits, and how many molecules are there.

        Use it before :meth:`zstack_image` to choose ``z``. A stack usually reaches past the
        section on both sides, and a plane outside the tissue still gets a tissue mask (its
        Otsu is computed on that plane alone), so it costs a full plane of sampling and returns
        anchors whose vectors are ambient. Counting the molecules at each plane's own depth
        says which planes are worth stacking.

        The molecules' ``z`` comes in one of two units, and the count follows it:

        * **plane index** (MERSCOPE ``global_z`` is 0, 1, 2 ... one value per imaging plane) —
          each plane is matched by its OWN NUMBER, so the count is exact: no slab, and the
          physical spacing is not involved in it at all. ``z_step`` still has to be passed,
          because it is what puts the planes in µm for everything downstream.
        * **µm** (Xenium ``z_location`` is continuous) — a plane takes the molecules within
          ``slab`` of its µm position, ``z_step`` by default, so the slabs tile the axis.

        Which one it is, is detected from the values (all integers within the plane numbers ->
        plane index) and reported; pass ``z_unit='plane'`` / ``'um'`` to force it.

        :param source: what :meth:`zstack_image` would take -- a z-stack file, a folder, a glob
            or a list of per-plane files.
        :param sdata: SpatialData holding the molecules; the first points element is read
            unless ``element_name`` says otherwise.
        :param z: optional pre-selection, as in :meth:`zstack_image`; the census then describes
            only those planes, and counts the rest as outside.
        :param z_step: µm between consecutive planes. Read from the OME ``PhysicalSizeZ`` of a
            stack file (Xenium); for per-plane files (MERSCOPE) it defaults to 1, which puts
            each plane at its own number -- the unit ``global_z`` already uses, so the two line
            up without looking the spacing up. Pass the real µm (1.5 on MERSCOPE) when z
            distances or volumes have to be physical.
        :param transcripts_z: the z values directly, instead of ``sdata``.
        :param z_unit: ``'plane'`` / ``'um'`` to override the detection.
        :param slab: µm counted around each plane; ``z_step`` by default. Ignored for plane
            indices, where the match is exact.
        :returns: a :class:`pandas.DataFrame` indexed by the plane's own number, with its ``z``,
            the molecules it holds, that as a fraction of all of them, and the running total.
            ``df.attrs`` carries ``unit``, ``outside`` (molecules no kept plane holds) and
            ``slab``.
        """
        import pandas as pd
        planes, step = Analysis._image_planes(source, z_step=z_step, z_origin=0.0)
        planes, step, z_origin = Analysis._select_planes(planes, step, z, z_origin)
        if transcripts_z is None:
            if sdata is None:
                raise ValueError('pass sdata= (or transcripts_z=) to count the molecules')
            name = element_name or next(iter(getattr(sdata, 'points', {}) or {}), None)
            if name is None:
                raise ValueError('no points element in sdata')
            pts = sdata.points[name]
            if 'z' not in getattr(pts, 'columns', []):
                raise ValueError(f'points[{name!r}] has no z column, so there is nothing to '
                                 'count per plane')
            col = pts['z']
            transcripts_z = np.asarray(col.compute() if hasattr(col, 'compute') else col,
                                       dtype=np.float64)
        tz = np.asarray(transcripts_z, dtype=np.float64)
        idx = [int(pl['index']) for pl in planes]
        if z_unit is None:
            # A plane index is a small non-negative integer that NAMES a plane; a µm depth is
            # a continuous measurement. Do not compare against the planes on hand: a bundle
            # may ship images for only some of the planes its molecules were decoded on
            # (MsLiver has z0 and z3 of 7), and the molecules on the missing ones are then
            # counted as outside, not taken as proof that z is in µm.
            uniq = np.unique(tz) if tz.size else tz
            z_unit = ('plane' if (uniq.size and uniq.size <= 100
                                  and np.all(uniq == np.rint(uniq)) and uniq.min() >= 0)
                      else 'um')
        elif z_unit not in ('plane', 'um'):
            raise ValueError(f"z_unit={z_unit!r}: 'plane' or 'um'")
        half = float(slab if slab is not None else step) / 2.0

        rows, inside = [], np.zeros(len(tz), dtype=bool)
        for j, k in enumerate(idx):
            zc = float(z_origin) + j * float(step)
            m = (tz == k) if z_unit == 'plane' else ((tz >= zc - half) & (tz < zc + half))
            inside |= m
            rows.append(dict(plane=k, z=zc, molecules=int(m.sum())))
        df = pd.DataFrame(rows).set_index('plane')
        n = max(len(tz), 1)
        df['fraction'] = df['molecules'] / n
        df['cumulative'] = df['fraction'].cumsum()
        df.attrs.update(unit=z_unit, outside=int((~inside).sum()),
                        slab=(None if z_unit == 'plane' else 2 * half),
                        z_step=float(step), z_origin=float(z_origin))
        if verbose:
            how = ('molecule z is the plane index: counted by plane number'
                   if z_unit == 'plane' else
                   f'molecule z is µm: counted within a {2 * half:g} µm slab')
            print(f'[zstack_planes] {len(planes)} planes, step {step:g} µm, origin '
                  f'{z_origin:g} µm; {len(tz):,} molecules, {how}; '
                  f'{int((~inside).sum()):,} on no kept plane\n' +
                  df.to_string(formatters={'z': '{:g}'.format, 'molecules': '{:,}'.format,
                                           'fraction': '{:.2%}'.format,
                                           'cumulative': '{:.1%}'.format}))
        return df

    @staticmethod
    def zstack_image(source, *, like=None, z=None, z_step=None, z_origin=0.0,
                     scale_factors=None, chunk=2048):
        """A z-stack as a SpatialData 3D image element, ready for ``sdata.images[key] = ...``.

        :meth:`downsample_vector` reads every image from ``sdata.images``. For ``z=3`` the
        element named by ``images=`` must carry a ``z`` axis; this builds one, lazily, so it
        can be added to the SpatialData the rest of the analysis already uses::

            sdata.images['morphology_zstack'] = Analysis.zstack_image(
                '.../morphology.ome.tif', like=sdata.images['morphology_focus'])
            analysis.downsample_vector(sdata, z=3, images='morphology_zstack', ...)

        :param source: a ``ZYX`` stack file (Xenium ``morphology.ome.tif``), a folder of
            per-plane ``*_z<N>.tif`` files (MERSCOPE ``images/``), a glob pattern or a list
            of files — see :meth:`_image_planes`.
        :param like: the 2D image element whose pixel grid the planes share (e.g.
            ``sdata.images['morphology_focus']``). Its transformations give the element's
            x/y placement in every coordinate system, and its size is checked. Without it
            the x/y transformation is the identity.
        :param z: which planes to stack, by their own number -- the page in a z-stack file,
            the ``_z<N>`` in a per-plane file name. ``None`` (default) takes all of them; pass
            an iterable (``z=[5, 6, 7, 8]``) or a slice. The kept planes keep the z they had in
            the full stack, so ``z_origin`` still refers to plane 0 of the SOURCE and does not
            have to be moved by hand. They must be evenly spaced, since the element's z axis is
            a single scale + translation. :meth:`zstack_planes` says which planes are worth it.
        :param z_step: distance between consecutive planes in the transcripts' z units.
            Read from the OME ``PhysicalSizeZ`` of a stack file; for per-plane files it
            defaults to 1, i.e. the plane's own number (what MERSCOPE ``global_z`` counts).
        :param z_origin: z of plane 0 OF THE SOURCE, same units (not of the first kept plane).
        :param scale_factors: x/y pyramid factors, default ``[2, 2, 2]``; z is never
            downscaled, since each plane is an image of its own.
        :returns: an ``Image3DModel`` element ``(c, z, y, x)``. Plane ``k`` sits at
            ``z_origin + k * z_step`` in each coordinate system: SpatialData places pixel
            ``k``'s centre at ``k + 0.5``, which the z part of the transformation undoes.
        """
        import dask.array as _da
        import tifffile
        from spatialdata.models import Image3DModel
        from spatialdata.transformations import (Scale, Sequence, Translation,
                                                 get_transformation as _get_tf)

        planes, step = Analysis._image_planes(source, z_step=z_step, z_origin=0.0)
        planes, step, z_origin = Analysis._select_planes(planes, step, z, z_origin)

        def _lazy(src):
            path, _, k = src.partition('[Z=')
            store = tifffile.imread(path, aszarr=True, level=0)
            a = zarr.open(store, mode='r')
            if not hasattr(a, 'shape'):
                a = a['0']
            arr = _da.from_zarr(a)
            return arr[int(k.rstrip(']'))] if k else arr

        arr = _da.stack([_lazy(pl['source']) for pl in planes], axis=0)       # (z, y, x)
        if arr.ndim != 3:
            raise ValueError(f'{source}: planes are {arr.shape[1:]}, expected single-channel (y, x)')
        if like is not None:
            want = Analysis._element_hw(like)
            if tuple(arr.shape[1:]) != tuple(want):
                raise ValueError(f'{source}: planes are {tuple(arr.shape[1:])} px but the like= image is '
                                 f'{tuple(want)}; the planes must share its pixel grid')
        z_tf = Sequence([Translation([-0.5], axes=('z',)), Scale([float(step)], axes=('z',)),
                         Translation([float(z_origin)], axes=('z',))])
        xy = (_get_tf(like, get_all=True) if like is not None else None) or {'global': None}
        tfs = {cs: (Sequence([z_tf, t]) if t is not None else z_tf) for cs, t in xy.items()}
        sf = scale_factors if scale_factors is not None else [2, 2, 2]
        sf = [f if isinstance(f, dict) else {'x': int(f), 'y': int(f), 'z': 1} for f in sf]
        el = Image3DModel.parse(arr[None], dims=('c', 'z', 'y', 'x'), scale_factors=sf,
                                chunks=(1, 1, chunk, chunk), transformations=tfs)
        # where each plane lives, so a whole plane can be decoded straight from the file (in
        # parallel) instead of through the rechunked multiscale graph -- ~10x faster on JPEG2000
        el.attrs['ssaft_plane_sources'] = [pl['source'] for pl in planes]
        return el

    @staticmethod
    def _sdata_image_planes(sdata, images):
        """The planes of ``sdata.images[images]``, a 3D image element, for :meth:`_select_vectors_3d`.

        Each plane's ``z`` is its pixel centre pushed through the element's ``'global'``
        transformation, so it is in the transcripts' z units whenever the element was
        built that way (see :meth:`zstack_image`). ``read()`` returns the full-resolution
        plane as ``(c, y, x)``.

        ``images`` may be a **list of keys**, one per stain: the planes of every element
        are pooled into one list with a unique ``index`` (so each stain plane is
        thresholded and sampled on its own — the anchors of all stains are then pooled),
        and each entry carries its own ``image_key``. All the elements must share the
        pixel grid (built with the same ``like=``); their z axes need not coincide.

        :returns: list of ``{'index', 'z', 'read', 'source', 'image_key'}`` in plane order.
        """
        from spatialdata.transformations import get_transformation as _get_tf
        if isinstance(images, (list, tuple)):
            pooled = []
            for key in images:
                for pl in Analysis._sdata_image_planes(sdata, key):
                    pl = dict(pl); pl['image_key'] = key
                    pooled.append(pl)
            for i, pl in enumerate(pooled):
                pl['index'] = i          # unique across stains (plane_otsu key)
            return pooled
        if not isinstance(images, str):
            raise TypeError(f'images= is the key of a 3D image element in sdata.images '
                            f'(or a list of keys), got {type(images).__name__} {images!r}')
        if images not in sdata.images:
            raise KeyError(f"sdata.images has no {images!r} (available: {list(sdata.images.keys())}). "
                           f"For z=3 add the z-stack as a 3D image element first, e.g. "
                           f"sdata.images[{images!r}] = Analysis.zstack_image(path, like=sdata.images[...])")
        node = sdata.images[images]
        if not hasattr(node, 'values') or hasattr(node, 'children'):
            first = node[list(node.keys())[0]]
            leaf = first['image'] if hasattr(first, 'image') else first[list(first.data_vars)[0]]
        else:
            leaf = node
        if 'z' not in leaf.dims:
            raise ValueError(f"sdata.images[{images!r}] has dims {tuple(leaf.dims)}: z=3 needs a 3D "
                             f"image element with a z axis (see Analysis.zstack_image)")
        M = np.asarray(_get_tf(node, to_coordinate_system='global').to_affine_matrix(
            input_axes=('x', 'y', 'z'), output_axes=('x', 'y', 'z')), dtype=np.float64)
        if not (np.allclose(M[2, :2], 0.0) and np.allclose(M[:2, 2], 0.0)):
            raise ValueError(f"sdata.images[{images!r}]: its z axis is mixed with x/y in 'global'; "
                             f"only a z scale/translation on top of the x/y placement is supported")
        zc = (np.asarray(leaf.coords['z'].values, dtype=np.float64) if 'z' in leaf.coords
              else np.arange(leaf.sizes['z']) + 0.5)
        zs = M[2, 2] * zc + M[2, 3]
        srcs = (getattr(node, 'attrs', None) or {}).get('ssaft_plane_sources')
        if srcs is not None and len(srcs) == leaf.sizes['z']:
            def _read(k):
                import tifffile
                path, _, zk = str(srcs[k]).partition('[Z=')
                with tifffile.TiffFile(path) as tf:
                    if zk:
                        plane = tf.series[0].pages[int(zk.rstrip(']'))].asarray(maxworkers=16)
                    else:
                        plane = tf.series[0].asarray(maxworkers=16)
                plane = np.asarray(plane)
                return plane[None] if plane.ndim == 2 else plane
        else:
            def _read(k):
                return np.asarray(leaf.isel(z=k).data)
        return [dict(index=k, z=float(zs[k]), read=(lambda k=k: _read(k)),
                     source=f'images[{images!r}][z={k}]', image_key=images)
                for k in range(leaf.sizes['z'])]

    @staticmethod
    def _tile_z_range(tile_path):
        """``(z_min, z_max)`` of a tile's transcripts (the first points element, as
        :meth:`_parse_transcripts` reads it)."""
        import spatialdata as _sd
        s = _sd.read_zarr(str(tile_path), selection=('points',))
        pts = s.points[list(s.points.keys())[0]]
        if 'z' not in pts.columns:
            raise ValueError(f"{tile_path}: the transcripts carry no z column — re-slice "
                             f"without --drop-z")
        z = pts['z']
        lo, hi = z.min(), z.max()
        lo = lo.compute() if hasattr(lo, 'compute') else lo
        hi = hi.compute() if hasattr(hi, 'compute') else hi
        return float(lo), float(hi)

    @staticmethod
    def _rgb_to_intensity(img, intensity='luminance'):
        """Collapse an (H, W, 3) RGB image to a single-channel intensity.

        ``intensity='luminance'`` uses the BT.601 perceptual weighting
        (``0.299 R + 0.587 G + 0.114 B``) — correct for brightfield / H&E
        photographs where the channels are correlated. ``intensity='max'``
        uses the per-pixel channel maximum — correct for fluorescence
        composite images (e.g. CosMx CellComposite), where the tissue
        signal may live almost entirely in one channel (e.g. blue DAPI
        nuclei) and the luminance weighting would crush it (blue is only
        weighted 0.114, so a blue-dominant composite would be discarded).

        Parameters
        ----------
        img : ndarray (H, W, 3)
        intensity : {'luminance', 'max'}, default 'luminance'

        Returns
        -------
        gray : ndarray (H, W) float
        """
        rgb = img[:, :, :3].astype(np.float64)
        if intensity == 'max':
            return rgb.max(axis=-1)
        if intensity == 'luminance':
            return (0.299 * rgb[:, :, 0]
                    + 0.587 * rgb[:, :, 1]
                    + 0.114 * rgb[:, :, 2])
        raise ValueError(
            f"intensity must be 'luminance' or 'max', got {intensity!r}.")

    def _otsu_thresholding(self, he_imgs, revert_image=True,
                           intensity='luminance', mult=None):
        """
        Compute per-image Otsu threshold and return the median.

        The threshold is computed on a tissue-bright intensity image so the
        downstream mask (``_select_vectors``) keeps high-signal pixels.
        ``revert_image=True`` inverts a bright-background image (H&E) to make
        the tissue bright; ``revert_image=False`` uses the grayscale directly
        for an already tissue-bright image (polyT). The same ``revert_image``
        MUST be used in :meth:`_select_vectors` so the threshold and mask
        operate on the same representation.

        Parameters
        ----------
        he_imgs : list of ndarray (H, W, 3)  RGB images
        revert_image : bool, default True
            ``True``  → invert (bright background, dark tissue; e.g. H&E).
            ``False`` → no inversion (dark background, bright tissue; e.g. polyT).
        intensity : {'luminance', 'max'}, default 'luminance'
            How RGB is collapsed to intensity (see :meth:`_rgb_to_intensity`).
            ``'luminance'`` for brightfield/H&E; ``'max'`` for fluorescence
            composites (e.g. CosMx CellComposite) so a single-channel signal
            such as blue DAPI is not crushed. MUST match the value used in
            :meth:`_select_vectors`.
        mult : float, optional
            Scale every per-image Otsu value by this factor before returning it
            (``None`` = the Otsu values as computed). Below 1 lowers the threshold
            and widens the tissue mask.

        Returns
        -------
        median_otsu : float
        otsu_vals   : list of float  (per-image values)
        """
        import cv2

        otsu_vals = []
        for img in he_imgs:
            gray = self._rgb_to_intensity(img, intensity=intensity)
            inv = ((255.0 - gray) if revert_image else gray).astype(np.uint8)
            val, _ = cv2.threshold(inv, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            otsu_vals.append(float(val))
        if mult is not None:
            otsu_vals = [v * float(mult) for v in otsu_vals]
        return float(np.median(otsu_vals)), otsu_vals

    def _select_vectors(self, he_img, sdata, otsu_threshold,
                        element_name=None,
                        bandwidth=2.5, sampling_distance=1.0,
                        prune_coefficient=4.3, size_after_normalization=10,
                        log_transform=True, n_samples=5, block_um=10,
                        seed=42, filter_is_gene=True, genes=None,
                        revert_image=True, intensity='luminance',
                        kernel='gaussian', gpu_device='cuda',
):
        """
        Mask → position selection → KDE → normalization for a single image/sdata.

        Parameters
        ----------
        he_img        : ndarray (H, W, 3)  RGB image
        sdata         : spatialdata.SpatialData
        otsu_threshold: float  threshold to apply (from _otsu_thresholding)

        Returns
        -------
        gx_he, gy_he, gz_he : ndarray  voxel coords (local to this sdata)
        normalized_vectors  : ndarray (N, n_genes)
        vf_norm             : ndarray (N,)
        genes               : list[str]
        coordinate_offset   : list[float]  [x_min, y_min, z_min] in µm
        sample_stats        : dict
        """
        import cv2
        from scipy.ndimage import zoom
        from scipy.spatial import cKDTree
        from spatialdata.transformations import get_transformation as _get_tf

        H_he, W_he = he_img.shape[:2]

        he_gray = self._rgb_to_intensity(he_img, intensity=intensity)
        he_inv = (255.0 - he_gray) if revert_image else he_gray
        he_sub = np.clip(he_inv - otsu_threshold, 0.0, None)
        he_mask = (np.log1p(he_sub) > 0).astype(np.uint8)
        n_mask_pixels = int(he_mask.sum())

        if getattr(self.dataset, 'shape', None) is None:
            self._prepare_grid(
                sdata,
                element_name=element_name,
                bandwidth=bandwidth,
                sampling_distance=sampling_distance,
                prune_coefficient=prune_coefficient,
                size_after_normalization=size_after_normalization,
            )
        vx, vy, vz = self.dataset.shape

        he_grid = zoom(he_mask, (vy / H_he, vx / W_he), order=0) > 0

        meta = self._parse_transcripts(
            sdata, sampling_distance=sampling_distance,
            filter_is_gene=filter_is_gene
        )

        ndim = 2 if vz == 1 else 3
        # the anchor cut is "two molecules' worth of evidence", so it has to be this
        # kernel's own peak -- the gaussian formula would cut an Epanechnikov field
        # at the wrong density (a factor ~2 at the default bandwidth)
        from ._count import kernel_scale, kernel_peak
        expression_threshold = kernel_peak(kernel_scale(bandwidth, ndim, kernel), ndim, kernel)
        norm_threshold = expression_threshold * 2

        x_scale = y_scale = None
        for _etype, _ename, _elem in sdata.gen_elements():
            if _etype == 'points':
                _sc = _get_tf(_elem, get_all=True)
                _tf = _sc.get('global', None)
                if _tf is not None and hasattr(_tf, 'scale'):
                    x_scale = _tf.scale[0]
                    y_scale = _tf.scale[1]
                else:
                    try:
                        _aff = _tf.to_affine_matrix(input_axes=('x', 'y'), output_axes=('x', 'y'))
                        x_scale = abs(_aff[0, 0]) if _aff.shape[0] > 0 else 1.0
                        y_scale = abs(_aff[1, 1]) if _aff.shape[0] > 1 else 1.0
                    except Exception:
                        x_scale = 1.0
                        y_scale = 1.0
                break
        if x_scale is None:
            raise RuntimeError('No points element found in sdata.')

        ys_all, xs_all = np.where(he_grid)
        block_px = (block_um * x_scale) * (block_um * y_scale)
        n_select = min(int(n_mask_pixels / block_px * n_samples), len(ys_all))
        if n_select == 0:
            empty_int = np.empty(0, dtype=np.int64)
            empty_nvecs = np.empty((0, 0), dtype=np.float64)
            empty_vf = np.empty(0, dtype=np.float64)
            sample_stats = {
                'n_mask_pixels': n_mask_pixels,
                'block_px': float(block_px),
                'n_select': 0,
                'candidate_pixels': int(he_grid.sum()),
                'dropped_xy_without_transcript_support': 0,
                'oversampled_count': 0,
                'kept_count': 0,
                'norm_threshold': float(norm_threshold),
            }
            return (empty_int, empty_int, empty_int, empty_nvecs, empty_vf,
                    [], meta['coordinate_offset'], sample_stats)

        rng = np.random.default_rng(seed)
        idx = rng.choice(len(ys_all), n_select, replace=False)
        gx_cand = xs_all[idx].astype(np.int64)
        gy_cand = ys_all[idx].astype(np.int64)

        rp = int(np.floor((bandwidth * prune_coefficient) / sampling_distance + 1e-12))

        dropped_xy = 0
        if not meta['is_3d'] or vz == 1:
            gx_raw = gx_cand
            gy_raw = gy_cand
            gz_raw = np.zeros(len(gx_cand), dtype=np.int64)
        else:
            xy_grid_coords = np.column_stack([meta['xc_g'], meta['yc_g']]).astype(np.float64)
            xy_tree = cKDTree(xy_grid_coords)
            gx_list, gy_list, gz_list = [], [], []
            for x, y in zip(gx_cand, gy_cand):
                neighbors = xy_tree.query_ball_point([float(x), float(y)], r=rp, p=np.inf)
                if len(neighbors) == 0:
                    dropped_xy += 1
                    continue
                z_vals = meta['zc_g'][np.asarray(neighbors, dtype=np.int64)]
                z_vals = z_vals[(z_vals >= 0) & (z_vals < vz)]
                if z_vals.size == 0:
                    dropped_xy += 1
                    continue
                gx_list.append(x)
                gy_list.append(y)
                gz_list.append(vz // 2)
            if len(gx_list) == 0:
                raise RuntimeError('No sampled voxels had nearby transcript support.')
            gx_raw = np.asarray(gx_list, dtype=np.int64)
            gy_raw = np.asarray(gy_list, dtype=np.int64)
            gz_raw = np.asarray(gz_list, dtype=np.int64)

        coords = np.unique(np.stack([gx_raw, gy_raw, gz_raw], axis=1), axis=0)
        gx_raw, gy_raw, gz_raw = coords[:, 0], coords[:, 1], coords[:, 2]

        query_voxels = np.stack([gx_raw, gy_raw, gz_raw], axis=1)
        normalized_vectors, vf_norm, genes, coordinate_offset = \
            self._compute_normalized_vectors(
                query_voxels, meta,
                genes=genes,
                bandwidth=bandwidth,
                sampling_distance=sampling_distance,
                prune_coefficient=prune_coefficient,
                size_after_normalization=size_after_normalization,
                log_transform=log_transform,
                kernel=kernel, gpu_device=gpu_device,
            )

        keep = vf_norm > norm_threshold
        gx_he = gx_raw[keep]
        gy_he = gy_raw[keep]
        gz_he = gz_raw[keep]
        normalized_vectors = normalized_vectors[keep]
        vf_norm = vf_norm[keep]

        sample_stats = {
            'n_mask_pixels': n_mask_pixels,
            'block_px': float(block_px),
            'n_select': n_select,
            'candidate_pixels': int(he_grid.sum()),
            'dropped_xy_without_transcript_support': int(dropped_xy),
            'oversampled_count': len(gx_raw),
            'kept_count': int(keep.sum()),
            'norm_threshold': float(norm_threshold),
        }

        return gx_he, gy_he, gz_he, normalized_vectors, vf_norm, genes, coordinate_offset, sample_stats

    def _select_vectors_3d(self, planes, sdata, image_key, otsu_threshold=None, otsu_mult=None,
                           plane_otsu=None,
                           element_name=None, bandwidth=2.5, sampling_distance=1.0,
                           prune_coefficient=4.3, size_after_normalization=10,
                           log_transform=True, n_samples=5, block_um=10, seed=42,
                           filter_is_gene=True, genes=None, revert_image=True,
                           intensity='luminance', kernel='gaussian',
                           gpu_device='cuda'):
        """3D counterpart of :meth:`_select_vectors`: one tissue mask per imaging plane.

        Every plane of the z-stack is an image of its own, so it is thresholded with its
        own Otsu value (unless ``otsu_threshold`` is given) and sampled at the same density
        — ``n_samples`` per ``block_um`` x ``block_um`` µm of ITS mask. The total therefore
        grows with the number of planes, as a per-image rule should. An anchor sits on the
        grid plane nearest its image plane and its vector is the 3D KDE there, so the
        dictionary is built from the same kind of profile the 3D map is scored against; a
        z-projected vector mixes whatever lies above and below the point.

        ``planes`` comes from :meth:`_sdata_image_planes` on ``sdata.images[image_key]``, a 3D
        image element; each plane's ``z`` is in the units of the transcripts' z column.
        Planes outside the transcripts' z range are skipped.

        Returns the same tuple as :meth:`_select_vectors`;
        ``sample_stats['per_plane']`` holds each plane's threshold, mask and counts.
        """
        from scipy.ndimage import zoom
        from ._count import kernel_scale, kernel_peak

        if getattr(self.dataset, 'shape', None) is None:
            self._prepare_grid(sdata, element_name=element_name, bandwidth=bandwidth,
                               sampling_distance=sampling_distance,
                               prune_coefficient=prune_coefficient,
                               size_after_normalization=size_after_normalization,
                               n_dims=3)
        vx, vy, vz = self.dataset.shape
        meta = self._parse_transcripts(sdata, sampling_distance=sampling_distance,
                                       filter_is_gene=filter_is_gene, n_dims=3)
        norm_threshold = kernel_peak(kernel_scale(bandwidth, 3, kernel), 3, kernel) * 2

        x_scale, y_scale = self._points_px_per_um(sdata)
        block_px = (block_um * x_scale) * (block_um * y_scale)

        # _select_vectors' support test -- a transcript within rp nodes (inf-norm) in x/y --
        # as one integral image instead of a KD-tree query per candidate
        rp = int(np.floor((bandwidth * prune_coefficient) / sampling_distance + 1e-12))
        xc = np.clip(np.asarray(meta['xc_g'], dtype=np.int64), 0, vx)
        yc = np.clip(np.asarray(meta['yc_g'], dtype=np.int64), 0, vy)
        cnt = np.bincount(yc * (vx + 1) + xc,
                          minlength=(vy + 1) * (vx + 1)).reshape(vy + 1, vx + 1)
        ii = np.zeros((vy + 2, vx + 2), dtype=np.int64)
        ii[1:, 1:] = cnt.cumsum(0).cumsum(1)
        del cnt, xc, yc

        def _supported(gx, gy):
            ya, yb = np.clip(gy - rp, 0, vy + 1), np.clip(gy + rp + 1, 0, vy + 1)
            xa, xb = np.clip(gx - rp, 0, vx + 1), np.clip(gx + rp + 1, 0, vx + 1)
            return (ii[yb, xb] - ii[ya, xb] - ii[yb, xa] + ii[ya, xa]) > 0

        oz = float(meta['coordinate_offset'][2])
        coordinate_offset = meta['coordinate_offset']
        rng = np.random.default_rng(seed)
        out_x, out_y, out_z, out_v, out_n, per_plane = [], [], [], [], [], []
        for pl in planes:
            gz = int(round((float(pl['z']) - oz) / sampling_distance))
            row = dict(plane=int(pl['index']), z=float(pl['z']), grid_z=gz)
            if gz < 0 or gz >= vz:
                row['skipped'] = 'outside the transcripts z range'
                per_plane.append(row)
                if self.verbose:
                    print(f"  plane {row['plane']:>2d} (z={row['z']:.2f}): skipped, "
                          f"outside the transcripts' z range", flush=True)
                continue
            raw = pl['read']()
            pl_key = pl.get('image_key', image_key)   # this stain's element (shared grid)
            u8 = self._image_to_u8_plane(raw)
            if u8 is not None:
                # Single-channel plane (a DAPI stack): the RGB replication, the float intensity,
                # Otsu's uint8 image and the mask test all depend on the pixel value alone, so
                # they are tabulated over the 256 values and looked up -- the same numbers as
                # the full-array path, without float64 copies of a multi-gigapixel plane.
                del raw
                inv_lut, inv_u8_lut = self._plane_luts(revert_image, intensity)
                img = self._he_array_axis_aligned(
                    sdata, image_key=pl_key, element_name=element_name,
                    sampling_distance=sampling_distance, revert_image=revert_image,
                    arr=u8[..., None], gray_u8=True)
                if plane_otsu is not None and int(pl['index']) in plane_otsu:
                    thr = float(plane_otsu[int(pl['index'])])
                elif otsu_threshold is None:
                    import cv2
                    oimg = self._he_array_axis_aligned(
                        sdata, image_key=pl_key, element_name=element_name,
                        sampling_distance=sampling_distance, pad=False,
                        revert_image=revert_image, arr=u8[..., None], gray_u8=True)
                    val, _ = cv2.threshold(inv_u8_lut[oimg[..., 0]], 0, 255,
                                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    thr = float(val) * (float(otsu_mult) if otsu_mult is not None else 1.0)
                    del oimg
                else:
                    thr = float(otsu_threshold)
                del u8
                mask_lut = (np.log1p(np.clip(inv_lut - thr, 0.0, None)) > 0).astype(np.uint8)
                mask = mask_lut[img[..., 0]]
                del img
            else:
                img = self._he_array_axis_aligned(
                    sdata, image_key=pl_key, element_name=element_name,
                    sampling_distance=sampling_distance, revert_image=revert_image, arr=raw)
                if plane_otsu is not None and int(pl['index']) in plane_otsu:
                    # one threshold per plane decided across all tiles (chunked=True)
                    thr = float(plane_otsu[int(pl['index'])])
                elif otsu_threshold is None:
                    oimg = self._he_array_axis_aligned(
                        sdata, image_key=pl_key, element_name=element_name,
                        sampling_distance=sampling_distance, pad=False,
                        revert_image=revert_image, arr=raw)
                    thr, _ = self._otsu_thresholding([oimg], revert_image=revert_image,
                                                     intensity=intensity, mult=otsu_mult)
                    del oimg
                else:
                    thr = float(otsu_threshold)
                del raw
                gray = self._rgb_to_intensity(img, intensity=intensity)
                del img
                inv = (255.0 - gray) if revert_image else gray
                mask = (np.log1p(np.clip(inv - thr, 0.0, None)) > 0).astype(np.uint8)
                del gray, inv
            n_mask = int(mask.sum())
            grid = zoom(mask, (vy / mask.shape[0], vx / mask.shape[1]), order=0) > 0
            del mask
            ys, xs = np.where(grid)
            del grid
            n_sel = min(int(n_mask / block_px * n_samples), len(ys))
            row.update(otsu=float(thr), n_mask_pixels=n_mask, n_select=n_sel)
            if n_sel == 0:
                row.update(dropped_xy_without_transcript_support=0, kept=0)
                per_plane.append(row)
                continue
            pick = rng.choice(len(ys), n_sel, replace=False)
            gx = xs[pick].astype(np.int64)
            gy = ys[pick].astype(np.int64)
            del xs, ys, pick
            sup = _supported(gx, gy)
            coords = np.unique(np.stack([gx[sup], gy[sup],
                                         np.full(int(sup.sum()), gz, dtype=np.int64)], 1),
                               axis=0)
            nv, vfn, genes, coordinate_offset = self._compute_normalized_vectors(
                coords, meta, genes=genes, bandwidth=bandwidth,
                sampling_distance=sampling_distance, prune_coefficient=prune_coefficient,
                size_after_normalization=size_after_normalization,
                log_transform=log_transform, kernel=kernel, gpu_device=gpu_device)
            keep = vfn > norm_threshold
            out_x.append(coords[keep, 0]); out_y.append(coords[keep, 1])
            out_z.append(coords[keep, 2])
            out_v.append(np.asarray(nv[keep], dtype=np.float32))
            out_n.append(np.asarray(vfn[keep]))
            row.update(dropped_xy_without_transcript_support=int((~sup).sum()),
                       kept=int(keep.sum()))
            per_plane.append(row)
            if self.verbose:
                print(f"  plane {row['plane']:>2d} (z={row['z']:.2f}, grid z {gz}): "
                      f"otsu {thr:.0f}, mask {n_mask:,} px, selected {n_sel:,}, "
                      f"kept {row['kept']:,}", flush=True)
            del nv, vfn, coords, keep, sup, gx, gy

        sample_stats = dict(
            n_planes=len(planes), per_plane=per_plane,
            n_mask_pixels=int(sum(r.get('n_mask_pixels', 0) for r in per_plane)),
            block_px=float(block_px),
            n_select=int(sum(r.get('n_select', 0) for r in per_plane)),
            dropped_xy_without_transcript_support=int(
                sum(r.get('dropped_xy_without_transcript_support', 0) for r in per_plane)),
            kept_count=int(sum(r.get('kept', 0) for r in per_plane)),
            norm_threshold=float(norm_threshold))
        if not out_x:
            empty = np.empty(0, dtype=np.int64)
            return (empty, empty, empty, np.empty((0, 0), dtype=np.float32),
                    np.empty(0, dtype=np.float64), genes or [], coordinate_offset, sample_stats)
        cx = np.concatenate(out_x); cy = np.concatenate(out_y); cz = np.concatenate(out_z)
        cv = np.concatenate(out_v, axis=0); cn = np.concatenate(out_n)
        # With several stains each plane is sampled on its own, so two stains can land an
        # anchor on the same voxel; the 3D KDE vector there is identical, so keep one per
        # (x, y, z). No-op for a single stain (coords are already unique per plane).
        _, uniq = np.unique(np.stack([cx, cy, cz], 1), axis=0, return_index=True)
        if len(uniq) < len(cx):
            uniq.sort()
            sample_stats['pooled_before_dedup'] = int(len(cx))
            sample_stats['kept_count'] = int(len(uniq))
            cx, cy, cz, cv, cn = cx[uniq], cy[uniq], cz[uniq], cv[uniq], cn[uniq]
        return (cx, cy, cz, cv, cn, genes, coordinate_offset, sample_stats)

    def compute_vf(self, grid, *, bin_um=2.0, sampling_distance=None, panel=None,
                   sigma_bin=None, bandwidth_um=None, prune_coefficient=4.3,
                   kernel='gaussian',
                   size_after_normalization=10.0, device='cuda', chunk_h=64,
                   chunk_w=512, store_chunk=32, mode='auto',
                   log_transform=True):
        """Build the count-data vector field (GPU blur) and persist it to zarr.

        This is the COUNT-platform entry point (Stereo-seq / Visium HD /
        StrataMap): it rasterizes bin counts and Gaussian-blurs them into
        ``vf_normalized`` + ``vf_norm``, writing only the normalized field.
        Transcript platforms do NOT call this — their field is computed per-chunk
        inside :meth:`map_celltypes`.

        Two independent resolutions (see :func:`ssaft._count.compute_vf_blur`):

        Parameters
        ----------
        grid : GridDataset or AnnData
            A :class:`ssaft.GridDataset` (raw counts are aggregated to a
            ``bin_um`` grid using ``grid.origin``), or a pre-aggregated bin-level
            AnnData with ``obs['array_row'/'array_col']``.
        bin_um : float
            **Blur raster** bin size in µm — the grid the Gaussian is computed on
            (finer = more accurate blur). Ignored if ``grid`` is already gridded.
        sampling_distance : float, optional
            **Output vector-field** grid spacing in µm. The blurred raster is
            decimated by ``round(sampling_distance / bin_um)`` so the stored vf can
            be coarser than the blur raster (fewer voxels on disk). Defaults to
            ``bin_um`` (output == raster).
        bandwidth_um : float, optional
            Gaussian bandwidth (σ) in **µm** — the intuitive knob. Converted to
            bin units internally (``sigma_bin = bandwidth_um / bin_um``). Give
            EITHER ``bandwidth_um`` OR ``sigma_bin`` (bin units); ``bandwidth_um``
            wins if both. Default: ``sigma_bin=1`` (bandwidth = ``bin_um`` µm).
        prune_coefficient : float, default 4.3
            Gaussian kernel truncation radius in σ units: ``radius = ceil(
            prune_coefficient * sigma_bin)`` bins (= ``prune_coefficient *
            bandwidth`` µm). Default 4.3 (radius ≈ cell diameter);
            lower (e.g. 3.0 ≈ 99.7% mass) truncates tighter and runs faster.
            Applies to the gaussian and (at the same relative height) the
            exponential; the compact kernels stop at their own support.
        kernel : {'gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'}, default 'gaussian'
            KDE kernel (``ssaft._count.KDE_KERNELS``). The bandwidth knob
            (``sigma_bin`` / ``bandwidth_um``) means the same thing for all of them:
            each kernel's own scale is derived from it by variance matching
            (:func:`ssaft._count.kernel_scale`, e.g. the Epanechnikov support
            ``H = sigma * sqrt(ndim + 4)``), so they smooth over the same width. The
            compact kernels (tophat, Epanechnikov, linear, cosine) reach zero at
            their support instead of being truncated at ``4.3 sigma`` and so touch
            far fewer voxels. Applies to the count raster and to the transcript
            shortcut alike.

            On the COUNT raster a compact kernel is not always the fast choice: the
            gaussian is separable there, so its cost grows linearly with the radius
            while a radial kernel's grows quadratically. Measured end to end, the
            Epanechnikov wins below ``sigma_bin`` about 6 (~0.8x at 1-4) and loses
            above it (2.3x at 16). On the transcript path the compact kernels are
            2-3x faster at any width.
        panel : list[str], optional
            Gene panel to restrict to before rasterization.
        chunk_h, chunk_w : int
            GPU **compute** tile (output rows × cols per blur pass). Large = fewer
            kernel launches / less halo recomputation. Does NOT set the on-disk chunk.
        store_chunk : int or (h, w), default 32
            On-disk chunk shape of ``vf_normalized`` (gene axis always full). Small
            SQUARE tiles let a sparse ``downsample_vector`` gather decode only the
            tiles containing sampled voxels instead of full-width row bands. Keep
            ``chunk_h``/``chunk_w`` integer multiples of it (defaults are). ``None``
            makes the on-disk chunk match the compute tile instead.

        :param sigma_bin: Gaussian bandwidth in bin units. Alternative to
            ``bandwidth_um``; ignored when ``bandwidth_um`` is given. Defaults to 1.
        :type sigma_bin: float or None
        :param size_after_normalization: Target L1 scale each voxel vector is
            normalized to before the optional log1p.
        :type size_after_normalization: float
        :param device: Torch device string used for the GPU blur (e.g. ``'cuda'``).
        :type device: str
        :param mode: Input-regime selector; ``'auto'`` detects count vs transcript
            SpatialData, ``'transcript'`` forces the KDE-field producer branch.
        :type mode: str
        :param log_transform: If True, apply log1p to the normalized field.
        :type log_transform: bool
        """
        from ._count import compute_vf_blur
        from ._griddataset import GridDataset
        from ._io import aggregate_to_grid

        try:
            from spatialdata import SpatialData as _SD
        except Exception:  # noqa: BLE001
            _SD = ()
        if _SD and isinstance(grid, _SD):
            _gd = None
            _is_transcript = (mode == 'transcript')
            if mode == 'auto':
                try:
                    _gd = GridDataset.from_sdata(grid)
                except Exception:  # noqa: BLE001
                    _is_transcript = True
            if _is_transcript:
                bw = float(bandwidth_um) if bandwidth_um is not None else 2.5
                self._transcript_gpu(
                    grid, do_map=False, save_vf=True,
                    bandwidth=bw, kernel=kernel,
                    sampling_distance=(1.0 if sampling_distance is None
                                       else float(sampling_distance)),
                    prune_coefficient=prune_coefficient,
                    size_after_normalization=size_after_normalization,
                    log_transform=log_transform, gpu_device=device)
                return None
            self._count_sdata = grid
            self.dataset._input_data_type = 'spatialdata'
            self.dataset._original_input_data = grid
            grid = _gd if _gd is not None else GridDataset.from_sdata(grid)

        if isinstance(grid, GridDataset):
            adata = grid.raw if panel is None else grid.raw[:, [g for g in panel
                                                    if g in set(grid.raw.var_names)]]
            bin_adata = aggregate_to_grid(adata, bin_um, origin=grid.origin)
            self._count_grid = grid
        else:
            bin_adata = grid if panel is None else grid[:, [g for g in panel
                                                    if g in set(grid.var_names)]]
            bin_um = float(bin_adata.uns.get("bin_size_um", bin_um))
        if bandwidth_um is not None:
            sigma_bin = float(bandwidth_um) / float(bin_um)
        elif sigma_bin is None:
            sigma_bin = 1.0
        if sigma_bin < 0.5:
            # The blur happens on the bin raster, so a sigma under half a bin cannot reach the
            # neighbouring bin and the field stays a lattice of isolated bins. It means the
            # raster is finer than the data it carries (e.g. 8 µm Visium HD bins blurred on a
            # 2 µm raster) or the bandwidth is too small for the bin size.
            warnings.warn(
                f'bandwidth {sigma_bin * bin_um:g} µm is {sigma_bin:.2f} of a {bin_um:g} µm '
                'bin: the gaussian does not reach the next bin, so the field will be isolated '
                'bins rather than a continuous one. Raise the bandwidth or lower bin_um to the '
                "data's own bin size.", UserWarning, stacklevel=2)
        compute_vf_blur(self, bin_adata, sigma_bin=sigma_bin, chunk_h=chunk_h,
                        chunk_w=chunk_w, store_chunk=store_chunk,
                        bin_size_um=bin_um, sampling_distance=sampling_distance,
                        prune_coefficient=prune_coefficient, kernel=kernel,
                        size_after_normalization=size_after_normalization,
                        device=device)
        # A grid read from spatialdata_io output lives in that reader's pixel frame, not in
        # 'global' µm: record the frame so the map and the cells written later, in any
        # process, go back into it (Dataset._map_frame).
        zg = self.dataset.zarr_group
        frame = (grid.meta or {}).get('frame') if isinstance(grid, GridDataset) else None
        if frame is not None:
            zg.attrs['count_frame'] = dict(frame, origin=[float(grid.origin[0]),
                                                          float(grid.origin[1])])
        elif 'count_frame' in zg.attrs:
            del zg.attrs['count_frame']
        return bin_adata

    def downsample_vector(self, sdata=None, *, z, grid=None, images=None, auto_gpu=True,
                          gpu_device='cuda', **kwargs):
        """Public entry point with automatic GPU/CPU dispatch.

        Returns an :class:`anndata.AnnData` of the sampled voxels (``X`` =
        normalized_vectors, ``obsm['spatial']``, ``obs['vf_norm']``, ``var`` =
        genes) — feed it to ``sc.pp.pca`` / ``sc.pp.neighbors`` / ``sc.tl.leiden``::

            adata = analysis.downsample_vector(sdata, ...)
            sc.pp.pca(adata); sc.pp.neighbors(adata)
            sc.tl.leiden(adata)               # writes adata.obs['leiden'] in place


        Two modes, auto-detected:

        * **count** — when ``grid`` is given (or omitted but a ``compute_vf`` grid
          was remembered) AND the store already holds a blur ``vf_normalized``:
          voxels are sampled from the tissue image and *looked up* in that field
          (no KDE). Delegates to
          :func:`ssaft._count.select_vectors_from_image`.
        * **transcript** — otherwise: voxels are sampled from the H&E tissue mask
          and KDE is computed at each (the original path below).


        If ``auto_gpu`` is True (default) and a usable CUDA device is
        detected (``torch + triton + cuda.is_available()``), the underlying
        ``ssaft.utils.kde_at_points`` is transparently routed to the Triton
        GPU kernel for the duration of this call. Otherwise the original
        C++/OpenMP CPU implementation is used. The user therefore does
        not need to wrap the call in ``with gpu_kde(...): ...``.

        ``kernel`` (transcript mode; default ``'gaussian'``, any of
        ``ssaft._count.KDE_KERNELS``) builds the anchors with that kernel, on the GPU
        and the CPU alike. It MUST match the ``kernel`` later given to
        :meth:`map_celltypes`: these vectors are clustered, annotated and summed
        into the mapper's dictionary, so mixing two kernels scores one kernel's
        field against another's dictionary. The count mode ignores it: that field's
        kernel was fixed by :meth:`compute_vf`.

        All other arguments are forwarded verbatim to
        :meth:`_downsample_vector_impl` (see its docstring).

        Parameters
        ----------
        sdata : spatialdata.SpatialData, optional
            Forwarded to the implementation (used when ``chunked=False``).
        z : {2, 3}
            **Required.** Dimensionality of the analysis, stored as ``self.n_dims`` and in
            the store as ``attrs['n_dims']``; :meth:`map_celltypes` builds its field in the
            same dimension, so the dictionary and the voxels it scores never disagree.

            * ``2`` — the original path: the tissue mask comes from the 2D image in
              ``sdata.images[image_key]``, anchors sit on one plane and their vectors are
              a 2D KDE. Transcripts' z, if any, is ignored throughout.
            * ``3`` — every plane of ``sdata.images[images]`` is thresholded and sampled on
              its own and each anchor's vector is the 3D KDE at its plane (see
              :meth:`_select_vectors_3d`). Needs ``images``: a 2D image carries no depth,
              so it cannot say where along z the tissue is.
        images : str, optional
            ``z=3`` only. The key of a 3D image element ``(c, z, y, x)`` in
            ``sdata.images`` — images are always read from the SpatialData, never from a
            path. Add the z-stack to the SpatialData first, e.g.
            ``sdata.images['morphology_zstack'] = Analysis.zstack_image(
            '.../morphology.ome.tif', like=sdata.images['morphology_focus'])``. Each
            plane's z comes from the element's ``'global'`` transformation and must be in
            the transcripts' z units. With ``chunked=True`` every tile must hold the
            element (``slice_spatialdata.py --image-key`` keeps it).
        grid : GridDataset, optional
            Count-regime grid to look voxels up in the precomputed blur field.
            When None, a ``compute_vf`` grid remembered on the analysis is used.
        auto_gpu : bool, default True
            If False, force CPU regardless of CUDA availability.
        gpu_device : str, default ``'cuda'``
            Torch device string, e.g. ``'cuda'``, ``'cuda:0'``, ``'cuda:1'``.
        **kwargs
            All other ``downsample_vector`` parameters (chunked, sdata_dir,
            n_samples, block_um, bandwidth, sampling_distance,
            revert_image, ...).

            otsu_mult : float, optional
                Scale the automatically computed Otsu threshold by this factor.
                ``None`` (default) keeps the Otsu value as computed; ``0.6``
                lowers it to 60% and widens the tissue mask (every image in 2D,
                every tile when chunked, every plane in 3D). Transcript mode only,
                and not together with ``otsu_threshold``.
            revert_image : bool, default True
                Background polarity of the staining image used for the tissue
                mask. ``True`` for bright-background / dark-tissue images such
                as H&E (the image is inverted before Otsu thresholding).
                ``False`` for dark-background / bright-tissue images such as
                polyT (the grayscale is thresholded directly, no inversion).
            intensity : {'luminance', 'max'}, default 'luminance'
                How the RGB staining image is collapsed to a single-channel
                intensity for the tissue mask. ``'luminance'`` (BT.601
                weighting) suits brightfield / H&E photographs. ``'max'``
                (per-pixel channel maximum) suits fluorescence composites
                such as CosMx CellComposite, where the tissue signal can sit
                almost entirely in one channel (e.g. blue DAPI nuclei) and
                luminance would crush it (blue weighted only 0.114).
        """
        if z not in (2, 3):
            raise ValueError(f'z must be 2 or 3 (the dimensionality of the analysis), got {z!r}')
        z = int(z)
        if z == 3 and images is None:
            raise ValueError(
                "z=3 needs images=: the key of a 3D image element (c, z, y, x) in sdata.images. "
                "A 2D image carries no depth, so it cannot say where along z the tissue is. Add "
                "the z-stack first, e.g. sdata.images['morphology_zstack'] = "
                "Analysis.zstack_image('.../morphology.ome.tif', like=sdata.images['morphology_focus'])")
        if images is not None and not (isinstance(images, str) or
                (isinstance(images, (list, tuple)) and images and
                 all(isinstance(x, str) for x in images))):
            raise TypeError(f'images= is the key of an element in sdata.images (or a list of '
                            f'keys, one per stain), not {type(images).__name__} {images!r}; '
                            f'add a z-stack file with Analysis.zstack_image first')
        if z == 2 and images is not None:
            raise ValueError('images= is only used with z=3; with z=2 the tissue mask comes '
                             'from sdata.images[image_key]')
        self.n_dims = z
        grid = grid if grid is not None else getattr(self, '_count_grid', None)
        zg = self.dataset.zarr_group
        is_blur = ('vf_normalized' in zg) and (zg.attrs.get('backend') == 'blur')
        if grid is None and is_blur and sdata is not None:
            try:
                from spatialdata import SpatialData as _SD
            except Exception:  # noqa: BLE001
                _SD = ()
            if _SD and isinstance(sdata, _SD) and len(getattr(sdata, 'tables', {})):
                from ._griddataset import GridDataset
                grid = GridDataset.from_sdata(sdata)
                self._count_sdata = sdata
                self.dataset._input_data_type = 'spatialdata'
                self.dataset._original_input_data = sdata
        if grid is not None and is_blur:
            if z != 2:
                raise ValueError('the count backend field is a 2D blur raster: downsample it '
                                 'with z=2')
            if kwargs.get('otsu_mult') is not None:
                raise ValueError('otsu_mult is supported for the transcript backend only; the '
                                 'count backend takes otsu_threshold')
            zg.attrs['n_dims'] = 2
            from ._count import select_vectors_from_image
            grid_um = float(zg.attrs.get('sampling_distance',
                            zg.attrs.get('bin_size_um', 1.0)))
            sel_keys = ('n_samples_per_block', 'block_um', 'otsu_threshold',
                        'norm_threshold_mult', 'vf_norm_percentile',
                        'lookup_chunk_h', 'seed',
                        'lookup_n_workers', 'lookup_ram_gb')
            sel_kwargs = {k: kwargs[k] for k in sel_keys if k in kwargs}
            if 'n_samples' in kwargs and 'n_samples_per_block' not in sel_kwargs:
                sel_kwargs['n_samples_per_block'] = kwargs['n_samples']
            select_vectors_from_image(self, grid, bin_size_um=grid_um, **sel_kwargs)
            return self._downsampled_adata()

        from contextlib import nullcontext as _nullcontext
        if auto_gpu:
            try:
                from ._kde_gpu import auto_kde_context as _auto_ctx
                _ctx = _auto_ctx(gpu_device)
            except ImportError:
                _ctx = _nullcontext()
        else:
            _ctx = _nullcontext()
        with _ctx:
            kwargs.setdefault('gpu_device', gpu_device)
            self._downsample_stats = self._downsample_vector_impl(
                sdata=sdata, n_dims=z, images=images, **kwargs)
        return self._downsampled_adata()

    def _downsample_vector_impl(self, sdata=None, element_name=None,
                         bandwidth=2.5, sampling_distance=1.0, prune_coefficient=4.3,
                         kernel='gaussian', gpu_device='cuda',
                         size_after_normalization=10, log_transform=True,
                         n_samples=5, block_um=10, seed=42,
                         filter_is_gene=True, genes=None,
                         chunked=False, sdata_dir=None,
                         image_key='he', otsu_threshold=None,
                         revert_image=True, intensity='luminance',
                         n_dims=2, images=None, otsu_mult=None):
        """
        Select representative vectors from H&E-intensity-guided positions.

        Voxels are sampled uniformly at random from the H&E tissue mask,
        with the number of samples proportional to the mask area:
        ``n_select = int(n_mask_pixels / block_px * n_samples)``
        where ``block_px = (block_um * x_scale) * (block_um * y_scale)``.

        The H&E image is now read directly from ``sdata.images[image_key]``;
        the caller is responsible for ensuring it has been cropped to the
        sequenced region and rasterized onto a grid that spatially
        corresponds to the morphology image.

        Stores results in ``ds.downsampled_vectors``, ``ds.normalized_vectors``,
        ``ds.genes``, ``ds.coordinate_offset``, and persists them to zarr.

        Parameters
        ----------
        sdata : spatialdata.SpatialData
            Used when ``chunked=False``. Must contain ``images[image_key]``.
        sdata_dir : str or pathlib.Path
            Used when ``chunked=True``. Directory containing one or more
            ``*.zarr`` SpatialData stores; each must contain ``images[image_key]``.
        image_key : str, default 'he'
            Key under which the staining / morphology image (e.g. H&E or
            polyT) is stored in ``sdata.images``.
        n_samples : int
            Number of voxels to sample per ``block_um × block_um`` µm block.
        block_um : float
            Block size in µm used to determine sampling density.
        revert_image : bool, default True
            Background polarity of the staining image used to build the
            tissue mask (Otsu + mask + bbox padding).
            ``True``  → bright background, dark tissue (e.g. H&E); the image
            is inverted (``255 - gray``) before thresholding.
            ``False`` → dark background, bright tissue (e.g. polyT); the
            grayscale is used directly with no inversion.
        intensity : {'luminance', 'max'}, default 'luminance'
            How the RGB staining image is collapsed to intensity for the
            tissue mask. ``'luminance'`` (BT.601) for brightfield / H&E;
            ``'max'`` (per-pixel channel maximum) for fluorescence composites
            (e.g. CosMx CellComposite) so a single-channel signal such as blue
            DAPI is not crushed by the luminance weighting.
        n_dims : {2, 3}, default 2
            Dimensionality of the analysis (``downsample_vector(z=...)``). ``2`` is the
            original path: one tissue mask from ``sdata.images[image_key]``, anchors on a
            single plane, 2D KDE. ``3`` samples every plane of ``sdata.images[images]``
            (see :meth:`_select_vectors_3d`). Recorded in the store as ``attrs['n_dims']``.
        images : str, optional
            ``n_dims=3`` only: key of the 3D image element in ``sdata.images`` (in every
            tile's SpatialData when ``chunked=True``); see :meth:`_sdata_image_planes`.
        otsu_mult : float, optional
            Scale the automatically computed Otsu threshold by this factor
            (``None``, the default, keeps the Otsu value as computed). It scales
            every per-image value: the one image in 2D, each tile store when
            ``chunked=True`` (so their median scales too), each plane in 3D. A
            value below 1 lowers the threshold and widens the tissue mask — on the
            mouse-kidney Xenium section ``0.6`` took the threshold from 59.0 to 35.4
            and the mask from 12.0% to 23.3% of the lattice, and the anchor count
            from 258,582 to 502,339. Cannot be combined with ``otsu_threshold``,
            which fixes the threshold outright.

        Returns
        -------
        sample_stats : dict
        """
        from pathlib import Path as _Path
        import spatialdata as _sd

        helper_kwargs = dict(
            element_name=element_name,
            bandwidth=bandwidth, sampling_distance=sampling_distance,
            prune_coefficient=prune_coefficient,
            kernel=kernel, gpu_device=gpu_device,
            size_after_normalization=size_after_normalization,
            log_transform=log_transform,
            n_samples=n_samples, block_um=block_um, seed=seed,
            filter_is_gene=filter_is_gene, genes=genes,
            revert_image=revert_image, intensity=intensity,
        )

        self.n_dims = int(n_dims)
        if self.n_dims not in (2, 3):
            raise ValueError(f'n_dims must be 2 or 3, got {n_dims!r}')
        if otsu_mult is not None:
            if otsu_threshold is not None:
                raise ValueError('otsu_mult scales the Otsu threshold ssaft computes; with '
                                 'otsu_threshold given there is none to scale — pass one or '
                                 'the other')
            if not float(otsu_mult) > 0:
                raise ValueError(f'otsu_mult must be > 0, got {otsu_mult!r}')
            otsu_mult = float(otsu_mult)
        if not chunked and self.n_dims == 3:
            if sdata is None:
                raise ValueError('chunked=False requires sdata.')
            planes = self._sdata_image_planes(sdata, images)
            if self.verbose:
                print(f'downsample_vector (3D): {len(planes)} planes of images[{images!r}] at z '
                      f"{[round(pl['z'], 3) for pl in planes]}", flush=True)
            self.dataset.shape = None
            gx_he, gy_he, gz_he, normalized_vectors, vf_norm_all, genes, coordinate_offset, sample_stats = \
                self._select_vectors_3d(planes, sdata, images,
                                        otsu_threshold=otsu_threshold, otsu_mult=otsu_mult,
                                        **helper_kwargs)
            sample_stats['plane_z'] = [pl['z'] for pl in planes]
            if sample_stats.get('kept_count', 0) == 0:
                raise RuntimeError(
                    f"no anchor survived on any of the {len(planes)} planes "
                    f"(mask pixels {sample_stats.get('n_mask_pixels', 0)}). Check that "
                    f"revert_image matches the stain's polarity and that images[{images!r}]'s "
                    f"z transformation puts the planes inside the transcripts' z range.")
            if self.verbose:
                vx, vy, vz = self.dataset.shape
                print(f'downsample_vector: vf grid ({vx}, {vy}, {vz})')
                for k in ('n_mask_pixels', 'block_px', 'n_select',
                          'dropped_xy_without_transcript_support', 'kept_count'):
                    print(f'  {k:40s}: {sample_stats[k]}')

        elif not chunked:
            if sdata is None:
                raise ValueError('chunked=False requires sdata.')

            he_img = self._he_array_axis_aligned(
                sdata, image_key=image_key,
                element_name=element_name, sampling_distance=sampling_distance,
                revert_image=revert_image,
            )
            if otsu_threshold is None:
                otsu_img = self._he_array_axis_aligned(
                    sdata, image_key=image_key,
                    element_name=element_name, sampling_distance=sampling_distance,
                    pad=False, revert_image=revert_image,
                )
                otsu_threshold, _ = self._otsu_thresholding(
                    [otsu_img], revert_image=revert_image, intensity=intensity,
                    mult=otsu_mult)

            self.dataset.shape = None
            gx_he, gy_he, gz_he, normalized_vectors, vf_norm_all, genes, coordinate_offset, sample_stats = \
                self._select_vectors(he_img, sdata, otsu_threshold, **helper_kwargs)

            if sample_stats.get('n_select', 0) == 0:
                raise RuntimeError(
                    f"n_select is 0 (n_mask_pixels="
                    f"{sample_stats.get('n_mask_pixels', 0)}, "
                    f"otsu_threshold={otsu_threshold}). "
                    f"The H&E mask is empty after thresholding. Check "
                    f"n_samples / block_um, or verify that images[{image_key!r}] "
                    f"contains tissue signal."
                )

            if self.verbose:
                vx, vy, vz = self.dataset.shape
                print(f'downsample_vector: vf grid ({vx}, {vy}, {vz})')
                print(f'  otsu_threshold   : {otsu_threshold}')
                for k in ('n_mask_pixels', 'block_px', 'n_select',
                          'dropped_xy_without_transcript_support', 'kept_count'):
                    print(f'  {k:40s}: {sample_stats[k]}')

        else:
            if sdata_dir is None:
                raise ValueError('chunked=True requires sdata_dir.')

            sdata_dir = _Path(sdata_dir)
            zarr_paths = sorted(sdata_dir.glob('*.zarr'))
            if not zarr_paths:
                raise RuntimeError(f'No *.zarr stores found in {sdata_dir}.')

            if self.verbose:
                print(f'downsample_vector (chunked): {len(zarr_paths)} sdata stores found.')

            _OTSU_STRIDE = 4
            planes = plane_otsu = None
            if self.n_dims == 3:
                # One threshold per imaging PLANE, decided across the tiles the way the 2D
                # path decides one for the image: Otsu on each tile's crop of that plane
                # (the tile's own images[images]), median over tiles. Each tile is then
                # sampled with its plane's threshold.
                tile_planes_of = {}
                for zp in zarr_paths:
                    s = _sd.read_zarr(zp, selection=('images',))
                    tile_planes_of[zp] = (s, self._sdata_image_planes(s, images))
                planes = tile_planes_of[zarr_paths[0]][1]
                ref_z = np.array([pl['z'] for pl in planes])
                for zp, (_, tp) in tile_planes_of.items():
                    tz = np.array([pl['z'] for pl in tp])
                    if tz.shape != ref_z.shape or not np.allclose(tz, ref_z, atol=1e-6):
                        raise ValueError(f"{zp.name}: images[{images!r}] planes at z {tz.tolist()} "
                                         f"differ from {zarr_paths[0].name}'s {ref_z.tolist()}")
                zr = [self._tile_z_range(zp) for zp in zarr_paths]
                z_lo, z_hi = min(r[0] for r in zr), max(r[1] for r in zr)
                if self.verbose:
                    print(f"  3D: {len(planes)} planes of images[{images!r}] at z "
                          f"{[round(float(v), 3) for v in ref_z]}; transcripts' z "
                          f"{z_lo:.2f}..{z_hi:.2f}", flush=True)
                plane_otsu, otsu_vals = {}, {}
                for k, pl in enumerate(planes):
                    if not (z_lo - sampling_distance <= float(pl['z']) <= z_hi + sampling_distance):
                        continue                    # never sampled, so it needs no threshold
                    vals = []
                    for zp in zarr_paths:
                        win = self._image_to_rgb_uint8(tile_planes_of[zp][1][k]['read']())
                        _, v = self._otsu_thresholding(
                            [win[::_OTSU_STRIDE, ::_OTSU_STRIDE, :]], revert_image=revert_image,
                            intensity=intensity, mult=otsu_mult)
                        vals.append(v[0])
                        del win
                    plane_otsu[int(pl['index'])] = float(np.median(vals))
                    otsu_vals[int(pl['index'])] = vals
                    if self.verbose:
                        print(f"  plane {int(pl['index']):>2d} (z={float(pl['z']):.2f}): otsu per tile "
                              f"{[round(x, 1) for x in vals]} -> median "
                              f"{plane_otsu[int(pl['index'])]:.1f}", flush=True)
                del tile_planes_of
                otsu_threshold = None
            else:
                otsu_vals = []
                for zp in zarr_paths:
                    s = _sd.read_zarr(zp)
                    arr = self._he_array_from_sdata(s, image_key=image_key)
                    otsu_img = arr[::_OTSU_STRIDE, ::_OTSU_STRIDE, :]
                    _, single_vals = self._otsu_thresholding(
                        [otsu_img], revert_image=revert_image, intensity=intensity,
                        mult=otsu_mult)
                    otsu_vals.append(single_vals[0])
                    del s, arr, otsu_img
                otsu_threshold = float(np.median(otsu_vals))

                if self.verbose:
                    for zp, v in zip(zarr_paths, otsu_vals):
                        print(f'  {zp.stem}: otsu={v:.0f}')
                    print(f'  median otsu: {otsu_threshold}')

            import gc as _gc
            try:
                import torch as _torch
                _has_cuda = _torch.cuda.is_available()
            except ImportError:
                _torch, _has_cuda = None, False

            all_gx, all_gy, all_gz = [], [], []
            all_nvecs = []
            all_vfnorm = []
            all_chunk_genes = []
            all_offsets = []
            all_stats = []

            for zp in zarr_paths:
                if self.verbose:
                    print(f'\n  [{zp.stem}]')

                chunk_sdata = _sd.read_zarr(zp)
                self.dataset.shape = None
                if self.n_dims == 3:
                    # the tile's own crop of the z-stack, placed by its own transform
                    he_img = None
                    gx, gy, gz, nvecs, vfn, chunk_genes, coord_off, stats = \
                        self._select_vectors_3d(self._sdata_image_planes(chunk_sdata, images),
                                                chunk_sdata, images,
                                                plane_otsu=plane_otsu, otsu_mult=otsu_mult,
                                                **helper_kwargs)
                else:
                    he_img = self._he_array_axis_aligned(
                        chunk_sdata, image_key=image_key,
                        element_name=element_name, sampling_distance=sampling_distance,
                        revert_image=revert_image,
                    )
                    gx, gy, gz, nvecs, vfn, chunk_genes, coord_off, stats = \
                        self._select_vectors(he_img, chunk_sdata, otsu_threshold, **helper_kwargs)

                if stats.get('n_select', 1) == 0:
                    if self.verbose:
                        print(f'    skipped (blank tile, n_mask_pixels='
                              f'{stats.get("n_mask_pixels", 0)})')
                    del he_img, chunk_sdata, gx, gy, gz, nvecs, vfn
                    _gc.collect()
                    if _has_cuda:
                        _torch.cuda.empty_cache()
                    continue


                all_gx.append(gx)
                all_gy.append(gy)
                all_gz.append(gz)
                all_nvecs.append(nvecs)
                all_vfnorm.append(vfn)
                all_chunk_genes.append(chunk_genes)
                all_offsets.append(coord_off)
                all_stats.append(stats)

                if self.verbose:
                    print(f'    kept: {stats["kept_count"]:,}  offset: {coord_off}')

                del he_img, chunk_sdata
                _gc.collect()
                if _has_cuda:
                    _torch.cuda.empty_cache()

            if not all_offsets:
                raise RuntimeError(
                    f'All {len(zarr_paths)} chunks were blank after otsu '
                    f'thresholding (otsu_threshold={otsu_threshold}). No '
                    f'vectors were sampled. Check the H&E key '
                    f'({image_key!r}) and whether the image contains signal.'
                )

            global_x_min = min(off[0] for off in all_offsets)
            global_y_min = min(off[1] for off in all_offsets)
            global_z_min = min(off[2] for off in all_offsets)
            coordinate_offset = [global_x_min, global_y_min, global_z_min]

            gx_he_list, gy_he_list, gz_he_list = [], [], []
            for gx, gy, gz, off in zip(all_gx, all_gy, all_gz, all_offsets):
                dx = round((off[0] - global_x_min) / sampling_distance)
                dy = round((off[1] - global_y_min) / sampling_distance)
                dz = round((off[2] - global_z_min) / sampling_distance)
                gx_he_list.append(gx + dx)
                gy_he_list.append(gy + dy)
                gz_he_list.append(gz + dz)

            gx_he = np.concatenate(gx_he_list)
            gy_he = np.concatenate(gy_he_list)
            gz_he = np.concatenate(gz_he_list)

            gx_max_um = global_x_min + float(gx_he.max()) * sampling_distance
            gy_max_um = global_y_min + float(gy_he.max()) * sampling_distance
            gz_max_um = global_z_min + float(gz_he.max()) * sampling_distance
            self.dataset.coordinate_bounds = [
                gx_max_um - global_x_min,
                gy_max_um - global_y_min,
                gz_max_um - global_z_min,
            ]
            self.dataset.original_coordinate_bounds = [
                [global_x_min, gx_max_um],
                [global_y_min, gy_max_um],
                [global_z_min, gz_max_um],
            ]


            genes = sorted(set().union(*all_chunk_genes))
            gene_to_idx = {g: i for i, g in enumerate(genes)}
            n_union = len(genes)
            if self.verbose:
                per_chunk_sizes = [len(g) for g in all_chunk_genes]
                if min(per_chunk_sizes) != max(per_chunk_sizes) or max(per_chunk_sizes) != n_union:
                    print(f'  gene panel union: {n_union} '
                          f'(per-chunk min={min(per_chunk_sizes)}, max={max(per_chunk_sizes)})')

            aligned_nvecs = []
            for nvecs, chunk_genes in zip(all_nvecs, all_chunk_genes):
                if list(chunk_genes) == genes:
                    aligned_nvecs.append(nvecs)
                else:
                    aligned = np.zeros((nvecs.shape[0], n_union), dtype=nvecs.dtype)
                    col_idx = np.fromiter(
                        (gene_to_idx[g] for g in chunk_genes),
                        dtype=np.int64, count=len(chunk_genes),
                    )
                    aligned[:, col_idx] = nvecs
                    aligned_nvecs.append(aligned)
            normalized_vectors = np.concatenate(aligned_nvecs, axis=0)
            vf_norm_all = np.concatenate(all_vfnorm)

            sample_stats = {
                'n_chunks': len(zarr_paths),
                'otsu_threshold': otsu_threshold,
                'per_chunk': all_stats,
                'total_kept': int(len(gx_he)),
            }
            if self.n_dims == 3:
                sample_stats.update(plane_otsu=plane_otsu, otsu_per_tile=otsu_vals,
                                    n_planes=len(planes), plane_z=[pl['z'] for pl in planes])

            if self.verbose:
                print(f'\n  total kept (all chunks): {len(gx_he):,}')

            # Each tile went through _prepare_grid, which leaves the LAST tile's grid shape and
            # transcripts on this object. map_celltypes reuses both when they are set, so on the
            # same Analysis it would map a tile-sized lattice; clear them so it builds the grid
            # from the whole SpatialData it is given.
            self.dataset.shape = None
            self.locations = None

        vf_norm_all = np.asarray(vf_norm_all)
        self.dataset.downsampled_vectors = (gx_he, gy_he, gz_he)
        self.dataset.normalized_vectors = normalized_vectors
        self.dataset.downsampled_vf_norm = vf_norm_all
        self.dataset.coordinate_offset = coordinate_offset
        self.dataset.sampling_distance = sampling_distance
        self.dataset.genes = genes

        for key in ('downsampled_vectors', 'normalized_vectors',
                    'downsampled_vf_norm', 'genes'):
            if key in self.dataset.zarr_group:
                _remove_array(self.dataset.zarr_group, key)

        self.dataset.zarr_group['downsampled_vectors'] = np.stack([gx_he, gy_he, gz_he], axis=0)
        self.dataset.zarr_group['normalized_vectors'] = normalized_vectors
        self.dataset.zarr_group['downsampled_vf_norm'] = vf_norm_all
        self.dataset.zarr_group['genes'] = np.array(genes)
        self.dataset.zarr_group.attrs['n_dims'] = int(self.n_dims)
        self.dataset.zarr_group.attrs['otsu_mult'] = otsu_mult
        if self.n_dims == 3:
            self.dataset.zarr_group.attrs['image_planes'] = dict(
                images=str(images), n_planes=int(sample_stats.get('n_planes', 0)),
                z=[float(v) for v in sample_stats.get('plane_z', [])])

        for key in ('coordinate_offset', 'coordinate_bounds',
                    'original_coordinate_bounds'):
            value = getattr(self.dataset, key, None)
            if value is None:
                continue
            if key in self.dataset.zarr_group:
                _remove_array(self.dataset.zarr_group, key)
            self.dataset.zarr_group[key] = np.array(value)

        return sample_stats

    @staticmethod
    def read_marker_genes(source, *, label_column='label',
                          marker_column='key_markers', sep=';'):
        """Marker gene sets per celltype, as ``{label: [genes]}``.

        ``source`` is one of

        * a ``{label: [genes]}`` dict — returned as is;
        * ``celltype_markers.yaml`` (CeLLMan §10.3) — the agent-written panel, a TREE
          of the emitted labels and their ancestors, each node carrying only its OWN
          ``defining`` / ``required`` / ``supporting`` lists, already restricted to
          genes measured in the dataset;
        * ``expert_judgment_annotation.csv`` — falls back to the ``key_markers``
          column, a 3–5 gene *summary* of each decision. Usable, but thin: prefer the
          YAML when it exists.

        **The YAML tree is accumulated root-to-node.** The Cell-Ontology is normalized
        delta-only — a child never repeats a marker an ancestor already carries — so a
        celltype's real marker set is the union along the path from the root down to
        its node. ``glomerular endothelial cell`` therefore comes back as its own
        ``Ehd3, Kdr, Icam2, …`` *plus* the ``endothelial cell`` markers above it. Every
        node is returned, ancestors included, so an intermediate label a cluster was
        floored to resolves too.

        Clusters sharing a label are unioned either way.
        """
        if isinstance(source, dict):
            return {k: sorted(set(v)) for k, v in source.items()}
        if str(source).lower().endswith(('.yaml', '.yml')):
            import yaml
            with open(source) as f:
                raw = yaml.safe_load(f) or {}
            out = {}

            def _walk(node, inherited):
                for lab, entry in (node or {}).items():
                    if not isinstance(entry, dict):    # {label: [genes]} shorthand
                        own, kids = set(entry or []), {}
                    else:
                        own = set()
                        for t in ('defining', 'required', 'supporting'):
                            own.update(entry.get(t) or [])
                        kids = entry.get('children') or {}
                    genes = inherited | own
                    if genes:
                        out[str(lab)] = sorted(genes)
                    _walk(kids, genes)

            _walk(raw, set())
            return out
        df = pd.read_csv(source)
        for c in (label_column, marker_column):
            if c not in df.columns:
                raise ValueError(f"'{c}' not in {source} (columns: {list(df.columns)})")
        df = df[df[label_column].notna() & df[marker_column].notna()]
        out = {}
        for lab, grp in df.groupby(label_column):
            genes = set()
            for v in grp[marker_column]:
                genes.update(g.strip() for g in str(v).split(sep) if g.strip())
            if genes:
                out[str(lab)] = sorted(genes)
        return out

    @staticmethod
    def read_marker_tree(source, *, max_degree=2):
        """``(parents, own)`` — the hierarchy and the DELTA markers, from the same YAML.

        :meth:`read_marker_genes` accumulates root-to-node because a celltype's real
        marker set includes its ancestors'. Anything reasoning about a parent/child pair
        needs the opposite: each node's OWN ``defining`` / ``required`` genes only. A
        child expresses its
        parent's markers by inheritance — cortical TAL anchors carry more ``Umod``
        than plain TAL ones — so an inherited gene is worthless as evidence for either
        side of a parent/child pair, and accumulating would hand every child its
        parent's whole set.

        ``supporting`` is left out for the same reason it is not a KG defining marker:
        it is corroboration, not identity.

        A gene claimed by more than ``max_degree`` celltypes is dropped from all of
        them — "this celltype and one close relative" still identifies, a gene shared
        five ways does not.

        Non-YAML sources carry no tree: ``parents`` comes back empty, so any caller that
        wanted the hierarchy sees every celltype on its own rather than a guessed one.

        :returns: ``({label: parent label or None}, {label: [genes]})``
        """
        if not str(source).lower().endswith(('.yaml', '.yml')):
            return {}, ({k: sorted(set(v)) for k, v in source.items()}
                        if isinstance(source, dict) else {})
        import yaml
        from collections import Counter
        with open(source) as f:
            raw = yaml.safe_load(f) or {}
        parents, own = {}, {}

        def _walk(node, parent):
            for lab, entry in (node or {}).items():
                lab = str(lab)
                parents[lab] = parent
                if isinstance(entry, dict):
                    g = set()
                    for t in ('defining', 'required'):
                        g.update(entry.get(t) or [])
                    own[lab] = g
                    _walk(entry.get('children') or {}, lab)
                else:
                    own[lab] = set(entry or [])

        _walk(raw, None)

        # delta-ise: a gene an ancestor already claims is inherited, so it cannot tell
        # the child apart. The KG is delta-only in principle, but a node may still
        # repeat its parent's gene (mouse kidney: cortical TAL lists Slc12a1, its
        # parent's defining marker; DCT2 lists Slc12a3), and leaving it in would let
        # the child clear the gate on evidence that says nothing about which of the
        # two it is.
        def _chain(lab):
            out, cur = [], parents.get(lab)
            while cur is not None:
                out.append(cur)
                cur = parents.get(cur)
            return out

        own = {lab: set(g) - {x for a in _chain(lab) for x in own.get(a, ())}
               for lab, g in own.items()}
        deg = Counter(g for v in own.values() for g in v)
        own = {k: sorted(g for g in v if deg[g] <= int(max_degree))
               for k, v in own.items()}
        return parents, own

    def _make_anchor_prior(self, lda_anchor, n_classes, sampling_distance, mk):
        """The anchor prior, as a precomputed convolution when the anchors allow it.

        ``downsample_vectors`` are voxel indices, so anchors and voxels share one
        lattice and the prior is exactly a convolution — computed once instead of a
        48-nearest-neighbour query per voxel (36 s of every 54 s chunk on the
        mouse-kidney store). If the anchors are ever off-lattice, or ``gridded_prior``
        is set False, the KD-tree version still stands.
        """
        from ._lda import AnchorPrior, GriddedAnchorPrior
        xy, lab = lda_anchor
        xy = np.atleast_2d(np.asarray(xy, np.float64))
        d_max = float(mk.pop('d_max', 12.0))
        nu = float(mk.pop('nu', 2.0))
        k_max = int(mk.pop('k_max', 48))
        want = bool(mk.pop('gridded_prior', True))
        off = (np.asarray(self.dataset.zarr_group['coordinate_offset'][:], np.float64)
               if 'coordinate_offset' in self.dataset.zarr_group else np.zeros(3))
        # Anchors that span depth make the prior 3D, and the convolution field would then
        # be (nx, ny, nz, K) — tens of GB on a section — so those go to the KD-tree.
        spans_z = bool(xy.shape[1] > 2 and np.ptp(xy[:, 2]) > 0)
        rel = (xy[:, :2] - off[:2][None, :]) / float(sampling_distance)
        on_lattice = bool(np.allclose(rel, np.rint(rel), atol=1e-6))
        if spans_z:
            self._m('map_celltypes: anchors span z, so the prior is 3D: a nearest-k_max '
                    'lookup, on the GPU when the anchors sit on the voxel lattice and a '
                    'CUDA device is used, else the KD-tree (the gridded prior is a 2D '
                    'convolution)')
            return AnchorPrior(xy, lab, n_classes, d_max=d_max, nu=nu, k_max=k_max,
                               sampling_distance=float(sampling_distance))
        if want and on_lattice:
            self._m('map_celltypes: anchor prior precomputed on the voxel lattice '
                    '(convolution; k_max honoured exactly at the nodes that exceed it)')
            return GriddedAnchorPrior(
                xy, lab, n_classes, d_max=d_max, nu=nu, k_max=k_max,
                sampling_distance=float(sampling_distance), origin=off[:2],
                exact_k_max=bool(mk.pop('exact_k_max', True)),
                verbose=self.verbose)
        if want and not on_lattice:
            self._m('map_celltypes: anchors are not on the voxel lattice; '
                    'falling back to the KD-tree anchor prior')
        return AnchorPrior(xy, lab, n_classes, d_max=d_max, nu=nu, k_max=k_max)

    def _unknown_class_ids(self):
        """Ids of the annotation's ``unknown*`` / ``background`` classes, if any.

        ``map_celltypes`` always keeps these classes on the map so a voxel that
        matches no real celltype still gets a stable id — the map then shows *where*
        the annotation abstained. Hiding them is a display choice, so it belongs to
        :meth:`filter_celltypemaps` (``show_unknown=False``), not to the mapper:
        deciding a voxel's label and deciding whether that voxel survives are
        different jobs, and folding the second into the first meant re-running hours
        of mapping to change a filter setting.
        """
        import re
        names, _ = self._celltype_names_colors()
        return [i for i, n in enumerate(names)
                if re.fullmatch(r'unknown\d*|background', str(n).strip(), re.I)]

    def _marker_gene_exclusions(self, marker_genes, exclude_gene_indices=None):
        """Turn marker sets into an ``exclude_gene_indices`` list.

        Everything outside the UNION of the marker sets is dropped, so the mapping
        runs on a panel of only the genes the annotation actually leaned on. The
        centroids keep one shared gene axis — required by both the correlation and
        the GaussianNB backends — but that axis no longer carries the hundreds of
        genes that are flat across celltypes and only add noise to the metric.
        """
        mk = self.read_marker_genes(marker_genes)
        genes = [g.decode() if isinstance(g, bytes) else str(g)
                 for g in self.dataset.zarr_group['genes'][:]]
        union = set().union(*mk.values()) if mk else set()
        missing = sorted(union - set(genes))
        keep = [i for i, g in enumerate(genes) if g in union]
        if not keep:
            raise ValueError("no marker gene is present in the panel — check that the "
                             "marker names match the store's gene names")
        drop = sorted(set(range(len(genes))) - set(keep)
                      | set(exclude_gene_indices or []))
        self._m(f'map_celltypes: marker panel {len(keep)}/{len(genes)} genes '
                f'from {len(mk)} celltypes'
                + (f'; {len(missing)} markers not in panel: {missing[:5]}' if missing else ''))
        return drop

    def _marker_index_sets(self, marker_genes, gene_names_used, warn_few=True):
        """Per-celltype gene positions for the marker-restricted correlation.

        ``gene_names_used`` is the gene axis the mapping actually runs on — after
        the union restriction :meth:`_marker_gene_exclusions` applied — so the
        returned positions index a data chunk directly. See
        :mod:`ssaft._marker_corr` for why each celltype gets its own subset.
        """
        from ._marker_corr import marker_index_sets
        names, _ = self._celltype_names_colors()
        return marker_index_sets(names, self.read_marker_genes(marker_genes),
                                 gene_names_used, warn_few=warn_few,
                                 verbose=self.verbose, log=self._m)

    def _own_marker_sets(self, marker_genes, gene_names_used):
        """Per-celltype OWN-marker positions for the LDA dictionary.

        Same resolution as :meth:`_marker_index_sets` but WITHOUT the
        ``MIN_MARKER_GENES`` fallback. That fallback exists so a 2-marker Pearson
        cannot saturate at +/-1; here it would do real damage — a celltype whose
        only marker is a single gene (mouse-kidney ``mural cell``: ``Notch3``) would
        come back as "no markers", and the dictionary would then treat that one gene
        as somebody else's and replace it with the background rate, erasing the only
        evidence the celltype has.
        """
        from ._marker_corr import marker_index_sets
        names, _ = self._celltype_names_colors()
        return marker_index_sets(names, self.read_marker_genes(marker_genes),
                                 gene_names_used, min_genes=0, verbose=False)

    def map_celltypes(self, sdata=None, centroids=None, element_name=None,
                      method='lda', exclude_gene_indices=None, marker_genes_path=None,
                      show_unknown=None,
                      bandwidth=2.5, sampling_distance=1.0, prune_coefficient=4.3,
                      kernel='gaussian',
                      size_after_normalization=10, log_transform=True, overwrite=True,
                      gpu_device='cuda', force_cpu=False,
                      query_subchunk_size=None, gpu_target_bytes=int(1e9),
                      map_kwargs=None, save_vf=False,
                      h5ad=None, column='cellman_annotation', leiden_column='leiden',
                      cmap=None, table_key='ssaft_downsampled_vectors'):
        """Single entry point for celltype mapping.

        Two regimes, auto-detected from the store:

        * **count** (blur ``vf_normalized`` present) — the field is read from
          zarr and mapped by ``method``. No KDE.
        * **transcript** (no precomputed field) — KDE is computed chunk-by-chunk
          and mapped on the fly, dispatching to
          :meth:`map_celltypes_gpu` (CUDA available and not ``force_cpu``) or
          :meth:`map_celltypes_cpu`.

        Both regimes support ``method='corr'`` (centroid Pearson correlation) and
        ``method='gnb'`` (per-voxel Gaussian naive Bayes fitted on the labeled
        ``normalized_vectors``). ``'gnb'`` is generative — every class keeps its own
        region — so it preserves rare cell types the discriminative correlation
        argmax can drop.

        Parameters
        ----------
        method : {'corr', 'gnb'}, default 'corr'
            Mapping algorithm (both regimes). ``'corr'`` correlates each voxel's
            vector with the centroids and takes the argmax. ``'gnb'`` fits a
            :class:`~sklearn.naive_bayes.GaussianNB` (cuML on GPU, scikit-learn on
            CPU) on the labeled ``normalized_vectors`` and predicts each voxel. Both
            write the label raster to ``celltype_maps``, but the per-voxel score goes
            to the volume matching its metric: ``'corr'`` -> ``max_correlations``
            (filtered downstream by ``min_r``), ``'gnb'`` -> ``max_probabilities``
            (filtered by ``min_p``). Both volumes can coexist — writing one leaves
            the other method's score untouched, and ``mapping_method`` in the store
            attrs records which one the current ``celltype_maps`` came from so
            :meth:`filter_celltypemaps` picks the matching volume. Extra ``'gnb'``
            options via ``map_kwargs`` (e.g. ``{'row_normalize': False,
            'var_smoothing': 1e-9}``). On the transcript/imaging regime ``'gnb'`` uses
            the CPU KDE producer with the (GPU) classifier. With ``marker_genes_path``
            the GNB is instead fitted on the annotated h5ad with ``obs[column]`` as the
            target, on the marker genes only, using scikit-learn.
        map_kwargs : dict, optional
            Extra kwargs forwarded to the count mapper
            (:func:`ssaft._count.map_corr` /
            :func:`ssaft._gnb.map_gnb`).
        gpu_device : str, default ``'cuda'``
            CUDA device used by the GPU path.
        force_cpu : bool, default False
            If True, run on CPU regardless of GPU availability (transcript KDE and
            the ``'gnb'`` classifier both honor it).
        save_vf : bool, default False
            **Transcript regime only.** By default the imaging-based path
            (Xenium / MERFISH) computes the KDE vector field chunk-by-chunk,
            maps it on the fly, and DISCARDS the field (only ``celltype_maps`` /
            ``max_correlations`` / ``vf_norm`` are persisted). When True, the
            normalized field is also written to ``vf_normalized``
            ``(n_genes_use, vx, vy, vz)`` in zarr (gene axis first, same
            convention as the count backend), plus ``vf_normalized_genes`` naming
            the gene axis. Ignored in the count regime — array-based data
            (Visium HD / Stereo-seq) already stores ``vf_normalized`` at
            ``compute_vf`` time. Note: this holds one extra CPU buffer of
            ``n_genes_use × chunk_x × vy × vz`` floats per chunk and adds the
            field's on-disk size, so enable it only when you need the field.
        query_subchunk_size, gpu_target_bytes
            Forwarded to :meth:`map_celltypes_gpu` (ignored on CPU).
        Other parameters are shared by the transcript paths; see
        :meth:`map_celltypes_cpu`.

        :param sdata: Transcript SpatialData to map; required in the transcript
            regime, unused in the count regime.
        :type sdata: spatialdata.SpatialData or None
        :param centroids: Per-cluster centroid matrix; falls back to the store's
            ``centroids`` (or the h5ad-built ones) when None.
        :type centroids: numpy.ndarray or None
        :param h5ad: Annotated h5ad (row-aligned with ``normalized_vectors``). When
            given, ``cluster_labels`` + ``centroids`` are (re)built from it before
            mapping: vectors are grouped by the label string in ``obs[column]`` (one
            class per label; ``Unknown`` split by ``obs[leiden_column]`` into
            ``unknown1..N``; Background dropped). Required the first time — when the
            store has no centroids/cluster_labels and no ``centroids`` array is passed.
        :type h5ad: str or None
        :param column: obs column with the final celltype label (h5ad build).
        :type column: str
        :param leiden_column: obs column with the original cluster id, used to split
            ``Unknown`` when building from an h5ad.
        :type leiden_column: str
        :param cmap: palette fixing the per-celltype colours at build time; the
            same palette is reused by the labels rendering and any legend.
            ``None`` (default) = qualitative palette sized to the class count
            (tab20 / scanpy 28 / scanpy 102), which stays legible for tens of
            celltypes; a colormap name samples it continuously instead.
        :type cmap: str or matplotlib.colors.Colormap or None
        :param table_key: key of the annotated downsampled table written into
            ``sdata.tables``.
        :type table_key: str
        :param element_name: Name of the points element to parse from ``sdata``.
        :type element_name: str or None
        :param exclude_gene_indices: Gene indices to drop before correlation.
        :type exclude_gene_indices: list[int] or None
        :param show_unknown: **Moved to** :meth:`filter_celltypemaps`. The mapper now
            always keeps the annotation's ``unknown1..N`` classes on the map, so a
            voxel that matches no real celltype gets a stable id and the map shows
            *where* the annotation abstained; hiding them is a filter setting you can
            change without re-mapping. Passing it here logs a notice and is otherwise
            ignored.
        :type show_unknown: bool or None
            ``method='lda'`` reads the anchors for its dictionary instead: which genes a
            celltype owns comes from the anchors themselves (see
            :func:`ssaft._lda.build_factors`), so a celltype the YAML gives no marker —
            8 of 18 on the mouse-kidney panel — still gets a discriminating row. The
            sets resolved here are then only the axis ``celltype_marker_score`` is
            measured on, for ``filter_celltypemaps(hsm_f=...)``.

            The scorer's own knobs go through ``map_kwargs``: ``cand_nats`` (default
            ``1.0``) how far behind the best log-likelihood a class can be and still be
            a candidate, ``prior_weight`` (``0.5``) the weight on the log anchor prior,
            and ``prior_floor`` (``0.02``) the floor that keeps "no anchor in range"
            from being a veto. See :class:`ssaft._lda.LDAScorer`.
        :param marker_genes_path: Restrict the mapping to marker genes. ``None`` (default)
            keeps the whole panel and behaves exactly as before. Otherwise a
            ``{celltype: [genes]}`` dict, the CeLLMan ``celltype_markers.yaml`` (a tree
            of the emitted labels and their ancestors, each node carrying its own
            defining / required / supporting genes — preferred), or
            ``expert_judgment_annotation.csv`` (its thinner ``key_markers`` column).
            The YAML tree is accumulated root-to-node, so a celltype's marker set
            includes every ancestor's (see :meth:`read_marker_genes`).

            Three things then change, in both the count and the transcript regime:

            * the gene axis is cut to the UNION of the marker sets, so centroids and
              the GaussianNB are built from the genes the annotation leaned on instead
              of the whole panel;
            * ``method='corr'`` scores each celltype ONLY on its own markers instead of
              the shared axis — the same Pearson, just a different gene count per
              celltype (see :mod:`ssaft._marker_corr`, including why a celltype with
              fewer than 3 on-panel markers falls back to the full axis);
            * ``method='gnb'`` refits on the annotated h5ad itself with ``obs[column]``
              as ground truth (scikit-learn ``GaussianNB``, see
              :func:`ssaft._gnb.fit_gnb_h5ad`) when the h5ad is known — either passed
              as ``h5ad=`` or remembered in the store from the centroid build.

            Combined with ``exclude_gene_indices`` if both are given.
        :type marker_genes_path: str or pathlib.Path or dict or None
        :param bandwidth: KDE Gaussian bandwidth in µm (transcript regime).
        :type bandwidth: float
        :param sampling_distance: Vector-field grid spacing in µm.
        :type sampling_distance: float
        :param prune_coefficient: Kernel truncation radius in bandwidth units. The
            gaussian's, and the exponential's at the same relative height; the
            compact kernels stop at their own support.
        :type prune_coefficient: float
        :param kernel: KDE kernel, one of ``ssaft._count.KDE_KERNELS``:
            ``'gaussian'`` (default), ``'tophat'``, ``'epanechnikov'``,
            ``'exponential'``, ``'linear'`` or ``'cosine'``. ``bandwidth`` keeps the
            SAME meaning for all of them: each kernel's own scale is derived from it
            by variance matching (:func:`ssaft._count.kernel_scale`; e.g. the
            Epanechnikov support ``H = bandwidth * sqrt(ndim + 4)``, 6.61 µm at the
            default 2.5 in 3D), so they have the same width. The compact kernels
            reach zero at their support instead of being truncated at
            ``4.3 * bandwidth`` = 10.75 µm, so they touch ~3x fewer voxels per
            transcript and run 2-3x faster; the smooth ones (Epanechnikov, linear,
            cosine) give near-identical fields to the gaussian (per-voxel cosine
            >= 0.99 on Xenium kidney and Visium HD). Transcript (KDE) regime only;
            ignored by the count regime.
        :type kernel: str
        :param size_after_normalization: Target L1 scale of each voxel vector.
        :type size_after_normalization: float
        :param log_transform: If True, apply log1p to the normalized field.
        :type log_transform: bool
        :param overwrite: If True, remove and recreate existing zarr outputs.
        :type overwrite: bool

        Returns
        -------
        spatialdata.SpatialData
            The map itself is written into the store (``celltype_maps`` + the score
            volume for the chosen ``method`` — ``max_correlations`` for ``'corr'``,
            ``max_probabilities`` for ``'gnb'`` — in zarr, and mirrored on
            ``ds``). The returned SpatialData additionally carries
            ``tables[table_key]`` — the downsampled vectors annotated with the
            REBUILT celltype labels + the fixed colour palette. Feed it to
            :meth:`filter_celltypemaps`, which adds the raster elements.
        """
        zg = self.dataset.zarr_group
        sdata = sdata if sdata is not None else getattr(
            self.dataset, '_original_input_data', None)

        # The inputs (cluster_labels + centroids) are built here from the annotated
        # h5ad. Rebuild when an h5ad is passed; otherwise build only if the store is
        # missing them and no centroids array was given (else fall back to the store).
        need_build = ('centroids' not in zg) or ('cluster_labels' not in zg)
        if h5ad is not None or (need_build and centroids is None):
            if h5ad is None:
                raise ValueError(
                    "store has no centroids/cluster_labels. Pass h5ad=<annotated .h5ad> "
                    "so map_celltypes can group vectors by obs[column] and build them, "
                    "or pass centroids= directly.")
            from ._mapping_inputs import build_centroids_from_h5ad
            build_centroids_from_h5ad(
                zg, h5ad, column=column, leiden_column=leiden_column, cmap=cmap,
                store_dir=self.dataset._zarr_store_path(self.dataset.zarr_store))
            self.dataset.cluster_labels = zg['cluster_labels'][:]
            self.dataset.centroids = zg['centroids'][:]
            # remembered so a later gnb run can refit on the same annotation without
            # the caller having to pass h5ad= again
            zg.attrs['annotation_h5ad'] = os.path.abspath(h5ad)

        if show_unknown is not None:
            self._m("map_celltypes: show_unknown moved to filter_celltypemaps and is "
                    "ignored here — the map keeps the unknown classes; pass "
                    f"filter_celltypemaps(show_unknown={show_unknown}) to hide them.")

        # ``marker_genes_path`` always supplies the per-celltype marker sets. It ALSO
        # narrows the gene axis, but only for the metrics that are axis-sensitive: a
        # Pearson over the full panel is dominated by the hundreds of genes that are
        # flat across celltypes, so corr/gnb want the marker union and nothing else.
        # The lda likelihood is not — an uninformative gene enters every celltype's
        # score through the same multinomial term and cancels — and it needs the full
        # panel anyway, because its dictionary decides what a NON-marker gene says
        # about a celltype. Narrowing there would throw that evidence away.
        if marker_genes_path is not None and method != 'lda':
            exclude_gene_indices = self._marker_gene_exclusions(
                marker_genes_path, exclude_gene_indices)

        if ('vf_normalized' in zg) and (zg.attrs.get('backend') == 'blur'):
            mk = dict(map_kwargs or {})
            mk.setdefault('overwrite', overwrite)
            # The count backend maps straight off ``vf_normalized``, whose gene axis is
            # the store's full ``genes``; the transcript paths do their own subsetting,
            # so the marker panel has to be resolved against that axis here.
            keep_idx = marker_idx = None
            if marker_genes_path is not None:
                _genes = [g.decode() if isinstance(g, bytes) else str(g)
                          for g in zg['genes'][:]]
                keep_idx = sorted(set(range(len(_genes))) - set(exclude_gene_indices or []))
                marker_idx = (self._marker_index_sets(
                    marker_genes_path, [_genes[i] for i in keep_idx])
                    if method != 'lda' else None)
            if method == 'corr':
                mk.setdefault('device', gpu_device)
                mk.setdefault('gene_indices', keep_idx)
                mk.setdefault('marker_index_sets', marker_idx)
                from ._count import map_corr
                map_corr(self, centroids=centroids, **mk)
            elif method == 'gnb':
                mk.pop('device', None)
                mk.setdefault('gene_indices', keep_idx)
                mk.setdefault('column', column)
                if keep_idx is not None:
                    mk.setdefault('gene_names', [_genes[i] for i in keep_idx])
                    mk.setdefault('h5ad', h5ad or zg.attrs.get('annotation_h5ad'))
                from ._gnb import map_gnb
                map_gnb(self, centroids=centroids, gpu_device=gpu_device,
                        force_cpu=force_cpu, **mk)
            elif method == 'lda':
                # the count-backend LDA scorer runs on the GPU too: the likelihood is
                # a (voxels x genes) @ (genes x celltypes) GEMM
                mk.setdefault('device', None if force_cpu else gpu_device)
                mk.setdefault('gene_indices', keep_idx)
                if marker_genes_path is not None:
                    _g = [g.decode() if isinstance(g, bytes) else str(g)
                          for g in zg['genes'][:]]
                    _ax = [_g[i] for i in keep_idx] if keep_idx is not None else _g
                    # the dictionary owns its genes from the anchors now; these sets are
                    # only the axis celltype_marker_score is taken on
                    mk.setdefault('marker_sets',
                                  self._own_marker_sets(marker_genes_path, _ax))
                from ._lda import map_lda
                map_lda(self, centroids=centroids, **mk)
            else:
                raise ValueError(f"unknown mapping method {method!r} "
                                 "(count: 'corr'|'gnb'|'lda')")
            return self._finish_map(sdata, table_key)

        if method not in ('corr', 'gnb', 'lda'):
            raise ValueError(f"unknown mapping method {method!r} "
                             "(transcript: 'corr'|'gnb'|'lda')")
        if sdata is None:
            raise ValueError("transcript map_celltypes requires sdata.")
        shared = dict(
            centroids=centroids, element_name=element_name,
            exclude_gene_indices=exclude_gene_indices,
            marker_genes_path=marker_genes_path, h5ad=h5ad, column=column,
            map_kwargs=map_kwargs, bandwidth=bandwidth,
            sampling_distance=sampling_distance, prune_coefficient=prune_coefficient,
            kernel=kernel,
            size_after_normalization=size_after_normalization,
            log_transform=log_transform, overwrite=overwrite, save_vf=save_vf,
            method=method,
        )
        use_gpu = False
        if not force_cpu:
            try:
                from ._kde_gpu import gpu_available
                use_gpu = gpu_available(gpu_device)
            except Exception:
                use_gpu = False
        if use_gpu:
            self._m(f'map_celltypes: GPU path ({gpu_device})')
            self._map_celltypes_gpu(
                sdata, gpu_device=gpu_device,
                query_subchunk_size=query_subchunk_size,
                gpu_target_bytes=gpu_target_bytes, **shared,
            )
        else:
            self._m('map_celltypes: CPU path')
            self._map_celltypes_cpu(sdata, gpu_device=gpu_device,
                                    force_cpu=force_cpu, **shared)
        return self._finish_map(sdata, table_key)

    def _map_celltypes_cpu(self, sdata, centroids=None, element_name=None,
                         exclude_gene_indices=None, marker_genes_path=None,
                         bandwidth=2.5, sampling_distance=1.0, prune_coefficient=4.3,
                         kernel='gaussian',
                         size_after_normalization=10, log_transform=True, overwrite=True,
                         save_vf=False, method='corr', gpu_device='cuda', force_cpu=False,
                         h5ad=None, column='cellman_annotation', map_kwargs=None):
        """
        Stream-compute celltype_maps for H&E-sampled data.

        Reuses ``self.dataset.shape`` set by ``downsample_vector`` (or
        ``_prepare_grid``) so that SpatialData is NOT re-parsed for grid
        dimensions.  KDE is computed chunk-by-chunk directly from
        ``self.locations`` (already stored by ``_prepare_grid``).

        The KDE is the compiled ``ssaft.utils.accumulate_gene_vector``, which takes
        every kernel in ``ssaft._count.KDE_KERNELS``.

        Parameters
        ----------
        sdata : spatialdata.SpatialData
            Only used to populate ``self.locations`` if not already set.
        centroids : ndarray (n_clusters, n_genes), optional
            Falls back to ``self.dataset.centroids``.

        Returns
        -------
        celltype_maps, max_correlations : zarr arrays
        """
        from tqdm.auto import tqdm
        from ._count import resolve_kernel, kernel_code
        kernel = resolve_kernel(kernel)

        self.dataset.cluster_labels          = self.dataset.zarr_group['cluster_labels'][:]
        self.dataset.downsampled_vectors              = self.dataset.zarr_group['downsampled_vectors'][:]
        self.dataset.normalized_vectors      = self.dataset.zarr_group['normalized_vectors'][:]
        self.dataset.centroids               = self.dataset.zarr_group['centroids'][:]

        if centroids is None:
            if getattr(self.dataset, 'centroids', None) is None:
                raise ValueError('centroids not found. Pass h5ad=<annotated .h5ad> to map_celltypes to build them.')
            centroids = np.asarray(self.dataset.centroids, dtype=np.float32)
        else:
            centroids = np.asarray(centroids, dtype=np.float32)

        if getattr(self.dataset, 'shape', None) is None:
            self._prepare_grid(
                sdata,
                element_name=element_name,
                bandwidth=bandwidth,
                sampling_distance=sampling_distance,
                prune_coefficient=prune_coefficient,
                size_after_normalization=size_after_normalization,
            )
        else:
            if not hasattr(self, 'locations') or self.locations is None:
                self._prepare_grid(
                    sdata,
                    element_name=element_name,
                    bandwidth=bandwidth,
                    sampling_distance=sampling_distance,
                    prune_coefficient=prune_coefficient,
                    size_after_normalization=size_after_normalization,
                )

        vx, vy, vz = (int(self.dataset.shape[0]),
                      int(self.dataset.shape[1]),
                      int(self.dataset.shape[2]))

        # the kernel's own scale, as the GPU path records it (norm_threshold, V_eff)
        from ._count import resolve_kernel, kernel_scale, kernel_peak
        self.dataset.ndim = 2 if vz == 1 else 3
        self.dataset.bandwidth = kernel_scale(bandwidth, self.dataset.ndim, kernel)
        self.dataset.kde_kernel = resolve_kernel(kernel)
        self.dataset.expression_threshold = kernel_peak(
            self.dataset.bandwidth, self.dataset.ndim, kernel)
        self.dataset.norm_threshold = self.dataset.expression_threshold * 2

        grouped = self.locations.groupby('gene', sort=True)
        grouped_map = {gene: loc for gene, loc in grouped}

        if getattr(self.dataset, 'genes', None) is None:
            genes = sorted(grouped_map.keys())
            self.dataset.genes = genes
        else:
            genes = list(self.dataset.genes)

        if centroids.shape[1] != len(genes):
            raise ValueError(
                f'centroids gene dim ({centroids.shape[1]}) != gene count ({len(genes)})'
            )

        shared_gene_indices = [i for i, g in enumerate(genes) if g in grouped_map]
        shared_genes = [genes[i] for i in shared_gene_indices]
        missing_genes = [g for g in genes if g not in grouped_map]
        if len(shared_genes) == 0:
            raise ValueError('No common genes between input data and centroids.')
        if missing_genes and self.verbose:
            print(f'map_celltypes_cpu: dropping {len(missing_genes)} missing genes '
                  f'(e.g. {missing_genes[:5]})')

        grouped_list = [(g, grouped_map[g]) for g in shared_genes]

        centroids_shared = centroids[:, shared_gene_indices]

        if exclude_gene_indices is not None:
            idx_to_local = {gi: li for li, gi in enumerate(shared_gene_indices)}
            exclude_shared = sorted(
                idx_to_local[i] for i in exclude_gene_indices if i in idx_to_local
            )
            centroids_use = np.delete(centroids_shared, exclude_shared, axis=1)
        else:
            exclude_shared = None
            centroids_use = centroids_shared

        # Each celltype is correlated only on its own markers (+ its ancestors');
        # None = the whole axis, i.e. exactly the previous behaviour.
        genes_use = ([g for i, g in enumerate(shared_genes) if i not in set(exclude_shared)]
                     if exclude_shared is not None else list(shared_genes))
        # lda never reads this: its dictionary and its marker score both use the
        # fallback-free OWN sets below. The MIN_MARKER_GENES fallback here is a guard
        # for Pearson (n=2 is always +-1), so it stays on the corr path alone.
        marker_idx = (self._marker_index_sets(marker_genes_path, genes_use)
                      if (marker_genes_path is not None and method != 'lda') else None)
        _ms = None

        # method='gnb': fit a GaussianNB on the labeled normalized_vectors, restricted to the
        # SAME gene axis the KDE chunks (data_use) carry, so posteriors align voxel-to-voxel.
        gnb_clf = gnb_classes = gnb_backend = None
        if method == 'gnb':
            from ._gnb import fit_gnb, predict_gnb  # noqa: F401 (predict_gnb used in the loop)
            if exclude_shared is not None:
                _keep = np.delete(np.arange(len(shared_gene_indices)), exclude_shared)
                _gnb_gi = np.asarray(shared_gene_indices)[_keep]
            else:
                _gnb_gi = np.asarray(shared_gene_indices)
            _h5ad = h5ad or self.dataset.zarr_group.attrs.get('annotation_h5ad')
            if marker_genes_path is not None and _h5ad:
                from ._gnb import fit_gnb_h5ad
                gnb_clf, gnb_classes, gnb_backend = fit_gnb_h5ad(
                    _h5ad, gene_names=genes_use, column=column,
                    row_normalize=True, verbose=self.verbose)
            else:
                gnb_clf, gnb_classes, gnb_backend = fit_gnb(
                    self.dataset.zarr_group, gpu_device=gpu_device, force_cpu=force_cpu,
                    gene_indices=_gnb_gi, row_normalize=True)

        # method='lda': FICTURE-style multinomial likelihood x anchor prior. Factors and
        # anchors both come from the annotation, so nothing is fitted here — the scorer is
        # built once and applied per chunk (see ssaft._lda).
        lda_scorer = None
        if method == 'lda':
            from ._lda import build_factors, anchor_positions, AnchorPrior, LDAScorer
            _mk = dict(map_kwargs or {})
            _veff = _mk.pop('veff', None)
            if _veff is None:      # KDE density -> molecule count; see ssaft._lda
                _veff = 1.0 / float(self.dataset.expression_threshold)
            _gi = np.asarray([shared_gene_indices[i] for i, g in enumerate(shared_genes)
                              if g in set(genes_use)])
            _ms = (self._own_marker_sets(marker_genes_path, list(genes_use))
                   if marker_genes_path is not None else None)
            _beta, _ = build_factors(
                self.dataset.zarr_group, gene_indices=_gi, veff=_veff,
                size_after_normalization=size_after_normalization,
                log_transformed=log_transform, sampling_distance=sampling_distance,
                verbose=self.verbose)
            _prior = None
            if bool(_mk.pop('use_anchor_prior', True)):
                lda_anchor = anchor_positions(
                    self.dataset.zarr_group, sampling_distance=sampling_distance,
                    coordinate_offset=_mk.pop('anchor_offset', None))
                _prior = self._make_anchor_prior(
                    lda_anchor, _beta.shape[0], sampling_distance, _mk)
            _unk = np.zeros(_beta.shape[0], dtype=bool)
            _unk[[i for i in self._unknown_class_ids() if i < _beta.shape[0]]] = True
            lda_scorer = LDAScorer(
                _beta, prior=_prior, unknown=_unk,
                cand_nats=float(_mk.pop('cand_nats', 1.0)),
                prior_weight=float(_mk.pop('prior_weight', 0.5)),
                prior_floor=float(_mk.pop('prior_floor', 0.02)))
            lda_veff = _veff
            _mk.pop('stream_lda', None)

        # the per-voxel score is a correlation ('corr') or a posterior ('gnb'/'lda'); each
        # goes to its own volume so filter_celltypemaps thresholds it with min_r / min_p.
        from ._count import score_key_for, _other_score_key
        score_key = score_key_for(method)

        def _safe_remove(name):
            if name in self.dataset.zarr_group:
                _remove_array(self.dataset.zarr_group, name)

        if overwrite:
            for name in ['vf_norm', 'celltype_maps', score_key,
                         'celltype_marker_score',
                         'vf_normalized', 'vf_normalized_genes']:
                _safe_remove(name)
        # The other method's score volume is KEPT: re-running 'corr' after 'gnb' used to
        # delete max_probabilities, throwing away a result that costs hours to recompute.
        # Which one filter_celltypemaps thresholds is decided by this attribute, not by
        # which volume happens to exist.
        self.dataset.zarr_group.attrs['mapping_method'] = method

        import numcodecs as _numcodecs
        _safe_remove('genes')
        self.dataset.zarr_group['genes'] = np.asarray(genes)

        self.dataset._zarr_create_array(self.dataset.zarr_group, 'vf_norm', (vx, vy, vz), 'f4')
        self.dataset._zarr_create_array(self.dataset.zarr_group, 'celltype_maps', (vx, vy, vz), 'i4')
        self.dataset._zarr_create_array(self.dataset.zarr_group, score_key, (vx, vy, vz), 'f4')
        # How strongly each voxel expresses ITS OWN celltype's markers — the quantity
        # filter_celltypemaps(hsm_f=) thresholds. Only the lda consumer produces it, and
        # only when the markers are known; the per-gene field exists only inside the
        # chunk loop, so it has to be taken there (see ssaft._marker_corr.marker_score).
        _lda_ms = method == 'lda' and _ms is not None
        if _lda_ms:
            self.dataset._zarr_create_array(
                self.dataset.zarr_group, 'celltype_marker_score', (vx, vy, vz), 'f4')

        use_genes = ([shared_genes[i] for i in range(len(shared_genes))
                      if i not in set(exclude_shared)]
                     if exclude_shared is not None else list(shared_genes))
        n_genes_use = int(centroids_use.shape[1])
        _cpu_vf_ts = None; _cpu_vf_zarr = None; _cpu_inflight = []
        if save_vf:
            from ._count import _tensorstore_array
            _safe_remove('vf_normalized'); _safe_remove('vf_normalized_genes')
            self.dataset.zarr_group['vf_normalized_genes'] = np.asarray(use_genes)
            _sr = getattr(self.dataset.zarr_group.store, 'root', None)
            _gd = (os.path.join(str(_sr), self.dataset.zarr_group.path)
                   if self.dataset.zarr_group.path else str(_sr)) if _sr is not None else None
            _shape = (n_genes_use, vx, vy, vz); _chunks = (n_genes_use, 1, vy, vz)
            _cpu_vf_ts = _tensorstore_array(_gd, 'vf_normalized', _shape, _chunks)
            if _cpu_vf_ts is None:
                _cpu_vf_zarr = self.dataset.zarr_group.create_array(
                    'vf_normalized', shape=_shape, chunks=_chunks, dtype='f4', fill_value=0.0)

        cx, n_chunk = self.calc_chunk(vx, vy, vz, len(shared_genes), mode='map_celltypes_cpu')
        if cx is None or cx <= 0:
            raise RuntimeError('Cannot determine chunk size.')

        import numcodecs as _nc2
        _nc2.blosc.set_nthreads(self.ncores)

        # One bar for the whole run, not one per chunk: a per-chunk bar with
        # leave=True prints n_chunk finished lines (93 of them on a Xenium panel).
        gene_bar = tqdm(
            total=n_chunk * len(grouped_list), desc='KDE genes',
            leave=True, dynamic_ncols=True,
        ) if self.verbose else None
        corr_bar = tqdm(
            total=n_chunk * len(centroids_use), desc='Correlations',
            leave=True, dynamic_ncols=True,
        ) if (self.verbose and method != 'gnb') else None

        for chunk_idx in range(n_chunk):
            x0 = chunk_idx * cx
            x1 = min(vx, x0 + cx)
            local_cx = x1 - x0

            vf_chunk = np.zeros((local_cx, vy, vz, len(shared_genes)), dtype='f4')

            if gene_bar is not None:
                gene_bar.set_postfix_str(f'chunk {chunk_idx + 1}/{n_chunk}', refresh=False)

            for gidx, (_, loc) in enumerate(grouped_list):
                locs = np.asarray(loc)
                if locs.shape[1] == 2:
                    loc_x, loc_y = locs[:, 0].astype(np.float64), locs[:, 1].astype(np.float64)
                    loc_z = np.zeros(locs.shape[0], dtype=np.float64)
                    dim = 2
                else:
                    loc_x, loc_y, loc_z = (locs[:, 0].astype(np.float64),
                                           locs[:, 1].astype(np.float64),
                                           locs[:, 2].astype(np.float64))
                    dim = 3

                accumulate_gene_vector(
                    vf_chunk,
                    loc_x=loc_x, loc_y=loc_y, loc_z=loc_z,
                    sampling_distance=float(sampling_distance),
                    bandwidth=float(bandwidth),
                    prune_coefficient=float(prune_coefficient),
                    gidx=gidx, dim=int(dim), ncores=int(self.ncores),
                    x0=int(x0), y0=0, z0=0, kernel=kernel_code(kernel),
                )
                if gene_bar is not None:
                    gene_bar.update(1)

            vf_norm_chunk = vf_chunk.sum(axis=3)
            self.dataset.zarr_group['vf_norm'][x0:x1, :, :] = vf_norm_chunk

            if save_vf:
                vf_full = (vf_chunk
                           / np.clip(vf_norm_chunk[..., None], 1e-12, None)
                           * float(size_after_normalization))
                if log_transform:
                    np.log1p(vf_full, out=vf_full)
                if exclude_shared is not None:
                    vf_full = np.delete(vf_full, exclude_shared, axis=3)
                vf_full = np.ascontiguousarray(np.moveaxis(vf_full, 3, 0))
                if _cpu_vf_ts is not None:
                    _cpu_inflight.append(_cpu_vf_ts[:, x0:x1, :, :].write(vf_full))
                    while len(_cpu_inflight) > 4:
                        _cpu_inflight.pop(0).result()
                else:
                    _cpu_vf_zarr[:, x0:x1, :, :] = vf_full
                del vf_full

            local_ctmap = np.full((local_cx, vy, vz), -1, dtype=np.int32)
            local_corr = np.full((local_cx, vy, vz), -1, dtype=np.float32)
            local_ms = (np.zeros((local_cx, vy, vz), dtype=np.float32)
                        if _lda_ms else None)

            x_idx, y_idx, z_idx, data_chunk = normalize_vf(
                vf_chunk, vf_norm_chunk, size_after_normalization, x0, self.ncores
            )
            # LDA needs molecule counts, so grab the field BEFORE log1p. vf_norm at each
            # kept voxel turns the L1-normalized vector back into a local count.
            lda_counts = lda_xy = lda_z = None
            if method == 'lda' and data_chunk.shape[0] > 0:
                _raw = (np.delete(data_chunk, exclude_shared, axis=1)
                        if exclude_shared is not None else data_chunk)
                _vn = vf_norm_chunk[x_idx.astype(np.int64) - x0,
                                    y_idx.astype(np.int64), z_idx.astype(np.int64)]
                # float32 throughout: LDAScorer casts to it before the matmul anyway,
                # so the float64 intermediate was computed and then discarded.
                # one scalar pass instead of two: /san then *veff is *(veff/san)
                lda_counts = np.asarray(_raw, dtype=np.float32)
                lda_counts *= np.float32(lda_veff / size_after_normalization)
                lda_counts *= np.asarray(_vn, dtype=np.float32)[:, None]
                _off = np.asarray(self.dataset.zarr_group['coordinate_offset'][:],
                                  dtype=np.float64)
                lda_xy = np.stack([x_idx * sampling_distance + _off[0],
                                   y_idx * sampling_distance + _off[1]], 1)
                lda_z = z_idx * sampling_distance + _off[2]

            if log_transform:
                np.log1p(data_chunk, out=data_chunk)

            if data_chunk.shape[0] > 0:
                data_use = (np.delete(data_chunk, exclude_shared, axis=1)
                            if exclude_shared is not None else data_chunk)

                if method == 'lda':
                    best_idx, best_corr = lda_scorer(lda_counts, lda_xy,
                                                    lda_z)
                elif method == 'gnb':
                    # per-voxel GaussianNB posterior; best_idx = celltype id (= centroid row),
                    # best_corr slot holds the top-class probability (-> max_probabilities).
                    best_idx, best_corr = predict_gnb(
                        gnb_clf, gnb_classes, data_use, gnb_backend, row_normalize=True)
                elif marker_idx is not None:
                    if corr_bar is not None:
                        corr_bar.set_postfix_str(f'chunk {chunk_idx + 1}/{n_chunk}',
                                                 refresh=False)
                    from ._marker_corr import masked_pearson_max
                    best_idx, best_corr = masked_pearson_max(
                        data_use, centroids_use, marker_idx, bar=corr_bar)
                else:
                    best_corr = np.full(data_use.shape[0], -1, dtype=np.float32)
                    best_idx = np.full(data_use.shape[0], -1, dtype=np.int32)
                    if corr_bar is not None:
                        corr_bar.set_postfix_str(f'chunk {chunk_idx + 1}/{n_chunk}',
                                                 refresh=False)
                    for cidx, centroid in enumerate(centroids_use):
                        corr_chunk = calc_ctmap(centroid, data_use, self.ncores)
                        corr_chunk = np.nan_to_num(corr_chunk).astype(np.float32, copy=False)
                        improved = corr_chunk > best_corr
                        best_corr[improved] = corr_chunk[improved]
                        best_idx[improved] = cidx
                        if corr_bar is not None:
                            corr_bar.update(1)

                local_x = x_idx.astype(np.int64) - x0
                local_y = y_idx.astype(np.int64)
                local_z = z_idx.astype(np.int64)
                local_ctmap[local_x, local_y, local_z] = best_idx
                local_corr[local_x, local_y, local_z] = best_corr
                if _lda_ms:
                    from ._marker_corr import marker_score
                    local_ms[local_x, local_y, local_z] = marker_score(
                        lda_counts, best_idx, _ms) / max(lda_veff, 1e-12)

            self.dataset.zarr_group['celltype_maps'][x0:x1, :, :] = local_ctmap
            self.dataset.zarr_group[score_key][x0:x1, :, :] = local_corr
            if _lda_ms:
                self.dataset.zarr_group['celltype_marker_score'][x0:x1, :, :] = local_ms

            del vf_chunk, vf_norm_chunk, x_idx, y_idx, z_idx, data_chunk
            del local_ctmap, local_corr, local_ms
            gc.collect()

        if gene_bar is not None:
            gene_bar.close()
        if corr_bar is not None:
            corr_bar.close()

        for _f in _cpu_inflight:
            _f.result()
        import dask.array as _da
        self.dataset._vf_norm = _da.from_zarr(self.dataset.zarr_group['vf_norm'])
        self.dataset.celltype_maps = self.dataset.zarr_group['celltype_maps'][:]
        setattr(self.dataset, score_key, self.dataset.zarr_group[score_key][:])
        if _lda_ms:
            self.dataset.celltype_marker_score = \
                self.dataset.zarr_group['celltype_marker_score'][:]

    def filter_celltypemaps(self, sdata=None, min_p=_UNSET, min_r=0.6,
                            min_norm=0, norm_f=0.3, norm_depth=True, hsm_f=_UNSET,
                            show_unknown=True,
                            fill_blobs=True, min_blob_area=0, filter_params={},
                            output_mask=None, z=None, background="white",
                            labels_key="ssaft_celltype_map",
                            image_key="ssaft_celltype_map_rgb"):
        """
        Post-filter cell type maps created by `map_celltypes`.

        When ``sdata`` is given (or one was remembered from ``compute_vf`` /
        ``map_celltypes``) the filtered map is also written into it as two aligned
        raster elements and the SpatialData is returned:

        * ``labels[labels_key]`` — the categorical raster; pixel value =
          ``celltype id + 1`` (0 = background / filtered out). Labels, not an
          image: that is what makes the raster addressable as categories
          (``aggregate``, label picking, region queries). A 3D map is written with
          every layer, ``(z, y, x)``.
        * ``images[image_key]`` — an RGB rendering in the palette fixed at
          ``map_celltypes`` time, for quick visual inspection; on a 3D map, of
          layer ``z``.

        No table is written here — the celltype NAMES live once, in
        ``tables['ssaft_downsampled_vectors']`` from :meth:`map_celltypes`, whose
        categorical is index-aligned with these pixel values::

            names = sdata['ssaft_downsampled_vectors'].obs['cellman_annotation'].cat.categories
            name_of_pixel = names[pixel - 1]        # pixel 0 = background

        so re-filtering can mint fresh rasters as often as you like without ever
        duplicating (or desynchronizing) the naming.

        :param sdata: SpatialData to write the raster elements into. Defaults to
            the one remembered from ``compute_vf`` / ``map_celltypes``; pass
            ``None`` with nothing remembered to only update the zarr store.
        :type sdata: spatialdata.SpatialData or None
        :param min_p: minimum threshold of the posterior probability (default
            ``0``, i.e. keep every mapped voxel). Used when the store holds
            ``max_probabilities`` — i.e. the map came from
            ``map_celltypes(method='lda'|'gnb')``; ignored for a correlation map.

            The default is 0 because the posterior answers "which of these
            celltypes", not "is there a cell here", and how high it can go depends on
            the ontology rather than the data: sibling subtypes that share their
            markers split the mass between them, so the same 0.9 discarded 12% of the
            mapped voxels on one panel and 56% on another. Raise it deliberately when
            you want only unambiguous calls, and read a failure as "the siblings are
            not separable here", not as "nothing is there" — ``norm_f`` is the cut that
            answers the second question.
        :type min_p: float
        :param min_r: minimum threshold of the correlation (default ``0.6``). Used
            when the store holds ``max_correlations`` — i.e. the map came from
            ``map_celltypes(method='corr')``, which is filtered on this and
            ``min_norm`` alone; ignored for an lda/gnb map.
        :type min_r: float
        :param min_norm: minimum threshold of the vector norm. Default ``0`` keeps
            every voxel the mapping reached (``vf_norm > 0``), leaving the score
            threshold (``min_p`` / ``min_r``) as the only density-independent filter;
            raise it to also drop low-density voxels. A string selects a local
            thresholding method instead (see ``filter_params``).
            If a string is given instead, then the threshold is automatically determined using
            sklearn's `threshold filter functions <https://scikit-image.org/docs/dev/api/skimage.filters.html>`_ (The functions start with `threshold_`).
        :type min_norm: str or float
        :param norm_f: Per-celltype density cut, as a fraction of the celltype's own
            ``vf_norm`` peak (default ``0.3``; ``None`` switches it off). ``min_p``
            decides whether the label is right; this decides whether there is a cell at
            the voxel at all.

            Density differs between celltypes, so one ``min_norm`` is vacuous for one
            celltype or lethal for another — on the mouse-kidney Xenium section the best
            global cut kept 95% of the tubules and 59% of the fibroblasts. So each
            celltype is cut on its own scale: the half-sample-mode peak of its
            ``log10(vf_norm)``, times ``norm_f``
            (:func:`ssaft._marker_corr.density_cutoffs`). Every celltype has a density,
            so unlike ``hsm_f`` this reaches all of them. On that section 0.6 gave the
            same IoU against a reference segmentation as the global cut (0.505 vs 0.514)
            while keeping 83% of the fibroblasts and 74% of the endothelium; IoU is flat
            from 0.6 to 0.8 and falls above. On the 3D map of the same section
            (depth-normalised), 0.3 removes 8.3% of the mapped voxels against 28.1% at 0.6,
            which empties the medulla between tubules less, so 0.3 is the default.
        :type norm_f: float or None
        :param norm_depth: On a 3D map, divide each plane's density by that plane's
            median before the per-celltype peak and cut are taken (default ``True``).
            Density falls off with depth, so without it one cut per celltype removes the
            shallow and deep planes wholesale (2% kept at z=7 against 81% at z=13 on the
            kidney section; 79-82% across the tissue with it). No effect on a
            single-plane map.
        :type norm_depth: bool
        :param hsm_f: Optional per-celltype cut on how strongly a voxel expresses
            THAT celltype's own markers, as a fraction of the celltype's own peak.
            Off by default (``None``): it reaches only the celltypes the annotation gave
            a marker set, 91% of what it removes on the kidney section ``norm_f``
            removes anyway, and the remaining 9% are dense voxels of weakly marked
            cells (fibroblast, endothelium). Needs ``celltype_marker_score`` in the
            store and raises without it.

            So each celltype is cut on its own scale: the peak of its
            ``celltype_marker_score`` distribution, located by the half-sample mode,
            times ``hsm_f``. Around 0.4 is where the trade stops paying on the crops
            this was tuned on — below it recall barely moves while precision falls,
            above it real cells start going (at 0.6, 20% of the reference glomerular
            endothelium and pericytes were lost) — but the curve is flat enough
            between 0.2 and 0.5 that the exact value matters little.

            Requires the store to carry ``celltype_marker_score``, which
            ``map_celltypes(method='lda')`` writes only when told which markers each
            celltype owns.
        :type hsm_f: float or None
        :param show_unknown: Whether the annotation's ``unknown1..N`` classes stay on
            the filtered map. ``True`` (default) keeps them, so the map shows *where*
            the annotation abstained. ``False`` sends every voxel they won to
            background — dropped, never reassigned to the runner-up: the mapping
            already said no real celltype fits, and promoting a runner-up would invent
            a call the annotation refused to make. This used to be a ``map_celltypes``
            argument; it lives here because it decides what survives, not what a voxel
            is, and re-filtering is seconds where re-mapping is hours. No-op on stores
            whose annotation left no unknown class.
        :type show_unknown: bool
        :param fill_blobs: If True, fill enclosed holes inside the kept blobs.
        :type fill_blobs: bool
        :param min_blob_area: The blobs with its area less than this value will be removed.
        :type min_blob_area: int
        :param filter_params: Filter parameters used for the sklearn's threshold filter functions.
            Not used when `min_norm` is float.
        :type filter_params: dict
        :param output_mask: If given, the cell type maps will be filtered using the output mask.
        :type output_mask: np.ndarray(bool)
        :param z: 3D map: the layer rendered to ``images[image_key]``; defaults to
            the middle one, ``int(n_layers / 2)``. ``labels[labels_key]`` keeps every layer.
        :type z: int or None
        :param background: Colour the filtered-out voxels take in
            ``images[image_key]``. Defaults to ``"white"`` — the RGB element has no
            alpha channel, so background is painted, not left empty, and a figure is
            read on a white page. ``"black"`` restores the old look. The labels
            raster is unaffected: background stays pixel value 0 there.
        :type background: str or list(float)
        :param labels_key: Key of the integer labels element written into ``sdata``.
        :type labels_key: str
        :param image_key: Key of the RGB image element written into ``sdata``.
        :type image_key: str
        :returns: The SpatialData with the two raster elements added, or ``None``
            when no SpatialData was given or remembered.
        :rtype: spatialdata.SpatialData or None
        """

        hsm_explicit = hsm_f is not _UNSET
        if min_p is _UNSET:
            min_p = 0.0
        if not hsm_explicit:
            hsm_f = None

        # Both score volumes can now coexist (a corr run no longer deletes a gnb
        # posterior), so the live one is named by 'mapping_method'; fall back to
        # whichever is present for stores written before that attribute existed.
        _m = self.dataset.zarr_group.attrs.get('mapping_method')
        _use_p = (_m in ('gnb', 'lda')) if _m else (self.dataset.max_probabilities is not None)
        if _use_p and self.dataset.max_probabilities is not None:
            score_vol = self.dataset.max_probabilities
            if score_vol.ndim == 4:
                score_vol = score_vol[..., 0]
            r_thr = min_p
        else:
            score_vol = self.dataset.max_correlations
            r_thr = min_r

        celltype_maps = self.dataset.celltype_maps

        if isinstance(min_norm, str):
            _filter_params = filter_params.copy()
            filter_offset = _filter_params.pop('offset', 0)

            # Geometry comes from the map, not from ``dataset.shape`` — that attribute is
            # only set as a side effect of mapping, so re-filtering a store opened fresh
            # (the whole point of an adaptive threshold: retune without re-mapping) used
            # to raise AttributeError. A 2D ``vf_norm`` is lifted the same way the scalar
            # branch below lifts it.
            _vfn = self.dataset.vf_norm
            _flat = int(getattr(_vfn, 'ndim', 3)) == 2 and celltype_maps.ndim == 3
            mask = np.zeros(celltype_maps.shape, dtype=bool)
            for cidx in np.unique(celltype_maps):
                if cidx == -1:
                    continue
                ctcorr = np.full_like(score_vol, -1.0)
                cidx_sel = celltype_maps == cidx
                ctcorr[cidx_sel] = score_vol[cidx_sel]
                for z in range(celltype_maps.shape[2]):
                    vf_norm_z = (np.asarray(_vfn).T if _flat
                                 else np.asarray(_vfn[..., z]))
                    ctcorr_mask_z = ctcorr[..., z] > r_thr
                    if min_norm in ["local", "niblack", "sauvola", "localotsu"]:
                        im = np.zeros(vf_norm_z.shape)
                        im[ctcorr_mask_z] = vf_norm_z[ctcorr_mask_z]
                    if min_norm == "localotsu":
                        max_norm = np.max(im)
                        im /= max_norm
                        selem = disk(_filter_params['radius'])
                        min_norm_cut = filters.rank.otsu(im, selem) * max_norm
                    else:
                        filter_func = getattr(filters, "threshold_" + min_norm)
                        if min_norm in ["local", "niblack", "sauvola"]:
                            min_norm_cut = filter_func(im, **_filter_params)
                        else:
                            highr_norm = vf_norm_z[ctcorr_mask_z]
                            if len(highr_norm) == 0 or np.max(highr_norm) == np.min(highr_norm):
                                min_norm_cut = np.max(self.dataset.vf_norm)
                            else:
                                min_norm_cut = filter_func(highr_norm, **_filter_params)
                    min_norm_cut += filter_offset
                    mask[..., z][np.logical_and(vf_norm_z > min_norm_cut, ctcorr_mask_z)] = 1
        else:
            vf_norm = np.asarray(self.dataset.vf_norm)
            if vf_norm.ndim == 2 and score_vol.ndim == 3:
                vf_norm = vf_norm.T[..., None]
            mask = (vf_norm > min_norm) & (score_vol > r_thr) & (celltype_maps != -1)

        # The per-celltype density cut: after the score mask, so each celltype's peak is
        # read off the voxels that survived it, and before the marker cut and the blob
        # pass, like the rest of the thresholds.
        if norm_f is not None:
            from ._marker_corr import density_cutoffs
            _cm = np.asarray(celltype_maps)
            _vf = np.asarray(self.dataset.vf_norm, dtype=np.float32)
            if _vf.ndim == 2 and _cm.ndim == 3:
                _vf = _vf.T[..., None]
            if _vf.shape != _cm.shape:
                _vf = np.broadcast_to(_vf, _cm.shape)
            _cuts, _factor = density_cutoffs(_cm, _vf, float(norm_f), mask=mask,
                                             depth_normalize=bool(norm_depth))
            if _cm.ndim == 3 and _factor.size > 1 and not np.all(_factor == 1.0):
                _vf = _vf / np.nan_to_num(_factor, nan=1.0).astype(np.float32)[None, None, :]
            n_before = int(mask.sum())
            # One pass, not one per celltype. `mask &= ~((_cm == cidx) & (_vf < cut))` walks the
            # whole volume twice for every celltype -- 38 passes over 2.5e9 voxels on a section
            # this size, each allocating its own temporary. The cut is a property of the voxel's
            # celltype, so it can be gathered: cut_of[_cm] gives every voxel its own threshold and
            # the comparison happens once. Same mask, 1/19th of the traffic.
            _ncls = int(max(_cuts) if _cuts else 0) + 2
            cut_of = np.full(_ncls, -np.inf, dtype=np.float32)      # -inf: not cut at all
            for cidx, cut in _cuts.items():
                cut_of[int(cidx)] = cut
            _lastax = _cm.ndim - 1
            for _k in range(_cm.shape[_lastax]):                   # a plane at a time: flat memory
                _cmk = _cm[..., _k]
                _safe = np.where((_cmk >= 0) & (_cmk < _ncls - 1), _cmk, _ncls - 1)
                mask[..., _k] &= ~(_vf[..., _k] < cut_of[_safe])
            _depth = (f', density normalised over {int(np.isfinite(_factor).sum())} planes'
                      if _factor.size > 1 and not np.all(_factor == 1.0) else '')
            self._m(f"filter_celltypemaps: norm_f={norm_f} -> "
                    f"{n_before - int(mask.sum()):,} of {n_before:,} voxels below their "
                    f"celltype's density cut ({len(_cuts)} celltypes{_depth})")
            del _vf, _cm

        # After the score/density mask, so the peak is estimated from the voxels that
        # survived it, and before the blob pass, so a marker-poor region never counts
        # into a blob's area.
        # A correlation map is filtered on min_r alone. Both score volumes can coexist
        # in one store, so a leftover 'celltype_marker_score' from an earlier lda run
        # must not leak a marker cut into a corr map: the live method decides.
        if hsm_f is not None and not _use_p:
            if hsm_explicit:
                raise ValueError(
                    "hsm_f does not apply to a correlation map — it thresholds the "
                    "per-celltype marker score that only map_celltypes(method='lda') "
                    "produces. Filter a corr map with min_r (and min_norm).")
            hsm_f = None
        if hsm_f is not None:
            from ._marker_corr import marker_score_cutoffs
            ms = getattr(self.dataset, 'celltype_marker_score', None)
            if ms is None and 'celltype_marker_score' in self.dataset.zarr_group:
                ms = self.dataset.zarr_group['celltype_marker_score'][:]
            if ms is None:
                # corr/gnb maps, and lda maps run without markers, carry no such
                # volume. Asking for the cut by hand on one of those is a mistake worth
                # stopping for; arriving here on the default is not.
                if hsm_explicit:
                    raise ValueError(
                        "hsm_f needs 'celltype_marker_score' in the store. It is "
                        "written by map_celltypes(method='lda') only when the markers "
                        "each celltype owns are known - re-map with "
                        "marker_genes_path=...")
                self._m(f"filter_celltypemaps: the store has no "
                        f"'celltype_marker_score', so the default hsm_f={hsm_f} is "
                        "skipped (map with method='lda', marker_genes_path=... to "
                        "enable it)")
                hsm_f = None
        if hsm_f is not None:
            ms = np.asarray(ms, dtype=np.float64)
            if ms.ndim == 4:
                ms = ms[..., 0]
            if ms.shape != celltype_maps.shape:
                raise ValueError(
                    f"celltype_marker_score {ms.shape} does not match celltype_maps "
                    f"{celltype_maps.shape}; the store was re-mapped without it")
            # ``notes`` names the celltypes whose peak should not be trusted (their
            # score sits on the clip floor, or the peak is not central). They are not
            # printed here: filtering is re-run often while tuning, and a per-celltype
            # complaint on every call trains the reader to skip the output. The same
            # flags come back from marker_score_cutoffs, and notebooks/preview_hsm.py
            # shows them next to the distribution they refer to, which is where the
            # decision is actually made.
            cutoffs, _notes = marker_score_cutoffs(
                np.where(mask, celltype_maps, -1), ms, float(hsm_f))
            n_before = int(mask.sum())
            for cidx, cut in cutoffs.items():
                mask &= ~((celltype_maps == cidx) & (ms < cut))
            self._m(f"filter_celltypemaps: hsm_f={hsm_f} -> "
                    f"{n_before - int(mask.sum()):,} of {n_before:,} voxels below "
                    f"their celltype's marker cut ({len(cutoffs)} celltypes)")

        # Hidden before the blob pass so an unknown region is never counted into a
        # blob's area, and re-applied to the labels below because fill_blobs would
        # otherwise pour a kept celltype's holes back over it.
        unknown_ids = self._unknown_class_ids() if not show_unknown else []
        if unknown_ids:
            names, _ = self._celltype_names_colors()
            is_unknown = np.isin(celltype_maps, unknown_ids)
            n_unk = int((mask & is_unknown).sum())
            mask = mask & ~is_unknown
            self._m(f"filter_celltypemaps: show_unknown=False -> {n_unk:,} voxels of "
                    f"{[str(names[i]) for i in unknown_ids]} sent to background")

        if min_blob_area > 0:
            blob_labels = measure.label(mask, background=0)
            for bp in measure.regionprops(blob_labels):
                if bp.area_filled < min_blob_area:
                    mask[tuple(bp.coords.T)] = 0
                    continue
                if fill_blobs and bp.area != bp.area_filled:
                    minx, miny, minz, maxx, maxy, maxz = bp.bbox
                    mask[minx:maxx, miny:maxy, minz:maxz] |= bp.image_filled
        elif fill_blobs:
            mask = _binary_fill_holes_fast(mask)

        filtered_ctmaps = np.array(celltype_maps, copy=True)
        filtered_ctmaps[mask == False] = -1
        if unknown_ids:
            filtered_ctmaps[is_unknown] = -1

        if output_mask is not None:
            filtered_ctmaps[~output_mask.astype(bool)] = -1
        self.dataset.filtered_celltype_maps = filtered_ctmaps
        if 'filtered_celltype_maps' in self.dataset.zarr_group:
            _remove_array(self.dataset.zarr_group, 'filtered_celltype_maps')
        self.dataset.zarr_group['filtered_celltype_maps'] = self.dataset.filtered_celltype_maps

        sdata = sdata if sdata is not None else getattr(
            self.dataset, '_original_input_data', None)
        if sdata is None:
            self._m('filter_celltypemaps: no SpatialData given and none '
                    'remembered; store updated, rasters skipped')
            return None
        _, colors = self._celltype_names_colors()
        return self.dataset.write_celltype_map(
            sdata, colors=colors, z=z, min_r=min_r, background=background,
            labels_key=labels_key, image_key=image_key)

    def _celltype_map_affine(self, sdata, z=None):
        """``(map2d, sd_um, ox, oy)`` for the filtered map: the ``(y, x)`` label
        raster plus the same µm affine :meth:`Dataset.write_celltype_map` writes,
        so anything derived here overlays the rasters exactly."""
        data = np.asarray(self.dataset.filtered_celltype_maps)
        if data.ndim == 2:
            data = data[..., None]
        if z is None:
            z = int(data.shape[2] / 2)
        map2d = np.asarray(data[..., z].T)                 # (x,y,z) -> (y,x)
        zg = self.dataset.zarr_group
        sd_um = float(zg.attrs.get("sampling_distance",
                      (zg["vf_params"][0] if "vf_params" in zg else 1.0)))
        ox, oy = self.dataset._celltype_map_origin(sdata)
        sd_um *= float(getattr(self.dataset, "x_scale", 1.0) or 1.0)
        return map2d, sd_um, float(ox), float(oy), z

    def _nuclei_to_um(self, sdata, nuclei, frame=None):
        """Resolve ``nuclei`` to an ``(N, 2)`` array in the CELL-TYPE MAP's frame.

        Accepts an ``(N, 2)`` array-like (taken as already being in that frame), or
        the key of a SpatialData element: shapes (polygon centroids / circle
        centres) or points (x, y columns).

        The map is built from the INTRINSIC coordinates of whatever element ssaft
        was run on (``_validate_input_data`` reads ``df['x']`` straight off the
        points frame; the count backend uses ``obsm['spatial']``), and 'global'
        carries no unit contract in SpatialData — it is µm on VisiumHD and mosaic
        PIXELS on MERSCOPE. A marker element therefore may or may not already be in
        the map's frame, and neither 'always intrinsic' nor 'always global' is
        right:

        =========  ==================  =================
        dataset    intrinsic           global
        =========  ==================  =================
        VisiumHD   WSI pixels (2.5%)   µm (100%)
        MERFISH    µm (99.1%)          pixels (0.0%)
        =========  ==================  =================

        (percentages = markers landing inside the map's own coordinate bounds.)
        So both candidates are scored against those bounds and the one that
        actually lands on the map wins. Pass ``frame='intrinsic'`` / ``'global'``
        to skip the check and force one.
        """
        self._nuclei_global_affine = None
        self._cells_cs = None
        if not isinstance(nuclei, str):
            xy = np.asarray(nuclei, dtype=float)
            if xy.ndim != 2 or xy.shape[1] < 2:
                raise ValueError("nuclei must be an (N, 2) array of µm coordinates")
            return xy[:, :2]
        if sdata is None:
            raise ValueError(f"nuclei='{nuclei}' given but no SpatialData to read it from")
        if nuclei in getattr(sdata, "shapes", {}):
            el = sdata.shapes[nuclei]
            cen = el.geometry.centroid
            xy = np.c_[cen.x.to_numpy(), cen.y.to_numpy()]
        elif nuclei in getattr(sdata, "points", {}):
            el = sdata.points[nuclei]
            xy = el[["x", "y"]].compute().to_numpy(dtype=float)
        else:
            raise KeyError(f"'{nuclei}' is not a shapes or points element of the SpatialData")

        from spatialdata.transformations import get_transformation
        count_frame = self.dataset._count_frame()
        if count_frame is not None and frame is None:
            # spatialdata_io count input: the map's µm frame is tied to the reader's
            # coordinate system, so the markers go through it and nothing is guessed
            cs = count_frame["coordinate_system"]
            t_cs = get_transformation(el, get_all=True).get(cs)
            if t_cs is None:
                raise ValueError(f"'{nuclei}' has no transformation to {cs!r}, the coordinate "
                                 f"system the count data was read in")
            um_to_cs = np.asarray(count_frame["um_to_cs"], dtype=float)
            M = np.linalg.inv(um_to_cs) @ np.asarray(
                t_cs.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")), dtype=float)
            self._nuclei_global_affine = um_to_cs
            self._cells_cs = cs
            self._m(f"segment_cells: markers read through coordinate system {cs!r}")
            return (M[:2, :2] @ xy.T).T + M[:2, 2]
        t = get_transformation(el, "global")
        if t is None:
            self._nuclei_global_affine = np.eye(3)
            return xy
        A = np.asarray(t.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")), dtype=float)
        xy_global = (A[:2, :2] @ xy.T).T + A[:2, 2]
        if frame == "intrinsic":
            self._nuclei_global_affine = A
            return xy
        if frame == "global":
            self._nuclei_global_affine = np.eye(3)
            return xy_global

        # The two backends write these bounds in different layouts: the transcript
        # path stores (3, 2) [[x0, x1], [y0, y1], [z0, z1]], the count path a flat
        # [x0, y0, z0, x1, y1, z1].
        zg = getattr(self.dataset, "zarr_group", None)
        b = (np.asarray(zg["original_coordinate_bounds"][:], dtype=float)
             if zg is not None and "original_coordinate_bounds" in zg else None)
        if b is None:
            self._nuclei_global_affine = A
            return xy                                   # nothing to score against
        if b.ndim == 2 and b.shape[0] >= 2:
            (x0, x1), (y0, y1) = b[0][:2], b[1][:2]
        elif b.ndim == 1 and b.size == 6:
            # the count backend writes these bounds relative to the grid origin
            gx, gy = self.dataset._celltype_map_origin(sdata)
            x0, y0, x1, y1 = b[0] + gx, b[1] + gy, b[3] + gx, b[4] + gy
        elif b.ndim == 1 and b.size == 4:
            x0, y0, x1, y1 = b
        else:
            self._nuclei_global_affine = A
            return xy

        def hit(v):
            return float(((v[:, 0] >= x0) & (v[:, 0] <= x1) &
                          (v[:, 1] >= y0) & (v[:, 1] <= y1)).mean())
        h_int, h_glob = hit(xy), hit(xy_global)
        pick, frac = (xy, h_int) if h_int >= h_glob else (xy_global, h_glob)
        self._nuclei_global_affine = A if pick is xy else np.eye(3)
        self._m(f"segment_cells: markers read in "
                f"{'intrinsic' if pick is xy else 'global'} frame "
                f"({frac:.1%} inside the map; other frame {min(h_int, h_glob):.1%})")
        if frac < 0.05:
            raise ValueError(
                f"markers from '{nuclei}' do not land on the cell-type map in either "
                f"frame (intrinsic {h_int:.1%}, global {h_glob:.1%}); map bounds are "
                f"x [{x0:.0f}, {x1:.0f}] y [{y0:.0f}, {y1:.0f}]")
        return pick

    def _count_transcripts_per_cell(self, sdata, labels, lab_of_cell,
                                    sd_um, ox, oy, transcripts=None):
        """Aggregate a points element into a cell x gene count matrix.

        Each transcript falls in exactly one voxel of the ``labels`` raster, so the
        raster performs the assignment in one pass — no point-in-polygon. Returns
        ``(csr_matrix, gene_names, n_assigned)``. Transcripts outside every cell are
        dropped (that is the segmentation saying they belong to no cell)."""
        from scipy.sparse import coo_matrix

        zg = getattr(self.dataset, 'zarr_group', None)
        if transcripts is None and zg is not None and zg.attrs.get('backend') == 'blur':
            # a count store: any points element (Stereo-seq's bin positions) is not molecules
            return None, [], 0
        key = self._pick_element(getattr(sdata, 'points', {}), transcripts,
                                 kind='points', param='transcripts', required=False)
        if key is None:
            return None, [], 0
        self._m(f"segment_cells: counting points['{key}']")
        pts = sdata.points[key]
        if hasattr(pts, 'compute'):
            pts = pts.compute()
        if 'is_gene' in pts.columns:
            pts = pts[pts['is_gene'] == True]                      # noqa: E712
        gene_col = next((c for c in ('gene', 'gene_symbol', 'feature_name', 'target')
                         if c in pts.columns), None)
        if gene_col is None:
            raise ValueError(f"gene column not found in points['{key}']: "
                             f"{list(pts.columns)}")

        vx = np.rint((pts['x'].to_numpy(dtype=np.float64) - ox) / sd_um).astype(np.int64)
        vy = np.rint((pts['y'].to_numpy(dtype=np.float64) - oy) / sd_um).astype(np.int64)
        H, W = labels.shape
        ok = (vx >= 0) & (vx < W) & (vy >= 0) & (vy < H)
        lab = np.zeros(len(vx), dtype=np.int64)
        lab[ok] = labels[vy[ok], vx[ok]]

        # label value -> row in the cell table (0..n_cells-1); anything else drops out
        # size by both: labels can be all-background while cells still exist
        n_lab = int(max(labels.max(), lab_of_cell.max() if len(lab_of_cell) else 0)) + 1
        row_of_lab = np.full(n_lab, -1, dtype=np.int64)
        row_of_lab[lab_of_cell] = np.arange(len(lab_of_cell))
        rows = np.where(lab > 0, row_of_lab[np.clip(lab, 0, None)], -1)
        hit = rows >= 0

        genes = pts[gene_col].astype(str).to_numpy()[hit]
        var_names, cols = np.unique(genes, return_inverse=True)
        X = coo_matrix((np.ones(hit.sum(), dtype=np.float32), (rows[hit], cols)),
                       shape=(len(lab_of_cell), len(var_names))).tocsr()
        return X, list(var_names), int(hit.sum())

    def _resolve_bin_table(self, sdata, bins=None):
        """Find the bin x gene table to aggregate. Returns ``(table, label)``.

        Tried in order: the explicit ``bins`` argument (AnnData, GridDataset, or a
        table key of ``sdata``), the GridDataset ``compute_vf`` was run on, and
        finally a table of ``sdata``. The last step never guesses between several
        candidates — ``uns['cellman']`` marks the count table a count-platform
        loader wrote, and ssaft's own tables are excluded, so anything still
        ambiguous is raised rather than picked.
        """
        if bins is not None:
            if isinstance(bins, str):
                key = self._pick_element(getattr(sdata, 'tables', {}), bins,
                                         kind='table', param='bins')
                return sdata.tables[key], f"tables['{key}']"
            if hasattr(bins, 'obsm'):                       # AnnData
                return bins, 'the AnnData passed as bins='
            raw = getattr(bins, 'raw', None)                # GridDataset
            if raw is not None:
                return raw, 'the GridDataset passed as bins='
            raise ValueError("bins must be an AnnData, a GridDataset, or a table key")

        # the count regime remembers the GridDataset it was run on; the transcript
        # regime remembers a SpatialData, which has no .raw and falls through
        remembered = getattr(self.dataset, '_original_input_data', None)
        raw = getattr(remembered, 'raw', None)
        if raw is not None:
            return raw, "the GridDataset remembered by compute_vf"
        grid = getattr(self, '_count_grid', None)
        if grid is not None:
            return grid.raw, "the GridDataset compute_vf ran on"

        from ._io._spatialdata_io import grid_from_spatialdata_io, is_count_reader_output
        if sdata is not None and is_count_reader_output(sdata):
            f = self.dataset._count_frame() or {}
            g = grid_from_spatialdata_io(sdata, table_key=f.get('table_key'), load_image=False)
            return g.raw, f"tables['{f.get('table_key') or 'bins'}'] (µm lattice)"

        tables = getattr(sdata, 'tables', None) or {}
        marked = {k: t for k, t in tables.items()
                  if k not in self._OWN_ELEMENTS and 'cellman' in getattr(t, 'uns', {})}
        if len(marked) == 1:
            k = next(iter(marked))
            return tables[k], f"tables['{k}']"
        if marked:
            raise ValueError(f"{len(marked)} count tables carry uns['cellman'] — pass "
                             f"bins= to say which one: {sorted(marked)}")
        key = self._pick_element(tables, None, kind='table', param='bins',
                                 required=False)
        if key is None:
            return None, None
        return tables[key], f"tables['{key}']"

    def _count_bins_per_cell(self, sdata, labels, lab_of_cell,
                             sd_um, ox, oy, bins=None):
        """Aggregate a binned count table into a cell x gene count matrix.

        The array platforms (Visium HD, Stereo-seq, StrataMap) carry no molecule
        coordinates. Their localized signal is a bin x gene table whose
        ``obsm['spatial']`` holds each bin's centre in the same µm frame as the
        cell-type map, so the assignment is exactly the points path — read the
        label under the bin centre, one raster lookup, no point-in-polygon. Only
        the accumulation differs: a molecule contributes 1, a bin contributes its
        whole count vector.

        A bin is never split; it belongs to the cell its centre falls in. That is
        harmless while the bin pitch is at or below the map's voxel size (Visium HD
        2 µm bins on a 2 µm/px map), and the ratio is reported so a coarser bin
        cannot pass unnoticed. Bins whose centre lands on background are dropped —
        that is the segmentation saying they belong to no cell.

        Unlike the points path the full panel is kept in ``var``, including genes
        that ended up with zero counts, so the gene axis stays comparable across
        cells and datasets.

        Returns ``(csr_matrix, gene_names, n_counts_assigned)``, or
        ``(None, [], 0)`` when no bin table can be found.
        """
        from scipy.sparse import coo_matrix, csr_matrix, issparse

        tab, src = self._resolve_bin_table(sdata, bins)
        if tab is None:
            return None, [], 0
        self._m(f"segment_cells: aggregating {src}")
        if 'spatial' not in getattr(tab, 'obsm', {}):
            raise ValueError("bin table needs obsm['spatial'] (bin centres in µm)")
        xy = np.asarray(tab.obsm['spatial'], dtype=np.float64)

        C = tab.layers['counts'] if 'counts' in getattr(tab, 'layers', {}) else tab.X
        C = C.tocsr() if issparse(C) else csr_matrix(np.asarray(C))
        if C.dtype != np.float32:
            C = C.astype(np.float32)

        vx = np.rint((xy[:, 0] - ox) / sd_um).astype(np.int64)
        vy = np.rint((xy[:, 1] - oy) / sd_um).astype(np.int64)
        H, W = labels.shape
        ok = (vx >= 0) & (vx < W) & (vy >= 0) & (vy < H)
        lab = np.zeros(len(vx), dtype=np.int64)
        lab[ok] = labels[vy[ok], vx[ok]]

        # label value -> row in the cell table (0..n_cells-1); anything else drops out
        # size by both: labels can be all-background while cells still exist
        n_lab = int(max(labels.max(), lab_of_cell.max() if len(lab_of_cell) else 0)) + 1
        row_of_lab = np.full(n_lab, -1, dtype=np.int64)
        row_of_lab[lab_of_cell] = np.arange(len(lab_of_cell))
        rows = np.where(lab > 0, row_of_lab[np.clip(lab, 0, None)], -1)
        idx = np.flatnonzero(rows >= 0)

        # (cells x bins) selector @ (bins x genes) counts == group-sum by cell
        sel = coo_matrix((np.ones(len(idx), dtype=np.float32), (rows[idx], idx)),
                         shape=(len(lab_of_cell), C.shape[0])).tocsr()
        X = (sel @ C).tocsr()

        # a bin is assigned whole, so a pitch coarser than the voxel is worth seeing
        steps = np.diff(np.unique(xy[:, 0]))
        pitch = f"~{np.median(steps):.2f}µm" if steps.size else "unknown"
        self._m(f"segment_cells: {len(idx):,}/{len(xy):,} bins assigned, "
                f"{int(X.sum()):,} counts (bin pitch {pitch} vs map {sd_um:.2f}µm/px)")
        return X, list(map(str, tab.var_names)), int(X.sum())

    def _cells_global_affine(self, sdata):
        """``(3, 3)`` affine carrying the cell-type map's µm frame into ``'global'``.

        The frame the nuclei were read in decides (:meth:`_nuclei_to_um` records it); for an
        array of markers, which carries no frame, the map's own (:meth:`Dataset._map_to_global_affine`).
        """
        A = getattr(self, "_nuclei_global_affine", None)
        return np.asarray(A, dtype=float) if A is not None else self.dataset._map_frame(sdata)[1]

    def _excluded_celltype_ids(self, exclude, names):
        """Cell-type ids named by ``exclude`` (names, as in the map's celltype list, or ids).

        Matching on names is case-insensitive and ignores surrounding space; a name that is not in
        this map's cell types is an error rather than a silent no-op, because the caller asked for a
        cell type to be left whole and would not see it happen.
        """
        if exclude is None:
            return set()
        if isinstance(exclude, (str, int, np.integer)):
            exclude = [exclude]
        by_name = {str(n).strip().lower(): i for i, n in enumerate(names)}
        ids, missing = set(), []
        for e in exclude:
            if isinstance(e, (int, np.integer)) and not isinstance(e, bool):
                ids.add(int(e))
                continue
            i = by_name.get(str(e).strip().lower())
            if i is None:
                missing.append(str(e))
            else:
                ids.add(int(i))
        if missing:
            raise ValueError(f"exclude_celltypes: {missing} not among the map's cell types "
                             f"({len(names)}): {sorted(names)[:8]}...")
        return ids

    def _cells_coordinate_system(self, sdata):
        """Coordinate system the cells are written into (see :meth:`_cells_global_affine`)."""
        cs = getattr(self, "_cells_cs", None)
        return cs if cs is not None else self.dataset._map_frame(sdata)[0]

    def segment_cells(self, sdata=None, nuclei=None, *, z=None,
                      simplify=None, min_area_um2=5.0, max_area_um2=None,
                      max_volume_um3=None, exclude_celltypes=None, transcripts=None, bins=None,
                      shapes_key="ssaft_cell_boundaries_2d",
                      shapes_3d_key="ssaft_cell_boundaries_3d",
                      labels_3d_key="ssaft_cell_labels_3d", table_key="ssaft_cells",
                      column="cellman_annotation", nuclei_frame=None, tile=1024, pad_um=64.0,
                      nuclei_z=None):
        """Cut the cell-type map into single cells by a celltype-constrained watershed.

        The segmentation-free map says *what* each voxel is but not *which cell* it
        belongs to. Given nuclei as markers this splits it into cells under one rule:

        * two cells of **different** cell types are separated by the boundary the
          cell-type map already found;
        * two cells of the **same** cell type are separated midway between their
          nuclei — the map carries no information there.

        Both fall out of running one watershed per cell type, confined to that type's
        own region::

            for c in celltypes:
                mask_c = (ctmap == c)
                dist   = distance_transform_edt(markers_c == 0)
                labels = watershed(dist, markers_c, mask=mask_c)

        A nucleus sitting on background (``-1``, i.e. filtered out by
        :meth:`filter_celltypemaps`) is ignored; every other nucleus yields one cell.

        **2D** (``n_dims == 2``, or ``z`` given): one plane of the map, ``z`` or the middle
        one. Markers are the nucleus centroids, a marker's cell type the map's label under
        it. Writes ``shapes[shapes_key]`` and ``tables[table_key]`` annotating it.

        **3D** (the store's ``n_dims == 3`` and ``z`` not given): the watershed runs in 3D over
        every layer of the map. Which voxels are tissue is :meth:`filter_celltypemaps`' decision,
        not the segmentation's, so no layers are picked here.

        * *markers* -- a marker only says where the water is poured: each nucleus centroid is one
          seed voxel, and its cell type is the map's label there. The seed's layer is the
          nucleus's own z (``nuclei_z``) when it is known and lands on a labelled voxel; otherwise
          the labelled voxel of the nucleus's xy column nearest the middle of the column's
          labelled run (the local tissue midline). A nucleus whose column is background on every
          layer is ignored. One fixed layer for all nuclei would drop every nucleus where a tilted
          or uneven section does not cross that layer -- two thirds of them on a Xenium section
          spread over 60 µm in z.
        * *flood* -- per cell type, the 3D watershed of the Euclidean distance to the seeds,
          confined to that cell type's voxels, computed in xy tiles of ``tile`` voxels with
          ``pad_um`` of overlap (each tile keeps its core) on ``ncores`` processes.

        It writes

        * ``labels[labels_3d_key]`` -- the ``(z, y, x)`` cell labels (0 = no cell), value =
          ``label_id``;
        * ``shapes[shapes_3d_key]`` -- every cell's outline on every layer, one row per
          (cell, layer), with ``cell_id``, ``z_index`` and ``z_um``;
        * ``shapes[shapes_key]`` -- one outline per cell, on the cell's own seed layer (the
          nucleus's layer);
        * ``tables[table_key]`` annotating the 3D labels (``instance_key='label_id'``): one
          row per cell, transcripts counted by the 3D voxel they fall in (z clamped into the
          layers), ``obs`` with the cell type, ``volume_um3``, ``n_z``,
          ``max_plane_area_um2``, ``area_um2`` (seed layer) and ``seed_z_um``, ``obsm['spatial']`` the
          voxel centroid in µm (x, y, z).

        ``min_area_um2`` drops a 3D cell whose largest layer is below it.

        ``exclude_celltypes`` leaves a cell type whole. The watershed assumes one nucleus = one
        cell, which a syncytium breaks: syncytiotrophoblast is one multinucleated body, and cutting
        it midway between its nuclei invents cells that do not exist. A named type is skipped by the
        flood and labelled by CONNECTED COMPONENT instead, so each contiguous piece of it (one
        villous branch, say) becomes one object, exactly as the filtered map drew it. Its nuclei are
        not seeds; they are counted, and ``obs['n_nuclei']`` says how many fell in each body.
        ``obs['segmented']`` is False for those rows, and neither area bound applies to them -- a cap
        exists to stop a cell running away, which is not a question for a body the map itself
        delimited. The other cell types are untouched: each type floods only inside its own mask, so
        excluding one cannot move a voxel of another.

        ``max_volume_um3`` (3D) and ``max_area_um2`` (2D) cap how far a cell may flood -- both OFF
        by default. The bound follows the dimension because a cell's size does: in 3D a per-layer
        area bound would leave the total free to grow with the number of layers the cell spans
        (200 um^2 over 30 layers is 6,000 um^3), so the 3D run bounds the VOLUME and the 2D run the
        area. Passing the other dimension's bound is an error rather than a silent no-op. The water is poured on
        the nucleus, so that is the part a cap must never take: voxels are kept in order of their
        distance to the cell's own seed and everything past the cap is given back to the
        background, which eats the cell's outside inwards and always leaves the nucleus. Turn it
        on where a nucleus sitting alone in a wide region of its own celltype claims all of it --
        the watershed has no other cell there to stop it, so on a tissue edge or for a sparse
        celltype a single "cell" can grow to hundreds of µm². In 3D the cap is applied per layer,
        i.e. to the cross-sectional area, which is what ``area_um2`` reports. It is off by default
        because the honest size of a cell is the one the map and the neighbouring nuclei give it;
        a cap is a correction to be chosen per dataset, not a default (100 µm² is a reasonable
        starting point -- it is about a 11 µm disc).

        Polygons trace voxel BOUNDARIES (:func:`spatialdata.to_polygons`): a contour through
        boundary voxel centres would inset every cell by half a voxel and lose 1-2 voxel
        cells. Geometry is in the map's µm frame (so areas are µm²); the elements carry the
        transformation of the frame the nuclei were read in, so they overlay the image, the
        nuclei and the transcripts in ``'global'``. The cell-type categorical is pinned to the
        celltype id order and the palette travels in ``uns[f'{column}_colors']``, so a render
        comes out in the colours of ``images['ssaft_celltype_map_rgb']``::

            sdata.pl.render_labels('ssaft_cell_labels_3d', color='cellman_annotation',
                                   table_name='ssaft_cells').pl.show()        # 3D
            sdata.pl.render_shapes('ssaft_cell_boundaries_2d', color='cellman_annotation',
                                   table_name='ssaft_cells').pl.show()        # 2D

        :param sdata: SpatialData to read the nuclei from and write the cells into. Defaults to
            the one remembered from ``compute_vf`` / ``map_celltypes``.
        :param nuclei: Marker positions: an ``(N, 2)`` array in the map's µm frame, or the key
            of a shapes element (polygon centroids, e.g. Xenium ``'nucleus_boundaries'``) or a
            points element with ``x`` / ``y`` columns.
        :param z: 2D only: the map layer to segment. Giving it on a 3D store segments that one
            layer in 2D.
        :param simplify: Douglas-Peucker tolerance in µm for the polygons (``None`` = exact).
        :param min_area_um2: Drop cells smaller than this (2D: the polygon; 3D: its largest
            layer). ``0`` keeps all.
        :param exclude_celltypes: Cell types (names or ids) NOT to cut into cells: each stays the
            body the cell-type map found, one object per connected piece. See above.
        :param max_area_um2: 2D only. Cap on one cell's area, trimmed from the voxels farthest
            from the nucleus the cell was seeded on. ``None`` (the default) / ``0`` leaves the
            watershed uncapped.
        :param max_volume_um3: 3D only. Cap on one cell's volume, trimmed from the voxels farthest
            (in 3D) from its nucleus. ``None`` (the default) / ``0`` leaves it uncapped.
        :param transcripts: Key of the points element to count; defaults to the only one.
        :param bins: 2D count platforms: the bin table to aggregate when there are no points.
        :param shapes_key: Outlines on the segmented 2D plane / the 3D middle layer.
        :param shapes_3d_key: 3D only: per-layer outlines.
        :param labels_3d_key: 3D only: the 3D label raster.
        :param table_key: The cell x gene table.
        :param column: ``obs`` column for the cell type (as :meth:`map_celltypes` writes).
        :param nuclei_frame: ``'intrinsic'`` / ``'global'`` to force the frame the nuclei are
            read in (default: the one that lands on the map).
        :param tile: 3D only: xy tile size, voxels.
        :param pad_um: 3D only: tile overlap, µm.
        :param nuclei_z: 3D only: each nucleus's z in µm (the transcripts' z frame), in the order
            of ``nuclei``; ``NaN`` where unknown. E.g. the median z of the transcripts inside each
            nucleus (Xenium ``overlaps_nucleus``). Without it every seed sits on its column's
            tissue midline.
        :returns: The SpatialData with the elements added.
        """
        if nuclei is None:
            raise ValueError("nuclei= is required (array of µm coords, or a "
                             "shapes/points element key)")
        sdata = sdata if sdata is not None else getattr(
            self.dataset, '_original_input_data', None)
        if sdata is None:
            raise ValueError("no SpatialData given; pass sdata= or run compute_vf "
                             "with a SpatialData input so it is remembered")
        fm = getattr(self.dataset, 'filtered_celltype_maps', None)
        if fm is None:
            raise ValueError("no filtered cell-type map; run filter_celltypemaps first")
        three_d = (getattr(self, 'n_dims', None) == 3 and z is None
                   and np.ndim(fm) == 3 and np.shape(fm)[2] > 1)
        if three_d and max_area_um2:
            raise ValueError("max_area_um2 is the 2D bound; a 3D run caps the volume — pass "
                             "max_volume_um3=<um^3> instead")
        if not three_d and max_volume_um3:
            raise ValueError("max_volume_um3 is the 3D bound; this run is 2D — pass "
                             "max_area_um2=<um^2> instead")
        if three_d:
            return self._segment_cells_3d(
                sdata, nuclei, simplify=simplify, min_area_um2=min_area_um2,
                max_volume_um3=max_volume_um3, exclude_celltypes=exclude_celltypes,
                transcripts=transcripts, shapes_key=shapes_key, shapes_3d_key=shapes_3d_key,
                labels_3d_key=labels_3d_key, table_key=table_key, column=column,
                nuclei_frame=nuclei_frame, tile=tile, pad_um=pad_um, nuclei_z=nuclei_z)
        return self._segment_cells_2d(
            sdata, nuclei, z=z, simplify=simplify, min_area_um2=min_area_um2,
            max_area_um2=max_area_um2, exclude_celltypes=exclude_celltypes,
            transcripts=transcripts, bins=bins, shapes_key=shapes_key, table_key=table_key,
            column=column, nuclei_frame=nuclei_frame)

    def _segment_cells_3d(self, sdata, nuclei, *, simplify, min_area_um2, max_volume_um3,
                          exclude_celltypes, transcripts,
                          shapes_key, shapes_3d_key, labels_3d_key, table_key, column,
                          nuclei_frame, tile, pad_um, nuclei_z=None):
        """The 3D path of :meth:`segment_cells`."""
        import geopandas as gpd
        from joblib import Parallel, delayed
        from scipy.sparse import csr_matrix
        from spatialdata.models import Labels3DModel, ShapesModel, TableModel
        from spatialdata.transformations import Affine

        data = np.asarray(self.dataset.filtered_celltype_maps)          # (x, y, z)
        X, Y, Zall = data.shape
        zg = self.dataset.zarr_group
        sd_um = float(zg.attrs.get("sampling_distance",
                      (zg["vf_params"][0] if "vf_params" in zg else 1.0)))
        ox, oy = (float(v) for v in self.dataset._celltype_map_origin(sdata))
        off = (np.asarray(zg["coordinate_offset"][:], dtype=float)
               if "coordinate_offset" in zg else np.zeros(3))
        oz = float(off[2]) if off.size > 2 else 0.0
        tx_key = self._pick_element(getattr(sdata, 'points', {}), transcripts,
                                    kind='points', param='transcripts', required=False)

        # ---- every layer of the map: which voxels are tissue is the filter's decision
        z0, z1 = 0, Zall - 1
        slab = np.ascontiguousarray(data)
        del data
        Z = Zall
        names, ct_colors = self._celltype_names_colors()
        K = len(names)

        # ---- markers: where the water is poured -- each nucleus centroid, one voxel on its own layer
        xy_um = self._nuclei_to_um(sdata, nuclei, frame=nuclei_frame)
        vx = np.rint((xy_um[:, 0] - ox) / sd_um).astype(np.int64)
        vy = np.rint((xy_um[:, 1] - oy) / sd_um).astype(np.int64)
        inside = (vx >= 0) & (vx < X) & (vy >= 0) & (vy < Y)
        vz = np.full(len(vx), -1, dtype=np.int64)
        own = np.zeros(len(vx), dtype=bool)
        if nuclei_z is not None:
            zz = np.asarray(nuclei_z, dtype=np.float64).ravel()
            if zz.shape[0] != len(vx):
                raise ValueError(f"nuclei_z has {zz.shape[0]} values for {len(vx)} nuclei")
            has = inside & np.isfinite(zz)
            k = np.rint((zz[has] - oz) / sd_um).astype(np.int64)
            ok = (k >= 0) & (k < Zall)
            idx = np.flatnonzero(has)[ok]
            lab_ok = slab[vx[idx], vy[idx], k[ok]] >= 0
            vz[idx[lab_ok]] = k[ok][lab_ok]
            own[idx[lab_ok]] = True
        # the rest: the labelled voxel of the column nearest the middle of its labelled run
        rest = np.flatnonzero(inside & ~own)
        n_mid = 0
        for c0 in range(0, len(rest), 200_000):
            r = rest[c0:c0 + 200_000]
            col = slab[vx[r], vy[r], :] >= 0                          # (n, Z)
            anyc = col.any(1)
            lo = col.argmax(1)
            hi = Zall - 1 - col[:, ::-1].argmax(1)
            m = (lo + hi) // 2
            dz = np.where(col, np.abs(np.arange(Zall)[None, :] - m[:, None]), Zall + 1)
            vz[r[anyc]] = dz.argmin(1)[anyc]
            n_mid += int(anyc.sum())
        cts = np.full(len(vx), -1, dtype=np.int32)
        ok = inside & (vz >= 0)
        cts[ok] = slab[vx[ok], vy[ok], vz[ok]]
        use = ok & (cts >= 0) & (cts < K)
        if not use.any():
            raise ValueError("no nucleus falls on a labelled voxel of the cell-type map on any layer — "
                             "check that nuclei and the map share a coordinate system")
        ex_ids = self._excluded_celltype_ids(exclude_celltypes, names)
        seed_use = use & ~np.isin(cts, list(ex_ids)) if ex_ids else use
        if not seed_use.any() and ex_ids:
            raise ValueError("every nucleus sits on an excluded cell type; nothing would be segmented")
        seeds_xyz = np.c_[vx[seed_use], vy[seed_use], vz[seed_use]]
        seeds_ct = cts[seed_use]
        seeds_id = np.arange(1, int(seed_use.sum()) + 1, dtype=np.int32)

        # ---- 3D watershed in xy tiles
        pad = int(np.ceil(float(pad_um) / sd_um))
        tile = int(tile)
        jobs = [(x0, min(X, x0 + tile), y0, min(Y, y0 + tile))
                for x0 in range(0, X, tile) for y0 in range(0, Y, tile)]
        labels = np.zeros((X, Y, Z), dtype=np.int32)
        for core, x0, y0 in Parallel(n_jobs=max(1, min(int(self.ncores), len(jobs))))(
                delayed(_watershed3d_tile)(slab, seeds_xyz, seeds_ct, seeds_id, *j, pad, ex_ids)
                for j in jobs):
            labels[x0:x0 + core.shape[0], y0:y0 + core.shape[1], :] = core
        slab_ex = slab if ex_ids else None       # the excluded types are read off it below
        del slab

        # ---- cap each cell's VOLUME: the seed voxel is the nearest one to itself, so what a cell
        # loses is its outside (see _cap_label_volume)
        n_lab = int(seeds_id.max()) + 1
        seed_xyz = np.zeros((n_lab, 3), np.int64)
        seed_xyz[seeds_id] = seeds_xyz
        n_capped = len(_cap_label_volume(labels, seed_xyz, sd_um ** 3, max_volume_um3))

        # ---- the excluded cell types: one object per connected piece, after the cap so no bound of
        # the segmented cells can reach them (ndimage.label walks the whole volume once per type)
        blob_ct, n_blob = {}, 0
        if ex_ids:
            nxt = int(labels.max()) + 1
            for c in sorted(ex_ids):
                n = _blob_labels(slab_ex == c, nxt, labels)
                for i in range(n):
                    blob_ct[nxt + i] = int(c)
                nxt += n
                n_blob += n
                if self.verbose:
                    print(f"[segment_cells] {names[c] if c < len(names) else c}: "
                          f"{n} connected bodies, not segmented", flush=True)
            del slab_ex
        n_lab = max(n_lab, int(labels.max()) + 1)

        # ---- per-cell size, planes and centroid; drop cells whose largest layer is too small
        vol = np.zeros(n_lab, np.int64)
        cover = np.zeros(n_lab, np.int64)
        n_z = np.zeros(n_lab, np.int16)
        cover_k = np.zeros(n_lab, np.int64)
        sx, sy, sz = np.zeros(n_lab), np.zeros(n_lab), np.zeros(n_lab)
        xi = np.broadcast_to(np.arange(X, dtype=np.float64)[:, None], (X, Y)).ravel()
        yi = np.broadcast_to(np.arange(Y, dtype=np.float64)[None, :], (X, Y)).ravel()
        for k in range(Z):
            L = labels[:, :, k].ravel()
            c = np.bincount(L, minlength=n_lab)
            vol += c
            cover_k[c > cover] = k
            cover = np.maximum(cover, c)
            n_z += (c > 0)
            sx += np.bincount(L, weights=xi, minlength=n_lab)
            sy += np.bincount(L, weights=yi, minlength=n_lab)
            sz += c * k
        is_blob = np.zeros(n_lab, bool)
        if blob_ct:
            is_blob[list(blob_ct)] = True
        keep = (cover > 0) & ((cover * sd_um ** 2 >= (min_area_um2 or 0)) | is_blob)
        keep[0] = False
        drop_small = int(((cover > 0) & ~keep)[1:].sum())
        if drop_small:
            labels[~keep[labels] & (labels > 0)] = 0
        present = np.flatnonzero(keep)
        den = np.maximum(vol, 1)
        cen = np.c_[ox + sx / den * sd_um, oy + sy / den * sd_um, oz + (z0 + sz / den) * sd_um]

        # ---- outlines on every layer, and each cell's on its own seed layer
        planes = Parallel(n_jobs=max(1, min(int(self.ncores), Z)))(
            delayed(_plane_polygons)(labels[:, :, k], sd_um, ox, oy, simplify) for k in range(Z))
        seed_k = np.full(n_lab, -1, dtype=np.int64)
        seed_k[seeds_id] = seeds_xyz[:, 2] - z0
        seed_k[is_blob] = cover_k[is_blob]
        recs, mid, area_mid = [], [], np.full(n_lab, np.nan)
        for k, polys in enumerate(planes):
            for lb, g in polys:
                if lb < n_lab and keep[lb]:
                    recs.append((lb, z0 + k, g))
                    if k == seed_k[lb]:
                        area_mid[lb] = g.area
                        mid.append((lb, z0 + k, g))
        gdf3 = gpd.GeoDataFrame(
            {"cell_id": [f"cell_{r[0]}" for r in recs],
             "z_index": np.array([r[1] for r in recs], dtype=np.int32),
             "z_um": np.array([oz + r[1] * sd_um for r in recs], dtype=np.float32)},
            geometry=[r[2] for r in recs],
            index=pd.Index([f"cell_{r[0]}_z{r[1]}" for r in recs], name="plane_cell_id"))
        gdf2 = gpd.GeoDataFrame(geometry=[r[2] for r in mid],
                                index=pd.Index([f"cell_{r[0]}" for r in mid], name="cell_id"))

        # ---- cell x gene counts by 3D voxel
        Xc, var_names, n_in = self._count_transcripts_per_cell_3d(
            sdata, labels, present, sd_um, ox, oy, oz, z0, transcripts=tx_key)

        ctid = np.full(n_lab, -1, dtype=np.int32)
        ctid[seeds_id] = seeds_ct
        for lb, c in blob_ct.items():
            ctid[lb] = c
        n_nuclei = np.zeros(n_lab, np.int32)
        n_nuclei[seeds_id] = 1                        # one nucleus per segmented cell, by construction
        if ex_ids:
            exn = use & np.isin(cts, list(ex_ids))    # the nuclei inside the untouched bodies
            if exn.any():
                lb = labels[vx[exn], vy[exn], vz[exn] - z0]
                n_nuclei += np.bincount(lb[lb > 0], minlength=n_lab)[:n_lab].astype(np.int32)
        idx = pd.Index([f"cell_{i}" for i in present])
        obs = pd.DataFrame({
            column: pd.Categorical.from_codes(ctid[present], categories=list(names)),
            "celltype_id": ctid[present],
            "label_id": present.astype(np.int32),
            "cell_id": idx.to_numpy(),
            "volume_um3": (vol[present] * sd_um ** 3).astype(np.float32),
            "n_z": n_z[present],
            "max_plane_area_um2": (cover[present] * sd_um ** 2).astype(np.float32),
            "area_um2": area_mid[present].astype(np.float32),
            "seed_z_um": (oz + (z0 + seed_k[present]) * sd_um).astype(np.float32),
            "segmented": ~is_blob[present],
            "n_nuclei": n_nuclei[present],
            "region": pd.Categorical([labels_3d_key] * len(present)),
        }, index=idx)
        if Xc is None:
            Xc, var_names = csr_matrix((len(present), 0), dtype=np.float32), []
        adata = ad.AnnData(X=Xc, obs=obs, var=pd.DataFrame(index=pd.Index(var_names, name=None)))
        adata.obsm['spatial'] = cen[present].astype(np.float32)
        adata.uns[f'{column}_colors'] = list(ct_colors)
        adata.uns['cellman_celltype_names'] = list(names)
        adata.uns['segment_cells'] = dict(
            n_dims=3, z_layers=[int(z0), int(z1)],
            z_um=[float(oz + z0 * sd_um), float(oz + z1 * sd_um)],
            seeds="per nucleus: its own z (nuclei_z) when labelled there, else its column's "
                  "labelled voxel nearest the tissue midline",
            n_seed_own_z=int((use & own).sum()), n_seed_column=int((use & ~own).sum()),
            obsm_spatial="voxel centroid, µm (x, y, z)", area_um2="area on the cell's seed layer",
            shapes_2d="each cell's outline on its seed layer",
            excluded_celltypes=[names[i] if i < len(names) else int(i) for i in sorted(ex_ids)],
            n_whole_bodies=int(n_blob))

        # ---- elements, in the frame the nuclei were read in
        A = self._cells_global_affine(sdata)
        cells_cs = self._cells_coordinate_system(sdata)
        a, b, c = A[0]
        d, e, f = A[1]
        M4 = np.array([[sd_um, 0.0, 0.0, oz + z0 * sd_um],
                       [0.0, e * sd_um, d * sd_um, d * ox + e * oy + f],
                       [0.0, b * sd_um, a * sd_um, a * ox + b * oy + c],
                       [0.0, 0.0, 0.0, 1.0]])
        tf_xy = Affine(A, input_axes=("x", "y"), output_axes=("x", "y"))
        for key, store in ((shapes_key, sdata.shapes), (shapes_3d_key, sdata.shapes),
                           (table_key, sdata.tables), (labels_3d_key, sdata.labels)):
            if key in store:
                del store[key]
        sdata.labels[labels_3d_key] = Labels3DModel.parse(
            np.ascontiguousarray(labels.transpose(2, 1, 0)), dims=("z", "y", "x"),
            chunks=(Z, 1024, 1024),
            transformations={cells_cs: Affine(M4, input_axes=("z", "y", "x"),
                                              output_axes=("z", "y", "x"))})
        sdata.shapes[shapes_3d_key] = ShapesModel.parse(gdf3, transformations={cells_cs: tf_xy})
        sdata.shapes[shapes_key] = ShapesModel.parse(gdf2, transformations={cells_cs: tf_xy})
        sdata.tables[table_key] = TableModel.parse(
            adata, region=labels_3d_key, region_key="region", instance_key="label_id")

        n_bg = int((inside & ~use).sum())
        print(f"[segment_cells] labels['{labels_3d_key}'] {len(present)} cells in 3D from "
              f"{int(use.sum())}/{len(vx)} nuclei (seeds: {int((use & own).sum())} at the nucleus's own z, "
              f"{int((use & ~own).sum())} on the column's tissue midline; {n_bg} on a background column "
              f"ignored, {int((~inside).sum())} outside map"
              + (f", {drop_small} below {min_area_um2}µm² on every layer" if drop_small else "")
              + (f", {n_capped} trimmed to {max_volume_um3}µm³" if n_capped else "")
              + (f"; {n_blob} whole bodies of "
                 f"{', '.join(names[i] if i < len(names) else str(i) for i in sorted(ex_ids))}"
                 if n_blob else "")
              + f") @ all {Z} layers, {sd_um}µm/voxel")
        print(f"[segment_cells] shapes['{shapes_3d_key}'] {len(gdf3)} outlines on {Z} layers; "
              f"shapes['{shapes_key}'] {len(gdf2)} on each cell's seed layer")
        print(f"[segment_cells] tables['{table_key}'] {adata.shape} ({n_in:,} transcripts assigned, "
              f"median volume {float(np.median(obs['volume_um3'])):.0f}µm³)")
        return sdata

    def _count_transcripts_per_cell_3d(self, sdata, labels, lab_of_cell, sd_um, ox, oy, oz, z0,
                                       transcripts=None):
        """:meth:`_count_transcripts_per_cell` on an ``(x, y, z)`` label volume.

        A transcript is counted in the cell of the voxel it falls in; its z is clamped into the
        segmented layers, so a molecule just above or below them joins its column's cell."""
        from scipy.sparse import coo_matrix
        if transcripts is None:
            return None, [], 0
        self._m(f"segment_cells: counting points['{transcripts}'] in 3D")
        pts = sdata.points[transcripts]
        gene_col = next((c for c in ('gene', 'gene_symbol', 'feature_name', 'target')
                         if c in pts.columns), None)
        if gene_col is None:
            raise ValueError(f"gene column not found in points['{transcripts}']: {list(pts.columns)}")
        cols = ['x', 'y', 'z', gene_col] + (['is_gene'] if 'is_gene' in pts.columns else [])
        pts = pts[cols]
        if hasattr(pts, 'compute'):
            pts = pts.compute()
        if 'is_gene' in pts.columns:
            pts = pts[pts['is_gene'] == True]                      # noqa: E712
        X, Y, Z = labels.shape
        vx = np.rint((pts['x'].to_numpy(dtype=np.float64) - ox) / sd_um).astype(np.int64)
        vy = np.rint((pts['y'].to_numpy(dtype=np.float64) - oy) / sd_um).astype(np.int64)
        vz = np.clip(np.rint((pts['z'].to_numpy(dtype=np.float64) - oz) / sd_um).astype(np.int64) - z0,
                     0, Z - 1)
        ok = (vx >= 0) & (vx < X) & (vy >= 0) & (vy < Y)
        lab = np.zeros(len(vx), dtype=np.int64)
        lab[ok] = labels[vx[ok], vy[ok], vz[ok]]
        n_lab = int(max(labels.max(), lab_of_cell.max() if len(lab_of_cell) else 0)) + 1
        row_of_lab = np.full(n_lab, -1, dtype=np.int64)
        row_of_lab[lab_of_cell] = np.arange(len(lab_of_cell))
        rows = np.where(lab > 0, row_of_lab[np.clip(lab, 0, None)], -1)
        hit = rows >= 0
        genes = pts[gene_col].astype(str).to_numpy()[hit]
        var_names, gcols = np.unique(genes, return_inverse=True)
        Xm = coo_matrix((np.ones(hit.sum(), dtype=np.float32), (rows[hit], gcols)),
                        shape=(len(lab_of_cell), len(var_names))).tocsr()
        return Xm, list(var_names), int(hit.sum())

    def _segment_cells_2d(self, sdata, nuclei, *, z, simplify, min_area_um2, max_area_um2,
                          exclude_celltypes, transcripts, bins,
                          shapes_key, table_key, column, nuclei_frame):
        """One plane of the cell-type map cut into cells; the 2D path of :meth:`segment_cells`."""
        import geopandas as gpd
        from skimage.segmentation import watershed
        from spatialdata import to_polygons
        from spatialdata.models import Labels2DModel, ShapesModel, TableModel
        from spatialdata.transformations import Affine

        if nuclei is None:
            raise ValueError("nuclei= is required (array of µm coords, or a "
                             "shapes/points element key)")
        sdata = sdata if sdata is not None else getattr(
            self.dataset, '_original_input_data', None)
        if sdata is None:
            raise ValueError("no SpatialData given; pass sdata= or run compute_vf "
                             "with a SpatialData input so it is remembered")

        ctmap, sd_um, ox, oy, z = self._celltype_map_affine(sdata, z)
        H, W = ctmap.shape

        xy_um = self._nuclei_to_um(sdata, nuclei, frame=nuclei_frame)
        vox = np.rint((xy_um - (ox, oy)) / sd_um).astype(int)
        inside = ((vox[:, 0] >= 0) & (vox[:, 0] < W) &
                  (vox[:, 1] >= 0) & (vox[:, 1] < H))
        cts = np.full(len(vox), -1, dtype=np.int32)
        cts[inside] = ctmap[vox[inside, 1], vox[inside, 0]]
        use = inside & (cts >= 0)                    # background nuclei are ignored
        if not use.any():
            raise ValueError("no nucleus falls on a labelled voxel of the cell-type "
                             "map — check that nuclei and the map share a coordinate system")

        names, ct_colors = self._celltype_names_colors()
        ex_ids = self._excluded_celltype_ids(exclude_celltypes, names)
        seed_use = use & ~np.isin(cts, list(ex_ids)) if ex_ids else use
        if not seed_use.any() and ex_ids:
            raise ValueError("every nucleus sits on an excluded cell type; nothing would be segmented")

        seeds = np.zeros(ctmap.shape, dtype=np.int32)
        seeds[vox[seed_use, 1], vox[seed_use, 0]] = np.arange(1, int(seed_use.sum()) + 1)

        labels = np.zeros(ctmap.shape, dtype=np.int32)
        seed_ct = {}
        for c in np.unique(cts[seed_use]):
            mk = np.where(np.isin(seeds, seeds[vox[seed_use & (cts == c), 1],
                                               vox[seed_use & (cts == c), 0]]), seeds, 0)
            dist = ndimage.distance_transform_edt(mk == 0)
            lab_c = watershed(dist, mk, mask=(ctmap == c))
            hit = lab_c > 0
            labels[hit] = lab_c[hit]
            for s in np.unique(lab_c[hit]):
                seed_ct[int(s)] = int(c)

        # cap the flood: keep each cell's voxels nearest its own seed, give the rest back to
        # the background. The seed is at distance 0, so the nucleus the water was poured on is
        # never what gets trimmed (see _cap_label_area).
        n_lab = int(seed_use.sum()) + 1
        seed_a, seed_b = np.zeros(n_lab, np.int64), np.zeros(n_lab, np.int64)
        seed_a[1:] = vox[seed_use, 1]                      # axis 0 of labels is y
        seed_b[1:] = vox[seed_use, 0]
        n_capped = len(_cap_label_area(labels, seed_a, seed_b, sd_um ** 2, max_area_um2))

        # the excluded cell types: one object per connected piece, after the cap so no bound of the
        # segmented cells can reach them
        blob_ct, n_blob = {}, 0
        if ex_ids:
            nxt = int(labels.max()) + 1
            for c in sorted(ex_ids):
                n = _blob_labels(ctmap == c, nxt, labels)
                for i in range(n):
                    blob_ct[nxt + i] = int(c)
                    seed_ct[nxt + i] = int(c)
                nxt += n
                n_blob += n
                if self.verbose:
                    print(f"[segment_cells] {names[c] if c < len(names) else c}: "
                          f"{n} connected bodies, not segmented", flush=True)
        n_nuclei = {}
        if ex_ids:
            exn = use & np.isin(cts, list(ex_ids))
            if exn.any():
                lb = labels[vox[exn, 1], vox[exn, 0]]
                for v, n in zip(*np.unique(lb[lb > 0], return_counts=True)):
                    n_nuclei[int(v)] = int(n)

        # Raster -> vector. The watershed IS already the segmentation: every voxel
        # belongs to exactly one cell and the borders run along voxel EDGES. All
        # that is left is transcribing that partition faithfully, which means the
        # polygon has to follow the voxel BOUNDARY. Tracing the boundary voxels'
        # CENTRES instead (what cv2.findContours hands back) insets every cell by
        # half a voxel and so erases ~perimeter/2 of its area — a 3x3-voxel cell
        # keeps 4 of 9 — while 1-2 voxel cells yield fewer than 3 points and cannot
        # be closed into a ring at all, silently vanishing. At the cell sizes a
        # 2 µm/px map produces (median ~9 voxels) that costs half the tissue.
        # to_polygons marching-squares the 0.5 level, i.e. between the voxels, so
        # areas survive and every label becomes exactly one cell.
        el = Labels2DModel.parse(labels, chunks=labels.shape)
        poly = to_polygons(el)
        # One chunk on purpose: the chunked path vectorizes each piece on its own
        # and dissolves them afterwards, which insets — and so bleeds area — along
        # every seam a cell happens to straddle.
        #
        # to_polygons works in voxel index space. These shapes are written into
        # 'global' µm under Identity (see below), so bake the map's affine into the
        # coordinates instead of carrying it as a transformation: obs['area_um2']
        # is read straight off the geometry and would otherwise come out in px².
        poly.geometry = [_nest_rings(g) for g in poly.geometry]
        poly.geometry = poly.geometry.affine_transform([sd_um, 0, 0, sd_um, ox, oy])
        if simplify:
            poly.geometry = poly.geometry.simplify(simplify)
        recs, drop_small = [], 0
        for lb, g in zip(poly.index.to_numpy(), poly.geometry.to_numpy()):
            if g is None or g.is_empty:
                continue
            if min_area_um2 and g.area < min_area_um2 and int(lb) not in blob_ct:
                drop_small += 1
                continue
            ct_id = seed_ct.get(int(lb), -1)
            recs.append((f"cell_{int(lb)}", g, ct_id,
                         names[ct_id] if 0 <= ct_id < len(names) else "Unknown",
                         int(lb) not in blob_ct, n_nuclei.get(int(lb), 1)))

        idx = pd.Index([r[0] for r in recs], name="cell_id")
        gdf = gpd.GeoDataFrame(geometry=[r[1] for r in recs], index=idx)
        cen = gdf.geometry.centroid
        # Pin the categories to the celltype ID order (NOT the alphabetical order
        # pandas would infer) so the categorical is index-aligned with the palette
        # the RGB map was rendered in -- same contract as the downsampled table.
        ct_cats, ct_cols = list(names), list(ct_colors)
        if any(r[2] < 0 or r[2] >= len(names) for r in recs) and "Unknown" not in ct_cats:
            ct_cats.append("Unknown")
            ct_cols.append("#bbbbbb")
        obs = pd.DataFrame({
            column: pd.Categorical([r[3] for r in recs], categories=ct_cats),
            "celltype_id": np.array([r[2] for r in recs], dtype=np.int32),
            "area_um2": gdf.geometry.area.to_numpy(),
            "segmented": np.array([r[4] for r in recs], dtype=bool),
            "n_nuclei": np.array([r[5] for r in recs], dtype=np.int32),
            "region": pd.Categorical([shapes_key] * len(recs)),
            "cell_id": idx.to_numpy(),
        }, index=idx)

        # cell x gene counts: every transcript lands in exactly one label voxel, so
        # the raster does the assignment in O(n_transcripts) — no point-in-polygon.
        lab_of_cell = np.array([int(r[0].split('_')[1]) for r in recs], dtype=np.int64)
        X, var_names, n_in = self._count_transcripts_per_cell(
            sdata, labels, lab_of_cell, sd_um, ox, oy, transcripts=transcripts)
        unit = 'transcripts'
        if X is None:
            # array platform: no molecule coordinates, aggregate the bin table instead
            X, var_names, n_in = self._count_bins_per_cell(
                sdata, labels, lab_of_cell, sd_um, ox, oy, bins=bins)
            unit = 'bin counts'

        for key, store in ((shapes_key, sdata.shapes), (table_key, sdata.tables)):
            if key in store:
                del store[key]
        # The polygons are in the cell-type map's µm frame, the frame _nuclei_to_um read
        # the markers in. That frame's own transformation places them in 'global': the
        # nuclei element's (µm -> image pixels on Xenium / MERSCOPE) when the markers were
        # read intrinsically, the identity when they were read in 'global' (Visium HD).
        sdata.shapes[shapes_key] = ShapesModel.parse(
            gdf, transformations={self._cells_coordinate_system(sdata): Affine(self._cells_global_affine(sdata),
                                                   input_axes=("x", "y"), output_axes=("x", "y"))})
        adata = ad.AnnData(X=X, obs=obs,
                           var=pd.DataFrame(index=pd.Index(var_names, name=None)))
        adata.obsm['spatial'] = np.c_[cen.x.to_numpy(), cen.y.to_numpy()].astype(np.float32)
        # index-aligned with obs[column].cat.categories, so render_shapes(color=column)
        # picks the very colours the celltype map was drawn in
        adata.uns[f'{column}_colors'] = ct_cols
        adata.uns['cellman_celltype_names'] = list(names)
        sdata.tables[table_key] = TableModel.parse(
            adata, region=shapes_key, region_key="region", instance_key="cell_id")

        n_bg = int((inside & (cts < 0)).sum())
        print(f"[segment_cells] shapes['{shapes_key}'] {len(gdf)} cells "
              f"from {int(use.sum())}/{len(vox)} nuclei "
              f"({n_bg} on background ignored, {int((~inside).sum())} outside map"
              + (f", {drop_small} below {min_area_um2}µm²" if drop_small else "")
              + (f", {n_capped} trimmed to {max_area_um2}µm²" if n_capped else "")
              + (f"; {n_blob} whole bodies of "
                 f"{', '.join(names[i] if i < len(names) else str(i) for i in sorted(ex_ids))}"
                 if n_blob else "")
              + f") @ z={z}, {sd_um}µm/px")
        print(f"[segment_cells] tables['{table_key}'] {adata.shape} "
              f"({n_in:,} {unit} assigned, "
              f"median {float(np.median(np.asarray(X.sum(1)).ravel())):.0f}/cell)"
              if X is not None else
              f"[segment_cells] tables['{table_key}'] {adata.shape} — no counts source: "
              f"sdata has no points element and no bin table was found "
              f"(pass transcripts=<points key> or bins=<AnnData/GridDataset/table key>)")
        return sdata


    def _map_celltypes_gpu(self, sdata, centroids=None, element_name=None,
                          exclude_gene_indices=None, marker_genes_path=None,
                          h5ad=None, column='cellman_annotation', map_kwargs=None,
                          bandwidth=2.5, sampling_distance=1.0,
                          prune_coefficient=4.3, kernel='gaussian',
                          size_after_normalization=10,
                          log_transform=True, overwrite=True,
                          gpu_device='cuda', query_subchunk_size=None,
                          gpu_target_bytes=int(1e9), save_vf=False, method='corr'):
        """GPU equivalent of :meth:`map_celltypes_cpu` (single machine).

        Reads zarr state, KDE-computes the vector field chunk-by-chunk on
        a CUDA device (Triton kernel + torch GEMM Pearson correlation),
        and writes ``vf_norm``, ``celltype_maps``, ``max_correlations``.

        Numerical equivalence vs the CPU path: not bit-identical; expected
        per-voxel correlation diff <1e-5, with minor label flips possible
        at decision boundaries (much less than 0.1% of voxels for
        well-separated centroids).

        Parameters
        ----------
        gpu_device : str, default ``'cuda'``
        query_subchunk_size : int, optional
            Voxel queries per Triton launch; auto-sized from
            ``gpu_target_bytes / (n_genes_use * 4)`` when None.
        gpu_target_bytes : int, default 1 GB
            Target VRAM budget for the per-sub-chunk output buffer. Transcript regime
            only — the count regime sizes its own field band (see
            :func:`ssaft._lda.map_lda`) and takes its budget through
            ``map_kwargs={'gpu_target_bytes': ...}``.
        Other parameters are identical to :meth:`map_celltypes_cpu`.

        Built on a producer/consumer split: KDE field production
        (:meth:`_kde_field_iter_gpu`) is separated from the shared Pearson-max
        mapper (``_pearson_max_gpu``, identical to the count path), with the full
        normalized field optionally teed to a tensorstore writer (``save_vf``) in
        the same streaming pass. Correlations are computed in matmul batches, so
        they can differ from an unbatched reduction by <1e-5.

        ``method='gnb'`` has no GPU-KDE kernel, so it runs the CPU KDE producer while the
        GaussianNB classifier itself still predicts on the GPU (cuML); only the KDE step is
        CPU here.
        """
        if method == 'gnb':
            # no GPU KDE kernel for the gnb producer; lda stays here (its consumer is a
            # matmul, so it keeps the fast GPU KDE).
            return self._map_celltypes_cpu(
                sdata, centroids=centroids, element_name=element_name,
                exclude_gene_indices=exclude_gene_indices,
                marker_genes_path=marker_genes_path, h5ad=h5ad, column=column,
                map_kwargs=map_kwargs, bandwidth=bandwidth,
                sampling_distance=sampling_distance, prune_coefficient=prune_coefficient,
                kernel=kernel,
                size_after_normalization=size_after_normalization,
                log_transform=log_transform, overwrite=overwrite, save_vf=save_vf,
                method='gnb', gpu_device=gpu_device, force_cpu=False)
        return self._transcript_gpu(
            sdata, do_map=True, save_vf=save_vf,
            centroids=centroids, element_name=element_name,
            exclude_gene_indices=exclude_gene_indices,
            marker_genes_path=marker_genes_path, method=method,
            map_kwargs=map_kwargs, bandwidth=bandwidth,
            sampling_distance=sampling_distance, prune_coefficient=prune_coefficient,
            kernel=kernel,
            size_after_normalization=size_after_normalization,
            log_transform=log_transform, overwrite=overwrite,
            gpu_device=gpu_device, query_subchunk_size=query_subchunk_size,
            gpu_target_bytes=gpu_target_bytes,
        )

    map_celltypes_cpu = _map_celltypes_cpu
    map_celltypes_gpu = _map_celltypes_gpu

    def _prepare_transcript_field(self, sdata, *, centroids, element_name,
                                  exclude_gene_indices, bandwidth, sampling_distance,
                                  prune_coefficient, size_after_normalization,
                                  gpu_device, kernel='gaussian'):
        """Resolve everything the KDE producer/consumer need (grid, gene set,
        centroids, transcripts on GPU)."""
        import torch
        for key in ('cluster_labels', 'downsampled_vectors', 'normalized_vectors',
                    'centroids'):
            if key in self.dataset.zarr_group:
                setattr(self.dataset, key, self.dataset.zarr_group[key][:])
        produce_only = centroids is None and getattr(self.dataset, 'centroids', None) is None
        if produce_only:
            centroids = None
        elif centroids is None:
            centroids = np.asarray(self.dataset.centroids, dtype=np.float32)
        else:
            centroids = np.asarray(centroids, dtype=np.float32)

        if getattr(self.dataset, 'shape', None) is None or \
                not hasattr(self, 'locations') or self.locations is None:
            self._prepare_grid(
                sdata, element_name=element_name, bandwidth=bandwidth,
                sampling_distance=sampling_distance, prune_coefficient=prune_coefficient,
                size_after_normalization=size_after_normalization)
        vx, vy, vz = (int(self.dataset.shape[0]), int(self.dataset.shape[1]),
                      int(self.dataset.shape[2]))
        # Each kernel has its own scale (the support radius of a compact one); the
        # caller's ``bandwidth`` is gaussian-equivalent for all of them, so convert
        # here -- the grid dimensionality is only known now -- and let everything
        # downstream (norm_threshold, V_eff, vf_params) read the converted scale.
        from ._count import resolve_kernel, kernel_scale, kernel_peak
        _dim = 2 if vz == 1 else 3
        _kern = resolve_kernel(kernel)
        _bw_kde = kernel_scale(bandwidth, _dim, _kern)
        self.dataset.bandwidth = _bw_kde
        self.dataset.kde_kernel = _kern
        self.dataset.ndim = _dim
        self.dataset.expression_threshold = kernel_peak(_bw_kde, _dim, _kern)
        self.dataset.norm_threshold = self.dataset.expression_threshold * 2
        norm_threshold = float(self.dataset.norm_threshold)

        grouped = self.locations.groupby('gene', sort=True)
        grouped_map = {gene: loc for gene, loc in grouped}
        if getattr(self.dataset, 'genes', None) is None:
            genes = sorted(grouped_map.keys()); self.dataset.genes = genes
        else:
            genes = list(self.dataset.genes)
        if (not produce_only) and centroids.shape[1] != len(genes):
            raise ValueError(
                f'centroids gene dim ({centroids.shape[1]}) != gene count ({len(genes)})')
        shared_gene_indices = [i for i, g in enumerate(genes) if g in grouped_map]
        shared_genes = [genes[i] for i in shared_gene_indices]
        missing_genes = [g for g in genes if g not in grouped_map]
        if len(shared_genes) == 0:
            raise ValueError('No common genes between input data and centroids.')
        if missing_genes and self.verbose:
            print(f'map_celltypes: dropping {len(missing_genes)} missing genes '
                  f'(e.g. {missing_genes[:5]})')
        if produce_only:
            use_genes = list(shared_genes); centroids_use = None
        else:
            centroids_shared = centroids[:, shared_gene_indices]
            if exclude_gene_indices is not None:
                idx_to_local = {gi: li for li, gi in enumerate(shared_gene_indices)}
                exclude_shared = sorted(idx_to_local[i] for i in exclude_gene_indices
                                        if i in idx_to_local)
            else:
                exclude_shared = None
            if exclude_shared:
                keep_in_shared = [i for i in range(len(shared_genes))
                                  if i not in set(exclude_shared)]
                use_genes = [shared_genes[i] for i in keep_in_shared]
                centroids_use = centroids_shared[:, keep_in_shared]
            else:
                use_genes = list(shared_genes); centroids_use = centroids_shared
        n_genes_use = len(use_genes)

        all_x, all_y, all_z, all_g = [], [], [], []
        for gidx_use, g in enumerate(use_genes):
            locs = np.asarray(grouped_map[g], dtype=np.float64)
            if locs.shape[0] == 0:
                continue
            all_x.append(locs[:, 0]); all_y.append(locs[:, 1])
            all_z.append(locs[:, 2] if locs.shape[1] >= 3
                         else np.zeros(locs.shape[0], dtype=np.float64))
            all_g.append(np.full(locs.shape[0], gidx_use, dtype=np.int64))
        if all_x:
            loc_x_np = np.concatenate(all_x); loc_y_np = np.concatenate(all_y)
            loc_z_np = np.concatenate(all_z); gi_np = np.concatenate(all_g)
        else:
            loc_x_np = loc_y_np = loc_z_np = np.zeros(0, np.float64)
            gi_np = np.zeros(0, np.int64)
        cx, n_chunk = self.calc_chunk(vx, vy, vz, n_genes_use, mode='map_celltypes_cpu')
        if cx is None or cx <= 0:
            raise RuntimeError('Cannot determine chunk size.')
        # Large transcript sets stay on the host: uploading all of them (MERSCOPE: ~0.5 billion,
        # 16 GB as float64 + gene index) left no room on the card for the rest of the map. Each
        # x-slab then uploads only the transcripts within its reach, found through an x-sorted
        # index and put back in their original order, so the KDE sums them in the same order.
        host = None
        if len(loc_x_np) * 32 > int(os.environ.get('SSAFT_MAP_HOST_LOCS_BYTES', 2 << 30)):
            _order = np.argsort(loc_x_np, kind='stable')
            host = dict(x=np.ascontiguousarray(loc_x_np), y=np.ascontiguousarray(loc_y_np),
                        z=np.ascontiguousarray(loc_z_np), gi=np.ascontiguousarray(gi_np),
                        order=_order, xs=loc_x_np[_order])
            _dev = lambda a, dt: torch.zeros(0, dtype=dt, device=gpu_device)  # noqa: E731
        else:
            _dev = lambda a, dt: torch.as_tensor(a, dtype=dt, device=gpu_device)  # noqa: E731
        return dict(
            vx=vx, vy=vy, vz=vz, dim=_dim,
            sd_um=float(sampling_distance), bandwidth=float(_bw_kde),
            kernel=_kern, prune_coefficient=float(prune_coefficient),
            norm_threshold=norm_threshold,
            size_after_normalization=float(size_after_normalization),
            n_genes_use=int(n_genes_use), genes=genes, use_genes=use_genes,
            centroids_use=centroids_use, cx=int(cx), n_chunk=int(n_chunk),
            loc_x_t=_dev(loc_x_np, torch.float64), loc_y_t=_dev(loc_y_np, torch.float64),
            loc_z_t=_dev(loc_z_np, torch.float64), gi_t=_dev(gi_np, torch.int64),
            host_locs=host, gpu_device=gpu_device,
        )

    def _kde_field_iter_gpu(self, st, *, chunk_start, chunk_end, log_transform,
                            want_full_field, query_subchunk_size, gpu_target_bytes,
                            on_batch=None):
        """PRODUCER. Yield, per output x-slab, a dict with the vf_norm plane, the
        KEPT (vf_norm>threshold) voxels' local positions + normalized·log1p gene
        vectors (GPU tensor), and the full normalized field (only if
        ``want_full_field``).

        ``on_batch(vecs, kx, ky, kz, vf_norm_kept, x0)`` streams instead of accumulating.
        Without it a chunk's kept vectors are held on the device and then concatenated,
        which peaks at TWICE their size — the concatenation alone asked for another
        9 GB of a 31 GB card on a whole-transcriptome MERFISH store and died there.
        With it each sub-chunk is handed over and freed, so the peak is one sub-chunk
        (``query_subchunk_size`` rows) instead of a whole chunk, and the yielded
        ``vecs`` is ``None``."""
        from tqdm.auto import tqdm
        vx, vy, vz = st['vx'], st['vy'], st['vz']
        n_genes_use = st['n_genes_use']
        bandwidth = st['bandwidth']; prune_coefficient = st['prune_coefficient']
        kernel = st.get('kernel', 'gaussian')
        # a compact kernel's own support IS its radius; prune_coefficient cuts the others
        from ._count import kernel_reach
        reach = kernel_reach(bandwidth, kernel, prune_coefficient)
        cx = st['cx']; n_chunk = st['n_chunk']
        if query_subchunk_size is None:
            query_subchunk_size = max(1024, int(gpu_target_bytes // (n_genes_use * 4)))
        cs = 0 if chunk_start is None else chunk_start
        ce = n_chunk if chunk_end is None else chunk_end
        # ONE bar for the whole call, like the CPU path and the count backend, with the
        # chunk in the postfix. A bar per chunk (leave=True) printed n_chunk finished
        # lines and reset the ETA every few percent; the chunking is an internal memory
        # device, not something the caller is waiting on. Counted in voxels so the rate
        # is comparable across chunks of different widths.
        _total_vox = sum(min(vx, i * cx + cx) - i * cx for i in range(cs, ce)) * vy * vz
        kde_bar = tqdm(total=_total_vox, desc='KDE', unit='vox', unit_scale=True,
                       leave=True, dynamic_ncols=True) if self.verbose else None
        try:
            yield from self._kde_field_chunks_gpu(
                st, cs=cs, ce=ce, cx=cx, log_transform=log_transform,
                want_full_field=want_full_field,
                query_subchunk_size=query_subchunk_size, on_batch=on_batch,
                reach=reach, kernel=kernel, kde_bar=kde_bar)
        finally:
            if kde_bar is not None:
                kde_bar.close()

    def _kde_field_chunks_gpu(self, st, *, cs, ce, cx, log_transform, want_full_field,
                              query_subchunk_size, on_batch, reach, kernel, kde_bar):
        """The chunk loop of :meth:`_kde_field_iter_gpu`, split out only so the
        single progress bar can be closed in a ``finally`` around it."""
        import torch
        from ._mapping_gpu import _kde_setup_celllist, _kde_run_subchunk, _kde_nonzero_queries
        vx, vy, vz = st['vx'], st['vy'], st['vz']
        n_genes_use = st['n_genes_use']; sd_um = st['sd_um']
        bandwidth = st['bandwidth']; prune_coefficient = st['prune_coefficient']
        dim = st['dim']; norm_threshold = st['norm_threshold']
        san = st['size_after_normalization']; gpu_device = st['gpu_device']
        loc_x_t, loc_y_t, loc_z_t, gi_t = st['loc_x_t'], st['loc_y_t'], st['loc_z_t'], st['gi_t']
        n_chunk = st['n_chunk']
        for chunk_idx in range(cs, ce):
            x0 = chunk_idx * cx; x1 = min(vx, x0 + cx); local_cx = x1 - x0
            qx_grid = (np.arange(x0, x1) * sd_um).astype(np.float64)
            qy_grid = (np.arange(vy) * sd_um).astype(np.float64)
            qz_grid = (np.arange(vz) * sd_um).astype(np.float64)
            Qx, Qy, Qz = np.meshgrid(qx_grid, qy_grid, qz_grid, indexing='ij')
            qx_flat = Qx.ravel(); qy_flat = Qy.ravel(); qz_flat = Qz.ravel()
            N_total = int(qx_flat.size)

            vf_norm_chunk = np.zeros((local_cx, vy, vz), dtype=np.float32)
            full_chunk = (np.zeros((n_genes_use, local_cx, vy, vz), dtype=np.float32)
                          if want_full_field else None)
            keep_x, keep_y, keep_z, keep_vecs = [], [], [], []

            cqxmin = float(qx_grid.min()) if N_total > 0 else 0.0
            cqxmax = float(qx_grid.max()) if N_total > 0 else 0.0
            cqymin = float(qy_grid.min()) if N_total > 0 else 0.0
            cqymax = float(qy_grid.max()) if N_total > 0 else 0.0
            cqzmin = float(qz_grid.min()) if N_total > 0 else 0.0
            cqzmax = float(qz_grid.max()) if N_total > 0 else 0.0
            _Rw_pad = float(reach + sd_um)
            H = st.get('host_locs')
            if H is not None:
                _lo = np.searchsorted(H['xs'], cqxmin - _Rw_pad, side='left')
                _hi = np.searchsorted(H['xs'], cqxmax + _Rw_pad, side='right')
                _idx = np.sort(H['order'][_lo:_hi])
                c_x = torch.as_tensor(H['x'][_idx], dtype=torch.float64, device=gpu_device)
                c_y = torch.as_tensor(H['y'][_idx], dtype=torch.float64, device=gpu_device)
                c_z = torch.as_tensor(H['z'][_idx], dtype=torch.float64, device=gpu_device)
                c_g = torch.as_tensor(H['gi'][_idx], dtype=torch.int64, device=gpu_device)
                del _idx
            else:
                mask_chunk_t = (loc_x_t >= cqxmin - _Rw_pad) & (loc_x_t <= cqxmax + _Rw_pad)
                c_x, c_y, c_z, c_g = (loc_x_t[mask_chunk_t], loc_y_t[mask_chunk_t],
                                      loc_z_t[mask_chunk_t], gi_t[mask_chunk_t])
                del mask_chunk_t
            state = _kde_setup_celllist(
                c_x, c_y, c_z, c_g, cqxmin, cqymin, cqzmin, cqxmax, cqymax, cqzmax,
                sd_um, bandwidth, prune_coefficient, dim, kernel=kernel)

            # only the voxels with a transcript within the kernel's reach: the rest are exactly
            # zero (vf_norm 0, never kept) and stay as the zero-initialised arrays have them
            if os.environ.get('SSAFT_KDE_NO_SKIP') or N_total == 0:
                qsel = np.arange(N_total)
            else:
                qsel = np.flatnonzero(_kde_nonzero_queries(
                    c_x, c_y, c_z if dim == 3 else None, x0, local_cx, vy, vz,
                    sd_um, reach, kernel))
            del c_x, c_y, c_z, c_g
            if kde_bar is not None:
                kde_bar.update(N_total - len(qsel))
            n_q = len(qsel)
            n_sub = (n_q + query_subchunk_size - 1) // query_subchunk_size
            if kde_bar is not None:
                kde_bar.set_postfix_str(f'chunk {chunk_idx + 1}/{n_chunk}', refresh=False)
            for s in range(n_sub):
                s0 = s * query_subchunk_size; s1 = min(n_q, s0 + query_subchunk_size)
                sub_idx = qsel[s0:s1]
                kde_t = _kde_run_subchunk(state, qx_flat[sub_idx], qy_flat[sub_idx],
                                          qz_flat[sub_idx], n_genes_use, gpu_device)
                vf_norm_t = kde_t.sum(dim=1)
                local_x = sub_idx // (vy * vz); rem = sub_idx % (vy * vz)
                local_y = rem // vz; local_z = rem % vz
                vf_norm_chunk[local_x, local_y, local_z] = (
                    vf_norm_t.detach().cpu().numpy().astype(np.float32, copy=False))
                if kde_bar is not None:
                    kde_bar.update(s1 - s0)
                if want_full_field:
                    vf_full_t = (kde_t / torch.clamp(vf_norm_t, min=1e-12).unsqueeze(1)
                                 * float(san))
                    if log_transform:
                        vf_full_t = torch.log1p(vf_full_t)
                    full_chunk[:, local_x, local_y, local_z] = (
                        vf_full_t.T.detach().cpu().numpy().astype(np.float32, copy=False))
                    del vf_full_t
                mask_t = vf_norm_t > norm_threshold
                if int(mask_t.sum().item()) == 0:
                    del kde_t, vf_norm_t, mask_t
                    continue
                data_kept_t = kde_t[mask_t]
                vfn_kept_t = vf_norm_t[mask_t]
                mask_np = mask_t.detach().cpu().numpy()
                data_kept_t = data_kept_t / vfn_kept_t.unsqueeze(1) * float(san)
                if log_transform:
                    data_kept_t = torch.log1p(data_kept_t)
                _kxb, _kyb, _kzb = (local_x[mask_np], local_y[mask_np],
                                    local_z[mask_np])
                keep_x.append(_kxb); keep_y.append(_kyb); keep_z.append(_kzb)
                if on_batch is not None:
                    on_batch(data_kept_t, _kxb, _kyb, _kzb, vfn_kept_t, x0)
                    del data_kept_t
                else:
                    keep_vecs.append(data_kept_t)
                del kde_t, vf_norm_t, mask_t, vfn_kept_t
            kx = np.concatenate(keep_x) if keep_x else np.zeros(0, np.int64)
            ky = np.concatenate(keep_y) if keep_y else np.zeros(0, np.int64)
            kz = np.concatenate(keep_z) if keep_z else np.zeros(0, np.int64)
            vecs = torch.cat(keep_vecs, dim=0) if keep_vecs else None
            yield dict(x0=x0, x1=x1, vf_norm=vf_norm_chunk, full=full_chunk,
                       kx=kx, ky=ky, kz=kz, vecs=vecs)
            del Qx, Qy, Qz, qx_flat, qy_flat, qz_flat, state
            gc.collect(); torch.cuda.empty_cache()

    def _transcript_gpu(self, sdata, *, do_map, save_vf, centroids=None,
                        element_name=None, exclude_gene_indices=None,
                        marker_genes_path=None,
                        method='corr', map_kwargs=None,
                        bandwidth=2.5, sampling_distance=1.0, prune_coefficient=4.3,
                        kernel='gaussian',
                        size_after_normalization=10, log_transform=True, overwrite=True,
                        gpu_device='cuda', query_subchunk_size=None,
                        gpu_target_bytes=int(1e9), chunk_start=None, chunk_end=None,
                        cx=None, create_arrays=True, attach=True):
        """ORCHESTRATOR. Drive the KDE producer into the mapping consumer and/or a
        tensorstore field writer in ONE pass. Covers all transcript combos:
          map_celltypes save_vf=False → do_map=True,  save_vf=False (stream+map)
          map_celltypes save_vf=True  → do_map=True,  save_vf=True  (map+store, 1 pass)
          compute_vf(imaging)         → do_map=False, save_vf=True  (store only)
        """
        import torch
        from ._mapping_gpu import _pearson_max_gpu
        from ._count import _tensorstore_array
        if not torch.cuda.is_available():
            raise RuntimeError('transcript map/compute_vf GPU path requires CUDA')
        st = self._prepare_transcript_field(
            sdata, centroids=centroids, element_name=element_name,
            exclude_gene_indices=exclude_gene_indices, bandwidth=bandwidth,
            sampling_distance=sampling_distance, prune_coefficient=prune_coefficient,
            kernel=kernel,
            size_after_normalization=size_after_normalization, gpu_device=gpu_device)
        vx, vy, vz = st['vx'], st['vy'], st['vz']; n_genes_use = st['n_genes_use']
        # Each celltype correlated on its own markers (+ ancestors'); None = full axis.
        marker_idx = (self._marker_index_sets(marker_genes_path, st['use_genes'])
                      if (do_map and marker_genes_path is not None
                          and method != 'lda') else None)
        _ms = None
        zg = self.dataset.zarr_group
        from ._count import score_key_for
        score_key = score_key_for(method)
        _origin = np.asarray(zg['coordinate_offset'][:], dtype=np.float64)
        # method='lda' keeps this GPU KDE producer and only swaps the consumer: the
        # multinomial likelihood is one matmul, so there is no reason to fall back to
        # the CPU producer the way gnb has to.
        lda_scorer = lda_veff = None
        _lda_on_gpu = False
        _lda_Mfull = None
        _mk_batch = 1_000_000
        if do_map and method == 'lda':
            from ._lda import build_factors, anchor_positions, AnchorPrior, LDAScorer
            _mk = dict(map_kwargs or {})
            lda_veff = _mk.pop('veff', None) or 1.0 / float(
                self.dataset.expression_threshold)
            _gi = np.asarray([st['genes'].index(g) for g in st['use_genes']])
            _ms = (self._own_marker_sets(marker_genes_path, list(st['use_genes']))
                   if marker_genes_path is not None else None)
            _beta, _ = build_factors(
                zg, gene_indices=_gi, veff=lda_veff,
                size_after_normalization=size_after_normalization,
                log_transformed=log_transform, sampling_distance=sampling_distance,
                verbose=self.verbose)
            _prior = None
            if bool(_mk.pop('use_anchor_prior', True)):
                lda_anchor = anchor_positions(
                    zg, sampling_distance=sampling_distance,
                    coordinate_offset=_mk.pop('anchor_offset', None))
                _prior = self._make_anchor_prior(
                    lda_anchor, _beta.shape[0], sampling_distance, _mk)
            _unk = np.zeros(_beta.shape[0], dtype=bool)
            _unk[[i for i in self._unknown_class_ids() if i < _beta.shape[0]]] = True
            lda_scorer = LDAScorer(
                _beta, prior=_prior, unknown=_unk,
                cand_nats=float(_mk.pop('cand_nats', 1.0)),
                prior_weight=float(_mk.pop('prior_weight', 0.5)),
                prior_floor=float(_mk.pop('prior_floor', 0.02)))
            _mk.pop('stream_lda', None)
            # Score on the device when there is one; the CPU path stays the fallback
            # and the reference implementation.
            _lda_on_gpu = False
            _lda_Mfull = None
            _mk_batch = int(_mk.pop('score_batch', 1_000_000))
            if bool(_mk.pop('score_on_gpu', True)):
                try:
                    import torch
                    if torch.cuda.is_available():
                        _dev = torch.device(gpu_device)
                        lda_scorer.to_torch(_dev)
                        _lda_on_gpu = True
                        if _ms is not None:
                            _mf = np.zeros((_beta.shape[1], _beta.shape[0]), np.float32)
                            for _k, _idx in enumerate(_ms):
                                if _idx is not None and len(_idx):
                                    _mf[np.asarray(_idx), _k] = 1.0 / len(_idx)
                            _lda_Mfull = torch.as_tensor(_mf, device=_dev)
                        self._m(f'map_celltypes: LDA scored on {gpu_device}')
                except Exception as _e:
                    self._m(f'map_celltypes: GPU scoring unavailable ({_e}); using CPU')
                    _lda_on_gpu = False

        def _safe_remove(name):
            if name in zg:
                _remove_array(zg, name)

        if cx is not None:
            st['cx'] = int(cx)
            st['n_chunk'] = (vx + int(cx) - 1) // int(cx)

        if create_arrays:
            if overwrite:
                for nm in ('vf_norm', 'celltype_maps', score_key,
                           'celltype_marker_score',
                           'vf_normalized', 'vf_normalized_genes'):
                    _safe_remove(nm)
            _safe_remove('genes'); zg['genes'] = np.asarray(st['genes'])
            # what built the field, for provenance. Deliberately NOT the keys
            # analytic_vf_norm_threshold reads -- transcript stores keep deriving
            # their threshold from the live dataset, exactly as before.
            zg.attrs['kde_kernel'] = st.get('kernel', 'gaussian')
            zg.attrs['kde_bandwidth'] = float(st['bandwidth'])   # kernel's own scale (H if compact)
            self.dataset._zarr_create_array(zg, 'vf_norm', (vx, vy, vz), 'f4')
            if do_map:
                # gnb still routes to the CPU producer; corr and lda run here. The other
                # method's score volume stays on disk — 'mapping_method' names the live one.
                zg.attrs['mapping_method'] = method
                self.dataset._zarr_create_array(zg, 'celltype_maps', (vx, vy, vz), 'i4')
                self.dataset._zarr_create_array(zg, score_key, (vx, vy, vz), 'f4')
                if method == 'lda' and _ms is not None:
                    self.dataset._zarr_create_array(
                        zg, 'celltype_marker_score', (vx, vy, vz), 'f4')
        vf_ts = None; vf_zarr = None
        if save_vf and create_arrays:
            _safe_remove('vf_normalized'); _safe_remove('vf_normalized_genes')
            zg['vf_normalized_genes'] = np.asarray(st['use_genes'])
            store_root = getattr(zg.store, 'root', None)
            group_dir = (os.path.join(str(store_root), zg.path) if zg.path
                         else str(store_root)) if store_root is not None else None
            shape = (n_genes_use, vx, vy, vz); chunks = (n_genes_use, 1, vy, vz)
            vf_ts = _tensorstore_array(group_dir, 'vf_normalized', shape, chunks)
            if vf_ts is None:
                vf_zarr = zg.create_array('vf_normalized', shape=shape, chunks=chunks,
                                          dtype='f4', fill_value=0.0)
        elif save_vf:
            vf_zarr = zg['vf_normalized']

        centroids_t = (torch.as_tensor(st['centroids_use'], dtype=torch.float32,
                                       device=gpu_device) if do_map else None)
        inflight = []
        # STREAMING CONSUMER. The kept vectors of one chunk are several GB at whole-
        # transcriptome panels, and holding them to the end of the chunk (then
        # concatenating, which doubles them) is what runs the card out of memory. The
        # LDA consumer needs nothing but this batch, so it scores each sub-chunk as the
        # producer makes it and keeps only the (int32, float32) answers.
        _pend = {'i': [], 'c': [], 'm': [], 'x': [], 'y': [], 'z': []}

        def _lda_on_batch(vecs_t, kx, ky, kz, vfn_t, x0):
            import torch
            _xy = np.stack([(kx + x0) * sampling_distance + _origin[0],
                            ky * sampling_distance + _origin[1]], 1)
            _pz = kz * sampling_distance + _origin[2]
            n = vecs_t.shape[0]
            bi = np.empty(n, np.int32); bc = np.empty(n, np.float32)
            mv = np.empty(n, np.float32) if _lda_Mfull is not None else None
            for a in range(0, n, int(_mk_batch)):
                b = min(n, a + int(_mk_batch))
                t = vecs_t[a:b].clone()
                if log_transform:
                    torch.expm1_(t)
                t.mul_(float(lda_veff / size_after_normalization))
                t.mul_(vfn_t[a:b].unsqueeze(1))
                _bi, _bc = lda_scorer.torch_call(t, _xy[a:b], _pz[a:b])
                bi[a:b] = _bi; bc[a:b] = _bc
                if _lda_Mfull is not None:
                    sel = (t @ _lda_Mfull).gather(
                        1, torch.as_tensor(_bi, device=t.device).long().unsqueeze(1)
                    ).squeeze(1)
                    mv[a:b] = (sel / max(lda_veff, 1e-12)).cpu().numpy()
                del t
            _pend['i'].append(bi); _pend['c'].append(bc)
            _pend['x'].append(kx); _pend['y'].append(ky); _pend['z'].append(kz)
            if mv is not None:
                _pend['m'].append(mv)

        # escape hatch: map_kwargs={'stream_lda': False} restores the accumulate-then-
        # concatenate path, for comparing the two or for a consumer that needs the
        # whole chunk at once
        _stream = (do_map and method == 'lda' and _lda_on_gpu
                   and bool((map_kwargs or {}).get('stream_lda', True)))
        from ._count import _SlabWriter
        _slab = _SlabWriter(zg)
        it = self._kde_field_iter_gpu(
            st, chunk_start=chunk_start, chunk_end=chunk_end, log_transform=log_transform,
            want_full_field=save_vf, query_subchunk_size=query_subchunk_size,
            gpu_target_bytes=gpu_target_bytes,
            on_batch=(_lda_on_batch if _stream else None))
        for fc in it:
            x0, x1 = fc['x0'], fc['x1']
            _slab('vf_norm', x0, x1, fc['vf_norm'])
            if save_vf and fc['full'] is not None:
                if vf_ts is not None:
                    inflight.append(vf_ts[:, x0:x1, :, :].write(fc['full']))
                    while len(inflight) > 4:
                        inflight.pop(0).result()
                else:
                    vf_zarr[:, x0:x1, :, :] = fc['full']
            if do_map:
                local_cx = x1 - x0
                ctmap = np.full((local_cx, vy, vz), -1, dtype=np.int32)
                corr = np.full((local_cx, vy, vz), -1.0, dtype=np.float32)
                if _stream:
                    if _pend['i']:
                        kx = np.concatenate(_pend['x']); ky = np.concatenate(_pend['y'])
                        kz = np.concatenate(_pend['z'])
                        ctmap[kx, ky, kz] = np.concatenate(_pend['i'])
                        corr[kx, ky, kz] = np.concatenate(_pend['c'])
                        if _ms is not None and _pend['m']:
                            msv = np.zeros((local_cx, vy, vz), dtype=np.float32)
                            msv[kx, ky, kz] = np.concatenate(_pend['m'])
                            _slab('celltype_marker_score', x0, x1, msv)
                    for _k in _pend:
                        _pend[_k].clear()
                    _slab('celltype_maps', x0, x1, ctmap)
                    _slab(score_key, x0, x1, corr)
                    continue
                if fc['vecs'] is not None and fc['vecs'].shape[0] > 0:
                    if method == 'lda':
                        _xy = np.stack(
                            [(fc['kx'] + x0) * sampling_distance + _origin[0],
                             fc['ky'] * sampling_distance + _origin[1]], 1)
                        _pz = fc['kz'] * sampling_distance + _origin[2]
                        if _lda_on_gpu:
                            # The field is already on the device; scoring it there too
                            # skips a (voxels x genes) host copy per chunk. Done in
                            # sub-batches and in place: a chunk is ~7M x 63 float32, so
                            # every extra full-size temporary is another 1.7 GB and four
                            # of them overflow even a 32 GB card.
                            import torch
                            _src = fc['vecs']
                            _n = _src.shape[0]
                            _vfk = np.asarray(
                                fc['vf_norm'][fc['kx'], fc['ky'], fc['kz']],
                                dtype=np.float32)
                            _bs = int(_mk_batch)
                            best_idx = np.empty(_n, dtype=np.int32)
                            best_corr = np.empty(_n, dtype=np.float32)
                            _msv = (np.empty(_n, dtype=np.float32)
                                    if _lda_Mfull is not None else None)
                            for _s0 in range(0, _n, _bs):
                                _s1 = min(_n, _s0 + _bs)
                                _t = _src[_s0:_s1].clone()
                                if log_transform:
                                    torch.expm1_(_t)
                                _t.mul_(float(lda_veff / size_after_normalization))
                                _t.mul_(torch.as_tensor(_vfk[_s0:_s1],
                                                        device=_t.device).unsqueeze(1))
                                _bi, _bc = lda_scorer.torch_call(
                                    _t, _xy[_s0:_s1], _pz[_s0:_s1])
                                best_idx[_s0:_s1] = _bi
                                best_corr[_s0:_s1] = _bc
                                if _lda_Mfull is not None:
                                    _sel = (_t @ _lda_Mfull).gather(
                                        1, torch.as_tensor(_bi, device=_t.device)
                                        .long().unsqueeze(1)).squeeze(1)
                                    _msv[_s0:_s1] = (
                                        _sel / max(lda_veff, 1e-12)).cpu().numpy()
                                del _t
                            del _src
                        else:
                            _v = fc['vecs'].detach().cpu().numpy()
                            if log_transform:
                                _v = np.expm1(_v)      # LDA wants counts, not log1p
                            _X = np.asarray(_v, dtype=np.float32)
                            _X *= np.float32(lda_veff / size_after_normalization)
                            _X *= np.asarray(
                                fc['vf_norm'][fc['kx'], fc['ky'], fc['kz']],
                                dtype=np.float32)[:, None]
                            best_idx, best_corr = lda_scorer(_X, _xy, _pz)
                            if _ms is not None:
                                from ._marker_corr import marker_score
                                _msv = marker_score(_X, best_idx,
                                                    _ms) / max(lda_veff, 1e-12)
                        ctmap[fc['kx'], fc['ky'], fc['kz']] = best_idx
                        corr[fc['kx'], fc['ky'], fc['kz']] = best_corr
                        _slab('celltype_maps', x0, x1, ctmap)
                        _slab(score_key, x0, x1, corr)
                        if _ms is not None:
                            msv = np.zeros((local_cx, vy, vz), dtype=np.float32)
                            msv[fc['kx'], fc['ky'], fc['kz']] = _msv
                            _slab('celltype_marker_score', x0, x1, msv)
                        continue
                    if marker_idx is not None:
                        from ._marker_corr import masked_pearson_max_torch
                        best_idx, best_corr = masked_pearson_max_torch(
                            fc['vecs'], centroids_t, marker_idx)
                    else:
                        best_idx, best_corr = _pearson_max_gpu(centroids_t, fc['vecs'])
                    ctmap[fc['kx'], fc['ky'], fc['kz']] = best_idx.cpu().numpy()
                    corr[fc['kx'], fc['ky'], fc['kz']] = best_corr.cpu().numpy()
                _slab('celltype_maps', x0, x1, ctmap)
                _slab(score_key, x0, x1, corr)
        _slab.flush()
        for f in inflight:
            f.result()
        if attach:
            import dask.array as _da
            self.dataset._vf_norm = _da.from_zarr(zg['vf_norm'])
            if do_map:
                self.dataset.celltype_maps = zg['celltype_maps'][:]
                setattr(self.dataset, score_key, zg[score_key][:])
                if 'celltype_marker_score' in zg:
                    self.dataset.celltype_marker_score = zg['celltype_marker_score'][:]


    def init_map_celltypes(self, sdata, centroids=None, element_name=None,
                               exclude_gene_indices=None,
                               bandwidth=2.5, sampling_distance=1.0, prune_coefficient=4.3,
                               kernel='gaussian',
                               size_after_normalization=10, log_transform=True,
                               method='corr', marker_genes_path=None, h5ad=None,
                               column='cellman_annotation', leiden_column='leiden', cmap=None):
        """
        Initialize zarr arrays for a distributed ``map_celltypes_cpu`` run.

        Must be called **once** on any single machine before launching
        parallel workers.  After this returns, call
        ``map_celltypes_partial`` on each worker with the returned
        chunk info.

        Parameters
        ----------
        sdata : spatialdata.SpatialData
        centroids : ndarray (n_clusters, n_genes), optional
            Falls back to ``self.dataset.centroids``.
        bandwidth, sampling_distance, prune_coefficient,
        size_after_normalization, log_transform
            Same as ``map_celltypes_cpu``; recorded so workers can use
            identical values.

        :param element_name: Name of the points element to parse from ``sdata``.
        :type element_name: str or None
        :param exclude_gene_indices: Gene indices to drop before correlation.
        :type exclude_gene_indices: list[int] or None
        :param marker_genes_path: Restrict the mapping to marker genes. ``None`` (default)
            keeps the whole panel and behaves exactly as before. Otherwise a
            ``{celltype: [genes]}`` dict, the CeLLMan ``celltype_markers.yaml``
            (defining / required / supporting per label, already restricted to genes
            measured in the dataset; all three tiers are unioned — preferred), or
            ``expert_judgment_annotation.csv``
            (its thinner ``key_markers`` column). The gene axis is cut down to the UNION
            of the marker sets and everything else is dropped, so the centroids are
            built from the genes the annotation leaned on instead of the whole panel —
            flat, non-discriminative genes stop diluting the correlation / likelihood.
            Combined with ``exclude_gene_indices`` if both are given.
        :type marker_genes_path: str or pathlib.Path or dict or None

        Returns
        -------
        dict
            ``{'cx': int, 'n_chunk': int, 'vx': int, 'vy': int, 'vz': int,
               'bandwidth': float, 'sampling_distance': float,
               'prune_coefficient': float, 'kernel': str,
               'size_after_normalization': float, 'log_transform': bool}``
        """
        # The classes come from the annotation, exactly as in map_celltypes. A distributed run
        # has to fix them BEFORE the workers start: each worker rebuilds the LDA factors from
        # this store, so they must all read the same classes.
        if h5ad is not None:
            from ._mapping_inputs import build_centroids_from_h5ad
            build_centroids_from_h5ad(
                self.dataset.zarr_group, h5ad, column=column, leiden_column=leiden_column,
                cmap=cmap, store_dir=self.dataset._zarr_store_path(self.dataset.zarr_store))
            self.dataset.zarr_group.attrs['annotation_h5ad'] = os.path.abspath(h5ad)
        self.dataset.cluster_labels          = self.dataset.zarr_group['cluster_labels'][:]
        self.dataset.downsampled_vectors              = self.dataset.zarr_group['downsampled_vectors'][:]
        self.dataset.normalized_vectors      = self.dataset.zarr_group['normalized_vectors'][:]
        self.dataset.centroids               = self.dataset.zarr_group['centroids'][:]

        if centroids is None:
            if getattr(self.dataset, 'centroids', None) is None:
                raise ValueError('centroids not found. Pass h5ad=<annotated .h5ad> to map_celltypes to build them.')
            centroids = np.asarray(self.dataset.centroids, dtype=np.float32)
        else:
            centroids = np.asarray(centroids, dtype=np.float32)

        if getattr(self.dataset, 'shape', None) is None or \
                not hasattr(self, 'locations') or self.locations is None:
            self._prepare_grid(sdata, element_name=element_name,
                           bandwidth=bandwidth,
                           sampling_distance=sampling_distance,
                           prune_coefficient=prune_coefficient,
                           size_after_normalization=size_after_normalization)

        vx, vy, vz = (int(self.dataset.shape[0]),
                      int(self.dataset.shape[1]),
                      int(self.dataset.shape[2]))

        from ._count import kernel_scale
        self.dataset.ndim = 2 if vz == 1 else 3
        self.dataset.bandwidth = kernel_scale(bandwidth, self.dataset.ndim, kernel)

        grouped = self.locations.groupby('gene', sort=True)
        grouped_map = {gene: loc for gene, loc in grouped}

        if getattr(self.dataset, 'genes', None) is None:
            genes = sorted(grouped_map.keys())
            self.dataset.genes = genes
        else:
            genes = list(self.dataset.genes)

        if centroids.shape[1] != len(genes):
            raise ValueError(
                f'centroids gene dim ({centroids.shape[1]}) != gene count ({len(genes)})'
            )

        shared_gene_indices = [i for i, g in enumerate(genes) if g in grouped_map]
        shared_genes = [genes[i] for i in shared_gene_indices]
        if len(shared_genes) == 0:
            raise ValueError('No common genes between input data and centroids.')

        def _safe_remove(name):
            if name in self.dataset.zarr_group:
                _remove_array(self.dataset.zarr_group, name)

        # correlation-only path; drop any posterior volume from an earlier gnb run so
        # filter_celltypemaps does not threshold this map with min_p.
        from ._count import score_key_for
        score_key = score_key_for(method)
        for name in ['vf_norm', 'celltype_maps', 'max_correlations', 'max_probabilities',
                     'celltype_marker_score']:
            _safe_remove(name)
        self.dataset.max_probabilities = None

        import numcodecs as _numcodecs
        _safe_remove('genes')
        self.dataset.zarr_group['genes'] = np.asarray(genes)

        self.dataset._zarr_create_array(self.dataset.zarr_group, 'vf_norm', (vx, vy, vz), 'f4')
        self.dataset._zarr_create_array(self.dataset.zarr_group, 'celltype_maps', (vx, vy, vz), 'i4')
        self.dataset._zarr_create_array(self.dataset.zarr_group, score_key, (vx, vy, vz), 'f4')
        if method == 'lda' and marker_genes_path is not None:
            self.dataset._zarr_create_array(
                self.dataset.zarr_group, 'celltype_marker_score', (vx, vy, vz), 'f4')
        self.dataset.zarr_group.attrs['mapping_method'] = method
        # the workers create no arrays, so the field's provenance is recorded here
        from ._count import resolve_kernel, kernel_scale
        _dim = 2 if vz == 1 else 3
        self.dataset.zarr_group.attrs['kde_kernel'] = resolve_kernel(kernel)
        self.dataset.zarr_group.attrs['kde_bandwidth'] = kernel_scale(bandwidth, _dim, kernel)

        cx, n_chunk = self.calc_chunk(vx, vy, vz, len(shared_genes), mode='map_celltypes_cpu')
        if cx is None or cx <= 0:
            raise RuntimeError('Cannot determine chunk size.')

        if self.verbose:
            print(f'init_map_celltypes: voxel grid ({vx}, {vy}, {vz}), '
                  f'cx={cx}, n_chunk={n_chunk}, shared_genes={len(shared_genes)}')

        return {
            'cx': cx,
            'n_chunk': n_chunk,
            'method': method,
            'vx': vx, 'vy': vy, 'vz': vz,
            'bandwidth': bandwidth,
            'sampling_distance': sampling_distance,
            'prune_coefficient': prune_coefficient,
            'kernel': kernel,
            'size_after_normalization': size_after_normalization,
            'log_transform': log_transform,
        }

    def map_celltypes_partial(self, sdata, chunk_start, chunk_end, cx,
                                  centroids=None, element_name=None,
                                  exclude_gene_indices=None,
                                  bandwidth=2.5, sampling_distance=1.0, prune_coefficient=4.3,
                                  kernel='gaussian',
                                  size_after_normalization=10, log_transform=True,
                                  gpu_device=None, force_cpu=False,
                                  query_subchunk_size=None, gpu_target_bytes=int(1e9),
                                  method='corr', marker_genes_path=None, map_kwargs=None):
        """
        Process a contiguous range of X-axis chunks for a distributed run.

        Call ``init_map_celltypes`` once first to create zarr arrays and
        obtain the shared ``cx`` / ``n_chunk`` values.  Then call this
        function on each worker with non-overlapping ``[chunk_start,
        chunk_end)`` ranges.

        Each worker runs on **GPU** when ``gpu_device`` is given and a CUDA
        device is available (and ``force_cpu`` is False); otherwise it
        falls back to the CPU implementation. GPU and CPU paths share the
        same I/O contract — both write the zarr slices ``[x0:x1, :, :]``
        for the chunks they own — so workers may even mix backends.

        Parameters
        ----------
        sdata : spatialdata.SpatialData
            Used to populate ``self.locations`` via ``_prepare_grid``.
        chunk_start : int
            First chunk index to process (inclusive).
        chunk_end : int
            Last chunk index to process (exclusive).
        cx : int
            Chunk width in voxels along the X axis.  **Must be the same
            value across all workers.**  Obtain from
            ``init_map_celltypes`` return value.
        gpu_device : str, optional
            CUDA device for this worker (e.g. ``'cuda:0'``). If None, the
            worker uses the CPU path.
        force_cpu : bool, default False
            Force the CPU path even if ``gpu_device`` is given.
        method : {'corr', 'lda'}, default 'corr'
            Must match ``init_map_celltypes``: it decides which score volume the worker writes
            (``max_correlations`` / ``max_probabilities``) and, for ``'lda'``, that the factors are
            rebuilt from the store. Every worker reads the same store, so they build the same
            factors and their slices join into the single-process result.
        marker_genes_path, map_kwargs
            As ``map_celltypes``. ``marker_genes_path`` is what makes ``'lda'`` write
            ``celltype_marker_score``, so pass the same value to ``init_map_celltypes``.
        query_subchunk_size, gpu_target_bytes
            Forwarded to the GPU core (ignored on CPU).
        bandwidth, sampling_distance, prune_coefficient,
        size_after_normalization, log_transform, centroids,
        element_name, exclude_gene_indices
            Same as ``map_celltypes_cpu``.  Values must match those used
            during ``init_map_celltypes``.

        Returns
        -------
        None
            Results are written to the zarr store slices
            ``[x0:x1, :, :]`` for each processed chunk.
        """
        use_gpu = False
        if gpu_device is not None and not force_cpu:
            try:
                from ._kde_gpu import gpu_available
                use_gpu = gpu_available(gpu_device)
            except Exception:
                use_gpu = False
        if use_gpu:
            self._m(f'map_celltypes_partial: GPU path ({gpu_device}) '
                    f'chunks {chunk_start}..{chunk_end - 1}')
            return self._transcript_gpu(
                sdata, do_map=True, save_vf=False,
                chunk_start=chunk_start, chunk_end=chunk_end, cx=cx,
                create_arrays=False, attach=False,
                method=method, marker_genes_path=marker_genes_path, map_kwargs=map_kwargs,
                centroids=centroids, element_name=element_name,
                exclude_gene_indices=exclude_gene_indices, bandwidth=bandwidth,
                sampling_distance=sampling_distance, prune_coefficient=prune_coefficient,
                kernel=kernel,
                size_after_normalization=size_after_normalization,
                log_transform=log_transform, overwrite=False,
                gpu_device=gpu_device, query_subchunk_size=query_subchunk_size,
                gpu_target_bytes=gpu_target_bytes,
            )

        from tqdm.auto import tqdm
        from ._count import resolve_kernel, kernel_code
        kernel = resolve_kernel(kernel)

        self.dataset.cluster_labels          = self.dataset.zarr_group['cluster_labels'][:]
        self.dataset.downsampled_vectors              = self.dataset.zarr_group['downsampled_vectors'][:]
        self.dataset.normalized_vectors      = self.dataset.zarr_group['normalized_vectors'][:]
        self.dataset.centroids               = self.dataset.zarr_group['centroids'][:]

        if centroids is None:
            if getattr(self.dataset, 'centroids', None) is None:
                raise ValueError('centroids not found. Pass h5ad=<annotated .h5ad> to map_celltypes to build them.')
            centroids = np.asarray(self.dataset.centroids, dtype=np.float32)
        else:
            centroids = np.asarray(centroids, dtype=np.float32)

        if getattr(self.dataset, 'shape', None) is None or \
                not hasattr(self, 'locations') or self.locations is None:
            self._prepare_grid(sdata, element_name=element_name,
                           bandwidth=bandwidth,
                           sampling_distance=sampling_distance,
                           prune_coefficient=prune_coefficient,
                           size_after_normalization=size_after_normalization)

        vx, vy, vz = (int(self.dataset.shape[0]),
                      int(self.dataset.shape[1]),
                      int(self.dataset.shape[2]))

        # the kernel's own scale, as the GPU path records it (norm_threshold, V_eff)
        from ._count import resolve_kernel, kernel_scale, kernel_peak
        self.dataset.ndim = 2 if vz == 1 else 3
        self.dataset.bandwidth = kernel_scale(bandwidth, self.dataset.ndim, kernel)
        self.dataset.kde_kernel = resolve_kernel(kernel)
        self.dataset.expression_threshold = kernel_peak(
            self.dataset.bandwidth, self.dataset.ndim, kernel)
        self.dataset.norm_threshold = self.dataset.expression_threshold * 2

        grouped = self.locations.groupby('gene', sort=True)
        grouped_map = {gene: loc for gene, loc in grouped}

        if getattr(self.dataset, 'genes', None) is None:
            genes = sorted(grouped_map.keys())
            self.dataset.genes = genes
        else:
            genes = list(self.dataset.genes)

        if centroids.shape[1] != len(genes):
            raise ValueError(
                f'centroids gene dim ({centroids.shape[1]}) != gene count ({len(genes)})'
            )

        shared_gene_indices = [i for i, g in enumerate(genes) if g in grouped_map]
        shared_genes = [genes[i] for i in shared_gene_indices]
        missing_genes = [g for g in genes if g not in grouped_map]
        if len(shared_genes) == 0:
            raise ValueError('No common genes between input data and centroids.')
        if missing_genes and self.verbose:
            print(f'map_celltypes_partial: dropping {len(missing_genes)} missing genes '
                  f'(e.g. {missing_genes[:5]})')

        grouped_list = [(g, grouped_map[g]) for g in shared_genes]
        centroids_shared = centroids[:, shared_gene_indices]

        if exclude_gene_indices is not None:
            idx_to_local = {gi: li for li, gi in enumerate(shared_gene_indices)}
            exclude_shared = sorted(
                idx_to_local[i] for i in exclude_gene_indices if i in idx_to_local
            )
            centroids_use = np.delete(centroids_shared, exclude_shared, axis=1)
        else:
            exclude_shared = None
            centroids_use = centroids_shared

        import numcodecs as _nc2
        _nc2.blosc.set_nthreads(self.ncores)

        n_chunks_total = chunk_end - chunk_start
        if self.verbose:
            print(f'map_celltypes_partial: processing chunks '
                  f'{chunk_start}..{chunk_end - 1} (cx={cx}, vx={vx})')

        # One bar for this worker's whole range, not one per chunk.
        gene_bar = tqdm(
            total=n_chunks_total * len(grouped_list), desc='KDE genes',
            leave=True, dynamic_ncols=True,
        ) if self.verbose else None
        corr_bar = tqdm(
            total=n_chunks_total * len(centroids_use), desc='Correlations',
            leave=True, dynamic_ncols=True,
        ) if self.verbose else None

        for local_idx, chunk_idx in enumerate(range(chunk_start, chunk_end)):
            x0 = chunk_idx * cx
            x1 = min(vx, x0 + cx)
            local_cx = x1 - x0

            vf_chunk = np.zeros((local_cx, vy, vz, len(shared_genes)), dtype='f4')

            if gene_bar is not None:
                gene_bar.set_postfix_str(
                    f'chunk {chunk_idx} ({local_idx + 1}/{n_chunks_total})', refresh=False)

            for gidx, (_, loc) in enumerate(grouped_list):
                locs = np.asarray(loc)
                if locs.shape[1] == 2:
                    loc_x, loc_y = locs[:, 0].astype(np.float64), locs[:, 1].astype(np.float64)
                    loc_z = np.zeros(locs.shape[0], dtype=np.float64)
                    dim = 2
                else:
                    loc_x, loc_y, loc_z = (locs[:, 0].astype(np.float64),
                                           locs[:, 1].astype(np.float64),
                                           locs[:, 2].astype(np.float64))
                    dim = 3

                accumulate_gene_vector(
                    vf_chunk,
                    loc_x=loc_x, loc_y=loc_y, loc_z=loc_z,
                    sampling_distance=float(sampling_distance),
                    bandwidth=float(bandwidth),
                    prune_coefficient=float(prune_coefficient),
                    gidx=gidx, dim=int(dim), ncores=int(self.ncores),
                    x0=int(x0), y0=0, z0=0, kernel=kernel_code(kernel),
                )
                if gene_bar is not None:
                    gene_bar.update(1)

            vf_norm_chunk = vf_chunk.sum(axis=3)
            self.dataset.zarr_group['vf_norm'][x0:x1, :, :] = vf_norm_chunk

            local_ctmap = np.full((local_cx, vy, vz), -1, dtype=np.int32)
            local_corr = np.full((local_cx, vy, vz), -1, dtype=np.float32)

            x_idx, y_idx, z_idx, data_chunk = normalize_vf(
                vf_chunk, vf_norm_chunk, size_after_normalization, x0, self.ncores
            )
            if log_transform:
                np.log1p(data_chunk, out=data_chunk)

            if data_chunk.shape[0] > 0:
                data_use = (np.delete(data_chunk, exclude_shared, axis=1)
                            if exclude_shared is not None else data_chunk)
                best_corr = np.full(data_use.shape[0], -1, dtype=np.float32)
                best_idx = np.full(data_use.shape[0], -1, dtype=np.int32)

                if corr_bar is not None:
                    corr_bar.set_postfix_str(
                        f'chunk {chunk_idx} ({local_idx + 1}/{n_chunks_total})',
                        refresh=False)

                for cidx, centroid in enumerate(centroids_use):
                    corr_chunk = calc_ctmap(centroid, data_use, self.ncores)
                    corr_chunk = np.nan_to_num(corr_chunk).astype(np.float32, copy=False)
                    improved = corr_chunk > best_corr
                    best_corr[improved] = corr_chunk[improved]
                    best_idx[improved] = cidx
                    if corr_bar is not None:
                        corr_bar.update(1)

                local_x = x_idx.astype(np.int64) - x0
                local_y = y_idx.astype(np.int64)
                local_z = z_idx.astype(np.int64)
                local_ctmap[local_x, local_y, local_z] = best_idx
                local_corr[local_x, local_y, local_z] = best_corr

            self.dataset.zarr_group['celltype_maps'][x0:x1, :, :] = local_ctmap
            self.dataset.zarr_group['max_correlations'][x0:x1, :, :] = local_corr

            del vf_chunk, vf_norm_chunk, x_idx, y_idx, z_idx, data_chunk
            del local_ctmap, local_corr
            gc.collect()

        if gene_bar is not None:
            gene_bar.close()
        if corr_bar is not None:
            corr_bar.close()

        if self.verbose:
            print(f'map_celltypes_partial: done chunks {chunk_start}..{chunk_end - 1}')
