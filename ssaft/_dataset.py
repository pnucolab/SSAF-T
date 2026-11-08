"""The dataset object: the store behind one analysis, and the plots drawn from it.

:class:`Dataset` owns the zarr group that holds every array an analysis produces,
from the vector field through the downsampled vectors to the cell-type map, and it
lazily loads each of them on first access. It also carries the coordinate information
needed to put those arrays back into the micrometre frame of the input, and the
plotting and writing methods that do so.

An :class:`~ssaft.Analysis` computes; a ``Dataset`` remembers and renders.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import seaborn as sns
import multiprocessing
import sys, os
sns.set()
sns.set_style("whitegrid", {'axes.grid' : False})
from sklearn.decomposition import PCA
from tempfile import mkdtemp
from matplotlib.colors import ListedColormap 
from sklearn import preprocessing
import numbers
import zarr
import dask.array as da 

try:
    import xarray as xr
    from spatialdata.models import Image2DModel, Labels2DModel, Labels3DModel, TableModel
    from spatialdata.transformations import Affine
    from spatialdata.transformations.operations import get_transformation, set_transformation
    _SPATIALDATA_AVAILABLE = True
except ImportError:
    _SPATIALDATA_AVAILABLE = False

try:
    from .utils import corr
except ImportError:
    import numpy as np
    corr = lambda a, b: np.corrcoef(a, b)[0, 1]

class Dataset(object):
    """Hold the inputs and the results of one ``ssaft`` analysis.

    :param genes: The genes used for the analysis.
    :type genes: list(str)
    :param locations: Locations of the molecules in micrometres, given as a list of
        arrays of shape N by D, where N is the number of molecules and D the number of
        dimensions.
    :type locations: list(numpy.ndarray)
    :param width: Width of the image in micrometres.
    :type width: float
    :param height: Height of the image in micrometres.
    :type height: float
    :param depth: Depth of the image in micrometres. A depth of one means a planar image.
    :type depth: float
    :param save_dir: Directory in which intermediate data such as the density and the
        vector field are stored as zarr groups. Any data already present there is loaded
        and reused.
    :type save_dir: str
    """
        
    def __init__(self, store=None, in_memory=False):
        """Open or create the backing store and clear every cached result.

        :param store: Path to a zarr store, an already-open zarr store, or ``None`` for an
            in-memory group that is never written to disk.
        :type store: str or zarr store or None
        :param in_memory: Whether to materialize the vector field in memory rather than
            leaving it lazy. This trades memory for speed on repeated access.
        :type in_memory: bool
        """
        self._vf = None
        self._vf_norm = None
        self._vf_normalized = None
        self.bandwidth = None
        self._downsampled_vectors = None
        self._downsampled_vf_norm = None
        self._selected_vectors = None
        self.normalized_vectors = None
        self.expanded_vectors = None
        self.cluster_labels = None
        self.tsne = None
        self.umap = None
        self.excluded_clusters = None
        self.celltype_binned_counts = None
        self.max_probabilities = None
        self.zarr_store, self.zarr_group = self._get_zarr_group(store)
        self.in_memory = in_memory

        self.x_scale = None
        self.y_scale = None
    
    @staticmethod
    def _get_zarr_group(store):
        """Open a zarr group, preserving the on-disk format version of an existing store.

        A store written by zarr version 2 carries a ``.zgroup`` file and one written by
        version 3 carries ``zarr.json``. The format is detected from whichever is present and
        passed back when opening, so that an existing store is never reopened under the other
        version's layout. A store holding both is treated as version 2.

        :param store: Path to a store, an already-open store, or ``None``.
        :returns: A pair holding the store, which is ``None`` for an in-memory group, and the
            opened group.
        :rtype: tuple
        """
        if store is None:
            return None, zarr.group()

        zarr_format = None
        if isinstance(store, str):
            import os as _os
            store_path = store
            has_v2 = _os.path.exists(_os.path.join(store_path, '.zgroup'))
            has_v3 = _os.path.exists(_os.path.join(store_path, 'zarr.json'))
            if has_v2 and not has_v3:
                zarr_format = 2
            elif has_v3 and not has_v2:
                zarr_format = 3
            elif has_v2 and has_v3:
                zarr_format = 2

            if hasattr(zarr, 'storage') and hasattr(zarr.storage, 'LocalStore'):
                store = zarr.storage.LocalStore(store_path)
            else:
                store = zarr.DirectoryStore(store_path)

        open_kwargs = dict(store=store, mode="a")
        if zarr_format is not None:
            open_kwargs['zarr_format'] = zarr_format
        return store, zarr.open_group(**open_kwargs)

    @staticmethod
    def _zarr_create_array(group, name, shape, dtype):
        """Create a zero-filled array inside a group, respecting the group's format version.

        The ``zeros`` method of a zarr version 3 group always writes version 3 metadata,
        regardless of the parent group's format, which breaks lookups when that group is
        opened in version 2 mode. This helper prefers ``create_array``, which respects the
        group's own metadata, and falls back to ``create_dataset`` for older stores.

        :param group: The group to create the array in.
        :param name: Name of the new array.
        :type name: str
        :param shape: Shape of the new array.
        :param dtype: Element type of the new array.
        :returns: The created array.
        """
        shape = tuple(shape) if hasattr(shape, '__iter__') else (int(shape),)
        if hasattr(group, 'create_array'):
            return group.create_array(name=name, shape=shape, dtype=dtype)
        return group.create_dataset(name=name, shape=shape, dtype=dtype)

    @staticmethod
    def _zarr_store_path(store):
        """Return the on-disk path of a zarr store, for either format version.

        :param store: The store to inspect, which may be ``None``.
        :returns: The path as a string, or ``None`` when the store has none.
        :rtype: str or None
        """
        if store is None:
            return None
        for attr in ('root', 'path'):
            p = getattr(store, attr, None)
            if p is not None:
                return str(p)
        return None
    
    def _try_flush(self):
        """Flush the backing store if it has one, ignoring any failure.

        Flushing is a durability convenience rather than a correctness requirement, so a
        store that does not support it, or that fails to do it, is passed over silently.
        """
        if self.zarr_store is not None:
            try:
                self.zarr_store.flush()
            except:
                pass
    
    def extract_spatialdata_scale(self, sdata, element_name='transcripts'):
        """Read the coordinate transformation of a points element and remember its scale.

        The scale is needed to map analysis coordinates back onto the input's own frame. The
        full affine is also kept, in ``_global_affine``, so that a rotated or sheared frame is
        handled rather than only a scaled one.

        :param sdata: SpatialData object containing the points.
        :type sdata: spatialdata.SpatialData
        :param element_name: Name of the points element to read the scale from.
        :type element_name: str
        :returns: The pair of scales along x and y, or ``None`` when the element is absent,
            when spatialdata is not installed, or when the transformation could not be read.
        :rtype: tuple or None
        """
        if not _SPATIALDATA_AVAILABLE:
            print("Warning: SpatialData not available. Cannot extract scale information.")
            return None
            
        try:
            for element_type, elem_name, element in list(sdata.gen_elements()):
                if element_type == 'points' and elem_name == element_name:
                    transforms = get_transformation(element, get_all=True)
                    t = transforms['global']
                    matrix = t.to_affine_matrix(input_axes=('x', 'y'), output_axes=('x', 'y'))
                    self.x_scale = matrix[0, 0]
                    self.y_scale = matrix[1, 1]
                    self._global_affine = np.asarray(matrix[:2, :], dtype=np.float64)
                    return (self.x_scale, self.y_scale)
            
            print(f"Warning: Points element '{element_name}' not found in SpatialData.")
            return None
            
        except Exception as e:
            print(f"Warning: Failed to extract scale from SpatialData: {e}")
            return None
    
    @property
    def downsampled_vectors(self):
        """The sampled lattice positions, as voxel indices."""
        return self._downsampled_vectors
    
    @downsampled_vectors.setter
    def downsampled_vectors(self, v):
        """Set the sampled lattice positions and invalidate the vectors cached from them.

        :param v: The new positions, as voxel indices.
        """
        self._downsampled_vectors = v
        self._selected_vectors = None

    @property
    def downsampled_vf_norm(self):
        """The local density at each sampled position.

        This is the value of ``vf_norm`` at the position of each downsampled vector, of shape
        ``(N,)`` and aligned row for row with ``downsampled_vectors`` and
        ``normalized_vectors``. It is computed in ``_select_vectors``, persisted by
        ``downsample_vector``, and used downstream as the density signal for detecting
        background and low-density voxels.
        """
        if self._downsampled_vf_norm is not None:
            return self._downsampled_vf_norm
        if 'downsampled_vf_norm' in self.zarr_group:
            self._downsampled_vf_norm = np.asarray(self.zarr_group['downsampled_vf_norm'][:])
            return self._downsampled_vf_norm
        return None

    @downsampled_vf_norm.setter
    def downsampled_vf_norm(self, v):
        """Set the per-position density, or clear it.

        :param v: The new densities, or ``None`` to clear them.
        """
        self._downsampled_vf_norm = None if v is None else np.asarray(v)

    @property
    def selected_vectors(self):
        """The raw field vectors at the sampled positions.

        They are read from the store when it holds them, and otherwise gathered from the
        vector field at the downsampled positions and cached.
        """
        if 'selected_vectors' in self.zarr_group:
            self._selected_vectors = da.from_zarr(self.zarr_group['selected_vectors'])
            return self._selected_vectors
        
        if self._selected_vectors is None:
            assert self._downsampled_vectors is not None
            mask = np.zeros(self.vf_norm.shape, dtype=bool)
            mask[self.downsampled_vectors] = True
            self._selected_vectors = self.vf.reshape([-1, len(self.genes)])[np.ravel(mask)].compute()
        return self._selected_vectors
    
    @selected_vectors.setter
    def selected_vectors(self, v):
        """Set the raw field vectors at the sampled positions.

        :param v: The new vectors.
        """
        self._selected_vectors = v

    @property
    def vf(self):
        """The vector field, as an array of shape ``(x, y, z, genes)``."""
        return self._vf
    
    @vf.setter
    def vf(self, vf):
        """Set the vector field and invalidate the norm computed from it.

        The field is materialized in memory when the dataset was opened with ``in_memory``,
        and left lazy otherwise. Any ``vf_norm`` already in the store is deleted, because it
        no longer describes this field.

        :param vf: The new vector field.
        """
        if self.in_memory:
            self._vf = da.array(vf.compute())
        else:
            self._vf = vf
        self._vf_norm = None
        try:
            del self.zarr_group['vf_norm']
        except:
            pass
        
    @property
    def vf_normalized(self):
        """The row-normalized vector field, which is what the mappers score against."""
        return self._vf_normalized
        
    @vf_normalized.setter
    def vf_normalized(self, vf_normalized):
        """Set the row-normalized vector field.

        It is materialized in memory when the dataset was opened with ``in_memory``, and left
        lazy otherwise.

        :param vf_normalized: The new normalized field.
        """
        if self.in_memory:
            self._vf_normalized = da.array(vf_normalized.compute())
        else:
            self._vf_normalized = vf_normalized

    @property
    def vf_norm(self):

        """The total density per voxel, summed over genes.

        The value is returned from the cache when it is loaded, read from the store when it
        is present there, and otherwise computed from the vector field, written to the store
        and then cached. It is ``None`` when no vector field exists yet.
        """
        if self._vf_norm is not None:
            return self._vf_norm

        if 'vf_norm' in self.zarr_group:
            self._vf_norm = da.from_zarr(self.zarr_group['vf_norm'])
            return self._vf_norm

        if self.vf is None:
            return None

        self.zarr_group.zeros(name='vf_norm', shape=self.vf.shape[:-1])
        self.zarr_group['vf_norm'] = self.vf.sum(axis=3).compute()
        self._try_flush()
        self._vf_norm = da.from_zarr(self.zarr_group['vf_norm'])
        return self._vf_norm

    def get_celltype_correlation(self, idx):
        """Return the correlation map restricted to one cell type.

        :param idx: Index of the cluster.
        :type idx: int
        :return: The correlation values where the map holds that cluster, and minus one
            everywhere else.
        :rtype: numpy.ndarray
        """
        rtn = np.zeros_like(self.max_correlations) - 1
        rtn[self.celltype_maps == idx] = self.max_correlations[self.celltype_maps == idx]
        return rtn

    def get_celltype_probability(self, idx):
        """Return the probability map restricted to one cell type.

        :param idx: Index of the cluster.
        :type idx: int
        :return: The probabilities where the map holds that cluster, and minus one everywhere
            else.
        :rtype: numpy.ndarray
        """
        rtn = np.zeros_like(self.max_probabilities) - 1
        rtn[self.celltype_maps == idx] = self.max_probabilities[self.celltype_maps == idx]
        return rtn


    def _grid_to_original_coords(self, grid_x, grid_y, grid_z=None):
        """Convert lattice indices to the coordinate frame of the input.

        Each index is scaled by the sampling distance and shifted by the coordinate offset.
        When the dataset carries neither, the indices are returned unchanged.

        :param grid_x: Lattice indices along x.
        :param grid_y: Lattice indices along y.
        :param grid_z: Lattice indices along z, or ``None`` for a planar map.
        :returns: The three coordinate arrays, where the third is ``None`` if no depth was
            given or the offset carries none.
        :rtype: tuple
        """
        if not hasattr(self, 'coordinate_offset') or not hasattr(self, 'sampling_distance'):
            return grid_x, grid_y, grid_z

        real_x = grid_x * self.sampling_distance + self.coordinate_offset[0]
        real_y = grid_y * self.sampling_distance + self.coordinate_offset[1]
        real_z = None
        if grid_z is not None and len(self.coordinate_offset) > 2:
            real_z = grid_z * self.sampling_distance + self.coordinate_offset[2]
            
        return real_x, real_y, real_z
    
    @property
    def downsampled_vectors_original_coords(self):
        """The sampled positions, converted to the coordinate frame of the input.

        This is ``None`` when no positions have been sampled yet.
        """
        if not hasattr(self, 'downsampled_vectors') or self.downsampled_vectors is None:
            return None
            
        grid_x, grid_y, grid_z = self.downsampled_vectors
        orig_x, orig_y, orig_z = self._grid_to_original_coords(grid_x, grid_y, grid_z)
        
        if orig_z is not None:
            return (orig_x, orig_y, orig_z)
        else:
            return (orig_x, orig_y, grid_z)

    def _get_plot_extent(self):
        """Compute the extent that ``imshow`` should be given for this dataset.

        ``original_coordinate_bounds`` is written in two layouts by different producers. The
        molecule path stores it nested, as pairs of low and high values per axis, while the
        count path of the grid backend stores it flat, as the three minima followed by the
        three maxima. Both are handled here, as is the four-element flat form.

        :returns: The extent as left, right, bottom and top, or ``None`` when the dataset
            carries no bounds or carries them in an unrecognized layout.
        :rtype: list or None
        """
        if not hasattr(self, 'original_coordinate_bounds') or self.original_coordinate_bounds is None:
            return None
        b = np.asarray(self.original_coordinate_bounds, dtype=float)
        if b.ndim == 2 and b.shape[0] >= 2:               # nested [[x0,x1],[y0,y1],...]
            return [b[0, 0], b[0, 1], b[1, 0], b[1, 1]]
        b = b.ravel()
        if b.size == 6:                                    # flat [minx,miny,minz, maxx,maxy,maxz]
            return [b[0], b[3], b[1], b[4]]
        if b.size == 4:                                    # flat [minx,maxx,miny,maxy]
            return [b[0], b[1], b[2], b[3]]
        return None

    def _plot_celltypes_map(self, data, background="black", centroid_indices=[], colors=None, cmap='jet', rotate=0, min_r=0.6, set_alpha=False, z=None, use_original_coords=True):
        """Draw one layer of a cell-type map onto the current axes.

        A planar map is treated as a single layer. Voxels that hold no cell type are drawn
        fully transparent rather than in the background colour, so that whatever was plotted
        underneath stays visible.

        :param data: The cell-type map, of shape ``(x, y)`` or ``(x, y, z)``, where minus one
            means no cell type.
        :param background: Colour of the axes behind the map.
        :type background: str or list(float)
        :param centroid_indices: The cell types to draw. An empty list draws all of them.
        :type centroid_indices: list(int)
        :param colors: Colour per cell type, which overrides ``cmap``.
        :type colors: list or None
        :param cmap: Colormap sampled for the cell types when ``colors`` is not given.
        :type cmap: str or matplotlib.colors.Colormap
        :param rotate: Rotation of the plot, as 0, 1, 2 or 3 quarter turns.
        :type rotate: int
        :param min_r: Lowest opacity used when ``set_alpha`` scales opacity by score.
        :type min_r: float
        :param set_alpha: Whether to scale each pixel's opacity by its correlation or
            probability.
        :type set_alpha: bool
        :param z: Layer to draw. Defaults to the middle layer of the map.
        :type z: int or None
        :param use_original_coords: Accepted for signature compatibility. The extent is taken
            from the dataset's own bounds in either case.
        :type use_original_coords: bool
        """
        data = np.asarray(data)
        if data.ndim == 2:                       # 2D map (no z axis) -> add a singleton z plane
            data = data[..., None]
        if z is None:
            z = int(data.shape[2] / 2)           # z from the MAP's own shape, not self.shape (vf_norm may be 2D)
        num_ctmaps = np.max(data) + 1

        if len(centroid_indices) == 0:
            centroid_indices = list(range(num_ctmaps))

        if colors is None:
            cmap_internal = plt.get_cmap(cmap)
            colors = cmap_internal([float(i) / (num_ctmaps - 1) for i in range(num_ctmaps)])

        all_colors = [background if not j in centroid_indices else colors[i] for i, j in enumerate(range(num_ctmaps))]
        cmap_internal = ListedColormap(all_colors)

        celltype_maps_internal = np.array(data[..., z].T, copy=True)
        empty_mask = celltype_maps_internal == -1
        celltype_maps_internal[empty_mask] = 0
        sctmap = cmap_internal(celltype_maps_internal)
        sctmap[empty_mask] = (0, 0, 0, 0)

        if set_alpha:
            score = self.max_correlations if getattr(self, "max_correlations", None) is not None \
                else getattr(self, "max_probabilities", None)
            if score is not None:
                alpha = np.array(score[..., z].T, copy=True)
                alpha[alpha < 0] = 0
                alpha = min_r + alpha / (np.max(alpha) / (1.0 - min_r))
                sctmap[..., 3] = alpha

        if rotate == 1 or rotate == 3:
            sctmap = sctmap.swapaxes(0, 1)


        extent = self._get_plot_extent()
        if extent is not None:
            xs = self.x_scale if self.x_scale is not None else 1.0   # store may not carry a scale (GridDataset load)
            ys = self.y_scale if self.y_scale is not None else 1.0
            extent[0] *= xs
            extent[1] *= xs
            extent[2], extent[3] = extent[3], extent[2]
            extent[2] *= ys
            extent[3] *= ys

        plt.gca().set_facecolor(background)
        plt.imshow(sctmap, interpolation='nearest', extent=extent)

        if rotate == 1:
            plt.gca().invert_yaxis()
        elif rotate == 2:
            plt.gca().invert_xaxis()
            plt.gca().invert_yaxis()
        elif rotate == 3:
            plt.gca().invert_xaxis()

    def plot_celltypes_map(self, background="black", centroid_indices=[], colors=None, cmap='jet', rotate=0, min_r=0.6, set_alpha=False, z=None, save_to_spatialdata=True, show_annotation=False, annotation_csv=None, n_celltypes=None):
        """Plot the filtered cell-type map.

        :param background: Background colour of the cell-type map.
        :type background: str or list(float)
        :param centroid_indices: The cell types to include. When not given, every cell type is
            drawn.
        :type centroid_indices: list(int)
        :param colors: Colour of the clusters, which overrides the ``cmap`` parameter.
        :type colors: list(str), list(list(float))
        :param cmap: Colormap for the clusters.
        :type cmap: str or matplotlib.colors.Colormap
        :param rotate: Rotate the plot. The possible values are 0, 1, 2 and 3.
        :type rotate: int
        :param min_r: Minimum correlation threshold for the cell-type map. This value affects
            the plot only and does not change the maps produced by ``filter_celltypemaps``.
        :type min_r: float
        :param set_alpha: Set the opacity of each pixel from its correlation. This is not
            fully implemented and does not behave correctly against a background other than
            black.
        :type set_alpha: bool
        :param z: Index of the layer to slice out of a three-dimensional cell-type map. When
            not given, the middle layer is used.
        :type z: int
        :param save_to_spatialdata: Whether to write the cell-type map into the SpatialData
            images as well, when the input was a SpatialData object.
        :type save_to_spatialdata: bool
        :param show_annotation: Whether to draw a legend of cell-type names, which also
            requires ``annotation_csv``.
        :type show_annotation: bool
        :param annotation_csv: Path to an annotation CSV file, which must have a ``cluster``
            and a ``celltype`` column and may have a ``global_cluster`` column.
        :type annotation_csv: str or None
        :param n_celltypes: Total number of globally unique cell types, used together with the
            ``global_cluster`` column to keep colours consistent across tiles. When ``None`` it
            is inferred as one more than the largest global cluster.
        :type n_celltypes: int or None
        """

        plot_colors = colors
        ann_df = None
        # Default palette/legend come from the store (written by map_celltypes),
        # indexed by celltype id. The legacy path below derives colours from the
        # RANK of each CSV row against `max(id)+1`, which mismatches whenever
        # filtering leaves the id space non-contiguous or short of K.
        store_names, store_colors = self.celltype_names_colors()
        if colors is None and annotation_csv is None and store_colors is not None:
            plot_colors = store_colors
        if annotation_csv is not None and colors is None:
            import pandas as pd
            ann_df = pd.read_csv(annotation_csv)
            if 'global_cluster' in ann_df.columns:
                cmap_internal = plt.get_cmap(cmap)
                if n_celltypes is not None:
                    n_global = n_celltypes
                elif 'n_global_celltypes' in ann_df.columns:
                    n_global = int(ann_df['n_global_celltypes'].iloc[0])
                else:
                    n_global = int(ann_df['global_cluster'].max() + 1)
                num_ctmaps = int(np.max(self.filtered_celltype_maps) + 1)
                sorted_rows = ann_df.sort_values('cluster')
                plot_colors = [None] * num_ctmaps
                for rank, (_, row) in enumerate(sorted_rows.iterrows()):
                    if rank < num_ctmaps:
                        global_id = int(row['global_cluster'])
                        plot_colors[rank] = cmap_internal(float(global_id) / max(n_global - 1, 1))
                plot_colors = [c if c is not None else background for c in plot_colors]

        self._plot_celltypes_map(self.filtered_celltype_maps, background, centroid_indices, plot_colors, cmap, rotate, min_r, set_alpha, z)

        if show_annotation and annotation_csv is None and store_names is not None:
            from matplotlib.patches import Patch
            present = [i for i in np.unique(self.filtered_celltype_maps) if i >= 0]
            pal = plot_colors if plot_colors is not None else store_colors
            plt.legend(handles=[Patch(facecolor=pal[i], label=store_names[i])
                                for i in present if i < len(store_names)],
                       bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        elif show_annotation and annotation_csv is not None:
            import pandas as pd
            from matplotlib.patches import Patch
            if ann_df is None:
                ann_df = pd.read_csv(annotation_csv)
            patches = []
            if colors is None:
                cmap_internal = plt.get_cmap(cmap)
                if 'global_cluster' in ann_df.columns:
                    if n_celltypes is not None:
                        n_global = n_celltypes
                    elif 'n_global_celltypes' in ann_df.columns:
                        n_global = int(ann_df['n_global_celltypes'].iloc[0])
                    else:
                        n_global = int(ann_df['global_cluster'].max() + 1)
                    for _, row in ann_df.sort_values('cluster').iterrows():
                        global_id = int(row['global_cluster'])
                        color = cmap_internal(float(global_id) / max(n_global - 1, 1))
                        patches.append(Patch(facecolor=color, label=row['celltype']))
                else:
                    num_ctmaps = int(np.max(self.filtered_celltype_maps) + 1)
                    for rank, (_, row) in enumerate(ann_df.sort_values('cluster').iterrows()):
                        color = cmap_internal(float(rank) / max(num_ctmaps - 1, 1))
                        patches.append(Patch(facecolor=color, label=row['celltype']))
            else:
                for _, row in ann_df.sort_values('cluster').iterrows():
                    cluster_idx = int(row['cluster'])
                    if cluster_idx < len(colors):
                        patches.append(Patch(facecolor=colors[cluster_idx], label=row['celltype']))
            plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        if (save_to_spatialdata and
            _SPATIALDATA_AVAILABLE and
            hasattr(self, '_input_data_type') and
            self._input_data_type == 'spatialdata' and
            hasattr(self, '_original_input_data')):

            self._save_celltype_map_to_spatialdata(
                self.filtered_celltype_maps,
                background,
                centroid_indices,
                plot_colors,
                cmap,
                rotate,
                min_r,
                set_alpha,
                z,
                image_name="celltype_map"
            )

        return

    def write_celltype_map(self, sdata=None, *, colors=None, cmap='jet',
                           background="white", z=None, min_r=0.6, set_alpha=False,
                           labels_key="ssaft_celltype_map",
                           image_key="ssaft_celltype_map_rgb"):
        """Write the filtered cell-type map into a SpatialData as two aligned rasters.

        This is normally reached through :meth:`Analysis.filter_celltypemaps`, which passes
        the palette fixed at ``map_celltypes`` time.

        The first raster, under ``labels_key``, is an integer one whose pixel value is the
        cell-type identifier plus one, with zero meaning background or filtered out. That is
        the idiomatic categorical representation of a segmentation-free cell-type map. A map
        with more than one layer is written whole, as a three-dimensional labels element
        indexed by depth, row and column, whose depth axis is the map's own depth in
        micrometres.

        The second raster, under ``image_key``, is an RGB rendering for quick visual
        inspection, showing layer ``z`` of a three-dimensional map. SpatialData keys are
        globally unique across element types, so the rendering cannot share the labels' key
        and takes a suffix instead.

        Both carry the same planar affine. A scale by the sampling distance followed by a
        translation to the origin takes a pixel into the map's micrometre frame, and that
        frame is carried into the global one by :meth:`_map_to_global_affine`. That is the
        transcripts element's own transformation when there is one, as on Xenium and
        MERSCOPE, where it maps micrometres to image pixels, and the identity otherwise, as
        on the count platforms, whose tables are already in global micrometres. The map
        therefore overlays the input table, points and staining image.

        No annotating table is written. The cell-type names live once, in the
        ``ssaft_downsampled_vectors`` table that :meth:`Analysis.map_celltypes` attaches,
        whose categorical is index-aligned with these pixel values, so that the name of a
        pixel is the entry one position below its value. Re-filtering therefore only ever
        replaces rasters.

        :param sdata: Target SpatialData to write the elements into. Defaults to the
            SpatialData remembered from ``compute_vf`` or ``downsample_vector``.
        :type sdata: spatialdata.SpatialData or None
        :param colors: Colour per cell type, indexed by identifier. Pass the palette stored at
            ``map_celltypes`` time so that every view agrees. Passing ``None`` samples ``cmap``
            over the identifiers present in the map.
        :type colors: list or None
        :param cmap: Colormap used for the clusters when ``colors`` is not given.
        :type cmap: str or matplotlib.colors.Colormap
        :param background: Colour the filtered-out voxels are composited onto. This defaults
            to white, because the rendering carries no alpha channel, so whatever is behind a
            background voxel becomes part of the picture, and a figure is read on a white
            page. Passing black restores the older look. Anything
            ``matplotlib.colors.to_rgba`` accepts will work.
        :type background: str or list(float)
        :param z: On a three-dimensional map, the layer rendered into the image element. It
            defaults to the middle one. The labels keep every layer.
        :type z: int or None
        :param min_r: Lowest opacity used when ``set_alpha`` scales opacity.
        :type min_r: float
        :param set_alpha: Whether to scale each pixel's opacity by its correlation or
            probability.
        :type set_alpha: bool
        :param labels_key: Key of the integer labels element written into ``sdata``.
        :type labels_key: str
        :param image_key: Key of the RGB image element written into ``sdata``.
        :type image_key: str
        :returns: The SpatialData object, with the two elements added.
        :rtype: spatialdata.SpatialData
        :raises ImportError: If spatialdata is not installed.
        :raises ValueError: If no SpatialData was given and none was remembered.
        """
        if not _SPATIALDATA_AVAILABLE:
            raise ImportError("spatialdata not available — cannot write celltype map")
        sdata = sdata if sdata is not None else getattr(self, "_original_input_data", None)
        if sdata is None:
            raise ValueError("no SpatialData given; pass sdata= or run compute_vf "
                             "with a SpatialData input so it is remembered")

        data = np.asarray(self.filtered_celltype_maps)
        if data.ndim == 2:                       # 2D map (no z axis) -> add a singleton z plane
            data = data[..., None]
        three_d = data.shape[2] > 1
        if z is None:
            z = int(data.shape[2] / 2)           # z from the MAP's own shape, not self.shape (vf_norm may be 2D)
        lab2d = np.asarray(data[..., z].T)
        n_present = int(data.max()) + 1 if (data.size and data.max() >= 0) else 0

        if three_d:                              # (x, y, z) -> (z, y, x), every layer
            labels_arr = np.ascontiguousarray(data.transpose(2, 1, 0), dtype=np.int32) + 1
        else:
            labels_arr = (lab2d + 1).astype(np.int32)

        # K counts every class the mapping can emit, not just the ones that won a
        # voxel here: `colors` is indexed BY CELLTYPE ID, so sizing the lookup by
        # lab2d.max() would silently recolour ids when filtering drops the tail.
        if colors is None:
            cmap_internal = plt.get_cmap(cmap)
            colors = cmap_internal([float(i) / max(n_present - 1, 1)
                                    for i in range(max(n_present, 1))])
        K = len(colors)
        lut = ListedColormap([to_rgba(c) for c in colors]) if K else ListedColormap(["black"])
        rgba = lut(np.clip(lab2d, 0, None))
        rgba[lab2d == -1] = (0, 0, 0, 0)
        _score = self.max_correlations if getattr(self, "max_correlations", None) is not None \
            else getattr(self, "max_probabilities", None)
        if set_alpha and _score is not None:
            a = np.asarray(_score[..., z].T, dtype=float); a[a < 0] = 0
            if a.max() > 0:
                rgba[..., 3] = min_r + a / (a.max() / (1.0 - min_r))
        # Composite over `background` instead of premultiplying alpha, which is the
        # same thing only when the background is black. Filtered-out voxels have
        # alpha 0 and so come out as the background colour exactly; with
        # set_alpha=True a weak call fades toward it rather than toward black.
        bg = np.asarray(to_rgba(background), dtype=float)[:3]
        rgb = rgba[..., :3] * rgba[..., 3:4] + bg * (1.0 - rgba[..., 3:4])
        rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
        cyx = np.moveaxis(rgb, -1, 0)

        zg = self.zarr_group
        sd_um = float(zg.attrs.get("sampling_distance",
                      (zg["vf_params"][0] if "vf_params" in zg else 1.0)))
        ox, oy = self._celltype_map_origin(sdata)
        cs, A = self._map_frame(sdata)
        if np.allclose(A, np.eye(3)):
            xs = float(getattr(self, "x_scale", 1.0) or 1.0)
            ys = float(getattr(self, "y_scale", 1.0) or 1.0)
        else:
            xs = ys = 1.0            # the points element's transformation already carries the scale
        mat = A @ np.array([[sd_um * xs, 0.0, ox], [0.0, sd_um * ys, oy], [0.0, 0.0, 1.0]])
        tf = Affine(mat, input_axes=("x", "y"), output_axes=("x", "y"))

        for key, store in ((labels_key, sdata.labels), (image_key, sdata.images)):
            if key in store:
                del store[key]
        if three_d:
            off = getattr(self, "coordinate_offset", None)
            if off is None and "coordinate_offset" in zg:
                off = zg["coordinate_offset"][:]
            oz = float(off[2]) if off is not None and len(off) > 2 else 0.0
            # rows/columns in (z, y, x) order; z stays in the map's µm, xy as the 2D affine
            m4 = np.array([[sd_um, 0.0, 0.0, oz],
                           [0.0, mat[1, 1], mat[1, 0], mat[1, 2]],
                           [0.0, mat[0, 1], mat[0, 0], mat[0, 2]],
                           [0.0, 0.0, 0.0, 1.0]])
            sdata.labels[labels_key] = Labels3DModel.parse(
                labels_arr, dims=("z", "y", "x"),
                chunks=(labels_arr.shape[0], 1024, 1024),
                transformations={cs: Affine(m4, input_axes=("z", "y", "x"),
                                                  output_axes=("z", "y", "x"))})
        else:
            sdata.labels[labels_key] = Labels2DModel.parse(
                labels_arr, dims=("y", "x"), transformations={cs: tf})
        sdata.images[image_key] = Image2DModel.parse(
            cyx, dims=("c", "y", "x"), transformations={cs: tf})
        print(f"[write_celltype_map] labels['{labels_key}'] {labels_arr.shape}"
              + (" (z, y, x)" if three_d else "")
              + f" + images['{image_key}'] " + (f"layer z={z} " if three_d else "")
              + f"({K} celltypes) @ {sd_um}µm/px, origin=({ox:.1f},{oy:.1f})")
        return sdata

    def _count_frame(self):
        """Return the reader frame ``compute_vf`` recorded for count input, if any.

        :returns: The recorded frame, or ``None`` when the input was not read by
            ``spatialdata_io`` or the frame was never recorded.
        :rtype: dict or None
        """
        zg = getattr(self, "zarr_group", None)
        if zg is None:
            return None
        f = zg.attrs.get("count_frame")
        return dict(f) if f else None

    def _map_frame(self, sdata):
        """Return the coordinate system and affine that carry the map into a SpatialData.

        Count input read straight from ``spatialdata_io`` has no global frame in micrometres,
        so the map goes back into the reader's own coordinate system, which ``compute_vf``
        recorded. Everything else is written into the global frame, as before.

        :param sdata: The SpatialData the map will be written into.
        :type sdata: spatialdata.SpatialData
        :returns: A pair holding the name of the target coordinate system and the 3 by 3
            affine into it.
        :rtype: tuple
        """
        f = self._count_frame()
        if f is not None:
            return f["coordinate_system"], np.asarray(f["um_to_cs"], dtype=float)
        return "global", self._map_to_global_affine(sdata)

    @staticmethod
    def _map_to_global_affine(sdata):
        """Return the affine from the cell-type map's micrometre frame to the global one.

        The transcript backend builds the map in the intrinsic coordinates of the points
        element it read, so that element's own transformation is what carries the map into the
        global frame. On Xenium and MERSCOPE that transformation maps micrometres to mosaic or
        morphology pixels. Count platforms have no points element, and their tables are
        already in global micrometres, so there the map needs no further transformation.

        :param sdata: The SpatialData the map will be written into.
        :type sdata: spatialdata.SpatialData
        :returns: The 3 by 3 affine, which is the identity when there is nothing to apply.
        :rtype: numpy.ndarray
        """
        from spatialdata.transformations import get_transformation
        points = getattr(sdata, "points", None) or {}
        if not len(points):
            return np.eye(3)
        t = get_transformation(points[next(iter(points))], "global")
        if t is None:
            return np.eye(3)
        return np.asarray(t.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")), dtype=float)

    def _celltype_map_origin(self, sdata):
        """Return the world origin of the vector-field lattice, in micrometres.

        Count platforms store it in the count table, under ``uns['cellman']['origin']``.
        Failing that, the recorded reader frame is consulted, then the dataset's own
        coordinate offset, and the origin falls back to zero.

        :param sdata: The SpatialData the map will be written into.
        :type sdata: spatialdata.SpatialData
        :returns: The origin as a pair of coordinates in micrometres.
        :rtype: tuple
        """
        try:
            for t in sdata.tables.values():
                cm = t.uns.get("cellman", {})
                if "origin" in cm:
                    o = cm["origin"]; return float(o[0]), float(o[1])
        except Exception:  # noqa: BLE001
            pass
        f = self._count_frame()
        if f is not None and "origin" in f:
            return float(f["origin"][0]), float(f["origin"][1])
        off = getattr(self, "coordinate_offset", None)
        if off is not None and len(off) >= 2:
            return float(off[0]), float(off[1])
        return 0.0, 0.0

    def celltype_names_colors(self):
        """Return the cell-type names and colours recorded in the store.

        These are written by ``map_celltypes`` and are the single source of cell-type naming
        and colour, shared by the labels rendering, the annotated table and any legend.

        :returns: A pair holding the names, indexed by cell-type identifier, and the colours.
            Both are ``None`` when the store holds no names, and the colours alone are
            ``None`` when no palette was recorded.
        :rtype: tuple
        """
        zg = self.zarr_group
        if 'celltype_names' not in zg:
            return None, None
        names = [str(x) for x in zg['celltype_names'][:]]
        colors = list(zg.attrs.get('cellman_celltype_colors', [])) or None
        return names, colors

    def _save_celltype_map_to_spatialdata(self, data, background="black",
                                          centroid_indices=[], colors=None, cmap='jet',
                                          rotate=0, min_r=0.6, set_alpha=False, z=None,
                                          image_name="ssaft_celltype_map"):
        """Write the cell-type map into the remembered SpatialData, ignoring any failure.

        ``plot_celltypes_map`` routes here when asked to save, and this method delegates to
        :meth:`write_celltype_map`, which writes the labels and the RGB rendering.

        :param data: Accepted for signature compatibility. The filtered map is read from the
            dataset instead.
        :param background: Colour the filtered-out voxels are composited onto.
        :param centroid_indices: Accepted for signature compatibility and unused.
        :param colors: Colour per cell type.
        :param cmap: Colormap used when no colours are given.
        :param rotate: Accepted for signature compatibility and unused.
        :param min_r: Lowest opacity used when ``set_alpha`` scales opacity.
        :param set_alpha: Whether to scale opacity by score.
        :param z: Layer rendered into the image element.
        :param image_name: Accepted for signature compatibility. The element keys are those of
            :meth:`write_celltype_map`.
        """
        try:
            self.write_celltype_map(colors=colors, cmap=cmap, background=background,
                                    z=z, min_r=min_r, set_alpha=set_alpha)
        except Exception as e:  # noqa: BLE001
            print(f"Warning: Failed to save celltype map to SpatialData: {e}")

    def plot_image(self, source, image_key='he', aspect='equal', **kwargs):
        """Display a staining or morphology image in global micrometre coordinates.

        Call this before :meth:`plot_downsampled_vector` in the same code cell, so that the
        two plots overlay on the same axes. Matplotlib then uses the same flipped convention
        along y, thanks to the inverted extent.

        The function accepts two forms of input.

        The first is a file path, as a string or a path object, which is opened with PIL and
        may be an OME-TIFF, a PNG or a JPEG. The image is assumed to span exactly the
        molecule bounding box recorded in ``original_coordinate_bounds``, which is the usual
        case for a cropped tile. Its extent in global micrometres is that box scaled by the
        dataset's own scales.

        The second is a SpatialData object, in which case the image element named by
        ``image_key`` is rendered over its own global extent, obtained through
        ``spatialdata.get_extent``. That respects whatever transformation lives on the
        element, which is the correct alignment with cell and molecule coordinates.

        :param source: Either a path to an image file, or a SpatialData object containing the
            image element.
        :type source: str, pathlib.Path, or spatialdata.SpatialData
        :param image_key: Image element key, used when ``source`` is a SpatialData object. Use
            ``'morphology_focus'`` on Xenium, ``'DAPI'`` on MERSCOPE or ``'hires_image'`` on
            Visium to render the morphology channel instead of the bright-field image. It is
            ignored when ``source`` is a path.
        :type image_key: str
        :param aspect: Aspect ratio passed to ``plt.imshow``.
        :type aspect: str
        :param kwargs: Further keyword arguments forwarded to ``plt.imshow``.
        """
        import matplotlib.pyplot as plt
        from pathlib import Path as _Path

        def _display_array(arr):
            """Put an image array into a layout ``imshow`` accepts.

            A channel-first array is moved to channel-last, a single channel is squeezed to a
            plain two-dimensional image, three channels are kept as RGB, and anything else is
            reduced to its first channel.

            :param arr: The image array.
            :returns: The array, ready to display.
            :rtype: numpy.ndarray
            """
            arr = np.asarray(arr)
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (3, 4):
                arr = np.moveaxis(arr, 0, -1)
            if arr.ndim == 3:
                if arr.shape[-1] == 1:
                    return arr[..., 0]
                if arr.shape[-1] == 3:
                    return arr
                if arr.shape[-1] == 4:
                    if arr.dtype == np.uint8:
                        return arr[..., :3]
                    return arr[..., 0]
            return arr

        is_path = isinstance(source, (str, _Path))

        if is_path:
            from PIL import Image as _Image
            he_img = np.array(_Image.open(str(source)).convert('RGB'))

            x_scale = self.x_scale if self.x_scale is not None else 1.0
            y_scale = self.y_scale if self.y_scale is not None else 1.0

            extent = None
            if hasattr(self, 'original_coordinate_bounds'):
                x_bounds = self.original_coordinate_bounds[0]
                y_bounds = self.original_coordinate_bounds[1]
                x_min = x_bounds[0] * x_scale
                x_max = x_bounds[1] * x_scale
                y_min = y_bounds[0] * y_scale
                y_max = y_bounds[1] * y_scale
                extent = [x_min, x_max, y_max, y_min]
        else:
            try:
                import spatialdata as _sd
            except ImportError as e:
                raise ImportError(
                    "plot_he with a SpatialData input requires the spatialdata "
                    "package."
                ) from e
            try:
                import spatialdata_plot  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "plot_he with a SpatialData input requires the "
                    "spatialdata-plot package."
                ) from e
            if not isinstance(source, _sd.SpatialData):
                raise TypeError(
                    f"plot_he source must be a path or SpatialData, "
                    f"got {type(source).__name__}."
                )
            if image_key not in source.images:
                raise KeyError(
                    f"sdata.images has no key '{image_key}'. "
                    f"Available: {list(source.images.keys())}"
                )

            title              = kwargs.pop('title', image_key)
            coordinate_systems = kwargs.pop('coordinate_systems', 'global')
            figsize            = kwargs.pop('figsize', None)
            ax                 = kwargs.pop('ax', None)

            _render_keys = {'channel', 'cmap', 'norm', 'na_color',
                            'palette', 'alpha', 'scale'}
            render_kwargs = {k: kwargs.pop(k) for k in list(kwargs)
                             if k in _render_keys}

            if ax is None:
                if figsize is not None:
                    fig = plt.figure(figsize=figsize)
                    ax = fig.add_subplot(111)
                else:
                    ax = plt.gca()

            source.pl.render_images(image_key, **render_kwargs).pl.show(
                title=title,
                coordinate_systems=coordinate_systems,
                ax=ax,
                **kwargs,
            )
            ax.set_aspect(aspect)
            plt.sca(ax)
            return

        plt.imshow(he_img, extent=extent, aspect=aspect, **kwargs)

    plot_he = plot_image

    def plot_downsampled_vector(self, c='red', s=0.3, alpha=0.45, **kwargs):
        """Scatter the sampled lattice positions in micrometre coordinates.

        The positions are the local maxima that the tissue mask guided the sampling to. Call
        :meth:`plot_image` first in the same code cell to overlay them on the staining image.
        ``downsample_vector`` must have been run first.

        :param c: Colour of the scatter dots.
        :type c: color
        :param s: Size of the scatter dots.
        :type s: float
        :param alpha: Transparency of the scatter dots.
        :type alpha: float
        :param kwargs: Further keyword arguments forwarded to ``plt.scatter``.
        :raises ValueError: If no positions have been sampled yet.
        """
        import matplotlib.pyplot as plt

        if self.downsampled_vectors is None:
            raise ValueError('downsampled_vectors not set. Run downsample_vector() first.')

        plot_coords = self.downsampled_vectors_original_coords
        if plot_coords is None:
            raise ValueError('downsampled_vectors not set. Run downsample_vector() first.')

        x_coords, y_coords = plot_coords[0], plot_coords[1]

        M = getattr(self, '_global_affine', None)
        if M is not None:
            px = M[0, 0] * x_coords + M[0, 1] * y_coords + M[0, 2]
            py = M[1, 0] * x_coords + M[1, 1] * y_coords + M[1, 2]
        else:
            x_scale = self.x_scale if self.x_scale is not None else 1.0
            y_scale = self.y_scale if self.y_scale is not None else 1.0
            px = x_coords * x_scale
            py = y_coords * y_scale

        plt.scatter(px, py, s=s, c=c, alpha=alpha, **kwargs)
