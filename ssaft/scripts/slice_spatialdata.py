"""Slice a SpatialData zarr into a tile grid; save each tile as a separate zarr.

Usage:
    .venv/bin/python scripts/slice_spatialdata.py \\
        --sdata-path  dataset/<DATASET>/ssaft_sdata.zarr \\
        --tile-size   10000 \\
        --output-dir  dataset/<DATASET>/spatialdata_crops
"""
import argparse
import json
import numpy as np
import os
import sys
import dask.array as da
import pandas as pd
import xarray as xr
import spatialdata as sd
from spatialdata import get_extent, bounding_box_query
from spatialdata.models import Image2DModel, Image3DModel, Labels2DModel, Labels3DModel, PointsModel
from spatialdata.transformations import Affine, Identity, get_transformation

#: Where the transcripts' ``z`` waits while the tile is cut out. ``bounding_box_query`` scans
#: a points frame's columns for axis names, and a column called ``z`` against a 2D
#: transformation is an unsupported case, so the depth is carried under a name that is
#: not an axis and renamed back before the tile is written.
Z_STASH = '_z_depth'


def parse_args():
    """Parse the command-line arguments of this script.

    :returns: The parsed arguments, carrying the input store, the tile size, the output
        directory and the flags controlling which elements each tile keeps.
    :rtype: argparse.Namespace
    """
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        '--sdata-path', required=True,
        help='Path to the input SpatialData zarr.',
    )
    p.add_argument(
        '--tile-size', type=float, required=True,
        help='Tile size in global coordinate units.',
    )
    p.add_argument(
        '--output-dir', required=True,
        help='Output directory for r{row:02d}_c{col:02d}.zarr tiles.',
    )
    p.add_argument(
        '--min-tile-size', type=float, default=None,
        help='Minimum size for the trailing row/col (default: tile_size / 2). '
             'If the last edge segment is smaller than this it is merged into '
             'the previous tile.',
    )
    p.add_argument(
        '--coordinate-system', default='global',
        help="SpatialData coordinate system to use (default: 'global').",
    )
    p.add_argument(
        '--image-key', nargs='+', default=None,
        help='Image keys to keep (e.g. the 2D staining image and, for '
             'downsample_vector(z=3), the 3D z-stack element: --image-key morphology_mip '
             'morphology_zstack). If set, all other image elements are dropped from each tile.',
    )
    p.add_argument(
        '--keep-points', nargs='*', default=None,
        help='Whitespace-separated points keys to keep. Default: keep all.',
    )
    p.add_argument(
        '--drop-labels', action='store_true',
        help='Drop all labels elements from each tile.',
    )
    p.add_argument(
        '--drop-shapes', action='store_true',
        help='Drop all shapes elements from each tile.',
    )
    p.add_argument(
        '--drop-tables', action='store_true',
        help='Drop all tables elements from each tile.',
    )
    p.add_argument(
        '--drop-z', action='store_true',
        help="Drop the transcripts' z column from each tile, as earlier versions did. "
             'By default z is kept, which Analysis.downsample_vector(z=3, chunked=True) '
             'needs.',
    )
    return p.parse_args()


def _highest_res_dataarray(tree):
    """Return the highest-resolution DataArray of a multiscale DataTree.

    The node named ``scale0`` is preferred, and the first node is used when there is none.

    :param tree: The multiscale element.
    :returns: The DataArray holding the finest level.
    :rtype: xarray.DataArray
    """
    children = list(tree.items())
    pick_name = 'scale0' if 'scale0' in dict(children) else children[0][0]
    sub = tree[pick_name] if pick_name in tree else children[0][1]
    ds = sub.ds
    var_name = list(ds.data_vars)[0]
    return ds[var_name]


def _rechunk_dataarray(xda, target_chunk=2048):
    """Give a DataArray regular chunks along its spatial axes.

    The y and x axes are chunked to at most ``target_chunk``, while the channel and z
    axes are left whole. An array that is not backed by dask is returned unchanged.

    :param xda: The array to rechunk.
    :type xda: xarray.DataArray
    :param target_chunk: Largest chunk size along y and x.
    :type target_chunk: int
    :returns: The rechunked array, carrying the original dimensions, coordinates,
        attributes and name.
    :rtype: xarray.DataArray
    """
    arr = xda.data
    if not isinstance(arr, da.Array):
        return xda
    new_chunks = []
    for dim in xda.dims:
        size = xda.sizes[dim]
        if size <= 0:
            new_chunks.append(1)
            continue
        if dim in ('y', 'x'):
            new_chunks.append(min(target_chunk, size))
        elif dim in ('c', 'z'):
            new_chunks.append(size)
        else:
            new_chunks.append(size)
    rechunked = arr.rechunk(tuple(new_chunks))
    return xr.DataArray(rechunked, dims=xda.dims, coords=xda.coords, attrs=xda.attrs, name=xda.name)


def rebuild_raster(elem, kind):
    """Rebuild a raster element so that it has regular chunks.

    Any multiscale pyramid is dropped and only the finest level is kept. The element's
    transformations are carried over unchanged.

    :param elem: The image or labels element to rebuild.
    :param kind: Either ``'image'`` or anything else, which is treated as labels. The
        choice selects the SpatialData model, and the presence of a z axis selects between
        its two-dimensional and three-dimensional form.
    :type kind: str
    :returns: The rebuilt element.
    """
    try:
        transformations = get_transformation(elem, get_all=True)
        if not isinstance(transformations, dict):
            transformations = {'global': transformations}
    except Exception:
        transformations = None

    if isinstance(elem, xr.DataArray):
        xda = elem
    else:
        xda = _highest_res_dataarray(elem)

    new_xda = _rechunk_dataarray(xda)
    dims = tuple(new_xda.dims)
    if kind == 'image':
        model_cls = Image3DModel if 'z' in dims else Image2DModel
    else:
        model_cls = Labels3DModel if 'z' in dims else Labels2DModel

    parsed = model_cls.parse(new_xda.data, dims=dims, transformations=transformations)
    return parsed


def rechunk_all_rasters(cropped_sdata):
    """Rebuild every image and labels element of a tile so that its chunks are regular.

    The elements are replaced in place.

    :param cropped_sdata: The cropped tile to rewrite.
    :type cropped_sdata: spatialdata.SpatialData
    """
    for k in list(cropped_sdata.images.keys()):
        cropped_sdata.images[k] = rebuild_raster(cropped_sdata.images[k], 'image')
    for k in list(cropped_sdata.labels.keys()):
        cropped_sdata.labels[k] = rebuild_raster(cropped_sdata.labels[k], 'label')


def strip_points_meta_transform(cropped_sdata):
    """Rebuild each points element so that it can be written to parquet.

    This works around a serialization bug in the ``io_points`` module of spatialdata.
    Writing the points triggers a dask computation that reaches
    ``pa.Table.from_pandas(partition)``. pyarrow then tries to encode
    ``partition.attrs`` as JSON for the pandas metadata of the parquet file, and chokes on
    an ``Identity`` transformation. spatialdata deletes ``attrs['transform']`` only on the
    wrapper copy, not on ``_meta`` and not on the partition payload already baked into the
    dask expression graph.

    The fix is to rebuild the dask DataFrame: compute it to pandas, clear every attribute,
    and parse it again through ``PointsModel`` with the same transformation. Points cropped
    to one tile should already fit in memory, so computing them is acceptable.

    The depth column set aside by :func:`normalize_points_indices` is renamed back to
    ``z`` here, so that each written tile is three-dimensional again.

    :param cropped_sdata: The cropped tile whose points are rebuilt in place.
    :type cropped_sdata: spatialdata.SpatialData
    """
    from spatialdata.models import PointsModel
    from spatialdata.transformations import get_transformation

    for k, p in list(cropped_sdata.points.items()):
        try:
            tx = get_transformation(p, get_all=True)
            if not isinstance(tx, dict):
                tx = {'global': tx}
        except Exception:
            tx = None

        spatialdata_attrs = p.attrs.get('spatialdata_attrs', {}) if hasattr(p, 'attrs') else {}
        feature_key = spatialdata_attrs.get('feature_key')

        pdf = p.compute()
        pdf.attrs = {}
        pdf = pdf.reset_index(drop=True)
        coords = {'x': 'x', 'y': 'y'}
        if Z_STASH in pdf.columns:
            # the depth set aside for the crop (see normalize_points_indices) goes back
            pdf = pdf.rename(columns={Z_STASH: 'z'})
            coords['z'] = 'z'

        kw = {'coordinates': coords, 'transformations': tx}
        if feature_key is not None and feature_key in pdf.columns:
            kw['feature_key'] = feature_key
        cropped_sdata.points[k] = PointsModel.parse(pdf, **kw)


def points_count(p):
    """Count the rows of a points element, without failing on an unusual one.

    The length is read from the shape, computing it when the shape is lazy, and falls back
    to the built-in length. Anything that raises yields zero rather than propagating.

    :param p: The points element.
    :returns: The number of rows, or zero when it could not be determined.
    :rtype: int
    """
    try:
        n = int(p.shape[0].compute()) if hasattr(p.shape[0], 'compute') else int(p.shape[0])
        return n
    except Exception:
        try:
            return len(p)
        except Exception:
            return 0


def slim_cropped(cropped, image_key=None, keep_points=None,
                 drop_labels=False, drop_shapes=False, drop_tables=False):
    """Drop the unused elements of a cropped tile, in place.

    This reduces the footprint of each tile on disk when the downstream consumer needs
    only a subset of the elements, for example the chosen images together with the points
    for ``Analysis.downsample_vector``.

    :param cropped: The cropped tile to slim.
    :type cropped: spatialdata.SpatialData
    :param image_key: Image key, or several, to keep. Every other image is dropped.
        Passing ``None`` keeps them all.
    :type image_key: str or iterable of str, optional
    :param keep_points: Points keys to keep. Every other points element is dropped.
        Passing ``None`` keeps them all.
    :type keep_points: iterable of str, optional
    :param drop_labels: Whether to drop every labels element.
    :type drop_labels: bool
    :param drop_shapes: Whether to drop every shapes element.
    :type drop_shapes: bool
    :param drop_tables: Whether to drop every tables element.
    :type drop_tables: bool
    :returns: The same tile, slimmed.
    :rtype: spatialdata.SpatialData
    :raises KeyError: If a requested image key is not present after the crop.
    """
    if drop_labels:
        for k in list(cropped.labels.keys()):
            del cropped.labels[k]
    if drop_shapes:
        for k in list(cropped.shapes.keys()):
            del cropped.shapes[k]
    if drop_tables:
        for k in list(cropped.tables.keys()):
            del cropped.tables[k]
    if image_key is not None:
        keep_images = {image_key} if isinstance(image_key, str) else set(image_key)
        missing = sorted(keep_images - set(cropped.images.keys()))
        if missing:
            raise KeyError(
                f"--image-key {missing} not found after crop. "
                f"Available images: {list(cropped.images.keys())}"
            )
        for k in list(cropped.images.keys()):
            if k not in keep_images:
                del cropped.images[k]
    if keep_points is not None:
        keep_set = set(keep_points)
        for k in list(cropped.points.keys()):
            if k not in keep_set:
                del cropped.points[k]
    return cropped


def has_biological_data(cropped):
    """Report whether a tile holds anything worth writing.

    A tile is meaningful only when it contains transcripts or cells, meaning a non-empty
    points element or a non-empty table.

    :param cropped: The cropped tile, which may be ``None``.
    :type cropped: spatialdata.SpatialData or None
    :returns: ``True`` when the tile holds data, ``False`` otherwise.
    :rtype: bool
    """
    if cropped is None:
        return False
    for k, p in cropped.points.items():
        if points_count(p) > 0:
            return True
    for k, t in cropped.tables.items():
        if t.n_obs > 0:
            return True
    return False


def normalize_points_indices(sdata, keep_z=True):
    """Rewrite every points element with a clean global range index.

    Two problems have to be fixed before ``get_extent`` or ``bounding_box_query`` is
    called.

    The first is duplicate labels. A Vizgen MERSCOPE ``detected_transcripts`` parquet file
    is read as a dask DataFrame of many partitions, each carrying its own range index from
    zero, so labels repeat across partitions. ``spatialdata.transform`` performs
    ``pd.Series(values, index=transformed.index)``, which materializes that index and
    raises a ValueError about reindexing on an axis with duplicate labels. Both
    ``get_extent`` and ``bounding_box_query`` go through ``transform``.

    The second is a mismatch between three-dimensional data and a two-dimensional
    transformation. The source has an x, y and z column trio while the transformation
    stored on disk is planar. ``PointsModel.parse`` reports that the z column will be
    ignored, but the column survives, and ``get_axes_names`` later scans the columns and
    reports three dimensions. ``bounding_box_query`` then hits its unsupported case of
    three-dimensional data against a two-dimensional transformation. The crop itself needs
    only x and y, so z is set aside under :data:`Z_STASH`, a name no axis scan picks up,
    and :func:`strip_points_meta_transform` restores it on each tile. Earlier versions
    dropped the column instead, which left every tile planar and made
    ``Analysis.downsample_vector(z=3, chunked=True)`` impossible. Passing ``keep_z`` as
    ``False``, which the ``--drop-z`` option does, still drops it.

    The DataFrame is materialized to pandas once, which for MERSCOPE transcripts is
    typically a few hundred megabytes and fits in memory. Computing it also forces the
    gene column's categorical dtype to have known categories, which silences the
    performance warning ``PointsModel.parse`` otherwise emits on a partially known
    categorical.

    :param sdata: The SpatialData whose points are rewritten in place.
    :type sdata: spatialdata.SpatialData
    :param keep_z: Whether to set the depth aside and keep it, rather than dropping it.
    :type keep_z: bool
    """
    def _project_to_2d(t):
        """Reduce a transformation to its planar part.

        An identity stays an identity, an affine is reduced to the x and y axes, and anything
        that cannot be reduced falls back to an identity.

        :param t: The transformation to project.
        :returns: The planar transformation.
        """
        if isinstance(t, Identity):
            return Identity()
        try:
            m2d = t.to_affine_matrix(input_axes=('x', 'y'), output_axes=('x', 'y'))
            return Affine(m2d, input_axes=('x', 'y'), output_axes=('x', 'y'))
        except Exception:
            return Identity()

    for k, p in list(sdata.points.items()):
        try:
            tx = get_transformation(p, get_all=True)
            if not isinstance(tx, dict):
                tx = {'global': tx}
        except Exception:
            tx = None

        if tx is not None:
            tx = {cs: _project_to_2d(t) for cs, t in tx.items()}

        sdata_attrs = p.attrs.get('spatialdata_attrs', {}) if hasattr(p, 'attrs') else {}
        feature_key = sdata_attrs.get('feature_key')

        pdf = p.compute()
        if 'z' in pdf.columns:
            pdf = (pdf.rename(columns={'z': Z_STASH}) if keep_z
                   else pdf.drop(columns=['z']))
        pdf = pdf.reset_index(drop=True)
        pdf.attrs = {}

        kw = {'coordinates': {'x': 'x', 'y': 'y'}, 'transformations': tx}
        if feature_key is not None and feature_key in pdf.columns:
            kw['feature_key'] = feature_key
        sdata.points[k] = PointsModel.parse(pdf, **kw)


def main():
    """Cut the input store into a grid of tiles and write each tile as its own zarr.

    The global extent is measured in the chosen coordinate system and divided into tiles
    of the requested size. A trailing row or column smaller than the minimum is merged
    into the one before it. Each tile is cropped by bounding box, slimmed to the requested
    elements, rechunked, and written together with a small JSON file recording its source,
    its grid position and its bounding box. A tile holding neither transcripts nor cells
    is skipped.
    """
    args = parse_args()
    SDATA_PATH = args.sdata_path
    TILE_SIZE = float(args.tile_size)
    OUTPUT_DIR = args.output_dir
    MIN_TILE_SIZE = (
        float(args.min_tile_size) if args.min_tile_size is not None else TILE_SIZE / 2.0
    )
    COORDINATE_SYSTEM = args.coordinate_system

    print(f'SDATA_PATH    : {SDATA_PATH}')
    print(f'TILE_SIZE     : {TILE_SIZE}')
    print(f'OUTPUT_DIR    : {OUTPUT_DIR}')
    print(f'MIN_TILE_SIZE : {MIN_TILE_SIZE}')
    print(f'COORD_SYSTEM  : {COORDINATE_SYSTEM}\n')

    sdata = sd.read_zarr(SDATA_PATH)

    print('SpatialData loaded:')
    print(f'  images : {list(sdata.images.keys())}')
    print(f'  labels : {list(sdata.labels.keys())}')
    print(f'  points : {list(sdata.points.keys())}')
    print(f'  shapes : {list(sdata.shapes.keys())}')
    print(f'  tables : {list(sdata.tables.keys())}')

    normalize_points_indices(sdata, keep_z=not args.drop_z)
    print(f"  points z   : {'dropped' if args.drop_z else 'kept'}")

    try:
        extent = get_extent(sdata, coordinate_system=COORDINATE_SYSTEM)
    except ValueError as e:
        if 'duplicate labels' not in str(e):
            raise
        print(f'  WARN: get_extent failed on points ({e}); '
              f'recomputing from images/labels/shapes only.', flush=True)
        extent = get_extent(
            sdata, coordinate_system=COORDINATE_SYSTEM, has_points=False
        )
    x_min_global, x_max_global = extent['x']
    y_min_global, y_max_global = extent['y']

    x_edges = np.arange(x_min_global, x_max_global, TILE_SIZE).tolist() + [x_max_global]
    y_edges = np.arange(y_min_global, y_max_global, TILE_SIZE).tolist() + [y_max_global]

    if len(x_edges) > 2 and (x_edges[-1] - x_edges[-2]) < MIN_TILE_SIZE:
        x_edges.pop(-2)
    if len(y_edges) > 2 and (y_edges[-1] - y_edges[-2]) < MIN_TILE_SIZE:
        y_edges.pop(-2)

    N_COLS = len(x_edges) - 1
    N_ROWS = len(y_edges) - 1

    print(f'\nGlobal extent: x=[{x_min_global:.2f}, {x_max_global:.2f}], y=[{y_min_global:.2f}, {y_max_global:.2f}]')
    print(f'x_edges: {[f"{v:.1f}" for v in x_edges]}')
    print(f'y_edges: {[f"{v:.1f}" for v in y_edges]}')
    print(f'Grid: {N_ROWS} rows x {N_COLS} cols = {N_ROWS * N_COLS} tiles\n')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    saved = 0
    skipped = 0
    saved_files = []

    for row in range(N_ROWS):
        for col in range(N_COLS):
            x0, x1 = x_edges[col], x_edges[col + 1]
            y0, y1 = y_edges[row], y_edges[row + 1]
            out_path = os.path.join(OUTPUT_DIR, f'r{row:02d}_c{col:02d}.zarr')
            print(f'[{row},{col}] x=[{x0:.2f},{x1:.2f}], y=[{y0:.2f},{y1:.2f}] -> {out_path}', flush=True)

            try:
                cropped = bounding_box_query(
                    sdata,
                    axes=('x', 'y'),
                    min_coordinate=[x0, y0],
                    max_coordinate=[x1, y1],
                    target_coordinate_system=COORDINATE_SYSTEM,
                )
            except Exception as e:
                print(f'  ERROR during bounding_box_query: {type(e).__name__}: {e}', flush=True)
                raise

            if not has_biological_data(cropped):
                npts = sum(points_count(p) for p in cropped.points.values()) if cropped is not None else 0
                ntab = sum(t.n_obs for t in cropped.tables.values()) if cropped is not None else 0
                print(f'  -> empty (transcripts={npts}, cells={ntab}), skipping', flush=True)
                skipped += 1
                continue

            slim_cropped(
                cropped,
                image_key=args.image_key,
                keep_points=args.keep_points,
                drop_labels=args.drop_labels,
                drop_shapes=args.drop_shapes,
                drop_tables=args.drop_tables,
            )
            rechunk_all_rasters(cropped)
            strip_points_meta_transform(cropped)

            try:
                cropped.write(out_path, overwrite=True)
            except Exception as e:
                print(f'  ERROR during write: {type(e).__name__}: {e}', flush=True)
                raise

            try:
                written = sd.read_zarr(out_path)
                npts = sum(points_count(p) for p in written.points.values())
                ncells = sum(t.n_obs for t in written.tables.values())
                print(f'  -> saved (images={list(written.images.keys())}, labels={list(written.labels.keys())}, '
                      f'points={list(written.points.keys())}, shapes={list(written.shapes.keys())}, '
                      f'tables={list(written.tables.keys())}, transcripts={npts}, cells={ncells})', flush=True)
            except Exception as e:
                print(f'  -> saved but failed to verify: {e}', flush=True)

            side = dict(source=os.path.abspath(SDATA_PATH), row=row, col=col,
                        coordinate_system=COORDINATE_SYSTEM,
                        bbox=dict(x=[float(x0), float(x1)], y=[float(y0), float(y1)]),
                        points_z_kept=not args.drop_z,
                        images=list(cropped.images.keys()))
            with open(out_path + '.json', 'w') as fh:
                json.dump(side, fh, indent=1)

            saved += 1
            saved_files.append(os.path.basename(out_path))

    print(f'\nDone.')
    print(f'  Total grid cells : {N_ROWS * N_COLS}')
    print(f'  Tiles saved      : {saved}')
    print(f'  Tiles skipped    : {skipped}')
    print(f'  Output directory : {OUTPUT_DIR}')
    print(f'\nSaved files:')
    for f in saved_files:
        print(f'  {f}')


if __name__ == '__main__':
    main()
