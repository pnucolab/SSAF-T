"""Raw expression, laid out so that ONE GENE is one cheap read.

An array platform (Visium HD, Stereo-seq) measures a count per lattice bin rather than
single transcripts, and the bin x gene matrix a vendor writes is CSR: its rows are bins,
so one gene's column is scattered over every row of it. Reading a single gene means
touching the whole matrix -- a hundred million entries for one picture -- which is why
the viewer used to draw a gene from the KDE signal field (``ssaft_downsampled_vectors``)
instead: dense, so a gene was one column, but SMOOTHED, DOWNSAMPLED, and cut to the
signal genes.

This writes the raw counts in the layout the question actually wants: gene-major, on the
cell-type map's OWN lattice. A gene becomes one contiguous run of (cell, count), so
drawing it costs a slice rather than a scan, with nothing averaged away and every gene of
the panel kept. The lattice is the map's full one, so a reader that wants coarser bins
(the 8 um a Visium HD reader expects) divides the cell index down and sums -- no second
copy on disk.

The group sits at the store ROOT, beside ``images/``, ``labels/``, ``shapes/`` and
``tables/``, and is NOT a SpatialData element: it is an index, not something to draw or
to carry transformations, and a viewer that lists the store's elements must not offer it
as a layer. ``spatialdata.read_zarr`` keeps it untouched.

Layout, under ``ssaft_gene_bins/``::

    genes     vlen-utf8 (n_gene,)    the panel, in the order the runs are stored
    indptr    int64     (n_gene+1,)  gene g owns cell/count[indptr[g]:indptr[g+1]]
    cell      int32     (nnz,)       index into the lattice, row * w + col,
                                     ASCENDING inside a gene (delta-coded on disk)
    count     uint8     (nnz,)       the raw count in that bin
    total     uint16    (w*h,)       every gene's counts summed, per lattice cell

and ``attrs['ssaft']`` carries the lattice the cell index refers to.
"""

import json
import os
import shutil

import numpy as np

__all__ = ["write_gene_bins", "GENE_BINS_KEY"]

GENE_BINS_KEY = "ssaft_gene_bins"
FORMAT_VERSION = 1
_CHUNK = 1 << 19


def _zstd(level=5):
    return {"name": "zstd", "configuration": {"level": level, "checksum": False}}


def _consolidate_into_root(path, key, children):
    """Put the new group into the store's consolidated index, and change nothing else.

    A SpatialData store keeps an inline index of every node at its root, and a reader
    that trusts it -- zarr-python in read mode, and the browser reader the portal ships
    -- cannot see a group that is missing from it, however plainly the group sits on
    disk. Rewriting the whole index with ``zarr.consolidate_metadata`` would rebuild
    entries SpatialData wrote, so the entries are inserted one by one instead: an array's
    entry IS its own ``zarr.json``, and a group's is that plus an empty inline index.
    """
    import pathlib

    root = pathlib.Path(path) / "zarr.json"
    meta = json.loads(root.read_text())
    index = meta.get("consolidated_metadata")
    if not index or index.get("kind") != "inline":
        return False                       # nothing consolidated: readers walk the store
    own = json.loads((pathlib.Path(path) / key / "zarr.json").read_text())
    own["consolidated_metadata"] = {"kind": "inline", "must_understand": False, "metadata": {}}
    index["metadata"][key] = own
    for child in children:
        index["metadata"][f"{key}/{child}"] = json.loads(
            (pathlib.Path(path) / key / child / "zarr.json").read_text())
    tmp = root.with_suffix(".json.ssaft-tmp")
    tmp.write_text(json.dumps(meta))
    tmp.replace(root)                      # in place, so a reader never sees a half file
    return True


def _map_lattice(sdata, element):
    """The (w, h, voxel, origin) of the lattice a labels element is DRAWN on.

    Read through SpatialData's own transformation rather than the raw matrix, so a bin lands
    where the map it is laid over lands.

    A map registered onto a slide is not square with the world: Space Ranger's own stores
    carry a scale of about 7.3, a mirrored x, and half a degree of rotation.  The lattice
    taken here is the axis-aligned box the placed map covers, at the map's own pixel pitch --
    the same straightening a viewer must do to draw it at all -- so a turned map is laid out
    rather than refused.

    The frame is usually called ``global``, but a store written straight from Space Ranger
    names it after the run (``Visium_HD_3prime_Mouse_Brain``), so the name is read off the
    element rather than assumed.
    """
    from spatialdata.transformations import get_transformation

    el = sdata.labels[element]
    arr = el["scale0"] if hasattr(el, "children") else el
    arr = next(iter(arr.values())) if hasattr(arr, "values") and not hasattr(arr, "shape") else arr
    dims = {d: int(n) for d, n in zip(arr.dims, arr.shape)}
    systems = get_transformation(el, get_all=True)
    frame = "global" if "global" in systems else (
        next(iter(systems)) if len(systems) == 1 else None)
    if frame is None:
        raise ValueError(f"{element} is in several frames ({sorted(systems)}); none is `global`")
    m = get_transformation(el, frame).to_affine_matrix(
        input_axes=("x", "y"), output_axes=("x", "y"))
    w, h = dims["x"], dims["y"]
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], float) @ m[:2, :].T
    ox, oy = corners.min(axis=0)
    ex, ey = corners.max(axis=0)
    vx = float(np.hypot(m[0, 0], m[1, 0]))       # a column is where one pixel of x goes
    vy = float(np.hypot(m[0, 1], m[1, 1]))
    if not (vx > 0 and vy > 0):
        raise ValueError(f"{element} has no scale in `{frame}`")
    return dict(w=int(np.ceil((ex - ox) / vx)), h=int(np.ceil((ey - oy) / vy)),
                vx=vx, vy=vy, ox=float(ox), oy=float(oy))


def _counts_csr(table):
    """(indptr, gene index, count) of an AnnData whose X is a CSR count matrix."""
    X = table.X
    if not hasattr(X, "indptr") or getattr(X, "format", None) not in (None, "csr"):
        raise ValueError("X is not CSR; this expects the vendor's bin x gene matrix")
    data = np.asarray(X.data)
    if np.any(data < 0) or not np.all(data == np.floor(data)):
        raise ValueError("X does not hold whole counts")
    return np.asarray(X.indptr), np.asarray(X.indices), data


def write_gene_bins(sdata, *, table="table", map_element=None, store=None,
                    key=GENE_BINS_KEY, quiet=False):
    """Write ``ssaft_gene_bins`` for an array-platform store.

    :param sdata: the store, already written to disk, whose counts are to be indexed.
    :type sdata: spatialdata.SpatialData
    :param table: the table holding the vendor's raw bin x gene counts.
    :type table: str
    :param map_element: the labels element whose lattice the bins are laid on; the first
        ``ssaft_celltype_map`` when not given.
    :type map_element: str or None
    :param store: the zarr store to write into; ``sdata.path`` when not given.
    :param key: the root group to write.
    :type key: str
    :returns: the lattice the index was written on.
    :rtype: dict
    """
    import zarr
    from numcodecs.zarr3 import Delta

    adata = sdata.tables[table]
    if map_element is None:
        maps = [k for k in sdata.labels if k.startswith("ssaft_celltype_map")]
        if not maps:
            raise ValueError("no ssaft_celltype_map in this store; pass map_element")
        map_element = maps[0]
    lat = _map_lattice(sdata, map_element)

    indptr, gene_of, count = _counts_csr(adata)
    n_bin, n_gene = adata.shape
    xy = np.asarray(adata.obsm["spatial"], np.float64)

    # every nonzero's bin, then that bin's cell in the lattice; a bin off the map is
    # dropped rather than folded onto an edge cell
    per_bin = np.diff(indptr).astype(np.int64)
    ix = np.floor((xy[:, 0] - lat["ox"]) / lat["vx"]).astype(np.int64)
    iy = np.floor((xy[:, 1] - lat["oy"]) / lat["vy"]).astype(np.int64)
    on = (ix >= 0) & (iy >= 0) & (ix < lat["w"]) & (iy < lat["h"])
    cell_of_bin = np.where(on, iy * lat["w"] + ix, -1)
    cell = np.repeat(cell_of_bin, per_bin)

    keep = cell >= 0
    off = float((~keep).mean()) if keep.size else 0.0
    if off > 0.01:
        raise ValueError(f"{off:.0%} of the counts fall outside {map_element}'s lattice; the table "
                         f"and the map are not in the same frame")
    if not keep.all():
        cell, gene_of, count = cell[keep], gene_of[keep], count[keep]

    # gene-major: order by gene, and inside a gene by cell, which is what lets the cell
    # index delta-code to almost nothing and what makes one gene one slice
    order = np.lexsort((cell, gene_of))
    cell = cell[order]
    gene_of = gene_of[order]
    count = count[order]

    # A lattice cell can hold MORE THAN ONE bin -- a Stereo-seq run puts 42 million bins on a
    # 37-million-cell map -- and their counts are added rather than left as separate entries.
    # Left alone, a reader that lays the run out as `grid[cell] = count` would keep whichever
    # entry it wrote last and report a fraction of what was measured there.
    if cell.size:
        first = np.empty(cell.size, bool)
        first[0] = True
        first[1:] = (cell[1:] != cell[:-1]) | (gene_of[1:] != gene_of[:-1])
        if not first.all():
            at = np.flatnonzero(first)
            count = np.add.reduceat(count, at)
            cell, gene_of = cell[at], gene_of[at]

    cell = cell.astype(np.int32)
    out_indptr = np.zeros(n_gene + 1, np.int64)
    np.cumsum(np.bincount(gene_of, minlength=n_gene), out=out_indptr[1:])
    # a byte holds every count of most runs, but not of a deep one (380 in a bin, in one colon
    # section, and more once bins are added together), so the width follows the data
    top = int(count.max()) if count.size else 0
    cdt = "uint8" if top < 256 else ("uint16" if top < 65536 else "uint32")

    total = np.bincount(cell.astype(np.int64), weights=count.astype(np.float64),
                        minlength=lat["w"] * lat["h"])
    if total.max() > np.iinfo(np.uint16).max:
        total = np.minimum(total, np.iinfo(np.uint16).max)
    total = total.astype(np.uint16)

    names = [str(x) for x in adata.var_names]
    path = str(store or sdata.path)
    # A group left by an interrupted run is ON DISK but missing from the store's consolidated
    # index, so zarr reads `key in root` as absent and then trips over the directory.  Clear the
    # directory itself, which is true whichever the index says.
    stale = os.path.join(path, key)
    if os.path.isdir(stale):
        shutil.rmtree(stale)
    root = zarr.open_group(path, mode="a")
    if key in root:
        del root[key]
    g = root.create_group(key)
    g.attrs["ssaft"] = dict(kind="gene_bins", version=FORMAT_VERSION, lattice=lat,
                            source_table=table, map_element=map_element,
                            n_gene=int(n_gene), nnz=int(cell.size))
    g.create_array("genes", shape=(len(names),), dtype=str,
                   chunks=(len(names),), compressors=[_zstd()])[...] = np.array(names, dtype=object)
    g.create_array("indptr", shape=out_indptr.shape, dtype="int64",
                   chunks=out_indptr.shape, compressors=[_zstd()])[...] = out_indptr
    g.create_array("cell", shape=cell.shape, dtype="int32", chunks=(_CHUNK,),
                   filters=[Delta(dtype="int32")], compressors=[_zstd()])[...] = cell
    g.create_array("count", shape=count.shape, dtype=cdt, chunks=(_CHUNK,),
                   compressors=[_zstd()])[...] = count.astype(cdt)
    g.create_array("total", shape=total.shape, dtype="uint16", chunks=(_CHUNK,),
                   compressors=[_zstd()])[...] = total
    _consolidate_into_root(store or sdata.path, key,
                           ("genes", "indptr", "cell", "count", "total"))
    if not quiet:
        print(f"[gene_bins] {key}: {n_gene:,} genes, {cell.size:,} nonzeros on "
              f"{lat['w']}x{lat['h']} bins of {lat['vx']:g} um"
              + (f", {int((~keep).sum()):,} off the map dropped" if not keep.all() else ""))
    return lat


def read_gene(store, gene, *, key=GENE_BINS_KEY, bin_size=1):
    """One gene's counts on the lattice, as ``(cell index, count, lattice)``.

    ``bin_size`` sums the counts of every ``bin_size x bin_size`` block of lattice cells,
    which is how a coarser view (the 8 um a Visium HD reader expects from a 2 um map) is
    taken without a second copy on disk.
    """
    import zarr

    g = zarr.open_group(str(store), mode="r")[key]
    meta = dict(g.attrs)["ssaft"]
    lat = dict(meta["lattice"])
    names = [str(x) for x in g["genes"][:]]
    if gene not in names:
        raise KeyError(gene)
    i = names.index(gene)
    lo, hi = (int(x) for x in g["indptr"][i:i + 2])
    cell = np.asarray(g["cell"][lo:hi], np.int64)
    count = np.asarray(g["count"][lo:hi], np.int64)
    if bin_size > 1:
        w2 = -(-lat["w"] // bin_size)
        cell = (cell // lat["w"] // bin_size) * w2 + (cell % lat["w"] // bin_size)
        lat = dict(lat, w=w2, h=-(-lat["h"] // bin_size),
                   vx=lat["vx"] * bin_size, vy=lat["vy"] * bin_size)
        if cell.size:
            cell, back = np.unique(cell, return_inverse=True)
            count = np.bincount(back, weights=count, minlength=cell.size).astype(np.int64)
    return cell, count, lat
