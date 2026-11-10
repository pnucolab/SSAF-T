from pathlib import Path

ANCHORS = "ssaft_downsampled_vectors"
CONTAINERS = ("images", "labels", "points", "shapes", "tables")


def save_elements(sdata, names):
    present = [n for n in names if any(n in getattr(sdata, c) for c in CONTAINERS)]
    on_disk = {p.rsplit("/", 1)[-1] for p in sdata.elements_paths_on_disk()}
    for n in present:
        if n in on_disk:
            sdata.delete_element_from_disk(n)
    sdata.write_element(present)
    return present


def _grid(store, key):
    import zarr
    if not Path(store).exists():
        return None
    zg = zarr.open(str(store), mode="r")
    return [str(x) for x in zg[key][:]] if key in zg else None


def _anchors(sdata_path, key):
    import zarr
    from anndata.io import read_elem
    if sdata_path is None:
        return None
    p = Path(sdata_path) / "tables" / ANCHORS
    if not p.exists():
        return None
    zg = zarr.open(str(p), mode="r")
    for k in key.split("/"):
        if k not in zg:
            return None
        zg = zg[k]
    return read_elem(zg)


def panel_genes(store, sdata_path=None):
    g = _grid(store, "genes")
    if g is None:
        v = _anchors(sdata_path, "var")
        g = [str(x) for x in v.index] if v is not None else None
    if g is None:
        raise SystemExit(f"missing input: {store}/genes\n  this is the vector-field gene panel, "
                         f"written by downsample_vector; run run_ssaft.py down")
    return g


def celltype_names(store, sdata_path=None, column="cellman_annotation"):
    n = _grid(store, "celltype_names")
    if n is None:
        c = _anchors(sdata_path, f"obs/{column}")
        n = [str(x) for x in c.categories] if c is not None else None
    if n is None:
        raise SystemExit(f"missing input: {store}/celltype_names\n  these name the cell-type map's "
                         f"pixel values, written by map_celltypes; run run_ssaft.py map")
    return n
