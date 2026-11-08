import os
import sys
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.paths import REPO, outdir                              # noqa: E402

DS = "placenta"

PLACENTA = Path(os.environ.get("SSAFT_PLACENTA", REPO / "_notebooks/seg-free/human_placenta"))
SDATA = PLACENTA / "spatialdata.zarr"
STORE = PLACENTA / "ssaft_store_v2"
OUT = outdir(DS)
TAB = outdir(DS, "data")
DATA = TAB
SCRATCH = Path(os.environ.get("SSAFT_SCRATCH",
                              "/bce/groups/pnucolab/analysis/l202255179/tmp/suppl_placenta"))
SCRATCH.mkdir(parents=True, exist_ok=True)

SOURCE = os.environ.get("SOURCE", "map_v2")
assert SOURCE in ("map_v2", "colall"), SOURCE

MAP = "ssaft_celltype_map_v2"
LABELS = "ssaft_cell_labels_3d_colall"
TABLE = "ssaft_cells_colall"


Z0, Z1, ZEX = 4, 19, 11
ZLO, ZHI = 4, 20
WIN_X, WIN_Y, WIN = 4100, 2900, 400


HALO, MIN_CORE, ASSIGN_MAX, HOLE = 120, 5000, 20.0, 400
MIN_VOLUME = 3e4
NPROC = 16


def names():
    import anndata as ad
    n = [str(x) for x in ad.read_zarr(SDATA / "tables" / TABLE).uns["cellman_celltype_names"]]
    stored = [str(x) for x in np.asarray(zarr.open(str(STORE), mode="r")["celltype_names"])]
    assert n == stored, "the table and the store disagree about the cell-type names"
    return n


def palette():
    return list(zarr.open(str(STORE), mode="r").attrs["cellman_celltype_colors"])


def _cell_lut():
    import anndata as ad
    o = ad.read_zarr(SDATA / "tables" / TABLE).obs
    lut = np.zeros(int(o.label_id.max()) + 1, np.uint8)
    lut[o.label_id.values] = (o.celltype_id.values + 1).astype(np.uint8)
    return lut


def celltype_planes(z0, z1):
    if SOURCE == "map_v2":
        p = SDATA / "labels" / MAP / "s0"
        if not p.exists():
            raise SystemExit(f"{p} not found -- point SSAFT_PLACENTA at the run's directory")
        return np.asarray(zarr.open(str(p), mode="r")[z0:z1]).astype(np.uint8)
    return _cell_lut()[np.asarray(zarr.open(str(SDATA / "labels" / LABELS / "s0"), mode="r")[z0:z1])]


def tag():
    return "" if SOURCE == "map_v2" else "_colall"
