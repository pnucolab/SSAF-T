import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from _common.store import celltype_names, panel_genes  # noqa: E402

OUT, CACHE, TAB = C.OUT, C.CACHE, C.TAB
STORE = C.SDATA
GRID = C.SDATA.parent / "ssaft_store"
X0, Y0, W, H, PX = C.X0, C.Y0, C.W, C.H, C.PX
NX, NY = C.NX, C.NY

LAYERS = ["L2/3 IT glutamatergic neuron", "L4 IT glutamatergic neuron",
          "L5 ET glutamatergic neuron", "L6 IT glutamatergic neuron",
          "L6 CT glutamatergic neuron", "L6b glutamatergic neuron"]
LAYER_SHORT = {"L2/3 IT glutamatergic neuron": "L2/3 IT", "L4 IT glutamatergic neuron": "L4 IT",
               "L5 ET glutamatergic neuron": "L5 ET", "L6 IT glutamatergic neuron": "L6 IT",
               "L6 CT glutamatergic neuron": "L6 CT", "L6b glutamatergic neuron": "L6b"}

LAYER_RANK = {"L2/3 IT glutamatergic neuron": 1, "L4 IT glutamatergic neuron": 2,
              "L5 ET glutamatergic neuron": 3, "L6 IT glutamatergic neuron": 4,
              "L6 CT glutamatergic neuron": 4, "L6b glutamatergic neuron": 5}
PUB_LAYERS = ["EX L2/3", "EX L4", "EX L5/6", "EX L6"]
PUB_RANK = {"EX L2/3": 1, "EX L4": 2, "EX L5/6": 3, "EX L6": 4}
HIPPO = ["CA1 pyramidal neuron", "CA3 pyramidal neuron", "hippocampal pyramidal neuron",
         "dentate gyrus granule cell"]


def load_bin1():
    z = np.load(CACHE / "bin1.npz")
    return dict(gene=z["gene"], x=0.5 * z["x"].astype(np.float32) - X0,
                y=0.5 * z["y"].astype(np.float32) - Y0, umi=z["umi"], genes=z["genes"],
                gidx={g: i for i, g in enumerate(z["genes"])})


def raster(b, genes=None, px=PX, weight="umi"):
    m = np.ones(len(b["x"]), bool) if genes is None else np.isin(
        b["gene"], [b["gidx"][g] for g in genes if g in b["gidx"]])
    ix = np.clip((b["x"][m] // px).astype(np.int32), 0, W // px - 1)
    iy = np.clip((b["y"][m] // px).astype(np.int32), 0, H // px - 1)
    w = b["umi"][m].astype(np.float64) if weight == "umi" else None
    return np.bincount(ix * (H // px) + iy, weights=w,
                       minlength=(W // px) * (H // px)).reshape(W // px, H // px)


def load_map(filtered=True):
    import spatialdata as sd
    s = sd.read_zarr(C.SDATA, selection=("labels",))
    lab = np.asarray(s.labels[C.CELLTYPE_MAP])
    cm = lab[0] if lab.ndim == 3 and lab.shape[0] == 1 else lab
    names = celltype_names(GRID, C.SDATA, C.LABEL_COL)
    if cm.shape != (W, H):
        cm = cm.T
    assert cm.shape == (W, H), f"map {cm.shape} is not the {W} x {H} µm grid"
    return cm, names


def map_at(cm, px=PX):
    o = px // 2
    return cm[o::px, o::px][: W // px, : H // px]


def palette():
    return json.load(open(C.SDATA.parent / "celltype_colors.json"))["colors"]


def signal_genes():
    return panel_genes(GRID, C.SDATA)


def ssaft_cells():
    import spatialdata as sd
    return sd.read_zarr(C.SDATA, selection=("tables",)).tables[C.TABLE]
