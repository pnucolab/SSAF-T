import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.paths import REPO, outdir                              # noqa: E402

DS = "heart"

HEART = Path(os.environ.get("SSAFT_HEART", REPO / "_notebooks/seg-free/heart_2"))
OUT = outdir(DS)
TAB = outdir(DS, "data")


SAMP = [("NF", "NF_1561_v2"), ("NF", "NF_1691"), ("NF", "NF_1697"),
        ("pRV", "pRV_1567"), ("pRV", "pRV_1618"), ("pRV", "pRV_1692"),
        ("RVF", "RVF_1343"), ("RVF", "RVF_1467"), ("RVF", "RVF_1632")]
SECTIONS = [s for _, s in SAMP]


Z_RANGE = {"NF_1561_v2": (5, 8), "NF_1691": (5, 8), "NF_1697": (7, 10),
           "pRV_1567": (8, 11), "pRV_1618": (7, 9), "pRV_1692": (5, 8),
           "RVF_1343": (5, 8), "RVF_1467": (5, 9), "RVF_1632": (6, 10)}

CELLS = os.environ.get("CELLS", "published")
assert CELLS in ("published", "colall"), CELLS


K_FORCE = int(os.environ["K_FORCE"]) if os.environ.get("K_FORCE") else None


BLOCK_UM = 5
Z0, Z1 = 9, 22
NICHE_R = 100
GRID_STEP = 4
TISSUE_FRAC = 0.25


GRP = ["FB", "EC", "mural", "myeloid", "lymphoid", "adipocyte"]


NICHE_COLORS = ["#56b4e9", "#009e73", "#e69f00", "#cc79a7",
                "#d55e00", "#a68bd9", "#0072b2", "#b0b000"]


NICHE_NAMES = ["EC-rich myocardium I",
               "Perivascular myocardium",
               "Immune-rich stroma",
               "Adipose",
               "Immune-rich myocardium",
               "Mixed stromal myocardium",
               "EC-rich myocardium II",
               "Mural"]


SCALEBAR_UM = {"NF_1561_v2": 1000, "NF_1691": 500, "NF_1697": 500,
               "pRV_1567": 1000, "pRV_1618": 500, "pRV_1692": 1000,
               "RVF_1343": 500, "RVF_1467": 500, "RVF_1632": 1000}

GROUP_COLORS = {"NF": "#5aa0f0", "pRV": "#f08a5c", "RVF": "#9a8ee8"}
INK, BG = "#f2f2f2", "#0e1014"


def tag():
    return ("" if CELLS == "published" else "_colall") + (f"_k{K_FORCE}" if K_FORCE else "")


def store(section):
    p = HEART / section / "ssaft/ssaft_store"
    if not p.exists():
        raise SystemExit(f"{p} not found -- point SSAFT_HEART at the directory holding the "
                         f"nine <section>/ssaft runs")
    return p


def read_cells(section):
    import anndata as ad
    if CELLS == "colall":
        return ad.read_zarr(HEART / section / "ssaft/spatialdata.zarr/tables/ssaft_cells_colall")
    p = HEART / section / "ssaft/ssaft_cells.h5ad"
    if not p.exists():
        raise SystemExit(f"{p} not found -- this is the cell table the figure was made from")
    return ad.read_h5ad(p)


def grid_npz(section):
    return TAB / f"{section}_grid{BLOCK_UM}.npz"
