import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.paths import DATA, outdir, sdata                       # noqa: E402

DS = "stereoseq_brain"

SDATA = sdata(DS)
OUT = outdir(DS)
TAB = outdir(DS, "data")
CACHE = outdir(DS, "data/cache")

MOSTA = DATA / "stereoseq/mosta"
BIN1 = MOSTA / "Mouse_brain_Adult_GEM_bin1.tsv.gz"
CELLBIN = MOSTA / "Mouse_brain_Adult_GEM_CellBin.tsv.gz"


CELLTYPE_MAP = "ssaft_celltype_map"
VECTORS = "ssaft_downsampled_vectors"
SHAPES_2D = "ssaft_cell_boundaries_2d"
TABLE = "ssaft_cells"
LABEL_COL = "cellman_annotation"


X0, Y0 = 1612.5, 3087.5
W, H = 5250, 6975
PX = 5
NX, NY = W // PX, H // PX


DISPLAY_CHOICE = {"column_windows": [51, 52, 53], "rotation_deg": -141.0592038931098,
                  "unrolled_windows": [55, 63], "unrolled_length_mm": 2.25}
