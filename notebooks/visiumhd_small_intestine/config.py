import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.paths import DATA, outdir, sdata                       # noqa: E402

DS = "visiumhd_small_intestine"

SDATA = sdata(DS)
OUT = outdir(DS)
TAB = outdir(DS, "data")

RAW = DATA / "visiumhd_enact"
BINNED = RAW / "binned_outputs"
WSI = RAW / "Visium_HD_Mouse_Small_Intestine_tissue_image.btf"
ENACT_OUT = Path(os.environ.get(
    "ENACT_OUT", DATA.parent / "enact_celltype/enact_output/mouse_small_intestine"))


BIN_TABLE = "table"
SHAPES_2D = "ssaft_cell_boundaries_2d"
TABLE = "ssaft_cells"
CELLTYPE_MAP = "ssaft_celltype_map"
LABEL_COL = "cellman_annotation"


PX, X0, Y0 = 2.0, 262.0, 120.0
