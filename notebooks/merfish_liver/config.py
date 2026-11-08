import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.paths import DATA, outdir, sdata                       # noqa: E402

DS = "merfish_liver"

SDATA = sdata(DS)
OUT = outdir(DS)
TAB = outdir(DS, "data")

SUB = DATA / "merfish/paper_subset_2"
PUB_TX = SUB / "detected_transcripts.csv"
PUB_POLY = SUB / "cell_boundaries.parquet"
PUB_META = SUB / "cell_metadata.csv"
PUB_IMAGES = SUB / "images"


POINTS = "merfish_paper_subset_2_transcripts"
SHAPES_2D = "ssaft_cell_boundaries_2d"
SHAPES_3D = "ssaft_cell_boundaries_3d"
NUCLEI = "nucleus_boundaries"
TABLE = "ssaft_cells"
LABEL_COL = "cellman_annotation"
DAPI_IMAGE = "dapi_zstack"


PX_PER_UM = 9.2594
Z_PER_PLANE = 1.5
CONTROL_PREFIX = ("blank", "notarget", "negcontrol")
CROP_C = (3059.906, 1713.730, 200.0)
CROP_A = (2125.0, 2050.0, 500.0)
