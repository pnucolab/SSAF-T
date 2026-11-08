import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.paths import DATA, outdir, sdata                       # noqa: E402

DS = "xenium_kidney"

SDATA = sdata(DS)
OUT = outdir(DS)
TAB = outdir(DS, "data")

XEN = DATA / "xenium"
BAYSOR_TX = XEN / "ShamR/segmentation.csv"
BAYSOR_CELLS = XEN / "ShamR/segmentation_cell_stats.csv"
REF_H5AD = XEN / "ref_shamR.h5ad"
BUNDLE = XEN / "GSM8325616_xenium_shamR_male_output"


POINTS = "transcripts"
SHAPES_2D = "ssaft_cell_boundaries_2d"
SHAPES_3D = "ssaft_cell_boundaries_3d"
TABLE = "ssaft_cells"
LABEL_COL = "cellman_annotation"
DAPI_IMAGE = "morphology_focus"


PX_UM = 0.2125
VOXEL = 1.0
CONTROL_PREFIX = ("NegControl", "BLANK", "antisense", "Unassigned", "DeprecatedCodeword")
CROP_C = (3313.861, 3288.296, 200.0)
