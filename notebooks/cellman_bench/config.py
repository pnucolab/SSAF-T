import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.paths import PANEL, REPO                                # noqa: E402

ATLAS_ROOT = Path(os.environ.get("CELLMAN_BENCH_ATLAS", REPO / "_notebooks/cellman_bench"))
OUT_ROOT = Path(os.environ.get("CELLMAN_BENCH_OUT", PANEL / "cellman_bench"))

FIG = OUT_ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

ATLASES = {"TS": dict(root=ATLAS_ROOT / "TS", species="human", dir="validation_ts",
                      label="Tabula Sapiens (human)"),
           "TM": dict(root=ATLAS_ROOT / "TM", species="mouse", dir="validation_tm",
                      label="Tabula Muris Senis (mouse)")}


def tables(atlas_dir):
    p = OUT_ROOT / atlas_dir / "tables"
    p.mkdir(parents=True, exist_ok=True)
    return p


def figures(atlas_dir):
    p = OUT_ROOT / atlas_dir / "figures"
    p.mkdir(parents=True, exist_ok=True)
    return p
