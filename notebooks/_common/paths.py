import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

DATA = Path(os.environ.get("SSAFT_DATA", REPO / "_datasets"))
OUTPUT = Path(os.environ.get("SSAFT_OUT", REPO / "_output"))
PANEL = Path(os.environ.get("SSAFT_PANEL", REPO / "_manuscript/_result/_script_out"))

DATASETS = ("visiumhd_small_intestine", "stereoseq_brain", "xenium_kidney", "merfish_liver")


def sdata(dataset):
    p = OUTPUT / dataset / "ssaft_sdata.zarr"
    if not p.exists():
        raise SystemExit(f"{p} not found — run {dataset}/run_ssaft.py first, "
                         f"or point SSAFT_OUT at the directory that holds it")
    return p


def outdir(dataset, sub=""):
    p = PANEL / dataset / sub if sub else PANEL / dataset
    p.mkdir(parents=True, exist_ok=True)
    return p


def need(path, what):
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"missing input: {path}\n  this is {what}")
    return path
