import json
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "8")
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

STORE = C.SDATA
OUT = C.OUT
TAB = C.TAB
WORK = C.TAB / "work"
WORK.mkdir(parents=True, exist_ok=True)
COLORS = json.load(open(C.SDATA.parent / "celltype_colors.json"))["colors"]


PX, X0, Y0 = C.PX, C.X0, C.Y0


MOOR_BOTTOM = ['2010107E04Rik', '2210407C18Rik', 'Atp5a1', 'Atp5e', 'Atp5g3', 'Atp5h', 'Atp5j',
    'Atp5j2', 'Atp5l', 'Atp5o', 'Atpif1', 'Ccl25', 'Chchd10', 'Ckmt1', 'Cox4i1', 'Cox5a', 'Cox5b',
    'Cox6b1', 'Cox6c', 'Cox7a1', 'Cox7a2', 'Cox7b', 'Fabp1', 'Gpx2', 'Gsta1', 'Lgals4', 'Lypd8',
    'Minos1', 'Ndufa1', 'Ndufa4', 'Ndufa5', 'Ndufb6', 'Ndufb8', 'Ndufc1', 'Plac8', 'Reg3b', 'Reg3g',
    'Rpl18', 'Rpl35a', 'Rpl38', 'Rpl39', 'Rpl41', 'Rplp1', 'Rps12', 'Rps14', 'Rps18', 'Rps2', 'Rps27',
    'Rps27l', 'Rps28', 'Rps29', 'Rps8', 'Sis', 'Spink4', 'Tm4sf5', 'Tma7', 'Txn1', 'Uba52', 'Uqcr10',
    'Uqcr11', 'Uqcrh', 'Uqcrq']
MOOR_TOP = ['2010109I03Rik', 'Acad9', 'Ada', 'Ap1g1', 'Apoa4', 'Apoc3', 'Cep57', 'Cfap20', 'Chchd1',
    'Chek2', 'Clca4a', 'Cldn7', 'Fgd4', 'Gcn1l1', 'Glipr1', 'Gm10680', 'Gm20594', 'Ifrd1', 'Krt20',
    'Lgals3', 'Lrrc41', 'Mrpl48', 'Myo7a', 'Olfr1385', 'Olfr46', 'Pam', 'Pkib', 'Pmp22', 'Psma7',
    'Rab34', 'S100a10', 'S100a6', 'Serpinb1a', 'Slc17a5', 'Slc25a22', 'Slc28a2', 'Sprr2a2', 'Ssbp2',
    'Tbk1', 'Tlr1', 'Tmsb4x', 'Ythdc2', 'Zfp280d']

MOOR_TEXT_BOTTOM = ['Reg1', 'Reg3b', 'Reg3g', 'Nlrp6', 'Il18', 'Ccl25', 'Lypd8']
MOOR_TEXT_MID = ['Slc5a1', 'Slc2a2', 'Slc15a1', 'Apob', 'Npc1l1']
MOOR_TEXT_TIP = ['Nt5e', 'Enpp3', 'Ada', 'Slc28a2', 'Egfr', 'Klf4', 'Apoa4']
HOUSEKEEPING_PREFIX = ('Rpl', 'Rps', 'Atp5', 'Cox', 'Nduf', 'Uqcr', 'Mrpl', 'Chchd', 'Minos', 'Tma7',
                       'Uba52', 'Atpif', 'Ckmt')

MUSCLE = ['Myh11', 'Des', 'Actg2', 'Cnn1', 'Acta2', 'Tagln']
PANETH_RAW = ['Lyz1', 'Defa21', 'Defa26', 'Defa28', 'Defa35', 'Ang4', 'Itln1', 'Mptx2', 'Mmp7']
PANETH_HELDOUT = ['Defa21', 'Defa26', 'Defa28', 'Defa35', 'Ang4', 'Itln1', 'Mptx2']
STEM = ['Lgr5', 'Olfm4', 'Ascl2', 'Smoc2']
STEM_HELDOUT = ['Rgmb', 'Axin2', 'Slc12a2']
PROLIF = ['Mki67', 'Top2a', 'Cenpf', 'Birc5', 'Cdk1', 'Ccnb2', 'Stmn1', 'Pcna']
EPI = ['Epcam', 'Vil1', 'Cdh1', 'Krt8', 'Krt19', 'Krt20', 'Cdx2', 'Cldn7']


EXPECTED = {"intestinal stem cell": 0, "Paneth cell": 0, "transit amplifying cell": 1,
            "immature enterocyte": 2, "villus-bottom enterocyte": 3, "villus-tip enterocyte": 4}
EXPECTED_FULL = {**EXPECTED, "immature Paneth cell progenitor": 0}
ENACT_EXPECTED = {"Paneth cells": 0, "Crypt cells": 0, "Enterocytes": 1}


def kg_markers():
    idx = C.SDATA.parent / "kg_slim_index.json"
    mk = C.SDATA.parent / "celltype_markers.yaml"
    for p in (idx, mk):
        if not p.exists():
            raise SystemExit(f"missing input: {p}\n  this is written by the CeLLMan annotation "
                             f"step and ships with the SSAF-T output")
    g = set()
    for v in json.load(open(idx))["terms"].values():
        for f in ("R", "S", "N", "DF", "A", "MP"):
            g |= set(v.get(f) or [])

    def walk(node):
        for k, v in node.items():
            if not isinstance(v, dict):
                continue
            for f in ("defining", "required", "supporting", "negative"):
                g.update(v.get(f, []) or [])
            if "children" in v:
                walk(v["children"])

    walk(yaml.safe_load(open(mk)))
    return g


def ssaft_cells(backed=None):
    import spatialdata as sd
    return sd.read_zarr(STORE, selection=("tables",)).tables[C.TABLE]


def enact_cells(backed=None):
    import anndata as ad
    p = TAB / "enact_cells.h5ad"
    if not p.exists():
        raise SystemExit(f"missing input: {p}\n  run 00_prepare_enact.py first")
    return ad.read_h5ad(p, backed=backed)

def signal_genes():
    import spatialdata as sd
    dv = sd.read_zarr(STORE, selection=("tables",)).tables["ssaft_downsampled_vectors"]
    return [str(g) for g in dv.var_names]
