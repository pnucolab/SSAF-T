#!/usr/bin/env python
from __future__ import annotations

import os
import shutil
import sys

import h5py
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TS = os.path.abspath(os.path.join(HERE, "..", "TS"))


PATCHES = {
    "ovary": ("free_annotation", "granulosa cell", 655, "granulosa cell", "CL:0000501"),
    "mouth": ("cell_type",       "keratocyte",      14, "keratinocyte",   "CL:0000312"),
}


def read_cat(h: h5py.File, col: str) -> pd.Series:
    g = h["obs"][col]
    if isinstance(g, h5py.Group):
        return pd.Series(g["categories"][:].astype(str)[g["codes"][:]])
    return pd.Series(g[:].astype(str))


def write_cat(h: h5py.File, col: str, values: pd.Series) -> None:
    cat = pd.Categorical(values.astype(str))
    del h["obs"][col]
    g = h["obs"].create_group(col)
    g.attrs["encoding-type"] = "categorical"
    g.attrs["encoding-version"] = "0.2.0"
    g.attrs["ordered"] = np.False_
    d = g.create_dataset("categories", data=np.array(cat.categories, dtype=object),
                         dtype=h5py.special_dtype(vlen=str))
    d.attrs["encoding-type"] = "string-array"
    d.attrs["encoding-version"] = "0.2.0"
    codes = cat.codes.astype(np.int8 if len(cat.categories) < 128 else np.int32)
    d = g.create_dataset("codes", data=codes)
    d.attrs["encoding-type"] = "array"
    d.attrs["encoding-version"] = "0.2.0"


def state(path: str, sel_col: str, sel_val: str, new_ct: str) -> tuple[int, int]:
    with h5py.File(path, "r") as h:
        sel = read_cat(h, sel_col)
        ct = read_cat(h, "cell_type")
    return int((sel == sel_val).sum()), int((ct == new_ct).sum())


def patch(organ: str, verify_only: bool) -> None:
    sel_col, sel_val, n_exp, new_ct, new_id = PATCHES[organ]
    path = os.path.join(TS, organ, f"TS_{organ}_ref.h5ad")
    if not os.path.exists(path):
        print(f"[{organ}] MISSING {path}")
        return

    n_sel, n_already = state(path, sel_col, sel_val, new_ct)
    if n_already >= n_exp:
        print(f"[{organ}] already patched ({n_already} cells are '{new_ct}') -- nothing to do")
        return
    if n_sel != n_exp:
        print(f"[{organ}] ABORT: {sel_col}=='{sel_val}' matched {n_sel} cells, expected {n_exp}")
        return
    if verify_only:
        print(f"[{organ}] would patch {n_sel} cells -> '{new_ct}' / {new_id}")
        return

    backup = path + ".orig"
    if not os.path.exists(backup):
        print(f"[{organ}] backing up -> {os.path.basename(backup)}")
        shutil.copy2(path, backup)

    with h5py.File(path, "r+") as h:
        sel = read_cat(h, sel_col)
        mask = (sel == sel_val).to_numpy()
        for col, new in (("cell_type", new_ct), ("cell_type_ontology_term_id", new_id)):
            v = read_cat(h, col)
            old = sorted(set(v[mask]))
            v[mask] = new
            write_cat(h, col, v)
            print(f"[{organ}] {col}: {old} -> '{new}'  ({mask.sum()} cells)")

        print(f"[{organ}] broad_cell_class left as-is: {sorted(set(read_cat(h,'broad_cell_class')[mask]))}")


def main() -> None:
    verify_only = "--verify" in sys.argv
    for organ in PATCHES:
        patch(organ, verify_only)


if __name__ == "__main__":
    main()
