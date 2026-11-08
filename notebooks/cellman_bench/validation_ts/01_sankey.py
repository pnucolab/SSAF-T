#!/usr/bin/env python
from __future__ import annotations

import os
import sys

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vocab
from sankey import RELATION_COLORS, sankey

HERE = os.path.dirname(os.path.abspath(__file__))
TS = os.path.abspath(os.path.join(HERE, "..", "TS"))
FIG = os.path.join(HERE, "figures")
TAB = os.path.join(HERE, "tables")


def read_col(path: str, col: str) -> pd.Series:
    with h5py.File(path, "r") as h:
        g = h["obs"][col]
        idx = h["obs"][h["obs"].attrs["_index"]][:].astype(str)
        if isinstance(g, h5py.Group):
            vals = g["categories"][:].astype(str)[g["codes"][:]]
        else:
            vals = g[:].astype(str)
    return pd.Series(vals, index=idx, name=col)


def organs() -> list[str]:
    out = []
    for d in sorted(os.listdir(TS)):
        p = os.path.join(TS, d)
        if os.path.isdir(p) and os.path.exists(f"{p}/TS_{d}_ref.h5ad") \
           and os.path.exists(f"{p}/annotation_out/TS_{d}_annotated.h5ad"):
            out.append(d)
    return out


def pairs(organ: str) -> pd.DataFrame:
    p = os.path.join(TS, organ)
    cm = read_col(f"{p}/annotation_out/TS_{organ}_annotated.h5ad", "cellman_annotation")
    rf = read_col(f"{p}/TS_{organ}_ref.h5ad", "cell_type")
    assert cm.index.equals(rf.index), f"{organ}: obs index mismatch between annotated and ref"

    df = pd.DataFrame({"cm_raw": cm.values, "ref_raw": rf.values}, index=cm.index)
    df["cm"] = [vocab.harmonise(x, "cellman") for x in df["cm_raw"]]
    df["ref"] = [vocab.harmonise(x, "ref") for x in df["ref_raw"]]

    g = (df.groupby(["cm", "ref"]).size().rename("n_cells").reset_index())
    g["relation"] = [vocab.relation(a, b) for a, b in zip(g["cm"], g["ref"])]
    g["organ"] = organ
    return g.sort_values("n_cells", ascending=False)


def draw(organ: str, g: pd.DataFrame) -> None:
    os.makedirs(FIG, exist_ok=True)
    n = int(g["n_cells"].sum())
    flows = g.rename(columns={"cm": "source", "ref": "target", "n_cells": "value"})

    for tag, color_by in (("", "source"), ("_relation", "relation")):
        fig, ax = sankey(flows, color_by=color_by,
                         title=f"{organ.replace('_', ' ')}   n = {n:,} cells",
                         left_title="CeLLMan", right_title="Tabula Sapiens")
        if color_by == "relation":
            from matplotlib.patches import Patch
            share = g.groupby("relation")["n_cells"].sum() / n
            ax.legend(handles=[Patch(facecolor=RELATION_COLORS[k], alpha=.75,
                                     label=f"{k}  {share.get(k, 0):.1%}")
                               for k in ["same", "cellman_deeper", "ref_deeper",
                                         "conflict", "unknown"] if k in share.index],
                      loc="upper center", bbox_to_anchor=(0.5, -0.008), frameon=False,
                      fontsize=8.5, ncol=5, handlelength=1.2, columnspacing=1.8)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(FIG, f"sankey_{organ}{tag}.{ext}"), dpi=200,
                        bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)


def main(argv: list[str]) -> None:
    todo = argv[1:] or organs()
    os.makedirs(TAB, exist_ok=True)
    summary = []
    for organ in todo:
        g = pairs(organ)
        g.to_csv(os.path.join(TAB, f"pairs_{organ}.csv"), index=False)
        draw(organ, g)
        n = g["n_cells"].sum()
        row = {"organ": organ, "n_cells": int(n),
               "n_cellman_labels": g["cm"].nunique(), "n_ref_labels": g["ref"].nunique()}
        s = g.groupby("relation")["n_cells"].sum()
        for k in ["same", "cellman_deeper", "ref_deeper", "conflict", "unknown"]:
            row[k] = int(s.get(k, 0))
            row[f"{k}_frac"] = float(s.get(k, 0) / n)
        summary.append(row)
        print(f"[{organ:16s}] n={n:7,d}  same {row['same_frac']:5.1%}  "
              f"cm_deeper {row['cellman_deeper_frac']:5.1%}  "
              f"ref_deeper {row['ref_deeper_frac']:5.1%}  "
              f"conflict {row['conflict_frac']:5.1%}  unknown {row['unknown_frac']:5.1%}")
    pd.DataFrame(summary).to_csv(os.path.join(TAB, "relation_summary.csv"), index=False)


if __name__ == "__main__":
    main(sys.argv)
