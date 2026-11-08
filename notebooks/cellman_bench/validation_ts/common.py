from __future__ import annotations

import os
import sys

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vocab

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as _C
TS = str(_C.ATLASES["TS"]["root"])
FIG = str(_C.figures("validation_ts"))
TAB = str(_C.tables("validation_ts"))


def read_col(path: str, col: str) -> pd.Series:
    with h5py.File(path, "r") as h:
        g = h["obs"][col]
        idx = h["obs"][h["obs"].attrs["_index"]][:].astype(str)
        vals = (g["categories"][:].astype(str)[g["codes"][:]]
                if isinstance(g, h5py.Group) else g[:].astype(str))
    return pd.Series(vals, index=idx, name=col)


def load(organ: str, with_X: bool = True) -> ad.AnnData:
    p = os.path.join(TS, organ)
    a = ad.read_h5ad(f"{p}/annotation_out/TS_{organ}_annotated.h5ad")
    rf = read_col(f"{p}/TS_{organ}_ref.h5ad", "cell_type")
    assert a.obs_names.equals(pd.Index(rf.index)), f"{organ}: obs index mismatch"
    assert a.raw is None, "unexpected .raw -- would leak reference marker genes"

    a.obs["cm"] = pd.Categorical([vocab.harmonise(x, "cellman")
                                  for x in a.obs["cellman_annotation"].astype(str)])
    a.obs["ref"] = pd.Categorical([vocab.harmonise(x, "ref") for x in rf.values])
    a.obs["relation"] = pd.Categorical([vocab.relation(c, r) for c, r
                                        in zip(a.obs["cm"].astype(str), a.obs["ref"].astype(str))])
    return a


def load_markers(organ: str) -> dict[str, dict[str, list[str]]]:
    f = os.path.join(TS, organ, "annotation_out", "celltype_markers.yaml")
    tree = yaml.safe_load(open(f))
    out: dict[str, dict[str, list[str]]] = {}

    def walk(node: dict):
        for label, body in (node or {}).items():
            body = body or {}
            out[label] = {k: list(body.get(k, []) or []) for k in
                          ("defining", "required", "supporting")}
            walk(body.get("children"))

    walk(tree)
    return out


def kg_markers(organ: str, species: str = "human") -> dict[str, dict[str, list[str]]]:
    import json
    f = os.path.join(HERE, "kg_cache", f"{species}_{organ}.json")
    terms = json.load(open(f))["terms"]
    return {v["L"]: {"defining": list(v.get("DF", []) or []),
                     "required": list(v.get("R", []) or []),
                     "supporting": list(v.get("S", []) or []),
                     "parents": [terms[p]["L"] for p in v.get("P", []) if p in terms]}
            for v in terms.values()}


def mean_profile(a: ad.AnnData, mask: np.ndarray, genes: list[str] | None = None) -> pd.Series:
    idx = np.asarray(mask).nonzero()[0]
    sub = a[idx] if genes is None else a[idx, genes]
    X = sub.X
    m = np.asarray(X.mean(axis=0)).ravel()
    return pd.Series(m, index=(a.var_names if genes is None else pd.Index(genes)))


def zheatmap(M: np.ndarray, rows: list[str], genes: list[str], owner: list[str],
             title: str, path_noext: str, ns: list[int], cbar_label: str,
             pct: np.ndarray | None = None):
    import matplotlib.pyplot as plt

    sd = M.std(axis=0, ddof=0)
    Z = np.zeros_like(M)
    ok = sd > 1e-9
    Z[:, ok] = (M[:, ok] - M[:, ok].mean(axis=0)) / sd[ok]
    hit = np.array([rows[int(np.argmax(Z[:, j]))] == owner[j] for j in range(len(genes))])

    fig, ax = plt.subplots(figsize=(max(5.0, 0.34 * len(genes) + 3.6),
                                    max(2.4, 0.52 * len(rows) + 2.0)))
    vmax = float(np.nanmax(np.abs(Z))) or 1.0
    im = ax.imshow(Z, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    if pct is not None:
        for i in range(len(rows)):
            for j in range(len(genes)):
                ax.scatter(j, i - 0.26, s=6 + 60 * pct[i, j], facecolor="none",
                           edgecolor="#222222", linewidth=0.5, zorder=3)
    for i in range(len(rows)):
        for j in range(len(genes)):
            ax.text(j, i + 0.24, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.6,
                    color="#111111" if abs(Z[i, j]) < 0.7 * vmax else "#ffffff", zorder=4)
    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels(genes, rotation=90, fontsize=7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{k}  (n={n:,})" for k, n in zip(rows, ns)], fontsize=8)


    palette = plt.get_cmap("tab10")
    base = []
    for r in rows:
        b = r.split("\n")[0]
        if b not in base:
            base.append(b)
    cidx = {b: palette(i % 10) for i, b in enumerate(base)}
    for t, o in zip(ax.get_xticklabels(), owner):
        t.set_color(cidx.get(o.split("\n")[0], "#000000"))
    for b in np.flatnonzero(np.array(owner[:-1]) != np.array(owner[1:])):
        ax.axvline(b + 0.5, color="#333333", lw=0.8)
    ax.set_yticks(np.arange(len(rows)) - 0.5, minor=True)
    ax.grid(which="minor", axis="y", color="#333333", lw=0.8)
    ax.set_title(title, fontsize=9.5, pad=10)
    fig.colorbar(im, ax=ax, shrink=0.72, pad=0.012,
                 label=cbar_label).ax.tick_params(labelsize=7)
    fig.text(0.5, -0.02, "cell text = mean log1p expression;  circle size = fraction of cells "
                         "expressing;  colour = z across row groups",
             ha="center", fontsize=6.5, color="#555555")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_noext}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return Z, hit


def group_stats(a: ad.AnnData, masks: list[np.ndarray], genes: list[str]):
    gi = [a.var_names.get_loc(g) for g in genes]
    M = np.zeros((len(masks), len(genes)), dtype=float)
    P = np.zeros_like(M)
    for i, m in enumerate(masks):
        X = a[np.asarray(m).nonzero()[0]].X[:, gi]
        M[i] = np.asarray(X.mean(axis=0)).ravel()
        P[i] = np.asarray((X > 0).mean(axis=0)).ravel()
    return M, P


def hvg(a: ad.AnnData, n_top: int = 2000) -> list[str]:
    import scanpy as sc
    v = sc.pp.highly_variable_genes(a, n_top_genes=n_top, flavor="seurat", inplace=False)
    return list(a.var_names[np.asarray(v["highly_variable"])])
