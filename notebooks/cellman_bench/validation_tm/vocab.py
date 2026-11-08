from __future__ import annotations

import glob
import json
import os
from functools import lru_cache

KG_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kg_cache")


NOCALL = {"unknown", "background"}


REF_ALIAS = {

    "immature t cell": "thymocyte",
    "naive t cell": "T cell",
    "mature alpha-beta t cell": "T cell",
    "mature nk t cell": "natural killer T cell",
    "immature nk t cell": "natural killer T cell",

    "precursor b cell": "pre-B cell",
    "early pro-b cell": "Pro-B cell",
    "hematopoietic precursor cell": "hematopoietic progenitor cell",
    "macrophage dendritic cell progenitor": "hematopoietic progenitor cell",
    "lymphoid lineage restricted progenitor cell": "common lymphoid progenitor",
    "granulocyte monocyte progenitor cell": "granulocyte-monocyte progenitor",
    "megakaryocyte-erythroid progenitor cell": "megakaryocyte-erythroid progenitor",
    "erythroid progenitor cell, mammalian": "erythroid progenitor",
    "granulocytopoietic cell": "granulocyte",

    "myeloid dendritic cell": "conventional dendritic cell",
    "myeloid leukocyte": "myeloid cell",
    "intermediate monocyte": "monocyte",
    "lung macrophage": "macrophage",
    "plasmatocyte": "macrophage",
    "professional antigen presenting cell": "leukocyte",
    "blood cell": "leukocyte",
    "circulating fibrocyte": "leukocyte",

    "endothelial cell of coronary artery": "arterial endothelial cell",
    "aortic endothelial cell": "arterial endothelial cell",
    "kidney cortex artery cell": "arterial endothelial cell",
    "vein endothelial cell": "venous endothelial cell",
    "endothelial cell of lymphatic vessel": "lymphatic endothelial cell",
    "kidney capillary endothelial cell": "capillary endothelial cell",
    "endothelial cell of hepatic sinusoid": "sinusoidal endothelial cell",
    "fenestrated endothelial cell": "endothelial cell",

    "brain pericyte": "pericyte",
    "bronchial smooth muscle cell": "airway smooth muscle cell",
    "smooth muscle cell of trachea": "airway smooth muscle cell",
    "smooth muscle cell of the pulmonary artery": "vascular smooth muscle cell",
    "regular atrial cardiac myocyte": "atrial cardiomyocyte",
    "regular ventricular cardiac myocyte": "ventricular cardiomyocyte",
    "cardiac muscle cell": "cardiomyocyte",
    "cardiac valve cell": "valvular interstitial cell",
    "cell of skeletal muscle": "skeletal muscle cell",

    "mesenchymal stem cell": "mesenchymal cell",
    "stromal cell": "mesenchymal cell",
    "bladder cell": "mesenchymal cell",
    "mesenchymal stem cell of adipose tissue": "adipose stem and progenitor cell",
    "adventitial cell": "adventitial fibroblast",
    "fibroblast of lung": "fibroblast",
    "pulmonary interstitial fibroblast": "fibroblast",
    "kidney interstitial fibroblast": "fibroblast",

    "kidney proximal convoluted tubule epithelial cell": "proximal tubule cell",
    "epithelial cell of proximal tubule": "proximal tubule cell",
    "kidney proximal straight tubule epithelial cell": "proximal tubule cell S3",
    "kidney loop of henle thick ascending limb epithelial cell": "thick ascending limb cell",
    "kidney loop of henle ascending limb epithelial cell": "thick ascending limb cell",
    "kidney distal convoluted tubule epithelial cell": "distal convoluted tubule cell",
    "kidney collecting duct principal cell": "collecting duct principal cell",
    "kidney collecting duct epithelial cell": "collecting duct epithelial cell",
    "kidney cell": "epithelial cell",

    "enterocyte of epithelium of large intestine": "colonocyte",
    "epithelial cell of large intestine": "epithelial cell",
    "intestinal crypt stem cell": "intestinal stem cell",
    "large intestine goblet cell": "goblet cell",
    "mucus secreting cell": "goblet cell",
    "tuft cell of large intestine": "tuft cell",
    "pancreatic acinar cell": "acinar cell",
    "pancreatic ductal cell": "ductal cell",
    "duct epithelial cell": "ductal cell",
    "pancreatic pp cell": "gamma cell",

    "pulmonary alveolar type 1 cell": "alveolar type 1 cell",
    "pulmonary alveolar type 2 cell": "alveolar type 2 cell",
    "basal epithelial cell of tracheobronchial tree": "basal cell",
    "respiratory basal cell": "basal cell",
    "multiciliated columnar cell of tracheobronchial tree": "ciliated cell",
    "neuroendocrine cell": "neuroendocrine cell",

    "basal cell of epidermis": "basal keratinocyte",
    "stem cell of epidermis": "basal keratinocyte",
    "epidermal cell": "keratinocyte",
    "keratinocyte stem cell": "keratinocyte stem cell",
    "bladder urothelial cell": "urothelial cell",
    "basal cell": "myoepithelial cell",

    "epithelial cell of thymus": "thymic epithelial cell",
    "interneuron": "GABAergic neuron",
    "cardiac neuron": "neuron",
    "medium spiny neuron": "striatal GABAergic neuron",
    "bergmann glial cell": "Bergmann glia",
    "neuroepithelial stem cell": "neural stem cell",
    "neural stem cell": "neural stem cell",

    "unknown": "Unknown",
}


CM_ALIAS = {
    "background": "Unknown",
}


EXTRA_EDGES = [

    ("DN2 thymocyte", "double negative thymocyte"),
    ("DN3 thymocyte", "double negative thymocyte"),
    ("DN4 thymocyte", "double negative thymocyte"),
    ("double negative thymocyte", "thymocyte"),
    ("immature single-positive thymocyte", "thymocyte"),
    ("DP thymocyte", "thymocyte"),
    ("CD4 single-positive thymocyte", "thymocyte"),
    ("CD8 single-positive thymocyte", "thymocyte"),
    ("thymocyte", "T cell"),

    ("cortical thymic epithelial cell", "thymic epithelial cell"),
    ("medullary thymic epithelial cell", "thymic epithelial cell"),
    ("bipotent thymic epithelial progenitor", "thymic epithelial cell"),
    ("junctional thymic epithelial cell", "thymic epithelial cell"),
    ("thymic epithelial cell", "epithelial cell"),

    ("bulge hair follicle stem cell", "keratinocyte stem cell"),
    ("upper bulge hair follicle stem cell", "bulge hair follicle stem cell"),
    ("lower bulge hair follicle stem cell", "bulge hair follicle stem cell"),
    ("basal keratinocyte", "keratinocyte stem cell"),
    ("keratinocyte stem cell", "keratinocyte"),

    ("quiescent neural stem cell", "neural stem cell"),
    ("radial glia", "neural stem cell"),
    ("neuroblast", "neural stem cell"),

    ("pulmonary neuroendocrine cell", "neuroendocrine cell"),


    ("neutrophil", "granulocyte"),


    ("lingual basal keratinocyte", "basal keratinocyte"),
]


@lru_cache(maxsize=1)
def kg_parents() -> dict[str, frozenset[str]]:
    par: dict[str, set[str]] = {}
    for f in sorted(glob.glob(os.path.join(KG_CACHE, "mouse_*.json"))):
        terms = json.load(open(f))["terms"]
        for v in terms.values():
            lab = v["L"].strip().lower()
            par.setdefault(lab, set())
            for p in v.get("P", []):
                if p in terms:
                    par[lab].add(terms[p]["L"].strip().lower())
    for child, parent in EXTRA_EDGES:
        par.setdefault(child.lower(), set()).add(parent.lower())
        par.setdefault(parent.lower(), set())
    return {k: frozenset(v) for k, v in par.items()}


@lru_cache(maxsize=1)
def kg_display() -> dict[str, str]:
    disp: dict[str, str] = {}
    for f in sorted(glob.glob(os.path.join(KG_CACHE, "mouse_*.json"))):
        for v in json.load(open(f))["terms"].values():
            disp.setdefault(v["L"].strip().lower(), v["L"].strip())
    for child, parent in EXTRA_EDGES:
        disp.setdefault(child.lower(), child)
        disp.setdefault(parent.lower(), parent)
    return disp


def harmonise(label: str, side: str) -> str:
    low = str(label).strip().lower()
    alias = REF_ALIAS if side == "ref" else CM_ALIAS
    if low in alias:
        low = alias[low].strip().lower()
    if low in NOCALL:
        return "Unknown"
    return kg_display().get(low, str(label).strip())


@lru_cache(maxsize=4096)
def ancestors(label: str) -> frozenset[str]:
    par = kg_parents()
    out: set[str] = set()
    stack = [label.strip().lower()]
    seen = set(stack)
    while stack:
        for p in par.get(stack.pop(), ()):
            if p not in seen:
                seen.add(p)
                out.add(p)
                stack.append(p)
    return frozenset(out)


def relation(cm: str, ref: str) -> str:
    c, r = cm.strip().lower(), ref.strip().lower()
    if c in NOCALL or c == "unknown":
        return "unknown"
    if c == r:
        return "same"
    if r in ancestors(c):
        return "cellman_deeper"
    if c in ancestors(r):
        return "ref_deeper"
    return "conflict"


def unmapped(labels, side: str) -> list[str]:
    par = kg_parents()
    bad = []
    for l in labels:
        h = harmonise(l, side)
        if h != "Unknown" and h.strip().lower() not in par:
            bad.append(l)
    return sorted(set(bad))
