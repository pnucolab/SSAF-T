from __future__ import annotations

import glob
import json
import os
from functools import lru_cache

KG_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kg_cache")


NOCALL = {"unknown", "background"}


REF_ALIAS = {


    "mature nk t cell": "natural killer T cell",
    "naive thymus-derived cd4-positive, alpha-beta t cell": "naive CD4-positive, alpha-beta T cell",
    "naive thymus-derived cd8-positive, alpha-beta t cell": "naive CD8-positive, alpha-beta T cell",
    "activated cd4-positive, alpha-beta t cell": "CD4-positive, alpha-beta T cell",
    "activated cd8-positive, alpha-beta t cell": "CD8-positive, alpha-beta T cell",
    "cd4-positive, alpha-beta thymocyte": "CD4 single-positive thymocyte",
    "cd8-positive, alpha-beta thymocyte": "CD8 single-positive thymocyte",

    "myeloid dendritic cell": "conventional dendritic cell",
    "colon macrophage": "macrophage",
    "hematopoietic precursor cell": "hematopoietic progenitor cell",
    "hematopoietic cell": "leukocyte",
    "erythroid progenitor cell": "erythroid progenitor",
    "erythroid lineage cell": "erythroid cell",

    "endothelial cell of artery": "arterial endothelial cell",
    "endothelial cell of arteriole": "arterial endothelial cell",
    "vein endothelial cell": "venous endothelial cell",
    "endothelial cell of venule": "venous endothelial cell",
    "endothelial cell of lymphatic vessel": "lymphatic endothelial cell",
    "endothelial cell of vascular tree": "endothelial cell",
    "cardiac endothelial cell": "endothelial cell",
    "colon endothelial cell": "endothelial cell",
    "retinal blood vessel endothelial cell": "retinal vascular endothelial cell",

    "blood vessel smooth muscle cell": "vascular smooth muscle cell",
    "vascular associated smooth muscle cell": "vascular smooth muscle cell",
    "bronchial smooth muscle cell": "airway smooth muscle cell",
    "myometrial cell": "smooth muscle cell",
    "myofibroblast cell": "myofibroblast",

    "fibroblast of breast": "fibroblast",
    "alveolar adventitial fibroblast": "adventitial fibroblast",
    "adventitial cell": "adventitial fibroblast",
    "stromal cell": "mesenchymal cell",
    "stromal cell of ovary": "mesenchymal cell",
    "connective tissue cell": "mesenchymal cell",
    "mesenchymal stem cell of adipose tissue": "adipose stem and progenitor cell",
    "tendon cell": "tenocyte",
    "thymic fibroblast type 1": "thymic capsular fibroblast",
    "thymic fibroblast type 2": "medullary interlobular thymic fibroblast",

    "skeletal muscle satellite stem cell": "skeletal muscle satellite cell",
    "regular atrial cardiac myocyte": "atrial cardiomyocyte",
    "ventricular cardiac muscle cell": "ventricular cardiomyocyte",
    "fast muscle cell": "skeletal muscle cell",
    "slow muscle cell": "skeletal muscle cell",
    "tongue muscle cell": "tongue skeletal muscle cell",

    "pulmonary alveolar type 1 cell": "alveolar type 1 cell",
    "pulmonary alveolar type 2 cell": "alveolar type 2 cell",
    "lung multiciliated epithelial cell": "ciliated cell",
    "multiciliated columnar cell of tracheobronchial tree": "ciliated cell",
    "ciliated epithelial cell": "ciliated cell",
    "respiratory tract goblet cell": "goblet cell",
    "tracheal goblet cell": "goblet cell",
    "serous cell of epithelium of trachea": "submucosal gland serous cell",
    "serous cell of epithelium of bronchus": "submucosal gland serous cell",

    "enterocyte of epithelium proper of duodenum": "absorptive enterocyte",
    "enterocyte of epithelium proper of jejunum": "absorptive enterocyte",
    "enterocyte of epithelium proper of ileum": "absorptive enterocyte",
    "enterocyte of epithelium proper of small intestine": "absorptive enterocyte",
    "enterocyte of epithelium of large intestine": "absorptive enterocyte",
    "small intestine goblet cell": "goblet cell",
    "large intestine goblet cell": "goblet cell",
    "paneth cell of epithelium of small intestine": "Paneth cell",
    "paneth cell of colon": "Paneth cell",
    "intestinal crypt stem cell of small intestine": "intestinal stem cell",
    "intestinal crypt stem cell of colon": "intestinal stem cell",
    "transit amplifying cell of small intestine": "transit amplifying cell",
    "transit amplifying cell of colon": "transit amplifying cell",
    "intestinal tuft cell": "tuft cell",
    "tuft cell of colon": "tuft cell",
    "enteroendocrine cell of small intestine": "enteroendocrine cell",

    "acinar cell of salivary gland": "acinar cell",
    "pancreatic acinar cell": "acinar cell",
    "pancreatic ductal cell": "ductal cell",
    "duct epithelial cell": "ductal cell",
    "pancreatic pp cell": "pancreatic endocrine cell",
    "intrahepatic cholangiocyte": "cholangiocyte",

    "bladder urothelial cell": "urothelial cell",
    "luminal cell of prostate epithelium": "prostate luminal epithelial cell",
    "basal cell of prostate epithelium": "basal prostate epithelial cell",
    "epithelial cell of uterus": "endometrial epithelial cell",
    "glandular secretory epithelial cell": "secretory glandular endometrial epithelial cell",
    "ovarian surface epithelial cell": "ovarian surface / coelomic epithelium",
    "luminal epithelial cell of mammary gland": "mammary luminal epithelial cell",
    "progenitor cell of mammary luminal epithelium": "luminal secretory cell",
    "epithelial fate stem cell": "epithelial cell",

    "neuro-medullary thymic epithelial cell": "neuroendocrine medullary thymic epithelial cell",
    "myo-medullary thymic epithelial cell": "myoid thymic epithelial cell",

    "eye photoreceptor cell": "photoreceptor cell",
    "retinal bipolar neuron": "bipolar cell",
    "mueller cell": "Muller glia",
    "radial glial cell": "glial cell",
    "supporting cell of vestibular epithelium": "vestibular supporting cell",

    "male germ cell": "germ cell",

    "unknown": "Unknown",
}


CM_ALIAS = {
    "background": "Unknown",
}


EXTRA_EDGES = [
    ("prostate luminal epithelial cell", "epithelial cell"),
    ("luminal acinar prostate epithelial cell", "prostate luminal epithelial cell"),
    ("luminal ductal prostate epithelial cell", "prostate luminal epithelial cell"),
    ("mammary luminal epithelial cell", "epithelial cell"),
    ("luminal hormone-sensing cell", "mammary luminal epithelial cell"),
    ("luminal secretory cell", "mammary luminal epithelial cell"),
    ("photoreceptor cell", "neuron"),
    ("rod photoreceptor", "photoreceptor cell"),
    ("cone photoreceptor", "photoreceptor cell"),
]


@lru_cache(maxsize=1)
def kg_parents() -> dict[str, frozenset[str]]:
    par: dict[str, set[str]] = {}
    for f in sorted(glob.glob(os.path.join(KG_CACHE, "human_*.json"))):
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
    for f in sorted(glob.glob(os.path.join(KG_CACHE, "human_*.json"))):
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
