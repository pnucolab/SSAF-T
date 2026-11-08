#!/usr/bin/env python
import json
import numpy as np, pandas as pd
from common import *
import config as C

S = json.load(open(CACHE / "stats_cortex.json")); H = json.load(open(CACHE / "stats_hippo.json"))
HO = json.load(open(CACHE / "stats_heldout.json")); SP = json.load(open(CACHE / "stats_sparsity.json"))


QC = json.load(open(CACHE / "stats_cellbin_qc.json")); DC = C.DISPLAY_CHOICE
mids = (np.arange(20) + 0.5) / 20
def peaks(csv):
    t = pd.read_csv(OUT / csv, index_col=0); t = t.div(t.sum(0), axis=1)
    return {c: float(mids[np.argmax(np.convolve(t[c].values, np.ones(3) / 3, "same"))]) for c in t.columns}
pk_ss, pk_cb, pk_b50 = peaks("cortex_ssaft_depth_counts.csv"), peaks("cortex_cellbin_depth_counts.csv"), peaks("cortex_bin50_depth_counts.csv")
b50med = {m["label"]: m["median"] for m in S["bin50"]["medians"]}
REF = {"L2/3 IT glutamatergic neuron": "Cortex-L2/3", "L4 IT glutamatergic neuron": "Cortex_L4",
       "L5 ET glutamatergic neuron": "Cortex_L5", "L6 CT glutamatergic neuron": "Cortex_L6",
       "EX L2/3": "Cortex-L2/3", "EX L4": "Cortex_L4", "EX L5/6": "Cortex_L5", "EX L6": "Cortex_L6"}
agree = {}
for lab, pk in list(pk_ss.items()) + list(pk_cb.items()):
    if lab == "EX L5/6":
        ref = min(("Cortex_L5", "Cortex_L6"), key=lambda r: abs(pk - b50med[r]))
        agree[lab] = dict(peak_depth=pk, bin50_ref=ref, bin50_median=b50med[ref], abs_diff=abs(pk - b50med[ref]))
    elif lab in REF: agree[lab] = dict(peak_depth=pk, bin50_ref=REF[lab], bin50_median=b50med[REF[lab]], abs_diff=abs(pk - b50med[REF[lab]]))
ss_mad = float(np.mean([agree[l]["abs_diff"] for l in REF if l in pk_ss]))
cb_mad = float(np.mean([agree[l]["abs_diff"] for l in REF if l in pk_cb]))
print("peak-depth vs published bin50 layer median, mean |diff|: SSAF-T", round(ss_mad, 3), " CellBin", round(cb_mad, 3))
px = pd.read_parquet(CACHE / "cortex_pixels.parquet")
cbv = pd.read_parquet(CACHE / "cortex_cellbin.parquet")
exc_ss = px.label.isin(LAYERS + ["neocortical glutamatergic neuron", "glutamatergic neuron"])
exc_cb = cbv.annotation.isin(PUB_LAYERS + ["EX", "EX CA"])
layer_resolved = dict(
    ssaft_frac_of_excitatory_px_with_layer_label=float(px.label[exc_ss].isin(LAYERS).mean()),
    cellbin_frac_of_excitatory_cells_with_layer_label=float(cbv.annotation[exc_cb].isin(PUB_LAYERS).mean()),
    cellbin_EX_CA_cells_in_valid_neocortex=int((cbv.annotation == "EX CA").sum()),
    ssaft_L6IT=dict(n_px=int((px.label == "L6 IT glutamatergic neuron").sum()),
                    median_depth=float(px.depth[px.label == "L6 IT glutamatergic neuron"].median())),
    ssaft_striatal_GABA_px_in_cortex_frac=float((px.label == "striatal GABAergic neuron").mean()))
print(layer_resolved)
out = dict(
    frame="SSAF-T grid, 1 um; x_um = 0.5*x_dnb - 1612.5, y_um = 0.5*y_dnb - 3087.5; published CellBin cells placed by their own DNBs (label column), no fitted transform",
    cortex=dict(valid_columns=int(sum(w["valid"] for w in S["columns"])), valid_area_mm2=S["valid_area_mm2"],
                depth_axis_validation=dict(bin50_medians=b50med, canonical_gene_peaks={g: v["peak_depth"] for g, v in S["canonical_genes"].items()},
                                           canonical_genes_are_annotation_markers=True),
                ssaft=S["ssaft"], cellbin=S["cellbin"], bin50=S["bin50"],
                peak_depth=dict(ssaft=pk_ss, cellbin=pk_cb, bin50=pk_b50),
                peak_vs_bin50=agree, peak_vs_bin50_mean_abs_diff=dict(ssaft=ss_mad, cellbin=cb_mad),
                fraction_in_own_raw_gene_band=S["fraction_in_own_raw_band"], raw_gene_bands=S["raw_gene_bands"],
                layer_resolution=layer_resolved, label_composition=S["label_composition_valid_cortex"],
                heldout_beyond_depth=HO),
    hippocampus=H, sparsity=dict(**SP, cellbin_qc=QC), display_choice=DC)
json.dump(out, open(OUT / "stats.json", "w"), indent=1, default=float)
print("-> stats.json")
