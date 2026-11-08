#!/usr/bin/env python
import json
import numpy as np, pandas as pd
from common import *

b = load_bin1()
G = np.load(CACHE / "geometry.npz"); tissue = G["tissue"]
ix = np.clip((b["x"] // PX).astype(np.int32), 0, NX - 1); iy = np.clip((b["y"] // PX).astype(np.int32), 0, NY - 1)
intis = tissue[ix, iy]
xd = (2 * (b["x"] + X0)).astype(np.int64); yd = (2 * (b["y"] + Y0)).astype(np.int64)
key = xd * 100000 + yd
res = {}
k_all, inv = np.unique(key, return_inverse=True)
umi_dnb = np.bincount(inv, weights=b["umi"].astype(float))
dnb_in = np.bincount(inv, weights=intis.astype(float)) > 0
area_um2 = tissue.sum() * PX * PX
lattice = area_um2 / 0.25
res["bin1_0p5um"] = dict(
    tissue_area_mm2=float(area_um2 / 1e6), lattice_positions_in_tissue=float(lattice),
    captured_positions_in_tissue=int(dnb_in.sum()),
    frac_positions_with_any_UMI=float(dnb_in.sum() / lattice),
    umi_per_captured_DNB_mean=float(umi_dnb[dnb_in].mean()),
    umi_per_captured_DNB_median=float(np.median(umi_dnb[dnb_in])),
    umi_per_lattice_position_mean=float(umi_dnb[dnb_in].sum() / lattice),
    frac_UMI_outside_ssDNA_tissue=float(b["umi"][~intis].sum() / b["umi"].sum()))
print(res["bin1_0p5um"])


bx = xd // 50; by = yd // 50; bk = bx * 100000 + by
df = pd.DataFrame(dict(bk=bk, g=b["gene"], u=b["umi"]))
u50 = df.groupby("bk").u.sum(); g50 = df.groupby("bk").g.nunique()
cx = np.clip(((0.5 * (u50.index // 100000 * 50 + 25) - X0) // PX).astype(int), 0, NX - 1)
cy = np.clip(((0.5 * (u50.index % 100000 * 50 + 25) - Y0) // PX).astype(int), 0, NY - 1)
m = tissue[cx, cy]
res["bin50_25um"] = dict(n_bins_in_tissue=int(m.sum()), umi_median=float(u50[m].median()), genes_median=float(g50[m].median()),
                         umi_iqr=[float(u50[m].quantile(.25)), float(u50[m].quantile(.75))])
print(res["bin50_25um"])

cb = pd.read_parquet(CACHE / "cellbin_cells.parquet")
res["cellbin_cell"] = dict(n_cells=int(len(cb)), umi_median=float(cb.total_counts.median()),
                           genes_median=float(cb.n_genes.median()),
                           captured_DNB_median=float(cb.n_dnb.median()),
                           frac_below_200_genes=float((cb.n_genes < 200).mean()),
                           frac_below_500_UMI=float((cb.total_counts < 500).mean()),
                           frac_below_100_genes=float((cb.n_genes < 100).mean()),
                           umi_median_by_type=cb.groupby("annotation").total_counts.median().round(0).to_dict())
print({k: v for k, v in res["cellbin_cell"].items() if k != "umi_median_by_type"})
res["rna_diffusion_halo"] = dict(note="RNA-density tissue mask vs ssDNA tissue mask; distance of RNA-mask edge outside the ssDNA edge",
                                 rna_mask_area_mm2=float(G["rna_tissue"].sum() * PX * PX / 1e6),
                                 ssdna_area_mm2=float(area_um2 / 1e6))
json.dump(res, open(CACHE / "stats_sparsity.json", "w"), indent=1, default=float)
