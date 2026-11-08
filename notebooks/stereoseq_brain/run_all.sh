#!/usr/bin/env bash
# Stereo-seq mouse brain: the analysis chain and every panel, in order.
# The SSAF-T run itself is NOT here -- see run_ssaft.py, which has to finish first.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
export NUMPY_MADVISE_HUGEPAGE=0

echo "== 00  cache the MOSTA DNB table and the published CellBin cells   (~15 min, 1 GB of TSV)"
$PY -W ignore 00_prep.py
echo "== 01  overview rasters + the ssDNA stain on the 5 µm grid"
$PY -W ignore 01_overview.py
$PY -W ignore 01b_ssdna.py
echo "== 02  geometry: pia from ssDNA, white matter from raw myelin genes, cortical depth"
$PY -W ignore 02_geometry.py
echo "== 03  cortex: bin50 anatomy, valid cortical pixels, per-pixel depth"
$PY -W ignore 03_cortex.py
echo "== 04  hippocampus: CA1 / CA3 / DG masks and skeletons from the raw UMI bands"
$PY -W ignore 04_hippo.py
echo "== 03b/03c  held-out tests beyond depth; sparsity and CellBin QC"
$PY -W ignore 03b_heldout.py
$PY -W ignore 03c_sparsity.py
echo "== 04b  assemble stats.json"
$PY -W ignore 04b_assemble.py
echo "== 05  SSAF-T cells in the cortex, on the depth axis"
$PY -W ignore 05_cortex_cells.py

echo "== 10  panel J  hippocampus maps"
$PY -W ignore 10_panelJ_hippocampus.py
echo "== 11  panel K  subfield gene heatmap"
$PY -W ignore 11_panelK_subfield_heatmap.py
echo "== 12  panel K  layer-band coverage"
$PY -W ignore 12_panelK_layer_coverage.py
echo "== 13  panel H  cortical column: SSAF-T layers against raw layer genes"
$PY -W ignore 13_panelH_cortex_column.py
echo "== 14  panel I  normalised depth, pia to white matter"
$PY -W ignore 14_panelI_depth_stats.py
echo "done"
