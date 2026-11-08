#!/usr/bin/env bash
# Visium HD mouse small intestine: the analysis chain and every panel, in order.
# The SSAF-T run itself is NOT here -- see run_ssaft.py, which has to finish first.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
export NUMPY_MADVISE_HUGEPAGE=0

# ENACT is the comparison method and is not shipped with the data: run it yourself on the 10x
# binned outputs and point ENACT_OUT at its output directory (see 00_prepare_enact.py).
echo "== 00 ENACT cells"
$PY -W ignore 00_prepare_enact.py

echo "== 01  bins on the lattice + sparsity numbers   (~10 min, reads 5.5M bins)"
$PY -W ignore 01_bins.py
$PY -W ignore 01b_kg_bins.py
echo "== 02  gene-group rasters"
$PY -W ignore 02_rasters.py
echo "== 03  crypt-villus axes from raw signal only"
$PY -W ignore 03_axis.py
echo "== 04  crypt-villus units"
$PY -W ignore 04_units.py
echo "== 05  statistics"
$PY -W ignore 05_stats.py
$PY -W ignore 05b_crypt_confinement.py
$PY -W ignore 05c_equal_resolution.py      # panel F reads per_unit_equal_resolution.csv
echo "== 06  pick the exemplar units (geometry only)"
$PY -W ignore 06_pick_units.py
echo "== 07  held-out landmark tests"
$PY -W ignore 07_heldout_strict.py
$PY -W ignore 07b_heldout_leave_one_out.py

echo "== 10  panels C, D, E, F"
# the published figure draws the raw gene fields of panel D on WHITE; without FIELD_BG the same
# panel is drawn on black and saved under the name without the _w suffix.
FIELD_BG=white $PY -W ignore 10_panels_CDEF.py
$PY -W ignore 10_panels_CDEF.py
echo "== 11  panel G"
$PY -W ignore 11_panelG_composition.py
echo "done"
