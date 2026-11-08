#!/usr/bin/env bash
# Xenium mouse kidney: every analysis step and every panel, in order.
# The SSAF-T run itself is NOT here -- see run_ssaft.py, which has to finish first.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
export NUMPY_MADVISE_HUGEPAGE=0        # large allocations stall on transparent-hugepage hosts

echo "== 01 transcript table   (point in polygon over every molecule; ~20 min)"
$PY -W ignore 01_transcript_table.py
echo "== 02 similarity"
$PY -W ignore 02_similarity.py
echo "== 03 vendor identity    (~5 min)"
$PY -W ignore 03_vendor_identity.py

for s in 10_panelA_map 11_panelC_dapi_outlines 12_panelD_agreement 13_panelG_fate \
         14_panelH_falloff 15_panelI_recovered_where 16_panelJ_identity; do
  echo "== $s"
  $PY -W ignore "$s.py"
done
echo "done"
