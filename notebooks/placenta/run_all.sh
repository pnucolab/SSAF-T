#!/usr/bin/env bash
# Placenta suppl panels F and G. Logs go to <SSAFT_PANEL>/placenta/logs/.
#   ./run_all.sh                     the published panels (the SSAF-T cell-type map)
#   SOURCE=colall ./run_all.sh       the later column_all re-segmentation, for comparison
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-/bce/groups/pnucolab/analysis/l202255179/UNNAMED/.venv/bin/python}
export NUMPY_MADVISE_HUGEPAGE=0
LOG=$($PY -c 'import config as C; print(C.outdir(C.DS, "logs"))')
run() { echo "== $1 ${SOURCE:-map_v2}"; $PY -W ignore "$1" > "$LOG/${1%%_*}${SOURCE:+_$SOURCE}.log" 2>&1 \
        || { tail -20 "$LOG/${1%%_*}${SOURCE:+_$SOURCE}.log"; exit 1; }; }
run 01_panelF_stb_surface.py
run 02_build_villi.py
run 03_panelG_villus_size.py
echo "done"
